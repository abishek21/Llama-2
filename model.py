import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass()
class ModelArgs:
    dim: int  = 4096
    n_layers: int = 32
    n_heads: int =32
    n_kv_heads: Optional[int] = None
    vocab_size: int =-1
    multiple_of: int =256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5


     ## Needde for kv cache
    max_batch_size : int=32
    max_seq_len : int =2048

    device:str = None


def precomputer_theta_pos_frequencies(head_dim: int,seq_len:int, device:str,thetha:float = 10000.0):
    assert head_dim%2==0, "Dimension must be divisble by 2"

    thetha_numerator = torch.arange(0,head_dim,2).float()  ### This is the series [0,2,4,head_dim/2]

    ## formule 10000 ^ -(i/head_dim) , where i is [0,2,4,head_dim/2] , shape : head_dim/2
    thetha = 1.0/ (thetha**(thetha_numerator/head_dim)).to(device)

    ### now lets create m (positions)
    ## shape : (seq_len)
    m=torch.arange(0,seq_len).float()


    ## now get all possible combinations of thetha with m (outer product)
    ##shape : (seq_len,head_dim/2)
    freqs=torch.outer(m,thetha)

    ## now write this in complex form
    freqs_complex = torch.polar(torch.ones_like(freqs),freqs)

    return freqs_complex


def apply_rotary_embedding(x:torch.Tensor,freqs_complex:torch.Tensor,device:str):
    # Separate the last dimension pairs of two values, representing the real and imaginary parts of the complex number
    # Two consecutive values will become a single complex number
    # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Reshape the freqs_complex tensor to match the shape of the x_complex tensor. So we need to add the batch dimension and the head dimension
    # (Seq_Len, Head_Dim/2) --> (1, Seq_Len, 1, Head_Dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # Multiply each complex number in the x_complex tensor by the corresponding complex number in the freqs_complex tensor
    # Which results in the rotation of the complex number as shown in the Figure 1 of the paper
    # (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
    x_rotated = x_complex * freqs_complex
    # Convert the complex number back to the real number
    # (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


class RMSNorm(nn.Module):
    def __init__(self,dim: int,norm_eps: float = 1e-6):
        super().__init__()
        self.eps=norm_eps
        self.weight=nn.Parameter(torch.ones(dim))

    def _norm(self,x:torch.Tensor):
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x)
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)

    def forward(self,x:torch.Tensor):
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.weight*self._norm(x.float()).type_as(x)



def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # (B, Seq_Len, N_KV_Heads, 1, Head_Dim)
        x[:, :, :, None, :]
        # (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        # (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )


class SelfAttention(nn.Module):
    def __init__(self,args:ModelArgs):
        super().__init__()

        ## Indicates the number of heads for the Key and Values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
       ## Indicates the number of heads for the Queries
        self.n_heads_q = args.n_heads
        ## Indicates how many times the key and values should be repeated
        self.n_rep = self.n_heads_q//self.n_kv_heads
        ## Indicates the dimension of each head, that is the part of embedding that each head will be responsible for
        self.head_dim = args.dim//args.n_heads


        ## Weight for Q
        self.wq=nn.Linear(args.dim,self.n_heads_q*self.head_dim,bias=False)
        ## Weight for K
        self.wk=nn.Linear(args.dim,self.n_kv_heads*self.head_dim,bias=False)
        ## Weight for V
        self.wv=nn.Linear(args.dim,self.n_kv_heads*self.head_dim,bias=False)

        self.wo=nn.Linear(args.n_heads*self.head_dim,args.dim,bias=False)

        self.cache_k = torch.zeros((args.max_batch_size,args.max_seq_len,self.n_kv_heads,self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self,x:torch.Tensor,start_pos:int,freqs_complex:torch.Tensor):
        batch_size,seq_len, _ = x.shape ## (Batch,1,Dim)

        xq=self.wq(x)
        xk=self.wk(x)
        xv=self.wv(x)

        ## (B,1,H_Q*Head_dim) --> (B,seq_len,H_Q,head_dim)
        xq=xq.view(batch_size,seq_len,self.n_heads_q,self.head_dim)
        ## (B,1,H_Q*Head_dim) --> (B,seq_len,H_Q,head_dim)
        xk=xk.view(batch_size,seq_len,self.n_heads_q,self.head_dim)
        ## (B,1,H_Q*Head_dim) --> (B,seq_len,H_Q,head_dim)
        xv=xv.view(batch_size,seq_len,self.n_heads_q,self.head_dim)

        ## (B, seq_len, H_Q, head_dim) --> (B,seq_len,H_Q,head_dim)
        xq=apply_rotary_embedding(xq,freqs_complex,device=x.device)
        ## (B, seq_len, H_Q, head_dim) --> (B,seq_len,H_Q,head_dim)
        xk=apply_rotary_embedding(xk,freqs_complex,device=x.device)

        ## Replace the  etry in the cache for this token only on K and V
        self.cache_k[:batch_size,start_pos:start_pos+seq_len]=xk
        self.cache_v[:batch_size,start_pos:start_pos+seq_len]=xv

        ## Retrieve all cached values so far because we need it for matmul
        keys=self.cache_k[:batch_size,0:start_pos+seq_len]
        values=self.cache_v[:batch_size,0:start_pos+seq_len]

        ## Since every group Q shares the same K and V heads , just repeast K and V heads for every Q in the same group

        keys=repeat_kv(keys,self.n_rep)
        values=repeat_kv(values,self.n_rep)

        ## move head_dim before seq each head wil watch
        ##(B,1,H_Q,Head_dim) --> (B,H_Q,1,Head_dim)
        xq=xq.transpose(1,2)
        keys=keys.transpose(1,2)
        values=values.transpose(1,2)


        scores=torch.matmul(xq,keys.transpose(2,3)) / math.sqrt(self.head_dim)
        scores=F.softmax(scores.float(),dim=-1).type_as(xq)

        output = torch.matmul(scores,values)

        output = (output.transpose(1,2).contiguous().view(batch_size,seq_len,-1))
        return self.wo(output) ##(B,1,Dim) --> (B,1,Dim)





class FeedForward(nn.Module):
    def __init__(
        self,
        args: ModelArgs
    ):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        swish = F.silu(self.w1(x))
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.w3(x)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        x = self.w2(x)
        return x





class EncoderBlock(nn.Module):
    def __init__(self,args:ModelArgs):
        super().__init__()
        self.n_heads=args.n_heads
        self.dim=args.dim
        self.head_dim=args.dim//args.n_heads

        self.attention=SelfAttention(args)
        self.feed_forward=FeedForward(args)

        ## RMSlayer before self-attention
        self.attention_norm=RMSNorm(args.dim,norm_eps=args.norm_eps)

        ## RMSlayer before feed forward block
        self.ffn_norm=RMSNorm(args.dim,norm_eps=args.norm_eps)

    def forward(self,x:torch.Tensor,start_pos:int,freqs_complex:torch.Tensor):
        ##(B,seq_len,dim) + (B,seq_len,dim) ---> (B,seq_len,dim)
        h=x+self.attention.forward(self.attention_norm(x),start_pos,freqs_complex)
        out=h+self.feed_forward(self.ffn_norm(h))

        return out


class Transformer(nn.Module):
    def __init__(self,args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size!=-1, "vocab size must be set"
        self.args =args
        self.vocab_size=args.vocab_size
        self.n_layers=args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size,args.dim) ## each token is encoded in 4096 dimension
        self.layers=nn.ModuleList()

        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))


        self.norm=RMSNorm(args.dim,norm_eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size,bias=False)  ### output layer each vocab size gets a probability assigned

        self.freqs_complex = precomputer_theta_pos_frequencies(self.args.dim // self.args.n_heads,self.args.max_seq_len*2,device=self.args.device)


    def forward(self,tokens:torch.Tensor, start_pos : int):
        ## (batch_size,seq_len):
        batch_size,seq_len = tokens.shape
        assert seq_len==1 ## only one token at a time is processed"

        ## (b,seq_len) -- > (b,seq_len,dim)
        h = self.tok_embeddings(tokens)

        ## Retrieve the pairs (m,theta) corresponding to the positions [start_pos,start_pos+seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos+seq_len]


        for layer in self.layers:
            h=layer(h,start_pos,freqs_complex)

        h=self.norm(h)

        output=self.output(h).float()
        return output


# freqs=precomputer_theta_pos_frequencies(head_dim=4096/32,seq_len=2048,device='cpu')
#
# x=torch.randn((32,2048,32,128))
#
# x_rot=apply_rotary_embedding(x,freqs,device='cpu')