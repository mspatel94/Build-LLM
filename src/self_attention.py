import torch
from torch import nn

class GPTSelfAttentionV1(nn.Module):
    def __init__(self, kvq_dim:int = 512, embedding_dim:int=1024, kqv_bias:bool=False, dropout:float=0.5):
        super().__init__()
        self._k_matrix = nn.Linear(embedding_dim, kvq_dim, bias = kqv_bias)
        self._v_matrix = nn.Linear(embedding_dim, kvq_dim, bias = kqv_bias)
        self._q_matrix = nn.Linear(embedding_dim, kvq_dim, bias = kqv_bias)
        self.kvq_dim = kvq_dim
        self.dropout = nn.Dropout(p=dropout)
    
    # input (batch_size, seq_len, embedding_dim)
    # output (batch_size, seq_len, kvq_dim)
    def forward(self, inputs:torch.Tensor)->torch.Tensor:
        batch_size, num_tokens, num_dim = inputs.shape
        K = self._k_matrix(inputs)
        Q = self._q_matrix(inputs)
        V = self._v_matrix(inputs)
        attention_score = Q @ K.transpose(1,2)
        mask = torch.triu(torch.ones(self.kvq_dim, self.kvq_dim), diagonal=1)
        masked = attention_score.masked_fill(mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attention_weights = torch.softmax(masked/K.shape[-1]**0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vec = attention_weights @ V
        return context_vec


class MultiHeadedAttentionUnoptimized(nn.Module):
    def __init__(self, dim_in:int, dim_out:int, num_heads:int, kvq_bias:bool=False, dropout:float=0.5):
        super().__init__()
        self.heads = nn.ModuleList([GPTSelfAttentionV1(dim_out//num_heads, dim_in, kqv_bias=kvq_bias, dropout=dropout) for head in range(num_heads)])
    
    def forward(self, input):
        output = torch.cat([head(input) for head in self.heads], dim=-1)
        return output

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec
