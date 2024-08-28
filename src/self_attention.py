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


class MultiHeadedAttention(nn.Module):
    def __init__(self, dim_in:int, dim_out:int, num_heads:int, kvq_bias:bool=False, dropout:float=0.5):
        super().__init__()
        self.heads = nn.ModuleList([GPTSelfAttentionV1(dim_out, dim_in, kqv_bias=kvq_bias, dropout=dropout) for head in range(num_heads)])
    
    def forward(self, input):
        output = torch.cat([head(input) for head in self.heads], dim=-1)
        return output