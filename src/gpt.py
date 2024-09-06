from dataclasses import dataclass
import torch
import tiktoken
import self_attention

@dataclass
class GPTConfig:
    def __init__(self, seq_len:int, embedding_dim:int, num_heads:int, vocab_size:int, dropout:float=0.8):
        self.seq_len=seq_len
        self.embedding_dim=embedding_dim
        self.num_heads=num_heads
        self.vocab_size=vocab_size
        self.dropout=dropout

class GELU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class LayerNorm(torch.nn.Module):
    def __init__(self, embedding_dim:int):
        super().__init__()
        self.eps = 1e-5
        self.scale = torch.nn.Parameter(torch.ones(embedding_dim))
        self.bias = torch.nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, input: torch.Tensor)->torch.Tensor:
        mean = input.mean( dim=-1, keepdim=True)
        var = input.var(dim=-1, keepdim=True, unbiased=False)
        norm = (input - mean) / torch.sqrt(var + self.eps)
        return (self.scale * norm) + self.bias        

class FeedForwardLayer(torch.nn.Module):
    def __init__(self, interim_dim, output_dim):
        super().__init__()
        self.layer_1 = torch.nn.Linear(output_dim, interim_dim)
        self.activation = GELU()
        self.layer_2 = torch.nn.Linear(interim_dim, output_dim)
        self.processor = torch.nn.Sequential(*[self.layer_1, self.activation, self.layer_2])
    
    def forward(self, input):
        return self.processor(input)


class AttentionBlock(torch.nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.layer_norm_1 = LayerNorm(cfg.embedding_dim)
        self.mha = self_attention.MultiHeadedAttention(cfg.embedding_dim, cfg.embedding_dim, cfg.seq_len, cfg.dropout, cfg.num_heads, qkv_bias=False)
        self.layer_norm_2 = LayerNorm(cfg.embedding_dim)
        self.ffn = FeedForwardLayer(cfg.embedding_dim*4, cfg.embedding_dim)
        self.dropout = torch.nn.Dropout(cfg.dropout)

    def forward(self, inputs:torch.Tensor)->torch.Tensor:
        norm = self.layer_norm_1(inputs)
        attention = self.mha(inputs)
        attention = self.dropout(attention)
        x=attention+inputs
        norm = self.layer_norm_2(x)
        ffn_output = self.ffn(norm)
        ffn_output = self.dropout(ffn_output)
        return ffn_output+attention

class GPT(torch.nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg=cfg
        self.token_embeddings = torch.nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        self.position_embeddings = torch.nn.Embedding(cfg.seq_len, cfg.embedding_dim)
        self.dropout = torch.nn.Dropout(cfg.dropout)
        self.attention_heads = torch.nn.Sequential(*[AttentionBlock(cfg) for _ in range(cfg.num_heads)])
        self.layer_norm = LayerNorm(cfg.embedding_dim)
        self.nn_layer = torch.nn.Linear(cfg.embedding_dim, cfg.vocab_size, bias=False)

    def forward(self, input:torch.Tensor)->torch.Tensor:
        batch, tokens = input.shape
        embeddings = self.token_embeddings(input)
        pos_embeddings = self.position_embeddings(torch.arange(tokens))
        final_embeddings = embeddings + pos_embeddings
        dropped_embed = self.dropout(final_embeddings)
        attention_output = self.attention_heads(dropped_embed)
        norm = self.layer_norm(attention_output)
        logits = self.nn_layer(norm) #b, seq, vocab
        return logits
    
    def generate_simple_text(self, prefix:torch.Tensor, max_len:int=10):
        output = torch.Tensor([]).to(torch.int32)
        input = prefix
        for i in range(max_len):
            logits = self.forward(input[-self.cfg.seq_len:].view(1, -1))
            next_pred = logits[:,-1, :]
            print(next_pred.shape)
            next_pred = torch.softmax(next_pred, dim=-1)
            index = next_pred.argmax(dim=1, keepdims=True)
            print("index",index, index[0])
            input=torch.cat([input, index[0]], dim=0)
            output=torch.cat([output, index[0]], dim=0)
            print(input)
            print("outout", output)
        return output

if __name__=="__main__":
    torch.manual_seed(123)
    a = "Every effort moves you"
    b = "Every day holds a"
    encoder = tiktoken.get_encoding("gpt2")
    a_tensor = torch.Tensor(encoder.encode(a)).to(torch.int32)
    b_tensor = torch.Tensor(encoder.encode(b)).to(torch.int32)
    input = torch.stack([a_tensor,b_tensor], dim=0)
    cfg = GPTConfig(1024,768,12,50257,0.1)
    model = GPT(cfg)
    print(input)
    output = model(input)
    print(output, output.shape)
    text_output = model.generate_simple_text(a_tensor)
    print(f"input: {a}, output: {encoder.decode(text_output.squeeze(0).tolist())}")

