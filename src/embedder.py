import torch

class GPTEmbedder(torch.nn.Module):
    def __init__(self, vocab_size:int, dim:int = 256, max_seq_len:int=5):
        self._embeddings = torch.nn.Embedding(vocab_size, dim)
        self._pos_embeddings = torch.nn.Embedding(vocab_size, dim)
        self.max_seq_len= max_seq_len
    
    def forward(self, token_ids: torch.Tensor) -> torch.Any:
        semantic_embeddings = self._embeddings(token_ids)
        positional_embeddings = self._pos_embeddings(torch.arange(self.max_seq_len))
        return semantic_embeddings+positional_embeddings