from collections import defaultdict

class Vocabulary:
    def __init__(self, tokens:list[str], max_size=None):
        self.max_size = max_size
        self.token_to_id = {}
        self.id_to_token = {}
        self._oov_token_id = -1
        self.token_to_count = defaultdict(int)
        self._build(tokens)
    
    def _build(self, tokens=list[str])->None:
        for token in tokens:
            self.token_to_count[token]+=1
        
        cutoff_count = 0

        if self.max_size:
            cutoff_count = sorted(self.token_to_count.values(), reverse=True)[min(self.max_size, len(self.token_to_count)-1)]
        
        counter = 0
        for token in sorted(self.token_to_count.keys()):
            if cutoff_count<=self.token_to_count[token]:
                self.token_to_id[token]=counter
                self.id_to_token[counter]=token
                counter+=1
        
    def encode(self, tokens:list[str])->list[int]:
        encoded_tokens = []

        for token in tokens:
            if token in self.token_to_id:
                encoded_tokens.append(self.token_to_id[token])
            else:
                encoded_tokens.append(-1)
            
        return encoded_tokens
    
    def decode(self, ids:list[int])->list[str]:
        decoded_tokens = []

        for id in ids:
            if ids in self.id_to_token:
                decoded_tokens.append(self.id_to_token[id])
            else:
                decoded_tokens.append(" ")
            
        return decoded_tokens