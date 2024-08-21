import os
from abc import ABC, abstractmethod
import re
import vocabulary

DATA_FILE = f"{os.path.dirname(os.path.realpath(__file__))}/../data/verdict.txt"

def read_file(path: str)->str:
    with open(path) as file:
        return file.read()

class Tokenizer(ABC):
    def tokenize(self, data:str):
        pass

class WordTokenizer(Tokenizer):
    def __init__(self, split_punctuation=True):
        self._split_punctuation=split_punctuation
    
    def _remove_empty_str(self, data:list[str])->list[str]:
        return [item for item in data if item.strip()]
    
    def tokenize(self, data:str):
        if self._split_punctuation:
            return self._remove_empty_str(re.split(r'([,.:;?_!"()\']|--|\s)', data))
        else:
            return self._remove_empty_str(re.split("(\s)", data))


class TokenizerFactory:
    @staticmethod
    def get_tokenizer(type:str):
        if type.lower()=="word":
            return WordTokenizer()
        else:
            raise Exception("Not valid tokenizer asked")


if __name__=="__main__":
    data = read_file(DATA_FILE)
    tokenizer = TokenizerFactory.get_tokenizer("word")
    tokens = tokenizer.tokenize(data)
    vocab = vocabulary.Vocabulary(tokens)
    print(vocab.encode(["clasping"]))
