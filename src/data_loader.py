import os
import tiktoken
from torch.utils.data import Dataset, DataLoader
from typing import Any, Tuple, List
import torch

DATA_FILE = f"{os.path.dirname(os.path.realpath(__file__))}/../data/verdict.txt"

def read_file(path: str)->str:
    with open(path) as file:
        return file.read()


class GPTDatasetV1(Dataset):
    def __init__(self, txt:str, tokenizer:Any, max_length: int = 5, stride:int = 1):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids)-max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i:i+max_length]))
            self.target_ids.append(torch.tensor(token_ids[i+1:i+max_length+1]))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index) -> Any:
        return self.input_ids[index], self.target_ids[index]

def create_dataset_loader_v1(txt:str, batch_size:int = 4, max_length:int=4, stride:int=4, shuffle:bool=True, drop_last:bool=True, num_workers:int = 0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length=max_length, stride=stride)
    dataloader = DataLoader(dataset, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers, batch_size=batch_size)

    return dataloader

def get_train_and_test_dataset(txt:str, split_ratio:float=0.9)->Tuple[GPTDatasetV1, GPTDatasetV1]:
    split_index = int(len(txt)*split_ratio)
    train_dataset = create_dataset_loader_v1(txt[:split_index], batch_size=2, max_length=256, stride=256, shuffle=True, drop_last=True)
    test_dataset = create_dataset_loader_v1(txt[split_index:], batch_size=2, max_length=256, stride=256, shuffle=True, drop_last=False)
    return train_dataset, test_dataset

if __name__=="__main__":
    data = read_file(DATA_FILE)
    dataloader = create_dataset_loader_v1(data)

    iter_data = iter(dataloader)

    for i in range(10):
        print(next(iter_data))
    