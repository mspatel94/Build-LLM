import data_loader
from gpt import GPT, GPTConfig
from gpt2 import GPTModel
from torch.nn.functional import cross_entropy
from typing import Any
import torch


def get_cross_entropy(logits, target):
    return cross_entropy(logits.flatten(0,1), target.flatten())

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device) #A
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss



def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)  #A
    else:
        num_batches = min(num_batches, len(data_loader)) #B
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item() #C 
        else:
            break
    return total_loss / num_batches

def train_model_loop(model:GPTModel, train_data:torch.Tensor, test_data:torch.Tensor, optimizer:Any, learning_rate:float, epoch:int, device:str):
    train_loss, eval_loss = [], []
    
    for _ in range(epoch):
        for x,y in train_data:
            model.train()
            model.zero_grad()
            loss = calc_loss_batch(x,y, model, device)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            
        for x,y in test_data:
            model.eval()
            loss = calc_loss_batch(x, y, model, device)
            eval_loss.append(loss.item())
    
    return train_loss, eval_loss


if __name__=="__main__":
    txt=data_loader.read_file(data_loader.DATA_FILE)
    train,test = data_loader.get_train_and_test_dataset(txt)
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()  # disable dropout

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") #A
    device="cpu"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.0004, weight_decay=0.1)
    train_loss,eval_loss = train_model_loop(model, train, test, optimizer, 0, 3, device)

    print("Training loss:", train_loss)
    print("Validation loss:", eval_loss)



