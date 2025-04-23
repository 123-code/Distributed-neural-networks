import torch
import torch.nn as nn
import torch.nn.functional as F 
from model import GPT,GPTConfig,Block 

class ModelPart0(nn.Module):
    def __init__(self,original_model:GPT,start_layer:int,end_layer:int):
        super().__init__()
        self.config = original_model.config
        self.wte = original_model.transformer.wte
        self.wpe = original_model.transformer.wpe
        self.h = nn.ModuleList([original_model.transformer.h[i] for i in range(start_layer,end_layer + 1)])
    def forward(self,idx):
        B,T = idx.size()
        assert T<= self.config.block_size,"Cannot forward, model block size is exhausted."
        pos = torch.arange(0,T,dtype=torch.long,device=idx.device)
        pos_emb = self.wpe(pos)
        tok_emb = self.wte(idx)
        x = tok_emb + pos_emb
        for block in self.h:
            x = block(x)
        return x 
    


class ModelPartIntermediate(nn.Module):
    def __init__(self,original_model:GPT,start_layer:int,end_layer:int):
        super().__init__()
        self.config = original_model.config
        self.h = nn.ModuleList(original_model.transformer.h[i] for i in range(start_layer,end_layer + 1))
    def forward(self, x):
        for block in self.h:
            x = block(x)
        return x 
    
class ModelPartFinal_GPT(nn.Module):
     def __init__(self, original_model: GPT, start_layer: int, end_layer: int):
        super().__init__()
        self.config = original_model.config
        self.h = nn.ModuleList([original_model.transformer.h[i] for i in range(start_layer, end_layer + 1)])
        self.ln_f = original_model.transformer.ln_f
        self.lm_head = original_model.lm_head

     def forward(self, x):
    
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) 
        return logits