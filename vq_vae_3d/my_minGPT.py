# Based on Andrej Karpathy implementantion
# minGPT: https://github.com/karpathy/minGPT
# nanoGPT: https://github.com/karpathy/nanoGPT 

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from dataclasses import dataclass

class AttentionHead(nn.Module):
    def __init__(self, embedding_dim, head_dim, dropout, block_size):
        super().__init__()
        
        self.linear_Q = nn.Linear(embedding_dim, head_dim)
        self.linear_K = nn.Linear(embedding_dim, head_dim)
        self.linear_V = nn.Linear(embedding_dim, head_dim)
        self.dropout = nn.Dropout(dropout)
        self.block_size = block_size

    def scaled_dot_product_attetion(self, query, key, value, masked):
        B, T, d_k = query.size()

        #Calculating QV/sqrt(d_k)
        
        scores = torch.bmm(query, key.transpose(1, 2))/sqrt(d_k)
    
        self.register_buffer("bias",
                             torch.tril(torch.ones(self.block_size, self.block_size).to(scores.device))
                                    .view(1, 1, self.block_size, self.block_size))
        
        att_matrix = F.softmax(scores, dim=-1)
        
        if masked == True:
            att_matrix = att_matrix.masked_fill(self.bias[0,:,:T,:T]==0, float('-Inf'))
        
        
        att_matrix = self.dropout(att_matrix)
        output = torch.bmm(att_matrix, value)
        
        return output

    def forward(self, x, mask):
        
        one_head_attetion = self.scaled_dot_product_attetion(
            self.linear_Q(x),
            self.linear_V(x),
            self.linear_K(x),
            mask
        )
        return one_head_attetion

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embedding_dim % config.n_heads == 0
        self.block_size = config.block_size
        self.embedding_dim = config.embedding_dim
        self.n_heads = config.n_heads
        self.head_dim = config.embedding_dim//config.n_heads
        self.heads = nn.ModuleList([
            AttentionHead(self.embedding_dim, self.head_dim, config.dropout, self.block_size)
            for _ in range(self.n_heads )
        ])
        self.output_linear = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, masked=True):
        x = torch.cat([head(x, masked) for head in self.heads], dim=-1)
        x = self.resid_dropout(self.output_linear(x))
        return x

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.embedding_dim, 4*config.embedding_dim)
        self.linear2 = nn.Linear(4*config.embedding_dim, config.embedding_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x
        
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.embedding_dim)
        self.layer_norm2 = nn.LayerNorm(config.embedding_dim)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x

@dataclass
class GPTconfig:
    block_size: int = 1024
    vocab_size: int = 5025
    n_layers: int = 12
    n_heads: int = 12
    embedding_dim: int = 768
    dropout: float = 0.1

    
# From nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.embedding_dim),
            wpe = nn.Embedding(config.block_size, config.embedding_dim),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.embedding_dim),
        ))
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

        n_params = sum(p.numel() for p in self.transformer.parameters())
        print('number of parameters: %.2fM' % (n_params/1e6,))

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss
    
    def crop_block_size(self, block_size):
        assert block_size <=self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            block.attention.bias = block.attention.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # Run the model
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]/temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits<v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx










