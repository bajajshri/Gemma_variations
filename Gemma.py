import json 
import argparse
import math 
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class ModelConfig:            
    vocab_size: int = 4096
    d_model: int = 512
    n_layers: int = 6
    d_ff: int = 2048          # GeGLU hidden size
    n_heads: int = 8
    n_kv_heads: int = 8       
    head_dim: int = 64
    context: int = 512



def init_params(cfg: ModelConfig, device, seed=0) -> dict:
    g = torch.Generator().manual_seed(seed)
    # initalize weights (Transformers are initalized to a small std)
    def w(*shape, std = 0.02):
        return (torch.randn(*shape, generator=g)*std).to(device).requires_grad_()
    def zeros(n):  # Gemma RMSNorm scales by (1 + w), so w = 0 starts as the identity
            return torch.zeros(n, device=device, requires_grad=True)
    d, hd, out_std = cfg.d_model, cfg.head_dim,  0.02 / math.sqrt(2 * cfg.n_layers)
    p = {"embed": w(cfg.vocab_size, d)}
    for l in range(cfg.n_layers):
        p.update({
            f"{l}.wq": w(cfg.n_heads * hd, d),
            f"{l}.wk": w(cfg.n_kv_heads * hd, d),
            f"{l}.wv": w(cfg.n_kv_heads * hd, d),
            f"{l}.wo": w(d, cfg.n_heads * hd, std = out_std),
            f"{l}.ffn_norm": zeros(d),
            f"{l}.w_gate": w(cfg.d_ff, d),  # Geglu
            f"{l}.w_up": w(cfg.d_ff, d),    #Geglu
            f"{l}.w_down": w(d, cfg.d_ff, std = out_std),
        }
        )
    p["final_norm"] = zeros(d)
    return p

def rms_norm(x,w, eps):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * (1.0 + w.float())).type_as(x)

