import math 

import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
  def __init__(self, d_model: int, vocab_size: int):
    super().__init__()
    self.d_model = d_model
    self.vocab_size = vocab_size
    self.embedding = nn.Embedding(vocab_size, d_model)

  def forward(self, x):
    return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
    super().__init__()
    self.d_model = d_model
    self.seq_len = seq_len
    self.droput = nn.Dropout(dropout)

    # create a matrix of shape (seq_len, d_model)
    pe = torch.zeros(seq_len, d_model)
    # create a vector of shape (seq_len, 1)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    # apply the sin to the even positions
    pe[:, 0::2] = torch.sin(position * div_term)
    # apply the cos to the odd positions
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0) # (1, seq_len, d_model)

    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + (self.self.pe[:, :x.shape[1], :]).requires_grad(False)
    return self.dropout(x)

class LayerNormaliation(nn.Module):
  def __init__(self, eps: float = 10**-6) -> None:
    super().__init__()
    self.eps = eps
    self.alpha = nn.Parameter(torch.ones(1)) # multiplied
    self.bias = nn.Paramter(torch.zeros(1)) # added

  def forward(self, x):
    mean = x.mean(dim = -1, keepdim=True)
    std = x.std(dim = -1, keepdim=True)
    return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
  def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
    super().__init__()
    self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
    self.dropout = nn.Dropout(dropout)
    self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

  def forward(self, x):
    # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_ff) --> (Batch, Seq_Len, d_model)
    return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
  def __init__(self, d_model: int, h: int, dropout: float) -> None:
    super().__init__()
    self.d_model = d_model
    self.h = h
    assert d_model % h == 0, "d_model is not divisible by h"

    self.d_k = d_model // h    
    self.w_q = nn.Linear(d_model, d_model) # Wq
    self.w_k = nn.Linear(d_model, d_model) # Wk
    self.w_v = nn.Linear(d_model, d_model) # Wv



