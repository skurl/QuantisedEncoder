"""Minimal, torch-only model definition for the fungal PLM.

This is the STANDALONE inference copy of the encoder. It must stay structurally identical to
`bin/core.py`'s Vocab/Transformer so checkpoints load (`load_checkpoint` rebuilds from the saved
arch dict). Kept separate so `pip install fungalplm` pulls only torch -- not the training/pipeline
stack. If you change the architecture in core.py, mirror it here.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
SPECIALS = ["<pad>", "<mask>", "<cls>", "<unk>", "<eos>"]


@dataclass
class Vocab:
    aa_vocab: list        # input alphabet: specials + amino acids seen in the data
    classes: list         # the predictable standard amino acids

    def __post_init__(self):
        self.stoi = {s: i for i, s in enumerate(self.aa_vocab)}
        self.out_stoi = {a: i for i, a in enumerate(self.classes)}
        self.out_itos = {i: a for i, a in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        self.pad, self.mask, self.cls, self.unk, self.eos = (
            self.stoi[f"<{t}>"] for t in ("pad", "mask", "cls", "unk", "eos"))

    @classmethod
    def from_sequences(cls, seqs):
        aas = sorted(set("".join(seqs)))
        return cls(SPECIALS + aas, [a for a in aas if a in STANDARD_AA])


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build(max_len)

    def _build(self, n):
        freqs = torch.outer(torch.arange(n, device=self.inv_freq.device).float(), self.inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, n):
        if n > self.cos.size(0):
            self._build(n)
        return self.cos[:n], self.sin[:n]


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    cos = torch.cat([cos, cos], -1).unsqueeze(0).unsqueeze(0)
    sin = torch.cat([sin, sin], -1).unsqueeze(0).unsqueeze(0)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x, padding_mask=None):
        B, L, D = x.shape
        split = lambda t: t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(self.W_q(x)), split(self.W_k(x)), split(self.W_v(x))
        cos, sin = self.rope(L)
        q, k = apply_rope(q, k, cos, sin)
        attn_mask = padding_mask[:, None, None, :] if padding_mask is not None else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                             dropout_p=self.dropout if self.training else 0.0)
        return self.W_o(out.transpose(1, 2).contiguous().view(B, L, D))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), padding_mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.arch = dict(arch)
        self.pad_idx = arch["pad_idx"]
        self.embed = nn.Embedding(arch["vocab_size"], arch["d_model"], padding_idx=arch["pad_idx"])
        self.layers = nn.ModuleList([EncoderLayer(arch["d_model"], arch["num_heads"],
                                                  arch["d_ff"], arch["dropout"])
                                     for _ in range(arch["num_layers"])])
        self.final_norm = nn.LayerNorm(arch["d_model"])
        self.fc = nn.Linear(arch["d_model"], arch["num_classes"], bias=False)
        self.dropout = nn.Dropout(arch["dropout"])

    def forward(self, src, attention_mask=None, return_repr=False):
        x = self.dropout(self.embed(src))
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.final_norm(x)
        return x if return_repr else self.fc(x)


def load_checkpoint(path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = Transformer(ckpt["arch"]).to(map_location)
    model.load_state_dict(ckpt["model"])
    return model, Vocab(ckpt["vocab"], ckpt["classes"])
