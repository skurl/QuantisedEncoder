import math, random
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.parametrize import register_parametrization, remove_parametrizations
from torch.utils.data import Sampler

device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ModelArgs:
    out_dir = "./outputs"
    data_file = "./data/fungi.fasta"    # pipeline overrides via --data_file
    length_cutoff = 512
    split_ratios = (0.8, 0.1, 0.1)
    split_seed = 1234
    mask_rate = 0.15
    d_model = 512             # sweep ni3jx7oo best config
    num_heads = 8
    num_layers = 6
    d_ff = 512               # sweep preferred a narrow FFN
    dropout = 0
    batch_size = 64
    length_batching = True    # group similar-length sequences per batch -> less padding waste
    max_steps = 5000
    eval_every = 250          # eval + checkpoint cadence, in optimizer steps
    learning_rate = 5.54e-4   # sweep ni3jx7oo best config
    weight_decay = 0.1
    label_smoothing = 0.05
    grad_clip = 0.5
    grad_accum = 4
    warmup_steps = 500
    min_lr_ratio = 0.1
    use_ema = False
    ema_decay = 0.999
    amp = True
    log_every = 100
    seed = 42
    eval_mask_seed = 999
    eval_max_seqs = 5000      # cap val/test used for eval -> eval cost is fixed, not proportional to dataset size (0 = use all)
    select_metric = "ppl"     # checkpoint selection: "ppl" (lowest val perplexity), "blosum", or "pgym" (needs a matched panel)
    use_wandb = True
    wandb_project = "quantised-encoder"
    run_name = None
    wandb_group = None
    qat_bits = 0              # 0 = normal training; N = quantisation-aware training at N bits
    init_ckpt = ""           # start from this checkpoint's weights (QAT fine-tune) instead of a fresh init
    distill_teacher = ""     # HF model id (e.g. facebook/esm2_t12_35M_UR50D); "" = no distillation. Any id works -- swap teachers by config.
    distill_weight = 1.0     # weight on the representation-matching loss added to the MLM loss
    pgym_panel = ""          # dir of a FEW held-out DMS csvs; if set, select checkpoints on their mean Spearman (north star) instead of val BLOSUM
    pgym_panel_match = ""    # optional comma-sep filename filter for the panel (keep it DISJOINT from the assays REPORT scores on)

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
SPECIALS = ["<pad>", "<mask>", "<cls>", "<unk>", "<eos>"]


# VOCAB

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


# DATA

def read_fasta(lines):
    seq = []
    for line in lines:
        if line.startswith(">"):
            if seq:
                yield "".join(seq)
                seq = []
        else:
            seq.append(line.strip())
    if seq:
        yield "".join(seq)


def load_sequences(path, cutoff):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python bin/downloads.py fungi` first to fetch it")
    with open(path) as fh:
        seqs = [s.upper().replace("*", "") for s in read_fasta(fh)]
    seqs = [s for s in seqs if 0 < len(s) < cutoff]
    return sorted(set(seqs))


def random_split(n, ratios, seed):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_tr, n_va = int(ratios[0] * n), int(ratios[1] * n)
    groups = [idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]]
    print(f"[split] train {len(groups[0])}  val {len(groups[1])}  test {len(groups[2])}")
    return groups


# MASKING  (take the Vocab explicitly)

def mask_sequence(seq, rng, vocab):
    ids, labels = [vocab.cls], [-100]
    for aa in seq:
        tok = vocab.stoi.get(aa, vocab.unk)
        if aa in vocab.out_stoi and rng.random() < ModelArgs.mask_rate:
            labels.append(vocab.out_stoi[aa])
            r = rng.random()
            if r < 0.8:
                ids.append(vocab.mask)
            elif r < 0.9:
                ids.append(vocab.stoi[rng.choice(vocab.classes)])
            else:
                ids.append(tok)
        else:
            ids.append(tok)
            labels.append(-100)
    ids.append(vocab.eos)
    labels.append(-100)
    return ids, labels


def pad_batch(items, vocab):
    xs = [torch.tensor(a, dtype=torch.long) for a, _ in items]
    ys = [torch.tensor(b, dtype=torch.long) for _, b in items]
    lengths = torch.tensor([len(x) for x in xs])
    x = pad_sequence(xs, batch_first=True, padding_value=vocab.pad)
    y = pad_sequence(ys, batch_first=True, padding_value=-100)
    attn = torch.arange(x.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    return x, y, attn


def train_collate(batch, vocab):
    x, y, attn = pad_batch([mask_sequence(s, random, vocab) for s in batch], vocab)
    return x, y, attn, batch          # keep raw seqs for the (optional) distillation teacher


def precompute(seqs, seed, vocab):
    rng = random.Random(seed)
    return [mask_sequence(s, rng, vocab) for s in seqs]


class LengthBatchSampler(Sampler):
    """Batches of similar-length sequences (less padding). Sorts by length with a small jitter,
    chunks into batches, then shuffles batch ORDER each epoch so gradients stay stochastic."""
    def __init__(self, lengths, batch_size, seed=0):
        self.lengths, self.bs = lengths, batch_size
        self.rng = random.Random(seed)

    def __iter__(self):
        order = sorted(range(len(self.lengths)), key=lambda i: (self.lengths[i], self.rng.random()))
        batches = [order[i:i + self.bs] for i in range(0, len(order), self.bs)]
        self.rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return (len(self.lengths) + self.bs - 1) // self.bs


# MODEL  (RoPE + pre-norm + SDPA)

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

    def forward(self, x, padding_mask=None, return_attn=False):
        B, L, D = x.shape
        split = lambda t: t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(self.W_q(x)), split(self.W_k(x)), split(self.W_v(x))
        cos, sin = self.rope(L)
        q, k = apply_rope(q, k, cos, sin)
        attn_mask = padding_mask[:, None, None, :] if padding_mask is not None else None
        if return_attn:                          # un-fused path: recompute softmax(QK^T) to EXPOSE the map (probing only)
            scores = q @ k.transpose(-2, -1) * self.head_dim ** -0.5
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            attn = scores.softmax(-1)
            self._attn = attn.detach()           # [B, H, L, L] stashed for the contact probe; identical to what SDPA computes
            out = attn @ v
        else:
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

    def forward(self, x, padding_mask=None, return_attn=False):
        x = x + self.dropout(self.attn(self.norm1(x), padding_mask, return_attn))
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
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
                m.weight.data[self.pad_idx].zero_()
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, src, attention_mask=None, return_repr=False, return_attn=False):
        x = self.dropout(self.embed(src))
        maps = []
        for layer in self.layers:
            x = layer(x, attention_mask, return_attn=return_attn)
            if return_attn:
                maps.append(layer.attn._attn)          # [B, H, L, L] per layer, for the contact probe
        x = self.final_norm(x)
        out = x if return_repr else self.fc(x)
        return (out, maps) if return_attn else out


def build_arch(vocab):
    return {"vocab_size": len(vocab.aa_vocab), "d_model": ModelArgs.d_model, "num_heads": ModelArgs.num_heads,
            "num_layers": ModelArgs.num_layers, "d_ff": ModelArgs.d_ff, "dropout": ModelArgs.dropout,
            "num_classes": vocab.num_classes, "pad_idx": vocab.pad}


# QUANTISATION-AWARE TRAINING  (weight-only, per-channel symmetric RTN with straight-through backward)

def fake_quant_weight(w, bits):   # mirror of quantise.fake_quant_weight -- keep in sync
    qmax = 2 ** (bits - 1) - 1
    scale = (w.detach().abs().amax(dim=1) / qmax).clamp(min=1e-8)
    zp = torch.zeros(w.shape[0], dtype=torch.int32, device=w.device)
    return torch.fake_quantize_per_channel_affine(w, scale, zp, axis=0, quant_min=-qmax, quant_max=qmax)


class _FakeQuant(nn.Module):      # parametrization: module.weight returns the fake-quantized weight each forward
    def __init__(self, bits): super().__init__(); self.bits = bits
    def forward(self, w): return fake_quant_weight(w, self.bits)


def apply_qat(model, bits):       # fake-quant every Linear + the embedding table (the fragile layer)
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            register_parametrization(m, "weight", _FakeQuant(bits))
    return model


def bake_qat(model):              # collapse parametrizations so .weight HOLDS the quantized values (save-ready)
    for m in model.modules():
        if getattr(m, "parametrizations", None) and "weight" in m.parametrizations:
            remove_parametrizations(m, "weight", leave_parametrized=True)
    return model


# DISTILLATION  (representation transfer from a frozen HF teacher -- training-only, needs `transformers`)
# ponytail: match per-residue hidden states, not logits -> teacher-vocab-agnostic. Any HF encoder that
# tokenises 1-token-per-residue with a leading <cls> (ESM2, ESM-C, ...) aligns index-for-index with ours.

def load_teacher(name):
    from transformers import AutoModel, AutoTokenizer          # heavy import, only when distilling
    tok = AutoTokenizer.from_pretrained(name)
    teacher = AutoModel.from_pretrained(name).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher, tok, teacher.config.hidden_size


@torch.no_grad()
def teacher_reps(teacher, tok, seqs):                          # [B, L, D] last hidden states; residue i at index i
    enc = tok(list(seqs), return_tensors="pt", padding=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
        return teacher(**enc).last_hidden_state                # bf16 teacher forward ~2x faster; distill_loss casts to float anyway



def distill_loss(student_rep, proj, t_rep, valid):
    """1 - cosine between projected student reps and teacher reps, over valid (unmasked, non-special) residues.
    Student and teacher put residue i at the same index i, so we compare position-for-position."""
    L = min(student_rep.size(1), t_rep.size(1))
    m = valid[:, :L]
    s = proj(student_rep[:, :L][m])                            # [N, D_teacher]
    t = t_rep[:, :L][m]
    return (1 - F.cosine_similarity(s, t.float(), dim=-1)).mean()


def save_checkpoint(path, model, vocab):
    torch.save({"arch": model.arch, "model": model.state_dict(),
                "vocab": vocab.aa_vocab, "classes": vocab.classes}, path)


def load_checkpoint(path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = Transformer(ckpt["arch"]).to(map_location)
    model.load_state_dict(ckpt["model"])
    return model, Vocab(ckpt["vocab"], ckpt["classes"])


# EMBEDDINGS  (unmasked full sequences -> mean-pooled per-protein vectors)

@torch.no_grad()
def embed(model, seqs, vocab, batch_size=64):
    model.eval()
    out = []
    for i in range(0, len(seqs), batch_size):
        ids = [[vocab.cls] + [vocab.stoi.get(a, vocab.unk) for a in s] + [vocab.eos] for s in seqs[i:i + batch_size]]
        lengths = torch.tensor([len(t) for t in ids])
        x = pad_sequence([torch.tensor(t) for t in ids], batch_first=True, padding_value=vocab.pad).to(device)
        attn = (torch.arange(x.size(1))[None, :] < lengths[:, None]).to(device)
        rep = model(x, attn, return_repr=True)
        valid = attn.clone()
        valid[:, 0] = False                                         # drop <cls> (untrained here)
        valid[torch.arange(len(ids)), lengths - 1] = False          # drop <eos>
        mask = valid.unsqueeze(-1).float()
        out.append(((rep * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu())
    return torch.cat(out)


# EVAL + ANALYSIS

@torch.no_grad()
def evaluate(model, loader, vocab):
    model.eval()
    nll_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    total_nll = torch.zeros((), device=device)
    total_tok = torch.zeros((), device=device)
    t1 = torch.zeros((), device=device)
    t3 = torch.zeros((), device=device)
    t5 = torch.zeros((), device=device)
    for x, y, attn in loader:
        x, y, attn = x.to(device), y.to(device), attn.to(device).bool()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=(ModelArgs.amp and device == "cuda")):
            logits = model(x, attn)
        fl, fy = logits.float().reshape(-1, vocab.num_classes), y.reshape(-1)
        total_nll += nll_fn(fl, fy)
        m = fy != -100
        total_tok += m.sum()
        top5 = fl[m].topk(5, dim=-1).indices
        hits = top5 == fy[m].unsqueeze(-1)
        t1 += hits[:, :1].any(-1).sum()
        t3 += hits[:, :3].any(-1).sum()
        t5 += hits[:, :5].any(-1).sum()
    n = max(1, int(total_tok.item()))
    nll = total_nll.item() / n
    return {"nll": nll, "perplexity": math.exp(nll),
            "top1": 100 * t1.item() / n, "top3": 100 * t3.item() / n, "top5": 100 * t5.item() / n}


def unigram_baseline(train_seqs, test_data, vocab):
    counts = Counter(aa for s in train_seqs for aa in s if aa in vocab.out_stoi)
    n = sum(counts.values())
    freq = np.clip(np.array([counts[vocab.out_itos[i]] / n for i in range(vocab.num_classes)]), 1e-12, None)
    ranking = np.argsort(-freq)
    labels = np.array([t for _, y in test_data for t in y if t != -100])
    nll = float(-np.mean(np.log(freq[labels])))
    return {"top1": 100 * np.mean(labels == ranking[0]),
            "top3": 100 * np.mean(np.isin(labels, ranking[:3])),
            "top5": 100 * np.mean(np.isin(labels, ranking[:5])),
            "perplexity": math.exp(nll)}


AA_CLASS = {**{a: "hydrophobic" for a in "AVILM"}, **{a: "aromatic" for a in "FWY"},
            **{a: "polar" for a in "STNQ"}, **{a: "basic" for a in "KRH"},
            **{a: "acidic" for a in "DE"}, **{a: "special" for a in "GPC"}}


@torch.no_grad()
def biochemical_breakdown(model, loader, vocab):
    model.eval()
    tot, cor = defaultdict(int), defaultdict(int)
    class_correct = same_wrong = wrong = 0
    for x, y, attn in loader:
        x, y, attn = x.to(device), y.to(device), attn.to(device).bool()
        preds = model(x, attn).argmax(-1)
        m = y != -100
        for t, p in zip(y[m].cpu().tolist(), preds[m].cpu().tolist()):
            ta, pa = vocab.out_itos[t], vocab.out_itos[p]
            tot[ta] += 1
            same = AA_CLASS.get(ta) == AA_CLASS.get(pa)
            if ta == pa:
                cor[ta] += 1
            else:
                wrong += 1
                same_wrong += same
            class_correct += same
    total = sum(tot.values())
    return {"per_aa": {a: {"acc": 100 * cor[a] / tot[a], "n": tot[a]} for a in sorted(tot)},
            "biochemical_class_accuracy": 100 * class_correct / max(1, total),
            "wrong_but_same_class_rate": 100 * same_wrong / max(1, wrong)}


def _spearman(a, b):
    def rank(x):
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        csum = np.cumsum(cnt)
        return ((csum - cnt + csum - 1) / 2.0)[inv]
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


# BLOSUM62 (standard matrix, hardcoded so we don't pull in biopython for one constant)
_B62_ORDER = "ARNDCQEGHILKMFPSTWYV"
_B62 = [
    [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],
    [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],
    [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],
    [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],
    [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
    [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],
    [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],
    [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],
    [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],
    [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],
    [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],
    [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],
    [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],
    [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],
    [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],
    [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3],
    [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],
    [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4],
]
BLOSUM62 = {(a, b): _B62[i][j] for i, a in enumerate(_B62_ORDER) for j, b in enumerate(_B62_ORDER)}
assert all(BLOSUM62[a, b] == BLOSUM62[b, a] for a in _B62_ORDER for b in _B62_ORDER)                    # symmetric
assert (BLOSUM62["W", "W"], BLOSUM62["C", "C"], BLOSUM62["L", "I"], BLOSUM62["D", "E"], BLOSUM62["F", "Y"]) == (11, 9, 2, 2, 3)


def blosum_correlation_unigram(train_seqs, vocab):
    """Frequency-only NULL for blosum_correlation: a context-free model that always predicts the
    marginal AA frequency (conf[a][b] = f[b]). Its off-diagonal-vs-BLOSUM rho is the floor that
    substitution structure gives you for free -- the trained model's rho only means "learned
    biology, not just frequencies" insofar as it clears this. (Hou et al. 2026: a site-independent
    frequency model is a strong ProteinGym baseline, so this number must exist next to the headline.)"""
    counts = Counter(aa for s in train_seqs for aa in s if aa in vocab.out_stoi)
    n = max(1, sum(counts.values()))
    f = np.array([counts[vocab.out_itos[i]] / n for i in range(vocab.num_classes)])
    tiled = np.tile(f, (vocab.num_classes, 1))
    conf = 0.5 * (tiled + tiled.T)                              # symmetrised, exactly as blosum_correlation does
    aas = [vocab.out_itos[i] for i in range(vocab.num_classes)]
    B = np.array([[BLOSUM62[a, b] for b in aas] for a in aas], dtype=float)
    iu = np.triu_indices(vocab.num_classes, k=1)
    return _spearman(conf[iu], B[iu])


@torch.no_grad()
def blosum_correlation(model, loader, vocab):
    model.eval()
    nc = vocab.num_classes
    conf = torch.zeros(nc, nc, device=device)
    counts = torch.zeros(nc, device=device)
    for x, y, attn in loader:
        x, y, attn = x.to(device), y.to(device), attn.to(device).bool()
        probs = model(x, attn).softmax(-1).reshape(-1, nc)
        fy = y.reshape(-1)
        m = fy != -100
        conf.index_add_(0, fy[m], probs[m])
        counts.index_add_(0, fy[m], torch.ones_like(fy[m], dtype=conf.dtype))
    conf = (conf / counts.clamp(min=1).unsqueeze(1)).cpu().numpy()
    conf = 0.5 * (conf + conf.T)
    aas = [vocab.out_itos[i] for i in range(nc)]
    B = np.array([[BLOSUM62[a, b] for b in aas] for a in aas], dtype=float)
    iu = np.triu_indices(nc, k=1)
    return {"spearman_offdiag": _spearman(conf[iu], B[iu]), "aas": aas, "model_matrix": conf.tolist()}