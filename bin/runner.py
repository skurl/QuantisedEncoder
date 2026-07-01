import math, json, random, copy, time
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}\n")


@dataclass
class ModelArgs:
    out_dir = "./outputs"
    data_file = "./data/fungi_clustered.fasta"   # built by import_fasta.py (50%-identity reps)
    length_cutoff = 512
    split_ratios = (0.8, 0.1, 0.1)
    split_seed = 1234
    mask_rate = 0.15
    d_model = 256
    num_heads = 8
    num_layers = 6 # changed for now, because the current dataset is tiny (13k sequences), 2.9 mln tokens total
    d_ff = 4 * 256
    dropout = 0   # Dropout = 0 in the ESM paper, but with a smaller dataset it may overfit
    batch_size = 32 # changed from 64
    num_epochs = 200
    learning_rate = 1e-3
    weight_decay = 1e-2
    label_smoothing = 0.05
    grad_clip = 1.0
    grad_accum = 4 # this one might be interesting to tweak as well
    warmup_steps = 1000
    min_lr_ratio = 0.1
    use_ema = True
    ema_decay = 0.999
    amp = True          # bf16 mixed precision on CUDA (Ampere+); no GradScaler needed
    log_every = 100
    seeds = (42, )
    eval_mask_seed = 999


# DATA  (clustered fungal proteomes from import_fasta.py)

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
        raise FileNotFoundError(f"{path} missing - run `python bin/import_fasta.py` first to build it")
    with open(path) as fh:
        seqs = [s.upper().replace("*", "") for s in read_fasta(fh)]
    seqs = [s for s in seqs if 0 < len(s) < cutoff]
    return sorted(set(seqs))


# SPLIT  (random since the data is clustered)

def random_split(n, ratios, seed):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_tr, n_va = int(ratios[0] * n), int(ratios[1] * n)
    groups = [idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]]
    print(f"[split] train {len(groups[0])}  val {len(groups[1])}  test {len(groups[2])}")
    return groups


# MASKING

def mask_sequence(seq, rng):
    ids, labels = [cls_idx], [-100]
    for aa in seq:
        tok = aa_stoi.get(aa, unk_idx)
        if aa in output_stoi and rng.random() < ModelArgs.mask_rate:
            labels.append(output_stoi[aa])
            r = rng.random()
            if r < 0.8:
                ids.append(mask_idx)
            elif r < 0.9:
                ids.append(aa_stoi[rng.choice(classes)])
            else:
                ids.append(tok)
        else:
            ids.append(tok)
            labels.append(-100)
    ids.append(eos_idx)
    labels.append(-100)
    return ids, labels


def pad_batch(items):
    xs = [torch.tensor(a, dtype=torch.long) for a, _ in items]
    ys = [torch.tensor(b, dtype=torch.long) for _, b in items]
    lengths = torch.tensor([len(x) for x in xs])
    x = pad_sequence(xs, batch_first=True, padding_value=pad_idx)
    y = pad_sequence(ys, batch_first=True, padding_value=-100)
    attn = torch.arange(x.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    return x, y, attn


def train_collate(batch):
    return pad_batch([mask_sequence(s, random) for s in batch])


def precompute(seqs, seed):
    rng = random.Random(seed)
    return [mask_sequence(s, rng) for s in seqs]


# MODEL  (RoPE + pre-norm + nn.SDPA)

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
        self.arch = dict(arch)            # stored so checkpoints are self-describing and can be reused
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

    def forward(self, src, attention_mask=None):
        x = self.dropout(self.embed(src))
        for layer in self.layers:
            x = layer(x, attention_mask)
        return self.fc(self.final_norm(x))


def build_arch(vocab_size, num_classes, pad_idx):
    return {"vocab_size": vocab_size, "d_model": ModelArgs.d_model, "num_heads": ModelArgs.num_heads,
            "num_layers": ModelArgs.num_layers, "d_ff": ModelArgs.d_ff, "dropout": ModelArgs.dropout,
            "num_classes": num_classes, "pad_idx": pad_idx}


def save_checkpoint(path, model, vocab, classes):     # self-describing: arch + vocab travel with the weights, so quantise/analyze never re-guess the config
    torch.save({"arch": model.arch, "model": model.state_dict(),
                "vocab": list(vocab), "classes": list(classes)}, path)


def load_checkpoint(path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = Transformer(ckpt["arch"]).to(map_location)
    model.load_state_dict(ckpt["model"])
    return model, ckpt["vocab"], ckpt["classes"]


# SCHEDULER

def make_scheduler(opt, warmup, total, min_ratio):
    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return max(min_ratio, 0.5 * (1 + math.cos(math.pi * prog)))
    return optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# TRAIN + EVAL

@torch.no_grad()
def evaluate(model, loader):
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
        fl, fy = logits.float().reshape(-1, num_classes), y.reshape(-1)   # fp32 for stable ppl
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


def train(model, train_loader, seed):
    model.to(device)
    crit = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=ModelArgs.label_smoothing)
    opt = optim.AdamW(model.parameters(), lr=ModelArgs.learning_rate, weight_decay=ModelArgs.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / ModelArgs.grad_accum)
    total_steps = steps_per_epoch * ModelArgs.num_epochs
    sched = make_scheduler(opt, ModelArgs.warmup_steps, total_steps, ModelArgs.min_lr_ratio)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ModelArgs.ema_decay)) if ModelArgs.use_ema else None

    best_val, best_state = float("inf"), copy.deepcopy(model.state_dict())
    global_step, tic, tok_acc = 0, time.time(), torch.zeros((), device=device)

    for epoch in range(ModelArgs.num_epochs):
        model.train()
        opt.zero_grad()
        for step, (x, y, attn) in enumerate(train_loader):
            x, y, attn = x.to(device), y.to(device), attn.to(device).bool()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(ModelArgs.amp and device == "cuda")):
                loss = crit(model(x, attn).reshape(-1, num_classes), y.reshape(-1)) / ModelArgs.grad_accum
            loss.backward()                          # bf16: no GradScaler needed
            tok_acc += (y != -100).sum()
            if (step + 1) % ModelArgs.grad_accum == 0 or (step + 1) == len(train_loader):
                clip_grad_norm_(model.parameters(), ModelArgs.grad_clip)
                opt.step()
                opt.zero_grad()
                sched.step()
                if ema:
                    ema.update_parameters(model)
                global_step += 1
                if global_step % ModelArgs.log_every == 0:
                    el = time.time() - tic
                    eta = (total_steps - global_step) * (el / ModelArgs.log_every) / 60
                    print(f"  step {global_step:6d}/{total_steps} | lr {sched.get_last_lr()[0]:.2e} | "
                          f"{tok_acc.item()/max(el,1e-6):.0f} tok/s | eta {eta:.1f}m")
                    tic, _ = time.time(), tok_acc.zero_()

        eval_model = ema.module if ema else model
        val = evaluate(eval_model, val_loader)
        improved = val["nll"] < best_val - 1e-4
        if improved:
            best_val = val["nll"]
            best_state = copy.deepcopy(eval_model.state_dict())
            save_checkpoint(Path(ModelArgs.out_dir) / f"best_seed{seed}.pth", eval_model, aa_vocab, classes)
        print(f"  epoch {epoch+1:3d} | val ppl {val['perplexity']:.3f} | "
              f"top1 {val['top1']:.2f}% | top3 {val['top3']:.2f}% | top5 {val['top5']:.2f}% "
              f"{'*' if improved else ''}")

    model.load_state_dict(best_state)
    return model


# ANALYSIS (I dont really use it for the AA class but here is is just in case, BLOSUM62 is better)

def unigram_baseline():
    counts = Counter(aa for s in train_seqs for aa in s if aa in output_stoi)
    n = sum(counts.values())
    freq = np.clip(np.array([counts[output_itos[i]] / n for i in range(num_classes)]), 1e-12, None)
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
def biochemical_breakdown(model):
    model.eval()
    tot, cor = defaultdict(int), defaultdict(int)
    class_correct = same_wrong = wrong = 0
    for x, y, attn in test_loader:
        x, y, attn = x.to(device), y.to(device), attn.to(device).bool()
        preds = model(x, attn).argmax(-1)
        m = y != -100
        for t, p in zip(y[m].cpu().tolist(), preds[m].cpu().tolist()):
            ta, pa = output_itos[t], output_itos[p]
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


@torch.no_grad()
def blosum_correlation(model):
    from Bio.Align import substitution_matrices
    model.eval()
    conf = torch.zeros(num_classes, num_classes, device=device)
    counts = torch.zeros(num_classes, device=device)
    for x, y, attn in test_loader:
        x, y, attn = x.to(device), y.to(device), attn.to(device).bool()
        probs = model(x, attn).softmax(-1).reshape(-1, num_classes)
        fy = y.reshape(-1)
        m = fy != -100
        conf.index_add_(0, fy[m], probs[m])
        counts.index_add_(0, fy[m], torch.ones_like(fy[m], dtype=conf.dtype))
    conf = (conf / counts.clamp(min=1).unsqueeze(1)).cpu().numpy()
    conf = 0.5 * (conf + conf.T)
    blosum = substitution_matrices.load("BLOSUM62")
    aas = [output_itos[i] for i in range(num_classes)]
    B = np.array([[blosum[a, b] for b in aas] for a in aas], dtype=float)
    iu = np.triu_indices(num_classes, k=1)
    return {"spearman_offdiag": _spearman(conf[iu], B[iu]), "aas": aas, "model_matrix": conf.tolist()}


# RUN  (only runs when the runner.py is called directly, rest can be reused in other scripts)

if __name__ == "__main__":
    sequences = load_sequences(ModelArgs.data_file, ModelArgs.length_cutoff)
    print("Number of unique sequences:", len(sequences))

    # VOCAB
    STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
    SPECIALS = ["<pad>", "<mask>", "<cls>", "<unk>", "<eos>"]
    amino_acids = sorted(set("".join(sequences)))
    aa_vocab = SPECIALS + amino_acids
    aa_stoi = {s: i for i, s in enumerate(aa_vocab)}

    classes = [a for a in amino_acids if a in STANDARD_AA]
    output_stoi = {a: i for i, a in enumerate(classes)}
    output_itos = {i: a for a, i in output_stoi.items()}
    num_classes = len(classes)

    pad_idx = aa_stoi["<pad>"]
    mask_idx = aa_stoi["<mask>"]
    cls_idx = aa_stoi["<cls>"]
    unk_idx = aa_stoi["<unk>"]
    eos_idx = aa_stoi["<eos>"]
    print(f"Vocab: {len(aa_vocab)} tokens, {num_classes} output classes\n")

    tr_idx, va_idx, te_idx = random_split(len(sequences), ModelArgs.split_ratios, ModelArgs.split_seed)
    train_seqs = [sequences[i] for i in tr_idx]
    val_seqs = [sequences[i] for i in va_idx]
    test_seqs = [sequences[i] for i in te_idx]

    val_data = precompute(val_seqs, ModelArgs.eval_mask_seed)
    test_data = precompute(test_seqs, ModelArgs.eval_mask_seed + 1)
    val_loader = DataLoader(val_data, batch_size=ModelArgs.batch_size, shuffle=False, collate_fn=pad_batch)
    test_loader = DataLoader(test_data, batch_size=ModelArgs.batch_size, shuffle=False, collate_fn=pad_batch)

    Path(ModelArgs.out_dir).mkdir(parents=True, exist_ok=True)

    base = unigram_baseline()
    print(f"\n[baseline] unigram  top1 {base['top1']:.2f}%  top3 {base['top3']:.2f}%  "
          f"top5 {base['top5']:.2f}%  ppl {base['perplexity']:.3f}")

    arch = build_arch(len(aa_vocab), num_classes, pad_idx)
    runs, best = [], None
    for seed in ModelArgs.seeds:
        print(f"\n===== seed {seed} =====")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        g = torch.Generator()
        g.manual_seed(seed)
        train_loader = DataLoader(train_seqs, batch_size=ModelArgs.batch_size, shuffle=True,
                                  collate_fn=train_collate, generator=g)
        model = Transformer(arch)
        if seed == ModelArgs.seeds[0]:
            print(f"[model] {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
        model = train(model, train_loader, seed)
        tm = evaluate(model, test_loader)
        tm["seed"] = seed
        print(f"  TEST seed {seed}  top1 {tm['top1']:.2f}%  top3 {tm['top3']:.2f}%  "
              f"top5 {tm['top5']:.2f}%  ppl {tm['perplexity']:.3f}")
        runs.append(tm)
        if best is None or tm["top1"] > best[0]:
            best = (tm["top1"], model)

    agg = {k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs]))}
           for k in ["top1", "top3", "top5", "perplexity", "nll"]}
    bio = biochemical_breakdown(best[1])
    blosum = blosum_correlation(best[1])

    print("\n================  SUMMARY  ================")
    print(f"unigram   top1 {base['top1']:.2f}%  ppl {base['perplexity']:.3f}")
    print(f"model     top1 {agg['top1']['mean']:.2f}±{agg['top1']['std']:.2f}%  "
          f"ppl {agg['perplexity']['mean']:.3f}±{agg['perplexity']['std']:.3f}")
    print(f"[biochem] class-accuracy {bio['biochemical_class_accuracy']:.2f}%  | "
          f"wrong-but-same-class {bio['wrong_but_same_class_rate']:.2f}%")
    print(f"[blosum]  off-diagonal Spearman vs BLOSUM62: {blosum['spearman_offdiag']:.3f}")

    save_checkpoint(Path(ModelArgs.out_dir) / "model_best.pth", best[1], aa_vocab, classes)
    with open(Path(ModelArgs.out_dir) / "results.json", "w") as fh:
        json.dump({"baseline": base, "per_seed": runs, "aggregate": agg,
                   "biochemistry": bio, "blosum": blosum}, fh, indent=2)
    print(f"\nSaved model_best.pth and results.json to {ModelArgs.out_dir}")