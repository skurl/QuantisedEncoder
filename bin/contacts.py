"""Unsupervised contact probe (Rao et al. 2020): do the model's attention maps encode 3D contacts?

Per protein:
  attentions [n_layers][H,L,L]  ->  symmetrise + APC per head  ->  features [L, L, n_layers*H]
  labels: C-beta (C-alpha for Gly) distance < 8A from an AlphaFold structure, pLDDT-filtered
  probe:  logistic regression [n_layers*H] -> contact prob, fit on pooled train pairs (sep >= 6)
  metric: P@L (sep >= 6) and P@L-LR (sep >= 24) -- top-L predicted pairs, precision vs true contacts

Reports P@L-LR as the primary number (comparable to Candido et al. 2026's scaling curve -- we sit far
below their smallest point, which is the honest anchor). Also fits a per-layer probe (H features each)
to see where structure peaks in depth (Candido: contacts peak near the penultimate layer).

    python bin/downloads.py contacts                 # fetch AlphaFold structures -> data/contacts/
    python bin/contacts.py CKPT --pdb_dir data/contacts --out contacts.json
    python bin/contacts.py --demo
"""
import argparse, hashlib, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from core import device, load_checkpoint, Vocab, Transformer, build_arch

THREE2ONE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
             "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
             "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def parse_pdb(path):
    """AlphaFold pdb -> (coords [L,3], plddt [L], seq str). C-beta per residue (C-alpha for Gly);
    AlphaFold's per-atom B-factor column IS the pLDDT. Single chain, sequential residues."""
    ca, cb, plddt, res3 = {}, {}, {}, {}
    for line in open(path):
        if not line.startswith("ATOM"):
            continue
        atom, res = line[12:16].strip(), int(line[22:26])
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        if atom == "CA":
            ca[res], plddt[res], res3[res] = xyz, float(line[60:66]), line[17:20].strip()
        elif atom == "CB":
            cb[res] = xyz
    order = sorted(ca)
    coords = np.array([cb.get(r, ca[r]) for r in order])            # C-beta, fall back to C-alpha (glycine)
    seq = "".join(THREE2ONE.get(res3[r], "X") for r in order)
    return coords, np.array([plddt[r] for r in order]), seq


def contact_map(coords, thr=8.0):
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    return d < thr


def apc(S):                                                        # average product correction on a symmetric map
    r, c, t = S.sum(1, keepdims=True), S.sum(0, keepdims=True), S.sum() + 1e-8
    return S - r * c / t


@torch.no_grad()
def attn_features(model, vocab, seq, max_len=1024):
    """[L, L, n_layers*H] of symmetrised + APC-corrected attention maps (special tokens dropped)."""
    seq = seq[:max_len]
    ids = [vocab.cls] + [vocab.stoi.get(a, vocab.unk) for a in seq] + [vocab.eos]
    x = torch.tensor([ids], device=device)
    _, maps = model(x, torch.ones_like(x, dtype=torch.bool), return_attn=True)
    feats = []
    for m in maps:                                                 # m: [1, H, L+2, L+2]
        a = m[0, :, 1:-1, 1:-1].float().cpu().numpy()              # drop <cls>/<eos> -> [H, L, L]
        for h in range(a.shape[0]):
            feats.append(apc(a[h] + a[h].T))                       # symmetrise then APC
    return np.stack(feats, -1)                                     # [L, L, n_layers*H]


def pairs(feats, contacts, valid, sep):
    """Upper-triangle residue pairs with |i-j| >= sep and both residues pLDDT-valid -> (X [N,F], y [N])."""
    L = feats.shape[0]
    ii, jj = np.triu_indices(L, k=sep)
    keep = valid[ii] & valid[jj]
    ii, jj = ii[keep], jj[keep]
    return feats[ii, jj], contacts[ii, jj].astype(np.float32)


def fit_probe(X, y, steps=400, lr=0.05):
    """Logistic regression F -> 1. No sklearn: nn.Linear + BCE, positives up-weighted (contacts are sparse)."""
    X = torch.tensor(X, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.float32, device=device)
    pos = y.sum().clamp(min=1)
    lin = nn.Linear(X.shape[1], 1).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss(pos_weight=(len(y) - pos) / pos)
    for _ in range(steps):
        opt.zero_grad()
        lossf(lin(X).squeeze(-1), y).backward()
        opt.step()
    return lin


@torch.no_grad()
def probe_scores(lin, feats):
    L, _, F = feats.shape
    X = torch.tensor(feats.reshape(-1, F), dtype=torch.float32, device=device)
    return torch.sigmoid(lin(X).squeeze(-1)).cpu().numpy().reshape(L, L)


def precision_at_L(scores, contacts, valid, sep, topk=None):
    """Rank valid pairs (|i-j| >= sep) by score, take top-L, precision against true contacts."""
    L = scores.shape[0]
    ii, jj = np.triu_indices(L, k=sep)
    keep = valid[ii] & valid[jj]
    ii, jj = ii[keep], jj[keep]
    if len(ii) == 0:
        return float("nan")
    order = np.argsort(-scores[ii, jj])[:(topk or L)]
    return float(contacts[ii[order], jj[order]].mean())


def base_rate(contacts, valid, sep):        # a random scorer's expected P@L = fraction of eligible pairs that ARE contacts
    L = contacts.shape[0]
    ii, jj = np.triu_indices(L, k=sep)
    keep = valid[ii] & valid[jj]
    return float(contacts[ii[keep], jj[keep]].mean()) if keep.any() else float("nan")


def build(model, vocab, pdb_dir, min_plddt=70.0, max_len=1024):
    """Load every structure -> per-protein (features, contacts, valid mask). Skips length mismatches."""
    data = {}
    for pdb in sorted(Path(pdb_dir).glob("*.pdb")):
        coords, plddt, seq = parse_pdb(pdb)
        L = min(len(seq), max_len)
        if L < 30:
            continue
        coords, plddt, seq = coords[:L], plddt[:L], seq[:L]
        feats = attn_features(model, vocab, seq, max_len)
        if feats.shape[0] != L:                                    # non-standard residues can desync; skip
            continue
        data[pdb.stem] = (feats, contact_map(coords), plddt >= min_plddt)
    return data


def split(names, test_frac=0.2):
    """Deterministic hash-ranked split -> always >=1 test protein (bucket thresholds can empty the test set
    when there are few structures). ponytail: a proper MMseqs cluster-split is stricter; upgrade if redundancy bites."""
    ranked = sorted(names, key=lambda n: hashlib.md5(n.encode()).hexdigest())
    k = max(1, round(test_frac * len(names)))
    test = set(ranked[:k])
    return [n for n in names if n not in test], sorted(test)


def evaluate(model, vocab, pdb_dir, min_plddt=70.0, max_len=1024):
    data = build(model, vocab, pdb_dir, min_plddt, max_len)
    if len(data) < 4:
        raise SystemExit(f"need >=4 usable structures, got {len(data)} (run: python bin/downloads.py contacts)")
    tr, te = split(list(data))
    n_layers = model.arch["num_layers"]
    H = model.arch["num_heads"]

    Xtr = np.concatenate([pairs(*data[n], sep=6)[0] for n in tr])
    ytr = np.concatenate([pairs(*data[n], sep=6)[1] for n in tr])
    full = fit_probe(Xtr, ytr)
    per_layer = [fit_probe(Xtr[:, l * H:(l + 1) * H], ytr) for l in range(n_layers)]   # H features each

    res = {"n_train": len(tr), "n_test": len(te), "n_layers": n_layers, "p_at_L": [], "p_at_L_LR": [],
           "random_L": [], "random_LR": [], "per_layer_p_at_L": [[] for _ in range(n_layers)]}
    for n in te:
        feats, contacts, valid = data[n]
        s = probe_scores(full, feats)
        res["p_at_L"].append(precision_at_L(s, contacts, valid, 6))
        res["p_at_L_LR"].append(precision_at_L(s, contacts, valid, 24))
        res["random_L"].append(base_rate(contacts, valid, 6))
        res["random_LR"].append(base_rate(contacts, valid, 24))
        for l in range(n_layers):
            sl = probe_scores(per_layer[l], feats[:, :, l * H:(l + 1) * H])
            res["per_layer_p_at_L"][l].append(precision_at_L(sl, contacts, valid, 6))

    mean = lambda xs: float(np.nanmean(xs)) if xs else float("nan")
    return {"p_at_L": mean(res["p_at_L"]), "p_at_L_LR": mean(res["p_at_L_LR"]),
            "p_at_L_random": mean(res["random_L"]), "p_at_L_LR_random": mean(res["random_LR"]),
            "per_layer_p_at_L": [mean(v) for v in res["per_layer_p_at_L"]],
            "n_train": res["n_train"], "n_test": res["n_test"]}


def demo():
    # 1) the load-bearing correctness check: the recompute path must equal the SDPA forward.
    seqs = ["ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"]
    vocab = Vocab.from_sequences(seqs)
    arch = {"vocab_size": len(vocab.aa_vocab), "d_model": 32, "num_heads": 4, "num_layers": 3,
            "d_ff": 64, "dropout": 0.0, "num_classes": vocab.num_classes, "pad_idx": vocab.pad}
    model = Transformer(arch).to(device).eval()
    ids = [vocab.cls] + [vocab.stoi[a] for a in seqs[0]] + [vocab.eos]
    x = torch.tensor([ids], device=device); mask = torch.ones_like(x, dtype=torch.bool)
    o1 = model(x, mask)
    o2, maps = model(x, mask, return_attn=True)
    assert torch.allclose(o1, o2, atol=1e-5), "recompute-attention path must match the SDPA forward"
    assert len(maps) == arch["num_layers"] and maps[0].shape[-1] == len(ids)

    # 2) feature shape + APC symmetry
    L = len(seqs[0])
    feats = attn_features(model, vocab, seqs[0])
    assert feats.shape == (L, L, arch["num_layers"] * arch["num_heads"]), feats.shape

    # 3) metric behaves: an oracle scorer nails P@L, and beats a random scorer
    C = np.zeros((L, L), bool)
    for i in range(L - 4):
        C[i, i + 4] = C[i + 4, i] = True
    valid = np.ones(L, bool)
    oracle = C.astype(float) + 1e-6 * np.random.rand(L, L)
    ntrue = int(C[np.triu_indices(L, 3)].sum())
    assert precision_at_L(oracle, C, valid, sep=3, topk=ntrue) == 1.0
    assert precision_at_L(oracle, C, valid, 3) > precision_at_L(np.random.rand(L, L), C, valid, 3)
    print(f"demo OK (SDPA≡recompute, feats {feats.shape}, oracle P@L=1.0)")


def main():
    p = argparse.ArgumentParser(prog="contacts", description="Unsupervised attention-contact probe")
    p.add_argument("ckpt", nargs="?")
    p.add_argument("--pdb_dir", default="data/contacts")
    p.add_argument("--min_plddt", type=float, default=70.0)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--out", default="contacts.json")
    p.add_argument("--demo", action="store_true")
    a = p.parse_args()
    if a.demo or not a.ckpt:
        return demo()

    model, vocab = load_checkpoint(a.ckpt, device); model.eval()
    res = evaluate(model, vocab, a.pdb_dir, a.min_plddt, a.max_len)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"P@L-LR {res['p_at_L_LR']:.3f} (random {res['p_at_L_LR_random']:.3f})  |  "
          f"P@L {res['p_at_L']:.3f} (random {res['p_at_L_random']:.3f})  "
          f"(train {res['n_train']} / test {res['n_test']})")
    print("per-layer P@L:", " ".join(f"{v:.3f}" for v in res["per_layer_p_at_L"]))
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
