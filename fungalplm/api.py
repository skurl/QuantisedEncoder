"""Inference API + CLI for the fungal PLM.

    from fungalplm import FungalPLM
    plm = FungalPLM.load("fungal-plm.pth")            # or a HuggingFace repo id, once published
    emb = plm.embed(["MQIFVKTLTG...", "MKTAY..."])    # [N, d] per-protein, ESM-shaped
    res = plm.embed(seqs, per_residue=True)           # list of [Li, d]

    fungalplm embed proteins.fasta --ckpt m.pth -o emb.npz [--per-residue]
"""
import argparse

import numpy as np
import torch

from .model import Vocab, Transformer, load_checkpoint


def read_fasta(path):                       # (id, seq) pairs -- keep ids so embeddings stay labelled
    rid, seq = None, []
    for line in open(path):
        if line.startswith(">"):
            if rid is not None:
                yield rid, "".join(seq)
            rid, seq = line[1:].strip().split()[0] if line[1:].strip() else "seq", []
        else:
            seq.append(line.strip())
    if rid is not None:
        yield rid, "".join(seq)


class FungalPLM:
    MAX_LEN = 2046                          # RoPE ceiling (2048) minus <cls>/<eos>

    def __init__(self, model, vocab, device):
        self.model, self.vocab, self.device = model, vocab, device

    @classmethod
    def load(cls, ckpt, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, vocab = load_checkpoint(ckpt, device)
        model.eval()
        return cls(model, vocab, device)

    def _encode(self, seq):
        v = self.vocab
        return [v.cls] + [v.stoi.get(a, v.unk) for a in seq[:self.MAX_LEN]] + [v.eos]

    @torch.no_grad()
    def embed(self, sequences, per_residue=False, batch_size=32):
        """Per-protein vectors [N, d] (mean over residues), or a list of per-residue [Li, d]."""
        if isinstance(sequences, str):
            sequences = [sequences]
        out = []
        for i in range(0, len(sequences), batch_size):
            chunk = [s.upper() for s in sequences[i:i + batch_size]]
            ids = [self._encode(s) for s in chunk]
            L = max(len(t) for t in ids)
            x = torch.full((len(ids), L), self.vocab.pad, device=self.device)
            for r, t in enumerate(ids):
                x[r, :len(t)] = torch.tensor(t, device=self.device)
            rep = self.model(x, x != self.vocab.pad, return_repr=True)     # [B, L, d], pre-head repr
            for r, s in enumerate(chunk):
                res = rep[r, 1:min(len(s), self.MAX_LEN) + 1]              # drop <cls>/<eos>/pad -> [len, d]
                out.append(res.cpu() if per_residue else res.mean(0).cpu())
        return out if per_residue else torch.stack(out)


def demo():                                 # shape check: per-residue length must equal sequence length
    seqs = ["ACDEFGHIK", "MKT"]
    vocab = Vocab.from_sequences(seqs)
    arch = {"vocab_size": len(vocab.aa_vocab), "d_model": 32, "num_heads": 4, "num_layers": 2,
            "d_ff": 64, "dropout": 0.0, "num_classes": vocab.num_classes, "pad_idx": vocab.pad}
    plm = FungalPLM(Transformer(arch).eval(), vocab, "cpu")
    pooled, per = plm.embed(seqs), plm.embed(seqs, per_residue=True)
    assert pooled.dim() == 2 and pooled.shape[0] == 2, pooled.shape
    assert [p.shape[0] for p in per] == [9, 3], [p.shape[0] for p in per]
    print(f"demo OK: pooled {tuple(pooled.shape)}, per-residue {[p.shape[0] for p in per]}")


def main():
    p = argparse.ArgumentParser(prog="fungalplm", description="Embed protein sequences with the fungal PLM")
    p.add_argument("--demo", action="store_true")
    sub = p.add_subparsers(dest="cmd")
    e = sub.add_parser("embed")
    e.add_argument("fasta")
    e.add_argument("--ckpt", required=True)
    e.add_argument("-o", "--out", default="embeddings.npz")
    e.add_argument("--per-residue", action="store_true")
    a = p.parse_args()
    if a.demo or a.cmd is None:
        return demo()

    ids, seqs = zip(*read_fasta(a.fasta))
    plm = FungalPLM.load(a.ckpt)
    emb = plm.embed(list(seqs), per_residue=a.per_residue)
    if a.per_residue:
        np.savez(a.out, **{i: e.numpy() for i, e in zip(ids, emb)})       # one array per protein, keyed by id
    else:
        np.savez(a.out, ids=np.array(ids), emb=emb.numpy())              # [N, d] + parallel ids
    d = emb[0].shape[-1] if a.per_residue else emb.shape[1]
    print(f"embedded {len(seqs)} sequences (dim {d}) -> {a.out}")


if __name__ == "__main__":
    main()
