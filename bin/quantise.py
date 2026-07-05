import copy, json, sys
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from runner import (ModelArgs, device, load_checkpoint, load_sequences, random_split,
                    precompute, pad_batch, evaluate, biochemical_breakdown, blosum_correlation, embed)

BITS = [8, 4, 3, 2]


def fake_quant_weight(w, bits):   # native op == the one QAT uses (straight-through backward)
    qmax = 2 ** (bits - 1) - 1
    scale = (w.detach().abs().amax(dim=1) / qmax).clamp(min=1e-8)
    zp = torch.zeros(w.shape[0], dtype=torch.int32, device=w.device)
    return torch.fake_quantize_per_channel_affine(w, scale, zp, axis=0, quant_min=-qmax, quant_max=qmax)


@torch.no_grad()
def fake_quantize_(model, bits):   # PTQ: overwrite every Linear weight in place
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.weight.data = fake_quant_weight(m.weight.data, bits)
    return model


def metrics(model, loader, vocab, test_seqs, fp_emb):
    e = evaluate(model, loader, vocab)
    b = biochemical_breakdown(model, loader, vocab)
    s = blosum_correlation(model, loader, vocab)["spearman_offdiag"]
    cos = F.cosine_similarity(embed(model, test_seqs, vocab), fp_emb, dim=1).mean().item()
    return {"ppl": e["perplexity"], "top1": e["top1"], "top3": e["top3"], "top5": e["top5"],
            "biochem_class_acc": b["biochemical_class_accuracy"], "blosum_spearman": s, "emb_cos_vs_fp": cos}


def default_ckpts():
    d = Path(ModelArgs.out_dir)
    if (d / "model_best.pth").exists():
        return [str(d / "model_best.pth")]
    cands = sorted(d.glob("best_seed*.pth"))
    if not cands:
        sys.exit(f"no checkpoint in {d} - train first with `python bin/runner.py`")
    return [str(cands[-1])]


def check():
    lin = nn.Linear(8, 8, bias=False)
    before = lin.weight.detach().clone()
    fake_quantize_(lin, 8)
    assert (lin.weight - before).abs().max() < before.abs().max() * 0.02, "int8 should barely move weights"


def sweep(ckpt):
    model, vocab = load_checkpoint(ckpt, device)
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    seqs = load_sequences(ModelArgs.data_file, ModelArgs.length_cutoff)
    _, _, te = random_split(len(seqs), ModelArgs.split_ratios, ModelArgs.split_seed)
    test_seqs = [seqs[i] for i in te]
    loader = DataLoader(precompute(test_seqs, ModelArgs.eval_mask_seed + 1, vocab),
                        batch_size=ModelArgs.batch_size, shuffle=False, collate_fn=partial(pad_batch, vocab=vocab))
    fp_emb = embed(model, test_seqs, vocab)   # reference embeddings

    print(f"\n=== {ckpt}  ({nparams:.2f}M params, device {device}) ===")
    rows = {"fp": metrics(model, loader, vocab, test_seqs, fp_emb)}
    for bits in BITS:
        rows[f"int{bits}"] = metrics(fake_quantize_(copy.deepcopy(model), bits), loader, vocab, test_seqs, fp_emb)

    cols = ["ppl", "top1", "top3", "top5", "biochem_class_acc", "blosum_spearman", "emb_cos_vs_fp"]
    print(f"{'':6}" + "".join(f"{c:>16}" for c in cols))
    for name, r in rows.items():
        print(f"{name:6}" + "".join(f"{r[c]:>16.4f}" for c in cols))
    return {"checkpoint": ckpt, "params_M": nparams, "results": rows}


def main():
    check()
    ckpts = sys.argv[1:] or default_ckpts()
    allres = [sweep(c) for c in ckpts]
    out = Path(ModelArgs.out_dir) / "quant_results.json"
    out.write_text(json.dumps(allres, indent=2))
    print(f"\nsaved -> {out}  ({len(allres)} checkpoint(s))")


if __name__ == "__main__":
    main()
