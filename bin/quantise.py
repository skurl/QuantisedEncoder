import copy, json, sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import runner
from runner import (ModelArgs, load_checkpoint, load_sequences, random_split,
                    precompute, pad_batch, evaluate, biochemical_breakdown, blosum_correlation)

BITS = [8, 4, 3, 2]


def bind_vocab(vocab, classes): # reuses runners functions
    runner.aa_vocab = vocab
    runner.aa_stoi = {s: i for i, s in enumerate(vocab)}
    runner.classes = classes
    runner.output_stoi = {a: i for i, a in enumerate(classes)}
    runner.output_itos = {i: a for i, a in enumerate(classes)}
    runner.num_classes = len(classes)
    for name in ("pad", "mask", "cls", "unk", "eos"):
        setattr(runner, f"{name}_idx", vocab.index(f"<{name}>"))


def fake_quant_weight(w, bits): # native fake-quant op == the one QAT uses (has a straight-through backward)
    qmax = 2 ** (bits - 1) - 1
    scale = (w.detach().abs().amax(dim=1) / qmax).clamp(min=1e-8)          # observer, per output channel
    zp = torch.zeros(w.shape[0], dtype=torch.int32, device=w.device)      # symmetric -> zero_point 0
    return torch.fake_quantize_per_channel_affine(w, scale, zp, axis=0, quant_min=-qmax, quant_max=qmax)


@torch.no_grad()
def fake_quantize_(model, bits): # PTQ: overwrite every Linear weight with its fake-quantised version, in place
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.weight.data = fake_quant_weight(m.weight.data, bits)
    return model


def metrics(model, loader):
    runner.test_loader = loader
    e = evaluate(model, loader)
    b = biochemical_breakdown(model)
    s = blosum_correlation(model)["spearman_offdiag"]
    return {"ppl": e["perplexity"], "top1": e["top1"], "top3": e["top3"], "top5": e["top5"],
            "biochem_class_acc": b["biochemical_class_accuracy"], "blosum_spearman": s}


def default_ckpt():
    d = Path(ModelArgs.out_dir)
    if (d / "model_best.pth").exists():
        return str(d / "model_best.pth")
    cands = sorted(d.glob("best_seed*.pth"))
    if not cands:
        sys.exit(f"no checkpoint in {d} - train first with `python bin/runner.py`")
    return str(cands[-1])


def check(): # basically checks if the quantization is working as expected, and that the weights are not changing too much
    lin = nn.Linear(8, 8, bias=False)
    before = lin.weight.detach().clone()
    fake_quantize_(lin, 8)
    assert (lin.weight - before).abs().max() < before.abs().max() * 0.02, "int8 should barely move weights"


def main():
    check()
    ckpt = sys.argv[1] if len(sys.argv) > 1 else default_ckpt()
    model, vocab, classes = load_checkpoint(ckpt, runner.device)
    bind_vocab(vocab, classes)
    print(f"loaded {ckpt}  ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params, device {runner.device})")

    # rebuild the exact held-out test set the model was trained against (seeded and reproducible)
    seqs = load_sequences(ModelArgs.data_file, ModelArgs.length_cutoff)
    null, null, te = random_split(len(seqs), ModelArgs.split_ratios, ModelArgs.split_seed)
    test_data = precompute([seqs[i] for i in te], ModelArgs.eval_mask_seed + 1)
    loader = DataLoader(test_data, batch_size=ModelArgs.batch_size, shuffle=False, collate_fn=pad_batch)

    rows = {"fp": metrics(model, loader)}
    for bits in BITS:
        rows[f"int{bits}"] = metrics(fake_quantize_(copy.deepcopy(model), bits), loader)

    cols = ["ppl", "top1", "top3", "top5", "biochem_class_acc", "blosum_spearman"]
    print(f"\n{'':6}" + "".join(f"{c:>18}" for c in cols))
    for name, r in rows.items():
        print(f"{name:6}" + "".join(f"{r[c]:>18.3f}" for c in cols))

    out = Path(ModelArgs.out_dir) / "quant_results.json"
    out.write_text(json.dumps({"checkpoint": ckpt, "results": rows}, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
