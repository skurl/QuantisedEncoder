import argparse, copy, json, math
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core import (ModelArgs, device, load_checkpoint, load_sequences,
                  random_split, precompute, pad_batch, Transformer, Vocab)


def layer_hiddens(model, x, attn):
    with torch.no_grad():
        h = model.dropout(model.embed(x))          # dropout is a no-op in eval; kept to mirror forward()
        hs = [h]
        for layer in model.layers:
            h = layer(h, attn)
            hs.append(h)
    return hs


def make_heads(n_points, d_model, n_classes):
    # tuned lens: per-layer LayerNorm (absorbs cross-layer scale differences) + linear unembedding
    return nn.ModuleList([nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))
                          for _ in range(n_points)]).to(device)


def fit_probes(model, loader, vocab, steps, lr=1e-3):
    model.eval()
    d, C = model.arch["d_model"], vocab.num_classes
    heads = make_heads(model.arch["num_layers"] + 1, d, C)
    opt = torch.optim.AdamW(heads.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    step = 0
    while step < steps:
        for x, y, attn in loader:
            x, y, attn = x.to(device), y.to(device), attn.to(device).bool()
            hs = layer_hiddens(model, x, attn)
            yy = y.reshape(-1)
            loss = sum(ce(head(h).reshape(-1, C), yy) for head, h in zip(heads, hs))
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step >= steps:
                break
    return heads


@torch.no_grad()
def eval_probes(model, heads, loader, vocab):
    """Per-layer masked-LM NLL + perplexity, same masked-token protocol as core.evaluate()."""
    model.eval()
    C = vocab.num_classes
    ce = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    nll = [0.0] * len(heads)
    tok = 0
    for x, y, attn in loader:
        x, y, attn = x.to(device), y.to(device), attn.to(device).bool()
        hs = layer_hiddens(model, x, attn)
        yy = y.reshape(-1)
        tok += int((yy != -100).sum())
        for i, (head, h) in enumerate(zip(heads, hs)):
            nll[i] += ce(head(h).reshape(-1, C).float(), yy).item()
    tok = max(1, tok)
    return [{"layer": i, "nll": nll[i] / tok, "perplexity": math.exp(nll[i] / tok)} for i in range(len(heads))]


def _loader(seqs, seed, vocab):
    return DataLoader(precompute(seqs, seed, vocab), batch_size=ModelArgs.batch_size,
                      shuffle=False, collate_fn=partial(pad_batch, vocab=vocab))


def fake_quant_copy(model, bits):
    """PTQ variant: per-channel int-`bits` on every Linear AND the embedding (embed-included, i.e. the
    'true all-intN' the cliff finding is about -- not the Linear-only sweep)."""
    from core import fake_quant_weight
    m = copy.deepcopy(model)
    for mod in m.modules():
        if isinstance(mod, (nn.Linear, nn.Embedding)):
            mod.weight.data = fake_quant_weight(mod.weight.data, bits)
    return m


def run(args):
    trained, vocab = load_checkpoint(args.checkpoint, device)
    seqs = load_sequences(args.data_file, ModelArgs.length_cutoff)
    tr, _, te = random_split(len(seqs), ModelArgs.split_ratios, ModelArgs.split_seed)
    cap = ModelArgs.eval_max_seqs or len(seqs)
    train_loader = _loader([seqs[i] for i in tr[:cap]], ModelArgs.eval_mask_seed + 2, vocab)   # probe FIT
    test_loader = _loader([seqs[i] for i in te[:cap]], ModelArgs.eval_mask_seed + 1, vocab)     # probe EVAL

    extra = {}                                            # e.g. ubiquitin WT sequences
    if args.seqs:
        xseqs = load_sequences(args.seqs, ModelArgs.length_cutoff)   # same reader; e.g. ubiquitin WT FASTA
        extra[Path(args.seqs).stem] = _loader(xseqs, ModelArgs.eval_mask_seed + 1, vocab)

    variants = {"trained": trained,
                "untrained": Transformer(trained.arch).to(device)}   # true fresh __init__, identical config
    for bits in (args.quant.split(",") if args.quant else []):
        variants[f"int{bits.strip().lstrip('int')}"] = fake_quant_copy(trained, int(bits.strip().lstrip("int")))
    for spec in args.extra_ckpt or []:                    # name=path, e.g. qat_int2=results/.../qat.pth
        name, path = spec.split("=", 1)
        variants[name] = load_checkpoint(path, device)[0]

    out = {"checkpoint": args.checkpoint, "dataset": trained.arch.get("dataset", "?"),
           "hou_band": 1.2, "variants": {}}
    for name, m in variants.items():
        heads = fit_probes(m, train_loader, vocab, args.probe_steps)
        res = {"test": eval_probes(m, heads, test_loader, vocab)}
        for xname, xloader in extra.items():
            res[xname] = eval_probes(m, heads, xloader, vocab)
        out["variants"][name] = res
        fin = res["test"][-1]
        print(f"[{name:12}] final-layer test nll {fin['nll']:.3f} ppl {fin['perplexity']:.2f}"
              + "".join(f" | {x} nll {res[x][-1]['nll']:.3f}" for x in extra))

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"saved -> {args.out}")
    if args.fig:
        plot(out, args.fig, list(extra))


def plot(out, path, extra):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[fig] matplotlib not available -> skipped (JSON has the numbers)"); return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, res in out["variants"].items():
        xs = [r["layer"] for r in res["test"]]
        ax.plot(xs, [r["nll"] for r in res["test"]], marker="o", label=f"{name} (test)")
        for x in extra:                                  # ubiquitin dashed
            ax.plot(xs, [r["nll"] for r in res[x]], marker="^", ls="--", label=f"{name} ({x})")
    ax.axhline(out["hou_band"], color="grey", ls=":", lw=1)
    ax.text(0, out["hou_band"], " Hou fitness band (~1.2, final-layer ref)", color="grey", va="bottom", fontsize=8)
    ax.set_xlabel("layer (0 = embedding output)"); ax.set_ylabel("tuned-lens masked-LM NLL")
    ax.set_title("Per-layer NLL by checkpoint variant"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140)
    print(f"saved -> {path}")


def overlay(paths, figpath):
    """Replot several existing layerwise JSONs on one axis (e.g. fungal vs eukaryotic). No compute."""
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        print("[fig] matplotlib not available -> skipped"); return
    fig, ax = plt.subplots(figsize=(7.5, 4.5)); band = 1.2
    for p in paths:
        d = json.loads(Path(p).read_text()); band = d.get("hou_band", band)
        tag = d.get("dataset", Path(p).stem)
        for name, res in d["variants"].items():
            for split, curve in res.items():
                xs = [r["layer"] for r in curve]
                ax.plot(xs, [r["nll"] for r in curve], marker="o",
                        ls="-" if split == "test" else "--", label=f"{tag}:{name}:{split}")
    ax.axhline(band, color="grey", ls=":", lw=1)
    ax.set_xlabel("layer (0 = embedding output)"); ax.set_ylabel("tuned-lens masked-LM NLL")
    ax.set_title("Per-layer NLL — overlay"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figpath, dpi=140); print(f"saved -> {figpath}")


def check():
    """Self-test on random data: probe FITS (loss drops), NLL is finite, and a probe on a backbone
    fed real signal beats one on pure noise. No checkpoint or data needed."""
    torch.manual_seed(0)
    vocab = Vocab.from_sequences(["ACDEFGHIKLMNPQRSTVWY" * 3])
    arch = {"vocab_size": len(vocab.aa_vocab), "d_model": 32, "num_heads": 4, "num_layers": 3,
            "d_ff": 32, "dropout": 0, "num_classes": vocab.num_classes, "pad_idx": vocab.pad}
    model = Transformer(arch).to(device)
    seqs = ["".join(vocab.classes[i % vocab.num_classes] for i in range(20 + j)) for j in range(40)]
    loader = _loader(seqs, 1, vocab)
    heads = fit_probes(model, loader, vocab, steps=30)
    res = eval_probes(model, heads, loader, vocab)
    assert len(res) == arch["num_layers"] + 1, "one readout per layer + embedding"
    assert all(math.isfinite(r["nll"]) for r in res), "NLL must be finite"
    # a probe reading the embedding of a fixed-token corpus should beat the log(20) uniform floor
    assert res[0]["nll"] < math.log(vocab.num_classes), "fitted probe should beat the uniform-NLL floor"
    print("check OK:", [round(r["nll"], 2) for r in res])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", nargs="?")
    p.add_argument("--data_file", help="clustered fungal FASTA (same split as the canonical perplexity)")
    p.add_argument("--seqs", help="extra FASTA to probe (e.g. ubiquitin WT sequences)")
    p.add_argument("--quant", help="comma-sep PTQ bit-widths to add as variants, e.g. int8,int4,int2 (embed-included)")
    p.add_argument("--extra_ckpt", action="append", help="name=path extra checkpoint variant (e.g. qat_int2=...)")
    p.add_argument("--probe_steps", type=int, default=500, help="optimizer steps to fit the readout heads")
    p.add_argument("--out", default="layerwise.json")
    p.add_argument("--fig", help="write a per-layer NLL line plot here (PNG); needs matplotlib")
    p.add_argument("--overlay", help="comma-sep existing layerwise JSONs to replot together (e.g. fungal vs euk); needs --fig")
    p.add_argument("--check", action="store_true", help="run the self-test and exit")
    a = p.parse_args()
    if a.check:
        check(); return
    if a.overlay:
        overlay(a.overlay.split(","), a.fig or "overlay.png"); return
    if not a.checkpoint or not a.data_file:
        p.error("checkpoint and --data_file are required (or use --check)")
    run(a)


if __name__ == "__main__":
    main()
