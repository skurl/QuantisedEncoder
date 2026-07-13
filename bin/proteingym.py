"""ProteinGym zero-shot variant-effect scoring (ESM masked-marginals protocol).

Score of a mutation <wt><pos><mut>:  logP(mut) - logP(wt) at that position, masked.
Multi-mutants sum their per-position scores. Metric = Spearman(model_score, DMS_score) per assay.

Efficiency: we mask each *position* of the wild-type once (batched) and cache the logprob
vector -> O(L) forwards per assay, not one per variant. This is the standard ESM trick.

Data: point --dms_dir at a folder of ProteinGym substitution assay CSVs (columns `mutant`,
`mutated_sequence`, `DMS_score`). The wild-type is reconstructed from any row by reverting its
mutations, so no separate reference file is needed.

    python bin/proteingym.py results/best_s42/best_seed42.pth results/qat_s42/best_seed42.pth \
        --dms_dir ProteinGym/substitutions --names fp,qat_int2

Fungal-only model on a pan-organism benchmark: start with the fungal/eukaryotic assays.
"""
import argparse, csv, json
from pathlib import Path

import torch
import torch.nn.functional as F

from core import device, load_checkpoint, Vocab, Transformer, build_arch, _spearman


def parse_mutation(m):                        # "A25G" -> (wt='A', pos0=24, mut='G')
    return m[0], int(m[1:-1]) - 1, m[-1]


def wt_from_row(mutated_sequence, mutant):    # revert the variant's mutations -> wild-type
    seq = list(mutated_sequence)
    for a, p, _ in (parse_mutation(x) for x in mutant.split(":")):
        seq[p] = a
    return "".join(seq)


@torch.no_grad()
def wt_position_logprobs(model, vocab, wt, chunk=64):
    # mask each position once (batched) -> [L, num_classes] log-softmax over the 20 output AAs
    base = [vocab.cls] + [vocab.stoi.get(a, vocab.unk) for a in wt] + [vocab.eos]
    L = len(wt)
    out = torch.empty(L, vocab.num_classes)
    for s in range(0, L, chunk):
        idx = list(range(s, min(s + chunk, L)))
        batch = torch.tensor([base] * len(idx), device=device)
        for r, p in enumerate(idx):
            batch[r, p + 1] = vocab.mask          # +1 for the cls prefix
        logits = model(batch, torch.ones_like(batch, dtype=torch.bool))
        for r, p in enumerate(idx):
            out[p] = F.log_softmax(logits[r, p + 1], dim=-1).cpu()
    return out


def score_all(cache, vocab, wt, mutants):
    scores = []
    for m in mutants:
        s = 0.0
        for a, p, b in (parse_mutation(x) for x in m.split(":")):
            if a != wt[p] or b not in vocab.out_stoi or a not in vocab.out_stoi:
                s = float("nan"); break          # position mismatch / non-standard AA -> drop this variant
            s += (cache[p][vocab.out_stoi[b]] - cache[p][vocab.out_stoi[a]]).item()
        scores.append(s)
    return scores


def run_assay(models, vocab, csv_path, max_len, baselines=()):
    rows = list(csv.DictReader(open(csv_path)))
    if not rows or "mutated_sequence" not in rows[0]:
        return None
    mutants = [r["mutant"] for r in rows]
    dms = [float(r["DMS_score"]) for r in rows]
    wt = wt_from_row(rows[0]["mutated_sequence"], rows[0]["mutant"])
    if len(wt) > max_len:
        return {"assay": Path(csv_path).stem, "n": len(mutants), "L": len(wt), "skipped": "too_long"}
    res = {"assay": Path(csv_path).stem, "n": len(mutants), "L": len(wt)}
    for name, model in models.items():
        pred = score_all(wt_position_logprobs(model, vocab, wt), vocab, wt, mutants)
        keep = [(p, d) for p, d in zip(pred, dms) if p == p]          # drop NaN variants
        res[name] = _spearman(*zip(*keep)) if len(keep) > 2 else float("nan")
    for col in baselines:                                             # precomputed score columns (e.g. ESM2_8M)
        if col not in rows[0]:
            continue
        pairs = []
        for r, d in zip(rows, dms):
            try: v = float(r[col])
            except (ValueError, KeyError): continue
            if v == v: pairs.append((v, d))
        res[col] = _spearman(*zip(*pairs)) if len(pairs) > 2 else float("nan")
    return res


def demo():                                   # invariants: identity mutation scores 0, WT reconstructs
    seqs = ["ACDEFGHIKLMNPQRSTVWY"]
    vocab = Vocab.from_sequences(seqs)
    model = Transformer(build_arch(vocab)).to(device).eval()
    cache = wt_position_logprobs(model, vocab, seqs[0])
    assert abs(score_all(cache, vocab, seqs[0], ["A1A"])[0]) < 1e-6, "identity mutation must score 0"
    assert wt_from_row("GCDEFGHIKLMNPQRSTVWY", "A1G") == seqs[0], "WT reconstruction failed"
    print("demo OK (identity=0, WT reconstructs)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="*")                          # fp first, then quant(s)
    p.add_argument("--dms"); p.add_argument("--dms_dir")
    p.add_argument("--names"); p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--match", help="comma-sep substrings; keep only assays whose filename contains one (e.g. YEAST,RHOTO,LIPST)")
    p.add_argument("--baseline", help="comma-sep precomputed score columns to also Spearman (e.g. ESM2_8M) -- needs the zero-shot-scores CSVs")
    p.add_argument("--out", default="proteingym_results.csv")
    p.add_argument("--summary_json", help="write {model: mean_spearman} here (for the Nextflow REPORT stage)")
    p.add_argument("--demo", action="store_true")
    a = p.parse_args()
    if a.demo or not a.checkpoints:
        return demo()

    names = a.names.split(",") if a.names else [Path(c).parent.name for c in a.checkpoints]
    models, vocab = {}, None
    for name, c in zip(names, a.checkpoints):
        m, v = load_checkpoint(c, device); m.eval()
        models[name] = m; vocab = vocab or v

    files = [Path(a.dms)] if a.dms else sorted(Path(a.dms_dir).glob("*.csv"))
    if a.match:                                                     # e.g. --match YEAST,RHOTO,LIPST for the fungal subset
        pats = [s.lower() for s in a.match.split(",")]
        files = [f for f in files if any(s in f.stem.lower() for s in pats)]
    baselines = a.baseline.split(",") if a.baseline else []
    rows = []
    for f in files:
        r = run_assay(models, vocab, f, a.max_len, baselines)
        if r: rows.append(r); print(r)

    scored = [r for r in rows if "skipped" not in r]
    print(f"\n{len(scored)}/{len(files)} assays scored (rest too long)")
    means = {}
    for name in list(models) + baselines:                            # mean Spearman across assays
        vals = [r[name] for r in scored if name in r and r[name] == r[name]]
        means[name] = sum(vals) / len(vals) if vals else None
        if vals:
            print(f"  mean Spearman [{name}] = {means[name]:.4f}  (n={len(vals)})")
    if a.summary_json:
        Path(a.summary_json).write_text(json.dumps(means, indent=2))
    if scored:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(scored[0].keys())); w.writeheader(); w.writerows(scored)
        print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
