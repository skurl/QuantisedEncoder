import argparse, copy, json, math, random, time
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import wandb

from core import (ModelArgs, device, Vocab, load_sequences, random_split, precompute,
                  pad_batch, train_collate, LengthBatchSampler, Transformer, build_arch,
                  save_checkpoint, evaluate, unigram_baseline, biochemical_breakdown, blosum_correlation,
                  blosum_correlation_unigram, apply_qat, bake_qat, load_teacher, teacher_reps, distill_loss)
from proteingym import load_panel, score_panel


def wlog(data, step=None):   # no-op when there's no active run (disabled mode / not inited)
    if wandb.run is not None:
        wandb.log(data, step=step)


def make_scheduler(opt, warmup, total, min_ratio):
    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return max(min_ratio, 0.5 * (1 + math.cos(math.pi * prog)))
    return optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def train(model, train_loader, val_loader, vocab, seed, teacher=None, tok=None, proj=None, panel=()):
    model.to(device)
    model.train()
    crit = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=ModelArgs.label_smoothing)
    params = list(model.parameters()) + (list(proj.parameters()) if proj is not None else [])
    opt = optim.AdamW(params, lr=ModelArgs.learning_rate, weight_decay=ModelArgs.weight_decay)
    sched = make_scheduler(opt, ModelArgs.warmup_steps, ModelArgs.max_steps, ModelArgs.min_lr_ratio)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ModelArgs.ema_decay)) if ModelArgs.use_ema else None
    specials = torch.tensor([vocab.cls, vocab.eos, vocab.pad, vocab.mask, vocab.unk], device=device)

    best_score, best_state = float("-inf"), None            # higher = better (ModelArgs.select_metric; ppl is negated)
    global_step, micro, tic, tok_acc = 0, 0, time.time(), torch.zeros((), device=device)
    opt.zero_grad()

    while global_step < ModelArgs.max_steps:
        for x, y, attn, seqs in train_loader:                    # loop the data until the step budget is hit
            x, y, attn = x.to(device), y.to(device), attn.to(device).bool()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(ModelArgs.amp and device == "cuda")):
                rep = model(x, attn, return_repr=True)           # one forward -> reps for both MLM head and distill
                loss = crit(model.fc(rep).reshape(-1, vocab.num_classes), y.reshape(-1))
                if teacher is not None:                          # + match the teacher's per-residue reps
                    valid = attn & (y == -100) & ~torch.isin(x, specials)   # unmasked real residues only
                    loss = loss + ModelArgs.distill_weight * distill_loss(rep, proj, teacher_reps(teacher, tok, seqs), valid)
            (loss / ModelArgs.grad_accum).backward()
            tok_acc += (y != -100).sum()
            micro += 1
            if micro % ModelArgs.grad_accum != 0:
                continue
            gnorm = clip_grad_norm_(model.parameters(), ModelArgs.grad_clip)     # returned norm = stability signal
            opt.step()
            opt.zero_grad()
            sched.step()
            if ema:
                ema.update_parameters(model)
            global_step += 1

            if global_step % ModelArgs.log_every == 0:
                el = time.time() - tic
                tok_s = tok_acc.item() / max(el, 1e-6)
                eta = (ModelArgs.max_steps - global_step) * (el / ModelArgs.log_every) / 60
                print(f"  step {global_step:6d}/{ModelArgs.max_steps} | lr {sched.get_last_lr()[0]:.2e} | "
                      f"grad {gnorm:.2f} | {tok_s:.0f} tok/s | eta {eta:.1f}m")
                wlog({"train/loss": loss.item(), "train/lr": sched.get_last_lr()[0],
                      "train/grad_norm": float(gnorm), "train/tok_per_s": tok_s}, global_step)
                tic, _ = time.time(), tok_acc.zero_()

            if global_step % ModelArgs.eval_every == 0:
                em = ema.module if ema else model
                val = evaluate(em, val_loader, vocab)
                logs = {"val/ppl": val["perplexity"], "val/top1": val["top1"], "val/top3": val["top3"],
                        "val/top5": val["top5"], "val/nll": val["nll"]}
                metric = ModelArgs.select_metric
                if metric == "pgym" and panel:                   # held-out ProteinGym (only sensible with a domain-matched panel)
                    score = score_panel(em, vocab, panel); logs["val/pgym_panel"] = score
                elif metric == "blosum":                         # leakage-resistant biochemistry (extra val pass)
                    score = blosum_correlation(em, val_loader, vocab)["spearman_offdiag"]; logs["val/blosum"] = score
                else:                                            # "ppl" (default): lowest perplexity -> negate so higher = better
                    score = -val["perplexity"]
                improved = score > best_score + 1e-4
                if improved:
                    best_score = score
                    best_state = copy.deepcopy(em.state_dict())
                    save_checkpoint(Path(ModelArgs.out_dir) / f"best_seed{seed}.pth", em, vocab)
                wlog(logs, global_step)
                print(f"  [eval {global_step}] ppl {val['perplexity']:.3f} | top1 {val['top1']:.2f}% "
                      f"| best-on-{metric} {'*' if improved else ''}")
                model.train()                                    # back to train mode after eval

            if global_step >= ModelArgs.max_steps:
                break

    if best_state is not None:                               # keep the best eval checkpoint; else keep final weights
        model.load_state_dict(best_state)
    return model


def _coerce(default, raw):        # wandb agent passes every arg as a --key=value string
    if isinstance(default, bool): return raw.lower() in ("1", "true", "yes")
    if isinstance(default, int):  return int(raw)
    if isinstance(default, float): return float(raw)
    return raw                    # str, or None-defaulted (run_name/wandb_group)


def parse_args():                 # accept EVERY sweepable ModelArgs field, typed off its default
    fields = {k: v for k, v in vars(ModelArgs).items()
              if not k.startswith("__") and not callable(v) and not isinstance(v, tuple)}
    p = argparse.ArgumentParser()
    for name in fields:
        p.add_argument(f"--{name}")                             # raw string, coerced below
    for name, raw in vars(p.parse_args()).items():              # only override what was passed
        if raw is not None:
            setattr(ModelArgs, name, _coerce(fields[name], raw))


def main():
    parse_args()
    print(f"[device] {device}")                                  # cuda = GPU, cpu = no GPU allocated
    seed = ModelArgs.seed
    seqs = load_sequences(ModelArgs.data_file, ModelArgs.length_cutoff)
    print("Number of unique sequences:", len(seqs))
    vocab = Vocab.from_sequences(seqs)
    print(f"Vocab: {len(vocab.aa_vocab)} tokens, {vocab.num_classes} output classes\n")

    tr, va, te = random_split(len(seqs), ModelArgs.split_ratios, ModelArgs.split_seed)
    train_seqs, val_seqs, test_seqs = ([seqs[i] for i in g] for g in (tr, va, te))
    if ModelArgs.eval_max_seqs:                              # cap eval sets: 10% of a huge corpus makes every eval crawl
        val_seqs, test_seqs = val_seqs[:ModelArgs.eval_max_seqs], test_seqs[:ModelArgs.eval_max_seqs]
        print(f"[eval] capped val/test to {ModelArgs.eval_max_seqs} sequences each")
    collate = partial(pad_batch, vocab=vocab)
    val_loader = DataLoader(precompute(val_seqs, ModelArgs.eval_mask_seed, vocab),
                            batch_size=ModelArgs.batch_size, shuffle=False, collate_fn=collate)
    test_data = precompute(test_seqs, ModelArgs.eval_mask_seed + 1, vocab)
    test_loader = DataLoader(test_data, batch_size=ModelArgs.batch_size, shuffle=False, collate_fn=collate)

    Path(ModelArgs.out_dir).mkdir(parents=True, exist_ok=True)
    base = unigram_baseline(train_seqs, test_data, vocab)
    print(f"[baseline] unigram top1 {base['top1']:.2f}%  ppl {base['perplexity']:.3f}")

    cfg = {k: v for k, v in vars(ModelArgs).items() if not k.startswith("__")}
    wandb.init(project=ModelArgs.wandb_project,
               group=ModelArgs.wandb_group or f"d{ModelArgs.d_model}_L{ModelArgs.num_layers}",
               name=ModelArgs.run_name or f"seed{seed}",
               config=cfg, mode=None if ModelArgs.use_wandb else "disabled")
    if wandb.run is not None:
        wandb.define_metric("val/ppl", summary="min")          # default selector: track the lowest perplexity
        wandb.define_metric("val/pgym_panel", summary="max")   # alternate selector (select_metric="pgym")
        wandb.define_metric("val/blosum", summary="max")       # alternate selector (select_metric="blosum")
        wandb.define_metric("val/top1", summary="max")         # monitoring only

    random.seed(seed); torch.manual_seed(seed)
    collate_tr = partial(train_collate, vocab=vocab)
    if ModelArgs.length_batching:
        sampler = LengthBatchSampler([len(s) for s in train_seqs], ModelArgs.batch_size, seed)
        train_loader = DataLoader(train_seqs, batch_sampler=sampler, collate_fn=collate_tr)
    else:
        g = torch.Generator(); g.manual_seed(seed)
        train_loader = DataLoader(train_seqs, batch_size=ModelArgs.batch_size, shuffle=True,
                                  collate_fn=collate_tr, generator=g)
    model = Transformer(build_arch(vocab))
    print(f"[model] {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    if ModelArgs.init_ckpt:                                       # QAT fine-tune: start from a trained fp checkpoint
        model.load_state_dict(torch.load(ModelArgs.init_ckpt, map_location=device, weights_only=False)["model"])
        print(f"[init] loaded weights from {ModelArgs.init_ckpt}")
    if ModelArgs.qat_bits:                                        # fake-quant Linear+Embedding weights during training
        apply_qat(model, ModelArgs.qat_bits)
        print(f"[qat] int{ModelArgs.qat_bits} weight-only, straight-through")
    teacher = tok = proj = None
    if ModelArgs.distill_teacher:                                 # borrow a big model's representations (any HF encoder id)
        teacher, tok, t_dim = load_teacher(ModelArgs.distill_teacher)
        proj = nn.Linear(ModelArgs.d_model, t_dim).to(device)     # student d -> teacher d, distill-only, not saved
        print(f"[distill] teacher {ModelArgs.distill_teacher} (d={t_dim}), weight {ModelArgs.distill_weight}")
    panel = (load_panel(ModelArgs.pgym_panel, ModelArgs.pgym_panel_match)
             if ModelArgs.pgym_panel and ModelArgs.select_metric == "pgym" else [])   # skip the (fungal) panel unless selecting on it
    if panel:
        print(f"[pgym panel] selecting on {len(panel)} assays: {[a for a, *_ in panel]}")
    model = train(model, train_loader, val_loader, vocab, seed, teacher, tok, proj, panel)
    if ModelArgs.qat_bits:                                        # collapse fake-quant -> weights hold the int values
        bake_qat(model)
    save_checkpoint(Path(ModelArgs.out_dir) / f"best_seed{seed}.pth", model, vocab)   # always emit the ckpt Nextflow expects

    tm = evaluate(model, test_loader, vocab)
    bio = biochemical_breakdown(model, test_loader, vocab)
    blosum = blosum_correlation(model, test_loader, vocab)
    blosum_null = blosum_correlation_unigram(train_seqs, vocab)   # frequency-only floor for the BLOSUM claim
    print(f"\nTEST  top1 {tm['top1']:.2f}%  ppl {tm['perplexity']:.3f}  "
          f"biochem {bio['biochemical_class_accuracy']:.2f}%  "
          f"blosum {blosum['spearman_offdiag']:.3f} (freq-null {blosum_null:.3f})")

    if wandb.run is not None:
        wandb.summary.update({"test/top1": tm["top1"], "test/top3": tm["top3"], "test/top5": tm["top5"],
                              "test/ppl": tm["perplexity"], "biochem/class_acc": bio["biochemical_class_accuracy"],
                              "blosum/spearman": blosum["spearman_offdiag"], "blosum/spearman_null": blosum_null,
                              "baseline/unigram_top1": base["top1"], "baseline/unigram_ppl": base["perplexity"]})
    wandb.finish()   # checkpoint is captured by Nextflow publishDir; no wandb artifact needed (and its staging dir is read-only on compute nodes)

    with open(Path(ModelArgs.out_dir) / f"results_seed{seed}.json", "w") as fh:
        json.dump({"baseline": base, "test": tm, "biochemistry": bio, "blosum": blosum,
                   "blosum_null": blosum_null, "config": cfg}, fh, indent=2)
    print(f"Saved best_seed{seed}.pth and results_seed{seed}.json to {ModelArgs.out_dir}")


if __name__ == "__main__":
    main()
