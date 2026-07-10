import argparse, copy, json, math, random, time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import wandb

from core import (ModelArgs, device, Vocab, load_sequences, random_split, precompute,
                  pad_batch, train_collate, LengthBatchSampler, Transformer, build_arch,
                  save_checkpoint, evaluate, unigram_baseline, biochemical_breakdown, blosum_correlation)


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


def train(model, train_loader, val_loader, vocab, seed):
    model.to(device)
    model.train()
    crit = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=ModelArgs.label_smoothing)
    opt = optim.AdamW(model.parameters(), lr=ModelArgs.learning_rate, weight_decay=ModelArgs.weight_decay)
    sched = make_scheduler(opt, ModelArgs.warmup_steps, ModelArgs.max_steps, ModelArgs.min_lr_ratio)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ModelArgs.ema_decay)) if ModelArgs.use_ema else None

    best_val, best_state = float("inf"), None
    global_step, micro, tic, tok_acc = 0, 0, time.time(), torch.zeros((), device=device)
    opt.zero_grad()

    while global_step < ModelArgs.max_steps:
        for x, y, attn in train_loader:                          # loop the data until the step budget is hit
            x, y, attn = x.to(device), y.to(device), attn.to(device).bool()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(ModelArgs.amp and device == "cuda")):
                loss = crit(model(x, attn).reshape(-1, vocab.num_classes), y.reshape(-1)) / ModelArgs.grad_accum
            loss.backward()
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
                wlog({"train/loss": loss.item() * ModelArgs.grad_accum, "train/lr": sched.get_last_lr()[0],
                      "train/grad_norm": float(gnorm), "train/tok_per_s": tok_s}, global_step)
                tic, _ = time.time(), tok_acc.zero_()

            if global_step % ModelArgs.eval_every == 0:
                em = ema.module if ema else model
                val = evaluate(em, val_loader, vocab)
                improved = val["nll"] < best_val - 1e-4
                if improved:
                    best_val = val["nll"]
                    best_state = copy.deepcopy(em.state_dict())
                    save_checkpoint(Path(ModelArgs.out_dir) / f"best_seed{seed}.pth", em, vocab)
                wlog({"val/ppl": val["perplexity"], "val/top1": val["top1"], "val/top3": val["top3"],
                      "val/top5": val["top5"], "val/nll": val["nll"]}, global_step)
                print(f"  [eval {global_step}] val ppl {val['perplexity']:.3f} | top1 {val['top1']:.2f}% "
                      f"{'*' if improved else ''}")
                model.train()                                    # back to train mode after eval

            if global_step >= ModelArgs.max_steps:
                break

    if best_state is not None:                               # keep the best eval checkpoint; else keep final weights
        model.load_state_dict(best_state)
    return model


def parse_args():
    p = argparse.ArgumentParser()
    for name in ("d_model", "num_heads", "num_layers", "d_ff", "batch_size", "max_steps",
                 "eval_every", "grad_accum", "warmup_steps", "seed", "log_every"):
        p.add_argument(f"--{name}", type=int, default=None)
    for name in ("dropout", "learning_rate", "weight_decay", "label_smoothing",
                 "grad_clip", "ema_decay", "min_lr_ratio", "mask_rate"):
        p.add_argument(f"--{name}", type=float, default=None)
    for name in ("data_file", "out_dir", "run_name", "wandb_group", "wandb_project"):
        p.add_argument(f"--{name}", type=str, default=None)
    p.add_argument("--no_wandb", dest="use_wandb", action="store_false", default=None)
    p.add_argument("--no_length_batching", dest="length_batching", action="store_false", default=None)
    args = p.parse_args()
    for k, v in vars(args).items():                              # only override what was actually passed
        if v is not None:
            setattr(ModelArgs, k, v)


def main():
    parse_args()
    seed = ModelArgs.seed
    seqs = load_sequences(ModelArgs.data_file, ModelArgs.length_cutoff)
    print("Number of unique sequences:", len(seqs))
    vocab = Vocab.from_sequences(seqs)
    print(f"Vocab: {len(vocab.aa_vocab)} tokens, {vocab.num_classes} output classes\n")

    tr, va, te = random_split(len(seqs), ModelArgs.split_ratios, ModelArgs.split_seed)
    train_seqs, val_seqs, test_seqs = ([seqs[i] for i in g] for g in (tr, va, te))
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

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
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
    model = train(model, train_loader, val_loader, vocab, seed)
    save_checkpoint(Path(ModelArgs.out_dir) / f"best_seed{seed}.pth", model, vocab)   # always emit the ckpt Nextflow expects

    tm = evaluate(model, test_loader, vocab)
    bio = biochemical_breakdown(model, test_loader, vocab)
    blosum = blosum_correlation(model, test_loader, vocab)
    print(f"\nTEST  top1 {tm['top1']:.2f}%  ppl {tm['perplexity']:.3f}  "
          f"biochem {bio['biochemical_class_accuracy']:.2f}%  blosum {blosum['spearman_offdiag']:.3f}")

    if wandb.run is not None:
        wandb.summary.update({"test/top1": tm["top1"], "test/top3": tm["top3"], "test/top5": tm["top5"],
                              "test/ppl": tm["perplexity"], "biochem/class_acc": bio["biochemical_class_accuracy"],
                              "blosum/spearman": blosum["spearman_offdiag"],
                              "baseline/unigram_top1": base["top1"], "baseline/unigram_ppl": base["perplexity"]})
    wandb.finish()   # checkpoint is captured by Nextflow publishDir; no wandb artifact needed (and its staging dir is read-only on compute nodes)

    with open(Path(ModelArgs.out_dir) / f"results_seed{seed}.json", "w") as fh:
        json.dump({"baseline": base, "test": tm, "biochemistry": bio, "blosum": blosum, "config": cfg}, fh, indent=2)
    print(f"Saved best_seed{seed}.pth and results_seed{seed}.json to {ModelArgs.out_dir}")


if __name__ == "__main__":
    main()
