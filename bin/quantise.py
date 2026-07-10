import os
from pathlib import Path

os.environ.setdefault("WANDB_DIR", str(Path.cwd() / "wandb"))
os.environ.setdefault("WANDB_DATA_DIR", str(Path.cwd() / "wandb_data"))
os.environ.setdefault("WANDB_CACHE_DIR", str(Path.cwd() / "wandb_cache"))

import argparse, copy, json, sys
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from core import (ModelArgs, device, load_checkpoint, load_sequences, random_split,
                  precompute, pad_batch, evaluate, biochemical_breakdown,
                  blosum_correlation, embed)

BITS=[8,4,3,2]
TEST_SIZE=4000

def wlog(data):
    if wandb.run is not None:
        wandb.log(data)

def fake_quant_weight(w,bits):
    qmax=2**(bits-1)-1
    scale=(w.detach().abs().amax(dim=1)/qmax).clamp(min=1e-8)
    zp=torch.zeros(w.shape[0],dtype=torch.int32,device=w.device)
    return torch.fake_quantize_per_channel_affine(w,scale,zp,axis=0,quant_min=-qmax,quant_max=qmax)

@torch.no_grad()
def fake_quantize_(model,bits):
    for m in model.modules():
        if isinstance(m,nn.Linear):
            m.weight.data=fake_quant_weight(m.weight.data,bits)
    return model

def metrics(model,loader,vocab,test_seqs,fp_emb):
    e=evaluate(model,loader,vocab)
    b=biochemical_breakdown(model,loader,vocab)
    s=blosum_correlation(model,loader,vocab)["spearman_offdiag"]
    cos=F.cosine_similarity(embed(model,test_seqs,vocab),fp_emb,dim=1).mean().item()
    return {"ppl":e["perplexity"],"top1":e["top1"],"top3":e["top3"],"top5":e["top5"],
            "biochem_class_acc":b["biochemical_class_accuracy"],
            "blosum_spearman":s,"emb_cos_vs_fp":cos}

def default_ckpts():
    d=Path(ModelArgs.out_dir)
    if (d/"model_best.pth").exists():
        return [str(d/"model_best.pth")]
    cands=sorted(d.glob("best_seed*.pth"))
    if not cands:
        sys.exit(f"no checkpoint in {d}")
    return [str(cands[-1])]

def check():
    lin=nn.Linear(8,8,bias=False)
    before=lin.weight.detach().clone()
    fake_quantize_(lin,8)
    assert (lin.weight-before).abs().max()<before.abs().max()*0.02

def sweep(ckpt):
    model,vocab=load_checkpoint(ckpt,device)
    nparams=sum(p.numel() for p in model.parameters())/1e6
    seqs=load_sequences(ModelArgs.data_file,ModelArgs.length_cutoff)
    _,_,te=random_split(len(seqs),ModelArgs.split_ratios,ModelArgs.split_seed)
    test_seqs=[seqs[i] for i in te[:TEST_SIZE]]
    loader=DataLoader(precompute(test_seqs,ModelArgs.eval_mask_seed+1,vocab),
        batch_size=ModelArgs.batch_size,shuffle=False,
        collate_fn=partial(pad_batch,vocab=vocab))
    fp_emb=embed(model,test_seqs,vocab)
    rows={"fp":metrics(model,loader,vocab,test_seqs,fp_emb)}
    for bits in BITS:
        rows[f"int{bits}"]=metrics(fake_quantize_(copy.deepcopy(model),bits),loader,vocab,test_seqs,fp_emb)
    return {"checkpoint":ckpt,"params_M":nparams,"results":rows}

def log_wandb(res,run_name,group,use_wandb=True):
    rows=res["results"]
    cols=["ppl","top1","top3","top5","biochem_class_acc","blosum_spearman","emb_cos_vs_fp"]
    wandb.init(project=ModelArgs.wandb_project,
               group=group,name=run_name or "quant",
               job_type="quant",
               config={"checkpoint":res["checkpoint"],"params_M":res["params_M"],"bits":BITS},
               mode=None if use_wandb else "disabled")
    tbl=wandb.Table(columns=["format"]+cols)
    for fmt,r in rows.items():
        tbl.add_data(fmt,*[r[c] for c in cols])
        wlog({f"{fmt}/{k}":v for k,v in r.items()})
    wlog({"ptq_sweep":tbl})
    wandb.summary.update({f"{fmt}/{k}":r[k] for fmt,r in rows.items() for k in r})
    wandb.finish()

def main():
    check()
    p=argparse.ArgumentParser()
    p.add_argument("checkpoints",nargs="*")
    p.add_argument("--data_file")
    p.add_argument("--out_dir")
    p.add_argument("--run_name")
    p.add_argument("--wandb_group")
    p.add_argument("--no_wandb",dest="use_wandb",action="store_false",default=True)
    a=p.parse_args()
    if a.data_file: ModelArgs.data_file=a.data_file
    if a.out_dir: ModelArgs.out_dir=a.out_dir
    ckpts=a.checkpoints or default_ckpts()
    allres=[]
    for c in ckpts:
        r=sweep(c)
        allres.append(r)
        log_wandb(r,a.run_name,a.wandb_group,a.use_wandb)
    out=Path(ModelArgs.out_dir)/"quant_results.json"
    out.write_text(json.dumps(allres,indent=2))
    print(f"saved -> {out}")

if __name__=="__main__":
    main()