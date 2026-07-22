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


# GPTQ (Frantar et al. 2023): error-compensated PTQ. Quantise each Linear column-by-column, propagating
# each column's rounding error into the not-yet-quantised columns via the inverse input-Hessian. Same
# per-output-channel grid as RTN -> RTN-vs-GPTQ is apples-to-apples (only the assignment differs).
# ponytail: one-pass fp calibration (all Hessians from the fp model), not the full sequential recompute.
#   upgrade path if the int2 recovery underwhelms: re-derive activations after each quantised layer.

@torch.no_grad()
def collect_hessians(model,loader,n_batches=16):
    H={}; hooks=[]
    def mk(name):
        def hook(mod,inp,out):
            x=inp[0].reshape(-1,inp[0].shape[-1]).float()          # [tokens, in_features]
            H[name]=H.get(name,torch.zeros(x.shape[1],x.shape[1],device=x.device))+x.t()@x
        return hook
    for name,m in model.named_modules():
        if isinstance(m,nn.Linear):
            hooks.append(m.register_forward_hook(mk(name)))
    for i,(x,y,attn) in enumerate(loader):
        model(x.to(device),attn.to(device).bool())
        if i+1>=n_batches: break
    for h in hooks: h.remove()
    return H

def _gptq_layer(W,H,bits,damp=0.01):
    W=W.clone().float(); out_ch,in_ch=W.shape
    qmax=2**(bits-1)-1
    scale=(W.abs().amax(dim=1)/qmax).clamp(min=1e-8)                # [out], per-channel, from ORIGINAL W (== RTN grid)
    H=H.clone().float(); idx=torch.arange(in_ch,device=H.device)
    dead=torch.diag(H)==0; H[dead,dead]=1.0; W[:,dead]=0.0         # unused input dims -> don't quantise
    U=None
    for k in (damp,damp*10,damp*100):                              # bump damping until H is PD
        Hd=H.clone(); Hd[idx,idx]+=k*torch.diag(H).mean()
        try: U=torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hd)),upper=True); break
        except RuntimeError: U=None
    if U is None:                                                  # numerically hopeless -> plain RTN
        return torch.clamp(torch.round(W/scale.unsqueeze(1)),-qmax,qmax)*scale.unsqueeze(1)
    Q=torch.zeros_like(W)
    for i in range(in_ch):
        w=W[:,i]
        q=torch.clamp(torch.round(w/scale),-qmax,qmax)*scale       # quantise column i on the per-row grid
        Q[:,i]=q
        err=(w-q)/U[i,i]
        W[:,i:]-=torch.outer(err,U[i,i:])                          # push the error onto the remaining columns
    return Q

@torch.no_grad()
def gptq_quantize_(model,bits,hessians):
    for name,m in model.named_modules():
        if isinstance(m,nn.Linear) and name in hessians:
            m.weight.data=_gptq_layer(m.weight.data,hessians[name],bits).to(m.weight.dtype)
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
    net=nn.Sequential(nn.Linear(8,8,bias=False),nn.Linear(8,8,bias=False))   # quantize_layer_ isolates its target
    w0,w1=net[0].weight.detach().clone(),net[1].weight.detach().clone()
    quantize_layer_(net,"0",2)
    assert torch.equal(net[1].weight,w1), "quantize_layer_ leaked into another layer"
    assert not torch.equal(net[0].weight,w0), "quantize_layer_ did not change its target"
    X=torch.randn(50,8); Q,_=torch.linalg.qr(torch.randn(8,8))   # CKA must be 1 for a rotated copy
    assert abs(_cka(X,X)-1)<1e-4 and abs(_cka(X,X@Q)-1)<1e-4, "linear CKA must be rotation-invariant"
    W=torch.randn(6,8)                                           # GPTQ with an identity Hessian must reduce to plain per-channel RTN
    assert torch.allclose(fake_quant_weight(W.clone(),4),_gptq_layer(W.clone(),torch.eye(8),4),atol=1e-5), "GPTQ(H=I) must equal RTN"

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
    return {"checkpoint":ckpt,"dataset":model.arch.get("dataset","?"),"params_M":nparams,"results":rows}

def gptq_sweep(ckpt):
    # RTN vs GPTQ at each bit-width (Linear-only, same as sweep) -> does error compensation recover the int2 cliff?
    model,vocab=load_checkpoint(ckpt,device)
    nparams=sum(p.numel() for p in model.parameters())/1e6
    seqs=load_sequences(ModelArgs.data_file,ModelArgs.length_cutoff)
    _,_,te=random_split(len(seqs),ModelArgs.split_ratios,ModelArgs.split_seed)
    test_seqs=[seqs[i] for i in te[:TEST_SIZE]]
    loader=DataLoader(precompute(test_seqs,ModelArgs.eval_mask_seed+1,vocab),
        batch_size=ModelArgs.batch_size,shuffle=False,collate_fn=partial(pad_batch,vocab=vocab))
    fp_emb=embed(model,test_seqs,vocab)
    H=collect_hessians(model,loader)                                   # calibration Hessians from fp activations
    rows={"fp":metrics(model,loader,vocab,test_seqs,fp_emb)}
    for bits in BITS:
        rows[f"rtn_int{bits}"]=metrics(fake_quantize_(copy.deepcopy(model),bits),loader,vocab,test_seqs,fp_emb)
        rows[f"gptq_int{bits}"]=metrics(gptq_quantize_(copy.deepcopy(model),bits,H),loader,vocab,test_seqs,fp_emb)
    print(f"\n=== RTN vs GPTQ (Linear-only PTQ): {ckpt}  (fp top1 {rows['fp']['top1']:.2f}) ===")
    print(f"{'variant':14}{'top1':>8}{'ppl':>9}{'blosum':>9}{'emb_cos':>9}")
    for name,r in rows.items():
        print(f"{name:14}{r['top1']:>8.2f}{r['ppl']:>9.3f}{r['blosum_spearman']:>9.3f}{r['emb_cos_vs_fp']:>9.3f}")
    return {"checkpoint":ckpt,"dataset":model.arch.get("dataset","?"),"params_M":nparams,"method":"rtn_vs_gptq","results":rows}

def quantize_layer_(model,name,bits):   # quantize ONE named layer's weight in place
    for n,m in model.named_modules():
        if n==name:
            m.weight.data=fake_quant_weight(m.weight.data,bits)
    return model

def sensitivity(ckpt,bits=2):
    # quantize each weight layer ALONE to int{bits}, rest fp -> rank layers by damage.
    # answers: is the int2 cliff a few fragile layers (concentrated) or all at once (diffuse)?
    model,vocab=load_checkpoint(ckpt,device)
    seqs=load_sequences(ModelArgs.data_file,ModelArgs.length_cutoff)
    _,_,te=random_split(len(seqs),ModelArgs.split_ratios,ModelArgs.split_seed)
    test_seqs=[seqs[i] for i in te[:TEST_SIZE]]
    loader=DataLoader(precompute(test_seqs,ModelArgs.eval_mask_seed+1,vocab),
        batch_size=ModelArgs.batch_size,shuffle=False,collate_fn=partial(pad_batch,vocab=vocab))
    fp_emb=embed(model,test_seqs,vocab)
    fp=metrics(model,loader,vocab,test_seqs,fp_emb)
    # every weight layer; flag which the whole-model PTQ sweep touched (Linear only -> embed was left fp!)
    targets=[(n,isinstance(m,nn.Linear)) for n,m in model.named_modules() if isinstance(m,(nn.Linear,nn.Embedding))]
    rows=[]
    for name,in_sweep in targets:
        r=metrics(quantize_layer_(copy.deepcopy(model),name,bits),loader,vocab,test_seqs,fp_emb)
        rows.append({"layer":name,"in_ptq_sweep":in_sweep,"top1":r["top1"],"ppl":r["ppl"],
                     "d_top1":fp["top1"]-r["top1"],"blosum":r["blosum_spearman"],"emb_cos":r["emb_cos_vs_fp"]})
    rows.sort(key=lambda x:x["d_top1"],reverse=True)   # most damage first
    print(f"\n=== per-layer int{bits} sensitivity: {ckpt}  (fp top1 {fp['top1']:.2f}) ===")
    print(f"{'layer':30}{'in_sweep':9}{'dtop1':>8}{'top1':>8}{'ppl':>9}{'emb_cos':>9}")
    for r in rows:
        print(f"{r['layer']:30}{str(r['in_ptq_sweep']):9}{r['d_top1']:>8.2f}{r['top1']:>8.2f}{r['ppl']:>9.3f}{r['emb_cos']:>9.3f}")
    return {"checkpoint":ckpt,"bits":bits,"fp":fp,"layers":rows}

def mixed(ckpt,keep,lo=2,hi=8):
    # int{lo} everything EXCEPT the `keep` layers (kept at int{hi}) -> how much of the cliff protecting them buys back
    model,vocab=load_checkpoint(ckpt,device)
    seqs=load_sequences(ModelArgs.data_file,ModelArgs.length_cutoff)
    _,_,te=random_split(len(seqs),ModelArgs.split_ratios,ModelArgs.split_seed)
    test_seqs=[seqs[i] for i in te[:TEST_SIZE]]
    loader=DataLoader(precompute(test_seqs,ModelArgs.eval_mask_seed+1,vocab),
        batch_size=ModelArgs.batch_size,shuffle=False,collate_fn=partial(pad_batch,vocab=vocab))
    fp_emb=embed(model,test_seqs,vocab)
    def q(keepset):   # quantize a fresh copy: kept layers -> hi bits, everything else -> lo bits
        m=copy.deepcopy(model)
        for n,mod in m.named_modules():
            if isinstance(mod,(nn.Linear,nn.Embedding)):
                mod.weight.data=fake_quant_weight(mod.weight.data, hi if n in keepset else lo)
        return m
    rows={"fp":metrics(model,loader,vocab,test_seqs,fp_emb),
          f"all_int{lo}":metrics(q(set()),loader,vocab,test_seqs,fp_emb),            # true all-int{lo}, embed included
          f"keep[{'+'.join(keep)}]@int{hi}":metrics(q(set(keep)),loader,vocab,test_seqs,fp_emb)}
    cols=["top1","ppl","blosum_spearman","emb_cos_vs_fp"]
    print(f"\n=== mixed precision (int{lo} except {keep}@int{hi}): {ckpt} ===")
    print(f"{'config':28}"+"".join(f"{c:>16}" for c in cols))
    for name,r in rows.items():
        print(f"{name:28}"+"".join(f"{r[c]:>16.4f}" for c in cols))
    return {"checkpoint":ckpt,"keep":keep,"lo":lo,"hi":hi,"results":rows}

def _cka(X,Y):   # linear CKA: representation similarity that is INVARIANT to rotation/scale (Kornblith 2019)
    X=X-X.mean(0,keepdim=True); Y=Y-Y.mean(0,keepdim=True)
    return ((X.T@Y).norm()**2/((X.T@X).norm()*(Y.T@Y).norm())).item()

def emb_cos_vs(ckpt,ref_ckpt):
    # compare a checkpoint (e.g. QAT-int2) to a reference fp model on pooled embeddings.
    # cos = basis-DEPENDENT (drops under rotation); CKA = basis-INVARIANT -> disambiguates rotation vs real loss.
    model,vocab=load_checkpoint(ckpt,device)
    ref,_=load_checkpoint(ref_ckpt,device)
    seqs=load_sequences(ModelArgs.data_file,ModelArgs.length_cutoff)
    _,_,te=random_split(len(seqs),ModelArgs.split_ratios,ModelArgs.split_seed)
    test_seqs=[seqs[i] for i in te[:TEST_SIZE]]
    A,B=embed(model,test_seqs,vocab),embed(ref,test_seqs,vocab)
    cos=F.cosine_similarity(A,B,dim=1).mean().item()
    cka=_cka(A,B)
    print(f"{ckpt} vs {ref_ckpt}:  emb_cos={cos:.4f}  linear_CKA={cka:.4f}")
    return {"checkpoint":ckpt,"ref":ref_ckpt,"emb_cos_vs_fp":cos,"linear_cka":cka}

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
    p.add_argument("--sensitivity",action="store_true",help="per-layer int-N sensitivity instead of the bit sweep")
    p.add_argument("--bits",type=int,default=2,help="low bit-width (sensitivity scan / mixed-precision rest)")
    p.add_argument("--keep",help="comma-sep layer names to protect at --keep_bits while the rest go to --bits (mixed precision)")
    p.add_argument("--keep_bits",type=int,default=8,help="bit-width for the protected layers")
    p.add_argument("--emb_cos_vs",help="reference fp checkpoint: report pooled-embedding cosine vs it (did QAT keep the geometry?)")
    p.add_argument("--gptq",action="store_true",help="RTN vs GPTQ (error-compensated PTQ) across bit-widths -> gptq_results.json")
    a=p.parse_args()
    if a.data_file: ModelArgs.data_file=a.data_file
    if a.out_dir: ModelArgs.out_dir=a.out_dir
    ckpts=a.checkpoints or default_ckpts()
    if a.sensitivity:
        allres=[sensitivity(c,a.bits) for c in ckpts]
        out=Path(ModelArgs.out_dir)/"sensitivity_results.json"
        out.write_text(json.dumps(allres,indent=2))
        print(f"saved -> {out}")
        return
    if a.keep is not None:
        keep=[s for s in a.keep.split(",") if s]
        allres=[mixed(c,keep,a.bits,a.keep_bits) for c in ckpts]
        out=Path(ModelArgs.out_dir)/"mixed_results.json"
        out.write_text(json.dumps(allres,indent=2))
        print(f"saved -> {out}")
        return
    if a.emb_cos_vs is not None:
        allres=[emb_cos_vs(c,a.emb_cos_vs) for c in ckpts]
        out=Path(ModelArgs.out_dir)/"emb_cos_results.json"
        out.write_text(json.dumps(allres,indent=2))
        print(f"saved -> {out}")
        return
    if a.gptq:
        allres=[gptq_sweep(c) for c in ckpts]
        out=Path(ModelArgs.out_dir)/"gptq_results.json"
        out.write_text(json.dumps(allres,indent=2))
        print(f"saved -> {out}")
        return
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