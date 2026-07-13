# QuantisedEncoder

A small **protein language model trained exclusively on the fungal kingdom** (UniProt taxon 4751),
built to (a) study how low-bit **quantisation** degrades a protein PLM, and (b) become a useful
**drop-in encoder for fungal experiments**. Runs as a reproducible **Nextflow pipeline** on SLURM +
Singularity, logging to Weights & Biases.

The model is an ESM/BERT-style masked-LM encoder: RoPE, pre-norm, SDPA attention, ~9.5M params,
predicting the 20 standard amino acids at masked positions.

## Results so far

Base model (fungal held-out test, ~840k sequences clustered at 50% id):

| metric | value | baseline |
|---|---|---|
| top-1 accuracy | **23.9%** | unigram 9.2% |
| perplexity | **12.4** | unigram 18.0 |
| BLOSUM62 substitution ρ | **0.35** | — |

**Quantisation (weight-only RTN, per-channel, 3 seeds):**
- **int8 / int4 — free** (top-1 within 0.1pt, embeddings ≥99.7% aligned).
- **int3 — usable** (~1pt top-1 drop).
- **int2 — a cliff.** True all-int2 (embedding included) collapses to **6.8% top-1** — below unigram.
- Per-layer sensitivity: **the embedding table is the cliff** (int2 on it alone → 9% = unigram); the transformer blocks are diffuse-but-recoverable.
- **QAT reclaims int2:** fine-tuning with fake-quant in the loop recovers to **~22.5% top-1** (94% of fp) and preserves the representation up to a rotation (**linear CKA 0.92**). int2 goes from catastrophic to usable *if you train for it*.

**ProteinGym zero-shot (fungal subset, vs ESM2-8M):**
- Competitive with ESM2-8M on **well-conserved fungal proteins** — *beats* it on ubiquitin (`RL40A`), neck-and-neck on `GCN4`.
- Weak overall — the model is **data-limited** (fungal-only, 50%-clustered = homolog depth stripped, undertrained vs ESM2's UniRef50). `qat_int2` tracks `fp` here too.

## Repository layout

```
main.nf            Nextflow DAG:  PREP -> TRAIN -> gate -> {EVAL_PGYM, QUANTISE} -> REPORT
nextflow.config    params (cluster_ids, gate, pgym paths) + SLURM/Singularity resources
Dockerfile         builds the .sif environment (torch + mmseqs + wandb ...)
environment.yml    conda deps baked into the container
sweep.yaml         standalone W&B hyperparameter sweep (not part of the DAG)
bin/
  downloads.py     fetch inputs: data/fungi.fasta + data/proteingym/   (stdlib only)
  core.py          library: ModelArgs, Vocab, the Transformer, metrics, QAT helpers
  train.py         TRAIN stage (+ QAT fine-tune via --init_ckpt/--qat_bits)
  quantise.py      QUANTISE stage: PTQ sweep (+ --sensitivity, --keep, --emb_cos_vs)
  proteingym.py    EVAL_PGYM stage: masked-marginals zero-shot vs ESM2-8M
  report.py        REPORT stage: rank checkpoints -> champion.json + leaderboard.csv
```

## Setup

The container installs all runtime deps and one command runs the experiments, but bootstrapping a
cluster is a one-time checklist:

1. **Clone** the repo to the cluster.
2. **Build the container** (needs Docker; the pipeline assumes `quantised_encoder.sif` exists):
   ```bash
   docker buildx build --platform linux/amd64 -t ghcr.io/skurl/quantised-encoder:0.0.1 --push .
   singularity pull quantised_encoder.sif docker://ghcr.io/skurl/quantised-encoder:0.0.1
   ```
3. **Fetch Nextflow + data** (on the login node — needs network):
   ```bash
   curl -s https://get.nextflow.io | bash     # ./nextflow launcher (needs Java 11+)
   python bin/downloads.py                     # data/fungi.fasta + data/proteingym/  (~3.4 GB)
   ```
4. **W&B login** (once): `wandb login` — the pipeline logs live via `~/.netrc`.
5. **Set your SLURM partitions** in `nextflow.config`: `gpu-compute` for GPU stages, and change the
   `withName: PREP`/`REPORT` placeholder queue `compute` to your real CPU partition.

## Running

```bash
./nextflow run main.nf -resume                                   # the full experiment
./nextflow run main.nf -resume --cluster_ids '["0.5"]' --max_steps 200   # fast smoke test
```

Every compute step runs inside the `.sif`; `-resume` makes it incremental (safe to re-run / add
configs). Results land under `results/` and stream to W&B. `results/champion.json` +
`leaderboard.csv` rank the runs by in-domain ProteinGym signal.

Code changes just need `git pull` + re-run — rebuild the `.sif` **only** when `environment.yml` /
`Dockerfile` change.

## The experiment the pipeline is built around

`PREP` fans out over `params.cluster_ids = ['0.5', '0.9', 'none']`, producing one training set per
clustering threshold. The open question: **does keeping homolog depth (looser clustering) lift
ProteinGym?** The 50% clustering that helped MLM training likely stripped the conservation signal
zero-shot variant-effect needs — `leaderboard.csv` will show whether `0.9`/`none` beat `0.5`.

## Next steps

- [ ] **Broad training + fungal fine-tuning** — the biggest lever for a *good fungal drop-in*: pretrain
      broad (general protein biology) then specialize on fungal. Cheaper variant: **distill from ESM2**
      into the small model. This is what closes the ProteinGym gap.
- [ ] Finish the **cluster_ids experiment** (running) — confirm the homolog-depth hypothesis.
- [ ] **QAT-on-champion** stage — auto fine-tune the winning checkpoint to int2 for the deployment claim.
- [ ] Multi-seed error bars; per-layer sensitivity / mixed-precision on the champion.
- [ ] Parked: SAE feature interpretability (which features survive int-quant); native ternary vs PTQ-int2.

## Author
* Maciej Szczesny

![Length Distribution](https://github.com/skurl/QuantisedEncoder/blob/main/diagrams/length_distribution.png?raw=true)
![Mushroom](https://github.com/skurl/QuantisedEncoder/blob/main/diagrams/mushroom.png?raw=true)
