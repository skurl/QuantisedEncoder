# QuantisedEncoder

> Small models, narrow domains, low bits. Notes from an ongoing experiment.

A compact encoder that learns a language it was never told the grammar of, then gets
squeezed to see how much still survives. Trained on one corner of the tree of life; the
rest is left as an exercise. Runs as a **Nextflow pipeline** on SLURM + Singularity.

There is a plan. It is not in this file yet.

## Layout

```
main.nf            the pipeline:  PREP -> TRAIN -> gate -> {EVAL, QUANTISE} -> REPORT
nextflow.config    params + SLURM/Singularity resources
Dockerfile         the container
bin/
  downloads.py     fetch the inputs                       (stdlib only)
  core.py          the model, the metrics, the sharp edges
  train.py         training (+ QAT, + optional distillation)
  quantise.py      the squeeze:  PTQ sweep / sensitivity / mixed precision
  proteingym.py    an external yardstick
  report.py        rank the survivors -> champion.json + leaderboard.csv
```

The `fungalplm/` package is the quiet part — a two-line way to load a finished checkpoint
and get embeddings out. It stays torch-only and says nothing about how the weights were made.

## Setup

One-time, per cluster:

1. **Clone**, then **build the container** (`quantised_encoder.sif`):
   ```bash
   docker buildx build --platform linux/amd64 -t ghcr.io/skurl/quantised-encoder:0.0.1 --push .
   singularity pull quantised_encoder.sif docker://ghcr.io/skurl/quantised-encoder:0.0.1
   ```
2. **Nextflow + inputs** (login node, needs network):
   ```bash
   curl -s https://get.nextflow.io | bash
   python bin/downloads.py
   ```
3. `wandb login` once; set your SLURM partitions in `nextflow.config`.

## Running

```bash
./nextflow run main.nf -resume
./nextflow run main.nf -resume --cluster_ids '["0.5"]' --max_steps 200   # smoke test
```

`-resume` makes it incremental. Results land under `results/`; `champion.json` and
`leaderboard.csv` say who won.

Teacher-guided training, when it's wanted, is a config away:

```bash
--distill_teacher facebook/esm2_t12_35M_UR50D    # or any HF encoder id
```

## State of things

The squeeze is well understood: most of the range is free, one layer is a cliff, and training
*for* the low-bit regime buys most of it back. What's in progress is making the thing genuinely
good at its narrow domain rather than merely small — the distillation switch above is the current
lever. The yardstick will tell.

## Author
* Maciej Szczesny
