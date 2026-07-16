# QuantisedEncoder

A small (~9.5M parameter) masked-language-model protein encoder trained only on fungal proteins
(UniProt taxon 4751). It was built for two reasons: to study how low-bit **quantisation** degrades a
protein language model, and to be a usable, lightweight encoder for fungal proteins. It runs as a
**Nextflow pipeline** on SLURM + Singularity and logs to Weights & Biases.

The model is ESM/BERT-style: RoPE, pre-norm, SDPA attention, 6 layers, d_model 512, predicting the 20
standard amino acids at masked positions.

The trained weights are on the Hugging Face Hub (`szchesny/fungal-plm`), and the inference wrapper is
on PyPI — `pip install fungalplm`.

## Repository layout

```
main.nf            Nextflow pipeline:  PREP -> TRAIN -> gate -> {EVAL_PGYM, QUANTISE} -> REPORT
nextflow.config    parameters + SLURM/Singularity resources
Dockerfile         builds the container image
bin/
  downloads.py     fetch inputs: training FASTA, ProteinGym, AlphaFold structures (stdlib only)
  core.py          the model, data pipeline, metrics, and quantisation helpers
  train.py         training (+ QAT, + optional distillation)
  quantise.py      post-training quantisation: PTQ sweep, per-layer sensitivity, mixed precision
  proteingym.py    zero-shot variant-effect scoring against an ESM2 baseline
  contacts.py      unsupervised contact probe (Rao et al.): does attention encode 3D structure?
  report.py        rank checkpoints -> champion.json + leaderboard.csv
```

The `fungalplm/` package is the inference side: a two-line API to load a checkpoint (a local file, or
`szchesny/fungal-plm` from the Hub) and get per-protein or per-residue embeddings. Install it with
`pip install fungalplm`.

## Results

The quantisation picture is clear: most of the bit-width range is essentially free, one layer (the
embedding table) is a cliff, and training for the low-bit regime (QAT) recovers most of what plain
post-training quantisation loses at int2. That could be because this model is still very much overparameterised.

As for the models efficiency and accuracy, this model is worse than ESM2-8M, which is  the same size, but broadly trained, on broader tasks, with about
0.05 vs 0.19 mean Spearman across 14 fungal ProteinGym assays. But on proteins where its likelihood is
well-calibrated — ubiquitin, which sits near the fitness-optimal NLL band — it **beats** ESM2, and
that win holds across random seeds.

Distillation from ESM2 was tested and turned off. It improves the token-level metrics (top-1, perplexity)
but not fitness, and it actually *erodes* the ubiquitin advantage — matching a teacher that is only
mediocre on ubiquitin drags the student down to that level. Breadth, not representation-matching, is
the real lever for fitness, and that means a larger model than this one.

## Future plans

- **Quantisation** — turn the low-bit study into a deployable artifact: a QAT-int2 checkpoint that runs
  on CPU with no GPU, and a comparison of native ternary training against post-training int2.
- **Sparse autoencoders (SAEs)** — train SAEs on the model's activations to interpret the features it
  learns, and ask which of those features survive quantisation.
- **Contact maps** — run the attention-contact probe end to end (P@L / P@L-LR) and look at how
  structural information is distributed across layers.
- **Others** — broad pre-training / fine-tuning to close the fitness gap; exposing the wild-type-NLL
  confidence gate directly in the `fungalplm` API; per-position saturation-mutagenesis effect maps;
  and checking whether the ubiquitin result generalises to other conserved proteins.

## Author

Maciej Szczesny
