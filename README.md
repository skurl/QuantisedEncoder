# QuantisedEncoder

A small (~9.5M parameter) masked-language-model protein encoder trained only on fungal proteins
(UniProt taxon 4751). I built it to study the effects of quantisation, and which features get preserved, and to be a usable, lightweight encoder for fungal proteins.

The model is ESM/BERT-style: RoPE, pre-norm, SDPA attention, 6 layers, d_model 512, predicting the 20
standard amino acids at masked positions.

The trained weights are on the Hugging Face Hub (`szchesny/fungal-plm`), and the inference wrapper is
on PyPI — `pip install fungalplm`. The model was trained and evaluated with Nextflow, using Weights & Biases for hyperparameter optimisation.

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

Install the package using `pip install fungalplm`.

## Results

The model was evaluated on three axes: BLOSUM62 substitution correlation, ProteinGym zero-shot
variant-effect scoring (14 yeast assays vs an ESM2-8M baseline), and an unsupervised attention contact
probe.

BLOSUM proved to not directly translate to fitness. The model reaches 0.36 BLOSUM Spearman against a
0.13 frequency-null, yet a higher BLOSUM score
doesn't track fitness. The much better benchmark is ProteinGym, and fitness there is gated by wild-type NLL: the
model predicts a protein's mutational effects well only where its own likelihood sits near the 1.2
NLL band.

On aggregate it trails ESM2-8M — 0.08 vs 0.19 mean Spearman — as expected for the same ~9.5M
parameters trained on fungi alone rather than broadly. But on ubiquitin, the one protein it
calibrates well (WT-NLL ~1.4), it beats ESM2 on most assays, and the win holds across seeds.

int8 and int4 quantisation is essentially free (top-1 within 0.1
pt, embedding cosine ≥ 0.997); int3 is a soft edge (top-1 −1 pt, BLOSUM −18%); int2 is a cliff — full
int2 drops top-1 to 6.8% (≈ a unigram baseline), with the damage localised to the embedding layer.
Quantisation-aware training reclaims it: weight-only int2 QAT recovers top-1 to 22.5% (fp is 23.9%)
— int2 at roughly int3 quality. The QAT embeddings sit at 0.78 cosine to the fp model but 0.92 linear
CKA, so the drift is a basis rotation, not lost information.

Distillation from ESM2 lifts token metrics (top-1, perplexity) but adds no fitness
and erodes the ubiquitin edge, decided not to pursue it further in this project; the contact probe did rise 2-3x above the random noise levels, but nowhere near close to actually usuable models (below 0.10), so I decided not to use it further in this project.

An analogous run on all [eukaryotic sequences](https://github.com/skurl/EukaryoticEncoder) is in progress and shows the same pattern so far.

## Author

Maciej Szczesny
