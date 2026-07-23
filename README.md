# QuantisedEncoder

A small (~9.5M parameter) masked-language-model protein encoder trained only on fungal proteins
(UniProt taxon 4751). I built it to study the effects of quantisation, and which features get preserved, and to be a usable, lightweight encoder for fungal proteins.

It is better than ESM-2 at predicting ubiquitin zero-shot mutations. (See below), and that is probably its main usecase.

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

On aggregate it trails ESM2-8M — **0.01 ± 0.05 vs 0.19 mean Spearman across three seeds — as expected
for the same ~9.5M parameters trained on fungi alone rather than broadly; the wide seed spread reflects
how noisy a 14-assay panel leaves the estimate. But on ubiquitin it beats ESM2 on 2 of 3
assays (Mavor, Roscoe 2013), and only in the seeds that learned the protein: across three seeds the
number of ubiquitin wins rises with how far each drove wild-type NLL down (3/3 at NLL 1.3, 2/3 at 2.2,
none at 2.8). Fitness tracks calibration even within a single protein.

int8 and int4 quantisation is essentially free (top-1 within 0.1
pt, embedding cosine ≥ 0.997); int3 is a soft edge (top-1 −1 pt, BLOSUM −18%); int2 is a cliff — full
int2 drops top-1 to 6.8% (≈ a unigram baseline), with the damage localised to the embedding layer.
Quantisation-aware training reclaims it: weight-only int2 QAT recovers top-1 to 22.5% (fp is 23.9%)
— int2 at roughly int3 quality. The QAT embeddings sit at 0.78 cosine to the fp model but 0.92 linear
CKA, so the drift is a basis rotation, not lost information.

Distillation from ESM2 lifts token metrics (top-1, perplexity) but adds no fitness
and erodes the ubiquitin edge, decided not to pursue it further in this project; the contact probe did rise 2-3x above the random noise levels, but nowhere near close to actually usuable models (below 0.10), so I decided not to use it further in this project.

An analogous run on all [eukaryotic sequences](https://github.com/skurl/EukaryoticEncoder), and demonstrated the same results: this model seems to be really good (2x better than ESM2-8M) at Mavor 2016 and Roscore 2013 benchmarks for ubiquitin mutations.

## Per layer analysi

layer	fungi	eukaryote	untrained
0	2.61	2.58	~2.6
1	2.41	2.54	~2.55
2	2.42	2.40	~2.55
3	2.20	2.42	~2.55
4	1.99	2.26	~2.55
5	1.68	1.92	~2.52
6	1.08	1.46	~2.55


## Author

Maciej Szczesny
