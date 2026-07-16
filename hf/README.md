---
license: mit
library_name: fungalplm
pipeline_tag: fill-mask
tags:
  - protein-language-model
  - biology
  - fungi
  - masked-language-model
---

# FungalPLM v0.0.1

A small (**9.5M parameter**) masked-language-model protein encoder trained **exclusively on fungal
proteins** (UniProt taxon 4751). RoPE, pre-norm, SDPA attention; predicts the 20 standard amino acids
at masked positions. Loads in two lines via the [`fungalplm`](https://github.com/skurl/QuantisedEncoder)
package for per-protein or per-residue embeddings. It beats ESM2-8M on ubiquitin related tasks.

## Performance (measured, honest)

| metric | FungalPLM | reference |
|---|---|---|
| MLM test top-1 | 23.9% | unigram 9.2% |
| BLOSUM62 substitution ρ | **0.35** | frequency-null **0.13** (learned real biochemistry) |
| ProteinGym fungal (14 assays, mean Spearman) | **0.05 ± 0.02** (3 seeds) | ESM2-8M **0.19** |

On aggregate fitness prediction, **ESM2-8M (same size) beats this model**

On **ubiquitin (RL40A)** it beats ESM2-8M.

| ubiquitin assay | FungalPLM (s42) | ESM2-8M |
|---|---|---|
| RL40A_Mavor_2016 | **0.22** | 0.10 |
| RL40A_Roscoe_2013 | **0.23** | 0.12 |

### The confidence signal: wild-type NLL

Fitness quality tracks how well the model calibrates a given protein (per Hou et al. 2026's bell curve).
Trust this model's fitness calls where its wild-type per-residue NLL is low (1.2–1.6); defer to a
general model (ESM2) where it is high (2.5+). WT-NLL is a *predictive* gate, computed before you
trust a score.

## Usage

```bash
pip install fungalplm
```
```python
from fungalplm import FungalPLM
plm = FungalPLM.load("szchesny/fungal-plm")          # this repo
emb = plm.embed(["MQIFVKTLTGKTITLEVEPSDTIENVK..."]) # [N, d] per-protein
res = plm.embed(seqs, per_residue=True)            # list of [Li, d]
```

## Training

UniProt fungal (taxon 4751), length ≤ 512, non-fragment, PE 1–3 → ~3.66M sequences, MMseqs2-linclust
at 50% identity → **840k** unique training sequences. 10k steps, masked-LM (15% masking). See the
[pipeline](https://github.com/skurl/QuantisedEncoder) for the full recipe and the quantisation study.

## Limitations

- Fungal-only; do not expect sensible behaviour on non-fungal proteins.
- Not competitive with ESM2 on general fitness prediction (see table).
- The ubiquitin win does not (yet) demonstrably generalise to other conserved proteins.
- Sequences truncated to 2046 residues at inference.

## Citation

Szczesny, M. *QuantisedEncoder / FungalPLM* (2026). https://github.com/skurl/QuantisedEncoder
