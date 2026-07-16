# fungalplm

A tiny (9.5M-parameter) protein language model exclusively trained on fungal proteins. The
API allows for per-protein and per-residue embeddings extraction. Weights load straight from the Hugging Face Hub.

```bash
pip install fungalplm
```

```python
from fungalplm import FungalPLM

plm = FungalPLM.load("szchesny/fungal-plm")            # downloads the weights
emb = plm.embed(["MQIFVKTLTGKTITLEVEPSDTIENVK..."])    # [N, d] per-protein (mean-pooled)
res = plm.embed(seqs, per_residue=True)                # list of [Li, d]
```

Or from the command line:

```bash
fungalplm embed proteins.fasta --ckpt fungal-plm.pth -o embeddings.npz
```

## Usecase

This model beats ESM2-8M in ubiquitin prediction, with wild-type-NLL signal scores provided, that tells you when to trust it (1.2-1.4 most optimal). Full numbers, limitations, and the model card: **https://huggingface.co/szchesny/fungal-plm**

MIT licensed. Dependencies: torch, numpy, huggingface_hub.
