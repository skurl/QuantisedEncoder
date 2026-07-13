"""Fetch every input the pipeline needs. Run once on the login node (has network):

    python bin/downloads.py              # everything
    python bin/downloads.py fungi        # just the training set  -> data/fungi.fasta
    python bin/downloads.py proteingym   # just the eval data     -> data/proteingym/

stdlib only (urllib + gzip + zipfile), so no curl/unzip needed. Clustering is NOT done
here -- the Nextflow PREP stage clusters `data/fungi.fasta` at each params.cluster_ids.
"""
import gzip, shutil, ssl, sys, urllib.request
import zipfile
from pathlib import Path

DATA = Path("./data")
FUNGI_URL = ("https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=fasta"
             "&query=%28%28taxonomy_id%3A4751%29+AND+%28length%3A%5B*+TO+512%5D%29"
             "+AND+fragment%3Afalse+AND+existence%3A%5B1+TO+3%5D%29")   # fungal, len<=512, non-fragment, PE 1-3 (~3.66M seqs)
PGYM_BASE = "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3"
PGYM_ZIPS = ["DMS_ProteinGym_substitutions.zip", "zero_shot_substitutions_scores.zip"]


def fetch(url, dest):                      # normal TLS; fall back to unverified only if the CA bundle is broken
    if dest.exists():
        print(f"[have]  {dest.name}"); return
    print(f"[get]   {url}")
    try:
        urllib.request.urlretrieve(url, dest)
    except urllib.error.URLError as e:
        if "CERTIFICATE" not in str(e).upper():
            raise
        print("        TLS verify failed (broken CA bundle) -> retrying unverified")   # public data, known host
        with urllib.request.urlopen(url, context=ssl._create_unverified_context()) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)


def fungi():
    DATA.mkdir(parents=True, exist_ok=True)
    gz, raw = DATA / "fungi.fasta.gz", DATA / "fungi.fasta"
    fetch(FUNGI_URL, gz)
    if not raw.exists():
        print("[gunzip] -> data/fungi.fasta")
        with gzip.open(gz, "rt") as f, open(raw, "w") as g:
            shutil.copyfileobj(f, g)
    n = sum(line.startswith(">") for line in open(raw))
    print(f"fungi -> {raw}  ({n} proteins; clustering happens in the PREP pipeline stage)\n")


def proteingym():
    out = DATA / "proteingym"; out.mkdir(parents=True, exist_ok=True)
    for z in PGYM_ZIPS:
        zp = out / z
        fetch(f"{PGYM_BASE}/{z}", zp)
        sub = out / zp.stem                # own subfolder, not loose in proteingym/
        print(f"[unzip] {z} -> {sub.name}/")
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(sub)
    print(f"proteingym -> {out}/  (DMS assays + ESM2-8M baseline scores)\n")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which not in ("fungi", "proteingym", "all"):
        sys.exit("usage: python bin/downloads.py [fungi|proteingym|all]")
    if which in ("fungi", "all"):
        fungi()
    if which in ("proteingym", "all"):
        proteingym()


if __name__ == "__main__":
    main()
