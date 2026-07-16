"""Fetch every input the pipeline needs. Run once on the login node (has network):

    python bin/downloads.py              # everything
    python bin/downloads.py fungi        # just the training set  -> data/fungi.fasta
    python bin/downloads.py proteingym   # just the eval data     -> data/proteingym/

stdlib only (urllib + gzip + zipfile), so no curl/unzip needed. Clustering is NOT done
here -- the Nextflow PREP stage clusters `data/fungi.fasta` at each params.cluster_ids.
"""
import gzip, json, shutil, ssl, sys, urllib.request
import zipfile
from pathlib import Path

DATA = Path("./data")
FUNGI_URL = ("https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=fasta"
             "&query=%28%28taxonomy_id%3A4751%29+AND+%28length%3A%5B*+TO+512%5D%29"
             "+AND+fragment%3Afalse+AND+existence%3A%5B1+TO+3%5D%29")   # fungal, len<=512, non-fragment, PE 1-3 (~3.66M seqs)
PGYM_BASE = "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3"
PGYM_ZIPS = ["DMS_ProteinGym_substitutions.zip", "zero_shot_substitutions_scores.zip"]
AF_API = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"   # returns the CURRENT pdbUrl; AFDB drops old file versions (v4->v6...), so never hardcode the version


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
        sub = out / zp.stem
        print(f"[unzip] {z} -> {sub.name}/")
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(sub)
    print(f"proteingym -> {out}/  (DMS assays + ESM2-8M baseline scores)\n")


def _accessions(fasta, n):
    """First n UniProt accessions from fungi.fasta headers (>sp|ACC|... or >tr|ACC|...)."""
    accs = []
    for line in open(fasta):
        if line.startswith(">"):
            parts = line[1:].split("|")
            accs.append(parts[1] if len(parts) > 2 else line[1:].split()[0])
            if len(accs) >= n:
                break
    return accs


def _af_pdb_url(acc):                      # ask AFDB for the current file URL; [] / 404 = no model for this accession
    url = AF_API.format(acc=acc)
    try:
        with urllib.request.urlopen(url) as r:
            meta = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    except urllib.error.URLError as e:
        if "CERTIFICATE" not in str(e).upper():
            raise
        with urllib.request.urlopen(url, context=ssl._create_unverified_context()) as r:   # broken CA bundle, known host
            meta = json.load(r)
    return meta[0]["pdbUrl"] if meta else None


def contacts(n=300):
    """AlphaFold structures for the first n fungal proteins -> data/contacts/*.pdb (contact-probe labels)."""
    fasta = DATA / "fungi.fasta"
    if not fasta.exists():
        sys.exit("need data/fungi.fasta first -> python bin/downloads.py fungi")
    out = DATA / "contacts"; out.mkdir(parents=True, exist_ok=True)
    got = 0
    for acc in _accessions(fasta, n):
        dest = out / f"{acc}.pdb"
        if dest.exists():
            got += 1; continue
        url = _af_pdb_url(acc)
        if url is None:
            print(f"[skip]  {acc} (not in AFDB)")     # not every UniProt accession has a model
            continue
        fetch(url, dest); got += 1
    print(f"contacts -> {out}/  ({got}/{n} structures)\n")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which not in ("fungi", "proteingym", "contacts", "all"):
        sys.exit("usage: python bin/downloads.py [fungi|proteingym|contacts|all]")
    if which in ("fungi", "all"):
        fungi()
    if which in ("proteingym", "all"):
        proteingym()
    if which in ("contacts", "all"):
        contacts()


if __name__ == "__main__":
    main()
