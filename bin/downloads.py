"""Fetch every input the pipeline needs. Run once on the login node (has network):

    python bin/downloads.py              # everything (fungi + proteingym + contacts)
    python bin/downloads.py fungi        # just the training set  -> data/fungi.fasta
    python bin/downloads.py proteingym   # just the eval data     -> data/proteingym/
    python bin/downloads.py eukaryota [N]  # ALL eukaryotic seqs (huge!) -> data/eukaryota.fasta

stdlib only (urllib + gzip + zipfile), so no curl/unzip needed. Clustering is NOT done
here -- the Nextflow PREP stage clusters the FASTA at each params.cluster_ids.
"""
import gzip, json, shutil, ssl, subprocess, sys, urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlencode

DATA = Path("./data")
UNIPROT = "https://rest.uniprot.org/uniprotkb"
STREAM_CAP = 9_000_000                    # stay under UniProt's ~10M stream limit; split queries below this
FUNGI_URL = UNIPROT + "/stream?" + urlencode({   # fungal, len<=512, non-frag, PE 1-3 (~3.66M seqs)
    "compressed": "true", "format": "fasta",
    "query": "(taxonomy_id:4751) AND (length:[* TO 512]) AND fragment:false "
             "AND (existence:1 OR existence:2 OR existence:3)"})
PGYM_BASE = "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3"
PGYM_ZIPS = ["DMS_ProteinGym_substitutions.zip", "zero_shot_substitutions_scores.zip"]
PGYM_REF = "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv"   # per-assay taxon labels (Human/Eukaryote/Prokaryote/Virus) -> the --taxon eval set
AF_API = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"   # returns the CURRENT pdbUrl; AFDB drops old file versions (v4->v6...), so never hardcode the version


def _gz_complete(path):                    # a gzip is only complete if it reads to a valid end-of-stream trailer
    try:
        return subprocess.run(["gzip", "-t", str(path)], capture_output=True).returncode == 0
    except FileNotFoundError:              # no gzip CLI -> verify by fully decompressing in python
        try:
            with gzip.open(path, "rb") as f:
                while f.read(1 << 22):
                    pass
            return True
        except (OSError, EOFError):
            return False


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


def _euk_q(lo, hi):                        # same filter as fungi(), but taxon 2759 (Eukaryota) and an explicit length window
    # note: `existence` takes individual values, NOT a range ([1 TO 3] is rejected by the API)
    return (f"(taxonomy_id:2759) AND (length:[{lo} TO {hi}]) AND fragment:false "
            f"AND (existence:1 OR existence:2 OR existence:3)")


def _count(q):                            # UniProt returns the match count in the x-total-results header
    url = f"{UNIPROT}/search?" + urlencode({"query": q, "format": "list", "size": 0})
    try:
        with urllib.request.urlopen(url) as r:
            return int(r.headers.get("x-total-results", 0))
    except urllib.error.URLError as e:
        if "CERTIFICATE" not in str(e).upper():
            raise
        with urllib.request.urlopen(url, context=ssl._create_unverified_context()) as r:
            return int(r.headers.get("x-total-results", 0))


def _windows(lo, hi):                     # recursively halve the length range until each window is under the stream cap
    n = _count(_euk_q(lo, hi))
    if n <= STREAM_CAP or lo >= hi:
        if n > STREAM_CAP:                # a single length with >cap seqs -> stream truncates (won't happen for len<=512 eukaryota)
            print(f"[warn] length {lo}-{hi}: {n:,} > cap, stream will truncate")
        return [(lo, hi, n)]
    mid = (lo + hi) // 2
    return _windows(lo, mid) + _windows(mid + 1, hi)


def eukaryota(cap_seqs=None):
    """ALL eukaryotic (taxon 2759) sequences, filtered exactly like fungi() (len<=512, non-fragment, PE 1-3),
    downloaded as length-window batches under UniProt's ~10M stream cap and concatenated -> data/eukaryota.fasta.
    ~15.8M sequences / ~15 GB. Pass N to cap the joined output at ~N sequences. Windows are cached and their
    gzip integrity is checked, so a truncated part is re-downloaded automatically -- an interrupted run just
    resumes on re-run."""
    DATA.mkdir(parents=True, exist_ok=True)
    parts = DATA / "eukaryota_parts"; parts.mkdir(exist_ok=True)
    out = DATA / "eukaryota.fasta"
    windows = _windows(1, 512)
    total = sum(n for *_, n in windows)
    print(f"[plan]  {len(windows)} length windows, ~{total:,} sequences total")
    got, part_files = 0, []
    for lo, hi, n in windows:
        gz = parts / f"euk_{lo:04d}_{hi:04d}.fasta.gz"
        if gz.exists() and not _gz_complete(gz):          # a prior run left this part truncated -> don't trust it
            print(f"[bad]   {gz.name} is truncated -> re-downloading"); gz.unlink()
        fetch(f"{UNIPROT}/stream?" + urlencode({"compressed": "true", "format": "fasta", "query": _euk_q(lo, hi)}), gz)
        if not _gz_complete(gz):                          # the UniProt stream dropped mid-download
            gz.unlink()
            sys.exit(f"[error] {gz.name} came down truncated (stream dropped). Just re-run to retry this window.")
        part_files.append(gz); got += n
        if cap_seqs and got >= cap_seqs:
            print(f"[cap]   ~{got:,} sequences downloaded, stopping (cap {cap_seqs:,})"); break
    print(f"[join]  {len(part_files)} parts -> {out}")
    written = 0
    with open(out, "w") as g:
        for gz in part_files:
            with gzip.open(gz, "rt") as f:
                if cap_seqs is None:
                    shutil.copyfileobj(f, g); continue
                for line in f:
                    if line.startswith(">"):
                        if written >= cap_seqs:
                            break
                        written += 1
                    g.write(line)
            if cap_seqs and written >= cap_seqs:
                break
    n = sum(line.startswith(">") for line in open(out))
    print(f"eukaryota -> {out}  ({n:,} sequences; clustering happens in the PREP pipeline stage)\n")


def proteingym():
    out = DATA / "proteingym"; out.mkdir(parents=True, exist_ok=True)
    for z in PGYM_ZIPS:
        zp = out / z
        fetch(f"{PGYM_BASE}/{z}", zp)
        sub = out / zp.stem
        print(f"[unzip] {z} -> {sub.name}/")
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(sub)
    # taxon labels, dropped INTO the scored dir so the Nextflow REPORT stage stages it alongside the assays
    fetch(PGYM_REF, out / "zero_shot_substitutions_scores" / "DMS_substitutions.csv")
    print(f"proteingym -> {out}/  (DMS assays + ESM2-8M baseline scores + taxon reference)\n")


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
    args = sys.argv[1:]
    which = args[0] if args else "all"
    if which not in ("fungi", "proteingym", "contacts", "eukaryota", "all"):
        sys.exit("usage: python bin/downloads.py [fungi|proteingym|contacts|eukaryota|all]")
    if which in ("fungi", "all"):
        fungi()
    if which in ("proteingym", "all"):
        proteingym()
    if which in ("contacts", "all"):
        contacts()
    if which == "eukaryota":              # opt-in only -- NOT part of `all` (it's ~100M+ sequences)
        eukaryota(cap_seqs=int(args[1]) if len(args) > 1 else None)


if __name__ == "__main__":
    main()
