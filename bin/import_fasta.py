import gzip, shutil, subprocess, sys, urllib.request
from pathlib import Path

URL = ("https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=fasta"
       "&query=%28%28taxonomy_id%3A4751%29+AND+%28reviewed%3Atrue%29%29") # imports all the fungal +. reviewed proteomes
DATA = Path("./data")
RAW_GZ = DATA / "fungi_reviewed.fasta.gz"
RAW = DATA / "fungi_reviewed.fasta"
OUT = DATA / "fungi_clustered.fasta"
MIN_ID = 0.5    # 50% identity
COV = 0.8       # min alignment coverage


def count_seqs(path):
    with open(path) as fh:
        return sum(line.startswith(">") for line in fh)


def main():
    if shutil.which("mmseqs") is None:
        sys.exit("mmseqs not found -> install it:  brew install mmseqs2   (or conda install -c bioconda mmseqs2)")
    DATA.mkdir(parents=True, exist_ok=True)

    if not RAW_GZ.exists():
        print(f"[1/3] downloading {URL}")
        urllib.request.urlretrieve(URL, RAW_GZ)
    if not RAW.exists():
        with gzip.open(RAW_GZ, "rt") as f, open(RAW, "w") as g:
            shutil.copyfileobj(f, g)
    print(f"[1/3] {count_seqs(RAW)} reviewed fungal proteins")

    print(f"[2/3] clustering at {MIN_ID:.0%} identity (MMseqs2, coverage {COV:.0%})")
    prefix, tmp = DATA / "clust", DATA / "mmseqs_tmp"
    subprocess.run(["mmseqs", "easy-cluster", str(RAW), str(prefix), str(tmp),
                    "--min-seq-id", str(MIN_ID), "-c", str(COV)], check=True)

    Path(f"{prefix}_rep_seq.fasta").replace(OUT)   # easy-cluster writes one rep per cluster here
    print(f"[3/3] wrote {count_seqs(OUT)} cluster representatives -> {OUT}")

    shutil.rmtree(tmp, ignore_errors=True)          # drop mmseqs intermediates, keep only OUT
    for p in DATA.glob("clust_*"):
        p.unlink()


if __name__ == "__main__":
    main()
