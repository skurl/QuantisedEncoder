import gzip, shutil, subprocess, sys, urllib.request
from pathlib import Path

URL = ("https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=fasta"
       "&query=%28%28taxonomy_id%3A4751%29+AND+%28length%3A%5B*+TO+512%5D%29"
       "+AND+fragment%3Afalse+AND+existence%3A%5B1+TO+3%5D%29")  # fungal, len<=512, non-fragment, predicted and uncertain proteins removed, ~3.66M seqs)
DATA = Path("./data")
RAW_GZ = DATA / "fungi.fasta.gz"
RAW = DATA / "fungi.fasta"
OUT = DATA / "fungi_clustered.fasta"   # matches runner.py ModelArgs.data_file
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
    print(f"[1/3] {count_seqs(RAW)} fungal proteins (<=512, non-fragment, PE 1-3)")

    print(f"[2/3] clustering at {MIN_ID:.0%} identity (MMseqs2 linclust, coverage {COV:.0%})")
    prefix, tmp = DATA / "clust", DATA / "mmseqs_tmp"
    subprocess.run(["mmseqs", "easy-linclust", str(RAW), str(prefix), str(tmp),   # linclust: linear-time, scales to millions
                    "--min-seq-id", str(MIN_ID), "-c", str(COV)], check=True)

    Path(f"{prefix}_rep_seq.fasta").replace(OUT)   # easy-linclust writes one rep per cluster here
    print(f"[3/3] wrote {count_seqs(OUT)} cluster representatives -> {OUT}")

    shutil.rmtree(tmp, ignore_errors=True)          # drop mmseqs intermediates, keep only OUT
    for p in DATA.glob("clust_*"):
        p.unlink()


if __name__ == "__main__":
    main()
