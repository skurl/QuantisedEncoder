"""Aggregate per-checkpoint result rows -> champion.json + leaderboard.csv.

Each row (written by the EVAL_PGYM Nextflow stage) is a small json:
    {id, cluster_id, tag, seed, mlm_top1, pgym_fp, pgym_esm2}
Champion = best in-domain ProteinGym signal (pgym_fp), tiebreak MLM top1.
Usage:  python bin/report.py row_a.json row_b.json ...
"""
import csv, json, sys
from pathlib import Path


def main(paths):
    rows = [json.loads(Path(p).read_text()) for p in paths]
    if not rows:
        sys.exit("no result rows")
    key = lambda r: (r.get("pgym_fp") if r.get("pgym_fp") is not None else -1,
                     r.get("mlm_top1") if r.get("mlm_top1") is not None else -1)
    rows.sort(key=key, reverse=True)

    Path("champion.json").write_text(json.dumps(rows[0], indent=2))
    cols = sorted({k for r in rows for k in r})
    with open("leaderboard.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)

    c = rows[0]
    print(f"champion: {c.get('id')}  pgym_fp={c.get('pgym_fp')}  mlm_top1={c.get('mlm_top1')}  "
          f"(vs esm2 {c.get('pgym_esm2')})")
    print(f"leaderboard: {len(rows)} checkpoints -> leaderboard.csv")


if __name__ == "__main__":
    main(sys.argv[1:])
