"""Aggregate per-checkpoint result rows -> champion.json + leaderboard.csv.

Each row (written by the EVAL_PGYM Nextflow stage) is a small json:
    {id, cluster_id, tag, seed, mlm_top1, blosum, blosum_null, pgym_fp, pgym_esm2, wt_nll}
Champion = the (cluster_id, tag) config with the best MEAN pgym_fp across its seeds (robust to
seed luck; Hou et al. warn seed noise is real), represented by its best single-seed checkpoint.
Usage:  python bin/report.py row_a.json row_b.json ...
"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def main(paths):
    rows = [json.loads(Path(p).read_text()) for p in paths]
    if not rows:
        sys.exit("no result rows")
    pg = lambda r: r.get("pgym_fp") if r.get("pgym_fp") is not None else -1

    groups = defaultdict(list)                                        # seeds of the same (dataset, config)
    for r in rows:
        groups[(r.get("dataset"), r.get("cluster_id"), r.get("tag"))].append(r)
    stats = {g: (mean([pg(r) for r in rs]), pstdev([pg(r) for r in rs]), len(rs)) for g, rs in groups.items()}
    best = max(stats, key=lambda g: stats[g][0])                      # config with best MEAN pgym_fp

    champ = max(groups[best], key=pg)                                 # ship the best seed of the robust config
    champ = {**champ, "pgym_fp_mean": stats[best][0], "pgym_fp_sd": stats[best][1], "n_seeds": stats[best][2]}
    Path("champion.json").write_text(json.dumps(champ, indent=2))

    rows.sort(key=pg, reverse=True)
    cols = sorted({k for r in rows for k in r})
    with open("leaderboard.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)

    for (ds, cid, tag), (m, sd, n) in sorted(stats.items(), key=lambda kv: kv[1][0], reverse=True):
        print(f"  {ds}/{cid}/{tag}: pgym_fp {m:.4f} ± {sd:.4f}  (n={n})")
    print(f"champion: {champ.get('id')}  pgym_fp={champ.get('pgym_fp')} (config mean {champ['pgym_fp_mean']:.4f})  "
          f"blosum={champ.get('blosum')} vs null {champ.get('blosum_null')}  wt_nll={champ.get('wt_nll')}")
    print(f"leaderboard: {len(rows)} checkpoints -> leaderboard.csv")


if __name__ == "__main__":
    main(sys.argv[1:])
