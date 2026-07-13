#!/bin/bash
# One-time login-node bootstrap: Nextflow launcher + input data + prerequisite checks.
# Does NOT build the .sif container (that needs Docker) -- see README > Setup.
# Run once, on a node with network:   bash bootstrap.sh
set -e
cd "$(dirname "$0")"

# --- Nextflow launcher (needs Java 11+) ---
if [ -x ./nextflow ]; then
    echo "[nextflow] already present"
elif curl -fsSL https://get.nextflow.io | bash; then
    echo "[nextflow] installed -> ./nextflow"
else
    echo "[nextflow] auto-install failed (broken CA bundle?) -- install manually: https://www.nextflow.io/docs/latest/install.html"
fi

# --- Input data: fungi.fasta + ProteinGym (~3.4 GB; downloads.py skips what's already there) ---
python bin/downloads.py

# --- Checks bootstrap can't fix for you ---
echo; echo "[checks]"
chk() { command -v "$1" >/dev/null && echo "  $1: OK" || echo "  $1: MISSING ($2)"; }
chk java        "Nextflow needs Java 11+"
chk singularity "needed to run the .sif"
[ -f quantised_encoder.sif ] && echo "  .sif: OK" || echo "  .sif: MISSING -> build from Dockerfile (see README > Setup)"
grep -qs wandb "$HOME/.netrc" && echo "  wandb: logged in" || echo "  wandb: run 'wandb login' once"

echo
echo "Next: set your SLURM partitions in nextflow.config, then:  ./nextflow run main.nf -resume"
