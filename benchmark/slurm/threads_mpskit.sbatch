#!/bin/bash
# Suite 7 (thread scaling), MPSKit side. One exclusive node; one julia process per
# (JULIA_NUM_THREADS x BLAS-threads) grid point, run sequentially so every point sees an
# otherwise-idle node. Submit from the repo root:
#   sbatch benchmark/slurm/threads_mpskit.sbatch
#SBATCH --job-name=mpskit-threads
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --partition=ccq            # EDIT if needed
#SBATCH --constraint=icelake       # EDIT: pin ONE CPU generation, same for ALL jobs
#SBATCH --output=benchmark/slurm/logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p benchmark/slurm/logs

# (julia threads, blas threads) grid — JULIA_NUM_THREADS is launch-time-only, hence the
# one-process-per-point loop. Keep the grid identical to threads_itensor.sbatch.
GRID=("1 1" "4 1" "1 4" "4 4" "8 1" "1 8" "8 8")

for point in "${GRID[@]}"; do
    read -r nj nb <<< "${point}"
    echo "=== suite 7 (MPSKit): julia=${nj} blas=${nb} ==="
    JULIA_NUM_THREADS="${nj}" julia --project=benchmark benchmark/run.jl --suite=7 --blas-threads="${nb}"
done
