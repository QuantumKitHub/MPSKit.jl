#!/bin/bash
# Suite 7 (thread scaling), ITensorMPS side. Same structure and grid as
# threads_mpskit.sbatch. Submit from the repo root:
#   sbatch benchmark/slurm/threads_itensor.sbatch
#SBATCH --job-name=itensor-threads
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --partition=ccq            # EDIT if needed
#SBATCH --constraint=icelake       # EDIT: pin ONE CPU generation, same for ALL jobs
#SBATCH --output=benchmark/slurm/logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p benchmark/slurm/logs

# Keep the grid identical to threads_mpskit.sbatch.
GRID=("1 1" "4 1" "1 4" "4 4" "8 1" "1 8" "8 8")

for point in "${GRID[@]}"; do
    read -r nj nb <<< "${point}"
    echo "=== suite 7 (ITensorMPS): julia=${nj} blas=${nb} ==="
    JULIA_NUM_THREADS="${nj}" julia --project=benchmark/comparisons/itensor benchmark/comparisons/itensor/run.jl --suite=7 --blas-threads="${nb}"
done
