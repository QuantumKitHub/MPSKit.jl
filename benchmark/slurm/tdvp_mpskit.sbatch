#!/bin/bash
# Suite 5 (TDVP throughput), MPSKit side. Submit from the repo root:
#   sbatch benchmark/slurm/tdvp_mpskit.sbatch
#SBATCH --job-name=mpskit-tdvp
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --partition=ccq            # EDIT if needed
#SBATCH --constraint=icelake       # EDIT: pin ONE CPU generation, same for ALL jobs
#SBATCH --output=benchmark/slurm/logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p benchmark/slurm/logs

export JULIA_NUM_THREADS=1

julia --project=benchmark benchmark/run.jl --suite=5 --blas-threads=1
