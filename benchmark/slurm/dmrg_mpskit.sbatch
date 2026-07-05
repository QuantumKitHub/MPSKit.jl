#!/bin/bash
# Suites 1-2 (finite DMRG, trivial + U(1)), MPSKit side. Submit from the repo root:
#   sbatch benchmark/slurm/dmrg_mpskit.sbatch
# Instantiate the env on the login node first (compute nodes may lack registry access):
#   julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
#SBATCH --job-name=mpskit-dmrg
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --partition=ccq            # EDIT if needed
#SBATCH --constraint=icelake       # EDIT: pin ONE CPU generation, same for ALL jobs
#SBATCH --output=benchmark/slurm/logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p benchmark/slurm/logs

# Single-threaded headline comparison (methodology guardrail §4.3): identical thread
# settings on both sides. Suite 7 is the multithreaded story.
export JULIA_NUM_THREADS=1

julia --project=benchmark benchmark/run.jl --suite=1,2 --blas-threads=1
