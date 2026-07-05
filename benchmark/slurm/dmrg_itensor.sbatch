#!/bin/bash
# Suites 1-2 (finite DMRG, trivial + U(1)), ITensorMPS side. Submit from the repo root:
#   sbatch benchmark/slurm/dmrg_itensor.sbatch
# Instantiate the env on the login node first:
#   julia --project=benchmark/comparisons/itensor -e 'using Pkg; Pkg.instantiate()'
#SBATCH --job-name=itensor-dmrg
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --partition=ccq            # EDIT if needed
#SBATCH --constraint=icelake       # EDIT: pin ONE CPU generation, same for ALL jobs
#SBATCH --output=benchmark/slurm/logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p benchmark/slurm/logs

export JULIA_NUM_THREADS=1

julia --project=benchmark/comparisons/itensor benchmark/comparisons/itensor/run.jl --suite=1,2 --blas-threads=1
