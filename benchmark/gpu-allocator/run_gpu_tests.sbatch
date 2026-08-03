#!/bin/bash
# Run the MPSKit CUDA test suite, including the new algorithm-level allocator tests.
#
# There is no GPU on the workstation, so this is the only way to exercise
# `MPSKit.allocatortype(::Type{<:CuArray})` and confirm that a device-backed DMRG/TDVP draws its
# scratch space from `CUDAAllocator` rather than a host buffer.
#
# Submit from the repository root:
#     sbatch benchmark/gpu-allocator/run_gpu_tests.sbatch
#
# Adjust --partition/--gres to whatever GPU partition you have access to; the values below are
# placeholders. Check with `sinfo -o "%P %G %N"`.

#SBATCH --job-name=mpskit-gpu-alloc
#SBATCH --output=benchmark/gpu-allocator/logs/%x-%j.out
#SBATCH --error=benchmark/gpu-allocator/logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu          # placeholder
#SBATCH --gres=gpu:1             # placeholder

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?must be submitted with sbatch}"
mkdir -p benchmark/gpu-allocator/logs

# The depot is precompiled under 1.12.6; `module load julia` gives 1.11.2, which fails with a
# StaticData error. Use the juliaup install on GPFS directly.
JULIA="${HOME}/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia"
test -x "$JULIA" || { echo "julia not found at $JULIA" >&2; exit 1; }

echo "host:     $(hostname)"
echo "julia:    $("$JULIA" --version)"
nvidia-smi || echo "WARNING: nvidia-smi unavailable"

export JULIA_NUM_THREADS=4
# one BLAS thread per Julia thread would oversubscribe; the GPU does the heavy lifting anyway
export OPENBLAS_NUM_THREADS=1

"$JULIA" --project=test -e '
    using Pkg
    Pkg.instantiate()

    using CUDA, cuTENSOR
    @info "CUDA status" functional = CUDA.functional() cutensor = cuTENSOR.functional()
    CUDA.functional() || error("CUDA is not functional on this node")
    CUDA.versioninfo()

    include(joinpath(@__DIR__, "test", "setup", "testsetup.jl"))
    using .TestSetup
    using Test

    @testset "MPSKit CUDA" verbose = true begin
        for f in ("cu_adapt.jl", "states.jl", "operators.jl", "algorithms.jl")
            @testset "$f" begin
                include(joinpath(@__DIR__, "test", "gpu", "cuda", f))
            end
        end
    end
'
