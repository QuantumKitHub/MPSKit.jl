#!/bin/bash
# The ENTIRE comparative benchmark (suites 1, 2, 5, 7 x both libraries) as ONE job on ONE
# node, so every published pair of numbers comes from the same machine by construction.
# Submit from the repo root:
#
#     sbatch benchmark/slurm/run_all.sbatch
#
# Local end-to-end check (runs the same phases at smoke scale, as plain bash):
#
#     SMOKE=1 bash benchmark/slurm/run_all.sbatch
#
# Instantiate both environments on a login node first (compute nodes may lack registry
# access):
#     julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
#     julia --project=benchmark/comparisons/itensor -e 'using Pkg; Pkg.instantiate()'
#
# Core discipline (node sizes vary, e.g. 64-96 cores — the script adapts via nproc;
# nothing else running: --exclusive):
#   * Phase 1 (DMRG) and phase 2 (TDVP) run the MPSKit and ITensorMPS sides
#     CONCURRENTLY, one single-threaded process per SOCKET (disjoint `taskset` masks, so
#     2 of 96 cores are active and the two sides never share a core, an L3, or a memory
#     controller). The compared pair therefore also sees identical machine state.
#   * Phase 3 (thread-scaling grid) runs ONE grid point at a time with the node to
#     itself; the largest grid point uses 8 + 8 = 16 threads, far below 96.
#   * Inside every Julia process, ThreadPinning.jl pins Julia threads 1:1 to distinct
#     cores of the process's affinity mask (`pinthreads(:affinitymask)`) and, when the
#     OpenBLAS pool has real workers, pins those to further disjoint cores (see
#     `pin_cores` in benchmark/run.jl). Nothing is ever oversubscribed.
#SBATCH --job-name=mpskit-bench-all
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=96:00:00
#SBATCH --partition=gen
#SBATCH --constraint=icelake       # EDIT: pin ONE CPU generation (any single one)
#SBATCH --output=benchmark/slurm/logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
LOGDIR="benchmark/slurm/logs"
mkdir -p "${LOGDIR}"
TAG="${SLURM_JOB_ID:-local}"
trap 'echo ">>> FAILED — check ${LOGDIR}/${TAG}-*.log for the per-process error"' ERR

# --- Julia resolution -----------------------------------------------------------------
# Julia >= 1.12 is REQUIRED: benchmark/ is a Pkg [workspace] member of the repo root, and
# older julia ignores [workspace] entirely — the symptom is a bogus
# "Package X ... does not seem to be installed: Run Pkg.instantiate()" (seen with
# `module load julia` = 1.11.2, job 6579944). Do NOT `module load julia` here; batch
# shells also lack the interactive PATH (job 6579930: "taskset: failed to execute
# julia"). Instead: honor an explicit $JULIA, else pick the newest 1.12+ binary in the
# shared-home juliaup depot (~/.julia/juliaup — GPFS, so identical on workstations and
# cluster nodes, and the same install that resolved the manifests), else fall back to
# PATH and let the version gate below catch it.
if [[ -z "${JULIA:-}" ]]; then
    JULIA=$(ls -d "${HOME}"/.julia/juliaup/julia-1.1[2-9]*/bin/julia 2>/dev/null | sort -V | tail -1 || true)
    [[ -n "${JULIA}" && -x "${JULIA}" ]] || JULIA=julia
fi
echo ">>> julia: $("${JULIA}" --version) ($(command -v "${JULIA}" || echo "${JULIA}"))"
"${JULIA}" -e 'VERSION >= v"1.12" || (println(stderr, "ERROR: julia >= 1.12 required (benchmark/ is a Pkg workspace member; older julia cannot resolve it)"); exit(1))'

SMOKE_FLAG=""
if [[ "${SMOKE:-0}" == "1" ]]; then
    SMOKE_FLAG="--smoke"
    echo ">>> SMOKE mode: pipeline check only, timings meaningless"
fi

MPSKIT=("${JULIA}" --project=benchmark benchmark/run.jl)
ITENSOR=("${JULIA}" --project=benchmark/comparisons/itensor benchmark/comparisons/itensor/run.jl)

# One core per socket for the concurrent single-threaded pairs. Assumes the common
# contiguous numbering (socket 0 = cores 0..N/2-1, socket 1 = cores N/2..N-1); verify
# with `lscpu -e` on the target node and EDIT if the numbering is interleaved.
NCPU=$(nproc)
CORE_MPSKIT=0
CORE_ITENSOR=$(( NCPU / 2 ))
echo ">>> $(hostname): ${NCPU} CPU threads; MPSKit on core ${CORE_MPSKIT}, ITensorMPS on core ${CORE_ITENSOR}"

# Run one MPSKit/ITensorMPS pair concurrently, each single-threaded and pinned to its own
# socket. Fails the job if either side fails.
run_pair() {
    local suites="$1"
    local phase="$2"
    echo ">>> phase ${phase}: suites ${suites}, both libraries concurrently ($(date))"
    JULIA_NUM_THREADS=1 taskset -c "${CORE_MPSKIT}" \
        "${MPSKIT[@]}" ${SMOKE_FLAG} --suite="${suites}" --blas-threads=1 \
        > "${LOGDIR}/${TAG}-${phase}-mpskit.log" 2>&1 &
    local pid_mpskit=$!
    JULIA_NUM_THREADS=1 taskset -c "${CORE_ITENSOR}" \
        "${ITENSOR[@]}" ${SMOKE_FLAG} --suite="${suites}" --blas-threads=1 \
        > "${LOGDIR}/${TAG}-${phase}-itensor.log" 2>&1 &
    local pid_itensor=$!
    wait "${pid_mpskit}"
    wait "${pid_itensor}"
    echo ">>> phase ${phase} done ($(date))"
}

run_pair "1,2" "dmrg"
run_pair "5" "tdvp"

# Phase 3: thread-scaling grid, one point at a time, node otherwise idle. The affinity
# mask is the whole node; ThreadPinning places the point's threads compactly within it.
GRID=("1 1" "4 1" "1 4" "4 4" "8 1" "1 8" "8 8")
if [[ -n "${SMOKE_FLAG}" ]]; then
    GRID=("1 1" "2 2")
fi
for side in mpskit itensor; do
    for point in "${GRID[@]}"; do
        read -r nj nb <<< "${point}"
        echo ">>> phase threads (${side}): julia=${nj} blas=${nb} ($(date))"
        if [[ "${side}" == mpskit ]]; then
            JULIA_NUM_THREADS="${nj}" "${MPSKIT[@]}" ${SMOKE_FLAG} --suite=7 --blas-threads="${nb}" \
                > "${LOGDIR}/${TAG}-threads-${side}-j${nj}b${nb}.log" 2>&1
        else
            JULIA_NUM_THREADS="${nj}" "${ITENSOR[@]}" ${SMOKE_FLAG} --suite=7 --blas-threads="${nb}" \
                > "${LOGDIR}/${TAG}-threads-${side}-j${nj}b${nb}.log" 2>&1
        fi
    done
done

echo ">>> all phases done ($(date)). Results in benchmark/results/, logs in ${LOGDIR}/."
echo ">>> next: julia --project=benchmark benchmark/plot_results.jl"
