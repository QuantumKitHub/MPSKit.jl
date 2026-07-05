# Rusty submission script for the comparative benchmarks

**One script, one job, one node**: `run_all.sbatch` runs all four suites for both
libraries inside a single exclusive-node allocation, so every compared pair of numbers
comes from the same machine by construction (no cross-job node-type mismatch to check
away afterwards). It is submitted **manually by the maintainer** — nothing here
self-submits.

## Core discipline (96-core nodes)

- **Phases 1–2 (DMRG suites 1–2, then TDVP suite 5)**: the MPSKit and ITensorMPS sides
  run *concurrently*, each single-threaded (`JULIA_NUM_THREADS=1`, BLAS = 1) and
  `taskset`-pinned to one core on *opposite sockets* — 2 of 96 cores active, no shared
  core, L3, or memory controller, and the compared pair sees identical machine state.
- **Phase 3 (thread-scaling suite 7)**: one (JULIA_NUM_THREADS × BLAS-threads) grid point
  at a time with the node otherwise idle; the largest point uses 16 threads.
- **Inside every Julia process**, ThreadPinning.jl pins Julia threads 1:1 to distinct
  cores of the process affinity mask (`pinthreads(:affinitymask)`), and pins OpenBLAS
  workers to further disjoint mask cores when the pool is real (`pin_cores` in
  `benchmark/run.jl` / the ITensor `run.jl`). Nothing is oversubscribed at any point.
  If a log ever shows the "thread pinning failed" warning, the timings from that run are
  not publishable.

## Before submitting

1. Edit `--partition` / `--constraint` in `run_all.sbatch` (any single CPU generation).
2. Check the socket→core numbering assumption (`lscpu -e` on the target node): the script
   assumes socket 0 = cores 0..47, socket 1 = cores 48..95 and pins the concurrent pair
   to cores 0 and 48. Edit `CORE_MPSKIT` / `CORE_ITENSOR` if the numbering interleaves.
3. Instantiate both environments on a login node:

   ```bash
   julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
   julia --project=benchmark/comparisons/itensor -e 'using Pkg; Pkg.instantiate()'
   ```

4. Optional local rehearsal (same phases at smoke scale, plain bash, no Slurm):

   ```bash
   SMOKE=1 bash benchmark/slurm/run_all.sbatch
   ```

Then, from the repo root:

```bash
sbatch benchmark/slurm/run_all.sbatch
```

## Walltime calibration

The 2026-07-05 MPSKit pilot (χ = 256, N = 100, two-site, 8 workstation threads) averaged
≈ 4.5 min/sweep; single-threaded χ = 512 extrapolates to a few hours per sweep. Phases
run back-to-back: DMRG (dominant, tens of hours) → TDVP → thread grid; the 96 h limit is
sized for that with margin. χ = 1024 is deliberately out of the default schedule (days of
single-threaded node-time; extend later if the 512 numbers justify it). If the job times
out, note that each *suite* writes its JSON only at the end — completed χ points inside
an unfinished suite are lost, so bump `--time` rather than resubmitting blindly.

Per-phase, per-library logs land in `benchmark/slurm/logs/` (`<jobid>-<phase>-<side>.log`);
results accumulate in `benchmark/results/` (one JSON per suite run / grid point).

## Sanity-gate checklist before reading any timing

Run `julia --project=benchmark benchmark/plot_results.jl` on the synced `results/`
directory, then check:

1. **Energy gate (suites 1-2)**: at each matched χ, final MPSKit and ITensorMPS energies
   agree to ~1e-8 (relative). If not, no timing from that pair means anything.
2. **Observable gate (suite 5)**: the ⟨Sz⟩(t) trajectories of the two libraries lie on
   top of each other at matched χ (smoke-scale agreement was ~1e-13; visible divergence
   = protocol mismatch, not physics).
3. **Thread parity (all)**: `nthreads_julia`, `nthreads_blas`, and `blas_config` match
   between the two libraries' metadata for every compared pair (suite 7 pairs match
   point-by-point), and no log contains the "thread pinning failed" warning.
4. Only then look at wall times. Report losses as prominently as wins (§4.3).
