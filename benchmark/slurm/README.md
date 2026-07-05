# Rusty submission scripts for the comparative benchmarks

One sbatch script per (workload x library), all single exclusive node. These are
submitted **manually by the maintainer** — nothing here self-submits.

## Before submitting

1. **Pin one CPU generation** — edit `--constraint=` (and `--partition=` if needed) in
   ALL six scripts to the same value. Timings across different node types are not
   comparable, and the metadata in the result JSONs records only hostname, not node type.
2. **Instantiate both environments on a login node** (compute-node registry access is not
   guaranteed):

   ```bash
   julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
   julia --project=benchmark/comparisons/itensor -e 'using Pkg; Pkg.instantiate()'
   ```

3. Make sure `julia` is on `PATH` in batch jobs (juliaup or `module load julia` in your
   shell init).

## Submission order (from the repo root)

Nothing depends on anything else; submit all six whenever. The DMRG pair is the priority:

```bash
sbatch benchmark/slurm/dmrg_mpskit.sbatch      # suites 1-2, ~hours (chi up to 1024)
sbatch benchmark/slurm/dmrg_itensor.sbatch
sbatch benchmark/slurm/tdvp_mpskit.sbatch      # suite 5
sbatch benchmark/slurm/tdvp_itensor.sbatch
sbatch benchmark/slurm/threads_mpskit.sbatch   # suite 7, 7 grid points sequentially
sbatch benchmark/slurm/threads_itensor.sbatch
```

Results accumulate in `benchmark/results/` (one JSON per run/grid point); logs in
`benchmark/slurm/logs/`. Time limits are generous guesses — if a job times out, the χ
points already completed are still lost (one JSON per *suite*, written at the end), so
bump `--time` rather than resubmitting blindly.

## Sanity-gate checklist before reading any timing

Run `julia --project=benchmark benchmark/plot_results.jl` locally on the synced
`results/` directory, then check:

1. **Energy gate (suites 1-2)**: at each matched χ, final MPSKit and ITensorMPS energies
   agree to ~1e-8 (relative). If not, no timing from that pair means anything.
2. **Observable gate (suite 5)**: the ⟨Sz⟩(t) trajectories of the two libraries lie on
   top of each other at matched χ (they solve the same Schrödinger equation with the
   same integrator family; visible divergence = protocol mismatch, not physics).
3. **Thread parity (all)**: `nthreads_julia`, `nthreads_blas`, and `blas_config` match
   between the two libraries' metadata for every compared pair (suite 7 pairs match
   point-by-point).
4. **Same hardware**: all result files report hostnames of the same node type.
5. Only then look at wall times. Report losses as prominently as wins (§4.3).
