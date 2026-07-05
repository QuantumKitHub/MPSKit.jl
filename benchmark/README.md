# MPSKit benchmarks

This directory holds two things:

1. The legacy regression suite (`benchmarks.jl`, `MPSKitBenchmarks/`), unchanged.
2. A comparative benchmark harness (`run.jl`, `suites/`, `comparisons/`) measuring MPSKit against other MPS/DMRG libraries.

## Comparative harness

```
julia --project=benchmark benchmark/run.jl --smoke            # suites 1-2, small sizes, minutes
julia --project=benchmark benchmark/run.jl --smoke --suite=5,7
julia --project=benchmark benchmark/run.jl --suite=1          # full suite 1 (long; full runs belong on Rusty, see slurm/)
julia --project=benchmark/comparisons/itensor benchmark/comparisons/itensor/run.jl --smoke
julia --project=benchmark benchmark/plot_results.jl           # plots + comparison tables from results/*.json
julia --project=benchmark benchmark/profile_sweep.jl --chi=256  # diagnostic: where one DMRG2 sweep spends its time
```

Suites (defined in `docs/IMPROVEMENT_PLAN.md` §4.2; numbering follows the plan):

- **Suite 1** — finite two-site DMRG time-to-accuracy: spin-1 Heisenberg chain, N = 100, no symmetry, χ ∈ {64, 128, 256, 512, 1024}; per-sweep (energy, walltime, GC, allocations) trajectories.
- **Suite 2** — same with U(1) (Sz conservation).
- **Suite 5** — TDVP throughput: global quench from the Néel state, same chain; two-site growth phase then single-site measure phase, per-step (⟨Sz⟩, χ, walltime) trajectories, wall seconds per unit physical time at χ ∈ {64, 128, 256}.
- **Suite 7** — thread scaling: the suite-1 workload at χ = 512, one process per (JULIA_NUM_THREADS × BLAS-threads) grid point (see `slurm/threads_*.sbatch` for the grid driver).

Results land in `results/` as one JSON per run, carrying full metadata (Julia and package versions, thread counts, BLAS vendor/threads, hostname, timestamp) plus the trajectory. `slurm/` holds the Rusty submission scripts (manual submission; see its README). `ANALYSIS.md` is the investigation log — skeleton until real cluster results exist.

## Methodology decisions (2026-07-05, maintainer)

- **Investigation first**: this round is about understanding where MPSKit is faster or slower and why; populating `docs/src/benchmarks.md` is a later round.
- **Two-site DMRG on both sides**: ITensorMPS exposes only two-site `dmrg`, so both suites use MPSKit `DMRG2` with `trscheme = truncrank(χ)` against it. No single-site comparison exists.
- **Float64 on both sides**: the Hamiltonian is real-symmetric; matching real arithmetic removes a 2-4x BLAS confound (MPSKit's constructor default is ComplexF64, passed explicitly as Float64 here).
- **Sweep budget 10** (was a 30-sweep placeholder); to be confirmed by a χ = 256 pilot so every χ point plateaus. Per-sweep trajectories are recorded, so time-to-accuracy extraction does not need long post-convergence tails.
- **Fresh random start per χ** (seeded identically): symmetric across libraries and per-χ independent; χ-ramp warm-start protocols are future work.
- **Full runs happen on the Rusty cluster** via the scripts in `slurm/`, submitted manually by the maintainer; local runs are smoke/plumbing only.

## Methodology guardrails (non-negotiable before publishing any number)

- Compare **time-to-accuracy**, never time-per-iteration: sweep semantics and per-sweep costs differ across libraries even at matched two-site updates.
- Identical protocol knobs: same Hamiltonian, χ schedule, truncation policy (rank-χ, cutoff 0, no noise/expansion), element type, and identical `JULIA_NUM_THREADS` / BLAS backend and thread counts on the same machine.
- **Sanity gate**: at matched χ, the converged energies of the two libraries must agree to ~1e-8 before any timing from that pair is meaningful. Smoke runs (4 sweeps, tiny χ) are for plumbing only — they need agree only in leading digits.
- Competitor scripts follow the competitor's *official documented idiom* (citations in the source comments); where exact parity is impossible, the choice made favors the competitor and is documented in `comparisons/itensor/README.md`.
- Publish everything or publish nothing: scripts, environment manifests, raw result JSONs, and plotting code accompany any published table. Losses are reported as prominently as wins.

TeNPy comparisons are out of scope for this round (`comparisons/tenpy/`).
