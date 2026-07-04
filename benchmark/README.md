# MPSKit benchmarks

This directory holds two things:

1. The legacy regression suite (`benchmarks.jl`, `MPSKitBenchmarks/`), unchanged.
2. A comparative benchmark harness (`run.jl`, `suites/`, `comparisons/`) measuring MPSKit against other MPS/DMRG libraries.

## Comparative harness

```
julia --project=benchmark benchmark/run.jl --smoke            # both suites, small sizes, minutes
julia --project=benchmark benchmark/run.jl --suite=1          # full suite 1 (long)
julia --project=benchmark/comparisons/itensor benchmark/comparisons/itensor/run.jl --smoke
julia --project=benchmark benchmark/plot_results.jl           # plots from results/*.json
```

Suites (defined in `docs/IMPROVEMENT_PLAN.md` §4.2):

- **Suite 1** — finite DMRG time-to-accuracy: spin-1 Heisenberg chain, N = 100, no symmetry, χ ∈ {64, 128, 256, 512, 1024}; per-sweep (energy, walltime) trajectories.
- **Suite 2** — same with U(1) (Sz conservation).

Results land in `results/` as one JSON per run, carrying full metadata (Julia and package versions, thread counts, BLAS vendor/threads, hostname, timestamp) plus the trajectory.

## Methodology guardrails (non-negotiable before publishing any number)

- Compare **time-to-accuracy**, never time-per-iteration: sweep semantics differ across libraries (MPSKit runs single-site DMRG here; ITensorMPS runs its recommended two-site variant).
- Identical protocol knobs: same Hamiltonian, χ schedule, truncation policy (fixed χ, cutoff 0, no noise/expansion), and identical `JULIA_NUM_THREADS` / BLAS backend and thread counts on the same machine.
- **Sanity gate**: at matched χ, the converged energies of the two libraries must agree to ~1e-8 before any timing from that pair is meaningful. Smoke runs (4 sweeps, tiny χ) are for plumbing only — they need agree only in leading digits.
- Competitor scripts follow the competitor's *official documented idiom* (citations in the source comments); where exact parity is impossible, the choice made favors the competitor and is documented in `comparisons/itensor/README.md`.
- Publish everything or publish nothing: scripts, environment manifests, raw result JSONs, and plotting code accompany any published table. Losses are reported as prominently as wins.

## Known open items before full runs

- The U(1) virtual-sector allocation in `suites/suite2_dmrg_u1.jl` is a flagged judgment call (`# REVIEW`); the ITensor smoke run reached a lower energy at matched χ, so the MPSKit-side sector split must be reviewed by the maintainer before suite-2 timings mean anything.
- Full-mode `nsweeps = 30` is a placeholder; tune so every χ point converges past the 1e-8 gate.
- Each χ point currently starts from a fresh random state (no ramp-up schedule) on both sides — symmetric, but not how practitioners run DMRG; revisit together.

TeNPy comparisons are out of scope for this round (`comparisons/tenpy/`).
