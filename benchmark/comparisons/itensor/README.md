# ITensorMPS.jl parity benchmarks (suites 1, 2, 5, 7)

The ITensorMPS.jl side of the MPSKit competitive benchmark described in
`docs/IMPROVEMENT_PLAN.md` §4. These scripts reproduce, as closely as the two libraries
allow, the exact protocol of the MPSKit side (`benchmark/suites/`, `benchmark/run.jl`):
finite DMRG **time-to-accuracy** for the spin-1 Heisenberg chain, no symmetry (suite 1) and
with U(1) Sz conservation (suite 2); **TDVP throughput** for a global quench (suite 5, see
the parity notes in `suite5_tdvp.jl`, including the matched Krylov-exponentiation
tolerance); and **thread scaling** of the suite-1 workload (suite 7, one process per
thread-grid point — see `benchmark/slurm/run_all.sbatch`).

Everything here uses the **official, documented ITensorMPS idiom** so that a reader from the
ITensor community cannot call it a strawman. Every nontrivial API call carries a comment
citing the doc page it came from.

## Sanity gate (read this first)

**Energies must agree with the MPSKit run to ~1e-8 at matched χ before any timing is
meaningful.** Time-to-accuracy is only comparable if both libraries are solving the *same*
variational problem to the *same* energy. Before trusting a single wall-time number:

1. Run the MPSKit side and this side at the same N and χ schedule.
2. Compare the converged (final-sweep, largest-χ, full-run) energies. They must match
   to ~1e-8. If they do not, the Hamiltonian, the symmetry sector, or the χ convention is
   mismatched and **no timing from that run may be published.**

At the *smoke* scale (N = 20, χ ∈ {8, 16}, only 4 sweeps) exact agreement is **not**
expected — 4 sweeps is far from convergence, and fixed-χ variational optima depend on the
random start and can plateau in distinct local minima. Leading-digit agreement
(all energies ≈ −26.8) is the smoke-scale bar. The 1e-8 gate applies to the full,
converged run only.

## How to run

From the repository root:

```bash
# smoke check (N=20, χ∈[8,16], 4 sweeps) — verifies the pipeline, NOT a result
JULIA_NUM_THREADS=1 julia --project=benchmark/comparisons/itensor \
    benchmark/comparisons/itensor/run.jl --smoke

# full run (N=100, χ∈[64,128,256,512]; suite 1: 6 sweeps, suite 2: 10) — full runs
# belong in the single-node cluster job, benchmark/slurm/run_all.sbatch
JULIA_NUM_THREADS=1 julia --project=benchmark/comparisons/itensor \
    benchmark/comparisons/itensor/run.jl

# one suite only
julia --project=benchmark/comparisons/itensor benchmark/comparisons/itensor/run.jl --suite=1
```

First run instantiates the env (downloads ITensors/ITensorMPS; a few minutes):

```bash
julia --project=benchmark/comparisons/itensor -e 'using Pkg; Pkg.instantiate()'
```

Results are written to the shared `benchmark/results/` directory as
`itensor_suite1_dmrg_trivial_N*.json` / `itensor_suite2_dmrg_u1_N*.json`. They use the
**same JSON schema** as the MPSKit side (plus a `"library": "ITensorMPS"` field), so
`benchmark/plot_results.jl`'s `plot_suite_trajectories` reads them directly. The
`itensor_` filename prefix keeps them from colliding with the MPSKit files that
`plot_all_suite_results` auto-discovers.

### Threading / BLAS parity (non-negotiable, §4.3)

The MPSKit side records `nthreads_julia` (`Threads.nthreads()`) and `nthreads_blas`
(`LinearAlgebra.BLAS.get_num_threads()`) in `benchmark/suites/common.jl`; this side records
the **same fields** in `common.jl`'s `collect_metadata`. For the timings to mean anything,
**both sides must be launched with identical `JULIA_NUM_THREADS` and identical BLAS thread
counts.** Set `JULIA_NUM_THREADS` explicitly on both, confirm `nthreads_blas` matches across
the two result files' metadata before comparing, and document the hardware.

## Parity choices, and which side each favors

| # | Aspect | MPSKit side | ITensor side | Exact parity? | Favors |
|---|--------|-------------|--------------|---------------|--------|
| 1 | Model | `heisenberg_XXX(…; J=1, spin=1)` = J·Σ Sᵢ·Sⱼ | official `OpSum` with `"Sz"`,`"S+"`,`"S-"` (DMRG tutorial), same H | **exact** | neither |
| 2 | Fixed χ | `DMRG2` with `trscheme=truncrank(χ)`: every two-site update truncates back to ≤ χ | `random_mps(...; linkdims=χ)` + `maxdim=mindim=χ`, `cutoff=0`, `noise=0` | **matched** | neither |
| 3 | Truncation cutoff | none (`truncrank` is rank-only) | `cutoff=0.0` | **exact** | neither (0 = most favorable to ITensor anyway: never discards weight) |
| 4 | Element type | `Float64` (explicit) | `Float64` (ITensor default / documented idiom) | **exact** | neither |
| 5 | DMRG update | **two-site** (`DMRG2`, `trscheme=truncrank(χ)`), both suites | **two-site** (ITensor's default `dmrg`, the recommended idiom; no single-site `dmrg` exists) | **matched** | neither |
| 6 | Subspace expansion / noise | none | `noise=0.0` (matches MPSKit) | **exact** | conservative for ITensor — see note |
| 7 | Sweeps | `maxiter=nsweeps`, `tol=0` (runs all sweeps) | `nsweeps` identical, no early stop in observer | **exact** | neither |
| 8 | U(1) sector | Sz = 0 sector; small sector-diverse seed, **`DMRG2` distributes χ across sectors itself** | `random_mps(sites, state; linkdims=χ)` with alternating Néel state; **ITensor distributes χ across sectors itself** | **matched** (both libraries choose the split) | neither |
| 9 | Warmup | N=6, χ=4, 2 sweeps before timing | identical warmup | **exact** | neither |
| 10 | Per-sweep recording | `finalize` hook, one (E, t) per sweep; `t0` set just before solve | `AbstractObserver.measure!` on `sweep_is_done`; `t0` reset just before `dmrg` | **exact** | neither |

### Notes on the judgment calls

- **Element type (row 4).** ITensorMPS's `random_mps` is `Float64` by default and that is
  the correct, documented choice for this real-symmetric Hamiltonian. MPSKit's constructors
  default to `ComplexF64`, so the MPSKit side passes `Float64` explicitly. Earlier
  revisions ran MPSKit at its complex default — a 2-4x BLAS handicap that muddied the
  algorithmic comparison; matched real arithmetic on both sides was a maintainer decision
  (2026-07-05).

- **Update scheme (row 5).** Both suites now use two-site DMRG on both sides: ITensorMPS
  exposes only the two-site `dmrg` (its tutorial/FAQ-recommended idiom), so no single-site
  comparison exists, and the earlier single-site suite-1 variant was dropped (maintainer
  decision, 2026-07-05). Suite 2's `DMRG2` also replaced an earlier hand-picked static
  U(1) sector split that was demonstrably suboptimal (−26.4027 vs −26.8188 at the χ = 8
  smoke point). Sweep costs still differ in implementation detail —
  §4.3 mandates comparing **time-to-accuracy, not time-per-sweep**; `nsweeps` is matched
  on both sides only as a loop bound.

- **Initial state (rows 2/8).** Both sides start from a random state seeded identically
  per χ. Trivial suite: both start at full χ. U(1) suite: ITensor's `random_mps` starts
  at full χ with its own per-sector split, MPSKit grows from a small sector-diverse seed
  within the first sweeps (`DMRG2` redistributes χ per update). Each library follows its
  natural protocol; neither split is hand-picked.

- **Noise / subspace expansion (row 6).** ITensor's DMRG FAQ recommends a small, decreasing
  `noise` schedule to help two-site DMRG escape local minima at fixed χ. We set `noise=0.0`
  to match MPSKit's no-expansion (`alg_expand=nothing`) protocol exactly. This is the
  *conservative* choice for ITensor (noise would likely help it), chosen for strict
  truncation-policy parity, not to flatter either side. **A user wanting maximum ITensor
  robustness can pass a `noise` schedule to `dmrg_trajectory` / `dmrg`** — the trade-off is
  documented here rather than hidden. This mainly matters at very small χ and few sweeps;
  see the smoke observation below.

## Smoke verification (recorded 2026-07-05, workstation)

`--smoke` (N=20, χ∈{8,16}, 4 sweeps / 4 TDVP measure steps), `JULIA_NUM_THREADS=1`,
BLAS threads = 1, OpenBLAS, Julia 1.12.6, ITensorMPS 0.4.1 / ITensors 0.9.30, both sides
Float64 + two-site DMRG. Final-sweep energies vs the MPSKit smoke reference:

| suite | χ | ITensorMPS | MPSKit reference | |Δ| |
|-------|---|-----------|------------------|-----|
| 1 (trivial) | 8  | −26.8048 | −26.8188 | 0.0139 ⚑ |
| 1 (trivial) | 16 | −26.8280 | −26.8373 | 0.0093 |
| 2 (U(1))    | 8  | −26.8188 | −26.8188 | 6.0e−5 |
| 2 (U(1))    | 16 | −26.8313 | −26.8373 | 0.0060 |

All energies agree to leading digits (≈ −26.8), the smoke-scale bar. ⚑ The suite-1 χ=8
point: the ITensor run **plateaued in a fixed-χ local minimum** by sweep 2 while MPSKit's
run kept improving to −26.8188 from its different random start. This is exactly the
fixed-χ local-minimum behavior ITensor's FAQ recommends `noise` for (row 6), amplified by
only 4 sweeps — a smoke-scale convergence artifact, **not** a Hamiltonian/sector mismatch
(the U(1) χ=8 pair agrees to 6e−5 on the same Hamiltonian). The full run at large χ is
where the 1e-8 sanity gate must hold; verify it there before publishing any timing.

**Suite 5 (TDVP) smoke gate**: the mid-chain ⟨Sz⟩ after the full grow+measure trajectory
agrees between the libraries to ~1e−13 at both χ (χ=8: −0.722010909330507 vs
−0.722010909330496; χ=16: −0.660446916466855 vs −0.660446916466839) — the two sides
demonstrably integrate the same quench with the same protocol.

## Do not publish numbers from an unverified run

No timing here is a published result. Populate `docs/src/benchmarks.md` only after: (a) the
sanity gate passes on the full run, (b) MPSKit and ITensor were run on the same hardware with
matched thread/BLAS settings (cross-check the metadata in both result files), and (c) losses
are reported as honestly as wins (§4.3).
