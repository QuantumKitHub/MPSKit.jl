# ITensorMPS.jl parity benchmarks (suites 1 & 2)

The ITensorMPS.jl side of the MPSKit competitive benchmark described in
`docs/IMPROVEMENT_PLAN.md` §4. These scripts reproduce, as closely as the two libraries
allow, the exact protocol of the MPSKit side (`benchmark/suites/`, `benchmark/run.jl`):
finite DMRG **time-to-accuracy** for the spin-1 Heisenberg chain, no symmetry (suite 1) and
with U(1) Sz conservation (suite 2).

Everything here uses the **official, documented ITensorMPS idiom** so that a reader from the
ITensor community cannot call it a strawman. Every nontrivial API call carries a comment
citing the doc page it came from.

## Sanity gate (read this first)

**Energies must agree with the MPSKit run to ~1e-8 at matched χ before any timing is
meaningful.** Time-to-accuracy is only comparable if both libraries are solving the *same*
variational problem to the *same* energy. Before trusting a single wall-time number:

1. Run the MPSKit side and this side at the same N and χ schedule.
2. Compare the converged (final-sweep, largest-χ, full 30-sweep) energies. They must match
   to ~1e-8. If they do not, the Hamiltonian, the symmetry sector, or the χ convention is
   mismatched and **no timing from that run may be published.**

At the *smoke* scale (N = 20, χ ∈ {8, 16}, only 4 sweeps) exact agreement is **not**
expected — 4 sweeps is far from convergence, and fixed-χ variational optima differ slightly
across gauges, truncation policy, and single- vs two-site updates. Leading-digit agreement
(all energies ≈ −26.8) is the smoke-scale bar. The 1e-8 gate applies to the full,
converged run only.

## How to run

From the repository root:

```bash
# smoke check (N=20, χ∈[8,16], 4 sweeps) — verifies the pipeline, NOT a result
JULIA_NUM_THREADS=1 julia --project=benchmark/comparisons/itensor \
    benchmark/comparisons/itensor/run.jl --smoke

# full run (N=100, χ∈[64,128,256,512,1024], 30 sweeps)
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
| 2 | Fixed χ | random full-χ `FiniteMPS`, `alg_expand=nothing`, non-truncating gauge | `random_mps(...; linkdims=χ)` + `maxdim=mindim=χ`, `cutoff=0`, `noise=0` | **exact** | neither |
| 3 | Truncation cutoff | none (fixed χ) | `cutoff=0.0` | **exact** | neither (0 = most favorable to ITensor anyway: never discards weight) |
| 4 | Element type | `ComplexF64` (MPSKit default) | `Float64` (ITensor default / documented idiom) | no | **ITensor** (real is faster; forcing complex would only slow ITensor and isn't its idiom) |
| 5 | DMRG update | **single-site** (`DMRG`) | **two-site** (ITensor's default `dmrg`, the recommended idiom) | no | **ITensor** (two-site is more robust against local minima) |
| 6 | Subspace expansion / noise | none | `noise=0.0` (matches MPSKit) | **exact** | conservative for ITensor — see note |
| 7 | Sweeps | `maxiter=nsweeps`, `tol=0` (runs all sweeps) | `nsweeps` identical, no early stop in observer | **exact** | neither |
| 8 | U(1) sector | Sz = 0 sector; interior bond split hand-picked (`qmax`, a flagged REVIEW guess) | `random_mps(sites, state; linkdims=χ)` with alternating Néel state; **ITensor distributes χ across sectors itself** | protocol-equivalent | **ITensor** (no hand-tuning of the sector split — the library chooses) |
| 9 | Warmup | N=6, χ=4, 2 sweeps before timing | identical warmup | **exact** | neither |
| 10 | Per-sweep recording | `finalize` hook, one (E, t) per sweep; `t0` set just before solve | `AbstractObserver.measure!` on `sweep_is_done`; `t0` reset just before `dmrg` | **exact** | neither |

### Notes on the judgment calls

- **Element type (row 4).** ITensorMPS's `random_mps` is `Float64` by default and that is
  the correct, documented choice for this real-symmetric Hamiltonian. MPSKit's constructors
  default to `ComplexF64`. Real arithmetic is strictly cheaper, so this **favors ITensor**.
  We keep it: forcing ITensor into complex would slow it down and depart from its idiom.

- **Single- vs two-site DMRG (row 5).** The MPSKit protocol uses single-site DMRG; ITensor's
  default `dmrg` is two-site, which is also what its DMRG tutorial and FAQ recommend. We use
  ITensor's recommended two-site update. Because two-site DMRG is generally *more robust*
  against fixed-χ local minima, this **favors ITensor**. The two updates have different
  per-sweep cost and semantics — precisely why §4.3 mandates comparing **time-to-accuracy,
  not time-per-sweep**. `nsweeps` is matched on both sides only as a loop bound, not as a
  claim that a sweep costs the same.

- **Noise / subspace expansion (row 6).** ITensor's DMRG FAQ recommends a small, decreasing
  `noise` schedule to help two-site DMRG escape local minima at fixed χ. We set `noise=0.0`
  to match MPSKit's no-expansion (`alg_expand=nothing`) protocol exactly. This is the
  *conservative* choice for ITensor (noise would likely help it), chosen for strict
  truncation-policy parity, not to flatter either side. **A user wanting maximum ITensor
  robustness can pass a `noise` schedule to `dmrg_trajectory` / `dmrg`** — the trade-off is
  documented here rather than hidden. This mainly matters at very small χ and few sweeps;
  see the smoke observation below.

## Smoke verification (recorded 2026-07-04, this machine)

`--smoke` (N=20, χ∈{8,16}, 4 sweeps), `JULIA_NUM_THREADS=1`, OpenBLAS, 16 BLAS threads,
Julia 1.12.6, ITensorMPS 0.4.1 / ITensors 0.9.30. Final-sweep energies vs the MPSKit smoke
reference:

| suite | χ | ITensorMPS | MPSKit reference | |Δ| |
|-------|---|-----------|------------------|-----|
| 1 (trivial) | 8  | −26.8048 | −26.8188 | 0.0140 ⚑ |
| 1 (trivial) | 16 | −26.8280 | −26.8334 | 0.0053 |
| 2 (U(1))    | 8  | −26.8188 | — | — |
| 2 (U(1))    | 16 | −26.8313 | — | — |

All energies agree to leading digits (≈ −26.8), as expected at smoke scale. ⚑ The suite-1
χ=8 point is 0.014 off — just over the ~1e-2 flag threshold — because that fixed-χ=8,
no-noise, two-site run **plateaued in a local minimum** by sweep 2 (trajectory:
−26.7983 → −26.8048 → −26.8048 → −26.8048) while MPSKit's single-site run kept improving to
−26.8188. This is exactly the fixed-χ local-minimum behavior ITensor's FAQ recommends
`noise` for (row 6), amplified by only 4 sweeps. It is a smoke-scale convergence artifact,
**not** a Hamiltonian/sector mismatch — note the U(1) χ=8 run reaches −26.8188 (matching the
MPSKit reference) from its Néel-seeded start. The full 30-sweep run at large χ is where the
1e-8 sanity gate must hold; verify it there before publishing any timing.

## Do not publish numbers from an unverified run

No timing here is a published result. Populate `docs/src/benchmarks.md` only after: (a) the
sanity gate passes on the full run, (b) MPSKit and ITensor were run on the same hardware with
matched thread/BLAS settings (cross-check the metadata in both result files), and (c) losses
are reported as honestly as wins (§4.3).
