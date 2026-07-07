# MPSKit vs ITensorMPS — investigation notes

Status: **round-1 findings recorded** (2026-07-07), from Rusty job 6581850: one exclusive
node (`worker6192`, 64 CPU threads, 2 sockets), all four suites, both libraries, julia
1.12.6, OpenBLAS (LBT ILP64), N = 100 spin-1 Heisenberg throughout. Result files cited
inline; raw JSONs in `results/`, per-process logs in `slurm/logs/6581850-*`.

## Q1. Sanity gates — ALL PASS

- **Energy gate (suites 1–2)**: at every matched χ ∈ {64, 128, 256, 512}, final energies
  agree to ≤ 6.4e-7 absolute on |E| ≈ 138.94, i.e. relative ≤ 4.6e-9 < 1e-8. (Tables
  printed by `plot_results.jl`; files `suite{1,2}_*_N100_20260706-*.json` vs
  `itensor_suite{1,2}_*_N100_20260706-*.json`.)
- **Observable gate (suite 5)**: mid-chain ⟨Sz⟩(t) agrees to ~1.5e-8 at all χ after
  t = 1.3–1.4 of evolution (`suite5_tdvp_N100_20260707-015719.json` vs
  `itensor_suite5_tdvp_N100_20260707-093904.json`); trajectories are visually identical
  (`results/compare_suite5_tdvp.png`).
- **Thread/BLAS parity**: same node, same julia, `nthreads_julia`/`nthreads_blas` match
  per compared pair; suites 1/2/5 ran the pair concurrently pinned to opposite sockets.
  **Caveat**: the four grid points (1,4)/(1,8) × both libraries ran with *unpinned* BLAS
  workers (floating on the otherwise-idle node) due to a pin_cores readback bug, fixed
  after the run (see Q4). Symmetric across libraries; placement noise only.

## Q2. Time-to-accuracy, finite two-site DMRG (suites 1–2)

Time to E − E_ref ≤ 1e-8·|E_ref| (E_ref = best energy across both libraries), seconds:

| χ | MPSKit (trivial) | ITensor (trivial) | MPSKit (U(1)) | ITensor (U(1)) |
|---|---|---|---|---|
| 64  | 17.2 | 16.9 | 1.5 | 11.2 |
| 128 | 122  | 100  | 1.7 | 23.7 |
| 256 | 1017 | 653  | 2.3 | 80.4 |
| 512 | 5660 | 3685 | 2.3 | 301  |

- **Trivial suite (matched random full-χ starts — the apples-to-apples comparison):
  ITensor is ~1.5–1.6× faster at χ ≥ 128**; parity at χ = 64.
- **U(1) suite: MPSKit's numbers are protocol-dominated, not throughput.** Its
  grow-from-seed start crosses the gate during the cheap early growth sweeps (this gapped
  model already meets 1e-8-relative at effective χ ≈ 64), giving 2-second
  "time-to-accuracy" at every χ cap. Legitimate as an idiomatic-workflow comparison —
  each library runs its natural protocol, and practitioners do grow — but the per-sweep
  numbers below are the implementation comparison.
- Steady-state per-sweep cost at full χ = 512 (last sweeps): MPSKit U(1) ≈ 570–590 s vs
  ITensor ≈ 49 s — **ITensor ~12× faster per U(1) two-site sweep**; trivial: MPSKit
  erratic 308–7178 s (!) vs ITensor flat ≈ 608 s.

### Why: inner-eigensolver work policy, not kernels (root-cause hypothesis)

Verified from both sources:

- ITensorMPS `dmrg` caps the local solve at `eigsolve_krylovdim = 3`,
  `eigsolve_maxiter = 1` (src/dmrg.jl:168–170) — a fixed, tiny Krylov budget per site
  update; convergence comes from many cheap sweeps.
- MPSKit's `Defaults.alg_eigsolve` uses `krylovdim = 30` with **dynamic tolerances**
  (`DynamicTol`, `tol_min = 1e-14`, `tol_factor = 1e-3`; src/utility/defaults.jl:26–61):
  the inner tolerance tracks the outer error × 1e-3, floored at 1e-14. Under this
  suite's run-past-convergence protocol (`tol = 0`), the inner solves are driven toward
  machine precision with a 30-vector Krylov space — enormous and *variable* per-sweep
  work. This matches the observed erratic sweep costs (308 → 7178 → 4480 s at χ = 512)
  and allocation swings (120 → 2195 GiB/sweep) where ITensor is flat (608 s, 456 GiB).

Corroboration: in TDVP (Q3), where the inner Krylov tolerance is **matched** (1e-10 both
sides), MPSKit is uniformly 2–4× *faster* — the tensor-contraction stack is not the
bottleneck; the DMRG inner-solve policy is.
<!-- REVIEW: the policy attribution is a strong inference from source defaults + the
TDVP contrast, but the decisive experiment (suite 1 rerun with MPSKit's alg_eigsolve
capped like ITensor's: krylovdim = 3, maxiter = 1, fixed tol) has NOT been run yet. -->

## Q3. TDVP throughput (suite 5) — MPSKit wins 2–4×

Wall-seconds per unit physical time, single-site measure phase (matched Krylov tol
1e-10, ComplexF64 both sides):

| χ | MPSKit | ITensorMPS | ratio |
|---|---|---|---|
| 64  | 217  | 890   | 4.1× |
| 128 | 1237 | 3972  | 3.2× |
| 256 | 8490 | 18924 | 2.2× |

The two-site growth phase shows the same ordering (χ = 256: 6758 s vs 17803 s, 2.6×).
The ratio shrinks with χ as O(χ³) BLAS (identical on both sides) increasingly dominates.

## Q4. Thread scaling (suite 7, χ = 256, 3 sweeps from random full-χ start)

Total workload walltime (s) and speedup vs the library's own (1,1):

| (julia, blas) | MPSKit | speedup | ITensorMPS | speedup |
|---|---|---|---|---|
| (1,1) | 2670 | 1.00 | 335 | 1.00 |
| (4,1) | 2144 | 1.25 | 296 | 1.13 |
| (8,1) | 1837 | 1.45 | 278 | 1.21 |
| (1,4) | 1230 | 2.17 | 231 | 1.45 |
| (1,8) | 863  | 3.10 | 213 | 1.57 |
| (4,4) | 804  | 3.32 | 170 | 1.97 |
| (8,8) | 554  | 4.82 | 151 | 2.21 |

- MPSKit scales notably better (4.8× at (8,8) vs 2.2×) but from an 8× worse
  single-thread baseline on this workload (which Q2's inner-solver policy explains — the
  extra Krylov work is BLAS-heavy and parallelizes). Even at (8,8) MPSKit (554 s) does
  not catch ITensor's single-core 335 s here.
- For both libraries BLAS threads beat Julia threads on this dense two-site workload;
  mixed (8,8) is best and does not oversubscribe.
- (1,4)/(1,8) points: BLAS workers were unpinned (floating on the idle node) due to the
  `pin_cores` bug — it read the affinity mask *after* `pinthreads` had narrowed the
  calling thread to one core, so BLAS pinning bailed out (only manifest at
  `JULIA_NUM_THREADS = 1`; with > 1 the toplevel task sits on an unpinned thread and read
  the full mask). Fixed post-run: mask read before pinning + Julia threads re-pinned
  after `openblas_pinthreads` (whose pool includes the calling thread when
  `JULIA_NUM_THREADS = 1` and would otherwise relocate it). Symmetric noise; rerun of
  those four points is optional (~30 min total on one node).

## Q5. Where the time goes (diagnostics)

- GC fractions at χ = 512 (steady state): MPSKit 7% (trivial) / 18% (U(1)); ITensor 12%
  (trivial) / 31% (U(1)). ITensor pays a *higher* GC share yet is much faster per sweep —
  more evidence the gap is solver work, not memory management.
- Allocation volume tracks the Krylov work: MPSKit's expensive sweeps allocate ~2.2
  TiB/sweep (trivial χ = 512) vs ITensor's flat 456 GiB.
- `profile_sweep.jl` (workstation, χ = 32 shakedown) shows the sweep dominated by
  KrylovKit `eigsolve`/Lanczos over `tensorcontract!` — consistent with the policy story;
  a cluster-node profile at χ = 256–512 is queued as follow-up.

## Round-1 conclusions

1. Both libraries solve the same problems to the same accuracy (all gates pass); the
   harness, parity choices, and pinning discipline held up on the cluster.
2. **MPSKit's tensor stack is fast** — 2–4× ahead in TDVP at matched inner tolerances.
3. **MPSKit's DMRG defaults spend far more inner-eigensolver work per sweep** than
   ITensor's (krylovdim 30 + dynamic tol → machine precision, vs krylovdim 3 × 1 sweep),
   which costs it the two-site DMRG comparison (1.5–1.6× dense, ~12× per-sweep U(1))
   under this fixed-sweep protocol.
4. The U(1) grow-from-seed workflow is a massive practical win (gate in ~2 s at any χ
   cap) and worth advertising as such, clearly labelled as a workflow comparison.

## Follow-ups queued for later rounds

- **Decisive experiment**: rerun suite 1 (and the U(1) per-sweep measurement) with
  MPSKit `DMRG2(alg_eigsolve = ...)` matched to ITensor's inner budget (krylovdim = 3,
  maxiter = 1, fixed tol) — if the gap closes, the round-1 story is confirmed and a
  defaults discussion for MPSKit's ground-state solvers is warranted.
- Rerun grid points (1,4)/(1,8) with the fixed pin_cores (optional, ~30 min).
- Cluster-node `profile_sweep.jl` at χ = 256/512 to quantify eigsolve vs contraction vs
  SVD shares directly.
- χ-ramp warm-start protocol; SU(2) capability suite; TeNPy; quasi-2D cylinder.
- Publishing: populate `docs/src/benchmarks.md` + README chart only after the losses/wins
  above are reviewed and the follow-up experiments settle the DMRG story.
