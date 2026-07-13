# Maintainer review summary — docs improvement round 1

Round 1 of `docs/IMPROVEMENT_PLAN.md` (W0 trust sweep + W1 happy path + benchmark scaffold), executed 2026-07-04 on `docs/content-wave1`.
Everything below either needs your physics judgment or records a decision deferred to you.

## 1. REVIEW flags awaiting physics sign-off

**`docs/src/tutorials/first_groundstate.md` (7 flags)** — ferromagnetic framing of the −J σᶻσᶻ term; g = 0.5 as the ordered side; the DMRG sweep description; σᶻ as order parameter + variance phrasing; PBC-vs-OBC rationale for the sweep; the `abs(...)` symmetry argument; crossover-shape claims.
Empirical input for the last one: the verified run shows a **sharp step** at g ≈ 1 (M > 0.8 for g ≤ 0.9, ~1e-8 for g ≥ 1.0), not a smooth crossover — consistent with DMRG locking onto one symmetry-broken state; align the prose accordingly.

**`docs/src/tutorials/thermodynamic_limit.md` (6 flags)** — VUMPS one-line characterization; `correlation_length` description/units; growth toward criticality (verified numerically: ξ = 0.58 at g = 0.5 → 7.45 at g = 1.0); finite-D capping; sharper-than-finite comparison; the finite-D spontaneous-symmetry-breaking framing (the writer's least-confident claim).

**`examples/quantum1d/0.tfim-groundstate/main.jl` (8 flags)** — critical point/framing; D-sharpens-everything; abs() for the infinite branch; finite-vs-infinite rounding; `entropy(::InfiniteMPS)` shape assumption; entropy-peak interpretation; correlation-length capping; the three-diagnostics-agree summary.

**`docs/src/examples/index.md` (2 flags)** — Haldane-SPT teaser rigor; Bose-Hubbard "Mott/superfluid" gloss.

**`benchmark/suites/suite2_dmrg_u1.jl` (1 flag, blocking for suite-2 numbers)** — the hand-rolled U(1) virtual-sector allocation (uniform over charges −4:4) is demonstrably suboptimal: ITensor's automatic allocation reached −26.8188 at χ=8 where this split reached −26.4027. Fix before any suite-2 timing is published.

## 2. Code/doc mismatches found (pick a side)

- `correlator(ψ, O₁₂, i, js)` with `first(js) ≤ i` logs `@error` but does NOT throw (`src/algorithms/correlators.jl:17`); `howto/observables.md` says "will throw an error". Fix code or wording.
- `expectation_value(::InfiniteMPS, ::InfiniteMPOHamiltonian)` returns a scalar (unit-cell sum); the old `sum(real(E0))/length(mps)` idiom was a no-op sum on a scalar. New pages state it correctly; the convention is worth a docstring note.
- `howto/bond_dimension.md` documented a nonexistent `truncbelow(x)` scheme and twice credited MPSKit with re-exporting `truncrank` (it's TensorKit). Fixed (commit `ad646a7`), but this survived wave-1 review — a quick grep-audit of other wave-1 tables for phantom symbols is cheap insurance.

## 3. Content debt observed, not touched this round

- CHANGELOG entries stop at 0.13.11 though tags reach v0.13.13.
- Example pages dump raw solver logs inline (`1.ising-cft`, `3.ising-dqpt`, `4.xxz-heisenberg` ~400 lines, `6.hubbard`); suggest verbosity = 0 in their `main.jl`.
- `6.hubbard/index.md` has an empty citation link `Eq. (6.82) in []()`; `7.xy-finiteT` has an unresolved `!!! todo`.
- `find_groundstate`'s docstring still omits keyword defaults and the auto-selection heuristic; only `expectation_value` has a docstring doctest (the whole package has exactly one). This is workstream W4.

## 4. Deferred decisions (from the approved plan)

- Fate of `man/environments.md` (still commented out of the build).
- The `!!! warning`s in `lib/observables.md` (`expectation_value(::MultilineMPS)`, `variance(::InfiniteQP)`): fix or keep quarantined.
- Loosening pre-1.0 `[compat]` pins.
- Community venue (GitHub Discussions vs a chat channel) — README/landing link slots are ready when chosen.

## 5. Benchmark status

Harness + ITensorMPS parity scripts are smoke-verified end-to-end; energies agree to leading digits; every parity judgment call is documented in `benchmark/comparisons/itensor/README.md` and deliberately favors ITensor where exact parity is impossible (two-site vs single-site, Float64 vs ComplexF64).
Before full runs: fix the suite-2 sector allocation (above), tune `nsweeps`, and decide on ramp-up schedules.
No numbers are published anywhere, per the plan.
