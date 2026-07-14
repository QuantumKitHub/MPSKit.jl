# `ParallelBUG` implementation status

Companion to `PARALLELBUG_design.md`. Records what is implemented and what remains.
**The integrator works** (first-order gates pass); it remains marked experimental because the
local solves are still executed serially and step rejection is not implemented.

## What is implemented (branch `ld-parallelbug`)

* `ParallelBUG <: Algorithm` struct + kw-constructor + docstring, registered and exported
  (`src/algorithms/timestep/bug.jl`, which holds both BUG integrators).
* A `timestep!` + copying `timestep` implementing Ceruti et al. 2024 (arXiv:2412.00858) Alg. 1–4
  specialized to the caterpillar tree rooted at site `L`, in two phases:
  1. **Frozen-snapshot Galerkin evolutions** (Alg. 2/3): one frozen `ψ₀ = copy(ψ)` + `envs₀`; the
     interior amplitude-weighted centers `AC[i]` (the paper's `Y_τ⁰ = U_τ⁰S_τ⁰`) and the root
     center `AC[L]` are each evolved forward from that snapshot. These `L` local solves are
     mutually independent — the parallel-in-time structure (currently executed serially).
  2. **Leaves→root augmentation** (Alg. 4): at bond `i` the new directions `Ũᵢ` are orthonormalized
     against the zero-padded old isometry `[AL⁰ᵢ; 0]` from the evolved center *stacked with the
     first-order coupling block* `C̃ᵢ` on the new rows of bond `i-1`. The couplings are one-site
     effective derivatives with the **mixed** left environment `⟨Ũ-chain|H|AL⁰-chain⟩` (maintained
     with two `TransferMatrix` applications per site) and the frozen old right environment. The
     interior site tensors of the augmented state are the isometries `[old │ Ũᵢ]`; the root tensor
     is `[C̄_L(t₁); C̃_L]` — the amplitude and all first-order content enter exactly once, at the
     root. A final `SvdCut` sweep truncates; `notrunc()` restores the pre-step virtual space of
     every bond (per-bond `truncspace`, fixed-rank parallel BUG).
* `test/algorithms/parallelbug.jl` (40 tests, all passing): 2-site dense exactness, energy +
  eigenstate-phase conservation, TDVP agreement, imaginary-time monotone lowering + norm,
  bond growth under tight/loose `trscheme`, LazySum, **convergence order ≥ 1** (measured slope
  ≈ 1.95–1.98 at these bond dimensions), **accuracy improves with ϑ** (vs a ϑ→0 run of the same
  integrator, isolating the `c·n·ϑ` term: 5e-4 → 2.5e-6 → 1e-15), U(1)/Z2 charge + graded-structure
  preservation.

## What closed the first-order gap (was the open problem)

Two changes relative to the first WIP driver:

1. **The coupling blocks participate in the orthonormalization** (the `M = Û'U₀` reconciliation of
   Alg. 4): the earlier driver placed the coupling in the `(new-row, old-col)` block of each
   interior site tensor, which routes it through the amplitude-carrying root block and multiplies
   it by the old bond matrix (σ-suppressed) — the interior first-order terms effectively vanished
   (measured slope ≈ 0). In the correct assembly the interior tensors are *pure isometries*
   `[old │ Ũᵢ]` and the coupling data enters `Ũᵢ`'s span via the stacked `Ĉ¹ᵢ = [C̄¹ᵢ; C̃ᵢ]`, so
   deep new directions propagate to the root, where the single coupling row
   `C̃_L = dt′·⟨Ũ¹_{L-1}|H|ψ₀⟩` captures every first-order component in one exact projection.
   (Rank counting is fine: the needed new subspace at bond `b` is the range of the *summed*
   tangent components, ≤ r_b-dimensional, not one r-dim family per site.)
2. **Amplitude-weighted kets**: the interior Galerkin solves and couplings act on `AC⁰[i]`
   (`= AL⁰[i]·C⁰ᵢ`), not the bare isometry `AL⁰[i]`. With bare-`AL` kets the spans miss the needed
   directions (H_eff and the bond matrix do not commute through the parent leg; measured slope
   stayed ≈ 0). The amplitude these objects carry is discarded with the R-factor in the
   orthonormalization, so no phase/energy overcounting occurs — this resolves the earlier
   "full-environment effective Hamiltonian trilemma" (`assembly A/B/C`): the phase only ever
   enters the state through the root blocks.

## Reconciliation with the reference (2026-07-10)

Compared against the authors' reference `github.com/JonasKu/Publication-Parallel-BUG-for-TTNs`
(MATLAB `Section5.1`, Julia `Section5.2/5.3`). The core assembly matches on every load-bearing
point (frozen-`t₀` independent solves; old-first `[U₀│Ũ]` QR augmentation; first-order coupling with
zeroed new–new corners; amplitude carried once at the root; SVD truncation). Changes landed this
session (all 41 tests green, incl. threaded 4-thread run):

* **Threading (done)**: phase-1's `L` solves swapped from `map` onto `tmap!` gated on
  `Defaults.scheduler[]` (mirrors `tdvp.jl`). Finite envs mutate on lazy recompute, so they are
  warmed serially (`_pbug_warmup_envs!`, dispatching through `MultipleEnvironments`) before the
  parallel region.
* **Time-dependent coupling (fixed)**: `_pbug_coupling_hamiltonian(...)` was applied with the
  one-arg form — a `MethodError` for a genuine `TimedOperator` (no one-arg apply). Now applied at the
  midpoint `t + dt/2` (the anchor `integrate` freezes coefficients at). New `TimedOperator` smoke test.
* **Error estimator (done)**: `η = ‖C̃‖/dt` accumulated in `_pbug_assemble` — the coupling blocks
  `C̃ᵢ` ARE the frozen derivative projected onto the new directions (Ceruti et al. 2024, eq. 6;
  cf. `eta_check.m`, whose `ttm(F_root, {Ũ-projectors})` is exactly this projection). Logged `@debug`.
* **Step rejection (partial, opt-in `maxiter_rejection > 0`)**: the **rank-saturation** trigger
  (`rejection_check.m`) — if a truncating step keeps the full doubled space on a bond, recompute as
  two half-steps (sub-stepping lowers the per-step normal component so one doubling suffices).
  Bounded recursion. Fields `c` (default 10.0) and `maxiter_rejection` (default 0) added.
* **Type stability / cleanups (done)**: `Vector{Any}`→concrete `As`/`Cevo`.

## Remaining work (why still experimental)

1. **η-threshold rejection trigger**: `η` is computed but only the rank-saturation trigger drives
   the retry. Wiring `dt·η > c·ϑ` needs a scheme-generic way to extract the tolerance `ϑ` from an
   arbitrary `trscheme` (trivial for `truncerror`, unclear in general). The current retry also uses
   sub-stepping rather than the reference's **basis-enrichment** recompute (re-embed `ψ₀` in the
   augmented 2r basis, re-assemble → 4r); enrichment keeps the same `dt` and is the true
   rank-adaptive mechanism, but needs the matrix-paper [4] retry semantics + numerical validation
   (the caterpillar has no dropped corner until `L ≥ 3`, so there is no cheap 2-site oracle).
2. **Second-order variant** (`TTN_integrator_parallel_2nd_order_nonglobal.m` + `RK_2`): the reference
   is **NOT** a Strang composition — it is an **augmented-Galerkin** scheme that *keeps* the new–new
   coupling corners (a `3r` core: old block, new–old coupling `C̄`, and the new–new block `Cᵢ`),
   solving the enlarged core ODE. For MPS this means retaining (not discarding via the R-factor) the
   new–new corner amplitude at interior nodes — effectively a parallel analogue of the sequential
   `BUG`'s second order. Substantial; recommend as a focused follow-up. Same subtle corner
   machinery as (1)'s enrichment, so validate them together against an `L ≥ 3` reference.
3. Minor: `timestep!`'s `envs` argument is still only used for `L == 1`; the frozen-snapshot
   environments are recomputed each step. Reusing the caller's `envs` is blocked by the frozen-copy
   design (`ψ₀ = copy(ψ)` ⇒ identity-keyed envs cannot match); evolving from `ψ` directly (it is not
   mutated before the final overwrite) would enable reuse but trades away the copy's isolation.
