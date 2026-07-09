# `ParallelBUG` implementation status

Companion to `PARALLELBUG_design.md`. Records what is implemented and what remains.
**The integrator works** (first-order gates pass); it remains marked experimental because the
local solves are still executed serially and step rejection is not implemented.

## What is implemented (branch `ld-parallelbug`)

* `ParallelBUG <: Algorithm` struct + kw-constructor + docstring, registered and exported
  (`src/algorithms/timestep/parallelbug.jl`).
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

## Remaining work (why still experimental)

1. **Threading**: phase 1's `L` local exponential solves all read from the frozen snapshot and are
   mutually independent — swap the `map` onto `@sync`/`Threads.@spawn` gated on
   `Defaults.scheduler[]` (mirror `tdvp.jl`'s `InfiniteMPS` branch). Phase 2 is a cheap sequential
   sweep (QRs + one derivative apply per site), as in the paper.
2. **Step rejection** (paper §3.3): `η = ‖Ũ₁'F₀Ṽ₁‖`, reject & recompute on the augmented bases if
   `h·η > c·ϑ` or the truncation saturates the doubling cap; needs fields `c` (default 10) and
   `maxiter_rejection`.
3. Minor: `timestep!`'s `envs` argument is currently only used for `L == 1`; the frozen snapshot
   environments are recomputed each step (could be reused when the caller's `envs` are current).
