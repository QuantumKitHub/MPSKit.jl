# `ParallelBUG` implementation status

Companion to `PARALLELBUG_design.md`. Records what is implemented, what works, the one open
problem, and the recommended path to close it. **The integrator is experimental (WIP).**

## What is implemented (branch `ld-parallelbug`)

* `ParallelBUG <: Algorithm` struct + kw-constructor + docstring, registered and exported
  (`src/algorithms/timestep/parallelbug.jl`).
* A `timestep!` + copying `timestep` implementing the **frozen-`t₀` assembly** (the paper's
  Alg. 1–4 specialized to the caterpillar): one frozen snapshot `ψ₀ = copy(ψ)` + `envs₀`; the root
  center `AC[L]` and the interior isometries `AL[i]` are each evolved forward from that snapshot;
  bonds are augmented old-first with the new directions the interior evolutions discover, coupled by
  first-order blocks with zeroed multi-new corners; then an `SvdCut` sweep truncates.
* `test/algorithms/parallelbug.jl` covering the behaviours below.

## What works (asserted in the tests)

* **2-site: exact.** Reproduces the dense `exp(-iH·dt)` step to ~1e-12 (the matrix parallel-BUG
  formulas are exact here — this locks the block conventions / adjoints).
* **Energy + eigenstate phase: exact.** `angle⟨ψ₁|ψ₀⟩/(dt·E₀) = 1` for `L = 2,3,4` (amplitude is
  carried once, at the root), `|E₁−E₀| ~ 1e-15`.
* **Rank adaptivity: bonds grow.** A low-bond-dim start under a tight `truncerror` grows the bond
  (e.g. 2 → 8) via the augment-then-`SvdCut` mechanism.
* Imaginary-time renormalization and symmetric-tensor (U(1)) total-charge preservation — see the
  test file for the exact assertions that hold.

## The one open problem (marked `@test_broken`)

For `L > 2` the integrator does **not** yet attain the documented **first-order** accuracy: the
error does not decrease cleanly `∝ dt`, and tightening `ϑ` does not monotonically improve the
overlap with the dense reference. Root cause and analysis:

MPSKit's `AC_hamiltonian`/`C_hamiltonian` are **full-environment** effective Hamiltonians (each
carries the full energy `E₀`). This forces a trilemma when mapping the caterpillar (verified
numerically across three assemblies):

| assembly | interior "old" block | phase ratio | order |
|---|---|---|---|
| A: keep evolved connecting tensor as a block | evolved `C̄¹_i` (amplitude kept) | ≈ `L` (overcounts) | — |
| B: freeze the basis, couple new dirs | frozen `AL[i]` (phase stripped) | **1** ✓ | ~0 (shipped) |
| C: symmetric `AC`+`C` regauge | evolved `AC`+`C` daggered | **1** ✓, exact @ L=2 | 2 (parallel-TDVP, not BUG) |

The shipped driver is **assembly B**: it strips the interior phase correctly (ratio 1) and grows
bonds, but the first-order dynamics are not fully captured because the coupling-block
**reconciliation** is incomplete. Specifically, the paper's Alg. 4 builds the augmented basis from
the range of *both* the reprojected old connecting tensor `Ĉ⁰ = C⁰ ×_child Û'U⁰` **and** the
Galerkin-evolved `Ĉ¹`, and threads the overlap `M = Û'U₀` up toward the root. The shipped assembly
places the coupling directly in the new-child rows without that `M`-reconciliation, so the retained
subspace after `SvdCut` is not rotated by the correct `O(dt)` amount — hence the loss of clean first
order. (The `AC2_hamiltonian`-based coupling used for `C̃_i` is also a first suspect; the correct
one-site term is `dt·(AC_hamiltonian(i)·AC[i])` projected onto the new directions.)

## Recommended path to close it

1. Implement the leaves→root **`M = Û'U₀` reconciliation** (design doc §2b / paper Alg. 4A.2–A.4):
   at each interior node reproject the *old* connecting tensor onto the already-augmented child
   basis and build the parent-bond basis from the range of `[Mat₀(Ĉ⁰); Mat₀(Ĉ¹)]`, threading the
   overlap upward — rather than the direct block placement used now.
2. Replace the two-site `AC2_hamiltonian` coupling with the one-site `dt·(AC_hamiltonian(i)·AC[i])`
   projected onto `Ũ₁` (matrix eq. 3.1a analog).
3. Re-run `research/`'s gate harness (2-site exact / phase ratio / **slope ≈ 1** / **accuracy
   improves with ϑ** / bond growth / symmetric preservation). The scaffolding and tests are ready;
   flip the `@test_broken`s to `@test` once slope ≈ 1 holds.
4. Only then add step rejection (`η = ‖Ũ₁*F₀Ṽ₁‖`, `hη>cϑ`) and the `@sync`/`tmap!` threading (the
   K-steps already read from one frozen snapshot, so threading is a localized change).

Nothing here requires new MPSKit primitives — the earlier "need partial-energy node operators"
hypothesis is not necessary: the tree algorithm handles the full-energy operators by
*orthonormalizing* the interior evolved tensors (stripping their phase), which assembly B already
does; the remaining work is the coupling reconciliation, not the effective operators.
