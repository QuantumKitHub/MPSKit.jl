# [Troubleshooting convergence](@id howto_convergence_troubleshooting)

```@meta
DocTestSetup = quote
    using MPSKit, MPSKitModels, TensorKit
end
```

When [`find_groundstate`](@ref), [`leading_boundary`](@ref), or a time-evolution call does
not converge, the fix is almost always one of a handful of causes: too few iterations, a
bond dimension that is too small, a bad initial state, or a mismatch between the ansatz and
the physics (wrong unit cell, wrong symmetry sector).
This page is a diagnostic checklist: each section is a symptom, the diagnostic that
confirms it, and the concrete API knob that fixes it.
It assumes you already know how to run the algorithms — see
[Ground-state algorithms](@ref howto_groundstate_algorithms) and
[Controlling bond dimension](@ref howto_bond_dimension) for that.

All examples share a single namespace:

```@example conv
using MPSKit, MPSKitModels, TensorKit
```

---

## 1. Read the convergence report

Every optimizer returns three things, `(ψ, envs, ϵ)`, and `ϵ` is your primary diagnostic:

```@example conv
L = 16
H = transverse_field_ising(FiniteChain(L); g = 1.0)
ψ₀ = FiniteMPS(L, ℂ^2, ℂ^16)

ψ, envs, ϵ = find_groundstate(ψ₀, H, DMRG(; tol = 1.0e-10, maxiter = 30, verbosity = 0))
ϵ
```

`ϵ` is the convergence-error measure of whichever algorithm ran last, compared against its
`tol` each sweep.
The algorithm has converged when `ϵ < tol`; if it stops because it hit `maxiter` first, `ϵ`
tells you how far it still had to go.
What `ϵ` actually measures differs by algorithm:

| Algorithm | What `ϵ` measures |
|:----------|:------------------|
| [`DMRG`](@ref), [`VUMPS`](@ref) | Galerkin residual (norm of the projected gradient) |
| [`DMRG2`](@ref) | `1 - abs(overlap)` of the two-site tensor across the truncation |
| [`IDMRG`](@ref), [`IDMRG2`](@ref) | change in the bond matrix `C` between iterations |
| [`GradientGrassmann`](@ref) | norm of the Riemannian gradient |

Because the measures differ, a raw `ϵ` value is only meaningful *within* one algorithm; do
not compare `ϵ` from a `DMRG2` warm-up against `ϵ` from the following `DMRG` refinement.

To watch convergence as it happens rather than after the fact, raise `verbosity`.
The levels are shared by every algorithm and documented in
[Ground-state algorithms](@ref howto_groundstate_algorithms); the named constants
`MPSKit.VERBOSE_NONE` (`0`) through `MPSKit.VERBOSE_ALL` (`4`) are public but not exported.
`verbosity = 2` prints one convergence line per sweep, which is usually enough to see
whether `ϵ` is dropping, plateauing, or oscillating.

---

## 2. Converged but wrong: check the variance

A small `ϵ` does **not** by itself guarantee a good ground state.
At a fixed bond dimension the optimizer converges to the best MPS *within that manifold*, and
its convergence measure can drop below `tol` while the state is still far from the true
ground state.
The independent check is the energy variance
``\langle H^2 \rangle - \langle H \rangle^2``, which is zero only for an exact eigenstate:

```@example conv
ψ_small, _, ϵ_small = find_groundstate(
    FiniteMPS(L, ℂ^2, ℂ^2), H, DMRG(; tol = 1.0e-10, maxiter = 30, verbosity = 0)
)
(ϵ_small, variance(ψ_small, H))
```

Here `ϵ` sits comfortably below `tol` — the run reports success — yet the variance is large:
a bond dimension of 2 simply cannot represent this (critical) ground state.
Grow the bond dimension and the variance collapses:

```@example conv
ψ_big, _, ϵ_big = find_groundstate(
    FiniteMPS(L, ℂ^2, ℂ^32), H, DMRG(; tol = 1.0e-10, maxiter = 30, verbosity = 0)
)
(ϵ_big, variance(ψ_big, H))
```

!!! tip "Variance is your ground-truth check"
    Whenever a result looks suspicious despite a small `ϵ`, compute [`variance`](@ref).
    A variance that will not drop as you add bond dimension points at an undersized ansatz,
    not at an unconverged optimization.

The fix is to grow the bond dimension: use [`DMRG2`](@ref)/[`IDMRG2`](@ref), or expand
explicitly with [`changebonds`](@ref) and [`OptimalExpand`](@ref)/[`RandExpand`](@ref).
See [Controlling bond dimension](@ref howto_bond_dimension) for the full set of recipes.

---

## 3. `ϵ` is still dropping, or stalls just above `tol`

If the algorithm stopped on `maxiter` with `ϵ` still decreasing, it simply needs more
iterations — raise `maxiter`:

```@example conv
ψ_more, _, ϵ_more = find_groundstate(
    ψ₀, H, DMRG(; tol = 1.0e-10, maxiter = 100, verbosity = 0)
)
ϵ_more
```

You can also resume from an already-optimized state instead of restarting, passing the
previous environments as the optional fourth argument so they are reused rather than
recomputed:

```@example conv
ψ_resume, _, ϵ_resume = find_groundstate(
    ψ, H, DMRG(; tol = 1.0e-12, maxiter = 50, verbosity = 0), envs
)
ϵ_resume
```

If instead `ϵ` *plateaus* well above `tol` and more iterations do not help, the ansatz is
the bottleneck, not the iteration count: check the variance and grow the bond dimension as in
[§2](#2-converged-but-wrong-check-the-variance), or treat it as a local minimum
([§4](#4-stuck-in-a-local-minimum)).

!!! note "Adaptive sub-tolerances"
    By default MPSKit tightens the tolerances of the inner eigensolver, gauge, and
    environment solvers automatically as the outer error `ϵ` shrinks (the `DynamicTol`
    mechanism).
    You therefore rarely need to touch `alg_eigsolve`/`alg_gauge`/`alg_environments` by
    hand; set the outer `tol` and let the inner solvers follow.

---

## 4. Stuck in a local minimum

Symptom: `ϵ` plateaus above `tol`, adding bond dimension does not help, and the variance
stays stubbornly high.
Variational optimizers can get trapped in local minima, especially from an unlucky random
start or a symmetry-frustrated initial state.

Things to try, roughly in order:

- **Restart from a different initial state.** `FiniteMPS`/`InfiniteMPS` with a size argument
  produce a *random* state, so simply rebuilding the initial guess reseeds the search:

  ```@example conv
  ψ_restart, _, ϵ_restart = find_groundstate(
      FiniteMPS(L, ℂ^2, ℂ^16), H, DMRG(; tol = 1.0e-10, maxiter = 100, verbosity = 0)
  )
  ϵ_restart
  ```

- **Mix algorithms.** Different optimizers have different failure modes, so chaining them
  with `&` often escapes a minimum that traps one of them.
  A robust infinite-system default is a [`VUMPS`](@ref) pass to get close, polished by
  [`GradientGrassmann`](@ref) — exactly what `find_groundstate` does automatically once
  `tol` is tighter than `1e-4`:

  ```@example conv
  H_inf = transverse_field_ising(; g = 0.5)
  ψ_inf, _, ϵ_inf = find_groundstate(
      InfiniteMPS(ℂ^2, ℂ^6), H_inf,
      VUMPS(; tol = 1.0e-8, maxiter = 50, verbosity = 0) &
          GradientGrassmann(; tol = 1.0e-10, maxiter = 50, verbosity = 0)
  )
  ϵ_inf
  ```

- **Inject noise, then re-optimize.** Padding the state with orthogonal random directions via
  [`RandExpand`](@ref) perturbs it off the current (possibly stuck) point without changing
  what it represents, giving the next optimization new directions to explore:

  ```@example conv
  ψ_noisy = changebonds(ψ_small, RandExpand(; trscheme = truncrank(8)))
  ψ_kick, _, ϵ_kick = find_groundstate(
      ψ_noisy, H, DMRG(; tol = 1.0e-10, maxiter = 100, verbosity = 0)
  )
  (ϵ_kick, variance(ψ_kick, H))
  ```

  A two-site algorithm ([`DMRG2`](@ref)) or the Hamiltonian-aware
  [`OptimalExpand`](@ref)/CBE variants achieve a similar effect while also improving the
  energy, and are usually the better first choice — see
  [Controlling bond dimension](@ref howto_bond_dimension).

---

## 5. Infinite MPS won't converge — look at the transfer matrix

A distinctive infinite-system failure is a state whose transfer matrix has several
eigenvalues crowding the unit circle.
This signals that the state is close to *non-injective* — a superposition of several
injective states — which is numerically ill-conditioned and usually means the unit cell is
too small for the order you are trying to represent.

Diagnose it with [`transfer_spectrum`](@ref) (the leading transfer-matrix eigenvalues) or its
distilled form [`correlation_length`](@ref):

```@example conv
ψ_gs, _, _ = find_groundstate(
    InfiniteMPS(ℂ^2, ℂ^16), transverse_field_ising(; g = 2.0),
    VUMPS(; tol = 1.0e-10, maxiter = 100, verbosity = 0)
)
correlation_length(ψ_gs)
```

```@example conv
abs.(transfer_spectrum(ψ_gs; num_vals = 5))
```

The leading eigenvalue is `1` (a normalized state); the *gap* between it and the next
eigenvalue sets the correlation length.
When several eigenvalues sit almost at `1`, the correlation length diverges and the state is
near-degenerate.

!!! tip "The fix is usually a larger unit cell"
    If the transfer spectrum is near-degenerate, rebuild the initial state and the
    Hamiltonian on a larger unit cell (e.g. `InfiniteMPS(fill(ℂ^2, 2), fill(ℂ^16, 2))` and a
    matching two-site `InfiniteMPOHamiltonian`) and re-optimize.
    The gallery example [The XXZ model](@ref "The XXZ model") walks through exactly this
    diagnostic — a VUMPS run that refuses to converge, a `transferplot` revealing
    near-degeneracy, and the fix of moving to a two-site unit cell.

For extracting a physically meaningful correlation length from the finite-bond-dimension
spectrum, [`marek_gap`](@ref) implements the standard finite-entanglement-scaling gap
extrapolation.

---

## 6. Wrong symmetry sector

With a symmetric Hamiltonian, an MPS is confined to the symmetry sector fixed by its virtual
spaces at construction time.
The optimizer never leaves that sector, so if you build the initial state in the wrong one
you converge to the lowest state *of that sector*, not the global ground state.

Inspect the sector content of a state through its virtual spaces:

```@example conv
using TensorKit: sectors
L2 = 12
H_z2 = transverse_field_ising(Z2Irrep, FiniteChain(L2); g = 0.5)
ψ_z2 = FiniteMPS(L2, Z2Space(0 => 1, 1 => 1), Z2Space(0 => 8, 1 => 8))
collect(sectors(left_virtualspace(ψ_z2, 1)))
```

If the sectors present are not the ones the physics requires, rebuild the initial state with
the intended sectors in its physical and virtual spaces (see
[Using symmetries](@ref tutorial_using_symmetries) and
[Constructing states](@ref howto_states) for the `sector => dimension` syntax) before
optimizing.

---

## 7. `leading_boundary` and time evolution

[`leading_boundary`](@ref) reuses the same infinite-MPS optimizers ([`VUMPS`](@ref),
[`VOMPS`](@ref), [`IDMRG`](@ref), and friends) and returns the same `(ψ, envs, ϵ)` triple, so
every recipe above applies: read `ϵ`, raise `maxiter`/`tol`, grow the bond dimension, and
watch the transfer-matrix spectrum of the boundary MPS.

For time evolution, "non-convergence" instead means accumulated error over the trajectory.
The knobs are different:

- **Time step.** A smaller `dt` reduces the per-step integration and (for
  [`make_time_mpo`](@ref) with [`WII`](@ref)/[`TaylorCluster`](@ref)) Trotter-type error.
- **Bond dimension.** Real-time evolution grows entanglement, so a fixed bond dimension
  eventually cannot follow the state.
  Use the two-site [`TDVP2`](@ref) (which takes a `trscheme`) or a CBE-enabled
  [`TDVP`](@ref) to let the bond dimension grow during evolution.

See [Time evolution](@ref howto_time_evolution) for the full time-evolution interface.

---

## Where to go next

For growing and inspecting bond dimension, the most common single fix, see
[Controlling bond dimension](@ref howto_bond_dimension).
For choosing and chaining the ground-state algorithms referenced here, see
[Ground-state algorithms](@ref howto_groundstate_algorithms).
For the reasoning behind when each algorithm applies and how they compare, see
[The algorithm landscape](@ref concept_algorithm_landscape).
