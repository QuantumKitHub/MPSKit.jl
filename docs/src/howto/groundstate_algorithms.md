# [Ground-state algorithms](@id howto_groundstate_algorithms)

The examples on this page use MPSKit.jl, MPSKitModels.jl, TensorKit.jl, and TensorKitTensors.jl.
See [Installation](@ref tutorial_installation) for how to add these packages to your environment.

[`find_groundstate`](@ref) is the single entry point for optimizing an MPS towards the ground state of a Hamiltonian.
This page shows how to pick and configure the algorithm it runs, for both finite and infinite systems.
For what each algorithm actually does and why you would choose one over another, see [Ground-state algorithms](@ref lib_groundstate).
All examples share a single namespace:

```@example groundstate_algs
using MPSKit, MPSKitModels, TensorKit
using TensorKitTensors.SpinOperators: σˣ, σᶻ
```

---

## 1. Get a ground state with defaults

Called with just a state and a Hamiltonian, `find_groundstate` inspects the type of the initial state and picks a matching algorithm for you.
For a `FiniteMPS` it runs [`DMRG`](@ref) with the keywords you pass through (`tol`, `maxiter`, `verbosity`):

```@example groundstate_algs
L = 8
ψ₀ = FiniteMPS(L, ℂ^2, ℂ^8)
H = transverse_field_ising(FiniteChain(L); g = 0.5)

ψ, envs, ϵ = find_groundstate(ψ₀, H; tol = 1.0e-8, maxiter = 50, verbosity = 0)
ϵ
```

For an `InfiniteMPS` it instead runs [`VUMPS`](@ref), and if the requested `tol` is tighter than `1e-4` it chains a [`GradientGrassmann`](@ref) pass afterwards to polish the last few digits (see [§4](#4-refine-convergence-with-gradientgrassmann)):

```@example groundstate_algs
ψ₀_inf = InfiniteMPS(ℂ^2, ℂ^6)
H_inf = transverse_field_ising(; g = 0.5)

ψ_inf, envs_inf, ϵ_inf = find_groundstate(ψ₀_inf, H_inf; verbosity = 0)
ϵ_inf
```

Passing a `trunc` keyword switches on a two-site pre-pass that can grow the bond dimension before the single-site algorithm takes over.
On a `FiniteMPS` this prepends [`DMRG2`](@ref); on an `InfiniteMPS` it prepends [`IDMRG2`](@ref) (which needs a unit cell of at least two sites, see [§3](#3-configure-infinite-system-algorithms)).

```@example groundstate_algs
ψ_auto, envs_auto, ϵ_auto = find_groundstate(
    ψ₀, H;
    trunc = truncrank(16), verbosity = 0
)
ϵ_auto
```

!!! note "When to reach for an explicit algorithm"
    The keyword form above covers the common cases.
    Reach for an explicit algorithm struct (`DMRG`, `DMRG2`, `VUMPS`, `IDMRG`, `IDMRG2`, `GradientGrassmann`), or a chain of them with `&`, whenever you need finer control than the heuristic provides — the rest of this page shows how.

---

## 2. Configure finite-system DMRG

Pass a [`DMRG`](@ref) struct explicitly to set `tol`, `maxiter`, and `verbosity` directly:

```@example groundstate_algs
ψ_dmrg, envs_dmrg, ϵ_dmrg = find_groundstate(
    ψ₀, H,
    DMRG(; tol = 1.0e-8, maxiter = 50, verbosity = 0)
)
ϵ_dmrg
```

`DMRG` updates one site at a time, so with its defaults it cannot change the bond dimension: whatever bond dimension `ψ₀` starts with is what it keeps.
[`DMRG2`](@ref) optimizes two sites at once and truncates back down, which lets it grow (or shrink) the bond dimension as it sweeps, at extra cost per step.
Unlike `DMRG`, `DMRG2` has no default truncation scheme, so `trunc` is required:

```@example groundstate_algs
ψ_dmrg2, envs_dmrg2, ϵ_dmrg2 = find_groundstate(
    ψ₀, H,
    DMRG2(; trunc = truncrank(16), maxiter = 5, verbosity = 0)
)
ϵ_dmrg2
```

A common pattern is to warm up with `DMRG2` to grow the bond dimension, then refine with the cheaper single-site `DMRG`.
The `&` chaining operator runs the first algorithm to completion, then feeds its result into the second:

```@example groundstate_algs
warmup_then_refine = DMRG2(; trunc = truncrank(16), maxiter = 3, verbosity = 0) &
    DMRG(; tol = 1.0e-8, maxiter = 30, verbosity = 0)

ψ_c, envs_c, ϵ_c = find_groundstate(ψ₀, H, warmup_then_refine)
ϵ_c
```

For more on choosing `trunc` and growing bond dimension in general, see [Controlling bond dimension](@ref howto_bond_dimension).

---

## 3. Configure infinite-system algorithms

[`VUMPS`](@ref) is the default single-site algorithm for an `InfiniteMPS`, and takes the same `tol`/`maxiter`/`verbosity` keywords:

```@example groundstate_algs
ψ_v, envs_v, ϵ_v = find_groundstate(
    ψ₀_inf, H_inf,
    VUMPS(; tol = 1.0e-8, maxiter = 50, verbosity = 0)
)
ϵ_v
```

[`IDMRG`](@ref) is the infinite analogue of `DMRG`: it grows the system by repeatedly inserting sites in the middle and re-optimizing, until boundary effects wash out.

```@example groundstate_algs
ψ_i, envs_i, ϵ_i = find_groundstate(
    ψ₀_inf, H_inf,
    IDMRG(; tol = 1.0e-8, maxiter = 50, verbosity = 0)
)
ϵ_i
```

In practice, prefer `VUMPS` unless you specifically need `IDMRG`'s ability to change the bond dimension one site at a time.

[`IDMRG2`](@ref) is the two-site, bond-dimension-changing variant, and mirrors `DMRG2`: `trunc` is required, and it needs a unit cell of at least two sites.

!!! warning "Unit cell size"
    `IDMRG2` throws an `ArgumentError` on a single-site `InfiniteMPS`.
    Build the initial state and Hamiltonian with a unit cell of two (or more) sites instead.

```@example groundstate_algs
J = 1.0
g = 0.5
X = σˣ()
Z = σᶻ()

lattice_2 = PeriodicVector([ℂ^2, ℂ^2])
H_inf_2 = InfiniteMPOHamiltonian(
    lattice_2,
    (1, 2) => -J * X ⊗ X,
    (2, 3) => -J * X ⊗ X,
    (1,) => -g * Z,
    (2,) => -g * Z,
)
ψ₀_2 = InfiniteMPS([ℂ^2, ℂ^2], [ℂ^2, ℂ^2])

ψ_i2, envs_i2, ϵ_i2 = find_groundstate(
    ψ₀_2, H_inf_2,
    IDMRG2(; trunc = truncrank(16), maxiter = 5, verbosity = 0)
)
ϵ_i2
```

---

## 4. Refine convergence with GradientGrassmann

[`GradientGrassmann`](@ref) performs Riemannian gradient descent directly on the manifold of (finite or infinite) MPS, using an optimizer from OptimKit (`ConjugateGradient` by default via the `method` keyword).
Chain it after `VUMPS` (or `DMRG`) with `&` to combine both regimes in one call — this is exactly what `find_groundstate`'s heuristic does once `tol` is tighter than `1e-4`:

```@example groundstate_algs
refine = VUMPS(; tol = 1.0e-6, maxiter = 20, verbosity = 0) &
    GradientGrassmann(; tol = 1.0e-10, maxiter = 50, verbosity = 0)

ψ_g, envs_g, ϵ_g = find_groundstate(ψ₀_inf, H_inf, refine)
ϵ_g
```

Since `GradientGrassmann` is also a single-site algorithm, it cannot change the bond dimension either: grow it beforehand with `DMRG2`/`IDMRG2` or the `changebonds` recipes in [Controlling bond dimension](@ref howto_bond_dimension).

---

## 5. Control output and tolerances

Every algorithm accepts a `verbosity` keyword as a plain integer:

| `verbosity` | Output |
|:-----------:|:-------|
| `0` | nothing |
| `1` | warnings only |
| `2` | convergence information |
| `3` | per-iteration information (the default) |
| `4` | everything |

`find_groundstate` returns `(ψ, envs, ϵ)`.
`ϵ` is the final convergence-error measure (a Galerkin residual) of whichever algorithm ran last — it quantifies how well the sweeps converged, not the error in the energy itself.

The optional third positional argument to `find_groundstate` lets you reuse `envs` from a previous call instead of recomputing it, which is useful when tightening the tolerance on a state you already optimized:

```@example groundstate_algs
ψ_v2, envs_v2, ϵ_v2 = find_groundstate(
    ψ_v, H_inf,
    VUMPS(; tol = 1.0e-10, maxiter = 50, verbosity = 0),
    envs_v
)
ϵ_v2
```

---

## Where to go next

For growing, shrinking, and inspecting bond dimension during or between these calculations, see [Controlling bond dimension](@ref howto_bond_dimension).
For background on when each algorithm applies and how it relates to the others, see [Ground-state algorithms](@ref lib_groundstate).
To see `find_groundstate` used end to end on a finite chain, start from [Your first ground state](@ref tutorial_first_groundstate); for the infinite-system counterpart, see [The thermodynamic limit](@ref tutorial_thermodynamic_limit).
