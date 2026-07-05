# [Time evolution](@id howto_time_evolution)

The examples on this page use MPSKit.jl, MPSKitModels.jl, TensorKit.jl, and TensorKitTensors.jl.
See [Installation](@ref tutorial_installation) for how to add these packages to your environment.

MPSKit solves the (real- or imaginary-time) Schrödinger equation `i ∂ψ/∂t = H ψ` in two ways: by projecting the equation onto the MPS tangent space at every step ([`TDVP`](@ref)/[`TDVP2`](@ref)), or by first building an approximate evolution operator as an MPO ([`make_time_mpo`](@ref)) and repeatedly applying it.
This page gives task recipes for both routes.
All examples share a single namespace:

```@example time_evo
using MPSKit, MPSKitModels, TensorKit
using TensorKitTensors.SpinOperators: σˣ, σᶻ
```

---

## 1. Evolve a state through one time step

[`timestep`](@ref) advances a state by a single `dt` under a Hamiltonian, using whichever [`TDVP`](@ref)-family algorithm you pass.
Its argument order is state, Hamiltonian, current time, time step, algorithm:

```@example time_evo
L = 8
g₀ = 0.5
H₀ = transverse_field_ising(FiniteChain(L); g = g₀)

ψ₀ = FiniteMPS(L, ℂ^2, ℂ^8)
ψ₀, = find_groundstate(ψ₀, H₀, DMRG(; verbosity = 0))
expectation_value(ψ₀, 4 => σᶻ())
```

Now quench to a different transverse field `g₁` and take a single step:

```@example time_evo
g₁ = 2.0
H₁ = transverse_field_ising(FiniteChain(L); g = g₁)

dt = 0.05
ψ₁, envs₁ = timestep(ψ₀, H₁, 0.0, dt, TDVP())
expectation_value(ψ₁, 4 => σᶻ())
```

`timestep` returns the updated state together with an `envs` cache; reuse `envs₁` in the next call to avoid recomputation.
An in-place `timestep!` also exists, but only for finite MPS.

---

## 2. Evolve over a time span

[`time_evolve`](@ref) steps through an explicit vector of time points instead of a single `dt`, carrying the algorithm and environments through the whole span:

```@example time_evo
t_span = 0:dt:(4dt)
ψ_span, envs_span = time_evolve(ψ₀, H₁, t_span, TDVP(); verbosity = 0)
expectation_value(ψ_span, 4 => σᶻ())
```

`t_span` need not be uniformly spaced — `time_evolve` steps pairwise between consecutive entries, so any `AbstractVector` of increasing times works.
There is no exported `time_evolve!`; use `time_evolve` and rebind the result.

---

## 3. Grow the bond dimension while evolving

Single-site `TDVP` cannot change the bond dimension: whatever `ψ₀` starts with is what it keeps.
After a quench, entanglement typically grows and a fixed bond dimension eventually becomes insufficient.
<!-- REVIEW: confirm that entanglement growth after a quench is generic enough to state as a blanket recipe motivation, versus being model/quench-dependent. -->
[`TDVP2`](@ref) updates two sites at a time and truncates back down, so it can grow (or shrink) the bond dimension as it evolves.
Unlike `TDVP`, `TDVP2` requires `trscheme` — there is no default:

```@example time_evo
ψ_tdvp2, envs_tdvp2 = timestep(ψ₀, H₁, 0.0, dt, TDVP2(; trscheme = truncrank(16)))
dim(left_virtualspace(ψ_tdvp2, 4))
```

`TDVP2` only has a finite-MPS method.
For finite systems, an alternative to switching algorithms entirely is single-site `TDVP` with `alg_expand` set to a bond-expansion algorithm such as [`OptimalExpand`](@ref) (Controlled Bond Expansion, "CBE-TDVP"); see [Controlling bond dimension](@ref howto_bond_dimension) for `OptimalExpand` and other `changebonds` recipes.

---

## 4. Evolve an infinite state

Single-site `TDVP` also works directly on an `InfiniteMPS`:

```@example time_evo
ψ₀_inf = InfiniteMPS(ℂ^2, ℂ^8)
H₀_inf = transverse_field_ising(; g = g₀)
ψ₀_inf, = find_groundstate(ψ₀_inf, H₀_inf, VUMPS(; verbosity = 0))

H₁_inf = transverse_field_ising(; g = g₁)
ψ₁_inf, envs₁_inf = timestep(ψ₀_inf, H₁_inf, 0.0, dt, TDVP())
expectation_value(ψ₁_inf, 1 => σᶻ())
```

`TDVP2` has no `InfiniteMPS` method, and single-site `TDVP` cannot change the bond dimension on an infinite state either.
Grow the bond dimension beforehand with `changebonds` (e.g. `OptimalExpand` or `VUMPSSvdCut`) — see [Controlling bond dimension](@ref howto_bond_dimension) — then evolve at the fixed, larger bond dimension.

---

## 5. Imaginary-time evolution

Passing `imaginary_evolution = true` evolves under `exp(-H dt)` instead of `exp(-iH dt)`, using the same real `dt`:

```@example time_evo
ψ_im, envs_im = timestep(ψ₀, H₁, 0.0, dt, TDVP(); imaginary_evolution = true)
norm(ψ_im)
```

Imaginary-time evolution renormalizes the state at every step, so `norm(ψ_im)` stays `1` regardless of how the un-normalized weight would otherwise change.
Repeated imaginary-time steps damp excited-state components faster than the ground state, so this is often used as a (slower) alternative to `find_groundstate` for driving a state towards the ground state of `H₁`.
<!-- REVIEW: confirm this "cooling towards the ground state" framing and its relative merits/use cases versus find_groundstate — beyond the norm-renormalization behavior, which is docstring-backed, the rest is my inference. -->
`time_evolve` accepts the same `imaginary_evolution` keyword for a span of imaginary-time steps.

---

## 6. Build a time-evolution MPO

When `H` is time-independent and `dt` is fixed, an alternative to repeated `timestep` calls is to build the evolution operator once as an MPO with [`make_time_mpo`](@ref), then apply it repeatedly with [`approximate`](@ref).
<!-- REVIEW: confirm when this MPO route is actually preferable to TDVP in practice (e.g. amortizing the one-time MPO construction cost over many reuses at fixed dt, versus TDVP's per-step tangent-space projection cost) — flagging since I cannot benchmark this. -->

[`WII`](@ref) builds a low-order MPO approximation and works for both finite and infinite Hamiltonians:

```@example time_evo
O = make_time_mpo(H₁, dt, WII())
```

[`TaylorCluster`](@ref) gives higher-order control via its `N` keyword (and accepts an additional `tol` keyword in `make_time_mpo`):

```@example time_evo
O_taylor = make_time_mpo(H₁, dt, TaylorCluster(; N = 2); tol = 1.0e-10)
```

[`WI`](@ref) is a ready-made first-order `TaylorCluster` constant, so it is used as a value, not called as a constructor:

```@example time_evo
O_wi = make_time_mpo(H₁, dt, WI)
```

Applying the MPO to a finite state uses [`approximate`](@ref) with a finite ground-state-style algorithm such as [`DMRG2`](@ref), which returns a 3-tuple including the final convergence error:

```@example time_evo
ψ_mpo, envs_mpo, ϵ_mpo = approximate(
    ψ₀, (O, ψ₀), DMRG2(; trscheme = truncrank(16), verbosity = 0)
)
expectation_value(ψ_mpo, 4 => σᶻ())
```

Repeat the `approximate` call with the same `O` for successive time steps to build up a longer evolution.
Imaginary-time MPOs are built the same way, by passing `imaginary_evolution = true` to `make_time_mpo`; a real `dt` is promoted internally, so no manual complex conversion is needed.

---

## Where to go next

For choosing and configuring ground-state algorithms to prepare the pre-quench state, see [Ground-state algorithms](@ref howto_groundstate_algorithms).
For growing or shrinking bond dimension between or during evolution steps, see [Controlling bond dimension](@ref howto_bond_dimension).
For extracting expectation values and correlators from the evolved state, see [Computing observables](@ref howto_observables).
For background on the TDVP and time-evolution-MPO approaches and how they relate, see [Time evolution](@ref lib_time_evolution).
