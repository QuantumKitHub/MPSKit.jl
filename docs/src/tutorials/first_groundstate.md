# [Your first ground state](@id tutorial_first_groundstate)

This tutorial walks you through a complete MPSKit.jl calculation from start to finish: we build the transverse-field Ising model, find its ground state with DMRG, measure a few physical quantities, and finish with a plot of the magnetization across the model's phase transition.
It assumes only that you are comfortable with basic quantum mechanics and that you have finished [Installation](@ref tutorial_installation), so the packages used below are already available in your environment.

The transverse-field Ising model (TFIM) is the "hello world" of quantum many-body physics: it is the simplest model that still shows a genuine quantum phase transition, so it is the natural place to learn the tools.
On a chain of ``L`` spin-1/2 sites it is

```math
H = -J\left(\sum_{\langle i,j\rangle} \sigma^z_i\,\sigma^z_j + g\sum_i \sigma^x_i\right),
```

where the first sum runs over neighbouring pairs.
The coupling ``J`` sets the overall energy scale, and the dimensionless field ``g`` tunes the competition between the ferromagnetic ``\sigma^z\sigma^z`` interaction and the transverse ``\sigma^x`` field.

The ground state of ``H`` lives in a Hilbert space of dimension ``2^L``, which is far too large to store as a plain vector for any interesting ``L``.
A *matrix product state* (MPS) sidesteps this by storing the state as a chain of small tensors, one per site, whose sizes we control directly; this is what makes the calculation below tractable.
The details of that compression are the subject of the concept pages — here we simply use it.

<!-- REVIEW: physics framing — is it accurate to call the sigma^z sigma^z term "ferromagnetic" given the -J sign convention, and to describe g as tuning interaction-vs-field competition? -->

## Loading the packages

Every code block on this page shares one Julia session, so we only need to load packages once.
We take the model and lattice from MPSKitModels, the local spin operators from TensorKitTensors, and `Plots` for the final figure.

```@example first-groundstate
using MPSKit, MPSKitModels, TensorKit
using TensorKitTensors.SpinOperators: σˣ, σᶻ
using Plots
```

## 1. Build the Hamiltonian

We work with a chain of `L = 16` sites and fix the field to `g = 0.5` for now.
`transverse_field_ising` assembles the Hamiltonian above; passing `FiniteChain(L)` asks for a finite open chain of `L` sites.

```@example first-groundstate
L = 16
H = transverse_field_ising(FiniteChain(L); g = 0.5)
```

The returned object is an `MPOHamiltonian`: the Hamiltonian written in matrix-product-operator form, i.e. as a chain of small tensors just like the state it acts on.
You do not need to know its internals to use it — MPSKit's algorithms consume it directly.
For other ways to build Hamiltonians see [Building Hamiltonians](@ref howto_hamiltonians).

<!-- REVIEW: at g = 0.5 the finite chain is on the ordered (ferromagnetic) side of the transition (critical point g = 1 in the thermodynamic limit). Please confirm this characterization. -->

## 2. Build the initial state

DMRG is an optimization: it needs a starting state to improve.
We create a random `FiniteMPS` with the right structure.

```@example first-groundstate
D = 4
ψ₀ = FiniteMPS(L, ℂ^2, ℂ^D)
```

The two space arguments describe the two kinds of index every MPS tensor carries:

- `ℂ^2` is the **physical space** — the local Hilbert space of a single spin-1/2 site, which has dimension 2.
- `ℂ^D` is the **virtual (bond) space** — the internal index linking neighbouring tensors, whose dimension `D` is the *bond dimension*.

The bond dimension `D` is the accuracy knob of the whole method: a larger `D` lets the MPS capture more entanglement and represent the true ground state more faithfully, at the cost of more computation.
`D = 4` is deliberately small so this tutorial runs quickly; [Controlling bond dimension](@ref howto_bond_dimension) covers how to choose and grow it.

!!! warning "Pass spaces, not integers"
    The physical and virtual arguments must be *vector spaces* (`ℂ^2`, `ℂ^D`, or equivalently `ComplexSpace(2)`), never bare integers.
    Writing `FiniteMPS(16, 2, 4)` throws a `MethodError` — this is the single most common beginner mistake.

## 3. Find the ground state

Now we run the calculation.
`find_groundstate` takes the starting state, the Hamiltonian, and an algorithm; we pass [`DMRG`](@ref) explicitly so the algorithm is visible.

```@example first-groundstate
ψ, envs, ϵ = find_groundstate(ψ₀, H, DMRG())
```

DMRG (the density-matrix renormalization group) sweeps back and forth along the chain, locally optimizing each tensor while holding the others fixed, and repeats until the state stops changing.
The lines printed above are the per-iteration convergence log (shown at the default `verbosity`); each reports how much the state improved on that sweep.

<!-- REVIEW: is it correct to describe the printed iteration log as "how much the state improved on that sweep", and to say DMRG optimizes one tensor locally while holding the others fixed? -->

!!! note "The algorithm is optional"
    Calling `find_groundstate(ψ₀, H)` with no algorithm argument selects DMRG automatically for a finite input, so the explicit `DMRG()` above is only for clarity.
    `DMRG` accepts keywords such as `tol` (default `1e-10`), `maxiter` (default `200`), and `verbosity` (default `3`); we use `verbosity = 0` later to silence the log inside a loop.

`find_groundstate` returns a triple:

- `ψ` — the optimized ground-state MPS (a *new* state; `ψ₀` is left untouched, so we can reuse it below). A mutating variant `find_groundstate!` also exists.
- `envs` — the *environments*, cached partial contractions that later measurements can reuse to save work.
- `ϵ` — a convergence-error measure (the Galerkin residual). It quantifies how well the sweeps converged; note that it is **not** the error in the energy.

## 4. Measure observables

With a ground state in hand we can extract physical quantities.
The energy is the expectation value of the Hamiltonian itself — pass `H` directly, with no site index:

```@example first-groundstate
E = expectation_value(ψ, H)
```

For a Hermitian `H` and a normalized state this is real up to floating-point noise.

The order parameter of the TFIM is the local magnetization ``\langle\sigma^z_i\rangle``.
We measure it at every site by pairing each site index with the single-site operator `σᶻ()`:

```@example first-groundstate
[expectation_value(ψ, i => σᶻ()) for i in 1:L]
```

Finally, a good "how converged am I really?" check is the energy variance ``\langle H^2\rangle - \langle H\rangle^2``, which vanishes exactly when `ψ` is a true eigenstate:

```@example first-groundstate
variance(ψ, H)
```

A small variance indicates the state is close to an eigenstate of `H`.
More recipes for observables live in [Computing observables](@ref howto_observables).

<!-- REVIEW: please confirm langle sigma^z_i rangle is the appropriate order parameter for this ZZ-coupling / X-field convention, and that "variance small ⇒ close to eigenstate" is stated correctly. -->

## 5. Magnetization across the transition

The payoff: we sweep the field `g` from 0 to 2 and, for each value, find the ground state and record its average magnetization.
This traces out the phase transition.

For this sweep we place the chain on a ring rather than an open segment: `transverse_field_ising()` with no lattice argument builds the Hamiltonian for an infinite chain, and [`periodic_boundary_conditions`](@ref) wraps it onto a finite ring of `L` sites.

<!-- REVIEW: the README uses periodic boundary conditions for this magnetization sweep (Sections 1–4 above used the open FiniteChain). Is periodic preferred here to reduce boundary/finite-size effects and give a sharper-looking transition, or is another reason more accurate? Please confirm or correct. -->

```@example first-groundstate
g_values = 0:0.1:2
M = map(g_values) do g
    Hg = periodic_boundary_conditions(transverse_field_ising(; g = g), L)
    ψg, = find_groundstate(ψ₀, Hg; verbosity = 0)
    return abs(sum(expectation_value(ψg, i => σᶻ()) for i in 1:L)) / L
end
scatter(g_values, M; xlabel = "g", ylabel = "M", label = "D = $D", title = "TFIM magnetization")
```

Here we take the **absolute value** of the mean magnetization.
A finite chain does not spontaneously break its symmetry, so the raw ``\sum_i\langle\sigma^z_i\rangle`` can land on either sign (or average toward zero); taking `abs` collapses the two symmetry-related ground states onto a single, well-defined order-parameter curve.

<!-- REVIEW: this is the crucial physics point. Please verify the justification for abs(...): spontaneous symmetry breaking does not literally occur at finite L, so the sign is not fixed; is "abs collapses the two symmetry-related sectors" the right way to explain it here? -->

The plot should show the magnetization large on the ordered side (small `g`) and dropping toward zero on the disordered side (large `g`), with the crossover near `g = 1`.

<!-- REVIEW: at finite L = 16 and small D = 4 the crossover is a smooth crossover, not a sharp transition, and its apparent location may be shifted from the thermodynamic critical point g = 1. Please confirm the expected qualitative shape and that we should not claim a sharp transition or an exact critical point at this size/bond dimension. -->

## Where to go next

You have run a full MPSKit workflow: build a model, optimize an MPS ground state, measure observables, and scan a parameter.
A natural next step is to perform the same calculation directly in the thermodynamic limit, working with an `InfiniteMPS` instead of a finite chain, which removes the finite-size effects seen above and lets you locate the critical point more cleanly.

<!-- TODO(link): thermodynamic_limit -->

To go deeper on the individual steps, see [Constructing states](@ref howto_states), [Building Hamiltonians](@ref howto_hamiltonians), [Computing observables](@ref howto_observables), [Controlling bond dimension](@ref howto_bond_dimension), and [Entanglement entropy and spectrum](@ref howto_entanglement); the algorithm reference is [Ground-state algorithms](@ref lib_groundstate).
