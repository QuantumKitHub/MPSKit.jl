```@meta
DocTestSetup = :(using MPSKit, TensorKit, MPSKitModels)
```

# [Algorithms](@id um_algorithms)

Here is a collection of the algorithms that have been added to MPSKit.jl.
If a particular algorithm is missing, feel free to let us know via an issue, or contribute via a PR.

## Groundstates

One of the most prominent use-cases of MPS is to obtain the ground state of a given (quasi-) one-dimensional quantum Hamiltonian.
In MPSKit.jl, this can be achieved through `find_groundstate`:

```@docs; canonical=false
find_groundstate
```

The returned error measures convergence to a variational fixed point, which is not the same as accuracy; see [Ground-state accuracy](@ref).

There are a variety of algorithms that have been developed over the years, and many of them have been implemented in MPSKit.
Keep in mind that some of them are exclusive to finite or infinite systems, while others may work for both.
Many of these algorithms have different advantages and disadvantages, and figuring out the optimal algorithm is not always straightforward, since this may strongly depend on the model.
Here, we enumerate some of their properties in hopes of pointing you in the right direction. For convenience, the full list of algorithms is:

- [DMRG](@ref)
- [DMRG2](@ref)
- [VUMPS](@ref)
- [Gradient descent](@ref)
- [TDVP](@ref)
- [Time evolution MPO](@ref)
- [Quasiparticle Ansatz](@ref)
- [Finite excitations](@ref)
- ["Chepiga Ansatz"](@ref)

### DMRG

Probably the most widely used algorithm for optimizing groundstates with MPS is [`DMRG`](@ref) and its variants.
This algorithm sweeps through the system, optimizing a single site or pair of sites while keeping all others fixed.
Since this local problem can be solved efficiently, the global optimal state follows by alternating through the system.
However, because of the single-site nature of this algorithm, this can never alter the bond dimension of the state, such that there is no way of dynamically increasing the precision.
This can become particularly relevant in the cases where symmetries are involved, since then finding a good distribution of charges is also required.
To circumvent this, it is also possible to optimize over two sites at the same time with [`DMRG2`](@ref), followed by a truncation back to the single site states.
This can dynamically change the bond dimension but comes at an increase in cost.

```@docs; canonical=false
DMRG
DMRG2
```

For infinite systems, a similar approach can be used by dynamically adding new sites to the middle of the system and optimizing over them.
This gradually increases the system size until the boundary effects are no longer felt.
However, because of this approach, for critical systems this algorithm can be quite slow to converge, since the number of steps needs to be larger than the correlation length of the system.
Again, both a single-site and a two-site version are implemented, to have the option to dynamically increase the bond dimension at a higher cost.

```@docs; canonical=false
IDMRG
IDMRG2
```

### VUMPS

[`VUMPS`](@ref) is an (I)DMRG inspired algorithm that can be used to variationally find the ground state as a Uniform (infinite) Matrix Product State.
In particular, a local update is followed by a re-gauging procedure that effectively replaces the entire network with the newly updated tensor.
Compared to IDMRG, this often achieves a higher rate of convergence, since updates are felt throughout the system immediately.
Nevertheless, this algorithm only works whenever the state is injective, i.e. there is a unique ground state.
Since VUMPS is a single-site algorithm, it cannot alter the bond dimension.

```@docs; canonical=false
VUMPS
```

### Gradient descent

Both finite and infinite matrix product states can be parametrized by a set of isometric tensors,
which we can optimize over.
Making use of the geometry of the manifold (a Grassmann manifold), we can greatly outperform naive optimization strategies.
Compared to the other algorithms, quite often the convergence rate in the tail of the optimization procedure is higher, such that often the fastest method combines a different algorithm far from convergence with this algorithm close to convergence.
Since this is again a single-site algorithm, there is no way to alter the bond dimension.

```@docs; canonical=false
GradientGrassmann
```

## Time evolution

Given a particular state, it can also often be useful to examine the evolution of certain properties over time.
To that end, there are two main approaches to solving the Schrödinger equation in MPSKit.

```math
i \hbar \frac{d}{dt} \Psi = H \Psi \implies \Psi(t) = \exp{\left(-iH(t - t_0)\right)} \Psi(t_0)
```

```@docs; canonical=false
timestep
time_evolve
make_time_mpo
```

### TDVP

The first is focused around approximately solving the equation for a small timestep, and repeating this until the desired evolution is achieved.
This can be achieved by projecting the equation onto the tangent space of the MPS, and then solving the results.
This procedure is commonly referred to as the [`TDVP`](@ref) algorithm, which again has a two-site variant to allow for dynamically altering the bond dimension.

There are three ways to let the bond dimension follow the entanglement rather than fixing it up front:
- [`TDVP2`](@ref) evolves two sites at a time and splits the result back apart with a truncated SVD.
- [`TDVP`](@ref) with an `alg_expand` keeps the cheaper single-site update and instead expands the bond with directions orthogonal to the current state before each local update, recovering controlled bond expansion (CBE).
- [`BUG`](@ref) is a different integrator altogether: it advances basis and core tensors forward in time with no backward substep, which makes it better behaved for imaginary-time evolution, and it is rank-adaptive when given a `trunc`.

```@docs; canonical=false
TDVP
TDVP2
BUG
```

### Time evolution MPO

The other approach instead tries to first approximately represent the evolution operator, and only then attempts to apply this operator to the initial state.
Typically the first step happens through [`make_time_mpo`](@ref), while the second can be achieved through [`approximate`](@ref).
Here, there are several algorithms available

```@docs; canonical=false
WI
WII
TaylorCluster
```

Time evolution has three distinct error sources, only one of which is reported back to the user.
See [Time evolution accuracy](@ref).

## Excitations

It might also be desirable to obtain information beyond the lowest energy state of a given system, and study the dispersion relation.
While it is typically not feasible to resolve states in the middle of the energy spectrum, there are several ways to target a few of the lowest-lying energy states.
None of these report an error.
For what limits their accuracy, see [Excitation accuracy](@ref).

```@docs; canonical=false
excitations
```

```@setup excitations
using TensorKit, MPSKit, MPSKitModels
```

### Quasiparticle Ansatz

The Quasiparticle Ansatz offers an approach to compute low-energy eigenstates in quantum
systems, playing a key role in both finite and infinite systems. It leverages localized
perturbations for approximations, as detailed in [haegeman2013](@cite).

#### Finite Systems:

In finite systems, we approximate low-energy states by altering a single tensor in the
Matrix Product State (MPS) for each site, and summing these across all sites. This method
introduces additional gauge freedoms, utilized to ensure orthogonality to the ground state.
Optimizing within this framework translates to solving an eigenvalue problem. For example,
in the transverse field Ising model, we calculate the first excited state as shown in the
provided code snippet, and check the accuracy against theoretical values. Some deviations
are expected, both due to finite-bond-dimension and finite-size effects.

```@example excitations
# Model parameters
g = 10.0
L = 16
H = transverse_field_ising(FiniteChain(L); g)

# Finding the ground state
ψ₀ = FiniteMPS(L, ℂ^2, ℂ^32)
ψ, = find_groundstate(ψ₀, H; verbosity=0)

# Computing excitations using the Quasiparticle Ansatz
Es, ϕs = excitations(H, QuasiparticleAnsatz(), ψ; num=1)
isapprox(Es[1], 2(g - 1); rtol=1e-2)
```

#### Infinite Systems:

The ansatz in infinite systems maintains translational invariance by perturbing every site
in the unit cell in a plane-wave superposition, requiring momentum specification. The
[Haldane gap](https://iopscience.iop.org/article/10.1088/0953-8984/1/19/001) computation in
the Heisenberg model illustrates this approach.

```@example excitations
# Setting up the model and momentum
momentum = π
H = heisenberg_XXX()

# Ground state computation
ψ₀ = InfiniteMPS(ℂ^3, ℂ^48)
ψ, = find_groundstate(ψ₀, H; verbosity=0)

# Excitation calculations
Es, ϕs = excitations(H, QuasiparticleAnsatz(), momentum, ψ)
isapprox(Es[1], 0.41047925; atol=1e-4)
```

#### Charged excitations:

When dealing with symmetric systems, the default optimization is for eigenvectors with
trivial total charge. However, quasiparticles with different charges can be obtained using
the sector keyword. For instance, in the transverse field Ising model, we consider an
excitation built up of flipping a single spin, aligning with `Z2Irrep(1)`.

```@example excitations
g = 10.0
L = 16
H = transverse_field_ising(Z2Irrep, FiniteChain(L); g)
ψ₀ = FiniteMPS(L, Z2Space(0 => 1, 1 => 1), Z2Space(0 => 16, 1 => 16))
ψ, = find_groundstate(ψ₀, H; verbosity=0)
Es, ϕs = excitations(H, QuasiparticleAnsatz(), ψ; num=1, sector=Z2Irrep(1))
isapprox(Es[1], 2(g - 1); rtol=1e-2) # infinite analytical result
```

```@docs; canonical=false
QuasiparticleAnsatz
```

### Finite excitations

For finite systems we can also do something else - find the ground state of the Hamiltonian +
``\\text{weight} \sum_i | \\psi_i ⟩ ⟨ \\psi_i ``. This is also supported by calling

```@example excitations
# Model parameters
g = 10.0
L = 16
H = transverse_field_ising(FiniteChain(L); g)

# Finding the ground state
ψ₀ = FiniteMPS(L, ℂ^2, ℂ^32)
ψ, = find_groundstate(ψ₀, H; verbosity=0)

Es, ϕs = excitations(H, FiniteExcited(), ψ; num=1)
isapprox(Es[1], 2(g - 1); rtol=1e-2)
```

```@docs; canonical=false
FiniteExcited
```

### "Chepiga Ansatz"

Computing excitations in critical systems poses a significant challenge due to the diverging
correlation length, which requires very large bond dimensions. However, we can leverage this
long-range correlation to effectively identify excitations. In this context, the left/right
gauged MPS, serving as isometries, are effectively projecting the Hamiltonian into the
low-energy sector. This projection method is particularly effective in long-range systems,
where excitations are distributed throughout the entire system. Consequently, the low-lying
energy spectrum can be extracted by diagonalizing the effective Hamiltonian (without any
additional DMRG costs!). The states of these excitations are then represented by the ground
state MPS, with one site substituted by the corresponding eigenvector. This approach is
often referred to as the 'Chepiga ansatz', named after one of the authors of this paper
[chepiga2017](@cite).

This is supported via the following syntax:

```@example excitations
g = 10.0
L = 16
H = transverse_field_ising(FiniteChain(L); g)
ψ₀ = FiniteMPS(L, ComplexSpace(2), ComplexSpace(32))
ψ, envs, = find_groundstate(ψ₀, H; verbosity=0)
E₀ = real(sum(expectation_value(ψ, H, envs)))
Es, ϕs = excitations(H, ChepigaAnsatz(), ψ, envs; num=1)
isapprox(Es[1] - E₀, 2(g - 1); rtol=1e-2) # infinite analytical result
```

In order to improve the accuracy, a two-site version also exists, which varies two
neighbouring sites:

```@example excitations
Es, ϕs = excitations(H, ChepigaAnsatz2(), ψ, envs; num=1)
isapprox(Es[1] - E₀, 2(g - 1); rtol=1e-2) # infinite analytical result
```

## Errors and accuracy

Most algorithms in MPSKit report an error alongside their result, and the manual pages above refer to it as `ϵ` throughout.
That single name covers genuinely different quantities, and the differences matter.
This section clarifies the differences, and explains what they do and, just as important, what they don't measure.

### The error convention

Where `ϵ` is a *truncation* error, it is the 2-norm of the discarded singular values of a single factorisation, so that `ϵ²` is the discarded weight and the squared norm of the factorised tensor drops by exactly `ϵ²`.
The name "discarded weight" refers to it truly representing a probability: the ``\sigma_\alpha^2`` are the eigenvalues of the reduced density matrix across the cut, i.e. the statistical weights of the Schmidt states, summing to 1 for a normalised state.
So `ϵ²` is the probability weight thrown away and `ϵ` is the corresponding amplitude, which is why squares appear wherever these errors are combined.
Note that `ϵ` is absolute rather than relative: it is not divided by the norm of the state, which under a non-renormalising algorithm drifts away from 1 precisely as truncation accumulates.

What differs between algorithms is how the per-bond values are *aggregated*.
For this reason, values between algorithms are not directly comparable.

!!! warning
    Not every `ϵ` is a truncation error.
    The `ϵ` returned by [`find_groundstate`](@ref), [`leading_boundary`](@ref) and the iterative [`approximate`](@ref) algorithms is a *convergence* measure, with no truncation interpretation at all.
    The two are unrelated quantities that happen to share a name; see the two sections below.

### Ground-state accuracy

[`find_groundstate`](@ref), [`leading_boundary`](@ref) and the iterative [`approximate`](@ref) algorithms return the quantity their `tol` is compared against.
For the sweeping algorithms ([`DMRG`](@ref), [`DMRG2`](@ref), [`VUMPS`](@ref), [`IDMRG`](@ref), [`IDMRG2`](@ref)) this is the Galerkin error: the norm of the local gradient projected orthogonally to the current state.
It vanishes exactly at a variational fixed point.
[`GradientGrassmann`](@ref) instead reports the norm of the Riemannian gradient from its optimizer.
Both vanish at a fixed point and both are gradient norms.
However, since they are taken in different metrics, their magnitudes are not directly comparable.
In particular, a `tol` tuned for one is not a `tol` tuned for the other.

In other words, convergence is only defined relative to the manifold you are optimising over.
A single-site algorithm at a fixed bond dimension can drive its `ϵ` to machine precision and still be far from the true ground state, because the error that remains is the bond dimension itself, which no amount of further sweeping can address.
A small `ϵ` certifies a fixed point, not an accurate state.
Growing the bond dimension is the job of the two-site algorithms ([`DMRG2`](@ref), [`IDMRG2`](@ref)) or of a bond expansion ([`DMRG`](@ref) with an `alg_expand`, or an expanding `alg_gauge` such as [`DMRG3S`](@ref)); see also [`changebonds`](@ref).

Once an algorithm does truncate, the two error notions interact.
The Galerkin error cannot fall below the level set by the weight being discarded each sweep, so a truncating scheme converges once `ϵ` reaches the truncation error rather than the (unreachable) bare `tol`.
[`DMRG`](@ref)/[`DMRG2`](@ref) account for this: their stopping test is `ϵ ≤ max(tol, maximum(ϵ_trunc))`, which reduces to the plain `ϵ ≤ tol` when nothing is truncated.

Neither measure is an error bar on an observable.
For that, the standard route is the energy variance ``\langle H^2 \rangle - \langle H \rangle^2`` and extrapolation of observables towards zero variance ([Hubig et al.](@cite hubig2018)).

### Time evolution accuracy

Unlike a ground-state search, a time evolution has no convergence criterion to run to.
There is no fixed point, and the error is made at every step.
Three sources behave differently and only one of them is reported.

* **Truncation error.**
  Whenever a bond is cut back down, the discarded singular values are lost from the state.
  [`timestep`](@ref) and [`time_evolve`](@ref) return this as their third value `ϵ`, the norm of the discarded component, so that `ϵ²` is the discarded weight.
  It is the error you control through the algorithm's `trunc`, and the only one that is free to compute, since the truncating SVD produces it anyway.
  It is non-zero for [`TDVP2`](@ref), for [`BUG`](@ref) with a `trunc`, and for [`TDVP`](@ref) with a bond expansion.
  Plain single-site [`TDVP`](@ref) runs at fixed bond dimension and returns exactly `0`.

* **Projection error.**
  Single-site [`TDVP`](@ref) confines the evolution to the tangent space of a fixed-bond-dimension manifold, ``\lVert (1 - P_{T_\psi}) H \psi \rVert``.
  The component of the exact evolution pointing off that manifold is simply dropped, and this happens even with no truncation and exact local solves.
  It is not reported as measuring it costs an extra effective-Hamiltonian application per site.
  This is what a bond expansion (CBE) exists to reduce ([Li et al.](@cite li2024)).

* **Time-discretization error.**
  The projector splitting is globally ``O(dt^2)`` for the symmetric back-and-forth sweep ([Lubich et al.](@cite lubich2015), [Paeckel et al.](@cite paeckel2019)), so it is controlled by `dt` alone.
  This can only be estimated by comparing one step of `dt` against two of `dt / 2`.

A trustworthy run needs all three under control, not just a small `ϵ`.
In practice: pick `dt` from a convergence check, pick `trunc` from the accumulated `ϵ`, and use a bond-adaptive scheme ([`TDVP2`](@ref), [`BUG`](@ref), or [`TDVP`](@ref) with `alg_expand`) whenever entanglement grows during the evolution, since a fixed bond dimension silently converts entanglement growth into projection error.

#### Summing local errors in squares and relation to the norm

``\epsilon^2`` is the quantity that adds exactly.
Each local truncation is an orthogonal projection, so it removes exactly ``\epsilon_k^2`` from the squared norm, and the substeps between truncations preserve the norm.
Summing the squares therefore tracks a conserved "cost".
The alternative is to sum ``\epsilon_k``: ``\lVert \psi_{\text{untruncated}} - \psi \rVert \le \sum_k \epsilon_k``, a distance to the untruncated solution, which is a different and always larger quantity.

By default none of the time evolution algorithms renormalize (`normalize = false`), which is deliberate.
In real time the local exponentials are unitary, so truncation is the only thing that changes the norm and it becomes a running record of what truncation has cost,

```math
\lVert \psi \rVert^2 = \lVert \psi_0 \rVert^2 - \epsilon^2 .
```

The reported `ϵ` is the norm deficit, and this composes across steps.
This follows from the following two facts put together, one per half of a local update.
1) An SVD truncation is an orthogonal projection onto the kept Schmidt vectors and is 2-norm optimal at that rank ([Schollwöck](@cite schollwoeck2011)), so the kept and discarded parts are orthogonal.
By Pythagoras the squared norm drops by exactly the discarded weight, the usual way of quantifying truncation during a time evolution ([Paeckel et al.](@cite paeckel2019)).
2) The local exponentials of the projector-splitting sweep are unitary, so TDVP conserves the norm and the energy exactly when the local equations are solved exactly ([Paeckel et al.](@cite paeckel2019)), contributing nothing to the norm change.

These two hold for the local updates of a step, so composing over all steps gives the identity.

!!! note
    "Exactly" in the second fact is up to the tolerance of the local exponentials, and is thus in practice only approximate due to integrator tolerance.

It is also specific to real time with `normalize = false`:

* **Imaginary time** evolves with the non-unitary ``\exp(-H dt)``, which rescales the state on its own.
  The norm then moves for two independent reasons, namely the physical decay of the weight and the truncation loss. One cannot separate them from each other. `ϵ` still counts only the truncation.
* **`normalize = true`** renormalizes at every local update, destroying the identity by construction.
  This is usually what you want for imaginary-time evolution used as a ground-state or thermal-state search.
  `ϵ` is still reported and is unaffected.
* **No truncation at all** (plain single-site [`TDVP`](@ref), or [`BUG`](@ref) with a QR gauge) gives ``\epsilon = 0``, and in real time the norm is then conserved exactly.
* An **`InfiniteMPS`** is regauged to norm 1 per site structurally, so its norm carries no such information and `normalize` has no effect.

### Excitation accuracy

[`excitations`](@ref) returns only `(energies, states)`: there is no error term, and none of the sources below is reported back to you.
They are worth knowing about, because the dominant one is usually not the one the algorithm is working on.

* **Inherited ground-state error.**
  Every method here builds on a ground state you supply and treats it as exact.
  Its error propagates straight into the gap, and since a gap is a difference of two large energies, it is typically the limiting factor.
  A well-converged ground state (in the sense of `ϵ` and bond dimension) is necessary for a meaningful excitation calculation.

* **Ansatz limitation.**
  [`QuasiparticleAnsatz`](@ref) varies over the single-quasiparticle tangent space on top of a fixed ground state.
  It is variational within that space and well suited to isolated quasiparticle branches, but multi-particle continua are not representable in it, so results there are not to be trusted.
  For infinite systems the momentum superposition itself is exact, so momentum is a good quantum number and no error enters through it.

* **Eigensolver convergence.**
  The local eigenvalue problem is solved with KrylovKit, and a run that fails to converge `num` states emits a warning on the residual when the verbosity is set high enough.
  This residual is neither returned nor thrown, so it is worth not running with warnings suppressed.

* **Penalty-based orthogonality** ([`FiniteExcited`](@ref)).
  Higher states are found by minimising ``H + \lambda \sum_i |\psi_i\rangle\langle\psi_i|`` against the previously converged states, with ``\lambda`` the `weight` field.
  A finite `weight` enforces orthogonality only approximately, so a residual overlap with a lower state biases the reported energy downwards.
  Since the reported value is the expectation value of the bare `H`, this bias is invisible in the output.
  Raising `weight` suppresses it at the cost of stretching the spectrum and slowing down the eigensolver's per-gap eigensolves.

* **Truncation** ([`ChepigaAnsatz2`](@ref)).
  The two-site excited state is split back to single-site tensors with a truncated SVD governed by `trunc`, and the resulting discarded weight is not reported.

## `changebonds`

Many of the previously mentioned algorithms do not possess a way to dynamically change to
bond dimension. This is often a problem, as the optimal bond dimension is often not a priori
known, or needs to increase because of entanglement growth throughout the course of a
simulation. [`changebonds`](@ref) exposes a way to change the bond dimension of a given
state.

```@docs; canonical=false
changebonds
```

All of these are controlled by a `trunc`, and the weight they discard is measured the same way as explained in the `ϵ` convention under [The error convention](@ref).
`changebonds` does not report it, since every algorithm has its own interpretation of the discarded singular values.

There are several different algorithms implemented, each having their own advantages and
disadvantages:

* [`SvdCut`](@ref): The simplest method for changing the bond dimension is found by simply
  locally truncating the state using an SVD decomposition. This yields a (locally) optimal
  truncation, but clearly cannot be used to increase the bond dimension. Note that a
  globally optimal truncation can be obtained by using the [`SvdCut`](@ref) algorithm in
  combination with [`approximate`](@ref). Since the output of this method might have a
  truncated bond dimension, the new state might not be identical to the input state.
  The truncation is controlled through `trunc`, which dictates how the singular values of
  the original state are truncated.


* [`OptimalExpand`](@ref): This algorithm is based on the idea of expanding the bond
  dimension by investigating the two-site derivative, and adding the most important blocks
  which are orthogonal to the current state. From the point of view of a local two-site
  update, this procedure is *optimal*, but it requires to evaluate a two-site derivative,
  which can be costly when the physical space is large. The state will remain unchanged, but
  a one-site scheme will now be able to push the optimization further. The subspace used for
  expansion can be truncated through `trunc`, which dictates how many singular values will
  be added.

* [`RandExpand`](@ref): This algorithm similarly adds blocks orthogonal to the current
  state, but does not attempt to select the most important ones, and rather just selects
  them at random. The advantage here is that this is much cheaper than the optimal expand,
  and if the bond dimension is grown slow enough, this still obtains a very good expansion
  scheme. Again, The state will remain unchanged and a one-site scheme will now be able to 
  push the optimization further. The subspace used for expansion can be truncated through
  `trunc`, which dictates how many orthogonal vectors will be added.

* [`VUMPSSvdCut`](@ref): This algorithm is based on the [`VUMPS`](@ref) algorithm, and
  consists of performing a two-site update, and then truncating the state back down. Because
  of the two-site update, this can again become expensive, but the algorithm has the option
  of both expanding as well as truncating the bond dimension. Here, `trunc` controls the
  truncation of the full state after the two-site update.

## Leading boundary

For statistical mechanics partition functions we want to find the approximate leading
boundary MPS. Again this can be done with VUMPS:

```julia
th = nonsym_ising_mpo()
ts = InfiniteMPS([ℂ^2],[ℂ^20]);
(ts,envs,_) = leading_boundary(ts,th,VUMPS(maxiter=400,verbosity=false));
```

If the mpo satisfies certain properties (positive and hermitian), it may also be possible to
use GradientGrassmann.

```@docs; canonical=false
leading_boundary
```

## `approximate`

Often, it is useful to approximate a given MPS by another, typically by one of a different
bond dimension. This is achieved by approximating an application of an MPO to the initial
state, by a new state.

```@docs; canonical=false
approximate
```

## Varia

What follows is a medley of lesser known (or used) algorithms and don't entirely fit under
one of the above categories.

### Dynamical DMRG

Dynamical DMRG has been described in other papers and is a way to find the propagator. The
basic idea is that to calculate ``G(z) = ⟨ V | (H-z)^{-1} | V ⟩ `` , one can variationally
find ``(H-z) |W ⟩ = | V ⟩ `` and then the propagator simply equals ``G(z) = ⟨ V | W ⟩``.

```@docs; canonical=false
propagator
DynamicalDMRG
NaiveInvert
Jeckelmann
```

### fidelity susceptibility

The fidelity susceptibility measures how much the ground state changes when tuning a
parameter in your Hamiltonian. Divergences occur at phase transitions, making it a valuable
measure when no order parameter is known.

```@docs; canonical=false
fidelity_susceptibility
```

### Boundary conditions

You can impose periodic or open boundary conditions on an infinite Hamiltonian, to generate a finite counterpart.
In particular, for periodic boundary conditions we still return an MPO that does not form a closed loop, such that it can be used with regular matrix product states.
This is straightforward to implement but, and while this effectively squares the bond dimension, it is still competitive with more advanced periodic MPS algorithms.

```@docs; canonical=false
open_boundary_conditions
periodic_boundary_conditions
```

### Exact diagonalization

As a side effect, our code supports exact diagonalization. The idea is to construct a finite
matrix product state with maximal bond dimension, and then optimize the middle site. Because
we never truncate the bond dimension, this single site effectively parametrizes the entire
Hilbert space.

```@docs; canonical=false
exact_diagonalization
```
