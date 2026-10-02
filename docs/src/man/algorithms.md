```@meta
DocTestSetup = :(using MPSKit, TensorKit, MPSKitModels)
```

# [Algorithms](@id um_algorithms)

Here is a collection of the algorithms that have been added to MPSKit.jl.
If a particular algorithm is missing, feel free to let us know via an issue, or contribute via a PR.

## Ground states

One of the most prominent use-cases of MPS is to obtain the ground state of a given (quasi-) one-dimensional quantum Hamiltonian.
In MPSKit.jl, this can be achieved through `find_groundstate`:

```@docs; canonical=false
find_groundstate
```

See also [Ground state accuracy](@ref).

There are a variety of algorithms that have been developed over the years, and many of them have been implemented in MPSKit.
Keep in mind that some of them are exclusive to finite or infinite systems, while others may work for both.
Many of these algorithms have different advantages and disadvantages, and figuring out the optimal algorithm is not always straightforward, since this may strongly depend on the model.
Here, we enumerate some of their properties in hopes of pointing you in the right direction. For convenience, the full list of algorithms is:

- [DMRG](@ref)
- [VUMPS](@ref)
- [Gradient descent](@ref)
- [TDVP](@ref)
- [Time evolution MPO](@ref)
- [Quasiparticle Ansatz](@ref)
- [Finite excitations](@ref)
- ["Chepiga Ansatz"](@ref)

### DMRG

Probably the most widely used algorithm for optimizing ground states with MPS is [`DMRG`](@ref) and its variants.
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

See also [Time evolution accuracy](@ref).

## Excitations

It might also be desirable to obtain information beyond the lowest energy state of a given system, and study the dispersion relation.
While it is typically not feasible to resolve states in the middle of the energy spectrum, there are several ways to target a few of the lowest-lying energy states.
See also [Excitation accuracy](@ref).

```@docs; canonical=false
excitations
```

```@setup excitations
using TensorKit, MPSKit, MPSKitModels
```

### Quasiparticle Ansatz

The Quasiparticle Ansatz offers an approach to compute low-energy eigenstates in quantum systems, playing a key role in both finite and infinite systems.
It leverages localized perturbations for approximations, as detailed in [haegeman2013](@cite).

#### Finite Systems:

In finite systems, we approximate low-energy states by altering a single tensor in the Matrix Product State (MPS) for each site, and summing these across all sites.
This method introduces additional gauge freedoms, utilized to ensure orthogonality to the ground state.
Optimizing within this framework translates to solving an eigenvalue problem.
For example, in the transverse field Ising model, we calculate the first excited state as shown in the provided code snippet, and check the accuracy against theoretical values.
Some deviations are expected, both due to finite-bond-dimension and finite-size effects.

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

The ansatz in infinite systems maintains translational invariance by perturbing every site in the unit cell in a plane-wave superposition, requiring momentum specification.
The [Haldane gap](https://iopscience.iop.org/article/10.1088/0953-8984/1/19/001) computation in the Heisenberg model illustrates this approach.

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

When dealing with symmetric systems, the default optimization is for eigenvectors with trivial total charge.
However, quasiparticles with different charges can be obtained using the sector keyword.
For instance, in the transverse field Ising model, we consider an excitation built up of flipping a single spin, aligning with `Z2Irrep(1)`.

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

For finite systems we can instead find the ground state of the Hamiltonian where we project out previously obtained eigenstates, i.e. by projecting out ``\\text{weight} \sum_i | \\psi_i ⟩ ⟨ \\psi_i ``.
This is also supported by calling

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

Computing excitations in critical systems poses a significant challenge due to the diverging correlation length, which requires very large bond dimensions.
However, we can leverage this long-range correlation to effectively identify excitations.
In this context, the left/right gauged MPS, serving as isometries, are effectively projecting the Hamiltonian into the low-energy sector.
This projection method is particularly effective in long-range systems, where excitations are distributed throughout the entire system.
Consequently, the low-lying energy spectrum can be extracted by diagonalizing the effective Hamiltonian (without any additional DMRG costs!).
The states of these excitations are then represented by the ground state MPS, with one site substituted by the corresponding eigenvector.
This approach is often referred to as the 'Chepiga ansatz', named after one of the authors of this paper [chepiga2017](@cite).

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

In order to improve the accuracy, a two-site version also exists, which varies two neighbouring sites:

```@example excitations
Es, ϕs = excitations(H, ChepigaAnsatz2(), ψ, envs; num=1)
isapprox(Es[1] - E₀, 2(g - 1); rtol=1e-2) # infinite analytical result
```

## Errors and accuracy

The algorithms that solve for a state, particularly [`find_groundstate`](@ref), [`leading_boundary`](@ref), [`approximate`](@ref), [`timestep`](@ref) and [`time_evolve`](@ref), return an [`AlgorithmInfo`](@ref) as their last value, describing how they arrived at their result.

```@docs; canonical=false
AlgorithmInfo
```

The rest of this section explains what quantities can be reported by the algorithms, and how these can be interpreted.

### Ground state accuracy

[`find_groundstate`](@ref), [`leading_boundary`](@ref) and the iterative [`approximate`](@ref) algorithms stop once a convergence measure drops below their `tol`, and report that measure together with a `converged` flag.

**Different measures for different algorithms.**
Which quantity is monitored depends on the algorithm, and these quantities cannot be compared to one another.
Each is therefore stored under its own name: `galerkin` for [`DMRG`](@ref), [`DMRG2`](@ref), [`VUMPS`](@ref) and [`VOMPS`](@ref), `gradientnorm` for [`GradientGrassmann`](@ref), `bondresidual` for [`IDMRG`](@ref) and [`IDMRG2`](@ref), and `localchange` for [`DMRG`](@ref) and [`DMRG2`](@ref) inside [`approximate`](@ref).
[`convergence_measure`](@ref) returns whichever of them is present, for code that only wants the number.
For the same reason, a `tol` that works well for one algorithm need not work well for another.

**Interaction with a changing bond dimension.**
An algorithm that truncates changes the state by the discarded weight in every sweep, so its convergence measure cannot drop below the level set by what is discarded.
Rather than running into `maxiter` for a `tol` that cannot be reached, [`DMRG`](@ref) and [`DMRG2`](@ref) consider themselves converged once `galerkin ≤ max(tol, maximum(truncation_errors))`, which reduces to `galerkin ≤ tol` when nothing is truncated.
The reported `truncation_errors` are those of the final sweep, i.e. what the returned state still discards.

**Convergence is not accuracy.**
For algorithms that remain on a fixed bond dimension MPS manifold, one can get the ground state *on that manifold* to converge to machine precision.
This however might still be far from the true ground state, whenever the bond dimension is not high enough.
Growing the bond dimension is the job of the two-site algorithms ([`DMRG2`](@ref), [`IDMRG2`](@ref)) or of a bond expansion ([`DMRG`](@ref) with an `alg_expand`, or an expanding `alg_gauge` such as [`DMRG3S`](@ref)); see also [`changebonds`](@ref).
Likewise, none of these measures is an error bar on an observable.
An independent check is the energy [`variance`](@ref), ``\langle H^2 \rangle - \langle H \rangle^2``, which vanishes for an exact eigenstate, as is comparing results across bond dimensions.

### Time evolution accuracy

A time evolution has three distinct sources of error ([Paeckel et al.](@cite paeckel2019)):

- The **truncation error**, from discarding singular values to keep the bond dimension bounded.
  This is the only one that is reported, as the `truncation_errors` of [`timestep`](@ref) and [`time_evolve`](@ref).
  It is non-zero for [`TDVP2`](@ref), for [`BUG`](@ref) with a `trunc`, and for [`TDVP`](@ref) with a bond expansion.
- The **projection error**, from restricting the evolution to MPS of the current bond dimension.
  Measuring it costs an extra two-site effective-Hamiltonian application per bond, so it is not reported.
  A fixed bond dimension turns entanglement growth into projection error, which is what a bond expansion (CBE) exists to reduce ([Li et al.](@cite li2024)).
- The **time-step error**, from splitting each step into a sequence of local updates, similar to a Trotter error.
  It decreases with `dt`, and can be estimated by comparing one step of `dt` against two steps of `dt / 2`.

Each of these has its own control: `dt` for the time-step error, `trunc` for the truncation error, and a bond-adaptive scheme ([`TDVP2`](@ref), [`BUG`](@ref), or [`TDVP`](@ref) with `alg_expand`) for the projection error whenever entanglement grows.

These controls are not independent.
With a truncation strategy [`trunc::TruncationStrategy`](@extref MatrixAlgebraKit.TruncationStrategy), every step can discard weight, however small `dt` is, and reaching a fixed final time `T` takes `T / dt` steps.
Halving `dt` therefore lowers the time-step error, but doubles the number of truncations, so the accumulated truncation error can grow.
For a given `trunc` there is an optimal `dt`, below which taking smaller steps makes the result worse rather than better.

### Excitation accuracy

[`excitations`](@ref) returns only `(energies, states)`: there is no error term, and none of the sources below is reported back to you.
They are worth knowing about, because the dominant one is usually not the one the algorithm is working on.

- **Inherited ground state error.**
  Every method builds on the ground state you supply and treats it as exact.
  Since a gap is a difference of two large energies, that error propagates straight into it and is typically the limiting factor.

- **Ansatz limitation.**
  [`QuasiparticleAnsatz`](@ref) varies over the single-quasiparticle tangent space on top of a fixed ground state, so it is variational within that space and suited to isolated quasiparticle branches.
  Its error is bounded exponentially in the support of the local operator, at a rate set by the gaps below *and* above the targeted eigenvalue ([Haegeman et al.](@cite haegeman2013)).

- **Eigensolver convergence.**
  The eigenvalue problem is solved with KrylovKit, and a run that fails to converge `num` states emits a warning carrying the residual when the verbosity is high enough.
  That residual is neither returned nor thrown, so it is worth not suppressing warnings.
  Nearly degenerate levels are the ones most likely to come back unconverged.

- **Penalty-based orthogonality.**
  [`FiniteExcited`](@ref) minimises ``H + \lambda \sum_i |\psi_i\rangle\langle\psi_i|`` against the previously converged states, with ``\lambda`` the `weight` field.
  A finite `weight` enforces orthogonality only approximately, so a residual overlap with a lower state biases the energy downwards — invisibly, since the reported value is the expectation value of the bare `H`.
  Raising `weight` suppresses the bias at the cost of stretching the spectrum and slowing the eigensolver.

- **Truncation** ([`ChepigaAnsatz2`](@ref)).
  The two-site excited state is split back to single-site tensors with a truncated SVD governed by `trunc`, and the discarded weight is not reported.

## `changebonds`

Many of the previously mentioned algorithms do not possess a way to dynamically change to
bond dimension. This is often a problem, as the optimal bond dimension is often not a priori
known, or needs to increase because of entanglement growth throughout the course of a
simulation. [`changebonds`](@ref) exposes a way to change the bond dimension of a given
state.

```@docs; canonical=false
changebonds
```

All of these are controlled by a `trunc`, and the weight they discard is measured the same way as the `truncation_errors` of an [`AlgorithmInfo`](@ref).
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

The associated eigenvalue, i.e. the partition function per unit cell, is obtained with [`dominant_eigenvalue`](@ref).
This is deliberately not `expectation_value`: the boundary contracts against a different line than the one the operator acts on, so it is not an overlap.
For a single-line `InfiniteMPS`/`InfiniteMPO` pair the two coincide and `expectation_value` forwards accordingly.

```@docs; canonical=false
dominant_eigenvalue
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
