```@meta
DocTestSetup = quote
    using MPSKit, MPSKitModels, TensorKit
end
```

# [Algorithms](@id um_algorithms)

## Minimizing the energy

There are a number of different possible energy-minimization algorithms, depending on the
system size. Exclusive to finite systems are

    - DMRG

    - DMRG2

Exclusive to infinite systems are

    - IDMRG

    - IDMRG2

    - VUMPS

with a last algorithm - GradientGrassmann - implemented for both finite and infinite
systems.

WindowMPS, which is a finite patch of mutable tensors embedded in an infinite MPS, is
handled as a finite system where we only optimize over the patch of mutable tensors.

### DMRG

```julia
state = FiniteMPS(20,ℂ^2,ℂ^10);
operator = nonsym_ising_ham();
(groundstate,environments,delta) = find_groundstate!(state,operator,DMRG())
```

The DMRG algorithm sweeps through the system, optimizing every site. Because of its
single-site behaviour, this will always keep the bond dimension fixed. If you do want to
increase the bond dimension dynamically, then there are two options. Either you use the
two-site variant of DMRG (DMRG2()), or you make use of the finalize option. Finalize is a
function that gets called at the end of every DMRG iteration. Within that function call one
can modify the state.

```julia
function my_finalize(iter,state,H,envs)
    println("Hello from iteration $iter")
    return state,envs;
end

(groundstate,environments,delta) = find_groundstate!(state,operator,DMRG(finalize = my_finalize))
```

### DMRG2

```julia
state = FiniteMPS(20,ℂ^2,ℂ^10);
operator = nonsym_ising_ham();
(groundstate,environments,delta) = find_groundstate!(state,operator,DMRG2(trscheme=truncbelow(1e-7)));
```

The twosite variant of DMRG, which optimizes blocks of two sites and then decomposes them
into 2 MPS tensors using the svd decomposition. By truncating the singular values up to a
desired precision, one can dynamically grow and shrink the bond dimension as needed.
However, this truncation in turn introduces an error, which is why a state converged with
DMRG2 can often be slightly further converged by subsequently using DMRG.

### IDMRG

```julia
state = InfiniteMPS([ℂ^2],[ℂ^10]);
operator = nonsym_ising_ham();
(groundstate,environments,delta) = find_groundstate(state,operator,IDMRG1())
```

The DMRG algorithm for finite systems can be generalized to infinite MPS. The idea is to
start with a finite system and grow the system size, while we are sweeping through the
system. This is again a single site algorithm, and therefore preserves the initial bond
dimension.

### IDMRG2
```julia
state = InfiniteMPS([ℂ^2,ℂ^2],[ℂ^10,ℂ^10]);
operator = repeat(nonsym_ising_ham(),2);
(groundstate,environments,delta) = find_groundstate(state,operator,IDMRG2(trscheme=truncbelow(1e-5)))
```

The generalization of DMRG2 to infinite systems has the same caveats as its finite
counterpart. We furthermore require a unitcell ≥ 2. As a rule of thumb, a truncation cutoff
of 1e-5 is already really good.

### VUMPS

VUMPS is an (I)DMRG inspired algorithm that can be used to find the groundstate of infinite
matrix product states
```julia
state = InfiniteMPS([ℂ^2],[ℂ^10]);
operator = nonsym_ising_ham();
(groundstate,environments,delta) = find_groundstate(state,operator,VUMPS())
```

much like DMRG, it cannot modify the bond dimension, and this has to be done manually in the
finalize function.

### Gradient descent

Both finite and infinite matrix product states can be parametrized by a set of unitary
matrices, and we can then perform gradient descent on this unitary manifold. Due to some
technical reasons (gauge freedom), this manifold further restricts to a grassmann manifold.

```julia
state = InfiniteMPS([ℂ^2],[ℂ^10]);
operator = nonsym_ising_ham();
(groundstate,environments,delta) = find_groundstate(state,operator,GradientGrassmann())
```

## Time evolution

### TDVP

There is an implementation of the one-site TDVP scheme for finite and infinite MPS:
```julia
(newstate,environments) = timestep(state,operator,dt,TDVP())
```

and the two-site scheme for finite MPS (TDVP2()). Similarly to DMRG, the one site scheme
will preserve the bond dimension, and expansion has to be done manually.

### Time evolution MPO

We have rudimentary support for turning an MPO hamiltonian into a time evolution MPO.

```julia
make_time_mpo(H,dt,alg::WI)
make_time_mpo(H,dt,alg::WII)
```

two algorithms are available, corresponding to different orders of precision. It is possible
to then multiply a state by this MPO, or to approximate (MPO,state) by a new state

```julia
state = InfiniteMPS([ℂ^2],[ℂ^10]);
operator = nonsym_ising_ham();
mpo = make_time_mpo(operator, 0.1, WII());
approximate(state, (mpo, state), VUMPS())
```

This feature is at the moment not very well supported.

## Excitations

### Quasiparticle Ansatz

The Quasiparticle Ansatz offers an approach to compute low-energy eigenstates in quantum
systems, playing a key role in both finite and infinite systems. It leverages localized
perturbations for approximations, as detailed in [this
paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.111.080401).

#### Finite Systems:

In finite systems, we approximate low-energy states by altering a single tensor in the
Matrix Product State (MPS) for each site, and summing these across all sites. This method
introduces additional gauge freedoms, utilized to ensure orthogonality to the ground state.
Optimizing within this framework translates to solving an eigenvalue problem. For example,
in the transverse field Ising model, we calculate the first excited state as shown in the
provided code snippet, amd check the accuracy against theoretical values. Some deviations
are expected, both due to finite-bond-dimension and finite-size effects.

```jldoctest; output = false
# Model parameters
g = 10.0
L = 16
H = transverse_field_ising(; g)

# Finding the ground state
ψ₀ = FiniteMPS(L, ℂ^2, ℂ^32)
ψ, = find_groundstate(ψ₀, H; verbosity=0)

# Computing excitations using the Quasiparticle Ansatz
Es, ϕs = excitations(H, QuasiparticleAnsatz(), ψ; num=1)
isapprox(Es[1], 2(g - 1); rtol=1e-2)

# output

true
```

#### Infinite Systems:

The ansatz in infinite systems maintains translational invariance by perturbing every site
in the unit cell in a plane-wave superposition, requiring momentum specification. The
[Haldane gap](https://iopscience.iop.org/article/10.1088/0953-8984/1/19/001) computation in
the Heisenberg model illustrates this approach.

```jldoctest; output = false
# Setting up the model and momentum
momentum = π
H = heisenberg_XXX()

# Ground state computation
ψ₀ = InfiniteMPS(ℂ^3, ℂ^48)
ψ, = find_groundstate(ψ₀, H; verbosity=0)

# Excitation calculations
Es, ϕs = excitations(H, QuasiparticleAnsatz(), momentum, ψ)
isapprox(Es[1], 0.41047925; atol=1e-4)

# output

true
```

#### Charged excitations:

When dealing with symmetric systems, the default optimization is for eigenvectors with
trivial total charge. However, quasiparticles with different charges can be obtained using
the sector keyword. For instance, in the transverse field Ising model, we consider an
excitation built up of flipping a single spin, aligning with `Z2Irrep(1)`.

```jldoctest; output = false
g = 10.0
L = 16
H = transverse_field_ising(Z2Irrep; g)
ψ₀ = FiniteMPS(L, Z2Space(0 => 1, 1 => 1), Z2Space(0 => 16, 1 => 16))
ψ, = find_groundstate(ψ₀, H; verbosity=0)
Es, ϕs = excitations(H, QuasiparticleAnsatz(), ψ; num=1, sector=Z2Irrep(1))
isapprox(Es[1], 2(g - 1); rtol=1e-2) # infinite analytical result

# output

true
```

### Finite excitations

For finite systems we can also do something else - find the groundstate of the hamiltonian +
``weight \sum_i | psi_i > < psi_i ``. This is also supported by calling

```julia
th = nonsym_ising_ham()
ts = FiniteMPS(10,ℂ^2,ℂ^12);
(ts,envs,_) = find_groundstate(ts,th,DMRG(verbosity=0));
(energies,Bs) = excitations(th,FiniteExcited(),ts,envs);
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
often referred to as the 'Chepiga ansatz', named after one of the authors of this paper.

This is supported via the following syntax:

```jldoctest
g = 1.0
L = 16
H = transverse_field_ising(; g)
ψ₀ = FiniteMPS(L, ComplexSpace(2), ComplexSpace(32))
ψ, envs, = find_groundstate(ψ₀, H; verbosity=0)
E₀ = real(sum(expectation_value(ψ, H, envs)))
Es, ϕs = excitations(H, ChepigaAnsatz(), ψ, envs; num=1)
isapprox(Es[1], 2(g - 1); rtol=1e-2) # infinite analytical result

# output

true
```

In order to improve the accuracy, a two-site version also exists, which varies two
neighbouring sites:

```jldoctest
g = 1.0
L = 16
H = transverse_field_ising(; g)
ψ₀ = FiniteMPS(L, ComplexSpace(2), ComplexSpace(32))
ψ, envs, = find_groundstate(ψ₀, H; verbosity=0)
E₀ = real(sum(expectation_value(ψ, H, envs)))
Es, ϕs = excitations(H, ChepigaAnsatz2(), ψ, envs; num=1)
isapprox(Es[1], 2(g - 1); rtol=1e-2) # infinite analytical result

# output

true
```

The algorithm is described in more detail in the following paper:

- Chepiga, N., & Mila, F. (2017). Excitation spectrum and density matrix renormalization
  group iterations. Physical Review B, 96(5), 054425.

## `changebonds`

Many of the previously mentioned algorithms do not possess a way to dynamically change to
bond dimension. This is often a problem, as the optimal bond dimension is often not a priori
known, or needs to increase because of entanglement growth throughout the course of a
simulation. [`changebonds`](@ref) exposes a way to change the bond dimension of a given
state.

```@docs; canonical=false
changebonds
```

There are several different algorithms implemented, each having their own advantages and
disadvantages:

* [`SvdCut`](@ref): The simplest method for changing the bonddimension is found by simply
  locally truncating the state using an SVD decomposition. This yields a (locally) optimal
  truncation, but clearly cannot be used to increase the bond dimension. Note that a
  globally optimal truncation can be obtained by using the [`SvdCut`](@ref) algorithm in
  combination with [`approximate`](@ref). Since the output of this method might have a
  truncated bonddimension, the new state might not be identical to the input state.
  The truncation is controlled through `trscheme`, which dictates how the singular values of
  the original state are truncated.


* [`OptimalExpand`](@ref): This algorithm is based on the idea of expanding the bond
  dimension by investigating the two-site derivative, and adding the most important blocks
  which are orthogonal to the current state. From the point of view of a local two-site
  update, this procedure is *optimal*, but it requires to evaluate a two-site derivative,
  which can be costly when the physical space is large. The state will remain unchanged, but
  a one-site scheme will now be able to push the optimization further. The subspace used for
  expansion can be truncated through `trscheme`, which dictates how many singular values will
  be added.

* [`RandExpand`](@ref): This algorithm similarly adds blocks orthogonal to the current
  state, but does not attempt to select the most important ones, and rather just selects
  them at random. The advantage here is that this is much cheaper than the optimal expand,
  and if the bond dimension is grown slow enough, this still obtains a very good expansion
  scheme. Again, The state will remain unchanged and a one-site scheme will now be able to 
  push the optimization further. The subspace used for expansion can be truncated through
  `trscheme`, which dictates how many singular values will be added.

* [`VUMPSSvdCut`](@ref): This algorithm is based on the [`VUMPS`](@ref) algorithm, and
  consists of performing a two-site update, and then truncating the state back down. Because
  of the two-site update, this can again become expensive, but the algorithm has the option
  of both expanding as well as truncating the bond dimension. Here, `trscheme` controls the
  truncation of the full state after the two-site update.

## leading boundary

For statmech partition functions we want to find the approximate leading boundary MPS. Again
this can be done with VUMPS:

```julia
th = nonsym_ising_mpo()
ts = InfiniteMPS([ℂ^2],[ℂ^20]);
(ts,envs,_) = leading_boundary(ts,th,VUMPS(maxiter=400,verbosity=false));
```

if the mpo satisfies certain properties (positive and hermitian), it may also be possible to
use GradientGrassmann.

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
basic idea is that to calculate ``G(z) = < V | (H-z)^{-1} | V > `` , one can variationally
find ``(H-z) |W > = | V > `` and then the propagator simply equals ``G(z) = < V | W >``.

### fidelity susceptibility

The fidelity susceptibility measures how much the groundstate changes when tuning a
parameter in your hamiltonian. Divergences occur at phase transitions, making it a valuable
measure when no order parameter is known.

```julia
fidelity_susceptibility(groundstate,H_0,perturbing_Hams::AbstractVector)
```

### periodic boundary conditions

You can impose periodic boundary conditions on the hamiltonian itself, while still using a
normal OBC finite matrix product states. This is straightforward to implement but
competitive with more advanced PBC MPS algorithms.

### exact diagonalization

As a side effect, our code support exact diagonalization. The idea is to construct a finite
matrix product state with maximal bond dimension, and then optimize the middle site. Because
we never truncated the bond dimension, this single site effectively parametrizes the entire
hilbert space.

```julia
exact_diagonalization(periodic_boundary_conditions(su2_xxx_ham(spin=1),10),which=:SR) # find the groundstate on 10 sites
```

```@meta
DocTestSetup = nothing
```
