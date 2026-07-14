# [Public API](@id public_api)

This page is the curated, stable public API surface of MPSKit — the symbols that are exported and intended for direct use.
Each entry links to its full docstring in the [Library](@ref lib_index) index.
The category reference pages ([States](@ref lib_states), [Operators](@ref lib_operators), [Ground-state algorithms](@ref lib_groundstate)) group the same docstrings by topic.

!!! note
    Anything not listed here (or marked internal in the [Library](@ref lib_index) index) is
    not part of the public API and may change without notice.

## States

The matrix product state types — finite, infinite, windowed, and multi-line.

[`FiniteMPS`](@ref), [`InfiniteMPS`](@ref), [`WindowMPS`](@ref), [`MultilineMPS`](@ref)

## Operators and Hamiltonians

Matrix product operators and Hamiltonians, finite and infinite, plus the wrappers used to build time-dependent and summed operators.

[`AbstractMPO`](@ref), [`MPO`](@ref), [`FiniteMPO`](@ref), [`InfiniteMPO`](@ref), [`MultilineMPO`](@ref), [`MPOHamiltonian`](@ref), [`FiniteMPOHamiltonian`](@ref), [`InfiniteMPOHamiltonian`](@ref), [`JordanMPOTensor`](@ref), [`MultipliedOperator`](@ref), [`TimedOperator`](@ref), [`UntimedOperator`](@ref), [`LazySum`](@ref)

## Environments

The caches that store partially contracted tensor networks and are reused throughout the algorithms; see the concept page on [Environments](@ref concept_environments) for why they exist.

[`environments`](@ref)

## Ground states and boundaries

The ground-state search and 2D leading-boundary interface, and the DMRG/VUMPS/IDMRG family of algorithms that implement it.

[`find_groundstate`](@ref), [`leading_boundary`](@ref), [`approximate`](@ref), [`VUMPS`](@ref), [`VOMPS`](@ref), [`DMRG`](@ref), [`DMRG2`](@ref), [`IDMRG`](@ref), [`IDMRG2`](@ref), [`GradientGrassmann`](@ref)

## Bond dimension

Expanding or truncating a state's virtual spaces, and the algorithms that drive it.

[`changebonds`](@ref), [`OptimalExpand`](@ref), [`RandExpand`](@ref), [`SvdCut`](@ref), [`VUMPSSvdCut`](@ref)

## Time evolution

Real- and imaginary-time evolution drivers and the algorithms and MPO approximations that implement them.

[`time_evolve`](@ref), [`timestep`](@ref), [`make_time_mpo`](@ref), [`TDVP`](@ref), [`TDVP2`](@ref), [`WI`](@ref), [`WII`](@ref), [`TaylorCluster`](@ref)

## Excitations

The excitation interface and the quasiparticle-ansatz and finite-excited-state algorithms that produce excited states on top of a ground state.

[`excitations`](@ref), [`FiniteExcited`](@ref), [`QuasiparticleAnsatz`](@ref), [`ChepigaAnsatz`](@ref), [`ChepigaAnsatz2`](@ref)

## Linear problems and spectral functions

Solving the MPS linear problems behind dynamical/spectral quantities, such as propagators and susceptibilities.

[`propagator`](@ref), [`DynamicalDMRG`](@ref), [`NaiveInvert`](@ref), [`Jeckelmann`](@ref), [`exact_diagonalization`](@ref), [`fidelity_susceptibility`](@ref)

## Observables and analysis

Extracting physical quantities and analysis diagnostics from an MPS — expectation values, correlators, spectra, and entanglement.

[`expectation_value`](@ref), [`correlator`](@ref), [`variance`](@ref), [`correlation_length`](@ref), [`marek_gap`](@ref), [`transfer_spectrum`](@ref), [`entropy`](@ref), [`entanglement_spectrum`](@ref)

## Boundary conditions

Converting an infinite MPO into a finite one of a given length, either wrapping it (periodic) or truncating it (open).

[`open_boundary_conditions`](@ref), [`periodic_boundary_conditions`](@ref)

## Utility

Periodic and windowed array containers, virtual/physical space accessors, and a compact "braille" visualization of an MPO's sparsity structure.

[`PeriodicArray`](@ref), [`PeriodicVector`](@ref), [`PeriodicMatrix`](@ref), [`WindowArray`](@ref), [`left_virtualspace`](@ref), [`right_virtualspace`](@ref), [`physicalspace`](@ref), [`braille`](@ref)
