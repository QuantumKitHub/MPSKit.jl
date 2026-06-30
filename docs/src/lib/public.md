# [Public API](@id public_api)

This page is the curated, stable public API surface of MPSKit — the symbols that are exported and intended for direct use.
Each entry links to its full docstring in the [Library](@ref lib_index) index.
The category reference pages ([States](@ref lib_states), [Operators](@ref lib_operators), [Ground-state algorithms](@ref lib_groundstate)) group the same docstrings by topic.

!!! note
    Anything not listed here (or marked internal in the [Library](@ref lib_index) index) is
    not part of the public API and may change without notice.

## States

[`FiniteMPS`](@ref), [`InfiniteMPS`](@ref), [`WindowMPS`](@ref), [`MultilineMPS`](@ref)

## Operators and Hamiltonians

[`AbstractMPO`](@ref), [`MPO`](@ref), [`FiniteMPO`](@ref), [`InfiniteMPO`](@ref), [`MultilineMPO`](@ref), [`MPOHamiltonian`](@ref), [`FiniteMPOHamiltonian`](@ref), [`InfiniteMPOHamiltonian`](@ref), [`JordanMPOTensor`](@ref), [`MultipliedOperator`](@ref), [`TimedOperator`](@ref), [`UntimedOperator`](@ref), [`LazySum`](@ref)

## Environments

[`environments`](@ref)

## Ground states and boundaries

[`find_groundstate`](@ref), [`leading_boundary`](@ref), [`approximate`](@ref), [`VUMPS`](@ref), [`VOMPS`](@ref), [`DMRG`](@ref), [`DMRG2`](@ref), [`IDMRG`](@ref), [`IDMRG2`](@ref), [`GradientGrassmann`](@ref)

## Bond dimension

[`changebonds`](@ref), [`OptimalExpand`](@ref), [`RandExpand`](@ref), [`SvdCut`](@ref), [`VUMPSSvdCut`](@ref)

## Time evolution

[`time_evolve`](@ref), [`timestep`](@ref), [`make_time_mpo`](@ref), [`TDVP`](@ref), [`TDVP2`](@ref), [`WI`](@ref), [`WII`](@ref), [`TaylorCluster`](@ref)

## Excitations

[`excitations`](@ref), [`FiniteExcited`](@ref), [`QuasiparticleAnsatz`](@ref), [`ChepigaAnsatz`](@ref), [`ChepigaAnsatz2`](@ref)

## Linear problems and spectral functions

[`propagator`](@ref), [`DynamicalDMRG`](@ref), [`NaiveInvert`](@ref), [`Jeckelmann`](@ref), [`exact_diagonalization`](@ref), [`fidelity_susceptibility`](@ref)

## Observables and analysis

[`expectation_value`](@ref), [`correlator`](@ref), [`variance`](@ref), [`correlation_length`](@ref), [`marek_gap`](@ref), [`transfer_spectrum`](@ref), [`entropy`](@ref), [`entanglement_spectrum`](@ref)

## Boundary conditions

[`open_boundary_conditions`](@ref), [`periodic_boundary_conditions`](@ref)

## Utility

[`PeriodicArray`](@ref), [`PeriodicVector`](@ref), [`PeriodicMatrix`](@ref), [`WindowArray`](@ref), [`left_virtualspace`](@ref), [`right_virtualspace`](@ref), [`physicalspace`](@ref), [`braille`](@ref)
