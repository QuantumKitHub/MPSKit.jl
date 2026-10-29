# Based on the design of GPUArrays.jl

"""
    TestSuite

Suite of tests that can be run for any array type `AT` (e.g. `Vector`, `CuVector`, ...).

Every test file is included into its own submodule, so that the parameters a file
defines at top level stay private to that file instead of clobbering the identically
named parameters of its neighbours. Each file exports the test entry points it defines,
which `TestSuite` re-exports in turn, so that callers can use `TestSuite.some_test(AT)`.

Every entry point has the signature `some_test(AT; fast = false)`, where `fast` selects
a reduced set of cases.
"""
module TestSuite

# shared helpers live in `TestSetup`, which the test runner loads into `Main`
isdefined(Main, :TestSetup) ||
    Base.include(Main, joinpath(@__DIR__, "..", "setup", "testsetup.jl"))
using Main: TestSetup


# Algorithms
# ----------
module AlgorithmsAnyonicBraiding
    include("algorithms/anyonic_braiding.jl")
end
using .AlgorithmsAnyonicBraiding

module AlgorithmsChangebonds
    include("algorithms/changebonds.jl")
end
using .AlgorithmsChangebonds

module AlgorithmsCorrelators
    include("algorithms/correlators.jl")
end
using .AlgorithmsCorrelators

module AlgorithmsFidelitySusceptibility
    include("algorithms/fidelity_susceptibility.jl")
end
using .AlgorithmsFidelitySusceptibility

module AlgorithmsPeriodicBoundary
    include("algorithms/periodic_boundary.jl")
end
using .AlgorithmsPeriodicBoundary

module AlgorithmsPropagator
    include("algorithms/propagator.jl")
end
using .AlgorithmsPropagator


# Excitations
# -----------
module ExcitationsExcitations
    include("excitations/excitations.jl")
end
using .ExcitationsExcitations


# GroundState
# -----------
module GroundStateApproximate
    include("groundstate/approximate.jl")
end
using .GroundStateApproximate

module GroundStateGroundstate
    include("groundstate/groundstate.jl")
end
using .GroundStateGroundstate


# Hamiltonian
# -----------
module HamiltonianDerivatives
    include("hamiltonian/derivatives.jl")
end
using .HamiltonianDerivatives

module HamiltonianFinite
    include("hamiltonian/finite.jl")
end
using .HamiltonianFinite

module HamiltonianInfinite
    include("hamiltonian/infinite.jl")
end
using .HamiltonianInfinite

module HamiltonianMixed
    include("hamiltonian/mixed.jl")
end
using .HamiltonianMixed


# MPO
# ---
module MPODenseMPO
    include("mpo/dense_mpo.jl")
end
using .MPODenseMPO

module MPOFinite
    include("mpo/finite.jl")
end
using .MPOFinite

module MPOInfinite
    include("mpo/infinite.jl")
end
using .MPOInfinite

module MPOMixed
    include("mpo/mixed.jl")
end
using .MPOMixed

module MPOMultiline
    include("mpo/multiline.jl")
end
using .MPOMultiline


# Operators
# ---------
module OperatorsLazysum
    include("operators/lazysum.jl")
end
using .OperatorsLazysum

module OperatorsProjection
    include("operators/projection.jl")
end
using .OperatorsProjection

module OperatorsWindow
    include("operators/window.jl")
end
using .OperatorsWindow


# States
# ------
module StatesArithmetic
    include("states/arithmetic.jl")
end
using .StatesArithmetic

module StatesFinitemps
    include("states/finitemps.jl")
end
using .StatesFinitemps

module StatesInfinitemps
    include("states/infinitemps.jl")
end
using .StatesInfinitemps

module StatesMultilinemps
    include("states/multilinemps.jl")
end
using .StatesMultilinemps

module StatesQuasiparticle
    include("states/quasiparticle.jl")
end
using .StatesQuasiparticle

module StatesWindowmps
    include("states/windowmps.jl")
end
using .StatesWindowmps


# StatMech
# --------
module StatMechStatmech
    include("statmech/statmech.jl")
end
using .StatMechStatmech


# Symmetries
# ----------
module SymmetriesMultifusion
    include("symmetries/multifusion.jl")
end
using .SymmetriesMultifusion

module SymmetriesSectorConventions
    include("symmetries/sector_conventions.jl")
end
using .SymmetriesSectorConventions


# TimeEvolution
# -------------
module TimeEvolutionClusterexpansion
    include("timeevolution/clusterexpansion.jl")
end
using .TimeEvolutionClusterexpansion

module TimeEvolutionTaylorcluster
    include("timeevolution/taylorcluster.jl")
end
using .TimeEvolutionTaylorcluster

module TimeEvolutionTimestep
    include("timeevolution/timestep.jl")
end
using .TimeEvolutionTimestep

end
