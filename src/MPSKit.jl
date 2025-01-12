module MPSKit

using TensorKit
using TensorKit: BraidingTensor
using BlockTensorKit
using KrylovKit
using KrylovKit: KrylovAlgorithm
using OptimKit
using Base.Threads
using Base.Iterators
using RecipesBase
using VectorInterface
using Accessors
using HalfIntegers
import TupleTools as TT
using DocStringExtensions

using LinearAlgebra: diag, Diagonal
using LinearAlgebra: LinearAlgebra
using Random
using Base: @kwdef
using LoggingExtras
using OhMyThreads

# bells and whistles for mpses
export InfiniteMPS, FiniteMPS, WindowMPS, MultilineMPS
export PeriodicArray, PeriodicVector, PeriodicMatrix, WindowArray
export MPSTensor
export QP, LeftGaugedQP, RightGaugedQP
export r_LL, l_LL, r_RR, l_RR, r_RL, r_LR, l_RL, l_LR # should be properties

# useful utility functions?
export add_util_leg, max_Ds, recalculate!
export left_virtualspace, right_virtualspace, physicalspace
export entanglementplot, transferplot
export braille

# hamiltonian things
export AbstractMPO
export MPO, FiniteMPO, InfiniteMPO
export MPOHamiltonian, FiniteMPOHamiltonian, InfiniteMPOHamiltonian
export SparseMPO, DenseMPO, MultilineMPO
export UntimedOperator, TimedOperator, MultipliedOperator, LazySum

export ∂C, ∂AC, ∂AC2, environments, expectation_value, effective_excitation_hamiltonian
export leftenv, rightenv

# algos
export find_groundstate!, find_groundstate, leading_boundary
export VUMPS, VOMPS, DMRG, DMRG2, IDMRG1, IDMRG2, GradientGrassmann
export excitations, FiniteExcited, QuasiparticleAnsatz, ChepigaAnsatz, ChepigaAnsatz2
export marek_gap, correlation_length, correlator
export time_evolve, timestep!, timestep
export TDVP, TDVP2, make_time_mpo, WI, WII, TaylorCluster
export entanglement_spectrum, transfer_spectrum, variance
export changebonds!, changebonds, VUMPSSvdCut, OptimalExpand, SvdCut, RandExpand
export entropy
export propagator, NaiveInvert, Jeckelmann, DynamicalDMRG
export fidelity_susceptibility
export approximate!, approximate
export periodic_boundary_conditions, open_boundary_conditions
export exact_diagonalization

# transfer matrix
export TransferMatrix
export transfer_left, transfer_right

include("algorithms/algorithm.jl")

# submodules
include("utility/dynamictols.jl")
using .DynamicTols

include("utility/defaults.jl")
using .Defaults: VERBOSE_NONE, VERBOSE_WARN, VERBOSE_CONV, VERBOSE_ITER, VERBOSE_ALL
include("utility/logging.jl")
using .IterativeLoggers
include("utility/iterativesolvers.jl")

include("utility/periodicarray.jl")
include("utility/windowarray.jl")
include("utility/multiline.jl")
include("utility/utility.jl") # random utility functions
include("utility/plotting.jl")
include("utility/linearcombination.jl")

# maybe we should introduce an abstract state type
include("states/abstractmps.jl")
include("states/infinitemps.jl")
include("states/multilinemps.jl")
include("states/finitemps.jl")
include("states/windowmps.jl")
include("states/orthoview.jl")
include("states/quasiparticle_state.jl")
include("states/ortho.jl")

include("operators/abstractmpo.jl")
include("operators/mpo.jl")
include("operators/mpohamiltonian.jl") # the mpohamiltonian objects
include("operators/multilinempo.jl")
include("operators/projection.jl")
include("operators/timedependence.jl")
include("operators/multipliedoperator.jl")
include("operators/lazysum.jl")

include("transfermatrix/transfermatrix.jl")
include("transfermatrix/transfer.jl")

include("environments/abstract_envs.jl")
include("environments/finite_envs.jl")
include("environments/infinite_envs.jl")
include("environments/multiline_envs.jl")
include("environments/qp_envs.jl")
include("environments/multiple_envs.jl")
include("environments/lazylincocache.jl")

include("algorithms/fixedpoint.jl")
include("algorithms/derivatives.jl")
include("algorithms/expval.jl")
include("algorithms/toolbox.jl")
include("algorithms/grassmann.jl")
include("algorithms/correlators.jl")

include("algorithms/changebonds/changebonds.jl")
include("algorithms/changebonds/optimalexpand.jl")
include("algorithms/changebonds/vumpssvd.jl")
include("algorithms/changebonds/svdcut.jl")
include("algorithms/changebonds/randexpand.jl")

include("algorithms/timestep/tdvp.jl")
include("algorithms/timestep/timeevmpo.jl")
include("algorithms/timestep/integrators.jl")
include("algorithms/timestep/time_evolve.jl")

include("algorithms/groundstate/vumps.jl")
include("algorithms/groundstate/idmrg.jl")
include("algorithms/groundstate/dmrg.jl")
include("algorithms/groundstate/gradient_grassmann.jl")
include("algorithms/groundstate/find_groundstate.jl")

include("algorithms/propagator/corvector.jl")

include("algorithms/excitation/excitations.jl")
include("algorithms/excitation/quasiparticleexcitation.jl")
include("algorithms/excitation/dmrgexcitation.jl")
include("algorithms/excitation/chepigaansatz.jl")
include("algorithms/excitation/exci_transfer_system.jl")

include("algorithms/statmech/leading_boundary.jl")
include("algorithms/statmech/vumps.jl")
include("algorithms/statmech/vomps.jl")
include("algorithms/statmech/gradient_grassmann.jl")

include("algorithms/fidelity_susceptibility.jl")

include("algorithms/approximate/approximate.jl")
include("algorithms/approximate/vomps.jl")
include("algorithms/approximate/fvomps.jl")
include("algorithms/approximate/idmrg.jl")

include("algorithms/ED.jl")

include("algorithms/unionalg.jl")

function __init__()
    Defaults.set_scheduler!()
    return nothing
end

end
