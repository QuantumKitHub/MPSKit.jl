module MPSKit

# Public API
# ----------
# utility:
export PeriodicArray, PeriodicVector, PeriodicMatrix
export WindowArray
export left_virtualspace, right_virtualspace, physicalspace
export braille

# states:
export FiniteMPS
export InfiniteMPS
export WindowMPS
export MultilineMPS
export QP, LeftGaugedQP, RightGaugedQP

# operators:
export AbstractMPO
export MPO, FiniteMPO, InfiniteMPO
export MPOHamiltonian, FiniteMPOHamiltonian, InfiniteMPOHamiltonian
export MultilineMPO
export UntimedOperator, TimedOperator, MultipliedOperator, LazySum

# environments:
export environments
export leftenv, rightenv

# algorithms:
export find_groundstate, find_groundstate!
export leading_boundary
export approximate, approximate!
export VUMPS, VOMPS, DMRG, DMRG2, IDMRG, IDMRG2, GradientGrassmann
export excitations
export FiniteExcited, QuasiparticleAnsatz, ChepigaAnsatz, ChepigaAnsatz2
export time_evolve, timestep, timestep!, make_time_mpo
export TDVP, TDVP2, WI, WII, TaylorCluster
export changebonds, changebonds!
export VUMPSSvdCut, OptimalExpand, SvdCut, RandExpand
export propagator
export DynamicalDMRG, NaiveInvert, Jeckelmann
export exact_diagonalization, fidelity_susceptibility

# toolbox:
export expectation_value, correlator, variance
export correlation_length, marek_gap, transfer_spectrum
export entropy, entanglement_spectrum
export open_boundary_conditions, periodic_boundary_conditions
export entanglementplot, transferplot
export r_LL, l_LL, r_RR, l_RR, r_RL, r_LR, l_RL, l_LR # TODO: rename

# unexported
using Compat: @compat
@compat public DynamicTols
@compat public VERBOSE_NONE, VERBOSE_WARN, VERBOSE_CONV, VERBOSE_ITER, VERBOSE_ALL
@compat public IterLog, loginit!, logiter!, logfinish!, logcancel!

# Imports
# -------
using TensorKit
using TensorKit: BraidingTensor
using MatrixAlgebraKit
using MatrixAlgebraKit: TruncationStrategy, PolarViaSVD, LAPACK_SVDAlgorithm
using BlockTensorKit
using BlockTensorKit: TensorMapSumSpace
using TensorOperations
using TensorOperations: AbstractBackend, DefaultBackend, DefaultAllocator
using KrylovKit
using KrylovKit: KrylovAlgorithm
using OptimKit
using Base.Threads
using Base.Iterators
using RecipesBase
using VectorInterface
using Accessors
using HalfIntegers
using DocStringExtensions

using LinearAlgebra: diag, Diagonal
using LinearAlgebra: LinearAlgebra
using Random
using Base: @kwdef, @propagate_inbounds
using LoggingExtras
using OhMyThreads

# Includes
# --------
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
include("operators/jordanmpotensor.jl")
include("operators/mpohamiltonian.jl") # the mpohamiltonian objects
include("operators/ortho.jl")
include("operators/multilinempo.jl")
include("operators/projection.jl")
include("operators/timedependence.jl")
include("operators/multipliedoperator.jl")
include("operators/lazysum.jl")
include("operators/show.jl")

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
include("algorithms/derivatives/derivatives.jl")
include("algorithms/derivatives/mpo_derivatives.jl")
include("algorithms/derivatives/projection_derivatives.jl")
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
include("algorithms/timestep/taylorcluster.jl")
include("algorithms/timestep/wii.jl")
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
include("algorithms/statmech/idmrg.jl")

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
