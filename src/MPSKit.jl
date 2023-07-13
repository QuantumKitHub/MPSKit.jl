module MPSKit

using TensorKit, KrylovKit, OptimKit, FastClosures
using Base.Threads, FLoops, Transducers, FoldsThreads
using Base.Iterators
using RecipesBase

using LinearAlgebra: diag, Diagonal
using LinearAlgebra: LinearAlgebra
using Base: @kwdef

#bells and whistles for mpses
export InfiniteMPS, FiniteMPS, WindowMPS, MPSMultiline
export PeriodicArray, Window
export MPSTensor
export QP, LeftGaugedQP, RightGaugedQP
export leftorth, rightorth, leftorth!, rightorth!, poison!, uniform_leftorth,
       uniform_rightorth
export r_LL, l_LL, r_RR, l_RR, r_RL, r_LR, l_RL, l_LR #should be properties

#useful utility functions?
export add_util_leg, max_Ds, left_virtualspace, right_virtualspace, recalculate!
export entanglementplot, transferplot

#hamiltonian things
export Cache
export SparseMPO, MPOHamiltonian, DenseMPO, MPOMultiline, UntimedOperator, TimedOperator, SumOfOperators
export ∂C, ∂AC, ∂AC2, environments, expectation_value, effective_excitation_hamiltonian
export leftenv, rightenv

#algos
export find_groundstate!, find_groundstate, leading_boundary
export VUMPS, DMRG, DMRG2, IDMRG1, IDMRG2, GradientGrassmann
export excitations, FiniteExcited, QuasiparticleAnsatz
export marek_gap, correlation_length, correlator
export timestep!, timestep, TDVP, TDVP2, make_time_mpo, WI, WII, TaylorCluster
export ImplicitMidpoint, RK4, Taylor
export splitham, infinite_temperature, entanglement_spectrum, transfer_spectrum, variance
export changebonds!, changebonds, VUMPSSvdCut, OptimalExpand, SvdCut, UnionTrunc, RandExpand
export entropy
export propagator, NaiveInvert, Jeckelmann, DynamicalDMRG
export fidelity_susceptibility
export approximate!, approximate
export periodic_boundary_conditions
export exact_diagonalization

#transfer matrix
export TransferMatrix
export transfer_left, transfer_right

@deprecate virtualspace left_virtualspace # there is a possible ambiguity when C isn't square, necessitating specifying left or right virtualspace
@deprecate params(args...) environments(args...)
@deprecate InfiniteMPO(args...) DenseMPO(args...)

#default settings
module Defaults
const eltype = ComplexF64
const maxiter = 100
const tolgauge = 1e-14
const tol = 1e-12
const verbose = true
_finalize(iter, state, opp, envs) = (state, envs)

import KrylovKit: GMRES, Arnoldi
const linearsolver = GMRES(; tol, maxiter)
const eigsolver = Arnoldi(; tol, maxiter)
end

include("utility/periodicarray.jl")
include("utility/utility.jl") #random utility functions
include("utility/plotting.jl")

#maybe we should introduce an abstract state type
include("states/window.jl")
include("states/abstractmps.jl")
include("states/infinitemps.jl")
include("states/multiline.jl")
include("states/finitemps.jl")
include("states/windowmps.jl")
include("states/orthoview.jl")
include("states/quasiparticle_state.jl")
include("states/ortho.jl")

include("operators/densempo.jl")
include("operators/sparsempo/sparseslice.jl")
include("operators/sparsempo/sparsempo.jl")
include("operators/mpohamiltonian.jl") #the mpohamiltonian objects
include("operators/mpomultiline.jl")
include("operators/projection.jl")
include("operators/multipliedoperator.jl")
include("operators/sumofoperators.jl")

include("transfermatrix/transfermatrix.jl")
include("transfermatrix/transfer.jl")

abstract type Cache end #cache "manages" environments

include("environments/FinEnv.jl")
include("environments/abstractinfenv.jl")
include("environments/permpoinfenv.jl")
include("environments/mpohaminfenv.jl")
include("environments/multipleenv.jl")
include("environments/qpenv.jl")
include("environments/idmrgenv.jl")

abstract type Algorithm end

include("algorithms/derivatives.jl")
include("algorithms/expval.jl")
include("algorithms/toolbox.jl")
include("algorithms/grassmann.jl")
include("algorithms/correlators.jl")

include("algorithms/changebonds/optimalexpand.jl")
include("algorithms/changebonds/vumpssvd.jl")
include("algorithms/changebonds/svdcut.jl")
include("algorithms/changebonds/randexpand.jl")

include("algorithms/timestep/tdvp.jl")
include("algorithms/timestep/windowtdvp.jl")
include("algorithms/timestep/timeevmpo.jl")
include("algorithms/timestep/integrators.jl")

include("algorithms/groundstate/vumps.jl")
include("algorithms/groundstate/idmrg.jl")
include("algorithms/groundstate/dmrg.jl")
include("algorithms/groundstate/gradient_grassmann.jl")
include("algorithms/groundstate/find_groundstate.jl")

include("algorithms/propagator/corvector.jl")

include("algorithms/excitation/excitations.jl")
include("algorithms/excitation/quasiparticleexcitation.jl")
include("algorithms/excitation/dmrgexcitation.jl")
include("algorithms/excitation/exci_transfer_system.jl")

include("algorithms/statmech/vumps.jl")
include("algorithms/statmech/gradient_grassmann.jl")

include("algorithms/fidelity_susceptibility.jl")

include("algorithms/approximate/vomps.jl")
include("algorithms/approximate/fvomps.jl")
include("algorithms/approximate/idmrg.jl")

include("algorithms/ED.jl")

include("algorithms/unionalg.jl")

include("precompile.jl")
_precompile_()

end
