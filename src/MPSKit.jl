module MPSKit
    using TensorKit,KrylovKit,Parameters, OptimKit, FastClosures
    using Base.Threads, FLoops, Transducers, FoldsThreads
    using Base.Iterators
    using RecipesBase

    using LinearAlgebra:diag,Diagonal;
    import LinearAlgebra

    #bells and whistles for mpses
    export InfiniteMPS,FiniteMPS,MPSComoving,PeriodicArray,MPSMultiline
    export QP,LeftGaugedQP,RightGaugedQP
    export leftorth,rightorth,leftorth!,rightorth!,poison!,uniform_leftorth,uniform_rightorth
    export r_LL,l_LL,r_RR,l_RR,r_RL,r_LR,l_RL,l_LR #should be properties

    #useful utility functions?
    export add_util_leg,max_Ds,left_virtualspace,right_virtualspace, recalculate!
    export entanglementplot, transferplot

    #hamiltonian things
    export Cache
    export SparseMPO,MPOHamiltonian,DenseMPO,MPOMultiline
    export ∂C,∂AC,∂AC2,environments,expectation_value,effective_excitation_hamiltonian
    export leftenv,rightenv

    #algos
    export find_groundstate!, find_groundstate, VUMPS, DMRG, DMRG2, GradDesc, IDMRG1, IDMRG2, GradientGrassmann
    export leading_boundary
    export excitations,FiniteExcited,QuasiparticleAnsatz
    export marek_gap, correlation_length
    export timestep!,timestep,TDVP,TDVP2,make_time_mpo,WI,WII,TaylorCluster
    export splitham,infinite_temperature, entanglement_spectrum, transfer_spectrum, variance
    export changebonds!,changebonds,VUMPSSvdCut,OptimalExpand,SvdCut,UnionTrunc,RandExpand
    export entropy
    export dynamicaldmrg
    export fidelity_susceptibility
    export approximate!,approximate
    export periodic_boundary_conditions
    export exact_diagonalization

    #transfer matrix
    export TransferMatrix
    export transfer_left,transfer_right

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
        _finalize(iter,state,opp,envs) = (state,envs);

        import KrylovKit: GMRES,Arnoldi
        const linearsolver = GMRES(;tol,maxiter)
        const eigsolver = Arnoldi(;tol,maxiter)
    end

    include("utility/periodicarray.jl")
    include("utility/utility.jl") #random utility functions
    include("utility/plotting.jl")
    include("utility/linearcombination.jl")

    #maybe we should introduce an abstract state type
    include("states/abstractmps.jl")
    include("states/infinitemps.jl")
    include("states/multiline.jl")
    include("states/finitemps.jl")
    include("states/comoving.jl")
    include("states/orthoview.jl")
    include("states/quasiparticle_state.jl")
    include("states/ortho.jl")

    include("operators/densempo.jl")
    include("operators/sparsempo/sparseslice.jl")
    include("operators/sparsempo/sparsempo.jl")
    include("operators/mpohamiltonian.jl") #the mpohamiltonian objects
    include("operators/mpomultiline.jl")
    include("operators/projection.jl")


    include("transfermatrix/transfermatrix.jl")
    include("transfermatrix/transfer.jl")


    abstract type Cache end #cache "manages" environments

    include("environments/FinEnv.jl")
    include("environments/abstractinfenv.jl")
    include("environments/permpoinfenv.jl")
    include("environments/mpohaminfenv.jl")
    include("environments/qpenv.jl")
    include("environments/idmrgenv.jl")
    include("environments/lazylincocache.jl")

    abstract type Algorithm end

    include("algorithms/derivatives.jl")
    include("algorithms/expval.jl")
    include("algorithms/toolbox.jl")
    include("algorithms/grassmann.jl")

    include("algorithms/changebonds/optimalexpand.jl")
    include("algorithms/changebonds/vumpssvd.jl")
    include("algorithms/changebonds/svdcut.jl")
    include("algorithms/changebonds/randexpand.jl")

    include("algorithms/timestep/tdvp.jl")
    include("algorithms/timestep/timeevmpo.jl")

    include("algorithms/groundstate/vumps.jl")
    include("algorithms/groundstate/idmrg.jl")
    include("algorithms/groundstate/dmrg.jl")
    include("algorithms/groundstate/gradient_grassmann.jl")

    include("algorithms/propagator/corvector.jl")

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

end
