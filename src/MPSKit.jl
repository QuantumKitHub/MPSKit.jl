module MPSKit
    using TensorKit,KrylovKit,Parameters, Base.Threads,OptimKit

    using LinearAlgebra:diag,Diagonal;
    import LinearAlgebra

    #bells and whistles for mpses
    export InfiniteMPS,FiniteMPS,MPSComoving,PeriodicArray,MPSMultiline
    export transfer_left,transfer_right
    export leftorth,rightorth,leftorth!,rightorth!,poison!,uniform_leftorth,uniform_rightorth
    export r_LL,l_LL,r_RR,l_RR,r_RL,r_LR,l_RL,l_LR #should be properties
    export hamcat

    #useful utility functions?
    export spinmatrices,add_util_leg,full,nonsym_spintensors,nonsym_bosonictensors
    export max_Ds,virtualspace

    #hamiltonian things
    export Hamiltonian,Operator,Cache
    export MPOHamiltonian,contains,PeriodicMPO,ComAct,commutator,anticommutator
    export ac_prime,c_prime,params,ac2_prime,expectation_value,effective_excitation_hamiltonian
    export leftenv,rightenv

    #algos
    export find_groundstate, Vumps, Dmrg, Dmrg2, GradDesc, Idmrg1, Idmrg2, GradientGrassmann
    export leading_boundary, PowerMethod
    export quasiparticle_excitation, correlation_length
    export timestep,Tdvp,Tdvp2
    export splitham,mpo2mps,mps2mpo,infinite_temperature, entanglement_spectrum, transfer_spectrum
    export changebonds,VumpsSvdCut,OptimalExpand,SvdCut,UnionTrunc
    export entropy
    export dynamicaldmrg
    export fidelity_susceptibility

    #default settings
    module Defaults
        const eltype = ComplexF64
        const maxiter = 100
        const tolgauge = 1e-14
        const tol = 1e-12
        const verbose = true
        _finalize(iter,state,opp,pars) = (state,pars,true);
    end

    include("utility/periodicarray.jl")
    include("utility/utility.jl") #random utility functions

    #maybe we should introduce an abstract state type
    include("states/abstractmps.jl")
    include("states/transfer.jl") # mps transfer matrices
    include("states/infinitemps.jl")
    include("states/multiline.jl")
    include("states/finitemps.jl")
    include("states/comoving.jl")
    include("states/orthoview.jl")
    include("states/quasiparticle_state.jl")

    abstract type Operator end
    abstract type Hamiltonian <: Operator end

    include("operators/mpohamiltonian/mpohamiltonian.jl") #the mpohamiltonian objects
    include("operators/umpo.jl")
    include("operators/commutator.jl")

    abstract type Cache end #cache "manages" environments

    include("environments/FinEnv.jl")
    include("environments/abstractinfenv.jl")
    include("environments/permpoinfenv.jl")
    include("environments/mpohaminfenv.jl")
    include("environments/simpleenv.jl")
    include("environments/overlapenv.jl")
    include("environments/qpenv.jl")

    abstract type Algorithm end

    include("algorithms/derivatives.jl")
    include("algorithms/expval.jl")
    include("algorithms/toolbox.jl") #maybe move to utility, or move some utility functions to toolbox?
    include("algorithms/ortho.jl")

    include("algorithms/changebonds/optimalexpand.jl")
    include("algorithms/changebonds/vumpssvd.jl")
    include("algorithms/changebonds/svdcut.jl")
    include("algorithms/changebonds/changebonds.jl")

    include("algorithms/timestep/tdvp.jl")

    include("algorithms/groundstate/vumps.jl")
    include("algorithms/groundstate/idmrg.jl")
    include("algorithms/groundstate/dmrg.jl")
    include("algorithms/groundstate/gradient_grassmann.jl")

    include("algorithms/propagator/corvector.jl")

    include("algorithms/excitation/quasiparticleexcitation.jl")
    include("algorithms/excitation/TM_excitations.jl")

    include("algorithms/statmech/vumps.jl")
    include("algorithms/statmech/power.jl")

    include("algorithms/fidelity_susceptibility.jl")
end
