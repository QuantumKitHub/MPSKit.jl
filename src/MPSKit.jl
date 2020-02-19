module MPSKit
    using LinearAlgebra,TensorKit,KrylovKit,Parameters, Base.Threads, TimerOutputs

    #reexport optimkit things
    #export GradientDescent, ConjugateGradient, LBFGS
    #export FletcherReeves, HestenesStiefel, PolakRibierePolyak, HagerZhang, DaiYuan
    #export HagerZhangLineSearch
    export enable_benchmarks, disable_benchmarks, print_timer,reset_timer!

    #bells and whistles for mpses
    export InfiniteMPS,FiniteMPS,FiniteMPO,MPSComoving,PeriodicArray,MPSMultiline
    export transfer_left,transfer_right
    export leftorth,rightorth,leftorth!,rightorth!,poison!,uniform_leftorth,uniform_rightorth
    export r_LL,l_LL,r_RR,l_RR,r_RL,r_LR,l_RL,l_LR #should be properties
    export hamcat

    #useful utility functions?
    export spinmatrices,exp_decomp,add_util_leg,full,nonsym_spintensors

    #hamiltonian things
    export Hamiltonian,Operator,Cache
    export MPOHamiltonian,contains,PeriodicMPO,ComAct,commutator,anticommutator
    export ac_prime,c_prime,params,ac2_prime,expectation_value,effective_excitation_hamiltonian
    export leftenv,rightenv

    #algos
    export find_groundstate, Vumps, Dmrg, Dmrg2, GradDesc, Idmrg1, Idmrg2
    export leading_boundary, PowerMethod
    export quasiparticle_excitation
    export timestep,Tdvp,Tdvp2
    export splitham,mpo2mps,mps2mpo,infinite_temperature
    export changebonds,VumpsSvdCut,DoNothing,OptimalExpand,RandExpand,SvdCut,UnionTrunc
    export managebonds,SimpleManager
    export entropy
    export dynamicaldmrg

    #models
    export nonsym_xxz_ham,nonsym_ising_ham,su2_xxx_ham,nonsym_ising_mpo,u1_xxz_ham,su2u1_grossneveu

    #default settings
    module Defaults
        const eltype = ComplexF64
        const maxiter = 100
        const tolgauge = 1e-14
        const tol = 1e-12
        const verbose = true
    end

    include("customerrors.jl")
    include("benchmarks.jl")

    include("utility/periodicarray.jl")
    include("utility/mps_types.jl")
    include("utility/utility.jl") #random utility functions
    include("utility/apply_transfer.jl") #apply transfer matrices

    #maybe we should introduce an abstract state type
    include("states/umps.jl")
    include("states/multiline.jl")
    include("states/finitemps.jl")
    include("states/finitempo.jl")
    include("states/comoving.jl")

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

    abstract type Algorithm end

    include("algorithms/actions.jl")
    include("algorithms/toolbox.jl") #maybe move to utility, or move some utility functions to toolbox?
    include("algorithms/ortho.jl")

    include("algorithms/changebonds/bondmanage.jl")
    include("algorithms/changebonds/optimalexpand.jl")
    include("algorithms/changebonds/donothing.jl")
    include("algorithms/changebonds/vumpssvd.jl")
    include("algorithms/changebonds/randomexpand.jl")
    include("algorithms/changebonds/svdcut.jl")
    include("algorithms/changebonds/union.jl")

    include("algorithms/timestep/tdvp.jl")

    include("algorithms/groundstate/vumps.jl")
    include("algorithms/groundstate/idmrg.jl")
    include("algorithms/groundstate/dmrg.jl")
    #include("algorithms/groundstate/optimhook.jl")

    include("algorithms/propagator/corvector.jl")

    include("algorithms/excitation/quasiparticleexcitation.jl")

    include("algorithms/statmech/vumps.jl")
    include("algorithms/statmech/power.jl")

    include("models/xxz.jl")
    include("models/ising.jl")
    include("models/grossneveu.jl")
end
