# --- above === below ---
"
    This object manages the periodic mpo environments for an MPSMultiline
"
mutable struct PerMPOInfEnv{H<:PeriodicMPO,V,S<:MPSMultiline} <: AbstractInfEnv
    opp :: H

    dependency :: S
    tol :: Float64
    maxiter :: Int

    lw :: PeriodicArray{V,2}
    rw :: PeriodicArray{V,2}
end

function recalculate!(envs::PerMPOInfEnv,nstate::MPSMultiline{T}) where T
    sameDspace = reduce((prev,i) -> prev && _lastspace(envs.lw[i...]) == _firstspace(nstate.CR[i...])',
        Iterators.product(1:size(nstate,1),1:size(nstate,2)),init=true);

    init = collect(zip(envs.lw[:,1],envs.rw[:,end]))
    if !sameDspace
        init = gen_init_fps(nstate,envs.opp,nstate)
    end

    (envs.lw,envs.rw) = mixed_fixpoints(nstate,envs.opp,nstate,init,tol = envs.tol, maxiter = envs.maxiter);
    envs.dependency = nstate;

    envs
end

environments(state::InfiniteMPS,opp::PeriodicMPO;kwargs...) = environments(convert(MPSMultiline,state),opp;kwargs...);
function environments(state::MPSMultiline{T},mpo::PeriodicMPO;tol = Defaults.tol,maxiter=Defaults.maxiter) where T
    (lw,rw) = mixed_fixpoints(state,mpo,state;tol = tol, maxiter = maxiter)

    PerMPOInfEnv(mpo,state,tol,maxiter,lw,rw)
end

mutable struct MixPerMPOInfEnv{H<:PeriodicMPO,V,S<:MPSMultiline} <: AbstractInfEnv
    opp :: H

    above :: S
    below :: S

    tol :: Float64
    maxiter :: Int

    lw :: PeriodicArray{V,2}
    rw :: PeriodicArray{V,2}
end

function recalculate!(envs::MixPerMPOInfEnv,nstate::MPSMultiline{T}) where T
    sameDspace = reduce((prev,i) -> prev && _firstspace(envs.lw[i...]) == _firstspace(nstate.CR[i...]),
        Iterators.product(1:size(nstate,1),1:size(nstate,2)),init=true);

    init = collect(zip(envs.lw[:,1],envs.rw[:,end]))
    if !sameDspace
        init = gen_init_fps(envs.above,envs.opp,nstate)
    end

    (envs.lw,envs.rw) = mixed_fixpoints(envs.above,envs.opp,nstate,init,tol = envs.tol, maxiter = envs.maxiter);
    envs.below = nstate;

    envs
end

function environments(below::InfiniteMPS,toapprox::Tuple{<:InfiniteMPS,<:PeriodicMPO};kwargs...)
    (above,opp) = toapprox
    environments(convert(MPSMultiline,below),(convert(MPSMultiline,above),opp);kwargs...);
end
function environments(below::MPSMultiline{T},toapprox::Tuple{<:MPSMultiline,<:PeriodicMPO};tol = Defaults.tol,maxiter=Defaults.maxiter) where T
    (above,mpo) = toapprox;
    (lw,rw) = mixed_fixpoints(above,mpo,below;tol = tol, maxiter = maxiter)

    MixPerMPOInfEnv(mpo,above,below,tol,maxiter,lw,rw)
end

# --- utility functions ---

function gen_init_fps(above::MPSMultiline{T},mpo::PeriodicMPO,below::MPSMultiline{T}) where T
    map(1:size(mpo,1)) do cr
        L0::T = TensorMap(rand,eltype(T),space(below.AL[cr,1],1)*space(mpo.opp[cr,1],1)',space(above.AL[cr,1],1))
        R0::T = TensorMap(rand,eltype(T),space(above.AR[cr,1],1)*space(mpo.opp[cr,1],1),space(below.AR[cr,1],1))
        (L0,R0)
    end
end

function mixed_fixpoints(above::MPSMultiline{T},mpo::PeriodicMPO,below::MPSMultiline{T},init = gen_init_fps(above,mpo,below);tol = Defaults.tol, maxiter = Defaults.maxiter) where T
    #sanity check
    (numrows,numcols) = size(above)
    @assert size(above) == size(mpo)
    @assert size(below) == size(mpo);

    lefties = PeriodicArray{T,2}(undef,numrows,numcols);
    righties = PeriodicArray{T,2}(undef,numrows,numcols);

    for cr = 1:numrows
        (L0,R0) = init[cr]

        (_,Ls,convhist) = eigsolve(x-> transfer_left(x,mpo.opp[cr,:],above.AL[cr,:],below.AL[cr+1,:]),L0,1,:LM,Arnoldi(tol = tol,maxiter=maxiter))
        convhist.converged < 1 && @info "left eigenvalue failed to converge $(convhist.normres)"
        (_,Rs,convhist) = eigsolve(x-> transfer_right(x,mpo.opp[cr,:],above.AR[cr,:],below.AR[cr+1,:]),R0,1,:LM,Arnoldi(tol = tol,maxiter=maxiter))
        convhist.converged < 1 && @info "right eigenvalue failed to converge $(convhist.normres)"


        lefties[cr,1] = Ls[1]
        for loc in 2:numcols
            lefties[cr,loc] = transfer_left(lefties[cr,loc-1],mpo.opp[cr,loc-1],above.AL[cr,loc-1],below.AL[cr+1,loc-1])
        end

        renormfact = @tensor Ls[1][1,2,3]*above.CR[cr,0][3,4]*Rs[1][4,2,5]*conj(below.CR[cr+1,0][1,5])

        righties[cr,end] = Rs[1]/sqrt(renormfact);
        lefties[cr,1] /=sqrt(renormfact);

        for loc in numcols-1:-1:1
            righties[cr,loc] = transfer_right(righties[cr,loc+1],mpo.opp[cr,loc+1],above.AR[cr,loc+1],below.AR[cr+1,loc+1])

            renormfact = @tensor lefties[cr,loc+1][1,2,3]*above.CR[cr,loc][3,4]*righties[cr,loc][4,2,5]*conj(below.CR[cr+1,loc][1,5])
            righties[cr,loc]/=sqrt(renormfact)
            lefties[cr,loc+1]/=sqrt(renormfact)
        end
    end

    return (lefties,righties)
end
