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

function recalculate!(pars::PerMPOInfEnv,nstate)
    ndat = params(nstate,pars.opp,pars.lw,pars.rw,tol=pars.tol,maxiter=pars.maxiter);

    pars.lw = ndat.lw
    pars.rw = ndat.rw
    pars.dependency = ndat.dependency;
end

function params(state::InfiniteMPS,opp::PeriodicMPO,prevl = nothing,prevr = nothing;tol = Defaults.tol,maxiter=Defaults.maxiter)
    params(convert(MPSMultiline,state),opp,prevl,prevr,tol=tol,maxiter=maxiter);
end

function params(state::MPSMultiline{T},mpo::PeriodicMPO,prevl = nothing,prevr = nothing;tol = Defaults.tol,maxiter=Defaults.maxiter) where T
    (numrows,numcols) = size(state)
    @assert size(state) == size(mpo)

    lefties = PeriodicArray{T,2}(undef,numrows,numcols);
    righties = PeriodicArray{T,2}(undef,numrows,numcols);

    for cr = 1:numrows

        L0::T = TensorMap(rand,eltype(T),space(state.AL[cr,1],1)*space(mpo.opp[cr,1],1)',space(state.AL[cr,1],1))
        if prevl != nothing
            L0 = prevl[cr,1]
        end

        R0::T=TensorMap(rand,eltype(T),space(state.AR[cr,1],1)*space(mpo.opp[cr,1],1),space(state.AR[cr,1],1))
        if prevr != nothing
            R0 = prevr[cr,end]
        end

        alg=Arnoldi(tol = tol,maxiter=maxiter)

        (vals,Ls,convhist) = eigsolve(x-> transfer_left(x,mpo.opp[cr,:],state.AL[cr,1:end],state.AL[cr+1,1:end]),L0,1,:LM,alg)
        convhist.converged < 1 && @info "left eigenvalue failed to converge $(convhist.normres)"
        (_,Rs,convhist) = eigsolve(x-> transfer_right(x,mpo.opp[cr,:],state.AR[cr,1:end],state.AR[cr+1,1:end]),R0,1,:LM,alg)
        convhist.converged < 1 && @info "right eigenvalue failed to converge $(convhist.normres)"


        lefties[cr,1] = Ls[1]
        for loc in 2:numcols
            lefties[cr,loc] = transfer_left(lefties[cr,loc-1],mpo.opp[cr,loc-1],state.AL[cr,loc-1],state.AL[cr+1,loc-1])
        end

        renormfact = @tensor Ls[1][1,2,3]*state.CR[cr,0][3,4]*Rs[1][4,2,5]*conj(state.CR[cr+1,0][1,5])

        righties[cr,end] = Rs[1]/sqrt(renormfact);
        lefties[cr,1] /=sqrt(renormfact);

        for loc in numcols-1:-1:1
            righties[cr,loc] = transfer_right(righties[cr,loc+1],mpo.opp[cr,loc+1],state.AR[cr,loc+1],state.AR[cr+1,loc+1])

            renormfact = @tensor lefties[cr,loc+1][1,2,3]*state.CR[cr,loc][3,4]*righties[cr,loc][4,2,5]*conj(state.CR[cr+1,loc][1,5])
            righties[cr,loc]/=sqrt(renormfact)
            lefties[cr,loc+1]/=sqrt(renormfact)
        end

    end

    return PerMPOInfEnv(mpo,state,tol,maxiter,lefties,righties)
end
