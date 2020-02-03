#only implemented for mpscentergauged/multiline ...
#either we change the supertype Hamiltonian to something else, that also fits; or we don't make this type inherit from Hamiltonian
#untested...

struct PeriodicMpo{O<:MpoType} <: Operator
    opp::Periodic{O,2}
end

PeriodicMpo(t::AbstractTensorMap) = PeriodicMpo(fill(t,1,1));
PeriodicMpo(t::Array{T,2}) where T<:TensorMap = PeriodicMpo(Periodic(t));

Base.getindex(o::PeriodicMpo,i,j) = o.opp[i,j]
Base.size(o::PeriodicMpo,i) = size(o.opp,i);
Base.size(o::PeriodicMpo) = size(o.opp);

struct PeriodicMpoCache{T,O} <: Cache
    L::Periodic{T,2}
	R::Periodic{T,2}
    opp::PeriodicMpo{O}
end

params(state::MpsCenterGauged,mpo::PeriodicMpo,prevpars=nothing;tol=Defaults.tol,maxiter=Defaults.maxiter) = params(convert(MpsMultiline,state),mpo,prevpars;tol=tol,maxiter=maxiter)
function params(state::MpsMultiline{T},mpo::PeriodicMpo,prevpars=nothing;tol=Defaults.tol,maxiter=Defaults.maxiter) where T
    (numrows,numcols) = size(state)
    @assert size(state) == size(mpo)

    lefties = Periodic{T,2}(numrows,numcols);
    righties = Periodic{T,2}(numrows,numcols);

    for cr = 1:numrows

        if prevpars==nothing
            Linit=TensorMap(rand,eltype(T),space(state.AL[cr,1],1)*space(mpo.opp[cr,1],1)',space(state.AL[cr,1],1))
            Rinit=TensorMap(rand,eltype(T),space(state.AR[cr,1],1)*space(mpo.opp[cr,1],1),space(state.AR[cr,1],1))
        else
            Linit=prevpars.L[cr,1]
            Rinit=prevpars.R[cr,end]
        end

        #todo; check if this is needed
        L0::T = Linit;
        R0::T = Rinit;

        alg=Arnoldi(tol = tol,maxiter=maxiter)

        (vals,Ls,convhist) = eigsolve(x-> mps_apply_transfer_left(x,mpo.opp[cr,:],state.AL[cr,1:end],state.AL[cr+1,1:end]),L0,1,:LM,alg)
        convhist.converged < 1 && @info "left eigenvalue failed to converge $(convhist.normres)"
        (_,Rs,convhist) = eigsolve(x-> mps_apply_transfer_right(x,mpo.opp[cr,:],state.AR[cr,1:end],state.AR[cr+1,1:end]),R0,1,:LM,alg)
        convhist.converged < 1 && @info "right eigenvalue failed to converge $(convhist.normres)"



        lefties[cr,1] = Ls[1]
        for loc in 2:numcols
            lefties[cr,loc] = mps_apply_transfer_left(lefties[cr,loc-1],mpo.opp[cr,loc-1],state.AL[cr,loc-1],state.AL[cr+1,loc-1])
        end

        renormfact = @tensor Ls[1][1,2,3]*state.CR[cr,0][3,4]*Rs[1][4,2,5]*conj(state.CR[cr+1,0][1,5])

        righties[cr,end] = Rs[1]/sqrt(renormfact);
        lefties[cr,1] /=sqrt(renormfact);

        for loc in numcols-1:-1:1
            righties[cr,loc] = mps_apply_transfer_right(righties[cr,loc+1],mpo.opp[cr,loc+1],state.AR[cr,loc+1],state.AR[cr+1,loc+1])

            renormfact = @tensor lefties[cr,loc+1][1,2,3]*state.CR[cr,loc][3,4]*righties[cr,loc][4,2,5]*conj(state.CR[cr+1,loc][1,5])
            righties[cr,loc]/=sqrt(renormfact)
            lefties[cr,loc+1]/=sqrt(renormfact)
        end

    end

    return PeriodicMpoCache(lefties,righties,mpo)
end

leftenv(pars::PeriodicMpoCache,row,col)=pars.L[row,col]
rightenv(pars::PeriodicMpoCache,row,col)=pars.R[row,col]

ac_prime(x::TensorMap, row,col,mps, pars::PeriodicMpoCache)=@tensor toret[-1 -2;-3]:=leftenv(pars,row,col)[-1,2,1]*x[1,3,4]*(pars.opp[row,col])[2,-2,5,3]*rightenv(pars,row,col)[4,5,-3]
ac2_prime(x::TensorMap, row,col,mps, pars::PeriodicMpoCache)=@tensor toret[-1 -2;-3 -4]:=leftenv(pars,row,col)[-1,2,1]*x[1,3,4,5]*(pars.opp[row,col])[2,-2,6,3]*(pars.opp[row,col+1])[6,-3,7,4]*rightenv(pars,row,col+1)[5,7,-4]
c_prime(x::TensorMap, row,col, mps, pars::PeriodicMpoCache)=@tensor toret[-1;-2]:=leftenv(pars,row,col+1)[-1,3,1]*x[1,2]*rightenv(pars,row,col)[2,3,-2]
