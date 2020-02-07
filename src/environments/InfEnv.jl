"""
    This object manages the environments for an MpsCenterGauged / MpsMultiline
    If the state changes, it recalculates the entire environment
"""
mutable struct InfEnv{H<:Operator,V,S} <:Cache
    opp :: H

    dependency :: S
    tol :: Float64
    maxiter :: Int

    lw :: Periodic{V,2}
    rw :: Periodic{V,2}
end


function leftenv(pars::InfEnv,pos::Int,state::MpsCenterGauged)
    if !(state===pars.dependency)
        pars.dependency = state
        recalculate!(pars);
    end
    pars.lw[pos,:]
end
function leftenv(pars::InfEnv,row::Int,col::Int,state::Union{MpsCenterGauged,MpsMultiline})
    if !(state===pars.dependency)
        pars.dependency = state
        recalculate!(pars);
    end
    pars.lw[row,col]
end

function rightenv(pars::InfEnv,pos::Int,state::MpsCenterGauged)
    if !(state===pars.dependency)
        pars.dependency = state
        recalculate!(pars);
    end
    pars.rw[pos,:]
end
function rightenv(pars::InfEnv,row::Int,col::Int,state::Union{MpsCenterGauged,MpsMultiline})
    if !(state===pars.dependency)
        pars.dependency = state
        recalculate!(pars);
    end
    pars.rw[row,col]
end


function recalculate!(pars::InfEnv{H}) where H<:MpoHamiltonian
    pars.lw = calclw(pars.dependency,pars.opp,pars.lw,tol = pars.tol,maxiter = pars.maxiter)
    pars.rw = calcrw(pars.dependency,pars.opp,pars.rw,tol = pars.tol,maxiter = pars.maxiter)
end

function recalculate!(pars::InfEnv{H}) where H<:PeriodicMpo
    ndat = params(pars.dependency,pars.opp,pars.lw,pars.rw,tol=pars.tol,maxiter=pars.maxiter);

    pars.lw = ndat.lw
    pars.rw = ndat.rw
end

function params(state::MpsCenterGauged,opp::MpoHamiltonian;tol::Float64=Defaults.tol,maxiter::Int=Defaults.maxiter)
    lw = calclw(state,opp, nothing,tol=tol,maxiter=maxiter)
    rw = calcrw(state,opp, nothing,tol=tol,maxiter=maxiter)
    return InfEnv(opp,state,tol,maxiter,lw,rw)
end

function params(state::MpsCenterGauged,opp::PeriodicMpo,prevl = nothing,prevr = nothing;tol = Defaults.tol,maxiter=Defaults.maxiter)
    params(convert(MpsMultiline,state),opp,prevl,prevr,tol=tol,maxiter=maxiter);
end

function params(state::MpsMultiline{T},mpo::PeriodicMpo,prevl = nothing,prevr = nothing;tol = Defaults.tol,maxiter=Defaults.maxiter) where T
    (numrows,numcols) = size(state)
    @assert size(state) == size(mpo)

    lefties = Periodic{T,2}(numrows,numcols);
    righties = Periodic{T,2}(numrows,numcols);

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

    return InfEnv(mpo,state,tol,maxiter,lefties,righties)
end
function calclw(st::MpsCenterGauged,ham::MpoHamiltonian,prevca=nothing;tol::Float64=Defaults.tol,maxiter::Int=Defaults.maxiter)
    len=length(st);alg=GMRES(tol=tol,maxiter=maxiter)
    @assert sanitycheck(ham)
    @assert len == ham.period;

    #the start element
    leftutil=Tensor(I,eltype(st),space(ham[1,1,1],1))
    @tensor leftstart[-1 -2;-3]:=l_LL(st)[-1,-3]*conj(leftutil[-2])

    initguess = Periodic{typeof(leftstart),2}(len,ham.odim)

    if prevca == nothing
        for (i,j) in Iterators.product(1:len,1:ham.odim)
            initguess[i,j]=TensorMap(rand,eltype(st),space(st.AL[i],1)*space(ham[i,j,1],1)',space(st.AL[i],1))
        end
    else
        initguess = oftype(initguess,prevca)
    end

    #initialize the fixpoints array
    fixpoints = Periodic(zero.(initguess));
    fixpoints[1,1]+=leftstart

    (len>1) && left_cyclethrough(1,fixpoints,ham,st)

    for i=2:ham.odim

        left_cyclethrough(i,fixpoints,ham,st)

        if(isid(ham,i)) #identity matrices; do the hacky renormalization

            #summon cthulu
            @tensor tosvec[-1 -2;-3]:=fixpoints[1,i][-1,-2,-3]-fixpoints[1,i][1,-2,2]*r_LL(st)[2,1]*l_LL(st)[-1,-3]

            (fixpoints[1,i],convhist)=linsolve(tosvec,initguess[1,i],alg) do x
                x-mps_apply_transfer_left(x,st.AL[1:len],st.AL[1:len],rvec=r_LL(st),lvec=l_LL(st))
            end
            convhist.converged==0 && @info "calclw failed to converge $(convhist.normres)"

            (len>1) && left_cyclethrough(i,fixpoints,ham,st)

            for potatoe in 1:len
                @tensor fixpoints[potatoe,i][-1 -2;-3]-=fixpoints[potatoe,i][1,-2,2]*r_LL(st,potatoe-1)[2,1]*l_LL(st,potatoe)[-1,-3]
            end

        else #do the obvious thing
            if reduce((a,b)->a&&b, [contains(ham,x,i,i) for x in 1:len])
                (fixpoints[1,i],convhist)=linsolve(fixpoints[1,i],initguess[1,i],alg) do x
                    x-mps_apply_transfer_left(x,[ham[j,i,i] for j in 1:len],st.AL[1:len],st.AL[1:len])
                end
                convhist.converged==0 && @info "calclw failed to converge $(convhist.normres)"

            end
            (len>1) && left_cyclethrough(i,fixpoints,ham,st)
        end

    end


    return fixpoints
end

function calcrw(st::MpsCenterGauged,ham::MpoHamiltonian,prevca=nothing;tol::Float64=Defaults.tol,maxiter::Int=Defaults.maxiter)
    alg=GMRES(tol=tol,maxiter=maxiter);len=length(st)

    @assert sanitycheck(ham)
    @assert len == ham.period;

    #the start element
    rightutil=Tensor(I,eltype(st),space(ham[len,1,1],3))
    @tensor rightstart[-1 -2;-3]:=r_RR(st)[-1,-3]*conj(rightutil[-2])

    #initialize the fixpoints array
    initguess = Periodic{typeof(rightstart),2}(len,ham.odim)
    if prevca == nothing
        for (i,j) in Iterators.product(1:len,1:ham.odim)
            initguess[i,j]=TensorMap(zeros,eltype(st),space(st.AR[i],3)'*space(ham[i,1,j],3)',space(st.AR[i],3)')
        end
    else
        initguess = oftype(initguess,prevca);
    end

    fixpoints = Periodic(zero.(initguess));
    fixpoints[end,end]+=rightstart

    (len>1) && right_cyclethrough(ham.odim,fixpoints,ham,st) #populate other sites

    for i=(ham.odim-1):-1:1

        right_cyclethrough(i,fixpoints,ham,st)

        if(isid(ham,i)) #identity matrices; do the hacky renormalization

            #summon cthulu
            @tensor tosvec[-1 -2;-3]:=fixpoints[end,i][-1,-2,-3]-fixpoints[end,i][1,-2,2]*l_RR(st)[2,1]*r_RR(st)[-1,-3]

            (fixpoints[end,i],convhist)=linsolve(tosvec,initguess[end,i],alg) do x
                x-mps_apply_transfer_right(x,st.AR[1:len],st.AR[1:len],lvec=l_RR(st),rvec=r_RR(st))
            end

            convhist.converged==0 && @info "calcrw failed to converge $(convhist.normres)"

            len>1 && right_cyclethrough(i,fixpoints,ham,st)
            for potatoe in 1:len
                @tensor fixpoints[potatoe,i][-1 -2;-3]-=fixpoints[potatoe,i][1,-2,2]*l_RR(st,potatoe+1)[2,1]*r_RR(st,potatoe)[-1,-3]
            end
        else #do the obvious thing
            if reduce((a,b)->a&&b, [contains(ham,x,i,i) for x in 1:len])

                (fixpoints[end,i],convhist)=linsolve(fixpoints[end,i],initguess[end,i],alg) do x
                    x-mps_apply_transfer_right(x,[ham[j,i,i] for j in 1:len],st.AR[1:len],st.AR[1:len])
                end
                convhist.converged==0 && @info "calcrw failed to converge $(convhist.normres)"

            end

            (len>1) && right_cyclethrough(i,fixpoints,ham,st)
        end
    end

    return fixpoints
end

function left_cyclethrough(index::Int,fp,ham,st) #see code for explanation
    for i=1:size(fp,1)
        fp[mod1(i+1,end),index]=zero(fp[mod1(i+1,end),index])
        for j=index:-1:1
            if contains(ham,i,j,index)
                if j==index && isscal(ham,i,index)
                    fp[mod1(i+1,end),index]+=mps_apply_transfer_left(fp[i,j],st.AL[i],st.AL[i])*ham.scalars[i][index]
                else
                    fp[mod1(i+1,end),index]+=mps_apply_transfer_left(fp[i,j],ham[i,j,index],st.AL[i],st.AL[i])
                end
            end
        end
    end
end

function right_cyclethrough(index,fp,ham,st) #see code for explanation
    for i=size(fp,1):(-1):1
        fp[mod1(i-1,end),index]=zero(fp[mod1(i-1,end),index])
        for j=index:ham.odim
            if contains(ham,i,index,j)
                if j==index && isscal(ham,i,index)
                    fp[mod1(i-1,end),index]+=mps_apply_transfer_right(fp[i,j],st.AR[i],st.AR[i])*ham.scalars[i][index]
                else
                    fp[mod1(i-1,end),index]+=mps_apply_transfer_right(fp[i,j],ham[i,index,j],st.AR[i],st.AR[i])
                end
            end
        end
    end
end
