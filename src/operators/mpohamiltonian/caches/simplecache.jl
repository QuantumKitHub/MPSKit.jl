#=
    A simple cache object. Does not do anything fancy, simply store the relevant left and right environments
    Used for MpsCenterGauged, where the initialization happens by solving a linear problem
    Also used in the idmrg algorithms
=#
struct SimpleCache{H<:MpoHamiltonian,V} <:Cache
    ham :: H
    lw :: Array{V,2}
    rw :: Array{V,2}
end


leftenv(pars::SimpleCache,pos,mps) = pars.lw[mod1(pos,pars.ham.period)::Int,:]
rightenv(pars::SimpleCache,pos,mps) = pars.rw[mod1(pos,pars.ham.period)::Int,:]
function setleftenv!(pars::SimpleCache,pos,mps,lw)
    for i in 1:length(lw)
        pars.lw[mod1(pos,pars.ham.period)::Int,i] = lw[i]
    end
end

function setrightenv!(pars::SimpleCache,pos,mps,rw)
    for i in 1:length(rw)
        pars.rw[mod1(pos,pars.ham.period)::Int,i] = rw[i]
    end
end

function params(state::MpsCenterGauged,ham::MpoHamiltonian,prevpars=nothing;tol::Float64=Defaults.tol,maxiter::Int=Defaults.maxiter)
    lw=calclw(state,ham,(prevpars==nothing) ? nothing : prevpars.lw,tol=tol,maxiter=maxiter)
    rw=calcrw(state,ham,(prevpars==nothing) ? nothing : prevpars.rw,tol=tol,maxiter=maxiter)
    return SimpleCache(ham,lw,rw)
end

function calclw(st::MpsCenterGauged,ham::MpoHamiltonian,prevca=nothing;tol::Float64=Defaults.tol,maxiter::Int=Defaults.maxiter)
    len=length(st);alg=GMRES(tol=tol,maxiter=maxiter)
    @assert sanitycheck(ham)
    @assert len == ham.period;

    #the start element
    leftutil=Tensor(I,eltype(st),space(ham[1,1,1],1))
    @tensor leftstart[-1 -2;-3]:=l_LL(st)[-1,-3]*conj(leftutil[-2])

    initguess = Array{typeof(leftstart)}(undef,len,ham.odim)

    if prevca == nothing
        for (i,j) in Iterators.product(1:len,1:ham.odim)
            initguess[i,j]=TensorMap(rand,eltype(st),space(st.AL[i],1)*space(ham[i,j,1],1)',space(st.AL[i],1))
        end
    else
        initguess = oftype(initguess,prevca)
    end

    #initialize the fixpoints array
    fixpoints = zero.(initguess);
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
    initguess=Array{typeof(rightstart)}(undef,len,ham.odim)
    if prevca == nothing
        for (i,j) in Iterators.product(1:len,1:ham.odim)
            initguess[i,j]=TensorMap(zeros,eltype(st),space(st.AR[i],3)'*space(ham[i,1,j],3)',space(st.AR[i],3)')
        end
    else
        initguess = oftype(initguess,prevca);
    end

    fixpoints = zero.(initguess);
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
