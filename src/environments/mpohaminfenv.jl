"
    This object manages the hamiltonian environments for an MpsCenterGauged
"
mutable struct MpoHamInfEnv{H<:MpoHamiltonian,V,S<:MpsCenterGauged} <:AbstractInfEnv
    opp :: H

    dependency :: S
    tol :: Float64
    maxiter :: Int

    lw :: Periodic{V,2}
    rw :: Periodic{V,2}
end

#randomly initialize pars
function params(st::MpsCenterGauged,ham::MpoHamiltonian;tol::Float64=Defaults.tol,maxiter::Int=Defaults.maxiter)
    lw = Periodic{typeof(st.AL[1]),2}(length(st),ham.odim)
    rw = Periodic{typeof(st.AL[1]),2}(length(st),ham.odim)

    for (i,j) in Iterators.product(1:length(st),1:ham.odim)
        lw[i,j] = TensorMap(rand,eltype(st),space(st.AL[i],1)*space(ham[i,j,1],1)',space(st.AL[i],1))
        rw[i,j] = TensorMap(rand,eltype(st),space(st.AR[i],3)'*space(ham[i,1,j],3)',space(st.AR[i],3)')
    end

    return MpoHamInfEnv(ham,similar(st),tol,maxiter,lw,rw)
end


function recalculate!(pars::MpoHamInfEnv, nstate)
    pars.dependency = nstate;
    pars.lw = calclw(pars.dependency,pars.opp,pars.lw,tol = pars.tol,maxiter = pars.maxiter)
    pars.rw = calcrw(pars.dependency,pars.opp,pars.rw,tol = pars.tol,maxiter = pars.maxiter)
end


function calclw(st::MpsCenterGauged,ham::MpoHamiltonian,prevca;tol::Float64=Defaults.tol,maxiter::Int=Defaults.maxiter)
    len=length(st);alg=GMRES(tol=tol,maxiter=maxiter)

    @assert sanitycheck(ham)
    @assert len == ham.period;

    #the start element
    leftutil = Tensor(ones,eltype(st),space(ham[1,1,1],1))
    @tensor leftstart[-1 -2;-3]:=l_LL(st)[-1,-3]*conj(leftutil[-2])

    #initialize the fixpoints array
    fixpoints = Periodic(zero.(prevca));
    fixpoints[1,1] += leftstart

    (len>1) && left_cyclethrough(1,fixpoints,ham,st)

    for i = 2:ham.odim

        left_cyclethrough(i,fixpoints,ham,st)

        if(isid(ham,i)) #identity matrices; do the hacky renormalization

            #summon cthulu
            @tensor tosvec[-1 -2;-3]:=fixpoints[1,i][-1,-2,-3]-fixpoints[1,i][1,-2,2]*r_LL(st)[2,1]*l_LL(st)[-1,-3]

            (fixpoints[1,i],convhist)=linsolve(tosvec,prevca[1,i],alg) do x
                x-mps_apply_transfer_left(x,st.AL[1:len],st.AL[1:len],rvec=r_LL(st),lvec=l_LL(st))
            end
            convhist.converged==0 && @info "calclw failed to converge $(convhist.normres)"

            (len>1) && left_cyclethrough(i,fixpoints,ham,st)

            for potatoe in 1:len
                @tensor fixpoints[potatoe,i][-1 -2;-3]-=fixpoints[potatoe,i][1,-2,2]*r_LL(st,potatoe-1)[2,1]*l_LL(st,potatoe)[-1,-3]
            end

        else #do the obvious thing
            if reduce((a,b)->a&&b, [contains(ham,x,i,i) for x in 1:len])
                (fixpoints[1,i],convhist)=linsolve(fixpoints[1,i],prevca[1,i],alg) do x
                    x-mps_apply_transfer_left(x,[ham[j,i,i] for j in 1:len],st.AL[1:len],st.AL[1:len])
                end
                convhist.converged==0 && @info "calclw failed to converge $(convhist.normres)"

            end
            (len>1) && left_cyclethrough(i,fixpoints,ham,st)
        end

    end


    return fixpoints
end

function calcrw(st::MpsCenterGauged,ham::MpoHamiltonian,prevca;tol::Float64=Defaults.tol,maxiter::Int=Defaults.maxiter)
    alg=GMRES(tol=tol,maxiter=maxiter);len=length(st)

    @assert sanitycheck(ham)
    @assert len == ham.period;

    #the start element
    rightutil=Tensor(ones,eltype(st),space(ham[len,1,1],3))
    @tensor rightstart[-1 -2;-3]:=r_RR(st)[-1,-3]*conj(rightutil[-2])

    #initialize the fixpoints array
    fixpoints = Periodic(zero.(prevca));
    fixpoints[end,end]+=rightstart

    (len>1) && right_cyclethrough(ham.odim,fixpoints,ham,st) #populate other sites

    for i=(ham.odim-1):-1:1

        right_cyclethrough(i,fixpoints,ham,st)

        if(isid(ham,i)) #identity matrices; do the hacky renormalization

            #summon cthulu
            @tensor tosvec[-1 -2;-3]:=fixpoints[end,i][-1,-2,-3]-fixpoints[end,i][1,-2,2]*l_RR(st)[2,1]*r_RR(st)[-1,-3]

            (fixpoints[end,i],convhist)=linsolve(tosvec,prevca[end,i],alg) do x
                x-mps_apply_transfer_right(x,st.AR[1:len],st.AR[1:len],lvec=l_RR(st),rvec=r_RR(st))
            end

            convhist.converged==0 && @info "calcrw failed to converge $(convhist.normres)"

            len>1 && right_cyclethrough(i,fixpoints,ham,st)
            for potatoe in 1:len
                @tensor fixpoints[potatoe,i][-1 -2;-3]-=fixpoints[potatoe,i][1,-2,2]*l_RR(st,potatoe+1)[2,1]*r_RR(st,potatoe)[-1,-3]
            end
        else #do the obvious thing
            if reduce((a,b)->a&&b, [contains(ham,x,i,i) for x in 1:len])

                (fixpoints[end,i],convhist)=linsolve(fixpoints[end,i],prevca[end,i],alg) do x
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
        fp[i+1,index] *= 0;

        for j=index:-1:1
            if contains(ham,i,j,index)
                if j==index && isscal(ham,i,index)
                    fp[i+1,index] += mps_apply_transfer_left(fp[i,j],st.AL[i],st.AL[i])*ham.scalars[i][index]
                else
                    fp[i+1,index] += mps_apply_transfer_left(fp[i,j],ham[i,j,index],st.AL[i],st.AL[i])
                end
            end
        end
    end
end

function right_cyclethrough(index,fp,ham,st) #see code for explanation
    for i=size(fp,1):(-1):1
        fp[i-1,index] *= 0;
        for j=index:ham.odim
            if contains(ham,i,index,j)
                if j==index && isscal(ham,i,index)
                    fp[i-1,index] += mps_apply_transfer_right(fp[i,j], st.AR[i], st.AR[i]) * ham.scalars[i][index]
                else
                    fp[i-1,index] += mps_apply_transfer_right(fp[i,j], ham[i,index,j], st.AR[i], st.AR[i])
                end
            end
        end
    end
end
