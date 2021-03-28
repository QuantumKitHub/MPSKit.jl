"
    This object manages the hamiltonian environments for an InfiniteMPS
"
mutable struct MPOHamInfEnv{H<:MPOHamiltonian,V,S<:InfiniteMPS} <:AbstractInfEnv
    opp :: H

    dependency :: S
    tol :: Float64
    maxiter :: Int

    lw :: PeriodicArray{V,2}
    rw :: PeriodicArray{V,2}

    lock :: ReentrantLock
end

Base.copy(p::MPOHamInfEnv) = MPOHamInfEnv(p.opp,p.dependency,p.tol,p.maxiter,copy(p.lw),copy(p.rw));

function gen_lw_rw(st::InfiniteMPS{A,B},ham::MPOHamiltonian) where {A,B}
    lw = PeriodicArray{A,2}(undef,length(st),ham.odim)
    rw = PeriodicArray{A,2}(undef,length(st),ham.odim)

    for i = 1:length(st), j = 1:ham.odim
        lw[i,j] = TensorMap(rand,eltype(eltype(st)),space(st.AL[i],1)*space(ham[i,j,1],1)',space(st.AL[i],1))
        rw[i,j] = TensorMap(rand,eltype(eltype(st)),space(st.AR[i],3)'*space(ham[i,1,j],3)',space(st.AR[i],3)')
    end

    return (lw,rw)
end

#randomly initialize envs
function environments(st::InfiniteMPS,ham::MPOHamiltonian;tol::Float64=Defaults.tol,maxiter::Int=Defaults.maxiter)
    (lw,rw) = gen_lw_rw(st,ham);
    envs = MPOHamInfEnv(ham,similar(st),tol,maxiter,lw,rw,ReentrantLock())
    recalculate!(envs,st);
end


function recalculate!(envs::MPOHamInfEnv, nstate)
    sameDspace = reduce((prev,i) -> prev && _lastspace(envs.lw[i,1]) == _firstspace(nstate.CR[i])',1:length(nstate),init=true);

    if !sameDspace
        (envs.lw,envs.rw) = gen_lw_rw(nstate,envs.opp)
    end

    @sync begin
        @Threads.spawn calclw!(envs.lw,nstate,envs.opp,tol = envs.tol,maxiter = envs.maxiter)
        @Threads.spawn calcrw!(envs.rw,nstate,envs.opp,tol = envs.tol,maxiter = envs.maxiter)
    end

    envs.dependency = nstate;

    envs
end


function calclw!(fixpoints,st::InfiniteMPS,ham::MPOHamiltonian; tol = Defaults.tol, maxiter = Defaults.maxiter)
    len = length(st);
    alg = GMRES(tol=tol,maxiter=maxiter)

    #the start element
    leftutil = Tensor(ones,eltype(eltype(st)),space(ham[1,1,1],1))
    @tensor fixpoints[1,1][-1 -2;-3] = l_LL(st)[-1,-3]*conj(leftutil[-2])
    (len>1) && left_cyclethrough!(1,fixpoints,ham,st)

    for i = 2:ham.odim
        prev = copy(fixpoints[1,i]);

        rmul!(fixpoints[1,i],0);
        left_cyclethrough!(i,fixpoints,ham,st)

        if(isid(ham,i)) #identity matrices; do the hacky renormalization

            #subtract fixpoints
            @tensor tosvec[-1 -2;-3] := fixpoints[1,i][-1,-2,-3]-fixpoints[1,i][1,-2,2]*r_LL(st)[2,1]*l_LL(st)[-1,-3]

            (fixpoints[1,i],convhist) = @closure linsolve(tosvec,prev,alg) do x
                x-transfer_left(x,st.AL,st.AL,rvec=r_LL(st),lvec=l_LL(st))
            end

            convhist.converged==0 && @info "calclw failed to converge $(convhist.normres)"

            (len>1) && left_cyclethrough!(i,fixpoints,ham,st)

            #go through the unitcell, again subtracting fixpoints
            for potato in 1:len
                @tensor fixpoints[potato,i][-1 -2;-3]-=fixpoints[potato,i][1,-2,2]*r_LL(st,potato-1)[2,1]*l_LL(st,potato)[-1,-3]
            end

        else
            if reduce((a,b)->a&&b, [contains(ham,x,i,i) for x in 1:len])

                (fixpoints[1,i],convhist) = @closure linsolve(fixpoints[1,i],prev,alg) do x
                    x-transfer_left(x,[ham[j,i,i] for j in 1:len],st.AL,st.AL)
                end

                convhist.converged==0 && @info "calclw failed to converge $(convhist.normres)"

            end
            (len>1) && left_cyclethrough!(i,fixpoints,ham,st)
        end

    end


    return fixpoints
end

function calcrw!(fixpoints,st::InfiniteMPS,ham::MPOHamiltonian; tol = Defaults.tol, maxiter = Defaults.maxiter)
    len = length(st)
    alg = GMRES(tol=tol,maxiter=maxiter);

    #the start element
    rightutil = Tensor(ones,eltype(eltype(st)),space(ham[len,1,1],3))
    @tensor fixpoints[end,end][-1 -2;-3] = r_RR(st)[-1,-3]*conj(rightutil[-2])

    (len>1) && right_cyclethrough!(ham.odim,fixpoints,ham,st) #populate other sites

    for i = (ham.odim-1):-1:1
        prev = copy(fixpoints[end,i])
        rmul!(fixpoints[end,i],0);
        right_cyclethrough!(i,fixpoints,ham,st)


        if(isid(ham,i)) #identity matrices; do the hacky renormalization

            #subtract fixpoints
            @tensor tosvec[-1 -2;-3]:=fixpoints[end,i][-1,-2,-3]-fixpoints[end,i][1,-2,2]*l_RR(st)[2,1]*r_RR(st)[-1,-3]

            (fixpoints[end,i],convhist) = @closure linsolve(tosvec,prev,alg) do x
                x-transfer_right(x,st.AR,st.AR,lvec=l_RR(st),rvec=r_RR(st))
            end
            convhist.converged==0 && @info "calcrw failed to converge $(convhist.normres)"

            len>1 && right_cyclethrough!(i,fixpoints,ham,st)

            #go through the unitcell, again subtracting fixpoints
            for potatoe in 1:len
                @tensor fixpoints[potatoe,i][-1 -2;-3]-=fixpoints[potatoe,i][1,-2,2]*l_RR(st,potatoe+1)[2,1]*r_RR(st,potatoe)[-1,-3]
            end
        else
            if reduce((a,b)->a&&b, [contains(ham,x,i,i) for x in 1:len])

                (fixpoints[end,i],convhist) = @closure linsolve(fixpoints[end,i],prev,alg) do x
                    x-transfer_right(x,[ham[j,i,i] for j in 1:len],st.AR,st.AR)
                end
                convhist.converged==0 && @info "calcrw failed to converge $(convhist.normres)"

            end

            (len>1) && right_cyclethrough!(i,fixpoints,ham,st)
        end
    end

    return fixpoints
end

function left_cyclethrough!(index::Int,fp,ham,st) #see code for explanation
    for i=1:size(fp,1)
        rmul!(fp[i+1,index],0);

        for j=index:-1:1
            contains(ham,i,j,index) || continue

            if isscal(ham,i,j,index)
                fp[i+1,index] += transfer_left(fp[i,j],st.AL[i],st.AL[i])*ham.Os[i,j,index]
            else
                fp[i+1,index] += transfer_left(fp[i,j],ham[i,j,index],st.AL[i],st.AL[i])
            end
        end
    end
end

function right_cyclethrough!(index,fp,ham,st) #see code for explanation
    for i=size(fp,1):(-1):1
        rmul!(fp[i-1,index],0);

        for j=index:ham.odim
            contains(ham,i,index,j) || continue

            if isscal(ham,i,index,j)
                fp[i-1,index] += transfer_right(fp[i,j], st.AR[i], st.AR[i]) * ham.Os[i,index,j]
            else
                fp[i-1,index] += transfer_right(fp[i,j], ham[i,index,j], st.AR[i], st.AR[i])
            end
        end
    end
end
