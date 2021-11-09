"
    This object manages the hamiltonian environments for an InfiniteMPS
"
mutable struct MPOHamInfEnv{H<:MPOHamiltonian,V,S<:InfiniteMPS,A} <:AbstractInfEnv
    opp :: H

    dependency :: S
    solver :: A

    lw :: PeriodicArray{V,2}
    rw :: PeriodicArray{V,2}

    lock :: ReentrantLock
end

Base.copy(p::MPOHamInfEnv) = MPOHamInfEnv(p.opp,p.dependency,p.solver,copy(p.lw),copy(p.rw));

function gen_lw_rw(st::InfiniteMPS{A},ham::Union{SparseMPO,MPOHamiltonian}) where {A}
    lw = PeriodicArray{A,2}(undef,length(st),ham.odim)
    rw = PeriodicArray{A,2}(undef,length(st),ham.odim)

    for i = 1:length(st), j = 1:ham.odim
        lw[i,j] = TensorMap(rand,eltype(A),_firstspace(st.AL[i])*ham[i].domspaces[j]',_firstspace(st.AL[i]))
        rw[i,j] = TensorMap(rand,eltype(A),_lastspace(st.AR[i])'*ham[i].imspaces[j]',_lastspace(st.AR[i])')
    end

    return (lw,rw)
end

#randomly initialize envs
function environments(st::InfiniteMPS,ham::MPOHamiltonian; solver=Defaults.solver)
    (lw,rw) = gen_lw_rw(st,ham);
    envs = MPOHamInfEnv(ham,similar(st),solver,lw,rw,ReentrantLock())
    recalculate!(envs,st);
end


function recalculate!(envs::MPOHamInfEnv, nstate)
    sameDspace = reduce((prev,i) -> prev && _lastspace(envs.lw[i,1]) == _firstspace(nstate.CR[i])',1:length(nstate),init=true);

    if !sameDspace
        (envs.lw,envs.rw) = gen_lw_rw(nstate,envs.opp)
    end

    @sync begin
        @Threads.spawn calclw!(envs.lw,nstate,envs.opp; solver=envs.solver)
        @Threads.spawn calcrw!(envs.rw,nstate,envs.opp; solver=envs.solver)
    end

    envs.dependency = nstate;

    envs
end


function calclw!(fixpoints,st::InfiniteMPS,ham::MPOHamiltonian; solver=Defaults.solver)
    len = length(st);


    #the start element
    leftutil = Tensor(ones,eltype(eltype(st)),ham[1].domspaces[1])
    @plansor fixpoints[1,1][-1 -2;-3] = l_LL(st)[-1;-3]*conj(leftutil[-2])
    (len>1) && left_cyclethrough!(1,fixpoints,ham,st)
    for i = 2:ham.odim
        prev = copy(fixpoints[1,i]);

        rmul!(fixpoints[1,i],0);
        left_cyclethrough!(i,fixpoints,ham,st)

        if(isid(ham,i)) #identity matrices; do the hacky renormalization

            #subtract fixpoints
            @plansor tosvec[-1 -2;-3] := fixpoints[1,i][-1 -2;-3]-fixpoints[1,i][1 -2;2]*r_LL(st)[2;1]*l_LL(st)[-1;-3]

            (fixpoints[1,i],convhist) = @closure linsolve(tosvec,prev,solver) do x
                x-transfer_left(x,st.AL,st.AL,rvec=r_LL(st),lvec=l_LL(st))
            end

            convhist.converged==0 && @info "calclw failed to converge $(convhist.normres)"

            (len>1) && left_cyclethrough!(i,fixpoints,ham,st)

            #go through the unitcell, again subtracting fixpoints
            for potato in 1:len
                @plansor fixpoints[potato,i][-1 -2;-3]-=fixpoints[potato,i][1 -2;2]*r_LL(st,potato-1)[2;1]*l_LL(st,potato)[-1;-3]
            end

        else
            if reduce((a,b)->a&&b, [contains(ham[x],i,i) for x in 1:len])

                diag = map(b->b[i,i],ham[:]);
                (fixpoints[1,i],convhist) = @closure linsolve(fixpoints[1,i],prev,solver) do x
                    x-transfer_left(x,diag,st.AL,st.AL)
                end

                convhist.converged==0 && @info "calclw failed to converge $(convhist.normres)"

            end
            (len>1) && left_cyclethrough!(i,fixpoints,ham,st)
        end

    end


    return fixpoints
end

function calcrw!(fixpoints,st::InfiniteMPS,ham::MPOHamiltonian; solver=Defaults.solver)
    len = length(st)

    #the start element
    rightutil = Tensor(ones,eltype(eltype(st)),ham[len].imspaces[1])
    @plansor fixpoints[end,end][-1 -2;-3] = r_RR(st)[-1;-3]*conj(rightutil[-2])
    (len>1) && right_cyclethrough!(ham.odim,fixpoints,ham,st) #populate other sites

    for i = (ham.odim-1):-1:1
        prev = copy(fixpoints[end,i])
        rmul!(fixpoints[end,i],0);
        right_cyclethrough!(i,fixpoints,ham,st)


        if(isid(ham,i)) #identity matrices; do the hacky renormalization

            #subtract fixpoints
            @plansor tosvec[-1 -2;-3]:=fixpoints[end,i][-1 -2;-3]-fixpoints[end,i][1 -2;2]*l_RR(st)[2;1]*r_RR(st)[-1;-3]

            (fixpoints[end,i],convhist) = @closure linsolve(tosvec,prev,solver) do x
                x-transfer_right(x,st.AR,st.AR,lvec=l_RR(st),rvec=r_RR(st))
            end
            convhist.converged==0 && @info "calcrw failed to converge $(convhist.normres)"

            len>1 && right_cyclethrough!(i,fixpoints,ham,st)

            #go through the unitcell, again subtracting fixpoints
            for potatoe in 1:len
                @plansor fixpoints[potatoe,i][-1 -2;-3]-=fixpoints[potatoe,i][1 -2;2]*l_RR(st,potatoe+1)[2;1]*r_RR(st,potatoe)[-1;-3]
            end
        else
            if reduce((a,b)->a&&b, [contains(ham[x],i,i) for x in 1:len])

                diag = map(b->b[i,i],ham[:]);
                (fixpoints[end,i],convhist) = @closure linsolve(fixpoints[end,i],prev,solver) do x
                    x-transfer_right(x,diag,st.AR,st.AR)
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
            contains(ham[i],j,index) || continue

            if isscal(ham[i],j,index)
                fp[i+1,index] += transfer_left(fp[i,j],st.AL[i],st.AL[i])*ham.Os[i,j,index]
            else
                fp[i+1,index] += transfer_left(fp[i,j],ham[i][j,index],st.AL[i],st.AL[i])
            end
        end
    end
end

function right_cyclethrough!(index,fp,ham,st) #see code for explanation
    for i=size(fp,1):(-1):1
        rmul!(fp[i-1,index],0);

        for j=index:ham.odim
            contains(ham[i],index,j) || continue

            if isscal(ham[i],index,j)
                fp[i-1,index] += transfer_right(fp[i,j], st.AR[i], st.AR[i]) * ham.Os[i,index,j]
            else
                fp[i-1,index] += transfer_right(fp[i,j], ham[i][index,j], st.AR[i], st.AR[i])
            end
        end
    end
end
