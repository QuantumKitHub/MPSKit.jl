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
    lw = PeriodicArray{A,2}(undef,ham.odim,length(st))
    rw = PeriodicArray{A,2}(undef,ham.odim,length(st))

    for i = 1:length(st), j = 1:ham.odim
        lw[j,i] = similar(st.AL[1],_firstspace(st.AL[i])*ham[i].domspaces[j]'←_firstspace(st.AL[i]))
        rw[j,i] = similar(st.AL[1],_lastspace(st.AR[i])'*ham[i].imspaces[j]'←_lastspace(st.AR[i])')
    end

    randomize!.(lw);
    randomize!.(rw);

    return (lw,rw)
end

#randomly initialize envs
function environments(st::InfiniteMPS,ham::MPOHamiltonian; solver=Defaults.linearsolver)
    (lw,rw) = gen_lw_rw(st,ham);
    envs = MPOHamInfEnv(ham,similar(st),solver,lw,rw,ReentrantLock())
    recalculate!(envs,st);
end

function leftenv(envs::MPOHamInfEnv,pos::Int,state)
    check_recalculate!(envs,state);
    envs.lw[:,pos]
end

function rightenv(envs::MPOHamInfEnv,pos::Int,state)
    check_recalculate!(envs,state);
    envs.rw[:,pos]
end


function recalculate!(envs::MPOHamInfEnv, nstate)
    sameDspace = reduce(&,_lastspace.(envs.lw[1,:]) .== _firstspace.(nstate.CR))

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


function calclw!(fixpoints,st::InfiniteMPS,ham::MPOHamiltonian; solver=Defaults.linearsolver)
    len = length(st);
    @assert len == length(ham);

    #the start element
    leftutil = similar(st.AL[1],ham[1].domspaces[1]); fill_data!(leftutil,one);

    @plansor fixpoints[1,1][-1 -2;-3] = l_LL(st)[-1;-3]*conj(leftutil[-2])
    (len>1) && left_cyclethrough!(1,fixpoints,ham,st)
    for i = 2:size(fixpoints,1)
        prev = copy(fixpoints[i,1]);

        rmul!(fixpoints[i,1],0);
        left_cyclethrough!(i,fixpoints,ham,st)

        if(isid(ham,i)) #identity matrices; do the hacky renormalization
            tm = regularize(TransferMatrix(st.AL,st.AL),l_LL(st),r_LL(st));
            (fixpoints[i,1],convhist) = linsolve(flip(tm),fixpoints[i,1],prev,solver,1,-1)
            convhist.converged==0 && @info "calclw failed to converge $(convhist.normres)"

            (len>1) && left_cyclethrough!(i,fixpoints,ham,st)

            #go through the unitcell, again subtracting fixpoints
            for potato in 1:len
                @plansor fixpoints[i,potato][-1 -2;-3]-=fixpoints[i,potato][1 -2;2]*r_LL(st,potato-1)[2;1]*l_LL(st,potato)[-1;-3]
            end

        else
            if reduce(&,contains.(ham.data,i,i))

                diag = map(b->b[i,i],ham[:]);
                tm = TransferMatrix(st.AL,diag,st.AL);
                (fixpoints[i,1],convhist) = linsolve(flip(tm),fixpoints[i,1],prev,solver,1,-1)
                convhist.converged==0 && @info "calclw failed to converge $(convhist.normres)"

            end
            (len>1) && left_cyclethrough!(i,fixpoints,ham,st)
        end

    end


    return fixpoints
end

function calcrw!(fixpoints,st::InfiniteMPS,ham::MPOHamiltonian; solver=Defaults.linearsolver)
    len = length(st); odim = size(fixpoints,1);
    @assert len == length(ham);

    #the start element
    rightutil = similar(st.AL[1],ham[len].imspaces[1]); fill_data!(rightutil,one);
    @plansor fixpoints[end,end][-1 -2;-3] = r_RR(st)[-1;-3]*conj(rightutil[-2])
    (len>1) && right_cyclethrough!(odim,fixpoints,ham,st) #populate other sites

    for i = (odim-1):-1:1
        prev = copy(fixpoints[i,end])
        rmul!(fixpoints[i,end],0);
        right_cyclethrough!(i,fixpoints,ham,st)


        if(isid(ham,i)) #identity matrices; do the hacky renormalization

            #subtract fixpoints
            tm = regularize(TransferMatrix(st.AR,st.AR),l_RR(st),r_RR(st));
            (fixpoints[i,end],convhist) = linsolve(tm,fixpoints[i,end],prev,solver,1,-1)
            convhist.converged==0 && @info "calcrw failed to converge $(convhist.normres)"

            len>1 && right_cyclethrough!(i,fixpoints,ham,st)

            #go through the unitcell, again subtracting fixpoints
            for potatoe in 1:len
                @plansor fixpoints[i,potatoe][-1 -2;-3]-=fixpoints[i,potatoe][1 -2;2]*l_RR(st,potatoe+1)[2;1]*r_RR(st,potatoe)[-1;-3]
            end
        else
            if reduce(&, contains.(ham.data,i,i))

                diag = map(b->b[i,i],ham[:]);
                tm = TransferMatrix(st.AR,diag,st.AR);
                (fixpoints[i,end],convhist) = linsolve(tm,fixpoints[i,end],prev,solver,1,-1)
                convhist.converged==0 && @info "calcrw failed to converge $(convhist.normres)"

            end

            (len>1) && right_cyclethrough!(i,fixpoints,ham,st)
        end
    end

    return fixpoints
end

function left_cyclethrough!(index::Int,fp,ham,st) 
    for i=1:size(fp,2)
        rmul!(fp[index,i+1],0);

        for j=index:-1:1
            contains(ham[i],j,index) || continue

            if isscal(ham[i],j,index)
                axpy!(ham.Os[i,j,index],
                    fp[j,i]*TransferMatrix(st.AL[i],st.AL[i]),
                    fp[index,i+1])
            else
                axpy!(true,fp[j,i]*TransferMatrix(st.AL[i],ham[i][j,index],st.AL[i]),fp[index,i+1])
            end
        end
    end
end

function right_cyclethrough!(index::Int,fp,ham,st)
    for i=size(fp,2):(-1):1
        rmul!(fp[index,i-1],0);

        for j=index:size(fp,1)
            contains(ham[i],index,j) || continue

            if isscal(ham[i],index,j)
                axpy!(ham.Os[i,index,j],
                    TransferMatrix(st.AR[i], st.AR[i]) * fp[j,i],fp[index,i-1])
            else
                
                axpy!(true,TransferMatrix(st.AR[i], ham[i][index,j], st.AR[i]) * fp[j,i],fp[index,i-1])
            end
        end
    end
end
