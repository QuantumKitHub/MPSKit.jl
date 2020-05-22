"
    Abstract environment for an infinite state
    distinct from finite, because we have to recalculate everything when the state changes
"
abstract type AbstractInfEnv <: Cache end;

function leftenv(pars::AbstractInfEnv,pos::Int,state)
    check_recalculate!(pars,state)
    pars.lw[pos,:]
end

function rightenv(pars::AbstractInfEnv,pos::Int,state)
    check_recalculate!(pars,state)
    pars.rw[pos,:]
end

function leftenv(pars::AbstractInfEnv,row::Int,col::Int,state)
    check_recalculate!(pars,state)
    pars.lw[row,col]
end

function rightenv(pars::AbstractInfEnv,row::Int,col::Int,state)
    check_recalculate!(pars,state)
    pars.rw[row,col]
end

function poison!(pars::AbstractInfEnv)
    pars.dependency = similar(pars.dependency);
end

function check_recalculate!(pars::AbstractInfEnv,state)
    if !(pars.dependency === state)
        lock(pars.lock);

        #we have acquired the lock; maybe state has already been updated?
        !(pars.dependency === state) && recalculate!(pars,state)

        unlock(pars.lock);
    end
end
