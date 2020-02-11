"
    Abstract environment for an infinite state
    distinct from finite, because we have to recalculate everything when the state changes
"
abstract type AbstractInfEnv <: Cache end;

function leftenv(pars::AbstractInfEnv,pos::Int,state)
    !(state===pars.dependency) && recalculate!(pars,state);
    pars.lw[pos,:]
end

function rightenv(pars::AbstractInfEnv,pos::Int,state)
    !(state===pars.dependency) && recalculate!(pars,state);
    pars.rw[pos,:]
end

function leftenv(pars::AbstractInfEnv,row::Int,col::Int,state)
    !(state===pars.dependency) && recalculate!(pars,state);
    pars.lw[row,col]
end

function rightenv(pars::AbstractInfEnv,row::Int,col::Int,state)
    !(state===pars.dependency) && recalculate!(pars,state);
    pars.rw[row,col]
end
