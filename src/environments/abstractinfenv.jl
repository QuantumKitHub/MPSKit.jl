"
    Abstract environment for an infinite state
    distinct from finite, because we have to recalculate everything when the state changes
"
abstract type AbstractInfEnv <: Cache end;

leftenv(pars::AbstractInfEnv,pos::Int,state) = pars.lw[pos,:]
rightenv(pars::AbstractInfEnv,pos::Int,state) = pars.rw[pos,:]
leftenv(pars::AbstractInfEnv,row::Int,col::Int,state) = pars.lw[row,col]
rightenv(pars::AbstractInfEnv,row::Int,col::Int,state) = pars.rw[row,col]
