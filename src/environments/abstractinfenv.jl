"
    Abstract environment for an infinite state
    distinct from finite, because we have to recalculate everything when the state changes
"
abstract type AbstractInfEnv <: Cache end;

leftenv(envs::AbstractInfEnv,pos::Int,state) = envs.lw[pos,:]
rightenv(envs::AbstractInfEnv,pos::Int,state) = envs.rw[pos,:]
leftenv(envs::AbstractInfEnv,row::Int,col::Int,state) = envs.lw[row,col]
rightenv(envs::AbstractInfEnv,row::Int,col::Int,state) = envs.rw[row,col]
