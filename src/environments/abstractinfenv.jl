"
    Abstract environment for an infinite state
"
abstract type AbstractInfEnv <: Cache end;

leftenv(envs::AbstractInfEnv,pos::Int,state) = envs.lw[pos,:]
rightenv(envs::AbstractInfEnv,pos::Int,state) = envs.rw[pos,:]
leftenv(envs::AbstractInfEnv,row::Int,col::Int,state) = envs.lw[row,col]
rightenv(envs::AbstractInfEnv,row::Int,col::Int,state) = envs.rw[row,col]
