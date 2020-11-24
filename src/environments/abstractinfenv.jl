"
    Abstract environment for an infinite state
"
abstract type AbstractInfEnv <: Cache end;

function leftenv(envs::AbstractInfEnv,pos::Int,state)
    check_recalculate!(envs,state);
    envs.lw[pos,:]
end

function rightenv(envs::AbstractInfEnv,pos::Int,state)
    check_recalculate!(envs,state);
    envs.rw[pos,:]
end

function leftenv(envs::AbstractInfEnv,row::Int,col::Int,state)
    check_recalculate!(envs,state);
    envs.lw[row,col]
end

function rightenv(envs::AbstractInfEnv,row::Int,col::Int,state)
    check_recalculate!(envs,state);
    envs.rw[row,col]
end

function check_recalculate!(envs,state)
    if !(envs.dependency === state)
        #acquire the lock
        lock(envs) do
            if !(envs.dependency === state)
                recalculate!(envs,state);
            end
        end
    end

    return envs;
end

Base.lock(fun::Function,env::AbstractInfEnv) = lock(fun,env.lock)
Base.lock(env::AbstractInfEnv) = lock(env.lock);
Base.unlock(env::AbstractInfEnv) = unlock(env.lock);
