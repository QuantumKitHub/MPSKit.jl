"
    SimpleEnv does nothing fancy to ensure the correctness of the environments it returns.
    Supports setleftenv! and setrightenv!
    Only used internally (in idmrg); no public constructor is provided
"
struct SimpleEnv{H<:Operator,V} <:Cache
    opp :: H
    lw :: PeriodicArray{V,2}
    rw :: PeriodicArray{V,2}
end

function SimpleEnv(state,envs::AbstractInfEnv)
    lw = similar(envs.lw)
    rw = similar(envs.rw)
    for i = 1:length(state)
        lw[i,:] = leftenv(envs,i,state)
        rw[i,:] = rightenv(envs,i,state)
    end

    return SimpleEnv(envs.opp,lw,rw)
end

leftenv(envs::SimpleEnv,pos::Int,state) = envs.lw[pos,:];
function setleftenv!(envs::SimpleEnv,pos,mps,lw)
    for i in 1:length(lw)
        envs.lw[pos,i] = lw[i]
    end
end

rightenv(envs::SimpleEnv,pos::Int,state) = envs.rw[pos,:];
function setrightenv!(envs::SimpleEnv,pos,mps,rw)
    for i in 1:length(rw)
        envs.rw[pos,i] = rw[i]
    end
end
