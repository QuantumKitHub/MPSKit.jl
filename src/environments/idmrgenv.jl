#=
Idmrg environments are only to be used internally.
They have to be updated manually, without any kind of checks
=#

struct IDMRGEnv{H,V} <: Cache
    opp::H
    lw::PeriodicArray{V,2}
    rw::PeriodicArray{V,2}
end

function IDMRGEnv(ψ::Union{MPSMultiline,InfiniteMPS}, env)
    ψ === env.dependency || recalculate!(env, ψ)
    return IDMRGEnv(env.opp, deepcopy(env.lw), deepcopy(env.rw))
end

leftenv(envs::IDMRGEnv, pos::Int) = envs.lw[:, pos];
leftenv(envs::IDMRGEnv, row::Int, col::Int) = envs.lw[row, col];
leftenv(envs::IDMRGEnv, pos::Int, ψ::InfiniteMPS) = envs.lw[:, pos];
function setleftenv!(envs::IDMRGEnv, pos, lw)
    return envs.lw[:, pos] = lw[:]
end
function setleftenv!(envs::IDMRGEnv, row, col, val)
    return envs.lw[row, col] = val
end

rightenv(envs::IDMRGEnv, row::Int, col::Int) = envs.rw[row, col];
rightenv(envs::IDMRGEnv, pos::Int) = envs.rw[:, pos];
rightenv(envs::IDMRGEnv, pos::Int, ψ::InfiniteMPS) = envs.rw[:, pos];
function setrightenv!(envs::IDMRGEnv, pos, rw)
    return envs.rw[:, pos] = rw[:]
end
function setrightenv!(envs::IDMRGEnv, row, col, val)
    return envs.rw[row, col] = val
end

# For MultipleEnvironments

function IDMRGEnv(ψ::Union{MPSMultiline,InfiniteMPS}, env::MultipleEnvironments)
    tmp = map(env.envs) do subenv
        ψ === subenv.dependency || recalculate!(subenv, ψ)
        return subenv.opp, IDMRGEnv(subenv.opp, deepcopy(subenv.lw), deepcopy(subenv.rw))
    end
    Hs, envs = collect.(zip(tmp...))
    return MultipleEnvironments(LazySum(Hs), envs)
end

function update_rightenv!(envs::MultipleEnvironments{<:LazySum,<:IDMRGEnv}, st, H,
                          pos::Int)
    for (subH, subenv) in zip(H, envs.envs)
        tm = TransferMatrix(st.AR[pos + 1], subH[pos + 1], st.AR[pos + 1])
        setrightenv!(subenv, pos, tm * rightenv(subenv, pos + 1))
    end
end

function update_leftenv!(envs::MultipleEnvironments{<:LazySum,<:IDMRGEnv}, st, H,
                         pos::Int)
    for (subH, subenv) in zip(H, envs.envs)
        tm = TransferMatrix(st.AL[pos - 1], subH[pos - 1], st.AL[pos - 1])
        setleftenv!(subenv, pos, leftenv(subenv, pos - 1) * tm)
    end
end

function update_rightenv!(envs::IDMRGEnv, st, H, pos::Int)
    tm = TransferMatrix(st.AR[pos + 1], H[pos + 1], st.AR[pos + 1])
    return setrightenv!(envs, pos, tm * rightenv(envs, pos + 1))
end

function update_leftenv!(envs::IDMRGEnv, st, H, pos::Int)
    tm = TransferMatrix(st.AL[pos - 1], H[pos - 1], st.AL[pos - 1])
    return setleftenv!(envs, pos, leftenv(envs, pos - 1) * tm)
end
