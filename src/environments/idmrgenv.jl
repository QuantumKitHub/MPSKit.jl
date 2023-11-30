#=
Idmrg environments are only to be used internally.
They have to be updated manually, without any kind of checks
=#

struct IDMRGEnv{H,V}
    opp::H
    lw::PeriodicArray{V,2}
    rw::PeriodicArray{V,2}
end

function IDMRGEnv(Ψ::Union{MPSMultiline,InfiniteMPS}, env)
    Ψ === env.dependency || recalculate!(env, Ψ)
    return IDMRGEnv(env.opp, deepcopy(env.lw), deepcopy(env.rw))
end

leftenv(envs::IDMRGEnv, pos::Int) = envs.lw[:, pos];
leftenv(envs::IDMRGEnv, row::Int, col::Int) = envs.lw[row, col];
leftenv(envs::IDMRGEnv, pos::Int, Ψ::InfiniteMPS) = envs.lw[:, pos];
function setleftenv!(envs::IDMRGEnv, pos, lw)
    return envs.lw[:, pos] = lw[:]
end
function setleftenv!(envs::IDMRGEnv, row, col, val)
    return envs.lw[row, col] = val
end

rightenv(envs::IDMRGEnv, row::Int, col::Int) = envs.rw[row, col];
rightenv(envs::IDMRGEnv, pos::Int) = envs.rw[:, pos];
rightenv(envs::IDMRGEnv, pos::Int, Ψ::InfiniteMPS) = envs.rw[:, pos];
function setrightenv!(envs::IDMRGEnv, pos, rw)
    return envs.rw[:, pos] = rw[:]
end
function setrightenv!(envs::IDMRGEnv, row, col, val)
    return envs.rw[row, col] = val
end

# For MultipleEnvironments

function IDMRGEnv(Ψ::Union{MPSMultiline,InfiniteMPS}, env::MultipleEnvironments)
    tmp = map(env.envs) do subenv
        Ψ === subenv.dependency || recalculate!(subenv, Ψ)
        (subenv.opp, IDMRGEnv(subenv.opp, deepcopy(subenv.lw), deepcopy(subenv.rw)))
    end
    hams, envs = collect.(zip(tmp...))
    return MultipleEnvironments(LazySum(hams), envs)
end

function update_rightenv!(
    envs::MultipleEnvironments{<:LazySum,<:IDMRGEnv}, st, ham, pos::Int
)
    for (subham, subenv) in zip(ham, envs.envs)
        tm = TransferMatrix(st.AR[pos + 1], subham[pos + 1], st.AR[pos + 1])
        setrightenv!(subenv, pos, tm * rightenv(subenv, pos + 1))
    end
end

function update_leftenv!(
    envs::MultipleEnvironments{<:LazySum,<:IDMRGEnv}, st, ham, pos::Int
)
    for (subham, subenv) in zip(ham, envs.envs)
        tm = TransferMatrix(st.AL[pos], subham[pos], st.AL[pos])
        setleftenv!(subenv, pos, leftenv(subenv, pos-1) * tm)
    end
end

function update_rightenv!(envs::IDMRGEnv, st, ham, pos::Int)
    tm = TransferMatrix(st.AR[pos + 1], ham[pos + 1], st.AR[pos + 1])
    return setrightenv!(envs, pos, tm * rightenv(envs, pos + 1))
end

function update_leftenv!(envs::IDMRGEnv, st, ham, pos::Int)
    tm = TransferMatrix(st.AL[pos], ham[pos], st.AL[pos])
    return setleftenv!(envs, pos, leftenv(envs, pos-1) * tm)
end
