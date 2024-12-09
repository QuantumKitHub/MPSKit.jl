#=
Idmrg environments are only to be used internally.
They have to be updated manually, without any kind of checks
=#

struct IDMRGEnv{H,V} <: Cache
    operator::H
    lw::PeriodicArray{V,2}
    rw::PeriodicArray{V,2}
end

struct IDMRGEnvironments{O,V} <: Cache
    operator::O
    leftenvs::PeriodicVector{V}
    rightenvs::PeriodicVector{V}
end

function IDMRGEnv(ψ::Union{MPSMultiline,InfiniteMPS}, env)
    ψ === env.dependency || recalculate!(env, ψ)
    return IDMRGEnv(env.operator, deepcopy(env.leftenvs), deepcopy(env.rightenvs))
end

# TODO: change this function name
function IDMRGEnv(ψ::InfiniteMPS, envs::InfiniteEnvironments)
    check_recalculate!(envs, ψ)
    return IDMRGEnvironments(envs.operator,
                             deepcopy(envs.leftenvs),
                             deepcopy(envs.rightenvs))
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

leftenv(envs::IDMRGEnvironments, site::Int) = envs.leftenvs[site]
leftenv(envs::IDMRGEnvironments, site::Int, ::InfiniteMPS) = envs.leftenvs[site]
setleftenv!(envs::IDMRGEnvironments, site::Int, GL) = envs.leftenvs[site] = GL
rightenv(envs::IDMRGEnvironments, site::Int) = envs.rightenvs[site]
rightenv(envs::IDMRGEnvironments, site::Int, ::InfiniteMPS) = envs.rightenvs[site]
setrightenv!(envs::IDMRGEnvironments, site::Int, GR) = envs.rightenvs[site] = GR

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
        return subenv.operator,
               IDMRGEnvironments(subenv.operator, deepcopy(subenv.leftenvs),
                                 deepcopy(subenv.rightenvs))
    end
    Hs, envs = collect.(zip(tmp...))
    return MultipleEnvironments(LazySum(Hs), envs)
end

function update_rightenv!(envs::MultipleEnvironments{<:LazySum,<:IDMRGEnvironments}, st, H,
                          pos::Int)
    for (subH, subenv) in zip(H, envs.envs)
        tm = TransferMatrix(st.AR[pos + 1], subH[pos + 1], st.AR[pos + 1])
        setrightenv!(subenv, pos, tm * rightenv(subenv, pos + 1))
    end
end

function update_leftenv!(envs::MultipleEnvironments{<:LazySum,<:IDMRGEnvironments}, st, H,
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

function update_leftenv!(envs::IDMRGEnvironments, state, O, site::Int)
    T = TransferMatrix(state.AL[site - 1], O[site - 1], state.AL[site - 1])
    return setleftenv!(envs, site, leftenv(envs, site - 1) * T)
end
function update_rightenv!(envs::IDMRGEnvironments, state, O, site::Int)
    T = TransferMatrix(state.AR[site + 1], O[site + 1], state.AR[site + 1])
    return setrightenv!(envs, site, T * rightenv(envs, site + 1))
end
