#=
Idmrg environments are only to be used internally.
They have to be updated manually, without any kind of checks
=#
"""
    IDMRGEnvironments{O,V} <: AbstractMPSEnvironments

Environment manager for IDMRG
"""
struct IDMRGEnvironments{O,V} <: AbstractMPSEnvironments
    operator::O
    leftenvs::PeriodicMatrix{V}
    rightenvs::PeriodicMatrix{V}
end

function IDMRGEnvironments(ψ::InfiniteMPS, envs::InfiniteMPOHamiltonianEnvironments)
    check_recalculate!(envs, ψ)
    L = length(ψ)
    leftenvs = PeriodicMatrix(reshape(deepcopy(envs.leftenvs), (1, L)))
    rightenvs = PeriodicMatrix(reshape(deepcopy(envs.rightenvs), (1, L)))
    return IDMRGEnvironments(envs.operator, leftenvs, rightenvs)
end
function IDMRGEnvironments(ψ::Union{InfiniteMPS,MultilineMPS},
                           envs::InfiniteMPOEnvironments)
    check_recalculate!(envs, ψ)
    return IDMRGEnvironments(envs.operator, deepcopy(envs.leftenvs),
                             deepcopy(envs.rightenvs))
end

leftenv(envs::IDMRGEnvironments, site::Int) = envs.leftenvs[site]
leftenv(envs::IDMRGEnvironments, site::Int, ::InfiniteMPS) = envs.leftenvs[site]
leftenv(envs::IDMRGEnvironments, row::Int, col::Int) = envs.leftenvs[row, col]
setleftenv!(envs::IDMRGEnvironments, site::Int, GL) = (envs.leftenvs[site] = GL; envs)
function setleftenv!(envs::IDMRGEnvironments, row::Int, col::Int, GL)
    envs.leftenvs[row, col] = GL
    return envs
end

rightenv(envs::IDMRGEnvironments, site::Int) = envs.rightenvs[site]
rightenv(envs::IDMRGEnvironments, site::Int, ::InfiniteMPS) = envs.rightenvs[site]
rightenv(envs::IDMRGEnvironments, row::Int, col::Int) = envs.rightenvs[row, col]
setrightenv!(envs::IDMRGEnvironments, site::Int, GR) = (envs.rightenvs[site] = GR; envs)
function setrightenv!(envs::IDMRGEnvironments, row::Int, col::Int, GR)
    envs.rightenvs[row, col] = GR
    return envs
end

function update_leftenv!(envs::IDMRGEnvironments, state, O, site::Int)
    T = TransferMatrix(state.AL[site - 1], O[site - 1], state.AL[site - 1])
    return setleftenv!(envs, site, leftenv(envs, site - 1) * T)
end
function update_rightenv!(envs::IDMRGEnvironments, state, O, site::Int)
    T = TransferMatrix(state.AR[site + 1], O[site + 1], state.AR[site + 1])
    return setrightenv!(envs, site, T * rightenv(envs, site + 1))
end
