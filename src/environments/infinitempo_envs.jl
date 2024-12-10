"""
    InfiniteMPOEnvironments{O<:MPOMultiline,V,S<:MPSMultiline,A} <: AbstractMPSEnvironments

Environment manager for `InfiniteMPO` and its `Multiline` version.
"""
mutable struct InfiniteMPOEnvironments{O,V,S<:MPSMultiline,A} <:
               AbstractInfiniteEnvironments
    above::Union{S,Nothing}
    operator::O

    dependency::S
    solver::A

    leftenvs::PeriodicMatrix{V}
    rightenvs::PeriodicMatrix{V}

    lock::ReentrantLock
end
function InfiniteMPOEnvironments(bra, O, ket, solver, GL, GR)
    return InfiniteMPOEnvironments(bra, O, ket, solver, GL, GR, ReentrantLock())
end

# Constructors
# ------------
function environments(state::InfiniteMPS, O::InfiniteMPO; kwargs...)
    return environments(convert(MPSMultiline, state), convert(MPOMultiline, O); kwargs...)
end
function environments(below::InfiniteMPS,
                      (mpo, above)::Tuple{<:InfiniteMPO,<:InfiniteMPS}; kwargs...)
    return environments(convert(MPSMultiline, below),
                        (convert(MPOMultiline, mpo), convert(MPSMultiline, above));
                        kwargs...)
end

function environments(state::MPSMultiline, mpo::MPOMultiline; solver=Defaults.eigsolver)
    GL, GR = initialize_environments(state, mpo, state)
    envs = InfiniteMPOEnvironments(nothing, mpo, state, solver, GL, GR)
    return recalculate!(envs, state)
end

function environments(below::MPSMultiline,
                      (mpo, above)::Tuple{<:MPOMultiline,<:MPSMultiline};
                      solver=Defaults.eigsolver)
    GL, GR = initialize_environments(above, mpo, below)
    envs = InfiniteMPOEnvironments(above, mpo, below, solver, GL, GR)
    return recalculate!(envs, below)
end

function initialize_environments(ket::MPSMultiline, operator::MPOMultiline,
                                 bra::MPSMultiline=ket)
    # allocate
    GL = PeriodicArray([allocate_GL(bra[row], operator[row], ket[row], col)
                        for row in 1:size(ket, 1), col in 1:size(ket, 2)])
    GR = PeriodicArray([allocate_GR(bra[row], operator[row], ket[row], col)
                        for row in 1:size(ket, 1), col in 1:size(ket, 2)])

    # initialize: randomize
    foreach(randomize!, GL)
    foreach(randomize!, GR)

    return GL, GR
end

# Getter/Setters
# --------------
function leftenv(envs::InfiniteMPOEnvironments, pos::Int, state::InfiniteMPS)
    check_recalculate!(envs, state)
    return envs.leftenvs[1, pos]
end
function leftenv(envs::InfiniteMPOEnvironments, pos::Int, state::MPSMultiline)
    check_recalculate!(envs, state)
    return envs.leftenvs[:, pos]
end
function leftenv(envs::InfiniteMPOEnvironments, row::Int, col::Int, state)
    check_recalculate!(envs, state)
    return envs.leftenvs[row, col]
end

function rightenv(envs::InfiniteMPOEnvironments, pos::Int, state::InfiniteMPS)
    check_recalculate!(envs, state)
    return envs.rightenvs[1, pos]
end
function rightenv(envs::InfiniteMPOEnvironments, pos::Int, state::MPSMultiline)
    check_recalculate!(envs, state)
    return envs.rightenvs[:, pos]
end
function rightenv(envs::InfiniteMPOEnvironments, row::Int, col::Int, state)
    check_recalculate!(envs, state)
    return envs.rightenvs[row, col]
end

# Utility
# -------
function check_dependency(envs::InfiniteMPOEnvironments, state::MPSMultiline)
    return all(x -> ===(x...), zip(envs.dependency, state))
end

function issamespace(envs::InfiniteMPOEnvironments, state::MPSMultiline)
    for row in 1:size(state, 1)
        newstate = state[row]
        oldstate = envs.dependency[row]
        for col in 1:size(state, 2)
            if left_virtualspace(oldstate, col) != left_virtualspace(newstate, col)
                return false
            end
            if right_virtualspace(oldstate, col) != right_virtualspace(newstate, col)
                return false
            end
        end
    end
    return true
end

# Calculation
# -----------
function recalculate!(envs::InfiniteMPOEnvironments, nstate::InfiniteMPS; kwargs...)
    return recalculate!(envs, convert(MPSMultiline, nstate); kwargs...)
end

function compute_leftenv!(envs::InfiniteMPOEnvironments)
    below = envs.dependency
    above = something(envs.above, below)
    mpo = envs.operator

    # sanity check
    numrows, numcols = size(above)
    @assert size(above) == size(mpo)
    @assert size(below) == size(mpo)

    @threads for row in 1:numrows
        T = TransferMatrix(above[row].AL, mpo[row, :], below[row + 1].AL)
        _, envs.leftenvs[row, 1] = fixedpoint(flip(T), envs.leftenvs[row, 1], :LM,
                                              envs.solver)
        # compute rest of unitcell
        for col in 2:numcols
            envs.leftenvs[row, col] = envs.leftenvs[row, col - 1] *
                                      TransferMatrix(above[row].AL[col - 1],
                                                     mpo[row, col - 1],
                                                     below[row + 1].AL[col - 1])
        end
    end

    return envs
end

function compute_rightenv!(envs::InfiniteMPOEnvironments)
    below = envs.dependency
    above = something(envs.above, below)
    mpo = envs.operator

    # sanity check
    numrows, numcols = size(above)
    @assert size(above) == size(mpo)
    @assert size(below) == size(mpo)

    @threads for row in 1:numrows
        T = TransferMatrix(above[row].AR, mpo[row, :], below[row + 1].AR)
        _, envs.rightenvs[row, end] = fixedpoint(T, envs.rightenvs[row, end], :LM,
                                                 envs.solver)
        # compute rest of unitcell
        for col in (numcols - 1):-1:1
            envs.rightenvs[row, col] = TransferMatrix(above[row].AR[col + 1],
                                                      mpo[row, col + 1],
                                                      below[row].AR[col + 1]) *
                                       envs.rightenvs[row, col + 1]
        end
    end

    return envs
end

function TensorKit.normalize!(envs::InfiniteMPOEnvironments)
    below = envs.dependency
    above = something(envs.above, below)

    for row in 1:size(below, 1)
        # fix normalization
        CRs_top, CRs_bot = above[row].CR, below[row + 1].CR
        for col in 1:size(below, 2)
            λ = dot(CRs_bot[col],
                    MPO_∂∂C(envs.leftenvs[row, col + 1], envs.rightenvs[row, col]) *
                    CRs_top[col])
            scale!(envs.leftenvs[row, col + 1], inv(λ))
        end
    end
end
