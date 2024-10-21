# --- above === below ---
"
    This object manages the periodic mpo environments for an MPSMultiline
"
mutable struct InfiniteMPOEnvironments{O,V,S<:MPSMultiline,A} <: AbstractInfEnv
    above::Union{S,Nothing}
    operator::O

    dependency::S
    solver::A

    leftenvs::PeriodicMatrix{V}
    rightenvs::PeriodicMatrix{V}

    lock::ReentrantLock
end

# convert to multiline
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
    envs = InfiniteMPOEnvironments(nothing, mpo, state, solver, GL, GR, ReentrantLock())
    return recalculate!(envs, state)
end

function environments(below::MPSMultiline,
                      (mpo, above)::Tuple{<:MPOMultiline,<:MPSMultiline};
                      solver=Defaults.eigsolver)
    GL, GR = initialize_environments(above, mpo, below)
    envs = InfiniteMPOEnvironments(above, mpo, below, solver, GL, GR, ReentrantLock())
    return recalculate!(envs, below)
end

function initialize_environments(ket::MPSMultiline, operator::MPOMultiline,
                                 bra::MPSMultiline=ket)
    # allocate
    GL = PeriodicArray([allocate_GL(ket[row], operator[row], bra[row], col)
                        for row in 1:size(ket, 1), col in 1:size(ket, 2)])
    GR = PeriodicArray([allocate_GR(ket[row], operator[row], bra[row], col)
                        for row in 1:size(ket, 1), col in 1:size(ket, 2)])

    # initialize: randomize
    foreach(randomize!, GL)
    foreach(randomize!, GR)

    return GL, GR
end

function recalculate!(envs::InfiniteMPOEnvironments, nstate::InfiniteMPS; kwargs...)
    return recalculate!(envs, convert(MPSMultiline, nstate); kwargs...)
end
function recalculate!(envs::InfiniteMPOEnvironments, state::MPSMultiline;
                      tol=envs.solver.tol)
    if !issamespace(envs, state)
        envs.leftenvs, envs.rightenvs = initialize_environments(state, envs.operator)
    end

    solver = envs.solver
    solver = solver.tol == tol ? solver : @set solver.tol = tol
    envs.solver = solver
    envs.dependency = state

    # compute fixedpoints
    compute_leftenv!(envs)
    compute_rightenv!(envs)
    normalize!(envs)

    return envs
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
            scale!(envs.leftenvs[row, col + 1], 1 / sqrt(λ))
            scale!(envs.rightenvs[row, col], 1 / sqrt(λ))
        end
    end
end
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

# --- utility functions ---

# function gen_init_fps(above::MPSMultiline, mpo::Multiline{<:DenseMPO}, below::MPSMultiline)
#     T = eltype(above)
#
#     map(1:size(mpo, 1)) do cr
#         L0::T = randomize!(similar(above.AL[1, 1],
#                                    left_virtualspace(below, cr + 1, 0) *
#                                    _firstspace(mpo[cr, 1])',
#                                    left_virtualspace(above, cr, 0)))
#         R0::T = randomize!(similar(above.AL[1, 1],
#                                    right_virtualspace(above, cr, 0) *
#                                    _firstspace(mpo[cr, 1]),
#                                    right_virtualspace(below, cr + 1, 0)))
#         return (L0, R0)
#     end
# end
#
# function gen_init_fps(above::MPSMultiline, mpo::Multiline{<:SparseMPO}, below::MPSMultiline)
#     map(1:size(mpo, 1)) do cr
#         ham = mpo[cr]
#         ab = above[cr]
#         be = below[cr]
#
#         A = eltype(ab)
#
#         lw = Vector{A}(undef, ham.odim)
#         rw = Vector{A}(undef, ham.odim)
#
#         for j in 1:(ham.odim)
#             lw[j] = similar(ab.AL[1], _firstspace(be.AL[1]) * ham[1].domspaces[j]',
#                             _firstspace(ab.AL[1]))
#             rw[j] = similar(ab.AL[1], _lastspace(ab.AR[end])' * ham[end].imspaces[j]',
#                             _lastspace(be.AR[end])')
#         end
#
#         randomize!.(lw)
#         randomize!.(rw)
#
#         return (lw, rw)
#     end
# end
#
# function gen_init_fps(above::MPSMultiline, mpo::MPOMultiline, below::MPSMultiline=above)
#     return map(axes(mpo, 1)) do row
#         O = mpo[row]
#         ab = above[row]
#         be = below[row]
#
#         GL = allocate_GL(ab, O, be, 1)
#         GR = allocate_GR(ab, O, be, 1)
#         # GL = similar(ab.AL[1],
#         #              left_virtualspace(be, 1) ⊗ left_virtualspace(O, 1)' ←
#         #              left_virtualspace(ab, 1))
#         # GR = similar(ab.AL[1],
#         #              right_virtualspace(ab, 1) ⊗ right_virtualspace(O, 1)' ←
#         #              right_virtualspace(be, 1))
#         randomize!(GL)
#         randomize!(GR)
#
#         return GL, GR
#     end
# end

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

# function mixed_fixpoints(above::MPSMultiline, mpo::MPOMultiline, below::MPSMultiline,
#                          init=gen_init_fps(above, mpo, below); solver=Defaults.eigsolver)
#     # sanity check
#     numrows, numcols = size(above)
#     @assert size(above) == size(mpo)
#     @assert size(below) == size(mpo)
#
#     envtype = eltype(init[1])
#     GLs = PeriodicArray{envtype,2}(undef, numrows, numcols)
#     GRs = PeriodicArray{envtype,2}(undef, numrows, numcols)
#
#     @threads for row in 1:numrows
#         Os = mpo[row, :]
#         ALs_top, ALs_bot = above[row].AL, below[row + 1].AL
#         ARs_top, ARs_bot = above[row].AR, below[row + 1].AR
#         L0, R0 = init[row]
#         @sync begin
#             Threads.@spawn begin
#                 E_LL = TransferMatrix(ALs_top, Os, ALs_bot)
#                 _, GLs[row, 1] = fixedpoint(flip(E_LL), L0, :LM, solver)
#                 # compute rest of unitcell
#                 for col in 2:numcols
#                     GLs[row, col] = GLs[row, col - 1] *
#                                     TransferMatrix(ALs_top[col - 1], Os[col - 1],
#                                                    ALs_bot[col - 1])
#                 end
#             end
#
#             Threads.@spawn begin
#                 E_RR = TransferMatrix(ARs_top, Os, ARs_bot)
#                 _, GRs[row, end] = fixedpoint(E_RR, R0, :LM, solver)
#                 # compute rest of unitcell
#                 for col in (numcols - 1):-1:1
#                     GRs[row, col] = TransferMatrix(ARs_top[col + 1], Os[col + 1],
#                                                    ARs_bot[col + 1]) *
#                                     GRs[row, col + 1]
#                 end
#             end
#         end
#
#         # fix normalization
#         CRs_top, CRs_bot = above[row].CR, below[row + 1].CR
#         for col in 1:numcols
#             λ = dot(CRs_bot[col],
#                     MPO_∂∂C(GLs[row, col + 1], GRs[row, col]) * CRs_top[col])
#             scale!(GLs[row, col + 1], 1 / sqrt(λ))
#             scale!(GRs[row, col], 1 / sqrt(λ))
#         end
#     end
#
#     return GLs, GRs
# end
