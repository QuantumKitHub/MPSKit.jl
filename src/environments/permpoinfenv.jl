# --- above === below ---
"
    This object manages the periodic mpo environments for an MPSMultiline
"
mutable struct PerMPOInfEnv{H,V,S<:MPSMultiline,A} <: AbstractInfEnv
    above::Union{S,Nothing}

    opp::H

    dependency::S
    solver::A

    lw::PeriodicArray{V,2}
    rw::PeriodicArray{V,2}

    lock::ReentrantLock
end

function environments(state::InfiniteMPS, opp::DenseMPO; kwargs...)
    return environments(convert(MPSMultiline, state), convert(MPOMultiline, opp); kwargs...)
end;

function environments(state::MPSMultiline, mpo::MPOMultiline; solver=Defaults.eigsolver)
    (lw, rw) = mixed_fixpoints(state, mpo, state; solver)

    return PerMPOInfEnv(nothing, mpo, state, solver, lw, rw, ReentrantLock())
end

function environments(below::InfiniteMPS,
                      toapprox::Tuple{<:Union{SparseMPO,DenseMPO},<:InfiniteMPS}; kwargs...)
    (opp, above) = toapprox
    return environments(convert(MPSMultiline, below),
                        (convert(MPOMultiline, opp), convert(MPSMultiline, above));
                        kwargs...)
end
function environments(below::MPSMultiline, toapprox::Tuple{<:MPOMultiline,<:MPSMultiline};
                      solver=Defaults.eigsolver)
    (mpo, above) = toapprox
    (lw, rw) = mixed_fixpoints(above, mpo, below; solver)

    return PerMPOInfEnv(above, mpo, below, solver, lw, rw, ReentrantLock())
end

function recalculate!(envs::PerMPOInfEnv, nstate::InfiniteMPS; kwargs...)
    return recalculate!(envs, convert(MPSMultiline, nstate); kwargs...)
end;
function recalculate!(envs::PerMPOInfEnv, nstate::MPSMultiline; tol=envs.solver.tol)
    sameDspace = reduce(&, _firstspace.(envs.dependency.CR) .== _firstspace.(nstate.CR))

    above = isnothing(envs.above) ? nstate : envs.above
    init = collect(zip(envs.lw[:, 1], envs.rw[:, end]))
    if !sameDspace
        init = gen_init_fps(above, envs.opp, nstate)
    end

    solver = envs.solver
    solver = solver.tol == tol ? solver : @set solver.tol = tol
    (envs.lw, envs.rw) = mixed_fixpoints(above, envs.opp, nstate, init; solver)
    envs.dependency = nstate
    envs.solver = solver

    return envs
end

function leftenv(envs::PerMPOInfEnv, pos::Int, state::InfiniteMPS)
    check_recalculate!(envs, state)
    return envs.lw[1, pos]
end

function rightenv(envs::PerMPOInfEnv, pos::Int, state::InfiniteMPS)
    check_recalculate!(envs, state)
    return envs.rw[1, pos]
end

function leftenv(envs::PerMPOInfEnv, pos::Int, state::MPSMultiline)
    check_recalculate!(envs, state)
    return envs.lw[:, pos]
end

function rightenv(envs::PerMPOInfEnv, pos::Int, state::MPSMultiline)
    check_recalculate!(envs, state)
    return envs.rw[:, pos]
end

function leftenv(envs::PerMPOInfEnv, row::Int, col::Int, state)
    check_recalculate!(envs, state)
    return envs.lw[row, col]
end

function rightenv(envs::PerMPOInfEnv, row::Int, col::Int, state)
    check_recalculate!(envs, state)
    return envs.rw[row, col]
end

# --- utility functions ---

function gen_init_fps(above::MPSMultiline, mpo::Multiline{<:DenseMPO}, below::MPSMultiline)
    T = eltype(above)

    map(1:size(mpo, 1)) do cr
        L0::T = randomize!(similar(above.AL[1, 1],
                                   left_virtualspace(below, cr + 1, 0) *
                                   _firstspace(mpo[cr, 1])',
                                   left_virtualspace(above, cr, 0)))
        R0::T = randomize!(similar(above.AL[1, 1],
                                   right_virtualspace(above, cr, 0) *
                                   _firstspace(mpo[cr, 1]),
                                   right_virtualspace(below, cr + 1, 0)))
        return (L0, R0)
    end
end

function gen_init_fps(above::MPSMultiline, mpo::Multiline{<:SparseMPO}, below::MPSMultiline)
    map(1:size(mpo, 1)) do cr
        ham = mpo[cr]
        ab = above[cr]
        be = below[cr]

        A = eltype(ab)

        lw = Vector{A}(undef, ham.odim)
        rw = Vector{A}(undef, ham.odim)

        for j in 1:(ham.odim)
            lw[j] = similar(ab.AL[1], _firstspace(be.AL[1]) * ham[1].domspaces[j]',
                            _firstspace(ab.AL[1]))
            rw[j] = similar(ab.AL[1], _lastspace(ab.AR[end])' * ham[end].imspaces[j]',
                            _lastspace(be.AR[end])')
        end

        randomize!.(lw)
        randomize!.(rw)

        return (lw, rw)
    end
end

function mixed_fixpoints(above::MPSMultiline, mpo::MPOMultiline, below::MPSMultiline,
                         init=gen_init_fps(above, mpo, below); solver=Defaults.eigsolver)
    # sanity check
    numrows, numcols = size(above)
    @assert size(above) == size(mpo)
    @assert size(below) == size(mpo)

    envtype = eltype(init[1])
    GLs = PeriodicArray{envtype,2}(undef, numrows, numcols)
    GRs = PeriodicArray{envtype,2}(undef, numrows, numcols)

    @threads for row in 1:numrows
        Os = mpo[row, :]
        ALs_top, ALs_bot = above[row].AL, below[row + 1].AL
        ARs_top, ARs_bot = above[row].AR, below[row + 1].AR
        L0, R0 = init[row]
        @sync begin
            Threads.@spawn begin
                E_LL = TransferMatrix(ALs_top, Os, ALs_bot)
                _, GLs[row, 1] = fixedpoint(flip(E_LL), L0, :LM, solver)
                # compute rest of unitcell
                for col in 2:numcols
                    GLs[row, col] = GLs[row, col - 1] *
                                    TransferMatrix(ALs_top[col - 1], Os[col - 1],
                                                   ALs_bot[col - 1])
                end
            end

            Threads.@spawn begin
                E_RR = TransferMatrix(ARs_top, Os, ARs_bot)
                _, GRs[row, end] = fixedpoint(E_RR, R0, :LM, solver)
                # compute rest of unitcell
                for col in (numcols - 1):-1:1
                    GRs[row, col] = TransferMatrix(ARs_top[col + 1], Os[col + 1],
                                                   ARs_bot[col + 1]) *
                                    GRs[row, col + 1]
                end
            end
        end

        # fix normalization
        CRs_top, CRs_bot = above[row].CR, below[row + 1].CR
        for col in 1:numcols
            λ = dot(CRs_bot[col],
                    MPO_∂∂C(GLs[row, col + 1], GRs[row, col]) * CRs_top[col])
            scale!(GLs[row, col + 1], inv(λ))
        end
    end

    return GLs, GRs
end
