# Internal state of the multiline IDMRG/IDMRG2 approximation, where `ϵ` is the change of the
# center bond tensor over the last sweep. IDMRG2 reuses one `truncation_errors` matrix across sweeps.
struct IDMRGApproximateState{S, O, E, V, A}
    mps::S
    operator::O
    envs::E
    iter::Int
    ϵ::Float64
    truncation_errors::V
    allocator::A
end

function approximate!(
        ψ::MultilineMPS, toapprox::Tuple{<:MultilineMPO, <:MultilineMPS},
        alg::Union{IDMRG, IDMRG2}, envs = environments(ψ, toapprox...)
    )
    allocator = default_allocator(ψ, SerialScheduler())
    alg isa IDMRG2 && width(ψ) < 2 && throw(ArgumentError("unit cell should be >= 2"))
    log = IterLog(alg)
    ϵ_truncs = alg isa IDMRG2 ?
        PeriodicMatrix(zeros(real(scalartype(ψ)), length(ψ), width(ψ))) : nothing
    state = IDMRGApproximateState(ψ, toapprox, envs, 0, 2 * alg.tol, ϵ_truncs, allocator)
    it = IterativeSolver(alg, state)

    with_verbosity(; alg.verbosity) do
        @log_initialization loginit!(log, it.ϵ)
        for _ in Iterators.take(it, alg.maxiter)
            ϵ = it.ϵ
            if ϵ <= alg.tol
                @log_convergence logfinish!(log, it.iter, ϵ)
                break
            elseif it.iter == alg.maxiter
                @log_nonconvergence logcancel!(log, it.iter, ϵ)
            else
                @log_iteration logiter!(log, it.iter, ϵ)
            end
        end
    end

    (; iter, ϵ) = it.state

    # TODO: immediately compute in-place
    alg_gauge = adapt_solver(alg.alg_gauge; iter, g_global = ϵ)
    ψ′ = MultilineMPS(map(identity, ψ.AR); alg_gauge.tol, alg_gauge.maxiter)
    copy!(ψ, ψ′) # ensure output destination is unchanged

    recalculate!(envs, ψ, toapprox)
    info = AlgorithmInfo(;
        converged = ϵ <= alg.tol, bondresidual = ϵ,
        truncation_errors = isnothing(ϵ_truncs) ? nothing : parent(ϵ_truncs), numiter = iter,
    )
    return ψ, envs, info
end

function Base.iterate(it::IterativeSolver{<:Union{IDMRG, IDMRG2}}, state::IDMRGApproximateState)
    iter = state.iter + 1
    C_current = state.mps.C[:, 0]
    state = sweep!(it, state, Val(:right), iter)
    state = sweep!(it, state, Val(:left), iter)
    ϵ = _center_change(it.alg, C_current, state.mps.C[:, 0])
    it.state = IDMRGApproximateState(
        state.mps, state.operator, state.envs, iter, ϵ, state.truncation_errors, state.allocator,
    )
    return (it.state.mps, it.state.envs, it.state.ϵ), it.state
end

# change of the center bond tensors over a sweep; for IDMRG2 the bond dimension may have
# changed, so both are compared in their common subspace
_center_change(::IDMRG, C_old, C_new) = norm(C_old - C_new)
function _center_change(::IDMRG2, C_old, C_new)
    return sum(zip(C_old, C_new)) do (c1, c2)
        smallest = infimum(_firstspace(c1), _firstspace(c2))
        e1 = isometry(_firstspace(c1), smallest)
        e2 = isometry(_firstspace(c2), smallest)
        return norm(e2' * c2 * e2 - e1' * c1 * e1)
    end
end

function sweep!(it::IterativeSolver{<:IDMRG}, state::IDMRGApproximateState, ::Val{:right}, iter)
    alg, ψ, toapprox, envs, allocator = it.alg, state.mps, state.operator, state.envs, state.allocator
    for col in 1:width(ψ)
        for row in 1:size(ψ, 1)
            ψ.AC[row + 1, col] = AC_projection(
                CartesianIndex(row, col), ψ, toapprox, envs;
                alg.backend, allocator
            )
            normalize!(ψ.AC[row + 1, col])
            ψ.AL[row + 1, col], ψ.C[row + 1, col] = left_orth!(ψ.AC[row + 1, col])
        end
        transfer_leftenv!(envs, ψ, toapprox, col + 1)
    end
    return state
end

function sweep!(it::IterativeSolver{<:IDMRG}, state::IDMRGApproximateState, ::Val{:left}, iter)
    alg, ψ, toapprox, envs, allocator = it.alg, state.mps, state.operator, state.envs, state.allocator
    for col in reverse(1:width(ψ))
        for row in 1:size(ψ, 1)
            ψ.AC[row + 1, col] = AC_projection(
                CartesianIndex(row, col), ψ, toapprox, envs;
                alg.backend, allocator
            )
            normalize!(ψ.AC[row + 1, col])
            ψ.C[row + 1, col - 1], temp = right_orth!(_transpose_tail(ψ.AC[row + 1, col]))
            ψ.AR[row + 1, col] = _transpose_front(temp)
        end
        transfer_rightenv!(envs, ψ, toapprox, col - 1)
    end
    normalize!(envs, ψ, toapprox)
    return state
end

function sweep!(it::IterativeSolver{<:IDMRG2}, state::IDMRGApproximateState, ::Val{:right}, iter)
    alg, ψ, toapprox, envs, allocator = it.alg, state.mps, state.operator, state.envs, state.allocator
    ϵ_truncs = state.truncation_errors
    for site in 1:(width(ψ) - 1)
        for row in 1:size(ψ, 1)
            AC2′ = AC2_projection(
                CartesianIndex(row, site), ψ, toapprox, envs;
                kind = :ACAR, alg.backend, allocator
            )
            al, c, ar, ϵ_truncs[row + 1, site] = svd_trunc!(AC2′; trunc = alg.trunc, alg = alg.alg_svd)
            normalize!(c)

            ψ.AL[row + 1, site] = al
            ψ.C[row + 1, site] = complex(c)
            ψ.AR[row + 1, site + 1] = _transpose_front(ar)
            ψ.AC[row + 1, site + 1] = _transpose_front(c * ar)
        end

        transfer_leftenv!(envs, ψ, toapprox, site + 1)
        transfer_rightenv!(envs, ψ, toapprox, site)
    end

    # update the edge
    ψ.AL[:, end] .= ψ.AC[:, end] ./ ψ.C[:, end]
    ψ.AC[:, 1] .= _mul_tail.(ψ.AL[:, 1], ψ.C[:, 1])
    for row in 1:size(ψ, 1)
        AC2′ = AC2_projection(
            CartesianIndex(row, width(ψ)), ψ, toapprox, envs;
            kind = :ALAC, alg.backend, allocator
        )
        al, c, ar, ϵ_truncs[row + 1, end] = svd_trunc!(AC2′; trunc = alg.trunc, alg = alg.alg_svd)
        normalize!(c)

        ψ.AL[row + 1, end] = al
        ψ.C[row + 1, end] = complex(c)
        ψ.AR[row + 1, 1] = _transpose_front(ar)

        ψ.AC[row + 1, end] = _mul_tail(al, c)
        ψ.AC[row + 1, 1] = _transpose_front(c * ar)
        ψ.AL[row + 1, 1] = ψ.AC[row + 1, 1] / ψ.C[row + 1, 1]
    end
    transfer_leftenv!(envs, ψ, toapprox, 1)
    transfer_rightenv!(envs, ψ, toapprox, 0)

    normalize!(envs, ψ, toapprox)
    return state
end

function sweep!(it::IterativeSolver{<:IDMRG2}, state::IDMRGApproximateState, ::Val{:left}, iter)
    alg, ψ, toapprox, envs, allocator = it.alg, state.mps, state.operator, state.envs, state.allocator
    ϵ_truncs = state.truncation_errors
    for site in reverse(1:(width(ψ) - 1))
        for row in 1:size(ψ, 1)
            AC2′ = AC2_projection(
                CartesianIndex(row, site), ψ, toapprox, envs;
                kind = :ALAC, alg.backend, allocator
            )
            al, c, ar, ϵ_truncs[row + 1, site] = svd_trunc!(AC2′; trunc = alg.trunc, alg = alg.alg_svd)
            normalize!(c)

            ψ.AL[row + 1, site] = al
            ψ.C[row + 1, site] = complex(c)
            ψ.AR[row + 1, site + 1] = _transpose_front(ar)
        end

        transfer_leftenv!(envs, ψ, toapprox, site + 1)
        transfer_rightenv!(envs, ψ, toapprox, site)
    end

    # update the edge
    ψ.AC[:, end] .= _mul_front.(ψ.C[:, end - 1], ψ.AR[:, end])
    ψ.AR[:, 1] .= _transpose_front.(ψ.C[:, end] .\ _transpose_tail.(ψ.AC[:, 1]))
    for row in 1:size(ψ, 1)
        AC2′ = AC2_projection(
            CartesianIndex(row, 0), ψ, toapprox, envs;
            kind = :ACAR, alg.backend, allocator
        )
        al, c, ar, ϵ_truncs[row + 1, end] = svd_trunc!(AC2′; trunc = alg.trunc, alg = alg.alg_svd)
        normalize!(c)

        ψ.AL[row + 1, end] = al
        ψ.C[row + 1, end] = complex(c)
        ψ.AR[row + 1, 1] = _transpose_front(ar)

        ψ.AR[row + 1, end] = _transpose_front(ψ.C[row + 1, end - 1] \ _transpose_tail(al * c))
        ψ.AC[row + 1, 1] = _transpose_front(c * ar)
    end
    transfer_leftenv!(envs, ψ, toapprox, 1)
    transfer_rightenv!(envs, ψ, toapprox, 0)

    normalize!(envs, ψ, toapprox)
    return state
end
