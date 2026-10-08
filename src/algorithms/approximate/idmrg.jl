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
    log = IterLog(string(nameof(typeof(alg))))
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
    ϵ = approximate_sweep!(
        state.mps, state.operator, it.alg, state.envs, state.allocator, state.truncation_errors
    )
    it.state = IDMRGApproximateState(
        state.mps, state.operator, state.envs, state.iter + 1, ϵ,
        state.truncation_errors, state.allocator,
    )
    return (it.state.mps, it.state.envs, it.state.ϵ), it.state
end

function approximate_sweep!(
        ψ::MultilineMPS, toapprox, alg::IDMRG, envs, allocator, ::Nothing
    )
    C_current = ψ.C[:, 0]

    # left to right sweep
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

    # right to left sweep
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

    return norm(C_current - ψ.C[:, 0])
end

function approximate_sweep!(
        ψ::MultilineMPS, toapprox, alg::IDMRG2, envs, allocator, ϵ_truncs
    )
    C_current = ψ.C[:, 0]

    # sweep from left to right
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
    # update environments
    transfer_leftenv!(envs, ψ, toapprox, 1)
    transfer_rightenv!(envs, ψ, toapprox, 0)

    normalize!(envs, ψ, toapprox)

    # sweep from right to left
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

    # update error
    return sum(zip(C_current, ψ.C[:, 0])) do (c1, c2)
        smallest = infimum(_firstspace(c1), _firstspace(c2))
        e1 = isometry(_firstspace(c1), smallest)
        e2 = isometry(_firstspace(c2), smallest)
        return norm(e2' * c2 * e2 - e1' * c1 * e1)
    end
end
