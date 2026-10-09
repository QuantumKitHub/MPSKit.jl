# Internal state of the IDMRG/IDMRG2 leading boundary search, where `ϵ` is the change of the
# center bond tensor over the last sweep. IDMRG2 reuses one `truncation_errors` matrix across sweeps.
struct IDMRGBoundaryState{S, O, E, V, A}
    mps::S
    operator::O
    envs::E
    iter::Int
    ϵ::Float64
    truncation_errors::V
    allocator::A
end

function leading_boundary(
        ψ::InfiniteMultilineMPS, operator, alg::Union{IDMRG, IDMRG2},
        envs = environments(ψ, operator, ψ)
    )
    allocator = default_allocator(ψ, SerialScheduler())
    alg isa IDMRG2 && width(ψ) < 2 && throw(ArgumentError("unit cell should be >= 2"))
    log = IterLog(alg)
    ϵ_truncs = alg isa IDMRG2 ?
        PeriodicMatrix(zeros(real(scalartype(ψ)), length(ψ), width(ψ))) : nothing
    state = IDMRGBoundaryState(ψ, operator, envs, 0, 2 * alg.tol, ϵ_truncs, allocator)
    it = IterativeSolver(alg, state)

    with_verbosity(; alg.verbosity) do
        @log_initialization loginit!(log, it.ϵ, _boundary_objective(alg, ψ, operator, envs))
        for (ψ, envs, ϵ) in Iterators.take(it, alg.maxiter)
            if ϵ <= alg.tol
                @log_convergence logfinish!(log, it.iter, ϵ, _boundary_objective(alg, ψ, operator, envs))
                break
            elseif it.iter == alg.maxiter
                @log_nonconvergence logcancel!(log, it.iter, ϵ, _boundary_objective(alg, ψ, operator, envs))
            else
                @log_iteration logiter!(log, it.iter, ϵ, _boundary_objective(alg, ψ, operator, envs))
            end
        end
    end

    (; iter, ϵ) = it.state
    alg_gauge = adapt_solver(alg.alg_gauge; iter, g_global = ϵ)
    ψ = MultilineMPS(map(identity, ψ.AR); alg_gauge.tol, alg_gauge.maxiter)

    recalculate!(envs, ψ, operator, ψ)
    info = AlgorithmInfo(;
        converged = ϵ <= alg.tol, bondresidual = ϵ,
        truncation_errors = isnothing(ϵ_truncs) ? nothing : parent(ϵ_truncs), numiter = iter,
    )
    return ψ, envs, info
end

# only single-site IDMRG reports the leading eigenvalue in its log
_boundary_objective(::IDMRG, ψ, operator, envs) = leading_eigenvalue(ψ, operator, envs)
_boundary_objective(::IDMRG2, ψ, operator, envs) = nothing

function Base.iterate(it::IterativeSolver{<:Union{IDMRG, IDMRG2}}, state::IDMRGBoundaryState)
    iter = state.iter + 1
    C_current = state.mps.C[:, 0]
    state = sweep!(it, state, Val(:right), iter)
    state = sweep!(it, state, Val(:left), iter)
    ϵ = _center_change(it.alg, C_current, state.mps.C[:, 0])
    it.state = IDMRGBoundaryState(
        state.mps, state.operator, state.envs, iter, ϵ, state.truncation_errors, state.allocator,
    )
    return (it.state.mps, it.state.envs, it.state.ϵ), it.state
end

function sweep!(it::IterativeSolver{<:IDMRG}, state::IDMRGBoundaryState, ::Val{:right}, iter)
    alg, ψ, operator, envs, allocator = it.alg, state.mps, state.operator, state.envs, state.allocator
    alg_eigsolve = adapt_solver(alg.alg_eigsolve; iter, g_global = state.ϵ)
    for col in 1:width(ψ)
        Hac = AC_hamiltonian(col, ψ, operator, ψ, envs; alg.backend, allocator)
        _, ψ.AC[:, col] = fixedpoint(Hac, ψ.AC[:, col], :LM, alg_eigsolve)

        for row in 1:size(ψ, 1)
            ac = ψ.AC[row, col]
            (col == width(ψ)) && (ac = copy(ac)) # needed in next sweep
            ψ.AL[row, col], ψ.C[row, col] = left_orth!(ac)
        end

        transfer_leftenv!(envs, ψ, operator, ψ, col + 1)
    end
    return state
end

function sweep!(it::IterativeSolver{<:IDMRG}, state::IDMRGBoundaryState, ::Val{:left}, iter)
    alg, ψ, operator, envs, allocator = it.alg, state.mps, state.operator, state.envs, state.allocator
    alg_eigsolve = adapt_solver(alg.alg_eigsolve; iter, g_global = state.ϵ)
    for col in width(ψ):-1:1
        Hac = AC_hamiltonian(col, ψ, operator, ψ, envs; alg.backend, allocator)
        _, ψ.AC[:, col] = fixedpoint(Hac, ψ.AC[:, col], :LM, alg_eigsolve)

        for row in 1:size(ψ, 1)
            ψ.C[row, col - 1], temp = right_orth!(_transpose_tail(ψ.AC[row, col]; copy = true))
            ψ.AR[row, col] = _transpose_front(temp)
        end

        transfer_rightenv!(envs, ψ, operator, ψ, col - 1)
    end
    normalize!(envs, ψ, operator, ψ)
    return state
end

function sweep!(it::IterativeSolver{<:IDMRG2}, state::IDMRGBoundaryState, ::Val{:right}, iter)
    alg, ψ, operator, envs, allocator = it.alg, state.mps, state.operator, state.envs, state.allocator
    alg_eigsolve = adapt_solver(alg.alg_eigsolve; iter, g_global = state.ϵ)
    ϵ_truncs = state.truncation_errors
    for site in 1:(width(ψ) - 1)
        ac2 = AC2(ψ, site; kind = :ACAR)
        h = AC2_hamiltonian(site, ψ, operator, ψ, envs; alg.backend, allocator)
        _, ac2′ = fixedpoint(h, ac2, :LM, alg_eigsolve)

        for row in 1:size(ψ, 1)
            al, c, ar, ϵ_truncs[row + 1, site] = svd_trunc!(ac2′[row]; trunc = alg.trunc, alg = alg.alg_svd)
            normalize!(c)

            ψ.AL[row + 1, site] = al
            ψ.C[row + 1, site] = complex(c)
            ψ.AR[row + 1, site + 1] = _transpose_front(ar)
            ψ.AC[row + 1, site + 1] = _transpose_front(c * ar)
        end

        transfer_leftenv!(envs, ψ, operator, ψ, site + 1)
        transfer_rightenv!(envs, ψ, operator, ψ, site)
    end

    normalize!(envs, ψ, operator, ψ)

    # update the edge
    site = width(ψ)
    ψ.AL[:, end] .= ψ.AC[:, end] ./ ψ.C[:, end]
    ψ.AC[:, 1] .= _mul_tail.(ψ.AL[:, 1], ψ.C[:, 1])
    ac2 = AC2(ψ, site; kind = :ALAC)
    h = AC2_hamiltonian(site, ψ, operator, ψ, envs; alg.backend, allocator)
    _, ac2′ = fixedpoint(h, ac2, :LM, alg_eigsolve)

    for row in 1:size(ψ, 1)
        al, c, ar, ϵ_truncs[row + 1, site] = svd_trunc!(ac2′[row]; trunc = alg.trunc, alg = alg.alg_svd)
        normalize!(c)

        ψ.AL[row + 1, site] = al
        ψ.C[row + 1, site] = complex(c)
        ψ.AR[row + 1, site + 1] = _transpose_front(ar)

        ψ.AC[row + 1, site] = _mul_tail(al, c)
        ψ.AC[row + 1, 1] = _transpose_front(c * ar)
        ψ.AL[row + 1, 1] = ψ.AC[row + 1, 1] / ψ.C[row + 1, 1]
    end

    transfer_leftenv!(envs, ψ, operator, ψ, 1)
    transfer_rightenv!(envs, ψ, operator, ψ, 0)
    return state
end

function sweep!(it::IterativeSolver{<:IDMRG2}, state::IDMRGBoundaryState, ::Val{:left}, iter)
    alg, ψ, operator, envs, allocator = it.alg, state.mps, state.operator, state.envs, state.allocator
    alg_eigsolve = adapt_solver(alg.alg_eigsolve; iter, g_global = state.ϵ)
    ϵ_truncs = state.truncation_errors
    for site in reverse(1:(width(ψ) - 1))
        ac2 = AC2(ψ, site; kind = :ALAC)
        h = AC2_hamiltonian(site, ψ, operator, ψ, envs; alg.backend, allocator)
        _, ac2′ = fixedpoint(h, ac2, :LM, alg_eigsolve)

        for row in 1:size(ψ, 1)
            al, c, ar, ϵ_truncs[row + 1, site] = svd_trunc!(ac2′[row]; trunc = alg.trunc, alg = alg.alg_svd)
            normalize!(c)

            ψ.AL[row + 1, site] = al
            ψ.C[row + 1, site] = complex(c)
            ψ.AR[row + 1, site + 1] = _transpose_front(ar)
        end

        transfer_leftenv!(envs, ψ, operator, ψ, site + 1)
        transfer_rightenv!(envs, ψ, operator, ψ, site)
    end

    normalize!(envs, ψ, operator, ψ)

    # update the edge
    ψ.AC[:, end] .= _mul_front.(ψ.C[:, end - 1], ψ.AR[:, end])
    ψ.AC[:, 1] .= _mul_tail.(ψ.AL[:, 1], ψ.C[:, 1])
    ψ.AR[:, 1] .= _transpose_front.(ψ.C[:, end] .\ _transpose_tail.(ψ.AC[:, 1]))
    ac2 = AC2(ψ, 0; kind = :ACAR)
    h = AC2_hamiltonian(0, ψ, operator, ψ, envs; alg.backend, allocator)
    _, ac2′ = fixedpoint(h, ac2, :LM, alg_eigsolve)

    for row in 1:size(ψ, 1)
        al, c, ar, ϵ_truncs[row + 1, end] = svd_trunc!(ac2′[row]; trunc = alg.trunc, alg = alg.alg_svd)
        normalize!(c)

        ψ.AL[row + 1, end] = al
        ψ.C[row + 1, end] = complex(c)
        ψ.AR[row + 1, 1] = _transpose_front(ar)

        ψ.AR[row + 1, end] = _transpose_front(
            ψ.C[row + 1, end - 1] \ _transpose_tail(al * c)
        )
        ψ.AC[row + 1, 1] = _transpose_front(c * ar)
    end

    transfer_leftenv!(envs, ψ, operator, ψ, 1)
    transfer_rightenv!(envs, ψ, operator, ψ, 0)
    return state
end
