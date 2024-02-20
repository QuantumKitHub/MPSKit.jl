"""
    IDMRG1{A} <: Algorithm

Single site infinite DMRG algorithm for finding groundstates.

# Fields
- `tol::Float64`: tolerance for convergence criterium
- `tol_gauge::Float64`: tolerance for gauging algorithm
- `eigalg::A`: eigensolver algorithm
- `maxiter::Int`: maximum number of outer iterations
- `verbosity::Int`: display progress information
"""
struct IDMRG1{A} <: Algorithm
    tol::Float64
    tol_gauge::Float64
    eigalg::A
    maxiter::Int
    verbosity::Int
end
function IDMRG1(; tol::Real=Defaults.tol, tol_gauge::Real=Defaults.tolgauge,
                eigalg=Defaults.eigsolver, maxiter::Integer=Defaults.maxiter,
                verbosity=Defaults.verbosity, tol_galerkin=nothing, verbose=nothing)
    # Deprecation warnings
    actual_tol = if !isnothing(tol_galerkin)
        Base.depwarn("IDMRG1(; kwargs..., tol_galerkin=...) is deprecated. Use IDMRG1(; kwargs...) instead.",
                     :IDMRG1; force=true)
        tol_galerkin
    else
        tol
    end
    actual_verbosity = if !isnothing(verbose)
        Base.depwarn("IDMRG1(; kwargs..., verbose=...) is deprecated. Use IDMRG1(; kwargs..., verbosity=...) instead.",
                     :IDMRG1; force=true)
        verbose
    else
        verbosity
    end
    return IDMRG1{typeof(eigalg)}(actual_tol, tol_gauge, eigalg, maxiter, actual_verbosity)
end

function find_groundstate(ψ′::InfiniteMPS, H, alg::IDMRG1, oenvs=environments(ψ′, H))
    ψ = copy(ψ′)
    envs = IDMRGEnv(ψ′, oenvs)

    δ::Float64 = 2 * alg.tol

    for topit in 1:(alg.maxiter)
        δ = 0.0
        curc = ψ.CR[0]

        for pos in 1:length(ψ)
            h = ∂∂AC(pos, ψ, H, envs)
            _, vecs = eigsolve(h, ψ.AC[pos], 1, :SR, alg.eigalg)

            ψ.AC[pos] = vecs[1]
            ψ.AL[pos], ψ.CR[pos] = leftorth(vecs[1])

            update_leftenv!(envs, ψ, H, pos + 1)
        end

        for pos in length(ψ):-1:1
            h = ∂∂AC(pos, ψ, H, envs)
            _, vecs = eigsolve(h, ψ.AC[pos], 1, :SR, alg.eigalg)

            ψ.AC[pos] = vecs[1]
            ψ.CR[pos - 1], temp = rightorth!(_transpose_tail(vecs[1]))
            ψ.AR[pos] = _transpose_front(temp)

            update_rightenv!(envs, ψ, H, pos - 1)
        end

        δ = norm(curc - ψ.CR[0])
        δ < alg.tol && break
        alg.verbosity <= VERBOSE_ITER && @info "IDMRG iter $(topit) err $(δ)"
    end

    ψ″ = InfiniteMPS(ψ.AR[1:end]; tol=alg.tol_gauge)
    nenvs = environments(ψ″, H; solver=oenvs.solver)
    return ψ″, nenvs, δ
end

"""
    IDMRG2{A} <: Algorithm

2-site infinite DMRG algorithm for finding groundstates.

# Fields
- `tol::Float64`: tolerance for convergence criterium
- `tol_gauge::Float64`: tolerance for gauging algorithm
- `eigalg::A`: eigensolver algorithm
- `maxiter::Int`: maximum number of outer iterations
- `verbosity::Int`: display progress information
- `trscheme`: truncation algorithm for [tsvd][TensorKit.tsvd](@ref)
"""
struct IDMRG2{A} <: Algorithm
    tol::Float64
    tol_gauge::Float64
    eigalg::A
    maxiter::Int
    verbosity::Int
    trscheme::TruncationScheme
end
function IDMRG2(; tol::Real=Defaults.tol, tol_gauge::Real=Defaults.tolgauge,
                eigalg=Defaults.eigsolver, maxiter::Integer=Defaults.maxiter,
                verbosity=Defaults.verbosity,
                trscheme::TruncationScheme=truncerr(sqrt(tol)),
                tol_galerkin=nothing, verbose=nothing)
    # Deprecation warnings
    actual_tol = if !isnothing(tol_galerkin)
        Base.depwarn("IDMRG2(; kwargs..., tol_galerkin=...) is deprecated. Use IDMRG2(; kwargs...) instead.",
                     :IDMRG2; force=true)
        tol_galerkin
    else
        tol
    end
    actual_verbosity = if !isnothing(verbose)
        Base.depwarn("IDMRG2(; kwargs..., verbose=...) is deprecated. Use IDMRG2(; kwargs..., verbosity=...) instead.",
                     :IDMRG2; force=true)
        verbose
    else
        verbosity
    end

    return IDMRG2{typeof(eigalg)}(actual_tol, tol_gauge, eigalg, maxiter, actual_verbosity,
                                  trscheme)
end

function find_groundstate(ψ′::InfiniteMPS, H, alg::IDMRG2, envs′=environments(ψ′, H))
    length(ψ′) < 2 && throw(ArgumentError("unit cell should be >= 2"))

    ψ = copy(ψ′)
    envs = IDMRGEnv(ψ′, envs′)

    δ::Float64 = 2 * alg.tol

    for topit in 1:(alg.maxiter)
        δ = 0.0

        curc = ψ.CR[0]

        # sweep from left to right
        for pos in 1:(length(ψ) - 1)
            ac2 = ψ.AC[pos] * _transpose_tail(ψ.AR[pos + 1])
            h_ac2 = ∂∂AC2(pos, ψ, H, envs)
            _, vecs, _ = eigsolve(h_ac2, ac2, 1, :SR, alg.eigalg)

            (al, c, ar, ϵ) = tsvd(vecs[1]; trunc=alg.trscheme, alg=TensorKit.SVD())
            normalize!(c)

            ψ.AL[pos] = al
            ψ.CR[pos] = complex(c)
            ψ.AR[pos + 1] = _transpose_front(ar)
            ψ.AC[pos + 1] = _transpose_front(c * ar)

            update_leftenv!(envs, ψ, H, pos + 1)
            update_rightenv!(envs, ψ, H, pos)
        end

        # update the edge
        @plansor ac2[-1 -2; -3 -4] := ψ.AC[end][-1 -2; 1] * inv(ψ.CR[0])[1; 2] *
                                      ψ.AL[1][2 -4; 3] * ψ.CR[1][3; -3]
        h_ac2 = ∂∂AC2(0, ψ, H, envs)
        _, vecs, _ = eigsolve(h_ac2, ac2, 1, :SR, alg.eigalg)

        al, c, ar, ϵ = tsvd(vecs[1]; trunc=alg.trscheme, alg=TensorKit.SVD())
        normalize!(c)

        ψ.AC[end] = al * c
        ψ.AL[end] = al
        ψ.CR[end] = complex(c)
        ψ.AR[1] = _transpose_front(ar)
        ψ.AC[1] = _transpose_front(c * ar)
        ψ.AL[1] = ψ.AC[1] * inv(ψ.CR[1])

        curc = complex(c)

        # update environments
        update_leftenv!(envs, ψ, H, 1)
        update_rightenv!(envs, ψ, H, 0)

        # sweep from right to left
        for pos in (length(ψ) - 1):-1:1
            ac2 = ψ.AL[pos] * _transpose_tail(ψ.AC[pos + 1])
            h_ac2 = ∂∂AC2(pos, ψ, H, envs)
            _, vecs, _ = eigsolve(h_ac2, ac2, 1, :SR, alg.eigalg)

            al, c, ar, ϵ = tsvd!(vecs[1]; trunc=alg.trscheme, alg=TensorKit.SVD())
            normalize!(c)

            ψ.AL[pos] = al
            ψ.AC[pos] = al * c
            ψ.CR[pos] = complex(c)
            ψ.AR[pos + 1] = _transpose_front(ar)
            ψ.AC[pos + 1] = _transpose_front(c * ar)

            update_leftenv!(envs, ψ, H, pos + 1)
            update_rightenv!(envs, ψ, H, pos)
        end

        # update the edge
        @plansor ac2[-1 -2; -3 -4] := ψ.CR[end - 1][-1; 1] * ψ.AR[end][1 -2; 2] *
                                      inv(ψ.CR[end])[2; 3] * ψ.AC[1][3 -4; -3]
        h_ac2 = ∂∂AC2(0, ψ, H, envs)
        eigvals, vecs = eigsolve(h_ac2, ac2, 1, :SR, alg.eigalg)
        al, c, ar, ϵ = tsvd(vecs[1]; trunc=alg.trscheme, alg=TensorKit.SVD())
        normalize!(c)

        ψ.AR[end] = _transpose_front(inv(ψ.CR[end - 1]) * _transpose_tail(al * c))
        ψ.AL[end] = al
        ψ.CR[end] = complex(c)
        ψ.AR[1] = _transpose_front(ar)
        ψ.AC[1] = _transpose_front(c * ar)

        update_leftenv!(envs, ψ, H, 1)
        update_rightenv!(envs, ψ, H, 0)

        # update error
        smallest = infimum(_firstspace(curc), _firstspace(c))
        e1 = isometry(_firstspace(curc), smallest)
        e2 = isometry(_firstspace(c), smallest)
        δ = norm(e2' * c * e2 - e1' * curc * e1)
        alg.verbosity >= VERBOSE_ITER && @info "IDMRG2 iter $(topit) err $(δ)"

        δ < alg.tol && break
    end

    ψ″ = InfiniteMPS(ψ.AR[1:end]; tol=alg.tol_gauge)
    nenvs = environments(ψ″, H; solver=envs′.solver)
    return ψ″, nenvs, δ
end
