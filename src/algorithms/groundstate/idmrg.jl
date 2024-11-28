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
function IDMRG1(; tol=Defaults.tol, tol_gauge=Defaults.tolgauge, eigalg=(;),
                maxiter=Defaults.maxiter, verbosity=Defaults.verbosity)
    eigalg′ = eigalg isa NamedTuple ? Defaults.alg_eigsolve(; eigalg...) : eigalg
    return IDMRG1{typeof(eigalg′)}(tol, tol_gauge, eigalg′, maxiter, verbosity)
end

function find_groundstate(ost::InfiniteMPS, H, alg::IDMRG1, oenvs=environments(ost, H))
    ϵ::Float64 = calc_galerkin(ost, oenvs)
    ψ = copy(ost)
    envs = IDMRGEnv(ost, oenvs)
    log = IterLog("IDMRG")

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ, expectation_value(ψ, H, envs))
        for iter in 1:(alg.maxiter)
            alg_eigsolve = updatetol(alg.eigalg, iter, ϵ)
            C_current = ψ.CR[0]

            # left to right sweep
            for pos in 1:length(ψ)
                h = ∂∂AC(pos, ψ, H, envs)
                _, ψ.AC[pos] = fixedpoint(h, ψ.AC[pos], :SR, alg_eigsolve)
                ψ.AL[pos], ψ.CR[pos] = leftorth!(ψ.AC[pos])
                update_leftenv!(envs, ψ, H, pos + 1)
            end

            # right to left sweep
            for pos in length(ψ):-1:1
                h = ∂∂AC(pos, ψ, H, envs)
                _, ψ.AC[pos] = fixedpoint(h, ψ.AC[pos], :SR, alg_eigsolve)

                ψ.CR[pos - 1], temp = rightorth!(_transpose_tail(ψ.AC[pos]))
                ψ.AR[pos] = _transpose_front(temp)

                update_rightenv!(envs, ψ, H, pos - 1)
            end

            ϵ = norm(C_current - ψ.CR[0])

            if ϵ < alg.tol
                @infov 2 logfinish!(log, iter, ϵ, expectation_value(ψ, H, envs))
                break
            end
            if iter == alg.maxiter
                @warnv 1 logcancel!(log, iter, ϵ, expectation_value(ψ, H, envs))
            else
                @infov 3 logiter!(log, iter, ϵ, expectation_value(ψ, H, envs))
            end
        end
    end

    nst = InfiniteMPS(ψ.AR[1:end]; tol=alg.tol_gauge)
    nenvs = environments(nst, H; solver=oenvs.solver)
    return nst, nenvs, ϵ
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
- `trscheme::TruncationScheme`: truncation algorithm for [tsvd][TensorKit.tsvd](@ref)
"""
struct IDMRG2{A} <: Algorithm
    tol::Float64
    tol_gauge::Float64
    eigalg::A
    maxiter::Int
    verbosity::Int
    trscheme::TruncationScheme
end
function IDMRG2(; tol=Defaults.tol, tol_gauge=Defaults.tolgauge, eigalg=(;),
                maxiter=Defaults.maxiter, verbosity=Defaults.verbosity,
                trscheme=truncerr(1e-6))
    eigalg′ = eigalg isa NamedTuple ? Defaults.alg_eigsolve(; eigalg...) : eigalg
    return IDMRG2{typeof(eigalg′)}(tol, tol_gauge, eigalg′, maxiter, verbosity, trscheme)
end

function find_groundstate(ost::InfiniteMPS, H, alg::IDMRG2, oenvs=environments(ost, H))
    length(ost) < 2 && throw(ArgumentError("unit cell should be >= 2"))
    ϵ::Float64 = calc_galerkin(ost, oenvs)

    ψ = copy(ost)
    envs = IDMRGEnv(ost, oenvs)
    log = IterLog("IDMRG2")

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ)
        for iter in 1:(alg.maxiter)
            alg_eigsolve = updatetol(alg.eigalg, iter, ϵ)
            C_current = ψ.CR[0]

            # sweep from left to right
            for pos in 1:(length(ψ) - 1)
                ac2 = ψ.AC[pos] * _transpose_tail(ψ.AR[pos + 1])
                h_ac2 = ∂∂AC2(pos, ψ, H, envs)
                _, ac2′ = fixedpoint(h_ac2, ac2, :SR, alg_eigsolve)

                al, c, ar, = tsvd!(ac2′; trunc=alg.trscheme, alg=TensorKit.SVD())
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
            _, ac2′ = fixedpoint(h_ac2, ac2, :SR, alg_eigsolve)

            al, c, ar, = tsvd!(ac2′; trunc=alg.trscheme, alg=TensorKit.SVD())
            normalize!(c)

            ψ.AC[end] = al * c
            ψ.AL[end] = al
            ψ.CR[end] = complex(c)
            ψ.AR[1] = _transpose_front(ar)
            ψ.AC[1] = _transpose_front(c * ar)
            ψ.AL[1] = ψ.AC[1] * inv(ψ.CR[1])

            C_current = complex(c)

            # update environments
            update_leftenv!(envs, ψ, H, 1)
            update_rightenv!(envs, ψ, H, 0)

            # sweep from right to left
            for pos in (length(ψ) - 1):-1:1
                ac2 = ψ.AL[pos] * _transpose_tail(ψ.AC[pos + 1])
                h_ac2 = ∂∂AC2(pos, ψ, H, envs)
                _, ac2′ = fixedpoint(h_ac2, ac2, :SR, alg_eigsolve)

                al, c, ar, = tsvd!(ac2′; trunc=alg.trscheme, alg=TensorKit.SVD())
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
            _, ac2′ = fixedpoint(h_ac2, ac2, :SR, alg_eigsolve)
            al, c, ar, = tsvd!(ac2′; trunc=alg.trscheme, alg=TensorKit.SVD())
            normalize!(c)

            ψ.AR[end] = _transpose_front(inv(ψ.CR[end - 1]) * _transpose_tail(al * c))
            ψ.AL[end] = al
            ψ.CR[end] = complex(c)
            ψ.AR[1] = _transpose_front(ar)
            ψ.AC[1] = _transpose_front(c * ar)

            update_leftenv!(envs, ψ, H, 1)
            update_rightenv!(envs, ψ, H, 0)

            # update error
            smallest = infimum(_firstspace(C_current), _firstspace(c))
            e1 = isometry(_firstspace(C_current), smallest)
            e2 = isometry(_firstspace(c), smallest)
            ϵ = norm(e2' * c * e2 - e1' * C_current * e1)

            if ϵ < alg.tol
                @infov 2 logfinish!(log, iter, ϵ)
                break
            end
            if iter == alg.maxiter
                @warnv 1 logcancel!(log, iter, ϵ)
            else
                @infov 3 logiter!(log, iter, ϵ)
            end
        end
    end

    ψ′ = InfiniteMPS(ψ.AR[1:end]; tol=alg.tol_gauge)
    nenvs = environments(ψ′, H; solver=oenvs.solver)
    return ψ′, nenvs, ϵ
end
