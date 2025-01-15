"""
$(TYPEDEF)

Single site infinite DMRG algorithm for finding the dominant eigenvector.

## Fields

$(TYPEDFIELDS)
"""
struct IDMRG1{A} <: Algorithm
    "tolerance for convergence criterium"
    tol::Float64
    "tolerance for gauging algorithm"
    tol_gauge::Float64
    "algorithm used for the eigenvalue solvers"
    eigalg::A
    "maximal amount of iterations"
    maxiter::Int
    "setting for how much information is displayed"
    verbosity::Int
end
function IDMRG1(; tol=Defaults.tol, tol_gauge=Defaults.tolgauge, eigalg=(;),
                maxiter=Defaults.maxiter, verbosity=Defaults.verbosity)
    eigalg′ = eigalg isa NamedTuple ? Defaults.alg_eigsolve(; eigalg...) : eigalg
    return IDMRG1{typeof(eigalg′)}(tol, tol_gauge, eigalg′, maxiter, verbosity)
end

function find_groundstate(ost::InfiniteMPS, H, alg::IDMRG1, envs=environments(ost, H))
    ϵ::Float64 = calc_galerkin(ost, H, ost, envs)
    ψ = copy(ost)
    log = IterLog("IDMRG")

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ, expectation_value(ψ, H, envs))
        for iter in 1:(alg.maxiter)
            alg_eigsolve = updatetol(alg.eigalg, iter, ϵ)
            C_current = ψ.C[0]

            # left to right sweep
            for pos in 1:length(ψ)
                h = ∂∂AC(pos, ψ, H, envs)
                _, ψ.AC[pos] = fixedpoint(h, ψ.AC[pos], :SR, alg_eigsolve)
                ψ.AL[pos], ψ.C[pos] = leftorth!(ψ.AC[pos])
                transfer_leftenv!(envs, ψ, H, ψ, pos + 1)
            end

            # right to left sweep
            for pos in length(ψ):-1:1
                h = ∂∂AC(pos, ψ, H, envs)
                _, ψ.AC[pos] = fixedpoint(h, ψ.AC[pos], :SR, alg_eigsolve)

                ψ.C[pos - 1], temp = rightorth!(_transpose_tail(ψ.AC[pos]))
                ψ.AR[pos] = _transpose_front(temp)

                transfer_rightenv!(envs, ψ, H, ψ, pos - 1)
            end

            ϵ = norm(C_current - ψ.C[0])

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
    recalculate!(envs, nst, H, nst)

    return nst, envs, ϵ
end

"""
$(TYPEDEF)

Two-site infinite DMRG algorithm for finding the dominant eigenvector.

## Fields

$(TYPEDFIELDS)
- `tol::Float64`: tolerance for convergence criterium
- `tol_gauge::Float64`: tolerance for gauging algorithm
- `eigalg::A`: eigensolver algorithm
- `maxiter::Int`: maximum number of outer iterations
- `verbosity::Int`: display progress information
- `trscheme::TruncationScheme`: truncation algorithm for [`tsvd`][TensorKit.tsvd](@extref)
"""
struct IDMRG2{A} <: Algorithm
    "tolerance for convergence criterium"
    tol::Float64
    "tolerance for gauging algorithm"
    tol_gauge::Float64
    "algorithm used for the eigenvalue solvers"
    eigalg::A
    "maximal amount of iterations"
    maxiter::Int
    "setting for how much information is displayed"
    verbosity::Int
    "algorithm used for truncation of the two-site update"
    trscheme::TruncationScheme
end
function IDMRG2(; tol=Defaults.tol, tol_gauge=Defaults.tolgauge, eigalg=(;),
                maxiter=Defaults.maxiter, verbosity=Defaults.verbosity,
                trscheme=truncerr(1e-6))
    eigalg′ = eigalg isa NamedTuple ? Defaults.alg_eigsolve(; eigalg...) : eigalg
    return IDMRG2{typeof(eigalg′)}(tol, tol_gauge, eigalg′, maxiter, verbosity, trscheme)
end

function find_groundstate(ost::InfiniteMPS, H, alg::IDMRG2, envs=environments(ost, H))
    length(ost) < 2 && throw(ArgumentError("unit cell should be >= 2"))
    ϵ::Float64 = calc_galerkin(ost, H, ost, envs)

    ψ = copy(ost)
    log = IterLog("IDMRG2")

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ)
        for iter in 1:(alg.maxiter)
            alg_eigsolve = updatetol(alg.eigalg, iter, ϵ)
            C_current = ψ.C[0]

            # sweep from left to right
            for pos in 1:(length(ψ) - 1)
                ac2 = ψ.AC[pos] * _transpose_tail(ψ.AR[pos + 1])
                h_ac2 = ∂∂AC2(pos, ψ, H, envs)
                _, ac2′ = fixedpoint(h_ac2, ac2, :SR, alg_eigsolve)

                al, c, ar, = tsvd!(ac2′; trunc=alg.trscheme, alg=TensorKit.SVD())
                normalize!(c)

                ψ.AL[pos] = al
                ψ.C[pos] = complex(c)
                ψ.AR[pos + 1] = _transpose_front(ar)
                ψ.AC[pos + 1] = _transpose_front(c * ar)

                transfer_leftenv!(envs, ψ, H, ψ, pos + 1)
                transfer_rightenv!(envs, ψ, H, ψ, pos)
            end

            # update the edge
            @plansor ac2[-1 -2; -3 -4] := ψ.AC[end][-1 -2; 1] * inv(ψ.C[0])[1; 2] *
                                          ψ.AL[1][2 -4; 3] * ψ.C[1][3; -3]
            h_ac2 = ∂∂AC2(0, ψ, H, envs)
            _, ac2′ = fixedpoint(h_ac2, ac2, :SR, alg_eigsolve)

            al, c, ar, = tsvd!(ac2′; trunc=alg.trscheme, alg=TensorKit.SVD())
            normalize!(c)

            ψ.AC[end] = al * c
            ψ.AL[end] = al
            ψ.C[end] = complex(c)
            ψ.AR[1] = _transpose_front(ar)
            ψ.AC[1] = _transpose_front(c * ar)
            ψ.AL[1] = ψ.AC[1] * inv(ψ.C[1])

            C_current = complex(c)

            # update environments
            transfer_leftenv!(envs, ψ, H, ψ, 1)
            transfer_rightenv!(envs, ψ, H, ψ, 0)

            # sweep from right to left
            for pos in (length(ψ) - 1):-1:1
                ac2 = ψ.AL[pos] * _transpose_tail(ψ.AC[pos + 1])
                h_ac2 = ∂∂AC2(pos, ψ, H, envs)
                _, ac2′ = fixedpoint(h_ac2, ac2, :SR, alg_eigsolve)

                al, c, ar, = tsvd!(ac2′; trunc=alg.trscheme, alg=TensorKit.SVD())
                normalize!(c)

                ψ.AL[pos] = al
                ψ.AC[pos] = al * c
                ψ.C[pos] = complex(c)
                ψ.AR[pos + 1] = _transpose_front(ar)
                ψ.AC[pos + 1] = _transpose_front(c * ar)

                transfer_leftenv!(envs, ψ, H, ψ, pos + 1)
                transfer_rightenv!(envs, ψ, H, ψ, pos)
            end

            # update the edge
            @plansor ac2[-1 -2; -3 -4] := ψ.C[end - 1][-1; 1] * ψ.AR[end][1 -2; 2] *
                                          inv(ψ.C[end])[2; 3] * ψ.AC[1][3 -4; -3]
            h_ac2 = ∂∂AC2(0, ψ, H, envs)
            _, ac2′ = fixedpoint(h_ac2, ac2, :SR, alg_eigsolve)
            al, c, ar, = tsvd!(ac2′; trunc=alg.trscheme, alg=TensorKit.SVD())
            normalize!(c)

            ψ.AR[end] = _transpose_front(inv(ψ.C[end - 1]) * _transpose_tail(al * c))
            ψ.AL[end] = al
            ψ.C[end] = complex(c)
            ψ.AR[1] = _transpose_front(ar)
            ψ.AC[1] = _transpose_front(c * ar)

            transfer_leftenv!(envs, ψ, H, ψ, 1)
            transfer_rightenv!(envs, ψ, H, ψ, 0)

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
    recalculate!(envs, ψ′, H, ψ′)

    return ψ′, envs, ϵ
end
