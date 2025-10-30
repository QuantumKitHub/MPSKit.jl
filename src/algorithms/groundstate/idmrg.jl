"""
$(TYPEDEF)

Single site infinite DMRG algorithm for finding the dominant eigenvector.

## Fields

$(TYPEDFIELDS)
"""
@kwdef struct IDMRG{A} <: Algorithm
    "tolerance for convergence criterium"
    tol::Float64 = Defaults.tol

    "maximal amount of iterations"
    maxiter::Int = Defaults.maxiter

    "setting for how much information is displayed"
    verbosity::Int = Defaults.verbosity

    "algorithm used for gauging the MPS"
    alg_gauge = Defaults.alg_gauge()

    "algorithm used for the eigenvalue solvers"
    alg_eigsolve::A = Defaults.alg_eigsolve()
end

function find_groundstate(ost::AbstractInfiniteMPS, H, alg::IDMRG, envs = environments(ost, H))
    ϵ::Float64 = calc_galerkin(ost, H, ost, envs)
    ψ = copy(ost)
    log = IterLog("IDMRG")
    local iter, E_current

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 begin
            E_current = expectation_value(ψ, H, envs)
            loginit!(log, ϵ, E_current)
        end
        for outer iter in 1:(alg.maxiter)
            alg_eigsolve = updatetol(alg.alg_eigsolve, iter, ϵ)
            C_current = ψ.C[0]

            # left to right sweep
            for pos in 1:length(ψ)
                h = AC_hamiltonian(pos, ψ, H, ψ, envs)
                _, ψ.AC[pos] = fixedpoint(h, ψ.AC[pos], :SR, alg_eigsolve)
                if pos == length(ψ)
                    # AC needed in next sweep
                    ψ.AL[pos], ψ.C[pos] = left_orth(ψ.AC[pos])
                else
                    ψ.AL[pos], ψ.C[pos] = left_orth!(ψ.AC[pos])
                end
                transfer_leftenv!(envs, ψ, H, ψ, pos + 1)
            end

            # right to left sweep
            for pos in length(ψ):-1:1
                h = AC_hamiltonian(pos, ψ, H, ψ, envs)
                _, ψ.AC[pos] = fixedpoint(h, ψ.AC[pos], :SR, alg_eigsolve)

                ψ.C[pos - 1], temp = right_orth!(_transpose_tail(ψ.AC[pos]))
                ψ.AR[pos] = _transpose_front(temp)

                transfer_rightenv!(envs, ψ, H, ψ, pos - 1)
            end

            ϵ = norm(C_current - ψ.C[0])

            if ϵ < alg.tol
                @infov 2 begin
                    E_next = expectation_value(ψ, H, envs)
                    ΔE = E_next - E_current
                    E_current = E_next
                    logfinish!(log, iter, ϵ, ΔE)
                end
                break
            end
            if iter == alg.maxiter
                @warnv 1 begin
                    E_next = expectation_value(ψ, H, envs)
                    ΔE = E_next - E_current
                    E_current = E_next
                    logcancel!(log, iter, ϵ, ΔE)
                end
            else
                @infov 3 begin
                    E_next = expectation_value(ψ, H, envs)
                    ΔE = E_next - E_current
                    E_current = E_next
                    logiter!(log, iter, ϵ, ΔE)
                end
            end
        end
    end

    alg_gauge = updatetol(alg.alg_gauge, iter, ϵ)
    ψ′ = InfiniteMPS(ψ.AR[1:end]; alg_gauge.tol, alg_gauge.maxiter)
    recalculate!(envs, ψ′, H, ψ′)

    return ψ′, envs, ϵ
end

"""
$(TYPEDEF)

Two-site infinite DMRG algorithm for finding the dominant eigenvector.

## Fields

$(TYPEDFIELDS)
"""
@kwdef struct IDMRG2{A, S} <: Algorithm
    "tolerance for convergence criterium"
    tol::Float64 = Defaults.tol

    "maximal amount of iterations"
    maxiter::Int = Defaults.maxiter

    "setting for how much information is displayed"
    verbosity::Int = Defaults.verbosity

    "algorithm used for gauging the MPS"
    alg_gauge = Defaults.alg_gauge()

    "algorithm used for the eigenvalue solvers"
    alg_eigsolve::A = Defaults.alg_eigsolve()

    "algorithm used for the singular value decomposition"
    alg_svd::S = Defaults.alg_svd()

    "algorithm used for [truncation](@extref MatrixAlgebraKit.TruncationStrategy) of the two-site update"
    trscheme::TruncationStrategy
end

function find_groundstate(ost::AbstractInfiniteMPS, H, alg::IDMRG2, envs = environments(ost, H))
    length(ost) < 2 && throw(ArgumentError("unit cell should be >= 2"))
    ϵ::Float64 = calc_galerkin(ost, H, ost, envs)

    ψ = copy(ost)
    log = IterLog("IDMRG2")
    local iter

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ)
        for outer iter in 1:(alg.maxiter)
            alg_eigsolve = updatetol(alg.alg_eigsolve, iter, ϵ)
            C_current = ψ.C[0]

            # sweep from left to right
            for pos in 1:(length(ψ) - 1)
                ac2 = AC2(ψ, pos; kind = :ACAR)
                h_ac2 = AC2_hamiltonian(pos, ψ, H, ψ, envs)
                _, ac2′ = fixedpoint(h_ac2, ac2, :SR, alg_eigsolve)

                al, c, ar = svd_trunc!(ac2′; trunc = alg.trscheme, alg = alg.alg_svd)
                normalize!(c)

                ψ.AL[pos] = al
                ψ.C[pos] = complex(c)
                ψ.AR[pos + 1] = _transpose_front(ar)
                ψ.AC[pos + 1] = _transpose_front(c * ar)

                transfer_leftenv!(envs, ψ, H, ψ, pos + 1)
                transfer_rightenv!(envs, ψ, H, ψ, pos)
            end

            # update the edge
            ψ.AL[end] = ψ.AC[end] / ψ.C[end]
            ψ.AC[1] = _mul_tail(ψ.AL[1], ψ.C[1])
            ac2 = AC2(ψ, 0; kind = :ALAC)
            h_ac2 = AC2_hamiltonian(0, ψ, H, ψ, envs)
            _, ac2′ = fixedpoint(h_ac2, ac2, :SR, alg_eigsolve)

            al, c, ar = svd_trunc!(ac2′; trunc = alg.trscheme, alg = alg.alg_svd)
            normalize!(c)

            ψ.AL[end] = al
            ψ.C[end] = complex(c)
            ψ.AR[1] = _transpose_front(ar)

            ψ.AC[end] = _mul_tail(al, c)
            ψ.AC[1] = _transpose_front(c * ar)
            ψ.AL[1] = ψ.AC[1] / ψ.C[1]

            C_current = complex(c)

            # update environments
            transfer_leftenv!(envs, ψ, H, ψ, 1)
            transfer_rightenv!(envs, ψ, H, ψ, 0)

            # sweep from right to left
            for pos in (length(ψ) - 1):-1:1
                ac2 = AC2(ψ, pos; kind = :ALAC)
                h_ac2 = AC2_hamiltonian(pos, ψ, H, ψ, envs)
                _, ac2′ = fixedpoint(h_ac2, ac2, :SR, alg_eigsolve)

                al, c, ar = svd_trunc!(ac2′; trunc = alg.trscheme, alg = alg.alg_svd)
                normalize!(c)

                ψ.AL[pos] = al
                ψ.AC[pos] = _mul_tail(al, c)
                ψ.C[pos] = complex(c)
                ψ.AR[pos + 1] = _transpose_front(ar)
                ψ.AC[pos + 1] = _transpose_front(c * ar)

                transfer_leftenv!(envs, ψ, H, ψ, pos + 1)
                transfer_rightenv!(envs, ψ, H, ψ, pos)
            end

            # update the edge
            ψ.AC[end] = _mul_front(ψ.C[end - 1], ψ.AR[end])
            ψ.AR[1] = _transpose_front(ψ.C[end] \ _transpose_tail(ψ.AC[1]))
            ac2 = AC2(ψ, 0; kind = :ACAR)
            h_ac2 = AC2_hamiltonian(0, ψ, H, ψ, envs)
            _, ac2′ = fixedpoint(h_ac2, ac2, :SR, alg_eigsolve)
            al, c, ar = svd_trunc!(ac2′; trunc = alg.trscheme, alg = alg.alg_svd)
            normalize!(c)

            ψ.AL[end] = al
            ψ.C[end] = complex(c)
            ψ.AR[1] = _transpose_front(ar)

            ψ.AR[end] = _transpose_front(ψ.C[end - 1] \ _transpose_tail(al * c))
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

    alg_gauge = updatetol(alg.alg_gauge, iter, ϵ)
    ψ′ = InfiniteMPS(ψ.AR[1:end]; alg_gauge.tol, alg_gauge.maxiter)
    recalculate!(envs, ψ′, H, ψ′)

    return ψ′, envs, ϵ
end
