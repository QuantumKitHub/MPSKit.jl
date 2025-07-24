"""
$(TYPEDEF)

Single-site DMRG algorithm for finding the dominant eigenvector.

## Fields

$(TYPEDFIELDS)
"""
struct DMRG{A, F} <: Algorithm
    "tolerance for convergence criterium"
    tol::Float64

    "maximal amount of iterations"
    maxiter::Int

    "setting for how much information is displayed"
    verbosity::Int

    "algorithm used for the eigenvalue solvers"
    alg_eigsolve::A

    "callback function applied after each iteration, of signature `finalize(iter, ψ, H, envs) -> ψ, envs`"
    finalize::F
end
function DMRG(;
        tol = Defaults.tol, maxiter = Defaults.maxiter, alg_eigsolve = (;),
        verbosity = Defaults.verbosity, finalize = Defaults._finalize
    )
    alg_eigsolve′ = alg_eigsolve isa NamedTuple ? Defaults.alg_eigsolve(; alg_eigsolve...) :
        alg_eigsolve
    return DMRG(tol, maxiter, verbosity, alg_eigsolve′, finalize)
end

function find_groundstate!(ψ::AbstractFiniteMPS, H, alg::DMRG, envs = environments(ψ, H))
    ϵs = map(pos -> calc_galerkin(pos, ψ, H, ψ, envs), 1:length(ψ))
    ϵ = maximum(ϵs)
    log = IterLog("DMRG")

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ, expectation_value(ψ, H, envs))
        for iter in 1:(alg.maxiter)
            alg_eigsolve = updatetol(alg.alg_eigsolve, iter, ϵ)

            zerovector!(ϵs)
            for pos in [1:(length(ψ) - 1); length(ψ):-1:2]
                h = AC_hamiltonian(pos, ψ, H, ψ, envs)
                _, vec = fixedpoint(h, ψ.AC[pos], :SR, alg_eigsolve)
                ϵs[pos] = max(ϵs[pos], calc_galerkin(pos, ψ, H, ψ, envs))
                ψ.AC[pos] = vec
            end
            ϵ = maximum(ϵs)

            ψ, envs = alg.finalize(iter, ψ, H, envs)::Tuple{typeof(ψ), typeof(envs)}

            if ϵ <= alg.tol
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
    return ψ, envs, ϵ
end

"""
$(TYPEDEF)

Two-site DMRG algorithm for finding the dominant eigenvector.

## Fields

$(TYPEDFIELDS)
"""
struct DMRG2{A, S, F} <: Algorithm
    "tolerance for convergence criterium"
    tol::Float64

    "maximal amount of iterations"
    maxiter::Int

    "setting for how much information is displayed"
    verbosity::Int

    "algorithm used for the eigenvalue solvers"
    alg_eigsolve::A

    "algorithm used for the singular value decomposition"
    alg_svd::S

    "algorithm used for [truncation](@extref TensorKit.tsvd) of the two-site update"
    trscheme::TruncationScheme

    "callback function applied after each iteration, of signature `finalize(iter, ψ, H, envs) -> ψ, envs`"
    finalize::F
end
# TODO: find better default truncation
function DMRG2(;
        tol = Defaults.tol, maxiter = Defaults.maxiter, verbosity = Defaults.verbosity,
        alg_eigsolve = (;), alg_svd = Defaults.alg_svd(), trscheme,
        finalize = Defaults._finalize
    )
    alg_eigsolve′ = alg_eigsolve isa NamedTuple ? Defaults.alg_eigsolve(; alg_eigsolve...) :
        alg_eigsolve
    return DMRG2(tol, maxiter, verbosity, alg_eigsolve′, alg_svd, trscheme, finalize)
end

function find_groundstate!(ψ::AbstractFiniteMPS, H, alg::DMRG2, envs = environments(ψ, H))
    ϵs = map(pos -> calc_galerkin(pos, ψ, H, ψ, envs), 1:length(ψ))
    ϵ = maximum(ϵs)
    log = IterLog("DMRG2")

    LoggingExtras.withlevel(; alg.verbosity) do
        for iter in 1:(alg.maxiter)
            alg_eigsolve = updatetol(alg.alg_eigsolve, iter, ϵ)
            zerovector!(ϵs)

            # left to right sweep
            for pos in 1:(length(ψ) - 1)
                @plansor ac2[-1 -2; -3 -4] := ψ.AC[pos][-1 -2; 1] * ψ.AR[pos + 1][1 -4; -3]
                Hac2 = AC2_hamiltonian(pos, ψ, H, ψ, envs)
                _, newA2center = fixedpoint(Hac2, ac2, :SR, alg_eigsolve)

                al, c, ar, = tsvd!(newA2center; trunc = alg.trscheme, alg = alg.alg_svd)
                normalize!(c)
                v = @plansor ac2[1 2; 3 4] * conj(al[1 2; 5]) * conj(c[5; 6]) *
                    conj(ar[6; 3 4])
                ϵs[pos] = max(ϵs[pos], abs(1 - abs(v)))

                ψ.AC[pos] = (al, complex(c))
                ψ.AC[pos + 1] = (complex(c), _transpose_front(ar))
            end

            # right to left sweep
            for pos in (length(ψ) - 2):-1:1
                @plansor ac2[-1 -2; -3 -4] := ψ.AL[pos][-1 -2; 1] * ψ.AC[pos + 1][1 -4; -3]
                Hac2 = AC2_hamiltonian(pos, ψ, H, ψ, envs)
                _, newA2center = fixedpoint(Hac2, ac2, :SR, alg_eigsolve)

                al, c, ar, = tsvd!(newA2center; trunc = alg.trscheme, alg = alg.alg_svd)
                normalize!(c)
                v = @plansor ac2[1 2; 3 4] * conj(al[1 2; 5]) * conj(c[5; 6]) *
                    conj(ar[6; 3 4])
                ϵs[pos] = max(ϵs[pos], abs(1 - abs(v)))

                ψ.AC[pos + 1] = (complex(c), _transpose_front(ar))
                ψ.AC[pos] = (al, complex(c))
            end

            ϵ = maximum(ϵs)
            ψ, envs = alg.finalize(iter, ψ, H, envs)::Tuple{typeof(ψ), typeof(envs)}

            if ϵ <= alg.tol
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
    return ψ, envs, ϵ
end

function find_groundstate(ψ, H, alg::Union{DMRG, DMRG2}, envs...; kwargs...)
    return find_groundstate!(copy(ψ), H, alg, envs...; kwargs...)
end
