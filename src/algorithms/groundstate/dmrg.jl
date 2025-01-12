"""
$(TYPEDEF)

Single-site DMRG algorithm for finding the dominant eigenvector.

## Fields

$(TYPEDFIELDS)
"""
struct DMRG{A,F} <: Algorithm
    "tolerance for convergence criterium"
    tol::Float64
    "maximal amount of iterations"
    maxiter::Int
    "callback function applied after each iteration, of signature `finalize(iter, ψ, H, envs) -> ψ, envs"
    finalize::F
    "setting for how much information is displayed"
    verbosity::Int
    "algorithm used for the eigenvalue solvers"
    eigalg::A
end
function DMRG(; tol=Defaults.tol, maxiter=Defaults.maxiter, eigalg=(;),
              verbosity=Defaults.verbosity, finalize=Defaults._finalize)
    eigalg′ = eigalg isa NamedTuple ? Defaults.alg_eigsolve(; eigalg...) : eigalg
    return DMRG{typeof(eigalg′),typeof(finalize)}(tol, maxiter, finalize, verbosity,
                                                  eigalg′)
end

function find_groundstate!(ψ::AbstractFiniteMPS, H, alg::DMRG, envs=environments(ψ, H))
    ϵs = map(pos -> calc_galerkin(pos, ψ, H, ψ, envs), 1:length(ψ))
    ϵ = maximum(ϵs)
    log = IterLog("DMRG")

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ, expectation_value(ψ, H, envs))
        for iter in 1:(alg.maxiter)
            alg_eigsolve = updatetol(alg.eigalg, iter, ϵ)

            zerovector!(ϵs)
            for pos in [1:(length(ψ) - 1); length(ψ):-1:2]
                h = ∂∂AC(pos, ψ, H, envs)
                _, vec = fixedpoint(h, ψ.AC[pos], :SR, alg_eigsolve)
                ϵs[pos] = max(ϵs[pos], calc_galerkin(pos, ψ, H, ψ, envs))
                ψ.AC[pos] = vec
            end
            ϵ = maximum(ϵs)

            ψ, envs = alg.finalize(iter, ψ, H, envs)::Tuple{typeof(ψ),typeof(envs)}

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
struct DMRG2{A,F} <: Algorithm
    "tolerance for convergence criterium"
    tol::Float64
    "maximal amount of iterations"
    maxiter::Int
    "callback function applied after each iteration, of signature `finalize(iter, ψ, H, envs) -> ψ, envs`"
    finalize::F
    "setting for how much information is displayed"
    verbosity::Int

    "algorithm used for the eigenvalue solvers"
    eigalg::A
    "algorithm used for truncation of the two-site update"
    trscheme::TruncationScheme
end
# TODO: find better default truncation
function DMRG2(; tol=Defaults.tol, maxiter=Defaults.maxiter, eigalg=(;),
               trscheme=truncerr(1e-6), verbosity=Defaults.verbosity,
               finalize=Defaults._finalize)
    eigalg′ = eigalg isa NamedTuple ? Defaults.alg_eigsolve(; eigalg...) : eigalg
    return DMRG2{typeof(eigalg′),typeof(finalize)}(tol, maxiter, finalize, verbosity,
                                                   eigalg′, trscheme)
end

function find_groundstate!(ψ::AbstractFiniteMPS, H, alg::DMRG2, envs=environments(ψ, H))
    ϵs = map(pos -> calc_galerkin(pos, ψ, H, ψ, envs), 1:length(ψ))
    ϵ = maximum(ϵs)
    log = IterLog("DMRG2")

    LoggingExtras.withlevel(; alg.verbosity) do
        for iter in 1:(alg.maxiter)
            alg_eigsolve = updatetol(alg.eigalg, iter, ϵ)
            zerovector!(ϵs)

            # left to right sweep
            for pos in 1:(length(ψ) - 1)
                @plansor ac2[-1 -2; -3 -4] := ψ.AC[pos][-1 -2; 1] * ψ.AR[pos + 1][1 -4; -3]

                _, newA2center = fixedpoint(∂∂AC2(pos, ψ, H, envs), ac2, :SR, alg_eigsolve)

                al, c, ar, = tsvd!(newA2center; trunc=alg.trscheme, alg=TensorKit.SVD())
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

                _, newA2center = fixedpoint(∂∂AC2(pos, ψ, H, envs), ac2, :SR, alg_eigsolve)

                al, c, ar, = tsvd!(newA2center; trunc=alg.trscheme, alg=TensorKit.SVD())
                normalize!(c)
                v = @plansor ac2[1 2; 3 4] * conj(al[1 2; 5]) * conj(c[5; 6]) *
                             conj(ar[6; 3 4])
                ϵs[pos] = max(ϵs[pos], abs(1 - abs(v)))

                ψ.AC[pos + 1] = (complex(c), _transpose_front(ar))
                ψ.AC[pos] = (al, complex(c))
            end

            ϵ = maximum(ϵs)
            ψ, envs = alg.finalize(iter, ψ, H, envs)::Tuple{typeof(ψ),typeof(envs)}

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

function find_groundstate(ψ, H, alg::Union{DMRG,DMRG2}, envs...)
    return find_groundstate!(copy(ψ), H, alg, envs...)
end
