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

function find_groundstate!(ψ::AbstractFiniteMPS, H, alg::DMRG, envs = environments(ψ, H, ψ))
    ϵs = map(pos -> calc_galerkin(pos, ψ, H, ψ, envs), 1:length(ψ))
    ϵ = maximum(ϵs)
    log = IterLog("DMRG")
    timeroutput = TimerOutput("DMRG")
    alg.verbosity > 3 || disable_timer!(timeroutput)

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ, expectation_value(ψ, H, envs))
        for iter in 1:(alg.maxiter)
            alg_eigsolve = updatetol(alg.alg_eigsolve, iter, ϵ)

            zerovector!(ϵs)
            @timeit timeroutput "sweep" begin
                for pos in [1:(length(ψ) - 1); length(ψ):-1:2]
                    local vec
                    @timeit timeroutput "AC_eigsolve" begin
                        h = AC_hamiltonian(pos, ψ, H, ψ, envs)
                        _, vec = fixedpoint(h, ψ.AC[pos], :SR, alg_eigsolve)
                    end
                    ϵs[pos] = max(ϵs[pos], calc_galerkin(pos, ψ, H, ψ, envs))
                    @timeit timeroutput "AC_update" ψ.AC[pos] = vec
                end
            end
            ϵ = maximum(ϵs)

            ψ, envs = @timeit timeroutput "finalize" alg.finalize(
                iter, ψ, H, envs
            )::Tuple{typeof(ψ), typeof(envs)}

            if ϵ <= alg.tol
                @infov 4 timeroutput
                @infov 2 logfinish!(log, iter, ϵ, expectation_value(ψ, H, envs))
                break
            end
            if iter == alg.maxiter
                @infov 4 timeroutput
                @warnv 1 logcancel!(log, iter, ϵ, expectation_value(ψ, H, envs))
            else
                @infov 3 logiter!(log, iter, ϵ, expectation_value(ψ, H, envs))
            end
        end
    end
    return ψ, envs, ϵ
end

struct CBEDMRG{A, F} <: Algorithm
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

    selection::TruncationStrategy

    "algorithm used for [truncation](@extref MatrixAlgebraKit.TruncationStrategy) of the two-site update"
    trscheme::TruncationStrategy
end

function find_groundstate!(ψ::AbstractFiniteMPS, H::FiniteMPOHamiltonian, alg::CBEDMRG, envs = environments(ψ, H))
    ϵs = map(pos -> calc_galerkin(pos, ψ, H, ψ, envs), 1:length(ψ))
    ϵ = maximum(ϵs)
    # extra debug error measures (do not drive convergence; ϵ remains the two-site fidelity)
    ϵs_galerkin = zero(ϵs)   # local (one-site) Galerkin error
    ϵs_trunc = zero(ϵs)      # bond-truncation error (discarded weight)
    ϵs_2site = zero(ϵs)      # two-site Galerkin error (complement-projected two-site gradient)
    log = IterLog("DMRG")

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ, expectation_value(ψ, H, envs))
        for iter in 1:(alg.maxiter)
            alg_eigsolve = updatetol(alg.alg_eigsolve, iter, ϵ)

            zerovector!(ϵs)
            zerovector!(ϵs_galerkin)
            zerovector!(ϵs_trunc)
            zerovector!(ϵs_2site)
            for pos in 1:(length(ψ) - 1)
                @plansor ac2[-1 -2; -3 -4] := ψ.AC[pos][-1 -2; 1] * ψ.AR[pos + 1][1 -4; -3]

                # exact two-site update at bond (pos, pos+1), projected onto the complement
                # of the current state (this is the exact analog of the randomized SVD)
                AC2 = AC2_projection(pos, ψ, H, ψ, envs)
                NL = left_null(ψ.AC[pos])
                NR = right_null!(_transpose_tail(ψ.AR[pos + 1]; copy = true))
                g2 = adjoint(NL) * AC2 * adjoint(NR)
                # two-site Galerkin: norm of the two-site gradient in the complement, normalized
                ϵs_2site[pos] = max(ϵs_2site[pos], norm(g2) / norm(AC2))
                intermediate = normalize!(g2)
                _, _, V = svd_trunc!(intermediate; trunc = alg.selection, alg = Defaults.alg_svd())

                # expand the bond: optimal vectors at pos+1, zero weight at pos
                ar_re = V * NR
                ar_le = zerovector!(similar(ψ.AC[pos], codomain(ψ.AC[pos]) ← space(V, 1)))
                Ql, C = qr_compact!(catdomain(ψ.AC[pos], ar_le))
                AR_exp = _transpose_front(catcodomain(_transpose_tail(ψ.AR[pos + 1]), ar_re))
                ψ.AC[pos] = (Ql, normalize!(C))
                ψ.AC[pos + 1] = (C, AR_exp)

                Hac = AC_hamiltonian(pos, ψ, H, ψ, envs)
                _, AC′ = fixedpoint(Hac, ψ.AC[pos], :SR, alg_eigsolve)
                # explicit truncated SVD: ϵ_trunc is the 2-norm of the discarded singular
                # values, computed directly from the spectrum (no 1 - ‖C‖² cancellation)
                U, S, Vᴴ, ϵ_trunc = svd_trunc!(AC′; trunc = alg.trscheme, alg = Defaults.alg_svd())
                ϵs_trunc[pos] = max(ϵs_trunc[pos], ϵ_trunc)
                AL′ = U
                C = S * Vᴴ
                normalize!(C)

                # DMRG2-style error: 1 - fidelity between the two-site state before the
                # eigensolver and the truncated, optimized two-site state after it. The
                # truncated bond is internal, so this -> 0 at convergence.
                @plansor ac2′[-1 -2; -3 -4] := AL′[-1 -2; 1] * C[1; 2] * AR_exp[2 -4; -3]
                ϵs[pos] = max(ϵs[pos], abs(1 - abs(dot(ac2, ac2′))))

                ψ.AC[pos] = (AL′, C)
                ϵs_galerkin[pos] = max(ϵs_galerkin[pos], calc_galerkin(pos, ψ, H, ψ, envs))
                # @debug "CBEDMRG L→R" pos ϵ = ϵs[pos]
            end

            for pos in length(ψ):-1:2
                @plansor ac2[-1 -2; -3 -4] := ψ.AL[pos - 1][-1 -2; 1] * ψ.AC[pos][1 -4; -3]

                # exact two-site update at bond (pos-1, pos), projected onto the complement
                AC2 = AC2_projection(pos - 1, ψ, H, ψ, envs)
                NL = left_null(ψ.AL[pos - 1])
                NR = right_null!(_transpose_tail(ψ.AC[pos]; copy = true))
                g2 = adjoint(NL) * AC2 * adjoint(NR)
                # two-site Galerkin: norm of the two-site gradient in the complement, normalized
                ϵs_2site[pos - 1] = max(ϵs_2site[pos - 1], norm(g2) / norm(AC2))
                intermediate = normalize!(g2)
                U, _, _ = svd_trunc!(intermediate; trunc = alg.selection, alg = Defaults.alg_svd())

                # optimal new left vectors for AL[pos-1]; zero padding for AC[pos]'s left virtual
                Q = NL * U
                AL = ψ.AL[pos - 1]
                al_le = zerovector!(
                    similar(ψ.AC[pos], space(Q, 3)' ⊗ physicalspace(ψ.AC[pos]) ← right_virtualspace(ψ.AC[pos]))
                )
                C, Qr = lq_compact!(catcodomain(_transpose_tail(ψ.AC[pos]), _transpose_tail(al_le)))
                AL_exp = catdomain(AL, Q)
                ψ.AC[pos] = (normalize!(C), _transpose_front(Qr))
                ψ.AC[pos - 1] = (AL_exp, C)

                Hac = AC_hamiltonian(pos, ψ, H, ψ, envs)
                _, AC′ = fixedpoint(Hac, ψ.AC[pos], :SR, alg_eigsolve)
                # explicit truncated SVD: ϵ_trunc is the 2-norm of the discarded singular
                # values, computed directly from the spectrum (no 1 - ‖C‖² cancellation)
                U, S, AR′, ϵ_trunc = svd_trunc!(_transpose_tail(AC′); trunc = alg.trscheme, alg = Defaults.alg_svd())
                ϵs_trunc[pos] = max(ϵs_trunc[pos], ϵ_trunc)
                C = U * S
                normalize!(C)
                AR_mps = _transpose_front(AR′)

                # DMRG2-style error: 1 - fidelity between the two-site state before the
                # eigensolver and the truncated, optimized two-site state after it. The
                # truncated bond is internal, so this -> 0 at convergence. AR_mps is in MPS
                # form (physical in codomain) so the overlap with ac2 stays planar.
                @plansor ac2′[-1 -2; -3 -4] := AL_exp[-1 -2; 1] * C[1; 2] * AR_mps[2 -4; -3]
                ϵs[pos] = max(ϵs[pos], abs(1 - abs(dot(ac2, ac2′))))
                # @debug "CBEDMRG R→L" pos ϵ = ϵs[pos]
                ψ.AC[pos] = (C, AR_mps)
                ϵs_galerkin[pos] = max(ϵs_galerkin[pos], calc_galerkin(pos, ψ, H, ψ, envs))
            end

            ϵ = maximum(ϵs)
            @infov 2 "CBEDMRG errors" iter fidelity = ϵ local_galerkin = maximum(ϵs_galerkin) twosite_galerkin = maximum(ϵs_2site) truncation = maximum(ϵs_trunc)

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

    "algorithm used for [truncation](@extref MatrixAlgebraKit.TruncationStrategy) of the two-site update"
    trscheme::TruncationStrategy

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

function find_groundstate!(ψ::AbstractFiniteMPS, H, alg::DMRG2, envs = environments(ψ, H, ψ))
    ϵs = map(pos -> calc_galerkin(pos, ψ, H, ψ, envs), 1:length(ψ))
    ϵ = maximum(ϵs)
    log = IterLog("DMRG2")
    timeroutput = TimerOutput("DMRG2")
    alg.verbosity > 3 || disable_timer!(timeroutput)

    LoggingExtras.withlevel(; alg.verbosity) do
        for iter in 1:(alg.maxiter)
            alg_eigsolve = updatetol(alg.alg_eigsolve, iter, ϵ)
            zerovector!(ϵs)

            @timeit timeroutput "sweep" begin
                # left to right sweep
                for pos in 1:(length(ψ) - 1)
                    local ac2, newA2center, al, c, ar
                    @timeit timeroutput "AC2_eigsolve" begin
                        @plansor ac2[-1 -2; -3 -4] := ψ.AC[pos][-1 -2; 1] * ψ.AR[pos + 1][1 -4; -3]
                        Hac2 = AC2_hamiltonian(pos, ψ, H, ψ, envs)
                        _, newA2center = fixedpoint(Hac2, ac2, :SR, alg_eigsolve)
                    end
                    @timeit timeroutput "svd_trunc" begin
                        al, c, ar = svd_trunc!(newA2center; trunc = alg.trscheme, alg = alg.alg_svd)
                        normalize!(c)
                        v = @plansor ac2[1 2; 3 4] * conj(al[1 2; 5]) * conj(c[5; 6]) * conj(ar[6; 3 4])
                        ϵs[pos] = max(ϵs[pos], abs(1 - abs(v)))
                    end
                    @timeit timeroutput "update_AC" begin
                        ψ.AC[pos] = (al, complex(c))
                        ψ.AC[pos + 1] = (complex(c), _transpose_front(ar))
                    end
                end

                # right to left sweep
                for pos in (length(ψ) - 2):-1:1
                    local ac2, newA2center, al, c, ar
                    @timeit timeroutput "AC2_eigsolve" begin
                        @plansor ac2[-1 -2; -3 -4] := ψ.AL[pos][-1 -2; 1] * ψ.AC[pos + 1][1 -4; -3]
                        Hac2 = AC2_hamiltonian(pos, ψ, H, ψ, envs)
                        _, newA2center = fixedpoint(Hac2, ac2, :SR, alg_eigsolve)
                    end
                    @timeit timeroutput "svd_trunc" begin
                        al, c, ar = svd_trunc!(newA2center; trunc = alg.trscheme, alg = alg.alg_svd)
                        normalize!(c)
                        v = @plansor ac2[1 2; 3 4] * conj(al[1 2; 5]) * conj(c[5; 6]) * conj(ar[6; 3 4])
                        ϵs[pos] = max(ϵs[pos], abs(1 - abs(v)))
                    end
                    @timeit timeroutput "update_AC" begin
                        ψ.AC[pos + 1] = (complex(c), _transpose_front(ar))
                        ψ.AC[pos] = (al, complex(c))
                    end
                end

            ϵ = maximum(ϵs)
            ψ, envs = @timeit timeroutput "finalize" alg.finalize(
                iter, ψ, H, envs
            )::Tuple{typeof(ψ), typeof(envs)}

            if ϵ <= alg.tol
                @infov 4 timeroutput
                @infov 2 logfinish!(log, iter, ϵ, expectation_value(ψ, H, envs))
                break
            end
            if iter == alg.maxiter
                @infov 4 timeroutput
                @warnv 1 logcancel!(log, iter, ϵ, expectation_value(ψ, H, envs))
            else
                @infov 3 logiter!(log, iter, ϵ, expectation_value(ψ, H, envs))
            end
        end
    end
    return ψ, envs, ϵ
end

function find_groundstate(ψ, H, alg::Union{DMRG, DMRG2, CBEDMRG}, envs...; kwargs...)
    return find_groundstate!(copy(ψ), H, alg, envs...; kwargs...)
end
