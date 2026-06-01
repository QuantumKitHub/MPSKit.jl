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

"""
$(TYPEDEF)

Single-site DMRG algorithm with Controlled Bond Expansion (CBE): before each single-site
optimization the bond is enriched with directions orthogonal to the current state, so the
optimization can grow the bond dimension while only ever diagonalizing a single-site effective
Hamiltonian. The expansion is delegated to a pluggable bond-expansion sub-algorithm,
allowing different selection strategies (e.g. [`OptimalExpand`](@ref),
[`RandExpand`](@ref)) to be swapped in.

## Fields

$(TYPEDFIELDS)
"""
struct CBEDMRG{A, F, E <: Algorithm} <: Algorithm
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

    "algorithm used to expand the bond dimension ahead of each single-site optimization"
    expand::E

    "algorithm used for [truncation](@extref MatrixAlgebraKit.TruncationStrategy) of the two-site update"
    trscheme::TruncationStrategy
end
function CBEDMRG(;
        tol = Defaults.tol, maxiter = Defaults.maxiter, verbosity = Defaults.verbosity,
        alg_eigsolve = (;), finalize = Defaults._finalize, expand, trscheme
    )
    alg_eigsolve′ = alg_eigsolve isa NamedTuple ? Defaults.alg_eigsolve(; alg_eigsolve...) :
        alg_eigsolve
    return CBEDMRG(tol, maxiter, verbosity, alg_eigsolve′, finalize, expand, trscheme)
end

# One CBEDMRG half-sweep: for each bond in the sweep direction, expand the bond with directions
# orthogonal to ψ, optimize the single-site tensor in the enlarged space, then truncate back down
# and move the center. The expansion is shared with `changebonds` via `changebond!`; the
# eigensolve + truncation + fidelity error are CBEDMRG-specific. The extra error measures
# (ϵs_galerkin/ϵs_trunc/ϵs_2site) are diagnostics and do not drive convergence; ϵs is the
# two-site fidelity that does.
function cbe_sweep!(
        ψ, H, envs, alg::CBEDMRG, alg_eigsolve, dir::Val{D},
        ϵs, ϵs_galerkin, ϵs_trunc, ϵs_2site
    ) where {D}
    positions = D === :right ? (1:(length(ψ) - 1)) : (length(ψ):-1:2)
    for pos in positions
        # the two MPS tensors of the pre-update two-site state, kept by reference (the expansion
        # builds new tensors rather than mutating these) for the DMRG2-style fidelity error
        # below, so the dense two-site tensor is never formed
        if D === :right
            bra_left, bra_right = ψ.AC[pos], ψ.AR[pos + 1]
            bond = pos
        else
            bra_left, bra_right = ψ.AL[pos - 1], ψ.AC[pos]
            bond = pos - 1
        end

        # enrich the bond ahead of the optimization with directions orthogonal to ψ
        _, info = changebond!(pos, dir, ψ, H, alg.expand, envs)
        ϵs_2site[bond] = max(ϵs_2site[bond], info.ϵ_2site)
        # the expanded neighbour tensor, kept for the fidelity contraction below
        neighbor = D === :right ? ψ.AR[pos + 1] : ψ.AL[pos - 1]

        # single-site optimization in the enlarged space
        Hac = AC_hamiltonian(pos, ψ, H, ψ, envs)
        _, AC′ = fixedpoint(Hac, ψ.AC[pos], :SR, alg_eigsolve)

        # explicit truncated SVD: ϵ_trunc is the 2-norm of the discarded singular values,
        # computed directly from the spectrum (no 1 - ‖C‖² cancellation). `v` is the overlap of
        # the pre-update two-site state with the truncated, optimized one, contracted directly
        # from the MPS factors in zipper order (peak intermediate is a single 3-leg tensor) so
        # neither dense two-site tensor is formed; the truncated bond is internal, so the
        # fidelity 1 - |v| -> 0 at convergence.
        if D === :right
            U, S, Vᴴ, ϵ_trunc = svd_trunc!(AC′; trunc = alg.trscheme, alg = Defaults.alg_svd())
            AL′ = U
            C = normalize!(S * Vᴴ)
            v = @plansor bra_left[1 2; 7] * conj(AL′[1 2; 5]) * conj(C[5; 6]) *
                bra_right[7 4; 3] * conj(neighbor[6 4; 3])
            ϵs[pos] = max(ϵs[pos], abs(1 - abs(v)))
            ψ.AC[pos] = (AL′, C)
        else
            U, S, AR′, ϵ_trunc = svd_trunc!(_transpose_tail(AC′); trunc = alg.trscheme, alg = Defaults.alg_svd())
            C = normalize!(U * S)
            AR_mps = _transpose_front(AR′)
            v = @plansor bra_left[1 2; 7] * conj(neighbor[1 2; 5]) * conj(C[5; 6]) *
                bra_right[7 4; 3] * conj(AR_mps[6 4; 3])
            ϵs[pos] = max(ϵs[pos], abs(1 - abs(v)))
            ψ.AC[pos] = (C, AR_mps)
        end
        ϵs_trunc[pos] = max(ϵs_trunc[pos], ϵ_trunc)
        ϵs_galerkin[pos] = max(ϵs_galerkin[pos], calc_galerkin(pos, ψ, H, ψ, envs))
    end
    return ψ
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

            cbe_sweep!(ψ, H, envs, alg, alg_eigsolve, Val(:right), ϵs, ϵs_galerkin, ϵs_trunc, ϵs_2site)
            cbe_sweep!(ψ, H, envs, alg, alg_eigsolve, Val(:left), ϵs, ϵs_galerkin, ϵs_trunc, ϵs_2site)

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
