# whether the gauge algorithm truncates the bond (SVD-based) or preserves it (QR-based, a plain
# center-move as in textbook single-site DMRG)
_truncates(::MatrixAlgebraKit.AbstractAlgorithm) = false
_truncates(::MatrixAlgebraKit.TruncatedAlgorithm) = true

"""
$(TYPEDEF)

Single-site DMRG algorithm for finding the dominant eigenvector.

Each site update is, in order: (1) an optional bond expansion (`alg_expand`), (2) a single-site
eigensolve, and (3) a gauge step (`alg_gauge`). With the defaults (`alg_expand = nothing` and a
non-truncating QR `alg_gauge`) this is textbook single-site DMRG, which cannot change the bond
dimension. Setting `alg_expand` to a bond-expansion algorithm (e.g. [`OptimalExpand`](@ref),
[`RandExpand`](@ref), [`SketchedExpand`](@ref)) enriches the bond with directions orthogonal to
the current state ahead of each eigensolve, recovering Controlled Bond Expansion (CBE) DMRG; a
truncating `alg_gauge` is then desirable to cut the enlarged bond back down.

The gauge algorithm is selected in the keyword constructor from the `trscheme` argument: when it
is `notrunc()` the gauge is a QR decomposition (`alg_orth`, [`Householder`](@extref
MatrixAlgebraKit.Householder) by default), otherwise it is a truncated SVD (`alg_svd` with the
given `trscheme`).

## Fields

$(TYPEDFIELDS)
"""
struct DMRG{A, F, E, G} <: Algorithm
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

    "algorithm used to expand the bond ahead of each local update, or `nothing` for none"
    alg_expand::E

    "factorization used for the post-update gauge: a QR algorithm (no truncation) or a truncated SVD"
    alg_gauge::G
end
function DMRG(;
        tol = Defaults.tol, maxiter = Defaults.maxiter, alg_eigsolve = (;),
        verbosity = Defaults.verbosity, finalize = Defaults._finalize,
        alg_expand = nothing, trscheme = notrunc(),
        alg_svd = Defaults.alg_svd(), alg_orth = Defaults.alg_orth()
    )
    alg_eigsolve′ = alg_eigsolve isa NamedTuple ? Defaults.alg_eigsolve(; alg_eigsolve...) :
        alg_eigsolve
    # a no-truncation `trscheme` selects a (bond-preserving) QR gauge, anything else a truncated SVD
    alg_gauge = trscheme isa MatrixAlgebraKit.NoTruncation ? alg_orth :
        MatrixAlgebraKit.TruncatedAlgorithm(alg_svd, trscheme)
    if !isnothing(alg_expand) && !_truncates(alg_gauge)
        @warn "DMRG with `alg_expand` but no truncation (`trscheme = notrunc()`): the bond dimension will grow unboundedly each sweep."
    end
    return DMRG(tol, maxiter, verbosity, alg_eigsolve′, finalize, alg_expand, alg_gauge)
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
                # left-to-right
                for pos in 1:(length(ψ) - 1)
                    local AC′
                    # convergence: pre-expansion single-site Galerkin error
                    ϵs[pos] = max(ϵs[pos], calc_galerkin(pos, ψ, H, ψ, envs))

                    # 1. expand
                    isnothing(alg.alg_expand) ||
                        @timeit timeroutput "expand" changebond!(pos, Val(:right), ψ, H, alg.alg_expand, envs)

                    # 2. local update
                    @timeit timeroutput "AC_eigsolve" begin
                        Hac = AC_hamiltonian(pos, ψ, H, ψ, envs)
                        _, AC′ = fixedpoint(Hac, ψ.AC[pos], :SR, alg_eigsolve)
                    end

                    # 3. gauge (QR center-move or truncated SVD, selected by `alg_gauge`)
                    @timeit timeroutput "gauge" left_gauge!(ψ, pos, AC′, alg.alg_gauge; normalize = true)
                end

                # right-to-left
                for pos in length(ψ):-1:2
                    local AC′
                    # convergence: pre-expansion single-site Galerkin error
                    ϵs[pos] = max(ϵs[pos], calc_galerkin(pos, ψ, H, ψ, envs))

                    # 1. expand
                    isnothing(alg.alg_expand) ||
                        @timeit timeroutput "expand" changebond!(pos, Val(:left), ψ, H, alg.alg_expand, envs)

                    # 2. local update
                    @timeit timeroutput "AC_eigsolve" begin
                        Hac = AC_hamiltonian(pos, ψ, H, ψ, envs)
                        _, AC′ = fixedpoint(Hac, ψ.AC[pos], :SR, alg_eigsolve)
                    end

                    # 3. gauge (QR center-move or truncated SVD, selected by `alg_gauge`)
                    @timeit timeroutput "gauge" right_gauge!(ψ, pos, AC′, alg.alg_gauge; normalize = true)
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

function find_groundstate(ψ, H, alg::Union{DMRG, DMRG2}, envs...; kwargs...)
    return find_groundstate!(copy(ψ), H, alg, envs...; kwargs...)
end
