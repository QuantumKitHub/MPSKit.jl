# whether the gauge algorithm truncates the bond (SVD-based) or preserves it (QR-based, a plain
# center-move as in textbook single-site DMRG)
_truncates(::MatrixAlgebraKit.AbstractAlgorithm) = false
_truncates(::MatrixAlgebraKit.TruncatedAlgorithm) = true
_truncates(alg::DMRG3S) = _truncates(alg.alg_gauge)

# a no-truncation `trunc` selects a (bond-preserving) QR gauge, anything else a truncated SVD
_build_inner_gauge(trunc, alg_svd, alg_orth) =
    trunc isa MatrixAlgebraKit.NoTruncation ? alg_orth :
    MatrixAlgebraKit.TruncatedAlgorithm(alg_svd, trunc)

_expands(alg) = false
_expands(::DMRG3S) = true

"""
$(TYPEDEF)

Density Matrix Renormalization Group algorithm for finding the dominant eigenvector.

Each site update is, in order: (1) an optional bond expansion (`alg_expand`), (2) a single-site
eigensolve, and (3) a gauge step (`alg_gauge`). With the defaults (`alg_expand = nothing` and
`alg_gauge = nothing`, a non-truncating QR gauge derived from `trunc = notrunc()`) this is
textbook single-site DMRG, which cannot change the bond dimension. Setting `alg_expand` to a
bond-expansion algorithm (e.g. [`OptimalExpand`](@ref), [`RandExpand`](@ref), [`SketchedExpand`](@ref))
expands the bond with directions orthogonal to the current state ahead of each eigensolve,
recovering Controlled Bond Expansion (CBE) DMRG. Setting `alg_gauge` to a bond-expanding gauge
algorithm (e.g. [`DMRG3S`](@ref)) instead expands the bond as part of the gauge step, after the
eigensolve. Either way, a truncating gauge (see below) is then desirable to cut the enlarged
bond back down.

# Choosing the gauge

By default, `alg_gauge` is built for you from `trunc`/`alg_svd`/`alg_orth`: `trunc =
notrunc()` (the default) gives a QR decomposition (`alg_orth`, [`Householder`](@extref
MatrixAlgebraKit.Householder) by default), any other `trunc` gives a truncated SVD (`alg_svd`
with that `trunc`).

```julia
DMRG()                            # QR gauge, no truncation
DMRG(; trunc = truncdim(50))   # truncated SVD gauge
```

To use a bond-expanding gauge such as [`DMRG3S`](@ref), pass it directly as `alg_gauge`; `trunc`
etc. are still routed through to build the *inner* gauge it wraps, exactly as above:

```julia
DMRG(; alg_gauge = DMRG3S(0.1, ExponentialDecay(0.7)), trunc = truncdim(50))
```

If `alg_gauge` is instead given with its inner gauge already set (e.g. `DMRG3S(0.1, sched,
some_gauge)`), `trunc`/`alg_svd`/`alg_orth` must be left at their defaults — passing both is an
error, since it leaves two conflicting sources for the same setting.

# Fields

$(TYPEDFIELDS)

# See also

Used as the `algorithm` argument of [`find_groundstate`](@ref) and [`approximate`](@ref).
"""
struct DMRG{A, F, E, G, B} <: Algorithm
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

    "gauge algorithm applied after each local update: `NoExpand` for a plain gauge step (a QR
    algorithm with no truncation, or a truncated SVD), or an algorithm that additionally expands
    the bond beforehand (e.g. [`DMRG3S`](@ref))"
    alg_gauge::G

    "backend for tensor contractions and index manipulations"
    backend::B
end
function DMRG(;
        tol = Defaults.tol, maxiter = Defaults.maxiter, alg_eigsolve = (;),
        verbosity = Defaults.verbosity, finalize = Defaults._finalize,
        alg_expand = nothing, alg_gauge = nothing, trunc = nothing,
        alg_svd = Defaults.alg_svd(), alg_orth = Defaults.alg_orth(),
        backend = Defaults.backend()
    )
    # single-site DMRG defaults to the per-bond adaptive controller (`AdaptiveKrylov`); pass
    # `alg_eigsolve = (; adaptive = false, ...)` to opt out (the splat overrides the default).
    alg_eigsolve′ = alg_eigsolve isa NamedTuple ?
        Defaults.alg_eigsolve(; adaptive = true, alg_eigsolve...) : alg_eigsolve

    if isnothing(alg_gauge) || isnothing(alg_gauge.alg_gauge)
        trunc = something(trunc, notrunc()) # enforce trunc default here
        inner_gauge = _build_inner_gauge(trunc, alg_svd, alg_orth)
        alg_gauge = set_alg_gauge(alg_gauge, inner_gauge)
    else
        isnothing(trunc) || throw(
            ArgumentError(
                "`trunc` was given together with an `alg_gauge` that already carries its own " *
                    "gauge algorithm (e.g. `DMRG3S(noise, schedule, some_gauge)`); set the truncation " *
                    "via one or the other, not both."
            )
        )
    end

    if (!isnothing(alg_expand) || _expands(alg_gauge)) && !_truncates(alg_gauge)
        @warn "DMRG with a bond-expanding `alg_expand` and/or `alg_gauge` but no truncation (`trunc = notrunc()`): the bond dimension will grow unboundedly each sweep."
    end
    return DMRG(tol, maxiter, verbosity, alg_eigsolve′, finalize, alg_expand, alg_gauge, backend)
end

function local_update!(
        site, direction,
        ψ, O, alg::DMRG, envs,
        ϵ_global, ϵ_trunc, decay_rate,
        iter, timeroutput, allocator
    )
    ϵ_local = calc_galerkin(site, ψ, O, ψ, envs)

    # 1. expand
    isnothing(alg.alg_expand) ||
        @timeit timeroutput "expand" changebond!(site, direction, ψ, O, alg.alg_expand, envs)

    # 2. local update
    alg_eigsolve = adapt_solver(alg.alg_eigsolve; decay_rate, g_local = ϵ_local, g_global = ϵ_global, eps_trunc = ϵ_trunc)
    ac_old = ψ.AC[site]
    λ, AC′, info = @timeit timeroutput "AC_eigsolve" begin
        H_effective = AC_hamiltonian(site, ψ, O, ψ, envs; alg.backend, allocator)
        fixedpoint(H_effective, ac_old, :SR, alg_eigsolve)
    end

    alg_gauge = _update_alg_gauge(alg.alg_gauge, iter, ϵ_global)

    # 3. gauge
    ψ, ϵ_trunc = @timeit timeroutput "gauge" gauge!(
        ψ, site, direction, O, envs, AC′, alg_gauge;
        normalize = true, alg.backend, allocator
    )

    # 4. bookkeeping: measured contraction factor per matvec, kept a strict contraction in (0, 1)
    decay_rate = clamp((first(info.normres) / ϵ_local)^(1 / max(1, info.numops)), 1.0e-3, 0.999)

    @debug "DMRG local update" site direction numops = info.numops normres = first(info.normres) krylovdim = alg_eigsolve.krylovdim maxiter = alg_eigsolve.maxiter tol = alg_eigsolve.tol decay_rate ϵ_local

    return ψ, ϵ_local, ϵ_trunc, decay_rate
end

"""
$(TYPEDEF)

Two-site DMRG algorithm for finding the dominant eigenvector.

# Fields

$(TYPEDFIELDS)

# See also

Used as the `algorithm` argument of [`find_groundstate`](@ref) and [`approximate`](@ref).
"""
struct DMRG2{A, G, F, B} <: Algorithm
    "tolerance for convergence criterium"
    tol::Float64

    "maximal amount of iterations"
    maxiter::Int

    "setting for how much information is displayed"
    verbosity::Int

    "algorithm used for the eigenvalue solvers"
    alg_eigsolve::A

    "factorization used for the post-update gauge: a truncated SVD (`alg_svd` with `trunc`)"
    alg_gauge::G

    "callback function applied after each iteration, of signature `finalize(iter, ψ, H, envs) -> ψ, envs`"
    finalize::F

    "backend for tensor contractions and index manipulations"
    backend::B
end
# TODO: find better default truncation
function DMRG2(;
        tol = Defaults.tol, maxiter = Defaults.maxiter, verbosity = Defaults.verbosity,
        alg_eigsolve = (;), alg_svd = Defaults.alg_svd(), trunc,
        finalize = Defaults._finalize,
        backend = Defaults.backend()
    )
    # two-site DMRG defaults to the per-bond adaptive controller (`AdaptiveKrylov`); pass
    # `alg_eigsolve = (; adaptive = false, ...)` to opt out (the splat overrides the default).
    alg_eigsolve′ = alg_eigsolve isa NamedTuple ?
        Defaults.alg_eigsolve(; adaptive = true, alg_eigsolve...) : alg_eigsolve
    # two-site DMRG always truncates the enlarged bond back down, so the gauge is a truncated SVD
    alg_gauge = MatrixAlgebraKit.TruncatedAlgorithm(alg_svd, trunc)
    return DMRG2(tol, maxiter, verbosity, alg_eigsolve′, alg_gauge, finalize, backend)
end

function local_update!(
        pos, direction,
        ψ, O, alg::DMRG2, envs,
        ϵ_global, ϵ_trunc, decay_rate,
        iter, timeroutput, allocator
    )
    Heff = @timeit timeroutput "AC2_hamiltonian" AC2_hamiltonian(pos, ψ, O, ψ, envs; alg.backend, allocator)

    kind = direction === Val(:right) ? :ACAR : :ALAC
    ac2 = AC2(ψ, pos; kind)
    AC2′ = normalize!(Heff * ac2)
    project_complement!(AC2′, ψ.AL[pos])
    ϵ_local = norm(AC2′)

    # 1. local two-site update
    alg_eigsolve = adapt_solver(alg.alg_eigsolve; decay_rate, g_local = ϵ_local, g_global = ϵ_global, eps_trunc = ϵ_trunc)
    newA2center, info = @timeit timeroutput "AC2_eigsolve" begin
        _, newA2center, info = fixedpoint(Heff, ac2, :SR, alg_eigsolve)
        (newA2center, info)
    end

    alg_gauge = _update_alg_gauge(alg.alg_gauge, iter, ϵ_global)

    # 2. gauge: truncated SVD split back into single-site tensors and install;
    #           the discarded weight is the truncation error
    ψ, ϵ_trunc = @timeit timeroutput "gauge" gauge2!(ψ, pos, direction, O, envs, newA2center, alg_gauge; normalize = true)

    # 3. bookkeeping: measured contraction factor per matvec, kept a strict contraction in (0, 1)
    decay_rate = clamp((first(info.normres) / ϵ_local)^(1 / max(1, info.numops)), 1.0e-3, 0.999)

    @debug "DMRG2 local update" pos direction numops = info.numops normres = first(info.normres) krylovdim = alg_eigsolve.krylovdim maxiter = alg_eigsolve.maxiter tol = alg_eigsolve.tol decay_rate ϵ_local ϵ_trunc

    return ψ, ϵ_local, ϵ_trunc, decay_rate
end

# Per-algorithm sweep geometry: single-site DMRG updates all `length(ψ)` sites (endpoints once,
# interior twice), whereas two-site DMRG2 updates the `length(ψ) - 1` bonds. `_num_updates`
# gives the number of per-update bookkeeping slots and `_sweep_ranges` the forward/backward
# index ranges; everything else in the sweep is shared.
_num_updates(::DMRG, ψ) = length(ψ)
_num_updates(::DMRG2, ψ) = length(ψ) - 1

_sweep_ranges(::DMRG, ψ) = (1:(length(ψ) - 1), length(ψ):-1:2)
_sweep_ranges(::DMRG2, ψ) = (1:(length(ψ) - 1), (length(ψ) - 2):-1:1)

inner_alg_gauge(alg::Union{DMRG, DMRG2}) = alg_gauge(alg.alg_gauge)

"""
    find_groundstate!(ψ, H, algorithm, [environments]) -> (ψ, environments, ϵ)

In-place version of [`find_groundstate`](@ref): optimize the finite MPS `ψ` for the
Hamiltonian `H`, overwriting the input state instead of working on a copy.
Currently supported for the finite-system algorithms [`DMRG`](@ref) and [`DMRG2`](@ref).

# Arguments

- `ψ::AbstractFiniteMPS`: initial guess, mutated in place
- `H`: operator for which to find the ground state
- `algorithm`: optimization algorithm
- `[environments]`: MPS environment manager

# Returns

- `ψ::AbstractFiniteMPS`: converged ground state
- `environments`: environments corresponding to the converged state
- `ϵ::Float64`: final convergence error upon terminating the algorithm
"""
function find_groundstate!(
        ψ::AbstractFiniteMPS, H, alg::Union{DMRG, DMRG2}, envs = environments(ψ, H, ψ)
    )
    # the sweep is serial, so a single allocator serves all local updates
    allocator = default_allocator(ψ, SerialScheduler())
    return _find_groundstate_sweep!(ψ, H, alg, envs, allocator)
end

function _find_groundstate_sweep!(
        ψ::AbstractFiniteMPS, H, alg::Union{DMRG, DMRG2}, envs, allocator
    )
    name = string(nameof(typeof(alg)))
    log = IterLog(name)
    timeroutput = TimerOutput(name)
    alg.verbosity > 3 || disable_timer!(timeroutput)

    Tr = real(scalartype(ψ))
    n = _num_updates(alg, ψ)
    ϵ_locals = ones(Tr, n)      # local Galerkin errors (drive both the eigensolver tol and the stop test)
    ϵ_global = maximum(ϵ_locals) # sweep-wide Galerkin error, the convergence measure
    ϵ_truncs = zeros(Tr, n)     # local truncation error
    decay_rates = zeros(n)      # local observed decay rate of eigensolver
    fwd, bwd = _sweep_ranges(alg, ψ)

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ_global, expectation_value(ψ, H, envs))
        for iter in 1:(alg.maxiter)
            @timeit timeroutput "sweep" begin
                # left-to-right
                for pos in fwd
                    ψ, ϵ_locals[pos], ϵ_truncs[pos], decay_rates[pos] =
                        local_update!(
                        pos, Val(:right),
                        ψ, H, alg, envs,
                        ϵ_global, ϵ_truncs[pos], decay_rates[pos],
                        iter, timeroutput, allocator
                    )
                    ϵ_global = maximum(ϵ_locals)
                end

                # right-to-left
                for pos in bwd
                    ψ, ϵ_locals[pos], ϵ_truncs[pos], decay_rates[pos] =
                        local_update!(
                        pos, Val(:left),
                        ψ, H, alg, envs,
                        ϵ_global, ϵ_truncs[pos], decay_rates[pos],
                        iter, timeroutput, allocator
                    )
                    ϵ_global = maximum(ϵ_locals)
                end
            end
            ϵ_global = maximum(ϵ_locals)

            ψ, envs = @timeit timeroutput "finalize" alg.finalize(
                iter, ψ, H, envs
            )::Tuple{typeof(ψ), typeof(envs)}

            # Truncation-aware convergence: the Galerkin gradient cannot drop below the level set by
            # the discarded weight, so a truncating scheme converges once `ϵ_global` reaches the
            # truncation error rather than the (unreachable) bare `tol`. With no truncation
            # (`ϵ_truncs .= 0`, e.g. single-site/QR gauge) this reduces to the plain `ϵ_global ≤ tol`.
            if ϵ_global <= max(alg.tol, maximum(ϵ_truncs))
                @infov 4 timeroutput
                @infov 2 logfinish!(log, iter, ϵ_global, expectation_value(ψ, H, envs))
                break
            end
            if iter == alg.maxiter
                @infov 4 timeroutput
                @warnv 1 logcancel!(log, iter, ϵ_global, expectation_value(ψ, H, envs))
            else
                @infov 3 logiter!(log, iter, ϵ_global, expectation_value(ψ, H, envs))
            end
        end
    end
    return ψ, envs, ϵ_global
end

function find_groundstate(ψ, H, alg::Union{DMRG, DMRG2}, envs...; kwargs...)
    return find_groundstate!(copy(ψ), H, alg, envs...; kwargs...)
end
