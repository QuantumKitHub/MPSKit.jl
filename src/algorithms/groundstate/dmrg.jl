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
    "convergence tolerance on the Galerkin error, see [Ground state accuracy](@ref)"
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
    # Prepare this window by absorbing the finalized tensor from the previous update.
    @timeit timeroutput "advance_env" begin
        if direction === Val(:right)
            if site > 1
                absorb_site!(envs, ψ, site - 1, Val(:right))
            elseif iter > 1
                # The preceding backward sweep ended after gauging site 2.
                absorb_site!(envs, ψ, 2, Val(:left))
            end
        elseif site == length(ψ)
            # The first backward one-site window needs the final forward AL tensor.
            absorb_site!(envs, ψ, site - 1, Val(:right))
        else
            absorb_site!(envs, ψ, site + 1, Val(:left))
        end
    end
    ϵ_local = calc_galerkin(site, ψ, O, ψ, envs; alg.backend, allocator)

    # 1. expand
    if !isnothing(alg.alg_expand)
        @timeit timeroutput "expand" begin
            # Expansion changes AR[site + 1] going forward or AL[site - 1] going backward.
            forward = direction === Val(:right)
            neighbor = forward ? site + 1 : site - 1
            tensors = forward ? ψ.AR : ψ.AL
            previous = tensors[neighbor]
            changebond!(site, direction, ψ, O, alg.alg_expand, envs; allocator)
            if tensors[neighbor] !== previous
                # Only the current one-site window uses this refreshed record before
                # the opposite sweep replaces it. Saturated bonds need no refresh.
                absorb_site!(
                    envs, ψ, neighbor, forward ? Val(:left) : Val(:right);
                    local_only = true,
                )
            end
        end
    end

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
    "convergence tolerance on the Galerkin error, see [Ground state accuracy](@ref)"
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
    # The previous forward pair finalized AL[pos - 1]; the previous backward pair
    # finalized AR[pos + 2]. At reversal this also absorbs the updated terminal AR.
    # IDMRG2 can use this interior rule, but its explicit (N, 1) seam solve must refresh
    # both unit-cell boundaries separately rather than use a finite endpoint condition.
    @timeit timeroutput "advance_env" begin
        if direction === Val(:right)
            pos > 1 && absorb_site!(envs, ψ, pos - 1, Val(:right))
        else
            absorb_site!(envs, ψ, pos + 2, Val(:left))
        end
    end
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
    #           the norm of the discarded singular values is the truncation error
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

# per-bond truncation errors of the last cut from the per-update slots: single-site slot `pos`
# holds the backward cut of bond `pos - 1`, which is the last one made there
_bond_truncation_errors(::DMRG, ϵ_truncs) = ϵ_truncs[2:end]
_bond_truncation_errors(::DMRG2, ϵ_truncs) = copy(ϵ_truncs)

inner_alg_gauge(alg::Union{DMRG, DMRG2}) = alg_gauge(alg.alg_gauge)

"""
    find_groundstate!(ψ, H, algorithm, [environments]) -> (ψ, environments, info)

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
- `info::AlgorithmInfo`: how the algorithm terminated. `info.galerkin` is the Galerkin error (also
    reachable as [`convergence_measure`](@ref)) and `info.converged` whether it met the
    stopping test. `info.truncation_errors` holds, per bond, what the last cut there discarded
    (all zero for a gauge that does not truncate). See [`AlgorithmInfo`](@ref).
"""
function find_groundstate!(
        ψ::AbstractFiniteMPS, H, alg::Union{DMRG, DMRG2}, envs = environments(ψ, H, ψ)
    )
    # the sweep is serial, so a single allocator serves all local updates
    allocator = default_allocator(ψ, SerialScheduler())

    name = string(nameof(typeof(alg)))
    timeroutput = alg.verbosity > 3 ? TimerOutput(name) : NoTimerOutput()

    return find_groundstate_sweep!(ψ, H, alg, envs, allocator, timeroutput)
end

# A solver iteration is one complete forward/backward sweep. The vectors retain the
# local history used by the adaptive solvers between sweeps.
struct DMRGState{S, O, E, R, V, D, T, A}
    mps::S
    operator::O
    envs::E
    iter::Int
    ϵ::R
    local_errors::V
    truncation_errors::V
    decay_rates::D
    timeroutput::T
    allocator::A
end

function DMRGState(ψ, H, alg::Union{DMRG, DMRG2}, envs, allocator, timeroutput)
    Tr = real(scalartype(ψ))
    n = _num_updates(alg, ψ)
    local_errors = ones(Tr, n)
    two_site = alg isa DMRG2 || !isnothing(alg.alg_expand)
    cache = initialize_sweep_cache(ψ, H, envs; alg.backend, allocator, one_site = alg isa DMRG, two_site)
    return DMRGState(
        ψ, H, cache, 0, maximum(local_errors), local_errors, zeros(Tr, n),
        zeros(n), timeroutput, allocator,
    )
end

function sweep!(it::IterativeSolver{<:Union{DMRG, DMRG2}}, state, direction, iter)
    fwd, bwd = _sweep_ranges(it.alg, state.mps)
    sites = direction === Val(:right) ? fwd : bwd
    ψ, ϵ = state.mps, state.ϵ
    for pos in sites
        ψ, state.local_errors[pos], state.truncation_errors[pos], state.decay_rates[pos] =
            local_update!(
            pos, direction, ψ, state.operator, it.alg, state.envs,
            ϵ, state.truncation_errors[pos], state.decay_rates[pos],
            iter, state.timeroutput, state.allocator,
        )
        # The next local solver uses the errors accumulated so far in this sweep.
        ϵ = maximum(state.local_errors)
    end
    return DMRGState(
        ψ, state.operator, state.envs, state.iter, ϵ, state.local_errors,
        state.truncation_errors, state.decay_rates, state.timeroutput, state.allocator,
    )
end

function Base.iterate(it::IterativeSolver{<:Union{DMRG, DMRG2}}, state = it.state)
    iter = state.iter + 1
    timeroutput = state.timeroutput
    state = @timeit timeroutput "sweep" begin
        state = sweep!(it, state, Val(:right), iter)
        sweep!(it, state, Val(:left), iter)
    end
    envs = unwrap_environments(state.envs)
    ψ, finalized_envs = @timeit timeroutput "finalize" it.finalize(
        iter, state.mps, state.operator, envs
    )::Tuple{typeof(state.mps), typeof(envs)}
    # The solve-owned cache assumes a read-only finalizer.
    @assert ψ === state.mps && finalized_envs === envs "DMRG sweep caches require a read-only finalizer"
    it.state = DMRGState(
        ψ, state.operator, state.envs, iter, state.ϵ, state.local_errors,
        state.truncation_errors, state.decay_rates, timeroutput, state.allocator,
    )
    return (ψ, envs, state.ϵ), it.state
end

# Truncation sets the attainable floor of the Galerkin error.
sweep_converged(alg, state::DMRGState) =
    state.ϵ <= max(alg.tol, maximum(state.truncation_errors))

function find_groundstate_sweep!(
        ψ::AbstractFiniteMPS, H, alg::Union{DMRG, DMRG2}, envs, allocator, timeroutput
    )
    log = IterLog(string(nameof(typeof(alg))))
    it = IterativeSolver(alg, DMRGState(ψ, H, alg, envs, allocator, timeroutput))

    with_verbosity(; alg.verbosity) do
        @log_initialization loginit!(log, it.ϵ, expectation_value(ψ, H, envs))
        for (ψ, envs, ϵ) in Iterators.take(it, alg.maxiter)
            if sweep_converged(alg, it.state)
                @info TimerReport(timeroutput) _group = :mpskit_timing
                @log_convergence logfinish!(log, it.iter, ϵ, expectation_value(ψ, H, envs))
                break
            elseif it.iter == alg.maxiter
                @info TimerReport(timeroutput) _group = :mpskit_timing
                @log_nonconvergence logcancel!(log, it.iter, ϵ, expectation_value(ψ, H, envs))
            else
                @log_iteration logiter!(log, it.iter, ϵ, expectation_value(ψ, H, envs))
            end
        end
    end

    state = it.state
    info = AlgorithmInfo(;
        converged = sweep_converged(alg, state), galerkin = state.ϵ,
        truncation_errors = _bond_truncation_errors(alg, state.truncation_errors),
        numiter = state.iter,
    )
    return state.mps, unwrap_environments(state.envs), info
end

function find_groundstate(ψ, H, alg::Union{DMRG, DMRG2}, envs...; kwargs...)
    return find_groundstate!(copy(ψ), H, alg, envs...; kwargs...)
end

# Explicit sweep cache lifecycle and center movement
# -------------------------------------------------
"""
    initialize_sweep_cache(ψ, O, envs; backend, allocator, one_site, two_site)

Build solve-owned snapshots for a fixed operator: prepare the left boundary and absorb
right-canonical tensors from the right boundary to initialize the first local window.
`one_site` and `two_site` explicitly select which local contributions to prepare.
Plain DMRG2 prepares pairs only; DMRG with expansion needs both. An explicit AC query
on a pair-only cache constructs its operator from the snapshots without publishing it.
Unsupported environment representations retain their existing update behavior.
"""
initialize_sweep_cache(ψ, O, envs; kwargs...) = envs
function initialize_sweep_cache(
        ψ::_HAM_MPS_TYPES, O::AbstractMPO, envs::FiniteEnvironments;
        backend, allocator, one_site::Bool = true, two_site::Bool = true,
    )
    @assert one_site || two_site "prepare at least one local Hamiltonian size"
    # Mixed bra/ket environments are outside the DMRG cache contract.
    isnothing(envs.above) || return envs
    data = cache_operator_data(O, envs.GLs[1], length(ψ))
    N = length(ψ)
    L = DMRGEnvironmentRecord(prepare_left_environment(envs.GLs[1], data, 1, backend, allocator; one_site, two_site)...)
    R = DMRGEnvironmentRecord(prepare_right_environment(envs.GRs[end], data, N, backend, allocator; one_site, two_site)...)
    cache = DMRGSweepCache(
        envs, data, fill!(Vector{Union{Nothing, typeof(L)}}(undef, N), nothing),
        fill!(Vector{Union{Nothing, typeof(R)}}(undef, N), nothing), backend, allocator, one_site, two_site,
    )
    cache.left[1] = L
    cache.right[N] = R
    for i in N:-1:2
        absorb_site!(cache, ψ, i, Val(:left))
    end
    return cache
end

function initialize_sweep_cache(ψ::WindowMPS{<:MPSTensor}, O::WindowMPOHamiltonian, envs::FiniteEnvironments; kwargs...)
    return initialize_sweep_cache(ψ, O.finite_ham, envs; kwargs...)
end
function initialize_sweep_cache(ψ, O::LazySum, envs::MultipleEnvironments; kwargs...)
    return MultipleEnvironments(
        map(O.ops, envs.envs) do op, env
            initialize_sweep_cache(ψ, op, env; kwargs...)
        end
    )
end
initialize_sweep_cache(ψ, O::MultipliedOperator, envs; kwargs...) =
    initialize_sweep_cache(ψ, O.op, envs; kwargs...)
function initialize_sweep_cache(ψ, O::LinearCombination, envs::LazyLincoCache; kwargs...)
    return LazyLincoCache(
        O, map(O.opps, envs.envs) do op, env
            initialize_sweep_cache(ψ, op, env; kwargs...)
        end
    )
end

unwrap_environments(envs) = envs
unwrap_environments(cache::DMRGSweepCache) = cache.environments
unwrap_environments(envs::MultipleEnvironments) = MultipleEnvironments(map(unwrap_environments, envs.envs))
unwrap_environments(envs::LazyLincoCache) = LazyLincoCache(envs.operator, map(unwrap_environments, envs.envs))

"""
    absorb_site!(envs, ψ, i, direction; local_only=false)

Contract the finalized tensor at `i` into an environment and publish its prepared record.
`Val(:right)` absorbs `AL[i]` into `GL`, producing the left record at `i + 1`;
`Val(:left)` absorbs `AR[i]` into `GR`, producing the right record at `i - 1`.
The underlying ordinary manager's tensor and dependency are updated at the same time.
`local_only=true` prepares only the one-site window needed after bond expansion.
"""
absorb_site!(envs, ψ, i, direction; kwargs...) = envs
function absorb_site!(cache::DMRGSweepCache{E, O, L}, ψ, i, ::Val{:right}; local_only::Bool = false) where {E, O, L}
    i < length(ψ) || return cache
    (; backend, allocator) = cache
    AL = ψ.AL[i]
    GL = cache.left[i].environment * TransferMatrix(AL, cache.environments.operator[i], AL; backend, allocator)
    cache.left[i + 1] = L(prepare_left_environment(GL, cache.operator_data, i + 1, backend, allocator; one_site = cache.one_site, two_site = cache.two_site, local_only)...)
    cache.environments.GLs[i + 1] = GL
    cache.environments.ldependencies[i] = AL
    return cache
end
function absorb_site!(cache::DMRGSweepCache{E, O, L, R}, ψ, i, ::Val{:left}; local_only::Bool = false) where {E, O, L, R}
    i > 1 || return cache
    (; backend, allocator) = cache
    AR = ψ.AR[i]
    GR = TransferMatrix(AR, cache.environments.operator[i], AR; backend, allocator) * cache.right[i].environment
    cache.right[i - 1] = R(prepare_right_environment(GR, cache.operator_data, i - 1, backend, allocator; one_site = cache.one_site, two_site = cache.two_site, local_only)...)
    cache.environments.GRs[i] = GR
    cache.environments.rdependencies[i] = AR
    return cache
end
function absorb_site!(envs::Union{MultipleEnvironments, LazyLincoCache}, ψ, i, direction; kwargs...)
    foreach(env -> absorb_site!(env, ψ, i, direction; kwargs...), envs.envs)
    return envs
end
