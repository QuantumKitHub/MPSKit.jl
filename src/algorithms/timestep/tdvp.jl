"""
$(TYPEDEF)

Single site MPS time-evolution algorithm based on the Time-Dependent Variational Principle.

For finite MPS, setting `alg_expand` to a bond-expansion algorithm (e.g. [`OptimalExpand`](@ref),
[`SketchedExpand`](@ref)) expands the bond with directions orthogonal to the current state
ahead of each local integration, recovering Controlled Bond Expansion (CBE) TDVP and lifting the
fixed-bond limitation of plain single-site TDVP. A truncating `trunc` is then required to cut
the enlarged bond back down (selecting the truncated-SVD gauge). The expansion is
state-preserving, as required for a consistent time evolution.

!!! note
    By default the norm is not preserved: neither the bond expansion nor the truncation renormalizes,
    so the state norm keeps useful information. In real time the squared norm drops by precisely the
    weight discarded by all cuts of the step, whereas in imaginary time the norm also carries the
    physical decay of the weight. Without `trunc` nothing is discarded at all and the norm is
    conserved exactly in real time.

    Pass `normalize = true` to `timestep`/`time_evolve` to renormalize at every step instead,
    like a ground state search. This is independent of `imaginary_evolution`. CBE is only available for finite MPS.

# Fields

$(TYPEDFIELDS)

# See also

Used as the `algorithm` argument of [`timestep`](@ref), [`timestep!`](@ref) and [`time_evolve`](@ref).

# References

* [Haegeman et al. Phys. Rev. Lett. 107 (2011)](@cite haegeman2011)
"""
struct TDVP{A, E, G, F, B} <: Algorithm
    "algorithm used in the exponential solvers"
    integrator::A

    "tolerance for gauging algorithm"
    tolgauge::Float64

    "maximal amount of iterations for gauging algorithm"
    gaugemaxiter::Int

    "algorithm used to expand the bond ahead of each local update, or `nothing` for none (finite CBE-TDVP)"
    alg_expand::E

    "factorization used for the post-update gauge: a QR algorithm (no truncation) or a truncated SVD"
    alg_gauge::G

    "callback function applied after each iteration, of signature `finalize(t, ψ, H, envs) -> ψ, envs`"
    finalize::F

    "backend for tensor contractions and index manipulations"
    backend::B
end
function TDVP(;
        integrator = Defaults.alg_expsolve(), tolgauge = Defaults.tolgauge,
        gaugemaxiter = Defaults.maxiter, finalize = Defaults._finalize,
        alg_expand = nothing, trunc = notrunc(),
        alg_svd = Defaults.alg_svd(), alg_orth = Defaults.alg_orth(),
        backend = Defaults.backend()
    )
    # a no-truncation `trunc` selects a (bond-preserving) QR gauge, anything else a truncated SVD
    alg_gauge = trunc isa MatrixAlgebraKit.NoTruncation ? alg_orth :
        MatrixAlgebraKit.TruncatedAlgorithm(alg_svd, trunc)
    if !isnothing(alg_expand) && !_truncates(alg_gauge)
        @warn "TDVP with `alg_expand` but no truncation (`trunc = notrunc()`): the bond dimension will grow unboundedly each sweep."
    end
    return TDVP(
        integrator, tolgauge, gaugemaxiter, alg_expand, alg_gauge, finalize, backend
    )
end

# Internal state of a single time step `t → t + dt`
struct TimeStepState{S, O, E, T, D, V, SC, A, TO}
    mps::S
    operator::O
    envs::E
    t::T
    dt::D
    truncation_errors::V
    scheduler::SC
    allocator::A
    timeroutput::TO
    imaginary_evolution::Bool
    normalize::Bool
end

# a single allocator serves all local updates of a step, and it can be carried over to later
# steps together with the scheduler that selected it
function TimeStepState(
        ψ, H, envs, t, dt; imaginary_evolution, normalize, timeroutput = NoTimerOutput(),
        scheduler = _step_scheduler(ψ), allocator = default_allocator(ψ, scheduler)
    )
    truncation_errors = _truncation_errors(ψ)
    return TimeStepState(
        ψ, H, envs, t, dt, truncation_errors, scheduler, allocator, timeroutput,
        imaginary_evolution, normalize,
    )
end

# a finite sweep is serial, whereas an infinite step updates all sites concurrently
_step_scheduler(::AbstractFiniteMPS) = SerialScheduler()
_step_scheduler(::InfiniteMPS) = Defaults.scheduler[]

# per-bond truncation errors of a finite sweep; an infinite step never truncates
_truncation_errors(ψ::AbstractFiniteMPS) = zeros(real(scalartype(ψ)), length(ψ) - 1)
_truncation_errors(ψ::InfiniteMPS) = nothing

# infinite one-site TDVP runs at fixed bond dimension and never truncates, so it doesn't report
# `truncation_errors` (rather than report zeros that would look like a measurement)
_step_info(state::TimeStepState) = isnothing(state.truncation_errors) ? AlgorithmInfo() :
    AlgorithmInfo(; truncation_errors = state.truncation_errors)

function _with_mps(state::TimeStepState, ψ; t = state.t)
    return TimeStepState(
        ψ, state.operator, state.envs, t, state.dt, state.truncation_errors,
        state.scheduler, state.allocator, state.timeroutput, state.imaginary_evolution,
        state.normalize,
    )
end

function timestep!(
        ψ::InfiniteMPS, H, t::Number, dt::Number, alg::TDVP,
        envs::AbstractMPSEnvironments = environments(ψ, H, ψ);
        leftorthflag = true, imaginary_evolution::Bool = false, normalize::Bool = false
    )
    # `normalize` is accepted for signature uniformity with the finite integrators, but an
    # `InfiniteMPS` is always normalized to norm-1-per-site by the gauge/reconstruction below
    # (a structural gauge requirement, not information erasure), so the flag has no effect here.
    # the scheduler is read here rather than below, so that the allocator it selects is inferable
    return _timestep_infinite!(
        ψ, H, t, dt, alg, envs, Defaults.scheduler[]; leftorthflag, imaginary_evolution
    )
end

function _timestep_infinite!(
        ψ::InfiniteMPS, H, t::Number, dt::Number, alg::TDVP, envs, scheduler::Scheduler;
        leftorthflag, imaginary_evolution
    )
    state = TimeStepState(ψ, H, envs, t, dt; imaginary_evolution, normalize = false, scheduler)
    state = _timestep!(alg, state; leftorthflag)
    # the gauge-fixing residual is controlled by `tolgauge`, not reported here
    return state.mps, state.envs, _step_info(state)
end

# an infinite step evolves every center tensor at once, then restores the uniform gauge and
# recomputes the environments
function _timestep!(alg::TDVP, state::TimeStepState{<:InfiniteMPS}; leftorthflag = true)
    ACs, Cs = localupdate_step!(alg, state)
    ψ = gauge_step!(alg, state, ACs, Cs; leftorthflag)
    recalculate!(state.envs, ψ, state.operator)
    return _with_mps(state, ψ; t = state.t + state.dt)
end

function localupdate_step!(alg::TDVP, state::TimeStepState)
    (; mps, operator, envs, t, dt, scheduler, allocator, imaginary_evolution) = state
    ψ, H = mps, operator
    temp_ACs = similar(ψ.AC)
    temp_Cs = similar(ψ.C)

    # both sweeps together are a single unit of concurrent work, and share one allocator
    ac_sweep!() = tforeach(1:length(ψ); scheduler) do loc
        Hac = AC_hamiltonian(loc, ψ, H, ψ, envs; alg.backend, allocator)
        temp_ACs[loc] = integrate(Hac, ψ.AC[loc], t, dt, alg.integrator; imaginary_evolution)
        return nothing
    end
    c_sweep!() = tforeach(1:length(ψ); scheduler) do loc
        Hc = C_hamiltonian(loc, ψ, H, ψ, envs; alg.backend, allocator)
        temp_Cs[loc] = integrate(Hc, ψ.C[loc], t, dt, alg.integrator; imaginary_evolution)
        return nothing
    end

    if scheduler isa SerialScheduler
        ac_sweep!()
        c_sweep!()
    else
        # the AC and C sweeps are independent, so run them concurrently with each other too
        @sync begin
            Threads.@spawn ac_sweep!()
            Threads.@spawn c_sweep!()
        end
    end
    return temp_ACs, temp_Cs
end

# the evolved center tensors are split into the new left (or right) isometries, from which the
# uniform gauge is restored in place
function gauge_step!(alg::TDVP, state::TimeStepState, ACs, Cs; leftorthflag)
    ψ = state.mps
    if leftorthflag
        regauge!.(ACs, Cs)
        copyto!(ψ.AL, ACs)
        gaugefix!(ψ, ψ.AL, ψ.C[end]; order = :R, tol = alg.tolgauge, maxiter = alg.gaugemaxiter)
    else
        circshift!(Cs, 1)
        regauge!.(Cs, ACs)
        copyto!(ψ.AR, ACs)
        gaugefix!(ψ, ψ.AR, ψ.C[0]; order = :L, tol = alg.tolgauge, maxiter = alg.gaugemaxiter)
    end
    mul!.(ψ.AC, ψ.AL, ψ.C)
    return ψ
end

# Single-site integrators visit every site, finishing a half-sweep by evolving only the center
# tensor at its far end.
_sweep_ranges(::TDVP, ψ) = (1:length(ψ), length(ψ):-1:1)

# bond cut by the update at `site`, or `nothing` at the far end of a single-site half-sweep
_swept_bond(alg, site, ψ, ::Val{:right}) = site < length(ψ) ? site : nothing
_swept_bond(alg, site, ψ, ::Val{:left}) = site > 1 ? site - 1 : nothing

function sweep!(alg, state::TimeStepState, direction)
    fwd, bwd = _sweep_ranges(alg, state.mps)
    sites = direction === Val(:right) ? fwd : bwd
    ψ = state.mps
    @timeit state.timeroutput "half-sweep" for site in sites
        ψ, ϵ = local_update!(site, direction, ψ, alg, state)
        bond = _swept_bond(alg, site, ψ, direction)
        isnothing(bond) || (state.truncation_errors[bond] = ϵ)
    end
    # each half-sweep evolves the state over `dt / 2`
    return _with_mps(state, ψ; t = state.t + state.dt / 2)
end

# a step is a left→right half-sweep `t → t + dt / 2` followed by a right→left half-sweep
# `t + dt / 2 → t + dt`
function _timestep!(alg, state::TimeStepState{<:AbstractFiniteMPS})
    state = sweep!(alg, state, Val(:right))
    return sweep!(alg, state, Val(:left))
end

function local_update!(site, direction::Val, ψ, alg::TDVP, state::TimeStepState)
    (; operator, envs, t, dt, allocator, imaginary_evolution, normalize) = state
    H = operator
    t_AC, t_C = t, t + dt / 2

    # at the far end of the sweep there is no bond ahead: only evolve the center tensor
    if site == _sweep_end(ψ, direction)
        Hac = AC_hamiltonian(site, ψ, H, ψ, envs; alg.backend, allocator)
        ψ.AC[site] = integrate(Hac, ψ.AC[site], t_AC, dt / 2, alg.integrator; imaginary_evolution)
        return ψ, zero(real(scalartype(ψ)))
    end

    # 1. optionally expand the bond ahead of the local update (CBE)
    isnothing(alg.alg_expand) ||
        changebond!(site, direction, ψ, H, alg.alg_expand, envs; normalize, allocator)

    # 2. evolve the (possibly expanded) center tensor forward
    Hac = AC_hamiltonian(site, ψ, H, ψ, envs; alg.backend, allocator)
    AC = integrate(Hac, ψ.AC[site], t_AC, dt / 2, alg.integrator; imaginary_evolution)

    # 3. gauge: split AC onto the bond ahead and move the center.
    #    By default the norm is preserved; `normalize` renormalizes.
    if direction === Val(:right)
        _, ϵ = left_gauge!(ψ, site, AC, alg.alg_gauge; normalize)
        bond = site
    else
        _, ϵ = right_gauge!(ψ, site, AC, alg.alg_gauge; normalize)
        bond = site - 1
    end

    # 4. evolve the bond tensor backward
    Hc = C_hamiltonian(bond, ψ, H, ψ, envs; alg.backend, allocator)
    ψ.C[bond] = integrate(Hc, ψ.C[bond], t_C, -dt / 2, alg.integrator; imaginary_evolution)

    return ψ, ϵ
end

"""
$(TYPEDEF)

Two-site MPS time-evolution algorithm based on the Time-Dependent Variational Principle.
See [`TDVP`](@ref) for more information.

# Fields

$(TYPEDFIELDS)

# See also

Used as the `algorithm` argument of [`timestep`](@ref), [`timestep!`](@ref) and [`time_evolve`](@ref).

# References

* [Haegeman et al. Phys. Rev. Lett. 107 (2011)](@cite haegeman2011)
"""
@kwdef struct TDVP2{A, S, F, B} <: Algorithm
    "algorithm used in the exponential solvers"
    integrator::A = Defaults.alg_expsolve()

    "tolerance for gauging algorithm"
    tolgauge::Float64 = Defaults.tolgauge

    "maximal amount of iterations for gauging algorithm"
    gaugemaxiter::Int = Defaults.maxiter

    "algorithm used for the singular value decomposition"
    alg_svd::S = Defaults.alg_svd()

    "algorithm used for truncation of the two-site update"
    trunc::TruncationStrategy

    "callback function applied after each iteration, of signature `finalize(t, ψ, H, envs) -> ψ, envs`"
    finalize::F = Defaults._finalize

    "backend for tensor contractions and index manipulations"
    backend::B = Defaults.backend()
end

# two-site TDVP visits every bond, and each update cuts the bond it acts on
_sweep_ranges(::TDVP2, ψ) = (1:(length(ψ) - 1), (length(ψ) - 1):-1:1)
_swept_bond(::TDVP2, pos, ψ, ::Val{:right}) = pos
_swept_bond(::TDVP2, pos, ψ, ::Val{:left}) = pos

function local_update!(pos, direction::Val, ψ, alg::TDVP2, state::TimeStepState)
    (; operator, envs, t, dt, allocator, imaginary_evolution, normalize) = state
    H = operator
    t_AC2, t_AC = t, t + dt / 2

    # 1. evolve the two-site center tensor at `(pos, pos + 1)` forward
    ac2 = if direction === Val(:right)
        _transpose_front(ψ.AC[pos]) * _transpose_tail(ψ.AR[pos + 1])
    else
        _transpose_front(ψ.AL[pos]) * _transpose_tail(ψ.AC[pos + 1])
    end
    Hac2 = AC2_hamiltonian(pos, ψ, H, ψ, envs; alg.backend, allocator)
    ac2′ = integrate(Hac2, ac2, t_AC2, dt / 2, alg.integrator; imaginary_evolution)

    # 2. gauge: the two-site center always has to be split back up, so this is always a
    #    truncated SVD, and the norm of the discarded singular values is the truncation error
    alg_gauge = MatrixAlgebraKit.TruncatedAlgorithm(alg.alg_svd, alg.trunc)
    _, ϵ = gauge2!(ψ, pos, direction, ac2′, alg_gauge; normalize)

    # 3. evolve the new single-site center backward, except at the far end of the sweep
    if direction === Val(:right) ? pos != length(ψ) - 1 : pos != 1
        site = direction === Val(:right) ? pos + 1 : pos
        Hac = AC_hamiltonian(site, ψ, H, ψ, envs; alg.backend, allocator)
        ψ.AC[site] = integrate(Hac, ψ.AC[site], t_AC, -dt / 2, alg.integrator; imaginary_evolution)
    end

    return ψ, ϵ
end

_promotes_to_complex(ψ, H, dt, ::Union{TDVP, TDVP2}; imaginary_evolution) =
    scalartype(ψ) <: Real && !(imaginary_evolution && isreal(dt))
