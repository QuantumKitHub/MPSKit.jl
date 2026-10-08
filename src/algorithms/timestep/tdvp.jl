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

function timestep(
        ψ::InfiniteMPS, H, t::Number, dt::Number, alg::TDVP,
        envs::AbstractMPSEnvironments = environments(ψ, H, ψ);
        leftorthflag = true, imaginary_evolution::Bool = false, normalize::Bool = false
    )
    # `normalize` is accepted for signature uniformity with the finite integrators, but an
    # `InfiniteMPS` is always normalized to norm-1-per-site by the gauge/reconstruction below
    # (a structural gauge requirement, not information erasure), so the flag has no effect here.
    # convert state to complex if necessary
    if scalartype(ψ) <: Real && (!imaginary_evolution || !isreal(dt))
        return timestep(complex(ψ), H, t, dt, alg, envs; leftorthflag, imaginary_evolution, normalize)
    end

    # the scheduler is read here rather than below, so that the allocator it selects is inferable
    return _timestep_infinite(
        ψ, H, t, dt, alg, envs, Defaults.scheduler[]; leftorthflag, imaginary_evolution
    )
end

function _timestep_infinite(
        ψ::InfiniteMPS, H, t::Number, dt::Number, alg::TDVP, envs, scheduler::Scheduler;
        leftorthflag, imaginary_evolution
    )
    temp_ACs = similar(ψ.AC)
    temp_Cs = similar(ψ.C)

    # both sweeps together are a single unit of concurrent work, and share one allocator
    allocator = default_allocator(ψ, scheduler)
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

    if leftorthflag
        regauge!.(temp_ACs, temp_Cs)
        ψ′ = InfiniteMPS(temp_ACs, ψ.C[end]; tol = alg.tolgauge, maxiter = alg.gaugemaxiter)
    else
        circshift!(temp_Cs, 1)
        regauge!.(temp_Cs, temp_ACs)
        ψ′ = InfiniteMPS(ψ.C[0], temp_ACs; tol = alg.tolgauge, maxiter = alg.gaugemaxiter)
    end

    recalculate!(envs, ψ′, H)
    # infinite one-site TDVP runs at fixed bond dimension and never truncates, so it doesn't
    # report `truncation_errors` (rather than report zeros that would look like a measurement)
    # the gauge-fixing residual is controlled by `tolgauge`, not reported here
    return ψ′, envs, AlgorithmInfo()
end

function timestep!(
        ψ::AbstractFiniteMPS, H, t::Number, dt::Number, alg::TDVP,
        envs::AbstractMPSEnvironments = environments(ψ, H, ψ);
        imaginary_evolution::Bool = false, normalize::Bool = false
    )
    # the sweep is serial, so a single allocator serves all local updates
    allocator = default_allocator(ψ, SerialScheduler())
    return _timestep_finite!(
        ψ, H, t, dt, alg, envs, allocator; imaginary_evolution, normalize
    )
end

# Start times of the forward center update and of the backward update that follows it, for the
# half-sweep in `direction` of a step `t → t + dt`
_half_sweep_times(::Val{:right}, t, dt) = (t, t + dt / 2)
_half_sweep_times(::Val{:left}, t, dt) = (t + dt / 2, t + dt)

function local_update!(
        site, direction::Val, ψ, H, alg::TDVP, envs, t, dt, allocator;
        imaginary_evolution, normalize
    )
    t_AC, t_C = _half_sweep_times(direction, t, dt)

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

    # 3. gauge: split AC onto the bond ahead (QR center-move, or truncated SVD cutting the
    #    enlarged bond back down) and move the center across it. By default the norm is
    #    preserved; `normalize` renormalizes.
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

function _timestep_finite!(
        ψ::AbstractFiniteMPS, H, t::Number, dt::Number, alg::TDVP, envs, allocator;
        imaginary_evolution::Bool, normalize::Bool
    )
    L = length(ψ)
    ϵ_truncs = zeros(real(scalartype(ψ)), L - 1)

    # left→right half-sweep: `t → t + dt / 2`
    for site in 1:L
        ψ, ϵ = local_update!(
            site, Val(:right), ψ, H, alg, envs, t, dt, allocator;
            imaginary_evolution, normalize
        )
        site < L && (ϵ_truncs[site] = ϵ)
    end

    # right→left half-sweep: `t + dt / 2 → t + dt`
    for site in L:-1:1
        ψ, ϵ = local_update!(
            site, Val(:left), ψ, H, alg, envs, t, dt, allocator;
            imaginary_evolution, normalize
        )
        site > 1 && (ϵ_truncs[site - 1] = ϵ)
    end

    return ψ, envs, AlgorithmInfo(; truncation_errors = ϵ_truncs)
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

function timestep!(
        ψ::AbstractFiniteMPS, H, t::Number, dt::Number, alg::TDVP2,
        envs::AbstractMPSEnvironments = environments(ψ, H, ψ);
        imaginary_evolution::Bool = false, normalize::Bool = false
    )
    # the sweep is serial, so a single allocator serves all local updates
    allocator = default_allocator(ψ, SerialScheduler())
    return _timestep2_finite!(
        ψ, H, t, dt, alg, envs, allocator; imaginary_evolution, normalize
    )
end

function local_update!(
        pos, direction::Val, ψ, H, alg::TDVP2, envs, t, dt, allocator;
        imaginary_evolution, normalize
    )
    t_AC2, t_AC = _half_sweep_times(direction, t, dt)

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

function _timestep2_finite!(
        ψ::AbstractFiniteMPS, H, t::Number, dt::Number, alg::TDVP2, envs, allocator;
        imaginary_evolution::Bool, normalize::Bool
    )
    ϵ_truncs = zeros(real(scalartype(ψ)), length(ψ) - 1)

    # left→right half-sweep: `t → t + dt / 2`
    for pos in 1:(length(ψ) - 1)
        ψ, ϵ_truncs[pos] = local_update!(
            pos, Val(:right), ψ, H, alg, envs, t, dt, allocator;
            imaginary_evolution, normalize
        )
    end

    # right→left half-sweep: `t + dt / 2 → t + dt`
    for pos in (length(ψ) - 1):-1:1
        ψ, ϵ_truncs[pos] = local_update!(
            pos, Val(:left), ψ, H, alg, envs, t, dt, allocator;
            imaginary_evolution, normalize
        )
    end

    return ψ, envs, AlgorithmInfo(; truncation_errors = ϵ_truncs)
end

# copying version
function timestep(
        ψ::AbstractFiniteMPS, H, time::Number, timestep::Number,
        alg::Union{TDVP, TDVP2}, envs::AbstractMPSEnvironments...;
        imaginary_evolution::Bool = false, normalize::Bool = false, kwargs...
    )
    isreal = (scalartype(ψ) <: Real && !imaginary_evolution)
    ψ′ = isreal ? complex(ψ) : copy(ψ)
    if length(envs) != 0 && isreal
        @warn "Currently cannot reuse real environments for complex evolution"
        envs′ = environments(ψ′, H, ψ′)
    elseif length(envs) == 1
        envs′ = only(envs)
    else
        @assert length(envs) == 0 "Invalid signature"
        envs′ = environments(ψ′, H, ψ′)
    end
    return timestep!(ψ′, H, time, timestep, alg, envs′; imaginary_evolution, normalize, kwargs...)
end
