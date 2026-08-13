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
    so the state norm keeps useful information. In real time this is exact, namely the squared norm
    drops by precisely the truncated ("discarded") weight,
    ``\\lVert \\psi \\rVert^2 = \\lVert \\psi_0 \\rVert^2 - \\epsilon^2``,
    with `ϵ` the truncation error returned by [`timestep`](@ref). In imaginary time the norm also carries
    the physical decay of the weight, so it no longer isolates the truncation. Without `trunc` nothing is
    discarded at all and the norm is conserved exactly in real time.

    Pass `normalize = true` to `timestep`/`time_evolve` to renormalize at every step instead,
    like a ground-state search. This is independent of `imaginary_evolution`. CBE is only available for finite MPS.

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
    # infinite one-site TDVP has a fixed bond dimension and never truncates so nothing is discarded
    # the gauge-fixing residual is controlled by `tolgauge`, not reported here
    return ψ′, envs, zero(real(scalartype(ψ′)))
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

function _timestep_finite!(
        ψ::AbstractFiniteMPS, H, t::Number, dt::Number, alg::TDVP, envs, allocator;
        imaginary_evolution::Bool, normalize::Bool
    )
    # discarded weight of the step, accumulated in squares over the local gauges
    # stays exactly zero for QR gauging
    ϵ² = zero(real(scalartype(ψ)))

    # sweep left to right
    for i in 1:(length(ψ) - 1)
        # 1. optionally expand the bond ahead of the local update (CBE)
        isnothing(alg.alg_expand) ||
            changebond!(i, Val(:right), ψ, H, alg.alg_expand, envs; normalize, allocator)

        # 2. evolve the (possibly expanded) center tensor forward
        Hac = AC_hamiltonian(i, ψ, H, ψ, envs; alg.backend, allocator)
        AC = integrate(Hac, ψ.AC[i], t, dt / 2, alg.integrator; imaginary_evolution)

        # 3. gauge: split AC -> AL[i], C[i] (QR center-move, or truncated SVD cutting the
        #    enlarged bond back down) and move the center to i+1. By default the norm is
        #    preserved; `normalize` renormalizes.
        _, ϵ = left_gauge!(ψ, i, AC, alg.alg_gauge; normalize)
        ϵ² += ϵ^2

        # 4. evolve the bond tensor backward
        Hc = C_hamiltonian(i, ψ, H, ψ, envs; alg.backend, allocator)
        ψ.C[i] = integrate(
            Hc, ψ.C[i], t + dt / 2, -dt / 2, alg.integrator;
            imaginary_evolution
        )
    end

    # edge case
    Hac = AC_hamiltonian(length(ψ), ψ, H, ψ, envs; alg.backend, allocator)
    ψ.AC[end] = integrate(Hac, ψ.AC[end], t, dt / 2, alg.integrator; imaginary_evolution)

    # sweep right to left
    for i in length(ψ):-1:2
        # 1. optionally expand the bond ahead of the local update (CBE)
        isnothing(alg.alg_expand) ||
            changebond!(i, Val(:left), ψ, H, alg.alg_expand, envs; normalize, allocator)

        # 2. evolve the (possibly expanded) center tensor forward
        Hac = AC_hamiltonian(i, ψ, H, ψ, envs; alg.backend, allocator)
        AC = integrate(
            Hac, ψ.AC[i], t + dt / 2, dt / 2, alg.integrator;
            imaginary_evolution
        )

        # 3. gauge: split AC -> C[i-1], AR[i] and move the center to i-1 (norm preserved by
        #    default; `normalize` renormalizes)
        _, ϵ = right_gauge!(ψ, i, AC, alg.alg_gauge; normalize)
        ϵ² += ϵ^2

        # 4. evolve the bond tensor backward
        Hc = C_hamiltonian(i - 1, ψ, H, ψ, envs; alg.backend, allocator)
        ψ.C[i - 1] = integrate(
            Hc, ψ.C[i - 1], t + dt, -dt / 2, alg.integrator;
            imaginary_evolution
        )
    end

    # edge case
    Hac = AC_hamiltonian(1, ψ, H, ψ, envs; alg.backend, allocator)
    ψ.AC[1] = integrate(
        Hac, ψ.AC[1], t + dt / 2, dt / 2, alg.integrator;
        imaginary_evolution
    )

    return ψ, envs, sqrt(ϵ²)
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

function _timestep2_finite!(
        ψ::AbstractFiniteMPS, H, t::Number, dt::Number, alg::TDVP2, envs, allocator;
        imaginary_evolution::Bool, normalize::Bool
    )
    # the two-site center always has to be split back up, so the gauge is always a truncated SVD
    alg_gauge = MatrixAlgebraKit.TruncatedAlgorithm(alg.alg_svd, alg.trunc)

    # discarded weight of the step, accumulated in squares over the local gauges
    ϵ² = zero(real(scalartype(ψ)))

    # sweep left to right
    for i in 1:(length(ψ) - 1)
        ac2 = _transpose_front(ψ.AC[i]) * _transpose_tail(ψ.AR[i + 1])
        Hac2 = AC2_hamiltonian(i, ψ, H, ψ, envs; alg.backend, allocator)
        ac2′ = integrate(Hac2, ac2, t, dt / 2, alg.integrator; imaginary_evolution)

        # the norm of the discarded singular values is the truncation error
        _, ϵ = gauge2!(ψ, i, Val(:right), ac2′, alg_gauge; normalize)
        ϵ² += ϵ^2

        if i != (length(ψ) - 1)
            Hac = AC_hamiltonian(i + 1, ψ, H, ψ, envs; alg.backend, allocator)
            ψ.AC[i + 1] = integrate(
                Hac, ψ.AC[i + 1], t + dt / 2, -dt / 2, alg.integrator;
                imaginary_evolution
            )
        end
    end

    # sweep right to left
    for i in length(ψ):-1:2
        ac2 = _transpose_front(ψ.AL[i - 1]) * _transpose_tail(ψ.AC[i])
        Hac2 = AC2_hamiltonian(i - 1, ψ, H, ψ, envs; alg.backend, allocator)
        ac2′ = integrate(Hac2, ac2, t + dt / 2, dt / 2, alg.integrator; imaginary_evolution)

        _, ϵ = gauge2!(ψ, i - 1, Val(:left), ac2′, alg_gauge; normalize)
        ϵ² += ϵ^2

        if i != 2
            Hac = AC_hamiltonian(i - 1, ψ, H, ψ, envs; alg.backend, allocator)
            ψ.AC[i - 1] = integrate(
                Hac, ψ.AC[i - 1], t + dt, -dt / 2, alg.integrator;
                imaginary_evolution
            )
        end
    end

    return ψ, envs, sqrt(ϵ²)
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
