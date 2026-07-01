"""
$(TYPEDEF)

Single site MPS time-evolution algorithm based on the Time-Dependent Variational Principle.

For finite MPS, setting `alg_expand` to a bond-expansion algorithm (e.g. [`OptimalExpand`](@ref),
[`SketchedExpand`](@ref)) enriches the bond with directions orthogonal to the current state
ahead of each local integration, recovering Controlled Bond Expansion (CBE) TDVP and lifting the
fixed-bond limitation of plain single-site TDVP. A truncating `trscheme` is then required to cut
the enlarged bond back down (selecting the truncated-SVD gauge). The expansion is
state-preserving, as required for a consistent time evolution.

!!! note
    Real-time evolution preserves the norm: neither the bond expansion nor the truncation
    renormalizes, so the state norm reflects the accumulated truncation error. Imaginary-time
    evolution instead renormalizes at every step, like a ground-state search. CBE is only
    available for finite MPS.

## Fields

$(TYPEDFIELDS)

## References

* [Haegeman et al. Phys. Rev. Lett. 107 (2011)](@cite haegeman2011)
"""
struct TDVP{A, E, G, F} <: Algorithm
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

    "callback function applied after each iteration, of signature `finalize(iter, ψ, H, envs) -> ψ, envs`"
    finalize::F
end
function TDVP(;
        integrator = Defaults.alg_expsolve(), tolgauge = Defaults.tolgauge,
        gaugemaxiter = Defaults.maxiter, finalize = Defaults._finalize,
        alg_expand = nothing, trscheme = notrunc(),
        alg_svd = Defaults.alg_svd(), alg_orth = Defaults.alg_orth()
    )
    # a no-truncation `trscheme` selects a (bond-preserving) QR gauge, anything else a truncated SVD
    alg_gauge = trscheme isa MatrixAlgebraKit.NoTruncation ? alg_orth :
        MatrixAlgebraKit.TruncatedAlgorithm(alg_svd, trscheme)
    if !isnothing(alg_expand) && !_truncates(alg_gauge)
        @warn "TDVP with `alg_expand` but no truncation (`trscheme = notrunc()`): the bond dimension will grow unboundedly each sweep."
    end
    return TDVP(integrator, tolgauge, gaugemaxiter, alg_expand, alg_gauge, finalize)
end

function timestep(
        ψ::InfiniteMPS, H, t::Number, dt::Number, alg::TDVP,
        envs::AbstractMPSEnvironments = environments(ψ, H, ψ);
        leftorthflag = true, imaginary_evolution::Bool = false
    )
    # convert state to complex if necessary
    if scalartype(ψ) <: Real && (!imaginary_evolution || !isreal(dt))
        return timestep(complex(ψ), H, t, dt, alg, envs; leftorthflag, imaginary_evolution)
    end

    temp_ACs = similar(ψ.AC)
    temp_Cs = similar(ψ.C)

    scheduler = Defaults.scheduler[]
    if scheduler isa SerialScheduler
        temp_ACs = tmap!(temp_ACs, 1:length(ψ); scheduler) do loc
            Hac = AC_hamiltonian(loc, ψ, H, ψ, envs)
            return integrate(Hac, ψ.AC[loc], t, dt, alg.integrator; imaginary_evolution)
        end
        temp_Cs = tmap!(temp_Cs, 1:length(ψ); scheduler) do loc
            Hc = C_hamiltonian(loc, ψ, H, ψ, envs)
            return integrate(Hc, ψ.C[loc], t, dt, alg.integrator; imaginary_evolution)
        end
    else
        @sync begin
            Threads.@spawn begin
                temp_ACs = tmap!(temp_ACs, 1:length(ψ); scheduler) do loc
                    Hac = AC_hamiltonian(loc, ψ, H, ψ, envs)
                    return integrate(
                        Hac, ψ.AC[loc], t, dt, alg.integrator;
                        imaginary_evolution
                    )
                end
            end
            Threads.@spawn begin
                temp_Cs = tmap!(temp_Cs, 1:length(ψ); scheduler) do loc
                    Hc = C_hamiltonian(loc, ψ, H, ψ, envs)
                    return integrate(
                        Hc, ψ.C[loc], t, dt, alg.integrator;
                        imaginary_evolution
                    )
                end
            end
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
    return ψ′, envs
end

function timestep!(
        ψ::AbstractFiniteMPS, H, t::Number, dt::Number, alg::TDVP,
        envs::AbstractMPSEnvironments = environments(ψ, H, ψ);
        imaginary_evolution::Bool = false
    )

    # sweep left to right
    for i in 1:(length(ψ) - 1)
        # 1. optionally expand the bond ahead of the local update (CBE)
        isnothing(alg.alg_expand) ||
            changebond!(i, Val(:right), ψ, H, alg.alg_expand, envs; normalize = imaginary_evolution)

        # 2. evolve the (possibly expanded) center tensor forward
        Hac = AC_hamiltonian(i, ψ, H, ψ, envs)
        AC = integrate(Hac, ψ.AC[i], t, dt / 2, alg.integrator; imaginary_evolution)

        # 3. gauge: split AC -> AL[i], C[i] (QR center-move, or truncated SVD cutting the
        #    enlarged bond back down) and move the center to i+1. Real-time evolution preserves
        #    the norm; imaginary-time evolution renormalizes.
        left_gauge!(ψ, i, AC, alg.alg_gauge; normalize = imaginary_evolution)

        # 4. evolve the bond tensor backward
        Hc = C_hamiltonian(i, ψ, H, ψ, envs)
        ψ.C[i] = integrate(
            Hc, ψ.C[i], t + dt / 2, -dt / 2, alg.integrator;
            imaginary_evolution
        )
    end

    # edge case
    Hac = AC_hamiltonian(length(ψ), ψ, H, ψ, envs)
    ψ.AC[end] = integrate(Hac, ψ.AC[end], t, dt / 2, alg.integrator; imaginary_evolution)

    # sweep right to left
    for i in length(ψ):-1:2
        # 1. optionally expand the bond ahead of the local update (CBE)
        isnothing(alg.alg_expand) ||
            changebond!(i, Val(:left), ψ, H, alg.alg_expand, envs; normalize = imaginary_evolution)

        # 2. evolve the (possibly expanded) center tensor forward
        Hac = AC_hamiltonian(i, ψ, H, ψ, envs)
        AC = integrate(
            Hac, ψ.AC[i], t + dt / 2, dt / 2, alg.integrator;
            imaginary_evolution
        )

        # 3. gauge: split AC -> C[i-1], AR[i] and move the center to i-1 (real-time preserves the
        #    norm; imaginary-time renormalizes)
        right_gauge!(ψ, i, AC, alg.alg_gauge; normalize = imaginary_evolution)

        # 4. evolve the bond tensor backward
        Hc = C_hamiltonian(i - 1, ψ, H, ψ, envs)
        ψ.C[i - 1] = integrate(
            Hc, ψ.C[i - 1], t + dt, -dt / 2, alg.integrator;
            imaginary_evolution
        )
    end

    # edge case
    Hac = AC_hamiltonian(1, ψ, H, ψ, envs)
    ψ.AC[1] = integrate(
        Hac, ψ.AC[1], t + dt / 2, dt / 2, alg.integrator;
        imaginary_evolution
    )

    return ψ, envs
end

"""
$(TYPEDEF)

Two-site MPS time-evolution algorithm based on the Time-Dependent Variational Principle.

## Fields

$(TYPEDFIELDS)

## References

* [Haegeman et al. Phys. Rev. Lett. 107 (2011)](@cite haegeman2011)
"""
@kwdef struct TDVP2{A, S, F} <: Algorithm
    "algorithm used in the exponential solvers"
    integrator::A = Defaults.alg_expsolve()

    "tolerance for gauging algorithm"
    tolgauge::Float64 = Defaults.tolgauge

    "maximal amount of iterations for gauging algorithm"
    gaugemaxiter::Int = Defaults.maxiter

    "algorithm used for the singular value decomposition"
    alg_svd::S = Defaults.alg_svd()

    "algorithm used for truncation of the two-site update"
    trscheme::TruncationStrategy

    "callback function applied after each iteration, of signature `finalize(iter, ψ, H, envs) -> ψ, envs`"
    finalize::F = Defaults._finalize
end

function timestep!(
        ψ::AbstractFiniteMPS, H, t::Number, dt::Number, alg::TDVP2,
        envs::AbstractMPSEnvironments = environments(ψ, H, ψ);
        imaginary_evolution::Bool = false
    )

    # sweep left to right
    for i in 1:(length(ψ) - 1)
        ac2 = _transpose_front(ψ.AC[i]) * _transpose_tail(ψ.AR[i + 1])
        Hac2 = AC2_hamiltonian(i, ψ, H, ψ, envs)
        ac2′ = integrate(Hac2, ac2, t, dt / 2, alg.integrator; imaginary_evolution)

        nal, nc, nar = svd_trunc!(ac2′; trunc = alg.trscheme, alg = alg.alg_svd)
        ψ.AC[i] = (nal, complex(nc))
        ψ.AC[i + 1] = (complex(nc), _transpose_front(nar))

        if i != (length(ψ) - 1)
            Hac = AC_hamiltonian(i + 1, ψ, H, ψ, envs)
            ψ.AC[i + 1] = integrate(
                Hac, ψ.AC[i + 1], t + dt / 2, -dt / 2, alg.integrator;
                imaginary_evolution
            )
        end
    end

    # sweep right to left
    for i in length(ψ):-1:2
        ac2 = _transpose_front(ψ.AL[i - 1]) * _transpose_tail(ψ.AC[i])
        Hac2 = AC2_hamiltonian(i - 1, ψ, H, ψ, envs)
        ac2′ = integrate(Hac2, ac2, t + dt / 2, dt / 2, alg.integrator; imaginary_evolution)

        nal, nc, nar = svd_trunc!(ac2′; trunc = alg.trscheme, alg = alg.alg_svd)
        ψ.AC[i - 1] = (nal, complex(nc))
        ψ.AC[i] = (complex(nc), _transpose_front(nar))

        if i != 2
            Hac = AC_hamiltonian(i - 1, ψ, H, ψ, envs)
            ψ.AC[i - 1] = integrate(
                Hac, ψ.AC[i - 1], t + dt, -dt / 2, alg.integrator;
                imaginary_evolution
            )
        end
    end

    return ψ, envs
end

# copying version
function timestep(
        ψ::AbstractFiniteMPS, H, time::Number, timestep::Number,
        alg::Union{TDVP, TDVP2}, envs::AbstractMPSEnvironments...;
        imaginary_evolution::Bool = false, kwargs...
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
    return timestep!(ψ′, H, time, timestep, alg, envs′; imaginary_evolution, kwargs...)
end
