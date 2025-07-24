"""
$(TYPEDEF)

Single site MPS time-evolution algorithm based on the Time-Dependent Variational Principle.

## Fields

$(TYPEDFIELDS)

## References

* [Haegeman et al. Phys. Rev. Lett. 107 (2011)](@cite haegeman2011)
"""
@kwdef struct TDVP{A, F} <: Algorithm
    "algorithm used in the exponential solvers"
    integrator::A = Defaults.alg_expsolve()

    "tolerance for gauging algorithm"
    tolgauge::Float64 = Defaults.tolgauge

    "maximal amount of iterations for gauging algorithm"
    gaugemaxiter::Int = Defaults.maxiter

    "callback function applied after each iteration, of signature `finalize(iter, ψ, H, envs) -> ψ, envs`"
    finalize::F = Defaults._finalize
end

function timestep(
        ψ_::InfiniteMPS, H, t::Number, dt::Number, alg::TDVP,
        envs::AbstractMPSEnvironments = environments(ψ_, H);
        leftorthflag = true
    )
    ψ = complex(ψ_)
    temp_ACs = similar(ψ.AC)
    temp_Cs = similar(ψ.C)

    scheduler = Defaults.scheduler[]
    if scheduler isa SerialScheduler
        temp_ACs = tmap!(temp_ACs, 1:length(ψ); scheduler) do loc
            Hac = AC_hamiltonian(loc, ψ, H, ψ, envs)
            return integrate(Hac, ψ.AC[loc], t, dt, alg.integrator)
        end
        temp_Cs = tmap!(temp_Cs, 1:length(ψ); scheduler) do loc
            Hc = C_hamiltonian(loc, ψ, H, ψ, envs)
            return integrate(Hc, ψ.C[loc], t, dt, alg.integrator)
        end
    else
        @sync begin
            Threads.@spawn begin
                temp_ACs = tmap!(temp_ACs, 1:length(ψ); scheduler) do loc
                    Hac = AC_hamiltonian(loc, ψ, H, ψ, envs)
                    return integrate(Hac, ψ.AC[loc], t, dt, alg.integrator)
                end
            end
            Threads.@spawn begin
                temp_Cs = tmap!(temp_Cs, 1:length(ψ); scheduler) do loc
                    Hc = C_hamiltonian(loc, ψ, H, ψ, envs)
                    return integrate(Hc, ψ.C[loc], t, dt, alg.integrator)
                end
            end
        end
    end

    if leftorthflag
        regauge!.(temp_ACs, temp_Cs; alg = TensorKit.QRpos())
        ψ′ = InfiniteMPS(temp_ACs, ψ.C[end]; tol = alg.tolgauge, maxiter = alg.gaugemaxiter)
    else
        circshift!(temp_Cs, 1)
        regauge!.(temp_Cs, temp_ACs; alg = TensorKit.LQpos())
        ψ′ = InfiniteMPS(ψ.C[0], temp_ACs; tol = alg.tolgauge, maxiter = alg.gaugemaxiter)
    end

    recalculate!(envs, ψ′, H)
    return ψ′, envs
end

function timestep!(
        ψ::AbstractFiniteMPS, H, t::Number, dt::Number, alg::TDVP,
        envs::AbstractMPSEnvironments = environments(ψ, H)
    )

    # sweep left to right
    for i in 1:(length(ψ) - 1)
        Hac = AC_hamiltonian(i, ψ, H, ψ, envs)
        ψ.AC[i] = integrate(Hac, ψ.AC[i], t, dt / 2, alg.integrator)

        Hc = C_hamiltonian(i, ψ, H, ψ, envs)
        ψ.C[i] = integrate(Hc, ψ.C[i], t + dt / 2, -dt / 2, alg.integrator)
    end

    # edge case
    Hac = AC_hamiltonian(length(ψ), ψ, H, ψ, envs)
    ψ.AC[end] = integrate(Hac, ψ.AC[end], t, dt / 2, alg.integrator)

    # sweep right to left
    for i in length(ψ):-1:2
        Hac = AC_hamiltonian(i, ψ, H, ψ, envs)
        ψ.AC[i] = integrate(Hac, ψ.AC[i], t + dt / 2, dt / 2, alg.integrator)

        Hc = C_hamiltonian(i - 1, ψ, H, ψ, envs)
        ψ.C[i - 1] = integrate(Hc, ψ.C[i - 1], t + dt, -dt / 2, alg.integrator)
    end

    # edge case
    Hac = AC_hamiltonian(1, ψ, H, ψ, envs)
    ψ.AC[1] = integrate(Hac, ψ.AC[1], t + dt / 2, dt / 2, alg.integrator)

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
    trscheme::TruncationScheme

    "callback function applied after each iteration, of signature `finalize(iter, ψ, H, envs) -> ψ, envs`"
    finalize::F = Defaults._finalize
end

function timestep!(
        ψ::AbstractFiniteMPS, H, t::Number, dt::Number, alg::TDVP2,
        envs::AbstractMPSEnvironments = environments(ψ, H)
    )

    # sweep left to right
    for i in 1:(length(ψ) - 1)
        ac2 = _transpose_front(ψ.AC[i]) * _transpose_tail(ψ.AR[i + 1])
        Hac2 = AC2_hamiltonian(i, ψ, H, ψ, envs)
        ac2′ = integrate(Hac2, ac2, t, dt / 2, alg.integrator)

        nal, nc, nar = tsvd!(ac2′; trunc = alg.trscheme, alg = alg.alg_svd)
        ψ.AC[i] = (nal, complex(nc))
        ψ.AC[i + 1] = (complex(nc), _transpose_front(nar))

        if i != (length(ψ) - 1)
            Hac = AC_hamiltonian(i + 1, ψ, H, ψ, envs)
            ψ.AC[i + 1] = integrate(Hac, ψ.AC[i + 1], t + dt / 2, -dt / 2, alg.integrator)
        end
    end

    # sweep right to left
    for i in length(ψ):-1:2
        ac2 = _transpose_front(ψ.AL[i - 1]) * _transpose_tail(ψ.AC[i])
        Hac2 = AC2_hamiltonian(i - 1, ψ, H, ψ, envs)
        ac2′ = integrate(Hac2, ac2, t + dt / 2, dt / 2, alg.integrator)

        nal, nc, nar = tsvd!(ac2′; trunc = alg.trscheme, alg = alg.alg_svd)
        ψ.AC[i - 1] = (nal, complex(nc))
        ψ.AC[i] = (complex(nc), _transpose_front(nar))

        if i != 2
            Hac = AC_hamiltonian(i - 1, ψ, H, ψ, envs)
            ψ.AC[i - 1] = integrate(Hac, ψ.AC[i - 1], t + dt, -dt / 2, alg.integrator)
        end
    end

    return ψ, envs
end

#copying version
function timestep(
        ψ::AbstractFiniteMPS, H, time::Number, timestep::Number,
        alg::Union{TDVP, TDVP2}, envs::AbstractMPSEnvironments...;
        kwargs...
    )
    isreal = scalartype(ψ) <: Real
    ψ′ = isreal ? complex(ψ) : copy(ψ)
    if length(envs) != 0 && isreal
        @warn "Currently cannot reuse real environments for complex evolution"
        envs′ = environments(ψ′, H)
    elseif length(envs) == 1
        envs′ = only(envs)
    else
        @assert length(envs) == 0 "Invalid signature"
        envs′ = environments(ψ′, H)
    end
    return timestep!(ψ′, H, time, timestep, alg, envs′; kwargs...)
end
