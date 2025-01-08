"""
    TDVP{A} <: Algorithm

Single site [TDVP](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.107.070601)
algorithm for time evolution.

# Fields
- `integrator::A`: integration algorithm (defaults to Lanczos exponentiation)
- `tolgauge::Float64`: tolerance for gauging algorithm
- `gaugemaxiter::Int`: maximum amount of gauging iterations
- `finalize::F`: user-supplied function which is applied after each timestep, with
    signature `finalize(t, Ψ, H, envs) -> Ψ, envs`
"""
@kwdef struct TDVP{A,F} <: Algorithm
    integrator::A = Defaults.alg_expsolve()
    tolgauge::Float64 = Defaults.tolgauge
    gaugemaxiter::Int = Defaults.maxiter
    finalize::F = Defaults._finalize
end

function timestep(ψ::InfiniteMPS, H, t::Number, dt::Number, alg::TDVP,
                  envs::AbstractMPSEnvironments=environments(ψ, H);
                  leftorthflag=true)
    temp_ACs = similar(ψ.AC)
    temp_Cs = similar(ψ.C)

    scheduler = Defaults.scheduler[]
    if scheduler isa SerialScheduler
        temp_ACs = tmap!(temp_ACs, 1:length(ψ); scheduler) do loc
            return integrate(∂∂AC(loc, ψ, H, envs), ψ.AC[loc], t, dt, alg.integrator)
        end
        temp_Cs = tmap!(temp_Cs, 1:length(ψ); scheduler) do loc
            return integrate(∂∂C(loc, ψ, H, envs), ψ.C[loc], t, dt, alg.integrator)
        end
    else
        @sync begin
            Threads.@spawn begin
                temp_ACs = tmap!(temp_ACs, 1:length(ψ); scheduler) do loc
                    return integrate(∂∂AC(loc, ψ, H, envs), ψ.AC[loc], t, dt,
                                     alg.integrator)
                end
            end
            Threads.@spawn begin
                temp_Cs = tmap!(temp_Cs, 1:length(ψ); scheduler) do loc
                    return integrate(∂∂C(loc, ψ, H, envs), ψ.C[loc], t, dt, alg.integrator)
                end
            end
        end
    end

    if leftorthflag
        regauge!.(temp_ACs, temp_Cs; alg=TensorKit.QRpos())
        ψ′ = InfiniteMPS(temp_ACs, ψ.C[end]; tol=alg.tolgauge, maxiter=alg.gaugemaxiter)
    else
        circshift!(temp_Cs, 1)
        regauge!.(temp_Cs, temp_ACs; alg=TensorKit.LQpos())
        ψ′ = InfiniteMPS(ψ.C[0], temp_ACs; tol=alg.tolgauge, maxiter=alg.gaugemaxiter)
    end

    recalculate!(envs, ψ′)
    return ψ′, envs
end

function timestep!(ψ::AbstractFiniteMPS, H, t::Number, dt::Number, alg::TDVP,
                   envs::AbstractMPSEnvironments=environments(ψ, H))

    # sweep left to right
    for i in 1:(length(ψ) - 1)
        h_ac = ∂∂AC(i, ψ, H, envs)
        ψ.AC[i] = integrate(h_ac, ψ.AC[i], t, dt / 2, alg.integrator)

        h_c = ∂∂C(i, ψ, H, envs)
        ψ.C[i] = integrate(h_c, ψ.C[i], t, -dt / 2, alg.integrator)
    end

    # edge case
    h_ac = ∂∂AC(length(ψ), ψ, H, envs)
    ψ.AC[end] = integrate(h_ac, ψ.AC[end], t, dt / 2, alg.integrator)

    # sweep right to left
    for i in length(ψ):-1:2
        h_ac = ∂∂AC(i, ψ, H, envs)
        ψ.AC[i] = integrate(h_ac, ψ.AC[i], t + dt / 2, dt / 2, alg.integrator)

        h_c = ∂∂C(i - 1, ψ, H, envs)
        ψ.C[i - 1] = integrate(h_c, ψ.C[i - 1], t + dt / 2, -dt / 2, alg.integrator)
    end

    # edge case
    h_ac = ∂∂AC(1, ψ, H, envs)
    ψ.AC[1] = integrate(h_ac, ψ.AC[1], t + dt / 2, dt / 2, alg.integrator)

    return ψ, envs
end

"""
    TDVP2{A} <: Algorithm

2-site [TDVP](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.107.070601)
algorithm for time evolution.

# Fields
- `integrator::A`: integrator algorithm (defaults to Lanczos exponentiation)
- `tolgauge::Float64`: tolerance for gauging algorithm
- `gaugemaxiter::Int`: maximum amount of gauging iterations
- `trscheme`: truncation algorithm for [tsvd][TensorKit.tsvd](@ref)
- `finalize::F`: user-supplied function which is applied after each timestep, with
    signature `finalize(t, Ψ, H, envs) -> Ψ, envs`
"""
@kwdef struct TDVP2{A,F} <: Algorithm
    integrator::A = Defaults.alg_expsolve()
    tolgauge::Float64 = Defaults.tolgauge
    gaugemaxiter::Int = Defaults.maxiter
    trscheme = truncerr(1e-3)
    finalize::F = Defaults._finalize
end

function timestep!(ψ::AbstractFiniteMPS, H, t::Number, dt::Number, alg::TDVP2,
                   envs::AbstractMPSEnvironments=environments(ψ, H))

    # sweep left to right
    for i in 1:(length(ψ) - 1)
        ac2 = _transpose_front(ψ.AC[i]) * _transpose_tail(ψ.AR[i + 1])
        h_ac2 = ∂∂AC2(i, ψ, H, envs)
        nac2 = integrate(h_ac2, ac2, t, dt / 2, alg.integrator)

        nal, nc, nar = tsvd!(nac2; trunc=alg.trscheme, alg=TensorKit.SVD())
        ψ.AC[i] = (nal, complex(nc))
        ψ.AC[i + 1] = (complex(nc), _transpose_front(nar))

        if i != (length(ψ) - 1)
            ψ.AC[i + 1] = integrate(∂∂AC(i + 1, ψ, H, envs), ψ.AC[i + 1], t, -dt / 2,
                                    alg.integrator)
        end
    end

    # sweep right to left
    for i in length(ψ):-1:2
        ac2 = _transpose_front(ψ.AL[i - 1]) * _transpose_tail(ψ.AC[i])
        h_ac2 = ∂∂AC2(i - 1, ψ, H, envs)
        nac2 = integrate(h_ac2, ac2, t + dt / 2, dt / 2, alg.integrator)

        nal, nc, nar = tsvd!(nac2; trunc=alg.trscheme, alg=TensorKit.SVD())
        ψ.AC[i - 1] = (nal, complex(nc))
        ψ.AC[i] = (complex(nc), _transpose_front(nar))

        if i != 2
            ψ.AC[i - 1] = integrate(∂∂AC(i - 1, ψ, H, envs), ψ.AC[i - 1], t + dt / 2,
                                    -dt / 2, alg.integrator)
        end
    end

    return ψ, envs
end

#copying version
function timestep(ψ::AbstractFiniteMPS, H, time::Number, timestep::Number,
                  alg::Union{TDVP,TDVP2}, envs::AbstractMPSEnvironments=environments(ψ, H);
                  kwargs...)
    return timestep!(copy(ψ), H, time, timestep, alg, envs; kwargs...)
end
