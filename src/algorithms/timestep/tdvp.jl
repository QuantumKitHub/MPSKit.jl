"""
    TDVP{A,G,F} <: Algorithm

Single site [TDVP](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.107.070601)
algorithm for time evolution.

# Fields
- `integrator::A`: integration algorithm (defaults to Lanczos exponentiation)
- `gaugealg::G`: gauge algorithm (defaults to UniformOrthogonalization)
- `verbosity::Int`: verbosity level
- `finalize::F`: user-supplied function which is applied after each timestep, with
    signature `finalize(t, Ψ, H, envs) -> Ψ, envs`
"""
struct TDVP{A,G,F} <: Algorithm
    integrator::A
    gaugealg::G
    verbosity::Int
    finalize::F
    
    # automatically fill type parameters
    function TDVP(integrator::A, gaugealg::G, verbosity, finalize::F) where {A,G,F}
        new{A,G,F}(integrator, gaugealg, verbosity, finalize)
    end
end
function TDVP(; tol=Defaults.tol, integrator=nothing, tolgauge=Defaults.tolgauge,
              gaugemaxiter::Integer=Defaults.maxiter, verbosity=Defaults.verbosity,
              finalize=Defaults._finalize)
    if isnothing(integrator)
        integrator = Lanczos(; tol)
    elseif !isnothing(tol)
        integrator = @set integrator.tol = tol
    end

    gaugealg = UniformOrthogonalization(; tol=tolgauge, maxiter=gaugemaxiter)

    return TDVP(integrator, gaugealg, verbosity, finalize)
end

function timestep(Ψ::InfiniteMPS, H, t::Number, dt::Number, alg::TDVP,
                  envs::Union{Cache,MultipleEnvironments}=environments(Ψ, H);
                  leftorthflag=true)
    temp_ACs = similar(Ψ.AC)
    temp_CRs = similar(Ψ.CR)
    @sync for (loc, (ac, c)) in enumerate(zip(Ψ.AC, Ψ.CR))
        Threads.@spawn begin
            h_ac = ∂∂AC(loc, Ψ, H, envs)
            temp_ACs[loc] = integrate(h_ac, ac, t, dt, alg.integrator)
        end

        Threads.@spawn begin
            h_c = ∂∂C(loc, Ψ, H, envs)
            temp_CRs[loc] = integrate(h_c, c, t, dt, alg.integrator)
        end
    end

    if leftorthflag
        for loc in 1:length(Ψ)
            # find AL that best fits these new Acenter and centers
            QAc, _ = leftorth!(temp_ACs[loc]; alg=TensorKit.QRpos())
            Qc, _ = leftorth!(temp_CRs[loc]; alg=TensorKit.QRpos())
            @plansor temp_ACs[loc][-1 -2; -3] = QAc[-1 -2; 1] * conj(Qc[-3; 1])
        end
        newΨ = InfiniteMPS(temp_ACs, Ψ.CR[end]; alg.gaugealg.tol, alg.gaugealg.maxiter)

    else
        for loc in 1:length(Ψ)
            # find AR that best fits these new Acenter and centers
            _, QAc = rightorth!(_transpose_tail(temp_ACs[loc]); alg=TensorKit.LQpos())
            _, Qc = rightorth!(temp_CRs[mod1(loc - 1, end)]; alg=TensorKit.LQpos())
            temp_ACs[loc] = _transpose_front(Qc' * QAc)
        end
        newΨ = InfiniteMPS(Ψ.CR[0], temp_ACs; alg.gaugealg.tol, alg.gaugealg.maxiter)
    end

    recalculate!(envs, newΨ)
    return newΨ, envs
end

function timestep!(Ψ::AbstractFiniteMPS, H, t::Number, dt::Number, alg::TDVP,
                   envs::Union{Cache,MultipleEnvironments}=environments(Ψ, H))

    # sweep left to right
    for i in 1:(length(Ψ) - 1)
        h_ac = ∂∂AC(i, Ψ, H, envs)
        Ψ.AC[i] = integrate(h_ac, Ψ.AC[i], t, dt / 2, alg.integrator)

        h_c = ∂∂C(i, Ψ, H, envs)
        Ψ.CR[i] = integrate(h_c, Ψ.CR[i], t, -dt / 2, alg.integrator)
    end

    # edge case
    h_ac = ∂∂AC(length(Ψ), Ψ, H, envs)
    Ψ.AC[end] = integrate(h_ac, Ψ.AC[end], t, dt / 2, alg.integrator)

    # sweep right to left
    for i in length(Ψ):-1:2
        h_ac = ∂∂AC(i, Ψ, H, envs)
        Ψ.AC[i] = integrate(h_ac, Ψ.AC[i], t + dt / 2, dt / 2, alg.integrator)

        h_c = ∂∂C(i - 1, Ψ, H, envs)
        Ψ.CR[i - 1] = integrate(h_c, Ψ.CR[i - 1], t + dt / 2, -dt / 2, alg.integrator)
    end

    # edge case
    h_ac = ∂∂AC(1, Ψ, H, envs)
    Ψ.AC[1] = integrate(h_ac, Ψ.AC[1], t + dt / 2, dt / 2, alg.integrator)

    return Ψ, envs
end

"""
    TDVP2{A} <: Algorithm

2-site [TDVP](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.107.070601)
algorithm for time evolution.

# Fields
- `integrator::A`: integrator algorithm (defaults to Lanczos exponentiation)
- `gaugealg::G`: gauge algorithm (defaults to UniformOrthogonalization)
- `verbosity::Int`: verbosity level
- `trscheme`: truncation algorithm for [tsvd][TensorKit.tsvd](@ref)
- `finalize::F`: user-supplied function which is applied after each timestep, with
    signature `finalize(t, Ψ, H, envs) -> Ψ, envs`
"""
struct TDVP2{A,G,F} <: Algorithm
    integrator::A
    gaugealg::G
    verbosity::Int
    trscheme::TruncationScheme
    finalize::F
    
    # automatically fill type parameters
    function TDVP2(integrator::A, gaugealg::G, verbosity, trscheme, finalize::F) where {A,G,F}
        new{A,G,F}(integrator, gaugealg, verbosity, trscheme, finalize)
    end
end
function TDVP2(; tol=Defaults.tol, integrator=nothing, tolgauge=Defaults.tolgauge,
               gaugemaxiter::Integer=Defaults.maxiter, verbosity=Defaults.verbosity,
               finalize=Defaults._finalize, trscheme=truncerr(1e-3))
    if isnothing(integrator)
        integrator = Lanczos(; tol)
    elseif !isnothing(tol)
        integrator = @set integrator.tol = tol
    end

    gaugealg = UniformOrthogonalization(; tol=tolgauge, maxiter=gaugemaxiter)
    
    return TDVP2(integrator, gaugealg, verbosity, trscheme, finalize)
end

function timestep!(Ψ::AbstractFiniteMPS, H, t::Number, dt::Number, alg::TDVP2,
                   envs=environments(Ψ, H))

    # sweep left to right
    for i in 1:(length(Ψ) - 1)
        ac2 = _transpose_front(Ψ.AC[i]) * _transpose_tail(Ψ.AR[i + 1])
        h_ac2 = ∂∂AC2(i, Ψ, H, envs)
        nac2 = integrate(h_ac2, ac2, t, dt / 2, alg.integrator)

        nal, nc, nar, = tsvd!(nac2; trunc=alg.trscheme, alg=TensorKit.SVD())
        Ψ.AC[i] = (nal, complex(nc))
        Ψ.AC[i + 1] = (complex(nc), _transpose_front(nar))

        if i != (length(Ψ) - 1)
            Ψ.AC[i + 1] = integrate(∂∂AC(i + 1, Ψ, H, envs), Ψ.AC[i + 1], t, -dt / 2,
                                    alg.integrator)
        end
    end

    # sweep right to left
    for i in length(Ψ):-1:2
        ac2 = _transpose_front(Ψ.AL[i - 1]) * _transpose_tail(Ψ.AC[i])
        h_ac2 = ∂∂AC2(i - 1, Ψ, H, envs)
        nac2 = integrate(h_ac2, ac2, t + dt / 2, dt / 2, alg.integrator)

        nal, nc, nar = tsvd!(nac2; trunc=alg.trscheme, alg=TensorKit.SVD())
        Ψ.AC[i - 1] = (nal, complex(nc))
        Ψ.AC[i] = (complex(nc), _transpose_front(nar))

        if i != 2
            Ψ.AC[i - 1] = integrate(∂∂AC(i - 1, Ψ, H, envs), Ψ.AC[i - 1], t + dt / 2,
                                    -dt / 2, alg.integrator)
        end
    end

    return Ψ, envs
end

#copying version
function timestep(Ψ::AbstractFiniteMPS, H, time::Number, timestep::Number,
                  alg::Union{TDVP,TDVP2}, envs=environments(Ψ, H); kwargs...)
    return timestep!(copy(Ψ), H, time, timestep, alg, envs; kwargs...)
end
