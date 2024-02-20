"""
    VUMPS{F} <: Algorithm

Variational optimization algorithm for uniform matrix product states, as introduced in
https://arxiv.org/abs/1701.07035.

# Fields
- `tol::Float64`: tolerance for convergence criterium
- `maxiter::Int`: maximum amount of iterations
- `orthmaxiter::Int`: maximum amount of gauging iterations
- `finalize::F`: user-supplied function which is applied after each iteration, with
    signature `finalize(iter, ψ, H, envs) -> ψ, envs`
- `verbosity::Int`: display progress information
- `dynamical_tols::Bool`: whether to use dynamically adjusted tolerances
- `tol_min::Float64`: minimum tolerance for subroutines
- `tol_max::Float64`: maximum tolerance for subroutines
- `eigs_tolfactor::Float64`: factor for dynamically setting the eigensolver tolerance  with
    respect to the current galerkin error
- `envs_tolfactor::Float64`: factor for dynamically setting the environment tolerance with
    respect to the current galerkin error
- `gauge_tolfactor::Float64`: factor for dynamically setting the gauging tolerance with
    respect to the current galerkin error
"""
struct VUMPS{F,A,B,C} <: Algorithm
    tol::Float64
    maxiter::Int
    verbosity::Int
    finalize::F
    eigalg::A
    gaugealg::B
    envalg::C
    function VUMPS(tol, maxiter, verbosity, finalize::F, eigalg::A, gaugealg::B,
                   envalg::C) where {F,A,B,C}
        return new{F,A,B,C}(tol, maxiter, verbosity, finalize, eigalg, gaugealg, envalg)
    end
end

function VUMPS(; tol::Real=Defaults.tol, maxiter::Integer=Defaults.maxiter,
               finalize=Defaults._finalize,
               verbosity::Integer=Defaults.verbosity,
               orthmaxiter::Integer=Defaults.maxiter,
               dynamic_tols::Bool=Defaults.dynamic_tols,
               tol_min=nothing, tol_max=nothing, eigs_tolfactor=nothing,
               envs_tolfactor=nothing, gauge_tolfactor=nothing, tol_galerkin=nothing,
               verbose=nothing, dynamical_tols=nothing)
    # Deprecation warnings
    actual_tol = if !isnothing(tol_galerkin)
        Base.depwarn("VUMPS(; kwargs..., tol_galerkin=...) is deprecated. Use VUMPS(; kwargs..., tol=...) instead.",
                     :VUMPS; force=true)
        tol_galerkin
    else
        tol
    end
    actual_verbosity = if !isnothing(verbose)
        Base.depwarn("VUMPS(; kwargs..., verbose=...) is deprecated. Use VUMPS(; kwargs..., verbosity=...) instead.",
                     :VUMPS; force=true)
        verbose ? VERBOSE_ITER : VERBOSE_WARN
    else
        verbosity
    end
    dynamic_tols = if !isnothing(dynamical_tols)
        Base.depwarn("VUMPS(; kwargs..., dynamic_tols=...) is deprecated. Use VUMPS(; kwargs..., dynamic_tols=...) instead.",
                     :VUMPS; force=true)
        dynamical_tols
    else
        dynamic_tols
    end

    # Keyword handling
    eigalg = Arnoldi(; tol, eager=true, verbosity=actual_verbosity - 2)
    gauge_alg = UniformGauging(; tol, maxiter=orthmaxiter, verbosity=actual_verbosity - 2)
    envalg = (; tol, verbosity=actual_verbosity - 2)

    if !dynamic_tols
        return VUMPS(actual_tol, maxiter, actual_verbosity, finalize, eigalg, gauge_alg,
                     envalg)
    end

    # Setup dynamic tolerances
    actual_tol_min = something(tol_min, Defaults.tol_min)
    actual_tol_max = something(tol_max, Defaults.tol_max)
    dyn_eigalg = ThrottledTol(eigalg, actual_tol_min, actual_tol_max,
                              something(eigs_tolfactor, Defaults.eigs_tolfactor))
    dyn_envalg = ThrottledTol(envalg, actual_tol_min, actual_tol_max,
                              something(envs_tolfactor, Defaults.envs_tolfactor))
    dyn_orthalg = ThrottledTol(gauge_alg, actual_tol_min, actual_tol_max,
                               something(gauge_tolfactor, Defaults.gauge_tolfactor))
    return VUMPS(actual_tol, maxiter, actual_verbosity, finalize, dyn_eigalg, dyn_orthalg,
                 dyn_envalg)
end

function Base.show(io::IO, ::MIME"text/plain", alg::VUMPS)
    fn = fieldnames(typeof(alg))
    if get(io, :compact, false)
        print(io, "VUMPS(; ")
        join(io, ("$(string(field))=$(string(getfield(alg, field)))" for field in fn), ", ")
        print(io, ")")
    else
        printstyled(io, "VUMPS:\n"; underline=true)
        join(io, (" ∘ $(field) = $(getfield(alg, field))" for field in fn), "\n")
    end
    return nothing
end

function find_groundstate(ψ::InfiniteMPS, H, alg::VUMPS, envs=environments(ψ, H))
    t₀ = Base.time_ns()
    ϵ::Float64 = calc_galerkin(ψ, envs)
    temp_ACs = similar.(ψ.AC)

    for iter in 1:(alg.maxiter)
        Δt = @elapsed begin
            eigalg = updatetol(alg.eigalg, iter, ϵ)

            # TODO: make the choice of QR or LQ together with the choice of gauging alg
            @static if Defaults.parallelize_sites
                @sync begin
                    for loc in 1:length(ψ)
                        Threads.@spawn begin
                            _vumps_localupdate!(temp_ACs[loc], loc, ψ, H, envs, eigalg)
                        end
                    end
                end
            else
                for loc in 1:length(ψ)
                    _vumps_localupdate!(temp_ACs[loc], loc, ψ, H, envs, eigalg)
                end
            end

            gaugealg = updatetol(alg.gaugealg, iter, ϵ)
            AL, AR, CR = uniform_gauge(temp_ACs, ψ.CR[end], gaugealg)
            ψ = InfiniteMPS(AL, AR, CR)

            # TODO: properly pass envalg to environments
            envalg = updatetol(alg.envalg, iter, ϵ)
            recalculate!(envs, ψ; envalg.tol)

            ψ, envs = alg.finalize(iter, ψ, H, envs)::typeof((ψ, envs))

            ϵ = calc_galerkin(ψ, envs)
        end

        alg.verbosity >= VERBOSE_ITER &&
            @info "VUMPS iteration:" iter ϵ λ = sum(expectation_value(ψ, H, envs)) Δt

        ϵ <= alg.tol && break
        alg.verbosity >= VERBOSE_WARN && iter == alg.maxiter &&
            @warn "VUMPS maximum iterations" iter ϵ λ = sum(expectation_value(ψ, H, envs))
    end

    Δt = (Base.time_ns() - t₀) / 1.0e9
    alg.verbosity >= VERBOSE_CONVERGENCE &&
        @info "VUMPS summary:" ϵ λ = sum(expectation_value(ψ, H, envs)) Δt
    return ψ, envs, ϵ
end

function _vumps_localupdate!(AC′, loc, ψ, H, envs, eigalg, factalg=QRpos())
    local Q_AC, Q_C
    @static if Defaults.parallelize_sites
        @sync begin
            Threads.@spawn begin
                _, acvecs = eigsolve(∂∂AC(loc, ψ, H, envs), ψ.AC[loc], 1, :SR, eigalg)
                Q_AC, _ = TensorKit.leftorth!(acvecs[1]; alg=factalg)
            end
            Threads.@spawn begin
                _, crvecs = eigsolve(∂∂C(loc, ψ, H, envs), ψ.CR[loc], 1, :SR, eigalg)
                Q_C, _ = TensorKit.leftorth!(crvecs[1]; alg=factalg)
            end
        end
    else
        _, acvecs = eigsolve(∂∂AC(loc, ψ, H, envs), ψ.AC[loc], 1, :SR, eigalg)
        Q_AC, _ = TensorKit.leftorth!(acvecs[1]; alg=factalg)
        _, crvecs = eigsolve(∂∂C(loc, ψ, H, envs), ψ.CR[loc], 1, :SR, eigalg)
        Q_C, _ = TensorKit.leftorth!(crvecs[1]; alg=factalg)
    end
    return mul!(AC′, Q_AC, adjoint(Q_C))
end
