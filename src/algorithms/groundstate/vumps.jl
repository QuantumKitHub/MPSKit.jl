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
struct VUMPS{F} <: Algorithm
    tol::Float64
    maxiter::Int
    orthmaxiter::Int
    finalize::F
    verbosity::Int
    dynamical_tols::Bool
    tol_min::Float64
    tol_max::Float64
    eigs_tolfactor::Float64
    envs_tolfactor::Float64
    gauge_tolfactor::Float64
end

function VUMPS(; tol::Real=Defaults.tol, maxiter::Integer=Defaults.maxiter,
               orthmaxiter::Integer=Defaults.maxiter,
               finalize=Defaults._finalize,
               verbosity::Integer=Defaults.verbosity,
               dynamical_tols::Bool=Defaults.dynamical_tols,
               tol_min::Real=Defaults.tol_min,
               tol_max::Real=Defaults.tol_max,
               eigs_tolfactor::Real=Defaults.eigs_tolfactor,
               envs_tolfactor::Real=Defaults.envs_tolfactor,
               gauge_tolfactor::Real=Defaults.gauge_tolfactor,
               tol_galerkin=nothing,
               verbose=nothing)
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
    return VUMPS{typeof(finalize)}(actual_tol, maxiter, orthmaxiter, finalize, actual_verbosity, dynamical_tols,
                    tol_min, tol_max, eigs_tolfactor, envs_tolfactor, gauge_tolfactor)
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

function updatetols(alg::VUMPS, iter, ϵ)
    if alg.dynamical_tols
        tol_eigs = between(alg.tol_min, ϵ * alg.eigs_tolfactor / sqrt(iter), alg.tol_max)
        tol_envs = between(alg.tol_min, ϵ * alg.envs_tolfactor / sqrt(iter), alg.tol_max)
        tol_gauge = between(alg.tol_min, ϵ * alg.gauge_tolfactor / sqrt(iter), alg.tol_max)
    else # preserve legacy behavior
        tol_eigs = alg.tol / 10
        tol_envs = Defaults.tol
        tol_gauge = Defaults.tolgauge
    end
    return tol_eigs, tol_envs, tol_gauge
end

function find_groundstate(ψ::InfiniteMPS, H, alg::VUMPS, envs=environments(ψ, H))
    t₀ = Base.time_ns()
    ϵ::Float64 = calc_galerkin(ψ, envs)
    temp_ACs = similar.(ψ.AC)

    for iter in 1:(alg.maxiter)
        tol_eigs, tol_envs, tol_gauge = updatetols(alg, iter, ϵ)
        Δt = @elapsed begin
            eigalg = Arnoldi(; tol=tol_eigs, eager=true)

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

            ψ = InfiniteMPS(temp_ACs, ψ.CR[end]; tol=tol_gauge, maxiter=alg.orthmaxiter)
            recalculate!(envs, ψ; tol=tol_envs)

            ψ, envs = alg.finalize(iter, ψ, H, envs)::Tuple{typeof(ψ),typeof(envs)}

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
