"""
    VUMPS{F} <: Algorithm

Variational optimization algorithm for uniform matrix product states, as introduced in
https://arxiv.org/abs/1701.07035.

# Fields
- `tol_galerkin::Float64`: tolerance for convergence criterium
- `maxiter::Int`: maximum amount of iterations
- `orthmaxiter::Int`: maximum amount of gauging iterations
- `finalize::F`: user-supplied function which is applied after each iteration, with
    signature `finalize(iter, ψ, H, envs) -> ψ, envs`
- `verbose::Bool`: display progress information
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
@kwdef struct VUMPS{F} <: Algorithm
    tol_galerkin::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
    orthmaxiter::Int = Defaults.maxiter
    finalize::F = Defaults._finalize
    verbose::Bool = Defaults.verbose
    dynamical_tols::Bool = Defaults.dynamical_tols
    tol_min::Float64 = Defaults.tol_min
    tol_max::Float64 = Defaults.tol_max
    eigs_tolfactor::Float64 = Defaults.eigs_tolfactor
    envs_tolfactor::Float64 = Defaults.envs_tolfactor
    gauge_tolfactor::Float64 = Defaults.gauge_tolfactor
end

function updatetols(alg::VUMPS, iter, ϵ)
    if alg.dynamical_tols
        tol_eigs = between(alg.tol_min, ϵ * alg.eigs_tolfactor / sqrt(iter), alg.tol_max)
        tol_envs = between(alg.tol_min, ϵ * alg.envs_tolfactor / sqrt(iter), alg.tol_max)
        tol_gauge = between(alg.tol_min, ϵ * alg.gauge_tolfactor / sqrt(iter), alg.tol_max)
    else # preserve legacy behavior
        tol_eigs = alg.tol_galerkin / 10
        tol_envs = Defaults.tol
        tol_gauge = Defaults.tolgauge
    end
    return tol_eigs, tol_envs, tol_gauge
end

"
    find_groundstate(ψ, H, alg, envs=environments(ψ, H))

find the groundstate for `H` using algorithm `alg`
"

function find_groundstate(ψ::InfiniteMPS, H, alg::VUMPS, envs=environments(ψ, H))
    t₀ = Base.time_ns()
    ϵ::Float64 = calc_galerkin(ψ, envs)
    temp_ACs = similar.(ψ.AC)

    for iter in 1:(alg.maxiter)
        tol_eigs, tol_envs, tol_gauge = updatetols(alg, iter, ϵ)
        Δt = @elapsed begin
            eigalg = Arnoldi(; tol=tol_eigs)

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

        alg.verbose &&
            @info "VUMPS iteration:" iter ϵ λ = sum(expectation_value(ψ, H, envs)) Δt

        ϵ <= alg.tol_galerkin && break
        iter == alg.maxiter &&
            @warn "VUMPS maximum iterations" iter ϵ λ = sum(expectation_value(ψ, H, envs))
    end

    Δt = (Base.time_ns() - t₀) / 1.0e9
    alg.verbose && @info "VUMPS summary:" ϵ λ = sum(expectation_value(ψ, H, envs)) Δt
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
