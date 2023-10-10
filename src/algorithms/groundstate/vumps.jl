"""
    VUMPS{F} <: Algorithm

Variational optimization algorithm for uniform matrix product states, as introduced in
https://arxiv.org/abs/1701.07035.

# Fields
- `tol_galerkin::Float64`: tolerance for convergence criterium
- `maxiter::Int`: maximum amount of iterations
- `orthmaxiter::Int`: maximum amount of gauging iterations
- `finalize::F`: user-supplied function which is applied after each iteration, with
    signature `finalize(iter, Ψ, H, envs) -> Ψ, envs`
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

function updatetols(alg::VUMPS, iter, ε)
    if alg.dynamical_tols
        tol_eigs = between(alg.tol_min, ε * alg.eigs_tolfactor / sqrt(iter), alg.tol_max)
        tol_envs = between(alg.tol_min, ε * alg.envs_tolfactor / sqrt(iter), alg.tol_max)
        tol_gauge = between(alg.tol_min, ε * alg.gauge_tolfactor / sqrt(iter), alg.tol_max)
    else # preserve legacy behavior
        tol_eigs = alg.tol_galerkin / 10
        tol_envs = Defaults.tol
        tol_gauge = Defaults.tolgauge
    end
    return tol_eigs, tol_envs, tol_gauge
end

"
    find_groundstate(Ψ,ham,alg,envs=environments(Ψ,ham))

    find the groundstate for ham using algorithm alg
"

function find_groundstate(Ψ::InfiniteMPS, H, alg::VUMPS, envs=environments(Ψ, H))
    t₀ = Base.time_ns()
    ε::Float64 = calc_galerkin(Ψ, envs)
    temp_ACs = similar.(Ψ.AC)

    for iter in 1:(alg.maxiter)
        tol_eigs, tol_envs, tol_gauge = updatetols(alg, iter, ε)
        Δt = @elapsed begin
            eigalg = Arnoldi(; tol=tol_eigs)

            @static if Defaults.parallelize_sites
                @sync begin
                    for loc in 1:length(Ψ)
                        Threads.@spawn begin
                            _vumps_localupdate!(temp_ACs[loc], loc, Ψ, H, envs, eigalg)
                        end
                    end
                end
            else
                for loc in 1:length(Ψ)
                    _vumps_localupdate!(temp_ACs[loc], loc, Ψ, H, envs, eigalg)
                end
            end

            Ψ = InfiniteMPS(temp_ACs, Ψ.CR[end]; tol=tol_gauge, maxiter=alg.orthmaxiter)
            recalculate!(envs, Ψ; tol=tol_envs)

            Ψ, envs = alg.finalize(iter, Ψ, H, envs)::Tuple{typeof(Ψ),typeof(envs)}

            ε = calc_galerkin(Ψ, envs)
        end

        alg.verbose &&
            @info "VUMPS iteration:" iter ε λ = sum(expectation_value(Ψ, H, envs)) Δt

        ε <= alg.tol_galerkin && break
        iter == alg.maxiter &&
            @warn "VUMPS maximum iterations" iter ε λ = sum(expectation_value(Ψ, H, envs)) Δt
    end

    alg.verbose && @info "VUMPS summary:" ε λ = sum(expectation_value(Ψ, H, envs)) Δt = (
        (Base.time_ns() - t₀) / 1.0e9
    )
    return Ψ, envs, ε
end

function _vumps_localupdate!(AC′, loc, Ψ, H, envs, eigalg, factalg=QRpos())
    local Q_AC, Q_C
    @static if Defaults.parallelize_sites
        @sync begin
            Threads.@spawn begin
                _, acvecs = eigsolve(∂∂AC(loc, Ψ, H, envs), Ψ.AC[loc], 1, :SR, eigalg)
                Q_AC, _ = TensorKit.leftorth!(acvecs[1]; alg=factalg)
            end
            Threads.@spawn begin
                _, crvecs = eigsolve(∂∂C(loc, Ψ, H, envs), Ψ.CR[loc], 1, :SR, eigalg)
                Q_C, _ = TensorKit.leftorth!(crvecs[1]; alg=factalg)
            end
        end
    else
        _, acvecs = eigsolve(∂∂AC(loc, Ψ, H, envs), Ψ.AC[loc], 1, :SR, eigalg)
        Q_AC, _ = TensorKit.leftorth!(acvecs[1]; alg=factalg)
        _, crvecs = eigsolve(∂∂C(loc, Ψ, H, envs), Ψ.CR[loc], 1, :SR, eigalg)
        Q_C, _ = TensorKit.leftorth!(crvecs[1]; alg=factalg)
    end
    return mul!(AC′, Q_AC, adjoint(Q_C))
end
