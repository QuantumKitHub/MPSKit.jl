"""
    VUMPS{F} <: Algorithm

Variational optimization algorithm for uniform matrix product states, as introduced in
https://arxiv.org/abs/1701.07035.

# Fields
- `tol_galerkin::Float64`: tolerance for convergence criterium
- `maxiter::Int`: maximum amount of iterations
- `finalize::F`: user-supplied function which is applied after each iteration, with
    signature `finalize(iter, ψ, H, envs) -> ψ, envs`
- `verbose::Bool`: display progress information

- `alg_gauge=Defaults.alg_gauge()`: algorithm for gauging
- `alg_eigsolve=Defaults.alg_eigsolve()`: algorithm for eigensolvers
- `alg_environments=Defaults.alg_environments()`: algorithm for updating environments
"""
@kwdef struct VUMPS{F} <: Algorithm
    tol_galerkin::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
    finalize::F = Defaults._finalize
    verbose::Bool = Defaults.verbose

    alg_gauge = Defaults.alg_gauge()
    alg_eigsolve = Defaults.alg_eigsolve()
    alg_environments = Defaults.alg_environments()
end

function find_groundstate(ψ::InfiniteMPS, H, alg::VUMPS, envs=environments(ψ, H))
    t₀ = Base.time_ns()
    ϵ::Float64 = calc_galerkin(ψ, envs)
    temp_ACs = similar.(ψ.AC)

    for iter in 1:(alg.maxiter)
        Δt = @elapsed begin
            alg_eigsolve = updatetol(alg.alg_eigsolve, iter, ϵ)
            @static if Defaults.parallelize_sites
                @sync begin
                    for loc in 1:length(ψ)
                        Threads.@spawn begin
                            _vumps_localupdate!(temp_ACs[loc], loc, ψ, H, envs,
                                                alg_eigsolve)
                        end
                    end
                end
            else
                for loc in 1:length(ψ)
                    _vumps_localupdate!(temp_ACs[loc], loc, ψ, H, envs, alg_eigsolve)
                end
            end

            alg_gauge = updatetol(alg.alg_gauge, iter, ϵ)
            ψ = InfiniteMPS(temp_ACs, ψ.CR[end]; alg_gauge.tol, alg_gauge.maxiter)

            alg_environments = updatetol(alg.alg_environments, iter, ϵ)
            recalculate!(envs, ψ; alg_environments.tol)

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
