"""
    VUMPS{F} <: Algorithm

Variational optimization algorithm for uniform matrix product states, as introduced in
https://arxiv.org/abs/1701.07035.

# Fields
- `tol_galerkin::Float64`: tolerance for convergence criterium
- `tol_gauge::Float64`: tolerance for gauging algorithm
- `maxiter::Int`: maximum amount of iterations
- `orthmaxiter::Int`: maximum amount of gauging iterations
- `finalize::F`: user-supplied function which is applied after each iteration, with
    signature `finalize(iter, Ψ, H, envs) -> Ψ, envs`
- `verbose::Bool`: display progress information
"""
@kwdef struct VUMPS{F} <: Algorithm
    tol_galerkin::Float64 = Defaults.tol
    tol_gauge::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
    orthmaxiter::Int = Defaults.maxiter
    finalize::F = Defaults._finalize
    verbose::Bool = Defaults.verbose
end

"
    find_groundstate(Ψ,ham,alg,envs=environments(Ψ,ham))

    find the groundstate for ham using algorithm alg
"

function find_groundstate(Ψ::InfiniteMPS, H, alg::VUMPS, envs=environments(Ψ, H))
    t₀ = Base.time_ns()
    ε::Float64 = 1 + alg.tol_galerkin
    temp_ACs = similar.(Ψ.AC)

    for iter in 1:(alg.maxiter)
        Δt = @elapsed begin
            eigalg = Arnoldi(; tol=ε / (4 * sqrt(iter)))

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

            Ψ = InfiniteMPS(temp_ACs, Ψ.CR[end]; tol=alg.tol_gauge, maxiter=alg.orthmaxiter)
            recalculate!(envs, Ψ)

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
