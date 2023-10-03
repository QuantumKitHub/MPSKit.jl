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
    galerkin::Float64 = 1 + alg.tol_galerkin
    iter = 1

    temp_ACs = similar.(Ψ.AC)

    while true
        eigalg = Arnoldi(; tol=galerkin / (4 * sqrt(iter)))

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

        galerkin = calc_galerkin(Ψ, envs)
        alg.verbose && @info "vumps @iteration $(iter) galerkin = $(galerkin)"

        if galerkin <= alg.tol_galerkin || iter >= alg.maxiter
            iter >= alg.maxiter && @warn "vumps didn't converge $(galerkin)"
            return Ψ, envs, galerkin
        end

        iter += 1
    end
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
