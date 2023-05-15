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
    tol_galerkin::Float64   = Defaults.tol
    tol_gauge::Float64      = Defaults.tolgauge
    maxiter::Int            = Defaults.maxiter
    orthmaxiter::Int        = Defaults.maxiter
    finalize::F             = Defaults._finalize
    verbose::Bool           = Defaults.verbose
end

"
    find_groundstate(Ψ,ham,alg,envs=environments(Ψ,ham))

    find the groundstate for ham using algorithm alg
"

function find_groundstate(Ψ::InfiniteMPS, H, alg::VUMPS, envs=environments(Ψ, H))
    galerkin::Float64 = 1 + alg.tol_galerkin
    iter = 1

    temp_ACs = similar.(Ψ.AC)
    temp_Cs = similar.(Ψ.CR)

    while true
        eigalg = Arnoldi(; tol=galerkin / (4 * sqrt(iter)))

        @sync for (loc, (ac, c)) in enumerate(zip(Ψ.AC, Ψ.CR))
            Threads.@spawn begin
                _, acvecs = eigsolve(∂∂AC($loc, $Ψ, $H, $envs), $ac, 1, :SR, eigalg)
                $temp_ACs[loc] = acvecs[1]
            end

            Threads.@spawn begin
                _, crvecs = eigsolve(∂∂C($loc, $Ψ, $H, $envs), $c, 1, :SR, eigalg)
                $temp_Cs[loc] = crvecs[1]
            end
        end

        for (i, (ac, c)) in enumerate(zip(temp_ACs, temp_Cs))
            QAc, _ = TensorKit.leftorth!(ac; alg=QRpos())
            Qc, _ = TensorKit.leftorth!(c; alg=QRpos())

            temp_ACs[i] = QAc * adjoint(Qc)
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
