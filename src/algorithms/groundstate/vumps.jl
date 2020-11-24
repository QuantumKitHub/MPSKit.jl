"""
see https://arxiv.org/abs/1701.07035
"""
@with_kw struct Vumps{F} <: Algorithm
    tol_galerkin::Float64 = Defaults.tol
    tol_gauge::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
    orthmaxiter::Int = Defaults.maxiter
    finalize::F = Defaults._finalize
    verbose::Bool = Defaults.verbose
end

"
    find_groundstate(state,ham,alg,envs=environments(state,ham))

    find the groundstate for ham using algorithm alg
"

function find_groundstate(state::InfiniteMPS{A,B}, H::Hamiltonian,alg::Vumps,envs::P=environments(state,H))::Tuple{InfiniteMPS{A,B},P,Float64} where {A,B,P}
    galerkin::Float64  = 1+alg.tol_galerkin
    iter      = 1

    temp_ACs = similar(state.AC);
    temp_Cs = similar(state.CR);

    while true
        eigalg = Arnoldi(tol=galerkin/(4*sqrt(iter)))

        @sync for (loc,(ac,c)) in enumerate(zip(state.AC,state.CR))
            @Threads.spawn begin
                (acvals,acvecs) = let state=state,envs=envs
                    eigsolve(ac, 1, :SR, eigalg) do x
                        ac_prime(x, loc, state, envs)
                    end
                end
                temp_ACs[loc] = acvecs[1];
            end

            @Threads.spawn begin
                (crvals,crvecs) = let state=state,envs=envs
                    eigsolve(c, 1, :SR, eigalg) do x
                        c_prime(x, loc, state, envs)
                    end
                end
                temp_Cs[loc] = crvecs[1];
            end

        end

        for (i,(ac,c)) in enumerate(zip(temp_ACs,temp_Cs))
            QAc,_ = TensorKit.leftorth!(ac, alg=QRpos())
            Qc,_  = TensorKit.leftorth!(c, alg=QRpos())

            temp_ACs[i] = QAc*adjoint(Qc)
        end

        state = InfiniteMPS(temp_ACs,state.CR[end]; tol = alg.tol_gauge, maxiter = alg.orthmaxiter)
        recalculate!(envs,state);

        galerkin   = calc_galerkin(state, envs)
        alg.verbose && @info "vumps @iteration $(iter) galerkin = $(galerkin)"

        (state,envs, external_conv) = alg.finalize(iter,state,H,envs) :: Tuple{InfiniteMPS{A,B},P,Bool};
        if (galerkin <= alg.tol_galerkin && external_conv ) || iter>=alg.maxiter
            iter>=alg.maxiter && @warn "vumps didn't converge $(galerkin)"
            return state, envs, galerkin
        end

        iter += 1
    end
end
