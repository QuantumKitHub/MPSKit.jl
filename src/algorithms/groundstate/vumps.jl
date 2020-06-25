"""
see https://arxiv.org/abs/1701.07035
"""
@with_kw struct Vumps <: Algorithm
    tol_galerkin::Float64 = Defaults.tol
    tol_gauge::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
    orthmaxiter::Int = Defaults.maxiter
    finalize::Function = (iter,state,ham,pars) -> (state,pars, true);
    verbose::Bool = Defaults.verbose
end

"
    find_groundstate(state,ham,alg,pars=params(state,ham))

    find the groundstate for ham using algorithm alg
"
function find_groundstate(state::InfiniteMPS, H::Hamiltonian,alg::Vumps,pars=params(state,H))
    galerkin  = 1+alg.tol_galerkin
    iter      = 1

    external_conv = false
    while true
        eigalg = Arnoldi(tol=galerkin/(4*sqrt(iter)))

        acjobs = map(enumerate(state.AC)) do (loc,ac)
            @Threads.spawn eigsolve(ac, 1, :SR, eigalg) do x
                ac_prime(x, loc, state, pars)
            end
        end

        cjobs = map(enumerate(state.CR)) do (loc,cr)
            @Threads.spawn eigsolve(cr, 1, :SR, eigalg) do x
                c_prime(x, loc, state, pars)
            end
        end

        newAs = map(zip(acjobs,cjobs)) do (acj,cj)
            (e,vac,ch) = fetch(acj)
            (e,vc,ch) = fetch(cj)

            QAc,_ = TensorKit.leftorth!(vac[1], alg=QRpos())
            Qc,_  = TensorKit.leftorth!(vc[1], alg=QRpos())

            QAc*adjoint(Qc)
        end


        state = InfiniteMPS(newAs; tol = alg.tol_gauge, maxiter = alg.orthmaxiter,leftgauged=true)
        galerkin   = calc_galerkin(state, pars)
        alg.verbose && @info "vumps @iteration $(iter) galerkin = $(galerkin)"

        if (galerkin <= alg.tol_galerkin && external_conv ) || iter>=alg.maxiter
            iter>=alg.maxiter && println("vumps didn't converge $(galerkin)")
            return state, pars, galerkin
        end

        (state,pars, external_conv) = alg.finalize(iter,state,H,pars);

        iter += 1
    end
end

"calculates the galerkin error"
calc_galerkin(state::InfiniteMPS, pars) = maximum([norm(leftnull(state.AC[loc])'*ac_prime(state.AC[loc], loc, state, pars)) for loc in 1:length(state)])
