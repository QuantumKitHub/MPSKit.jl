"""
see https://arxiv.org/abs/1701.07035
"""
@with_kw struct Vumps{TruncT<:Algorithm} <: Algorithm
    tol_galerkin::Float64 = Defaults.tol
    tol_gauge::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
    manager::TruncT = SimpleManager()
    verbose::Bool = Defaults.verbose
end

"
    find_groundstate(state,ham,alg,pars=params(state,ham))

    find the groundstate for ham using algorithm alg
"
function find_groundstate(state::InfiniteMPS, H::Hamiltonian,alg::Vumps,pars=params(state,H))
    galerkin  = 1+alg.tol_galerkin
    iter       = 1

    newAs = similar(state.AL)

    lee = galerkin;

    while true
        eigalg=Arnoldi(tol=galerkin/(4*sqrt(iter)))

        for loc in 1:size(state,1)


            (e,vac,ch)=let state=state,pars=pars,eigalg=eigalg
                eigsolve(state.AC[loc], 1, :SR, eigalg) do x
                    ac_prime(x, loc, state, pars)
                end
            end

            (e,vc,ch) = let state=state,pars=pars,eigalg=eigalg
                eigsolve(state.CR[loc], 1, :SR, eigalg) do x
                    c_prime(x, loc, state, pars)
                end
            end

            QAc,_ = TensorKit.leftorth!(vac[1], alg=TensorKit.Polar())
            Qc,_  = TensorKit.leftorth!(vc[1], alg=TensorKit.Polar())

            newAs[loc]     = QAc*adjoint(Qc)

        end


        state = InfiniteMPS(newAs; tol = alg.tol_gauge, maxiter = alg.maxiter,cguess = state.CR[end],leftgauged=true)
        galerkin   = calc_galerkin(state, pars)
        alg.verbose && println("vumps @iteration $(iter) galerkin = $(galerkin)")

        #dynamical bonds
        if galerkin < lee
            lee = galerkin
            state, pars = managebonds(state,H,alg.manager,pars)
        end

        if galerkin <= alg.tol_galerkin || iter>=alg.maxiter
            iter>=alg.maxiter && println("vumps didn't converge $(galerkin)")
            return state, pars, galerkin
        end

        iter += 1
    end
end

"calculates the galerkin error"
calc_galerkin(state::InfiniteMPS, pars) = maximum([norm(leftnull(state.AC[loc])'*ac_prime(state.AC[loc], loc, state, pars)) for loc in 1:length(state)])
