@with_kw struct PowerMethod{TruncT<:Algorithm} <: Algorithm
    tol_galerkin::Float64 = Defaults.tol
    tol_gauge::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
    truncalg::TruncT = SimpleManager()
    verbose::Bool = Defaults.verbose
end

function leading_boundary(state::MpsMultiline, H,alg::PowerMethod,pars=params(state,H))
    galerkin  = 1+alg.tol_galerkin
    iter       = 1

    newAs = similar(state.AL)

    lee = galerkin

    while true
        @threads for col in 1:size(state,2)

            vac = circshift([ac_prime(ac,row,col,state,pars) for (row,ac) in enumerate(state.AC[:,col])],1)
            vc  = circshift([c_prime(c,row,col,state,pars) for (row,c) in enumerate(state.CR[:,col])],1)

            for row in 1:size(state,1)
                QAc,_ = TensorKit.leftorth!(vac[row], alg=TensorKit.Polar())
                Qc,_  = TensorKit.leftorth!(vc[row], alg=TensorKit.Polar())
                newAs[row,col] = QAc*adjoint(Qc)
            end

        end

        state = MpsMultiline(newAs; tol = alg.tol_gauge, maxiter = alg.maxiter)
        galerkin   = calc_galerkin(state, pars)
        alg.verbose && println("powermethod @iteration $(iter) galerkin = $(galerkin)")

        #dynamical bonds
        if galerkin < lee
            lee = galerkin
            state, pars = managebonds(state,H,alg.truncalg,pars)
        end

        if galerkin <= alg.tol_galerkin || iter>=alg.maxiter
            iter>=alg.maxiter && println("powermethod didn't converge $(galerkin)")
            return state, pars, galerkin
        end

        iter += 1
    end
end
