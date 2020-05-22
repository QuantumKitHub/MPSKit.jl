"""
    PowerMethod way of finding the leading boundary mps
"""
@with_kw struct PowerMethod <: Algorithm
    tol_galerkin::Float64 = Defaults.tol
    tol_gauge::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
    finalize::Function = (iter,state,ham,pars) -> (state,pars);
    verbose::Bool = Defaults.verbose
end

function leading_boundary(state::MPSMultiline, H,alg::PowerMethod,pars=params(state,H))
    galerkin  = 1+alg.tol_galerkin
    iter       = 1

    newAs = similar(state.AL)

    while true

        @threads for col in 1:size(state,2)

            vac = let state=state,pars=pars
                circshift([ac_prime(ac,row,col,state,pars) for (row,ac) in enumerate(state.AC[:,col])],1)
            end
            vc  = let state=state,pars=pars
                circshift([c_prime(c,row,col,state,pars) for (row,c) in enumerate(state.CR[:,col])],1)
            end

            for row in 1:size(state,1)
                QAc,_ = leftorth!(vac[row], alg=TensorKit.Polar())
                Qc,_  = leftorth!(vc[row], alg=TensorKit.Polar())
                newAs[row,col] = QAc*adjoint(Qc)
            end

        end

        state = MPSMultiline(newAs; tol = alg.tol_gauge, maxiter = alg.maxiter)
        galerkin   = calc_galerkin(state, pars)
        alg.verbose && println("powermethod @iteration $(iter) galerkin = $(galerkin)")

        if galerkin <= alg.tol_galerkin || iter>=alg.maxiter
            iter>=alg.maxiter && println("powermethod didn't converge $(galerkin)")
            return state, pars, galerkin
        end

        (state,pars) = alg.finalize(iter,state,H,pars);

        iter += 1
    end
end
