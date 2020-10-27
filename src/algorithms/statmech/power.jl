"""
    PowerMethod way of finding the leading boundary mps
"""
@with_kw struct PowerMethod{F} <: Algorithm
    tol_galerkin::Float64 = Defaults.tol
    tol_gauge::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
    orthmaxiter::Int = Defaults.maxiter
    finalize::F = Defaults._finalize
    verbose::Bool = Defaults.verbose
end

function leading_boundary!(state::MPSMultiline, H,alg::PowerMethod,pars=params(state,H))
    galerkin  = 1+alg.tol_galerkin
    iter       = 1

    temp_ACs = similar(state.AC);
    temp_Cs = similar(state.CR);

    while true

        @sync for col in 1:size(state,2)
            @Threads.spawn begin
                temp_ACs[:,col] = let state=state,pars=pars
                    circshift([ac_prime(ac,row,col,state,pars) for (row,ac) in enumerate(state.AC[:,col])],1)
                end
            end

            @Threads.spawn begin
                temp_Cs  = let state=state,pars=pars
                    circshift([c_prime(c,row,col,state,pars) for (row,c) in enumerate(state.CR[:,col])],1)
                end
            end
        end

        for row in 1:size(state,1),col in 1:size(state,2)
            QAc,_ = leftorth!(temp_ACs[row,col], alg=TensorKit.QRpos())
            Qc,_  = leftorth!(temp_Cs[row,col], alg=TensorKit.QRpos())
            state.AL[row,col] = QAc*adjoint(Qc)
        end

        reorth!(state; tol = alg.tol_gauge, maxiter = alg.orthmaxiter);
        recalculate!(pars,state);

        galerkin   = calc_galerkin(state, pars)
        alg.verbose && @info "powermethod @iteration $(iter) galerkin = $(galerkin)"

        (state,pars,sc) = alg.finalize(iter,state,H,pars);

        if (galerkin <= alg.tol_galerkin && sc) || iter>=alg.maxiter
            iter>=alg.maxiter && @warn "powermethod didn't converge $(galerkin)"
            return state, pars, galerkin
        end



        iter += 1
    end
end
