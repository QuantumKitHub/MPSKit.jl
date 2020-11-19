function approximate(state,toapprox,alg,envs=environments(state,toapprox))
    copied_envs = deepcopy(envs);
    copied_state = deepcopy(state);

    approximate!(copied_state,toapprox,alg,copied_envs)
end

function approximate!(state::InfiniteMPS,toapprox::Tuple{<:InfiniteMPS,<:PeriodicMPO},alg::PowerMethod,envs=environments(state,toapprox))
    #PeriodicMPO's always act on MPSMultiline's. I therefore convert the imps to multilines, approximate! and convert back

    (multi,envs) = approximate!(convert(MPSMultiline,state),(envs.above,envs.opp),alg,envs)
    copyto!(state,multi)
    return (state,envs)
end


function approximate!(state::MPSMultiline,toapprox::Tuple{<:MPSMultiline,<:PeriodicMPO},alg::PowerMethod,envs = environments(init,toapprox))
    (above,mpo) = toapprox;

    above === state && throw(ArgumentError("cannot approximate and modify in-place the same mps"))

    galerkin  = 1+alg.tol_galerkin
    iter       = 1

    temp_ACs = similar(state.AC);
    temp_Cs = similar(state.CR);

    while true

        @sync for col in 1:size(state,2)
            @Threads.spawn begin
                temp_ACs[:,col] = let state=state,envs=envs
                    circshift([ac_prime(ac,row,col,state,envs) for (row,ac) in enumerate(above.AC[:,col])],1)
                end
            end

            @Threads.spawn begin
                temp_Cs[:,col]  = let state=state,envs=envs
                    circshift([c_prime(c,row,col,state,envs) for (row,c) in enumerate(above.CR[:,col])],1)
                end
            end
        end

        for row in 1:size(state,1),col in 1:size(state,2)
            QAc,_ = leftorth!(temp_ACs[row,col], alg=TensorKit.QRpos())
            Qc,_  = leftorth!(temp_Cs[row,col], alg=TensorKit.QRpos())
            state.AL[row,col] = QAc*adjoint(Qc)
        end

        reorth!(state; tol = alg.tol_gauge, maxiter = alg.orthmaxiter);
        recalculate!(envs,state);

        galerkin   = calc_galerkin(state, envs)
        alg.verbose && @info "powermethod @iteration $(iter) galerkin = $(galerkin)"

        (state,envs,sc) = alg.finalize!(iter,state,toapprox,envs);

        if (galerkin <= alg.tol_galerkin && sc) || iter>=alg.maxiter
            iter>=alg.maxiter && @warn "powermethod didn't converge $(galerkin)"
            return state, envs, galerkin
        end



        iter += 1
    end
end
