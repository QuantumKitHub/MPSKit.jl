"""
    approximate(state::InfiniteMPS, toapprox::Tuple{<:Union{SparseMPO,DenseMPO},<:InfiniteMPS}, alg, envs = environments(state,toapprox))
    approximate(state::MPSMultiline, toapprox::Tuple{<:MPOMultiline,<:MPSMultiline}, alg::VUMPS, envs = environments(state,toapprox))
    approximate(ost::MPSMultiline, toapprox::Tuple{<:MPOMultiline,<:MPSMultiline}, alg::IDMRG1, oenvs = environments(ost,toapprox))
    approximate(ost::MPSMultiline, toapprox::Tuple{<:MPOMultiline,<:MPSMultiline}, alg::IDMRG2, oenvs = environments(ost,toapprox))

    Approximate an infinite MPO-MPS state (a state obtained by applying an iMPO to an iMPS) by an iMPS with a smaller bond dimension.

    - The MPO-MPS state to be truncated is provided in `toapprox`, and the truncated state is initialized by the input `state`. The iMPO and iMPS can be periodic in the vertical direction (using `MPOMultiline` and `MPSMultiline`).
    - The truncation algorithm can be chosen among `VUMPS`, `IDMRG1`, `IDMRG2`. 
"""
function approximate end

function approximate(state::InfiniteMPS, toapprox::Tuple{<:Union{SparseMPO,DenseMPO},<:InfiniteMPS}, alg, envs = environments(state,toapprox))
    # PeriodicMPO's always act on MPSMultiline's. I therefore convert the imps to multilines, approximate and convert back
    (multi, envs) = approximate(convert(MPSMultiline, state), (convert(MPOMultiline, envs.opp),convert(MPSMultiline, envs.above)), alg, envs)
    state = convert(InfiniteMPS, multi)
    return (state, envs)
end
function approximate(state::MPSMultiline, toapprox::Tuple{<:MPOMultiline,<:MPSMultiline}, alg::VUMPS, envs = environments(state,toapprox))
    (mpo,above) = toapprox;

    galerkin  = 1+alg.tol_galerkin
    iter       = 1

    temp_ACs = similar.(state.AC);
    temp_Cs = similar.(state.CR);

    while true

        @sync for col in 1:size(state,2)
            @Threads.spawn $temp_ACs[:,col] = circshift([ac_proj(row,$col,$state,$envs) for row in 1:size($state,1)],1)
            @Threads.spawn $temp_Cs[:,col]  = circshift([c_proj(row,$col,$state,$envs) for row in 1:size($state,1)],1)
        end

        for row in 1:size(state,1),col in 1:size(state,2)
            QAc,_ = leftorth!(temp_ACs[row,col], alg=TensorKit.QRpos())
            Qc,_  = leftorth!(temp_Cs[row,col], alg=TensorKit.QRpos())
            temp_ACs[row,col] = QAc*adjoint(Qc)
        end

        state = MPSMultiline(temp_ACs,state.CR[:,end]; tol = alg.tol_gauge, maxiter = alg.orthmaxiter);
        recalculate!(envs,state);

        (state,envs) = alg.finalize(iter,state,toapprox,envs) :: Tuple{typeof(state),typeof(envs)};

        galerkin   = calc_galerkin(state, envs)
        alg.verbose && @info "vomps @iteration $(iter) galerkin = $(galerkin)"

        if (galerkin <= alg.tol_galerkin) || iter>=alg.maxiter
            iter>=alg.maxiter && @warn "vomps didn't converge $(galerkin)"
            return state, envs, galerkin
        end

        iter += 1
    end
end
