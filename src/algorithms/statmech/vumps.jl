#the statmech VUMPS
#it made sense to seperate both vumpses as
# - leading_boundary primarily works on MPSMultiline
# - they search for different eigenvalues
# - ham vumps should use Lanczos, this has to use arnoldi
# - this vumps updates entire collumns (state[:,i]); incompatible with InfiniteMPS
# - (a)c-prime takes a different number of arguments
# - it's very litle duplicate code, but together it'd be a bit more convoluted (primarily because of the indexing way)

"
    leading_boundary(state,opp,alg,envs=environments(state,ham))

    approximate the leading eigenvector for opp
"
function leading_boundary(state::InfiniteMPS, H, alg, envs=environments(state, H))
    (st, pr, de) = leading_boundary(convert(MPSMultiline, state), Multiline([H]), alg, envs)
    return convert(InfiniteMPS, st), pr, de
end

function leading_boundary(state::MPSMultiline, H, alg::VUMPS, envs=environments(state, H))
    galerkin = 1 + alg.tol_galerkin
    iter = 1

    temp_ACs = similar.(state.AC)
    temp_Cs = similar.(state.CR)

    while true
        eigalg = Arnoldi(; tol=alg.tol_galerkin / 10)

        @sync for col in 1:size(state, 2)
            Threads.@spawn begin
                H_AC = ∂∂AC($col, $state, $H, $envs)

                (vals_ac, vecs_ac) = eigsolve(
                    H_AC, RecursiveVec($state.AC[:, col]), 1, :LM, eigalg
                )
                $temp_ACs[:, col] = vecs_ac[1].vecs[:]
            end

            Threads.@spawn begin
                H_C = ∂∂C($col, $state, $H, $envs)

                (vals_c, vecs_c) = eigsolve(
                    H_C, RecursiveVec($state.CR[:, col]), 1, :LM, eigalg
                )
                $temp_Cs[:, col] = vecs_c[1].vecs[:]
            end
        end

        for row in 1:size(state, 1), col in 1:size(state, 2)
            QAc, _ = leftorth!(temp_ACs[row, col]; alg=TensorKit.QRpos())
            Qc, _ = leftorth!(temp_Cs[row, col]; alg=TensorKit.QRpos())
            temp_ACs[row, col] = QAc * adjoint(Qc)
        end

        state = MPSMultiline(
            temp_ACs, state.CR[:, end]; tol=alg.tol_gauge, maxiter=alg.orthmaxiter
        )
        recalculate!(envs, state)

        (state, envs) =
            alg.finalize(iter, state, H, envs)::Tuple{typeof(state),typeof(envs)}

        galerkin = calc_galerkin(state, envs)
        alg.verbose && @info "vumps @iteration $(iter) galerkin = $(galerkin)"

        if (galerkin <= alg.tol_galerkin) || iter >= alg.maxiter
            iter >= alg.maxiter && @warn "vumps didn't converge $(galerkin)"
            return state, envs, galerkin
        end

        iter += 1
    end
end
