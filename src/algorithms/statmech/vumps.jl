#the statmech VUMPS
#it made sense to seperate both vumpses as
# - leading_boundary primarily works on MPSMultiline
# - they search for different eigenvalues
# - ham vumps should use Lanczos, this has to use arnoldi
# - this vumps updates entire collumns (Ψ[:,i]); incompatible with InfiniteMPS
# - (a)c-prime takes a different number of arguments
# - it's very litle duplicate code, but together it'd be a bit more convoluted (primarily because of the indexing way)

"""
    leading_boundary(Ψ,opp,alg,envs=environments(Ψ,ham))

Approximate the leading eigenvector for opp.
"""
function leading_boundary(Ψ::InfiniteMPS, H, alg, envs=environments(Ψ, H))
    (st, pr, de) = leading_boundary(convert(MPSMultiline, Ψ), Multiline([H]), alg, envs)
    return convert(InfiniteMPS, st), pr, de
end

function leading_boundary(Ψ::MPSMultiline, H, alg::VUMPS, envs=environments(Ψ, H))
    galerkin = calc_galerkin(Ψ, envs)
    iter = 1

    temp_ACs = similar.(Ψ.AC)
    temp_Cs = similar.(Ψ.CR)

    while true
        tol_eigs, tol_gauge, tol_envs = updatetols(alg, iter, galerkin)

        eigalg = Arnoldi(; tol=tol_eigs)

        if Defaults.parallelize_sites
            @sync begin
                for col in 1:size(Ψ, 2)
                    Threads.@spawn begin
                        H_AC = ∂∂AC($col, $Ψ, $H, $envs)
                        ac = RecursiveVec($Ψ.AC[:, col])
                        _, acvecs = eigsolve(H_AC, ac, 1, :LM, eigalg)
                        $temp_ACs[:, col] = acvecs[1].vecs[:]
                    end

                    Threads.@spawn begin
                        H_C = ∂∂C($col, $Ψ, $H, $envs)
                        c = RecursiveVec($Ψ.CR[:, col])
                        _, cvecs = eigsolve(H_C, c, 1, :LM, eigalg)
                        $temp_Cs[:, col] = cvecs[1].vecs[:]
                    end
                end
            end
        else
            for col in 1:size(Ψ, 2)
                H_AC = ∂∂AC(col, Ψ, H, envs)
                ac = RecursiveVec(Ψ.AC[:, col])
                _, acvecs = eigsolve(H_AC, ac, 1, :LM, eigalg)
                temp_ACs[:, col] = acvecs[1].vecs[:]

                H_C = ∂∂C(col, Ψ, H, envs)
                c = RecursiveVec(Ψ.CR[:, col])
                _, cvecs = eigsolve(H_C, c, 1, :LM, eigalg)
                temp_Cs[:, col] = cvecs[1].vecs[:]
            end
        end

        for row in 1:size(Ψ, 1), col in 1:size(Ψ, 2)
            QAc, _ = leftorth!(temp_ACs[row, col]; alg=TensorKit.QRpos())
            Qc, _ = leftorth!(temp_Cs[row, col]; alg=TensorKit.QRpos())
            temp_ACs[row, col] = QAc * adjoint(Qc)
        end

        Ψ = MPSMultiline(temp_ACs, Ψ.CR[:, end]; tol=tol_gauge, maxiter=alg.orthmaxiter)
        recalculate!(envs, Ψ; tol=tol_envs)

        (Ψ, envs) = alg.finalize(iter, Ψ, H, envs)::Tuple{typeof(Ψ),typeof(envs)}

        galerkin = calc_galerkin(Ψ, envs)
        alg.verbose && @info "vumps @iteration $(iter) galerkin = $(galerkin)"

        if (galerkin <= alg.tol_galerkin) || iter >= alg.maxiter
            iter >= alg.maxiter && @warn "vumps didn't converge $(galerkin)"
            return Ψ, envs, galerkin
        end

        iter += 1
    end
end
