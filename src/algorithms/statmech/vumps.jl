#the statmech Vumps
#it made sense to seperate both vumpses as
# - leading_boundary primarily works on MPSMultiline
# - they search for different eigenvalues
# - ham vumps should use Lanczos, this has to use arnoldi
# - this vumps updates entire collumns (state[:,i]); incompatible with InfiniteMPS
# - (a)c-prime takes a different number of arguments
# - it's very litle duplicate code, but together it'd be a bit more convoluted (primarily because of the indexing way)

"
    leading_boundary(state,opp,alg,pars=params(state,ham))

    approximate the leading eigenvector for opp
"
function leading_boundary(state::InfiniteMPS,H,alg,pars=params(state,H))
    (st,pr,de) = leading_boundary(convert(MPSMultiline,state),H,alg,pars)
    return convert(InfiniteMPS,st),pr,de
end

function leading_boundary(state::MPSMultiline, H,alg::Vumps,pars = params(state,H))
    galerkin  = 1+alg.tol_galerkin
    iter       = 1

    newAs = similar(state.AL)

    while true
        eigalg = Arnoldi(tol=alg.tol_galerkin/10)

        acjobs = map(1:size(state,2)) do col
            @Threads.spawn eigsolve(RecursiveVec(state.AC[:,col]), 1, :LM, eigalg) do x
                tasks = map(1:length(x)) do row
                    @Threads.spawn ac_prime(x[row], row,col, state, pars)
                end

                RecursiveVec(circshift(fetch.(tasks),1))
            end
        end

        cjobs = map(1:size(state,2)) do col
            @Threads.spawn eigsolve(RecursiveVec(state.CR[:,col]), 1, :LM, eigalg) do x
                tasks = map(1:length(x)) do row
                    @Threads.spawn c_prime(x[row], row,col, state, pars)
                end

                RecursiveVec(circshift(fetch.(tasks),1))
            end
        end

        for col in 1:size(state,2)
            (e,vac,ch) = fetch(acjobs[col])
            (e,vc,ch) = fetch(cjobs[col])

            for row in 1:size(state,1)
                QAc,_ = TensorKit.leftorth!(vac[1][row], alg=TensorKit.QRpos())
                Qc,_  = TensorKit.leftorth!(vc[1][row], alg=TensorKit.QRpos())
                newAs[row,col] = QAc*adjoint(Qc)
            end

        end

        state = MPSMultiline(newAs; leftgauged=true,tol = alg.tol_gauge, maxiter = alg.orthmaxiter)
        galerkin = calc_galerkin(state, pars)
        alg.verbose && @info "vumps @iteration $(iter) galerkin = $(galerkin)"

        (state,pars,sc) = alg.finalize(iter,state,H,pars);
        if (galerkin <= alg.tol_galerkin && sc) || iter>=alg.maxiter
            iter>=alg.maxiter && @warn "vumps didn't converge $(galerkin)"
            return state, pars, galerkin
        end



        iter += 1
    end
end
