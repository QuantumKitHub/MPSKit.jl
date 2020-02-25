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

@bm function leading_boundary(state::MPSMultiline, H,alg::Vumps,pars = params(state,H))
    galerkin  = 1+alg.tol_galerkin
    iter       = 1

    newAs = similar(state.AL)

    lee = galerkin

    while true
        eigalg = Arnoldi(tol=alg.tol_galerkin/10)

        for col in 1:size(state,2)

            (e,vac,ch)=let state=state,pars=pars,eigalg=eigalg
                eigsolve(RecursiveVec(state.AC[:,col]), 1, :LM, eigalg) do x
                    RecursiveVec(circshift([ac_prime(x[row], row,col, state, pars) for row in 1:length(x)],1))
                end
            end

            (e,vc,ch) = let state=state,pars=pars,eigalg=eigalg
                eigsolve(RecursiveVec(state.CR[:,col]), 1, :LM, eigalg) do x
                    RecursiveVec(circshift([c_prime(x[row], row,col, state, pars) for row in 1:length(x)],1))
                end
            end

            for row in 1:size(state,1)
                QAc,_ = TensorKit.leftorth!(vac[1][row], alg=TensorKit.Polar())
                Qc,_  = TensorKit.leftorth!(vc[1][row], alg=TensorKit.Polar())
                newAs[row,col] = QAc*adjoint(Qc)
            end

        end

        state = MPSMultiline(newAs; tol = alg.tol_gauge, maxiter = alg.maxiter)
        galerkin = calc_galerkin(state, pars)
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

calc_galerkin(state::MPSMultiline, pars) = maximum([norm(leftnull(state.AC[row+1,col])'*ac_prime(state.AC[row,col], row,col, state, pars)) for (row,col) in Iterators.product(1:size(state,1),1:size(state,2))][:])
