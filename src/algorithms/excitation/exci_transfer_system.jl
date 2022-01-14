
function left_excitation_transfer_system(lBs, ham, exci; mom=exci.momentum, solver=Defaults.linearsolver)
    len = ham.period
    found = zero.(lBs)
    ids = collect(Iterators.filter(x->isid(ham,x),1:ham.odim));

    for i in 1:ham.odim


        #this operation can be sped up by at least a factor 2;  found mostly consists of zeros
        start = found
        for k in 1:len
            start = start*TransferMatrix(exci.right_gs.AR[k],ham[k],exci.left_gs.AL[k])*exp(conj(1im*mom))

            exci.trivial && for l in ids[2:end-1]
                @plansor start[l][-1 -2;-3 -4]-=start[l][1 4;-3 2]*r_RL(exci.right_gs,k)[2;3]*τ[3 4;5 1]*l_RL(exci.right_gs,k+1)[-1;6]*τ[5 6;-4 -2]
            end
        end

        #either the element i,i exists; in which case we have to solve a linear system
        #otherwise it's easy and we already know found[i]
        if reduce((a,b)->a&&contains(ham[b],i,i),1:len,init=true)
            tm = TransferMatrix(exci.right_gs.AR,[ham[b][i,i] for b in 1:len],exci.left_gs.AL)

            (found[i],convhist) = linsolve(lBs[i]+start[i],lBs[i]+start[i],solver,1,-exp(-1im*mom)) do y
                x = y*tm

                if exci.trivial && i in ids
                    @plansor x[-1 -2;-3 -4] -= x[3 4;-3 5]*r_RL(exci.left_gs)[5;2]*τ[2 4;6 3]*l_RL(exci.left_gs)[-1;1]*τ[6 1;-4 -2]
                end

                return x
            end
            convhist.converged<1 && @info "left $(i) excitation inversion failed normres $(convhist.normres)"

        else
            found[i]=lBs[i]+start[i]
        end
    end
    return found
end

function right_excitation_transfer_system(rBs, ham, exci; mom=exci.momentum, solver=Defaults.linearsolver)
    len = ham.period
    found = zero.(rBs)
    ids = collect(Iterators.filter(x->isid(ham,x),1:ham.odim));
    for i in ham.odim:-1:1

        start = found
        for k in len:-1:1
            start = TransferMatrix(exci.left_gs.AL[k],ham[k],exci.right_gs.AR[k])*start*exp(1im*mom)

            exci.trivial && for l in ids[2:end-1]
                @plansor start[l][-1 -2;-3 -4] -= τ[6 2;3 4]*start[l][3 4;-3 5]*l_LR(exci.right_gs,k)[5;2]*r_LR(exci.right_gs,k-1)[-1;1]*τ[-2 -4;1 6]
            end

        end

        if reduce((a,b)->a&&contains(ham[b],i,i),1:len,init=true)
            tm = TransferMatrix(exci.left_gs.AL,[ham[b][i,i] for b in 1:len],exci.right_gs.AR);
            (found[i],convhist) = linsolve(rBs[i]+start[i],rBs[i]+start[i],solver,1,-exp(1im*mom)) do y
                x = tm*y
                
                if exci.trivial && i in ids
                    @plansor x[-1 -2;-3 -4] -= τ[6 2;3 4]*x[3 4;-3 5]*l_LR(exci.right_gs)[5;2]*r_LR(exci.right_gs)[-1;1]*τ[-2 -4;1 6]
                end

                return y-x
            end
            convhist.converged<1 && @info "right $(i) excitation inversion failed normres $(convhist.normres)"

        else
            found[i]=rBs[i]+start[i]
        end
    end
    return found
end
