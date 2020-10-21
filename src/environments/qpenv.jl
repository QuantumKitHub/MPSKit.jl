#=
nothing fancy - only used internally (and therefore cryptic) - stores some partially contracted things
seperates out this bit of logic from effective_excitation_hamiltonian (now more readable)
can also - potentially - partially reuse this in other algorithms
=#
struct QPEnv{A,B} <: Cache
    lBs::Array{A,1}
    rBs::Array{A,1}

    lpars::B
    rpars::B
end

function params(exci::InfiniteQP,ham::MPOHamiltonian,lpars=params(exci.left_gs,ham),rpars=exci.trivial ? lpars : params(exci.right_gs,ham))
    ids = collect(Iterators.filter(x->isid(ham,x),2:ham.odim-1));

    #build lBs(c)
    lB_cur = [ TensorMap(zeros,eltype(exci),
                    virtualspace(exci.left_gs,0)*ham.domspaces[1,k]',
                    space(exci[1],3)'*virtualspace(exci.left_gs,0)) for k in 1:ham.odim]
    lBs = typeof(lB_cur)[]

    for pos = 1:length(exci)
        lB_cur = exci_transfer_left(lB_cur,ham,pos,exci.right_gs.AR[pos],exci.left_gs.AL[pos])*exp(conj(1im*exci.momentum))
        lB_cur += exci_transfer_left(leftenv(lpars,pos,exci.left_gs),ham,pos,exci[pos],exci.left_gs.AL[pos])*exp(conj(1im*exci.momentum))

        exci.trivial && for i in ids
            @tensor lB_cur[i][-1,-2,-3,-4] -= lB_cur[i][1,-2,-3,2]*r_RL(exci.left_gs,pos)[2,1]*l_RL(exci.left_gs,pos+1)[-1,-4]
        end


        push!(lBs,lB_cur)
    end

    #build rBs(c)
    rB_cur = [ TensorMap(zeros,eltype(exci),
                    virtualspace(exci.right_gs,length(exci))*space(exci[1],3),
                    ham.imspaces[length(exci),k]*virtualspace(exci.right_gs,length(exci))) for k in 1:ham.odim]
    rBs = typeof(rB_cur)[]

    for pos=length(exci):-1:1
        rB_cur = exci_transfer_right(rB_cur,ham,pos,exci.left_gs.AL[pos],exci.right_gs.AR[pos])*exp(1im*exci.momentum)
        rB_cur += exci_transfer_right(rightenv(rpars,pos,exci.right_gs),ham,pos,exci[pos],exci.right_gs.AR[pos])*exp(1im*exci.momentum)

        exci.trivial && for i in ids
            @tensor rB_cur[i][-1,-2,-3,-4] -= rB_cur[i][1,-2,-3,2]*l_LR(exci.left_gs,pos)[2,1]*r_LR(exci.left_gs,pos-1)[-1,-4]
        end

        push!(rBs,rB_cur)
    end
    rBs = reverse(rBs)

    lBE = left_excitation_transfer_system(lB_cur,ham,exci)
    rBE = right_excitation_transfer_system(rB_cur,ham,exci)

    lBs[end] = lBE;

    for i=1:length(exci)-1
        lBE = exci_transfer_left(lBE,ham,i,exci.right_gs.AR[i],exci.left_gs.AL[i])*exp(conj(1im*exci.momentum))

        exci.trivial && for k in ids
            @tensor lBE[k][-1,-2,-3,-4] -= lBE[k][1,-2,-3,2]*r_RL(exci.left_gs,i)[2,1]*l_RL(exci.left_gs,i+1)[-1,-4]
        end

        lBs[i] += lBE;
    end

    rBs[1] = rBE;

    for i=length(exci):-1:2
        rBE = exci_transfer_right(rBE,ham,i,exci.left_gs.AL[i],exci.right_gs.AR[i])*exp(1im*exci.momentum)

        exci.trivial && for k in ids
            @tensor rBE[k][-1,-2,-3,-4]-=rBE[k][1,-2,-3,2]*l_LR(exci.left_gs,i)[2,1]*r_LR(exci.left_gs,i-1)[-1,-4]
        end

        rBs[i] += rBE
    end

    return QPEnv(lBs,rBs,lpars,rpars)
end

function params(exci::FiniteQP,ham::MPOHamiltonian,lpars=params(exci.left_gs,ham),rpars=exci.trivial ? lpars : params(exci.right_gs,ham))
    #construct lBE
    lB_cur = [ TensorMap(zeros,eltype(exci),
                    virtualspace(exci.left_gs,0)*ham.domspaces[1,k]',
                    space(exci[1],3)'*virtualspace(exci.left_gs,0)) for k in 1:ham.odim]
    lBs = typeof(lB_cur)[]
    for pos = 1:length(exci)
        lB_cur = exci_transfer_left(lB_cur,ham,pos,exci.right_gs.AR[pos],exci.left_gs.AL[pos])
        lB_cur += exci_transfer_left(leftenv(lpars,pos,exci.left_gs),ham,pos,exci[pos],exci.left_gs.AL[pos])
        push!(lBs,lB_cur)
    end

    #build rBs(c)
    rB_cur = [ TensorMap(zeros,eltype(exci),
                    virtualspace(exci.right_gs,length(exci))*space(exci[1],3),
                    ham.imspaces[length(exci),k]*virtualspace(exci.right_gs,length(exci))) for k in 1:ham.odim]
    rBs = typeof(rB_cur)[]
    for pos=length(exci):-1:1
        rB_cur = exci_transfer_right(rB_cur,ham,pos,exci.left_gs.AL[pos],exci.right_gs.AR[pos])
        rB_cur += exci_transfer_right(rightenv(rpars,pos,exci.right_gs),ham,pos,exci[pos],exci.right_gs.AR[pos])
        push!(rBs,rB_cur)
    end
    rBs=reverse(rBs)

    return QPEnv(lBs,rBs,lpars,rpars)
end
