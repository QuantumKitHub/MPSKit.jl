#=
nothing fancy - only used internally (and therefore cryptic) - stores some partially contracted things
seperates out this bit of logic from effective_excitation_hamiltonian (now more readable)
can also - potentially - partially reuse this in other algorithms
=#
struct InfiniteQPEnv{A,B} <: Cache
    lBsc::Array{A,1}
    rBsc::Array{A,1}
    lBsE::A
    rBsE::A

    lpars::B
    rpars::B
end

struct FiniteQPEnv{A,B} <: Cache
    lBsc::Array{A,1}
    rBsc::Array{A,1}

    lpars::B
    rpars::B
end

function params(exci::InfiniteQP,ham::MPOHamiltonian,lpars=params(qp.left_gs,ham),rpars=exci.trivial ? lpars : params(qp.right_gs,ham))
    ids = collect(Iterators.filter(x->isid(ham,x),2:ham.odim-1));

    #build lBs(c)
    lBs = [ TensorMap(zeros,eltype(exci.left_gs),
                    space(lpars.lw[1,k],1)*space(lpars.lw[1,k],2),
                    space(exci[1],3)'*space(rpars.lw[1,k],3)') for k in 1:ham.odim]
    lBsc = typeof(lBs)[]

    for pos = 1:length(exci)
        lBs = exci_transfer_left(lBs,ham,pos,exci.right_gs.AR[pos],exci.left_gs.AL[pos])*exp(conj(1im*exci.momentum))
        lBs += exci_transfer_left(leftenv(lpars,pos,exci.left_gs),ham,pos,exci[pos],exci.left_gs.AL[pos])*exp(conj(1im*exci.momentum))

        exci.trivial && for i in ids
            @tensor lBs[i][-1,-2,-3,-4] -= lBs[i][1,-2,-3,2]*r_RL(exci.left_gs,pos)[2,1]*l_RL(exci.left_gs,pos)[-1,-4]
        end


        push!(lBsc,lBs)
    end

    #build rBs(c)
    rBs = [ TensorMap(zeros,eltype(exci.left_gs),
                    space(lpars.rw[end,k],1)*space(exci[1],3),
                    space(rpars.rw[end,k],2)'*space(rpars.rw[end,k],3)') for k in 1:ham.odim]
    rBsc = typeof(rBs)[]

    for pos=length(exci):-1:1
        rBs = exci_transfer_right(rBs,ham,pos,exci.left_gs.AL[pos],exci.right_gs.AR[pos])*exp(1im*exci.momentum)
        rBs += exci_transfer_right(rightenv(rpars,pos,exci.right_gs),ham,pos,exci[pos],exci.right_gs.AR[pos])*exp(1im*exci.momentum)

        exci.trivial && for i in ids
            @tensor rBs[i][-1,-2,-3,-4] -= rBs[i][1,-2,-3,2]*l_LR(exci.left_gs,pos)[2,1]*r_LR(exci.left_gs,pos)[-1,-4]
        end

        push!(rBsc,rBs)
    end
    rBsc = reverse(rBsc)

    lBsE = left_excitation_transfer_system(lBs,ham,exci)
    rBsE = right_excitation_transfer_system(rBs,ham,exci)

    return InfiniteQPEnv(lBsc,rBsc,lBsE,rBsE,lpars,rpars)
end

function params(exci::FiniteQP,ham::MPOHamiltonian,lpars=params(qp.left_gs,ham),rpars=exci.trivial ? lpars : params(qp.right_gs,ham))
    #construct lBsE
    lBs = [ TensorMap(zeros,eltype(exci),
                    space(leftenv(lpars,1,exci.left_gs)[k],1)*space(leftenv(lpars,1,exci.left_gs)[k],2),
                    space(exci[1],3)'*space(leftenv(lpars,1,exci.left_gs)[k],3)') for k in 1:ham.odim]
    lBsc = typeof(lBs)[]
    for pos = 1:length(exci)
        lBs = exci_transfer_left(lBs,ham,pos,exci.right_gs.AR[pos],exci.left_gs.AL[pos])
        lBs += exci_transfer_left(leftenv(lpars,pos,exci.left_gs),ham,pos,exci[pos],exci.left_gs.AL[pos])
        push!(lBsc,lBs)
    end

    #build rBs(c)
    rBs = [ TensorMap(zeros,eltype(exci),
                    space(rightenv(lpars,length(exci.left_gs),exci.left_gs)[k],1)*space(exci[1],3),
                    space(rightenv(lpars,length(exci.left_gs),exci.left_gs)[k],2)'
                    *space(rightenv(lpars,length(exci.left_gs),exci.left_gs)[k],3)') for k in 1:ham.odim]
    rBsc = typeof(rBs)[]
    for pos=length(exci):-1:1
        rBs = exci_transfer_right(rBs,ham,pos,exci.left_gs.AL[pos],exci.right_gs.AR[pos])
        rBs += exci_transfer_right(rightenv(rpars,pos,exci.right_gs),ham,pos,exci[pos],exci.right_gs.AR[pos])
        push!(rBsc,rBs)
    end
    rBsc=reverse(rBsc)

    return FiniteQPEnv(lBsc,rBsc,lpars,rpars)
end
