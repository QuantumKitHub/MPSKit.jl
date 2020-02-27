"calculates the entropy of a given state"
entropy(state::InfiniteMPS) = [sum([-j^2*2*log(j) for j in diag(convert(Array,tsvd(state.CR[i])[2]))]) for i in 1:length(state)]

"
given a thermal state, you can map it to an mps by fusing the physical legs together
to prepare a gibbs ensemble, you need to evolve this state with H working on both legs
here we return the 'superhamiltonian' (H*id,id*H)
"
function splitham(ham::MPOHamiltonian)
    fusers = [isomorphism(fuse(p*p'),p*p') for p in ham.pspaces]

    idham = Array{Union{Missing,eltype(ham.Os[1])},3}(missing,ham.period,ham.odim,ham.odim)
    hamid = Array{Union{Missing,eltype(ham.Os[1])},3}(missing,ham.period,ham.odim,ham.odim)

    for i in 1:ham.period
        for (k,l) in keys(ham,i)
            idt = isomorphism(ham.pspaces[i],ham.pspaces[i])

            @tensor hamid[i,k,l][-1 -2; -3 -4] :=   ham[i,k,l][-1,12,-3,15]*
                                                    idt[16,13]*
                                                    fusers[i][-2,12,13]*
                                                    conj(fusers[i][-4,15,16])

            @tensor idham[i,k,l][-1 -2; -3 -4] :=   ham[i,k,l][-1,13,-3,16]*
                                                    idt[12,15]*
                                                    fusers[i][-2,12,16]*
                                                    conj(fusers[i][-4,15,13])
        end
    end

    return MPOHamiltonian(hamid),MPOHamiltonian(idham)
end

mpo2mps(mpo::FiniteMPO) = FiniteMPS(mpo2mps.(mpo))
function mpo2mps(mpo::TensorMap)
    mpo = permute(mpo,(2,4),(1,3))
    mpo = TensorMap(mpo.data,fuse(codomain(mpo)),domain(mpo))
    mpo = permute(mpo,(2,1),(3,));
    return mpo
end

mps2mpo(mps::FiniteMPS,ospace::AbstractArray) = FiniteMPO([mps2mpo(mps[i],ospace[i]) for i in 1:length(mps)])
function mps2mpo(mps::TensorMap,ospace::VectorSpace)
    mps=permute(mps,(1,2),(3,))
    mpo=TensorMap(mps.data,space(mps,1)*ospace*ospace',space(mps,3)')
    return permute(mpo,(1,2,),(4,3))
end

infinite_temperature(ham::MPOHamiltonian) = [isomorphism(Matrix{eltype(ham[1,1,1])},oneunit(sp)*sp,oneunit(sp)*sp) for sp in ham.pspaces]
