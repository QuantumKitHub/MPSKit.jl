"calculates the entropy of a given state"
entropy(state::InfiniteMPS) = [sum([-j^2*2*log(j) for j in diag(convert(Array,tsvd(state.CR[i])[2]))]) for i in 1:length(state)]

"
given a thermal state, you can map it to an mps by fusing the physical legs together
to prepare a gibbs ensemble, you need to evolve this state with H working on both legs
here we return the 'superhamiltonian' (H*id,id*H)
"
function splitham(ham::MPOHamiltonian)
    fusers = [isomorphism(fuse(p*p'),p*p') for p in ham.pspaces]

    idham = Array{Union{Missing,typeof(ham[1,1,1])},3}(missing,ham.period,ham.odim,ham.odim)
    hamid = Array{Union{Missing,typeof(ham[1,1,1])},3}(missing,ham.period,ham.odim,ham.odim)

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

infinite_temperature(ham::MPOHamiltonian) = [permute(isomorphism(Matrix{eltype(ham[1,1,1])},oneunit(sp)*sp,oneunit(sp)*sp),(1,2,4),(3,)) for sp in ham.pspaces]

calc_galerkin(state::Union{InfiniteMPS,FiniteMPS,MPSComoving},loc,pars) = norm(leftnull(state.AC[loc])'*ac_prime(state.AC[loc], loc, state, pars))
"calculates the galerkin error"
calc_galerkin(state::Union{InfiniteMPS,FiniteMPS,MPSComoving}, pars) = maximum([calc_galerkin(state,loc,pars) for loc in 1:length(state)])
calc_galerkin(state::MPSMultiline, pars) = maximum([norm(leftnull(state.AC[row+1,col])'*ac_prime(state.AC[row,col], row,col, state, pars)) for (row,col) in Iterators.product(1:size(state,1),1:size(state,2))][:])


"
allows exact two point functions of operators instead of eigenvalues of TM which gives the leading order infinte range behavior.
"
function twopoint()
	throw("WIP BEN BEZIG AAN IETS VOOR DAAN")
end
