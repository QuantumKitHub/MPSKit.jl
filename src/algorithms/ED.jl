"""
Use krylovkit to perform exact diagonalization
"""
function exact_diagonalization(opp::MPOHamiltonian;sector = first(sectors(oneunit(opp.pspaces[1]))),len::Int = opp.period,num::Int = 1,which::Symbol=:LM,alg::KrylovKit.KrylovAlgorithm = Arnoldi())
    left = RepresentationSpace(sector=>1);
    right = oneunit(left);

    middle_site = Int(round(len/2));

    Ot = eltype(opp);
    tensors = Vector{tensormaptype(spacetype(Ot),2,1,eltype(Ot))}(undef,len);

    for i in 1:middle_site
        tensors[i] = isomorphism(storagetype(Ot),left*opp.pspaces[i],fuse(left*opp.pspaces[i]));
        left = _lastspace(tensors[i])';
    end
    for i in len:-1:middle_site+1
        tensors[i] = _permute_front(isomorphism(storagetype(Ot),fuse(opp.pspaces[i]'*right'),opp.pspaces[i]'*right));
        right = _firstspace(tensors[i]);
    end
    left == right || throw(ArgumentError("invalid sector"));
    
    #construct the largest possible finite mps of that length
    state = FiniteMPS(tensors,normalize=true);
    envs = environments(state,opp);

    #optimize the middle site. Because there is no truncation, this single site captures the entire possible hilbert space

    (vals,vecs,convhist) = eigsolve(state.AC[middle_site],num,which,alg) do x
        ac_prime(x,middle_site,state,envs)
    end

    state_vecs = map(vecs) do v
        cs = copy(state);
        cs.AC[middle_site] = v;
        cs
    end

    return vals,state_vecs,convhist
end
