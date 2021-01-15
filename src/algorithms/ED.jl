"""
Use krylovkit to perform exact diagonalization
"""
function exact_diagonalization(opp::MPOHamiltonian,len::Int = opp.period,num::Int = 1,which::Symbol=:LM,alg::KrylovKit.KrylovAlgorithm = Arnoldi())
    pspaces = [opp.pspaces[i] for i in 1:len];

    #construct the largest possible finite mps of that length
    state = FiniteMPS(rand,eltype(eltype(opp)),pspaces,fuse(prod(pspaces)))
    envs = environments(state,opp);

    #optimize the middle site. Because there is no truncation, this single site captures the entire possible hilbert space
    middle_site = Int(round(len/2));
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
