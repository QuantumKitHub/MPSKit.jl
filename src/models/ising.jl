function nonsym_ising_ham(;J = -1,spin = 1//2,lambda = 0.5,longit=0.0)
    (sx,sy,sz)=nonsym_spintensors(spin);
    id = one(sx);

    @tensor ham[-1 -2;-3 -4]:=(J*sz)[-1,-3]*sz[-2,-4]+(0.5*lambda*id)[-1,-3]*sx[-2,-4]+(0.5*lambda*sx)[-1,-3]*id[-2,-4]+(0.5*longit*id)[-1,-3]*sz[-2,-4]+(0.5*longit*sz)[-1,-3]*id[-2,-4]
    ham = MpoHamiltonian(ham);

    return ham
end

function nonsym_ising_mpo(;beta = log(1+sqrt(2))/2)
    t = [exp(beta) exp(-beta); exp(-beta) exp(beta)];
    r = eigen(t);t = r.vectors*sqrt(Diagonal(r.values))*r.vectors;

    O = zeros(2,2,2,2);
    O[1,1,1,1]=1; O[2,2,2,2]=1;

    @tensor toret[-1 -2;-3 -4] := O[1,2,3,4]*t[-1,1]*t[-2,2]*t[-3,3]*t[-4,4];

    torett = TensorMap(convert(Array{Defaults.eltype,4},toret),ComplexSpace(2)*ComplexSpace(2),ComplexSpace(2)*ComplexSpace(2));
    return PeriodicMpo(torett);
end
