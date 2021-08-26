using MPSKit,MPSKitModels,TensorKit,Test

let
    ham = nonsym_ising_ham(lambda=4.0);

    gs = FiniteMPS(fill(TensorMap(rand,ComplexF64,ℂ^1*ℂ^2,ℂ^1),10));

    (gs,envs,_) = find_groundstate(gs,ham,Dmrg2(trscheme=truncdim(10)));

    #we are in the groundstate
    #we expect to find a single isolated pole around the gs energy
    polepos = real(sum(expectation_value(gs,ham,envs)));

    vals = (-0.5:0.05:0.5).+polepos
    eta = 0.3im;

    predicted = [1/(v+eta-polepos) for v in vals];

    data = map(vals) do v
        first(dynamicaldmrg(gs,v+eta,ham,verbose=false))
    end

    @test norm(data-predicted) ≈ 0 atol=1e-8
end
