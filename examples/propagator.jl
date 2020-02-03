using MPSKit,TensorKit,Test

#let
    ham = nonsym_ising_ham(lambda=4.0);
    gs = FiniteMps([ℂ^2 for i in 1:10]);
    (gs,pars,_) = find_groundstate(gs,ham,Dmrg2(trscheme=truncdim(10)));

    #we are in the groundstate
    #we expect to find a single isolated pole around the gs energy
    polepos = real(sum(expectation_value(gs,ham,pars)));

    vals = (-0.5:0.05:0.5).+polepos
    eta = 0.3im;

    predicted = [1/(v+eta-polepos) for v in vals];

    data = similar(predicted);
    for (i,v) in enumerate(vals)
	global data
        (data[i],_) = dynamicaldmrg(gs,v+eta,ham,verbose=false)
    end

    @test norm(data-predicted) ≈ 0 atol=1e-8
#end
