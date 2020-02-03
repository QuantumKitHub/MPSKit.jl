using MPSKit,TensorKit,Test

let
    #defining the hamiltonian
    th = nonsym_ising_ham(lambda = 4.0);
    szt = TensorMap([1 0;0 -1],ℂ^2,ℂ^2)

    #finite mps (dmrg)
    ts = FiniteMps(fill(ℂ^2,10));
    (ts,pars,_)=find_groundstate(ts,th,Dmrg(manager=SimpleManager(12)));

    szval_finite= sum(expectation_value(ts,szt))/length(ts)
    @test szval_finite ≈ 0 atol=1e-12

    #twosite dmrg
    ts=MPSKit.generatefmps(10,2,10);
    (ts,pars,_)=find_groundstate(ts,th,Dmrg2(trscheme = truncdim(15)));

    szval_finite= sum(expectation_value(ts,szt))/length(ts)
    @test szval_finite ≈ 0 atol=1e-12

    #uniform mps
    ts=MpsCenterGauged([ℂ^2],[ℂ^50]);
    (ts,pars,_)=find_groundstate(ts,th,Vumps(maxiter=400));

    szval_infinite=@tensor ts.AC[1][1,2,3]*szt[4,2]*conj(ts.AC[1][1,4,3])
    @test szval_infinite ≈ 0 atol=1e-12

    #optimkit algorithms (experimental)
    #ts=MpsCenterGauged([ℂ^2],[ℂ^5]);
    #(ts,pars,_)=find_groundstate(ts,th,LBFGS(maxiter=100,verbosity=3));

    #szval_infinite=@tensor ts.AC[1][1,2,3]*szt[4,2]*conj(ts.AC[1][1,4,3])
    #@test szval_infinite ≈ 0 atol=1e-12
end
