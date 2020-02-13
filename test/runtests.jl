using MPSKit,TensorKit,LinearAlgebra,Test

@testset "States" begin
    @testset "MpsCenterGauged ($D,$d,$elt)" for (D,d,elt) in [
            (ComplexSpace(10),ComplexSpace(2),ComplexF64),
            (ℂ[SU₂](1=>1,0=>3),ℂ[SU₂](0=>1),ComplexF32)
            ]
        tol = Float64(eps(real(elt))*100);

        #=@inferred=# leftorth([TensorMap(rand,elt,D*d,D)],tol=tol);
        #=@inferred=# rightorth([TensorMap(rand,elt,D*d,D)],tol=tol);

        ts = #=@inferred=# MpsCenterGauged([TensorMap(rand,elt,D*d,D),TensorMap(rand,elt,D*d,D)],tol = tol);

        @tensor difference[-1,-2,-3] := ts.AL[1][-1,-2,1]*ts.CR[1][1,-3]-ts.CR[0][-1,1]*ts.AR[1][1,-2,-3];
        @test norm(difference,Inf)<tol;
    end

    @testset "FiniteMps ($D,$d,$elt)" for (D,d,elt) in [
            (ComplexSpace(10),ComplexSpace(2),ComplexF64),
            (ℂ[SU₂](1=>1,0=>3),ℂ[SU₂](0=>1),ComplexF32)
            ]

        data = [TensorMap(rand,elt,oneunit(D)*d,D)]
        for i in 1:3
            push!(data,TensorMap(rand,elt,D*d,D))
        end
        push!(data,TensorMap(rand,elt,D*d,oneunit(D)));

        ts = FiniteMps(data);

        ovl = dot(ts,ts);
        ts = rightorth(ts,renorm=false);
        @test ovl ≈ norm(ts[1])^2

        data2 = [TensorMap(rand,elt,oneunit(D)*d,D)]
        for i in 1:3
            push!(data2,TensorMap(rand,elt,D*d,D))
        end
        push!(data2,TensorMap(rand,elt,D*d,oneunit(D)));

        ts2 = FiniteMps(data2);

        ovl2 = dot(ts,ts2);

        ts3 = ts+ts2;

        @test ovl2+ovl ≈ dot(ts,ts3)
    end

    @testset "MpsComoving" begin
        ham = nonsym_ising_ham(lambda=4.0);
        (gs,_,_) = find_groundstate(MpsCenterGauged([ℂ^2],[ℂ^10]),ham,Vumps(verbose=false));

        window = MpsComoving(gs,[gs.AC[1];[gs.AR[i] for i in 2:10]],gs);

        e1 = expectation_value(window,ham);

        (window,pars,_) = find_groundstate(window,ham,Dmrg(verbose=false));

        e2 = expectation_value(window,ham);

        @test e1[1] ≈ e2[1]
        @test e1[2] ≈ e2[2]

        (window,pars) = timestep(window,ham,0.1,Tdvp2(),pars)
        (window,pars) = timestep(window,ham,0.1,Tdvp(),pars)

        e3 = expectation_value(window,ham);

        @test e1[1] ≈ e3[1]
        @test e1[2] ≈ e3[2]
    end
end

@testset "Operators" begin

    @testset "mpoham $(i)" for (i,(th,Dspaces)) in enumerate([
            (nonsym_ising_ham(),[ℂ^1]),
            (u1_xxz_ham(),[ℂ[U₁](1//2=>1)]),
            (repeat(su2_xxx_ham(),2),[ℂ[SU₂](0=>1),ℂ[SU₂](1//2=>1)])
            ])

        ts = MpsCenterGauged(th.pspaces,Dspaces); # generate a product state

        (ts,_) = changebonds(ts,th,OptimalExpand()) # optimal expand a la vumps paper
        ndim = dim(space(ts.AC[1],1))
        (ts,_) = changebonds(ts,th,VumpsSvdCut()) # idmrg2 step to expand the bond dimension
        @test dim(space(ts.AC[1],1)) > ndim;

        e1 = expectation_value(ts,th);

        e2 = expectation_value(ts,2*th); #multiplication with a constant
        @test 2*e1≈e2;

        e2 = expectation_value(ts,0.5*th+th); #addition
        @test 1.5*e1≈e2;

        th -= expectation_value(ts,th);
        v = expectation_value(ts,th*th);
        @test real(v[1])>=0;
    end

    @testset "comact $(i)" for (i,th) in enumerate([
            nonsym_ising_ham(),
            u1_xxz_ham(),
            su2_xxx_ham()
            ])

        len = 20;

        ts = FiniteMpo([TensorMap(rand,ComplexF64,
                    oneunit(th.pspaces[1]) * th.pspaces[j],
                    oneunit(th.pspaces[1]) * th.pspaces[j])
                    for j in 1:len]);

        (ts,_) = changebonds(ts,commutator(th),RandExpand());
        (ts,_) = changebonds(ts,anticommutator(th),OptimalExpand());

        e1 = expectation_value(ts,anticommutator(th));
        e2 = expectation_value(ts,2*anticommutator(th));

        @test 2*e1≈e2;

        e3 = expectation_value(ts,anticommutator(th)+commutator(th));
        e4 = expectation_value(ts,anticommutator(th)-commutator(th));

        @test e3+e4≈e2;

        diff = [rand() for i in th.pspaces];
        e5 = expectation_value(ts,anticommutator(th)-diff);
        @test sum([e1[j]-diff[mod1(j,end)] for j in 1:len])≈sum(e5);
    end

end

@testset "Algorithms" begin
    @testset "find_groundstate $(ind)" for (ind,(state,alg,ham)) in enumerate([
            (MpsCenterGauged([ℂ^2],[ℂ^10]),Vumps(tol_galerkin=1e-8,verbose=false),nonsym_ising_ham(lambda=2.0)),
            (MpsCenterGauged([ℂ^2],[ℂ^10]),Idmrg1(tol_galerkin=1e-8,maxiter=400,verbose=false),nonsym_ising_ham(lambda=2.0)),
            (FiniteMps(fill(TensorMap(rand,ComplexF64,ℂ^1*ℂ^2,ℂ^1),10)),Dmrg2(verbose=false),nonsym_ising_ham(lambda=2.0)),
            (FiniteMps(fill(TensorMap(rand,ComplexF64,ℂ^1*ℂ^2,ℂ^1),10)),Dmrg(manager=SimpleManager(10),verbose=false),nonsym_ising_ham(lambda=2.0))
            ])

        #vumps type inferrence got broken by @threads, so worth it?
        (ts,pars,delta) =  #=@inferred=# find_groundstate(state,ham,alg)

        @test sum(delta) < 1e-6

        evals = expectation_value(ts,ham);
        th = repeat(ham,length(evals))-evals

        @test real(sum(expectation_value(ts,th*th))) < 1e-2 #is the ground state variance relatively low?
    end

    @testset "timestep $(ind)" for (ind,(state,alg,opp)) in enumerate([
        (FiniteMpo(fill(TensorMap(rand,ComplexF64,ℂ^1*ℂ^2,ℂ^1*ℂ^2),5)),Tdvp(),commutator(nonsym_xxz_ham(spin=1//2))),
        (FiniteMpo(fill(TensorMap(rand,ComplexF64,ℂ^1*ℂ^2,ℂ^1*ℂ^2),7)),Tdvp2(),anticommutator(nonsym_xxz_ham(spin=1//2))),
        (MpsCenterGauged([ℂ^3,ℂ^3],[ℂ^50,ℂ^50]),Tdvp(),repeat(nonsym_xxz_ham(spin=1),2))
        ])

        edens = expectation_value(state,opp);

        (state,_) = timestep(state,opp,rand(),alg)

        @test sum(expectation_value(state,opp)) ≈ sum(edens) atol = 1e-3
    end

    @testset "leading_boundary $(ind)" for (ind,alg) in enumerate([
            Vumps(tol_galerkin=1e-10,verbose=false),
            PowerMethod(tol_galerkin=1e-10,verbose=false,maxiter=1000)])

        mpo = #=@inferred=# nonsym_ising_mpo();
        state = MpsCenterGauged([ℂ^2],[ℂ^10]);
        (state,pars,_) = leading_boundary(state,mpo,alg);

        @test expectation_value(state,mpo,pars)[1,1] ≈ 2.5337 atol=1e-3
    end


    @testset "quasiparticle_excitation" begin
        th = nonsym_xxz_ham()
        ts = MpsCenterGauged([ℂ^3],[ℂ^48]);
        (ts,pars,_) = find_groundstate(ts,th,Vumps(maxiter=400,verbose=false));
        (energies,Bs) = quasiparticle_excitation(th,Float64(pi),ts,pars);
        @test energies[1] ≈ 0.41047925 atol=1e-4
    end

    @testset "dynamicaldmrg" begin
        ham = nonsym_ising_ham(lambda=4.0);
        (gs,_,_) = find_groundstate(MpsCenterGauged([ℂ^2],[ℂ^10]),ham,Vumps(verbose=false));
        window = MpsComoving(gs,[gs.AC[1];[gs.AR[i] for i in 2:10]],gs);

        szd = TensorMap([1 0;0 -1],ℂ^2,ℂ^2);
        @test expectation_value(gs,szd)[1] ≈ expectation_value(window,szd)[1] atol=1e-10

        polepos = expectation_value(gs,ham,10)
        @test polepos ≈ expectation_value(window,ham)[2]

        vals = (-0.5:0.2:0.5).+polepos
        eta = 0.3im;

        predicted = [1/(v+eta-polepos) for v in vals];

        data = similar(predicted);
        for (i,v) in enumerate(vals)
            (data[i],_) = dynamicaldmrg(window,v+eta,ham,verbose=false)
        end

        @test data ≈ predicted atol=1e-8
    end

end
