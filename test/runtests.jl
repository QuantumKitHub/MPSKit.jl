using MPSKit,TensorKit,LinearAlgebra,Test

@testset "States" begin
    @testset "MpsCenterGauged" begin
        @testset "($D,$d,$elt)" for (D,d,elt) in [(ComplexSpace(10),ComplexSpace(2),ComplexF64),(ℂ[SU₂](1=>1,0=>3),ℂ[SU₂](0=>1),ComplexF32)]
            tol = Float64(eps(real(elt))*100);

            @inferred leftorth([TensorMap(rand,elt,D*d,D)],tol=tol);
            @inferred rightorth([TensorMap(rand,elt,D*d,D)],tol=tol);

            ts = @inferred MpsCenterGauged([TensorMap(rand,elt,D*d,D),TensorMap(rand,elt,D*d,D)],tol = tol);

            @tensor difference[-1,-2,-3] := ts.AL[1][-1,-2,1]*ts.CR[1][1,-3]-ts.CR[0][-1,1]*ts.AR[1][1,-2,-3];
            @test norm(difference,Inf)<tol;
        end
    end
end

@testset "Operators" begin
    @testset "mpoham" begin
        @testset "$(name)" for name in [:nonsym_ising_ham,:nonsym_xxz_ham,#=:su2_xxx_ham,=#:u1_xxz_ham]

            th = @eval $(name)()
            ts = MpsCenterGauged(th.pspaces); # generate a product state

            @test dim(space(ts.AC[1],1)) == 1;

            (ts,_) = changebonds(ts,th,OptimalExpand()) # optimal expand a la vumps paper

            ndim = dim(space(ts.AC[1],1))
            @test ndim > 1;

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
    end
end

@testset "Algorithms" begin
    @testset "find_groundstate" begin
        #defining the hamiltonian
        (sx,sy,sz,id) = nonsym_spintensors(1//2)
        @tensor ham[-1 -2;-3 -4]:=(-1.5*sz)[-1,-3]*sz[-2,-4]+id[-1,-3]*sx[-2,-4]
        th = @inferred MpoHamiltonian(ham)
        ts = @inferred MpsCenterGauged([TensorMap(rand,ComplexF64,ℂ^50*ℂ^2,ℂ^50)]);

        #vumps type inferrence got broken by @threads, so worth it?
        (ts,pars,delta) =  #=@inferred=# find_groundstate(ts,th,Vumps(tol_galerkin=1e-8,verbose=false))

        @test sum(delta)<1e-8 #we're in trouble when vumps even fails for ising

        th=th-expectation_value(ts,th)

        @inferred expectation_value(ts,th*th)
        @test real(expectation_value(ts,th*th)[1]) < 1e-2 #is the ground state variance relatively low?

        #finite mps
        ts = FiniteMps(th.pspaces[1:10]);
        (ts,pars,_) = #=@inferred=# find_groundstate(ts,th,Dmrg2(verbose=false));
        (ts,pars,_) = #=@inferred=# find_groundstate(ts,th,Dmrg(verbose=false));
        @inferred expectation_value(ts,th)
    end

    @testset "leading_boundary" begin
        mpo = @inferred nonsym_ising_mpo();
        state = MpsCenterGauged([ℂ^2],[ℂ^10]);
        (state,pars,_) = leading_boundary(state,mpo,Vumps(tol_galerkin=1e-10,verbose=false));

        @test expectation_value(state,mpo,pars)[1,1] ≈ 2.5337 atol=1e-3
    end

    @testset "quasiparticle_excitation" begin
        th = nonsym_xxz_ham()
        ts = MpsCenterGauged([ℂ^3],[ℂ^48]);
        (ts,pars,_) = find_groundstate(ts,th,Vumps(maxiter=400,verbose=false));
        (energies,Bs) = quasiparticle_excitation(th,Float64(pi),ts,pars);
        @test energies[1] ≈ 0.41047925 atol=1e-4
    end
end
