using MPSKit
using TensorKit
using LinearAlgebra
using Test

@testset "MpsCenterGauged tests" begin
    @inferred leftorth([TensorMap(rand,ComplexF64,ℂ^50*ℂ^2,ℂ^50)]);
    @inferred rightorth([TensorMap(rand,ComplexF64,ℂ^50*ℂ^2,ℂ^50)]);

    ts = @inferred MpsCenterGauged([TensorMap(rand,ComplexF64,ℂ^50*ℂ^2,ℂ^50),TensorMap(rand,ComplexF64,ℂ^50*ℂ^2,ℂ^50)]);

    @tensor difference[-1,-2,-3] := ts.AL[1][-1,-2,1]*ts.CR[1][1,-3]-ts.CR[0][-1,1]*ts.AR[1][1,-2,-3];
    @test norm(difference)<1e-12;
end

@testset "General Mpo Hamiltonian shenanigans" begin #ideally be iterating over different hamiltonians
    #defining the hamiltonian
    (sx,sy,sz,id) = nonsym_spintensors(1//2);
    @tensor ham[-1 -2;-3 -4]:=(-1.5*sz)[-1,-3]*sz[-2,-4]+id[-1,-3]*sx[-2,-4];
    th = MpoHamiltonian(ham);

    ts = MpsCenterGauged([TensorMap(rand,ComplexF64,ℂ^50*ℂ^2,ℂ^50)]);

    e1 = expectation_value(ts,th);

    e2 = expectation_value(ts,2*th); #multiplication with a constant
    @test 2*e1≈e2;

    e2 = expectation_value(ts,0.5*th+th); #addition
    @test 1.5*e1≈e2;

    th -= expectation_value(ts,th);
    v = expectation_value(ts,th*th);
    @test real(v)>=0;
end

@testset "findgroundstate derping" begin
    #defining the hamiltonian
    (sx,sy,sz,id) = nonsym_spintensors(1//2)
    @tensor ham[-1 -2;-3 -4]:=(-1.5*sz)[-1,-3]*sz[-2,-4]+id[-1,-3]*sx[-2,-4]
    th = @inferred MpoHamiltonian(ham)
    ts = @inferred MpsCenterGauged([TensorMap(rand,ComplexF64,ℂ^50*ℂ^2,ℂ^50)]);

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
