using MPSKit,TensorKit,LinearAlgebra,Test

println("------------------------------------")
println("|     States                       |")
println("------------------------------------")
@testset "FiniteMPS ($D,$d,$elt)" for (D,d,elt) in [
        (ComplexSpace(10),ComplexSpace(2),ComplexF64),
        (ℂ[SU₂](1=>1,0=>3),ℂ[SU₂](0=>1)*ℂ[SU₂](0=>1),ComplexF32)
        ]

    data = [TensorMap(rand,elt,oneunit(D)*d,D)]
    for i in 1:3
        push!(data,TensorMap(rand,elt,D*d,D))
    end
    push!(data,TensorMap(rand,elt,D*d,oneunit(D)));

    ts = FiniteMPS(data);

    ovl = dot(ts,ts);
    ts = rightorth(ts,normalize=false);
    @test ovl ≈ norm(ts.A[1])^2

    data2 = [TensorMap(rand,elt,oneunit(D)*d,D)]
    for i in 1:3
        push!(data2,TensorMap(rand,elt,D*d,D))
    end
    push!(data2,TensorMap(rand,elt,D*d,oneunit(D)));

    ts2 = FiniteMPS(data2);

    ovl2 = dot(ts,ts2);

    ts3 = ts+ts2;

    @test ovl2+ovl ≈ dot(ts,ts3)
end

@testset "InfiniteMPS ($D,$d,$elt)" for (D,d,elt) in [
        (ComplexSpace(10),ComplexSpace(2),ComplexF64),
        (ℂ[SU₂](1=>1,0=>3),ℂ[SU₂](0=>1),ComplexF32)
        ]
    tol = Float64(eps(real(elt))*100);

    #=@inferred=# uniform_leftorth([TensorMap(rand,elt,D*d,D)],tol=tol);
    #=@inferred=# uniform_rightorth([TensorMap(rand,elt,D*d,D)],tol=tol);

    ts = #=@inferred=# InfiniteMPS([TensorMap(rand,elt,D*d,D),TensorMap(rand,elt,D*d,D)],tol = tol);

    for i in 1:length(ts)
        @tensor difference[-1,-2,-3] := ts.AL[i][-1,-2,1]*ts.CR[i][1,-3]-ts.CR[i-1][-1,1]*ts.AR[i][1,-2,-3];
        @test norm(difference,Inf) < tol;

        @test transfer_left(l_LL(ts,i),ts.AL[i],ts.AL[i]) ≈ l_LL(ts,i+1)
        @test transfer_left(l_LR(ts,i),ts.AL[i],ts.AR[i]) ≈ l_LR(ts,i+1)
        @test transfer_left(l_RL(ts,i),ts.AR[i],ts.AL[i]) ≈ l_RL(ts,i+1)
        @test transfer_left(l_RR(ts,i),ts.AR[i],ts.AR[i]) ≈ l_RR(ts,i+1)

        @test transfer_right(r_LL(ts,i),ts.AL[i],ts.AL[i]) ≈ r_LL(ts,i+1)
        @test transfer_right(r_LR(ts,i),ts.AL[i],ts.AR[i]) ≈ r_LR(ts,i+1)
        @test transfer_right(r_RL(ts,i),ts.AR[i],ts.AL[i]) ≈ r_RL(ts,i+1)
        @test transfer_right(r_RR(ts,i),ts.AR[i],ts.AR[i]) ≈ r_RR(ts,i+1)
    end
end

@testset "MPSMultiline ($D,$d,$elt)" for (D,d,elt) in [
        (ComplexSpace(10),ComplexSpace(2),ComplexF64),
        (ℂ[SU₂](1=>1,0=>3),ℂ[SU₂](0=>1),ComplexF32)
        ]

    tol = Float64(eps(real(elt))*100);
    ts = MPSMultiline([TensorMap(rand,elt,D*d,D) TensorMap(rand,elt,D*d,D);TensorMap(rand,elt,D*d,D) TensorMap(rand,elt,D*d,D)],tol = tol);

    for (i,j) in Iterators.product(size(ts,1),size(ts,2))
        @tensor difference[-1,-2,-3] := ts.AL[i,j][-1,-2,1]*ts.CR[i,j][1,-3]-ts.CR[i,j-1][-1,1]*ts.AR[i,j][1,-2,-3];
        @test norm(difference,Inf) < tol;

        @test transfer_left(l_LL(ts,i,j),ts.AL[i,j],ts.AL[i,j]) ≈ l_LL(ts,i,j+1)
        @test transfer_left(l_LR(ts,i,j),ts.AL[i,j],ts.AR[i,j]) ≈ l_LR(ts,i,j+1)
        @test transfer_left(l_RL(ts,i,j),ts.AR[i,j],ts.AL[i,j]) ≈ l_RL(ts,i,j+1)
        @test transfer_left(l_RR(ts,i,j),ts.AR[i,j],ts.AR[i,j]) ≈ l_RR(ts,i,j+1)

        @test transfer_right(r_LL(ts,i,j),ts.AL[i,j],ts.AL[i,j]) ≈ r_LL(ts,i,j+1)
        @test transfer_right(r_LR(ts,i,j),ts.AL[i,j],ts.AR[i,j]) ≈ r_LR(ts,i,j+1)
        @test transfer_right(r_RL(ts,i,j),ts.AR[i,j],ts.AL[i,j]) ≈ r_RL(ts,i,j+1)
        @test transfer_right(r_RR(ts,i,j),ts.AR[i,j],ts.AR[i,j]) ≈ r_RR(ts,i,j+1)
    end
end

@testset "MPSComoving" begin
    ham = nonsym_ising_ham(lambda=4.0);
    (gs,_,_) = find_groundstate(InfiniteMPS([ℂ^2],[ℂ^10]),ham,Vumps(verbose=false));

    window = MPSComoving(gs,copy.([gs.AC[1];[gs.AR[i] for i in 2:10]]),gs);

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

println("------------------------------------")
println("|     Operators                    |")
println("------------------------------------")
@testset "mpoham $(i)" for (i,(th,Dspaces)) in enumerate([
        (nonsym_ising_ham(),[ℂ^1]),
        (u1_xxz_ham(),[ℂ[U₁](1//2=>1)]),
        (repeat(su2_xxx_ham(),2),[ℂ[SU₂](0=>1),ℂ[SU₂](1//2=>1)])
        ])

    ts = InfiniteMPS(th.pspaces,Dspaces); # generate a product state

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
        su2_xxx_ham(),
        su2u1_grossneveu()
        ])

    len = 20;

    inftemp = repeat(infinite_temperature(th),len);
    ndat = collect(map(x-> TensorMap(rand,ComplexF64,codomain(x),domain(x)),inftemp));
    ts = FiniteMPS(ndat);

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

println("------------------------------------")
println("|     Algorithms                   |")
println("------------------------------------")

@testset "find_groundstate $(ind)" for (ind,(state,alg,ham)) in enumerate([
        (InfiniteMPS([ℂ^2],[ℂ^10]),Vumps(tol_galerkin=1e-8,verbose=false),nonsym_ising_ham(lambda=2.0)),
        (InfiniteMPS([ℂ^2],[ℂ^10]),Idmrg1(tol_galerkin=1e-8,maxiter=400,verbose=false),nonsym_ising_ham(lambda=2.0)),
        (FiniteMPS(fill(TensorMap(rand,ComplexF64,ℂ^1*ℂ^2,ℂ^1),10)),Dmrg2(verbose=false),nonsym_ising_ham(lambda=2.0)),
        (FiniteMPS(fill(TensorMap(rand,ComplexF64,ℂ^1*ℂ^2,ℂ^1),10)),Dmrg(manager=SimpleManager(10),verbose=false),nonsym_ising_ham(lambda=2.0))
        ])

    #vumps type inferrence got broken by @threads, so worth it?
    (ts,pars,delta) =  #=@inferred=# find_groundstate(state,ham,alg)

    @test sum(delta) < 1e-6

    evals = expectation_value(ts,ham);
    th = repeat(ham,length(evals))-evals

    @test real(sum(expectation_value(ts,th*th))) < 1e-2 #is the ground state variance relatively low?
end

@testset "timestep $(ind)" for (ind,(state,alg,opp)) in enumerate([
    (FiniteMPS(fill(TensorMap(rand,ComplexF64,ℂ^1*ℂ^2*adjoint(ℂ^2),ℂ^1),5)),Tdvp(),commutator(nonsym_xxz_ham(spin=1//2))),
    (FiniteMPS(fill(TensorMap(rand,ComplexF64,ℂ^1*ℂ^2*adjoint(ℂ^2),ℂ^1),7)),Tdvp2(),anticommutator(nonsym_xxz_ham(spin=1//2))),
    (InfiniteMPS([ℂ^3,ℂ^3],[ℂ^50,ℂ^50]),Tdvp(),repeat(nonsym_xxz_ham(spin=1),2))
    ])

    edens = expectation_value(state,opp);

    (state,_) = timestep(state,opp,rand()/10,alg)

    @test sum(expectation_value(state,opp)) ≈ sum(edens) atol = 1e-2
end

@testset "leading_boundary $(ind)" for (ind,alg) in enumerate([
        Vumps(tol_galerkin=1e-5,verbose=false),
        PowerMethod(tol_galerkin=1e-5,verbose=false,maxiter=1000)])

    mpo = #=@inferred=# nonsym_ising_mpo();
    state = InfiniteMPS([ℂ^2],[ℂ^10]);
    (state,pars,_) = leading_boundary(state,mpo,alg);

    @test expectation_value(state,mpo,pars)[1,1] ≈ 2.5337 atol=1e-3
end


@testset "quasiparticle_excitation" begin
    th = nonsym_xxz_ham()
    ts = InfiniteMPS([ℂ^3],[ℂ^48]);
    (ts,pars,_) = find_groundstate(ts,th,Vumps(maxiter=400,verbose=false));
    (energies,Bs) = quasiparticle_excitation(th,Float64(pi),ts,pars);
    @test energies[1] ≈ 0.41047925 atol=1e-4
end

@testset "dynamicaldmrg" begin
    ham = nonsym_ising_ham(lambda=4.0);
    (gs,_,_) = find_groundstate(InfiniteMPS([ℂ^2],[ℂ^10]),ham,Vumps(verbose=false));
    window = MPSComoving(gs,[gs.AC[1];[gs.AR[i] for i in 2:10]],gs);

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

@testset "utility" begin
    tofit = x->x^-3;
    fitdist = 100;
    numexp = 10;

    (prefs,exps) = exp_decomp(tofit,fitdist = fitdist,numexp = numexp);

    exact = map(tofit,1:fitdist);
    aprox = map(x->sum([e^x*l for (e,l) in zip(exps,prefs)]),1:fitdist);

    @test exact ≈ aprox atol=1e-2
end
