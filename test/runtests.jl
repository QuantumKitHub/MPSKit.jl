using MPSKit,TensorKit,Test,OptimKit,MPSKitModels,TestExtras
using MPSKit:_transpose_tail,_transpose_front,@plansor;
include("planarspace.jl");

println("------------------------------------")
println("|     States                       |")
println("------------------------------------")
@timedtestset "FiniteMPS ($D,$d,$elt)" for (D,d,elt) in [
        (𝔹^10,𝔹^2,ComplexF64),
        (Rep[SU₂](1=>1,0=>3),Rep[SU₂](0=>1)*Rep[SU₂](0=>1),ComplexF32)
        ]

    ts = FiniteMPS(rand,elt,rand(3:20),d,D);

    ovl = dot(ts,ts);

    @test ovl ≈ norm(ts.AC[1])^2

    for i in 1:length(ts)
        @test ts.AC[i] ≈ ts.AL[i]*ts.CR[i]
        @test ts.AC[i] ≈ MPSKit._transpose_front(ts.CR[i-1]*MPSKit._transpose_tail(ts.AR[i]))
    end

    @test elt == eltype(eltype(ts))

    ts = ts*3
    @test ovl*9 ≈ norm(ts)^2
    ts = 3*ts
    @test ovl*9*9 ≈ norm(ts)^2

    @test norm(2*ts+ts-3*ts) ≈ 0.0 atol = sqrt(eps(real(elt)))
end

@timedtestset "InfiniteMPS ($D,$d,$elt)" for (D,d,elt) in [
        (𝔹^10,𝔹^2,ComplexF64),
        (Rep[U₁](1=>3),Rep[U₁](0=>1),ComplexF64)
        ]
    tol = Float64(eps(real(elt))*100);

    ts = InfiniteMPS([TensorMap(rand,elt,D*d,D),TensorMap(rand,elt,D*d,D)],tol = tol);

    for i in 1:length(ts)
        @plansor difference[-1 -2;-3] := ts.AL[i][-1 -2;1]*ts.CR[i][1;-3]-ts.CR[i-1][-1;1]*ts.AR[i][1 -2;-3];
        @test norm(difference,Inf) < tol*10;

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

@timedtestset "MPSMultiline ($D,$d,$elt)" for (D,d,elt) in [
        (𝔹^10,𝔹^2,ComplexF64),
        (Rep[U₁](1=>3),Rep[U₁](0=>1),ComplexF32)
        ]

    tol = Float64(eps(real(elt))*100);
    ts = MPSMultiline([TensorMap(rand,elt,D*d,D) TensorMap(rand,elt,D*d,D);TensorMap(rand,elt,D*d,D) TensorMap(rand,elt,D*d,D)],tol = tol);

    for i = 1:size(ts,1), j = 1:size(ts,2)
        @plansor difference[-1 -2;-3] := ts.AL[i,j][-1 -2;1]*ts.CR[i,j][1;-3]-ts.CR[i,j-1][-1;1]*ts.AR[i,j][1 -2;-3];
        @test norm(difference,Inf) < tol*10;

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


@timedtestset "MPSComoving" begin
    ham = force_planar(nonsym_ising_ham(lambda=4.0));
    (gs,_,_) = find_groundstate(InfiniteMPS([𝔹^2],[𝔹^10]),ham,Vumps(verbose=false));

    #constructor 1 - give it a plain array of tensors
    window_1 = MPSComoving(gs,copy.([gs.AC[1];[gs.AR[i] for i in 2:10]]),gs);

    #constructor 2 - used to take a "slice" from an infinite mps
    window_2 = MPSComoving(gs,10);

    # we should logically have that window_1 approximates window_2
    ovl = dot(window_1,window_2)
    @test ovl ≈ 1 atol=1e-8

    #constructor 3 - random initial tensors
    window = MPSComoving(rand,ComplexF64,10,𝔹^2,𝔹^10,gs,gs)
    normalize!(window);

    for i in 1:length(window)
        @test window.AC[i] ≈ window.AL[i]*window.CR[i]
        @test window.AC[i] ≈ MPSKit._transpose_front(window.CR[i-1]*MPSKit._transpose_tail(window.AR[i]))
    end

    @test norm(window) ≈ 1
    window = window*3
    @test 9 ≈ norm(window)^2
    window = 3*window
    @test 9*9 ≈ norm(window)^2
    normalize!(window)

    e1 = expectation_value(window,ham);

    v1 = variance(window,ham)
    (window,envs,_) = find_groundstate(window,ham,Dmrg(verbose=false));
    v2 = variance(window,ham)

    e2 = expectation_value(window,ham);

    @test v2<v1
    @test real(e2[2]) ≤ real(e1[2])

    (window,envs) = timestep(window,ham,0.1,Tdvp2(),envs)
    (window,envs) = timestep(window,ham,0.1,Tdvp(),envs)

    e3 = expectation_value(window,ham);

    @test e2[1] ≈ e3[1]
    @test e2[2] ≈ e3[2]
end

@timedtestset "Quasiparticle state" begin
    @timedtestset "Finite" for (th,D,d) in [
        (force_planar(nonsym_ising_ham()),𝔹^10,𝔹^2),
        (su2_xxx_ham(spin=1),Rep[SU₂](1=>1,0=>3),Rep[SU₂](1=>1))
        ]

        ts = FiniteMPS(rand,ComplexF64,rand(4:20),d,D);
        normalize!(ts);

        #rand_quasiparticle is a private non-exported function
        qst1 = MPSKit.LeftGaugedQP(rand,ts);
        qst2 = MPSKit.LeftGaugedQP(rand,ts);

        @test norm(axpy!(1,qst1,copy(qst2))) ≤ norm(qst1) + norm(qst2)
        @test norm(qst1)*3 ≈ norm(qst1*3)

        normalize!(qst1);

        qst1_f = convert(FiniteMPS,qst1);
        qst2_f = convert(FiniteMPS,qst2);

        ovl_f = dot(qst1_f,qst2_f)
        ovl_q = dot(qst1,qst2)
        @test ovl_f ≈ ovl_q atol=1e-5
        @test norm(qst1_f) ≈ norm(qst1) atol=1e-5

        ev_f = sum(expectation_value(qst1_f,th)-expectation_value(ts,th))
        ev_q = dot(qst1,effective_excitation_hamiltonian(th,qst1));
        @test ev_f ≈ ev_q atol=1e-5
    end

    @timedtestset "Infinite" for (th,D,d) in [
        (force_planar(nonsym_ising_ham()),𝔹^10,𝔹^2),
        (su2_xxx_ham(spin=1),Rep[SU₂](1=>1,0=>3),Rep[SU₂](1=>1))
        ]

        period = rand(1:4);
        ts = InfiniteMPS(fill(d,period),fill(D,period));

        #rand_quasiparticle is a private non-exported function
        qst1 = MPSKit.LeftGaugedQP(rand,ts);
        qst2 = MPSKit.LeftGaugedQP(rand,ts);

        @test norm(axpy!(1,qst1,copy(qst2))) ≤ norm(qst1) + norm(qst2)
        @test norm(qst1)*3 ≈ norm(qst1*3)

        @test dot(qst1,convert(MPSKit.LeftGaugedQP,convert(MPSKit.RightGaugedQP,qst1))) ≈ dot(qst1,qst1) atol=1e-10
    end

end

println("------------------------------------")
println("|     Operators                    |")
println("------------------------------------")

@timedtestset "mpoham $((pspace,Dspace))" for (pspace,Dspace) in [(𝔹^4,𝔹^10),
        (Rep[U₁](0=>2),Rep[U₁]((0=>20))),
        (Rep[SU₂](1=>1),Rep[SU₂](1//2=>10,3//2=>5,5//2=>1))]

    #generate a 1-2-3 body interaction
    n = TensorMap(rand,ComplexF64,pspace,pspace); n+= n';
    nn = TensorMap(rand,ComplexF64,pspace*pspace,pspace*pspace); nn+=nn';
    nnn = TensorMap(rand,ComplexF64,pspace*pspace*pspace,pspace*pspace*pspace); nnn+=nnn';

    #can you pass in a proper mpo?
    identity = complex(isomorphism(oneunit(pspace)*pspace,pspace*oneunit(pspace)));
    mpoified = MPSKit.decompose_localmpo(MPSKit.add_util_leg(nnn));
    d3 = Array{Union{Missing,typeof(identity)},3}(missing,1,4,4);
    d3[1,1,1] = identity;
    d3[1,end,end] = identity;
    d3[1,1,2] = mpoified[1];
    d3[1,2,3] = mpoified[2];
    d3[1,3,4] = mpoified[3];
    h1 = MPOHamiltonian(d3);

    #¢an you pass in the actual hamiltonian?
    h2 = MPOHamiltonian(nn);

    #can you generate a hamiltonian using only onsite interactions?
    d1 = Array{Any,3}(missing,2,3,3);
    d1[1,1,1] = 1; d1[1,end,end] = 1;
    d1[1,1,2] = n; d1[1,2,end] = n;
    d1[2,1,1] = 1; d1[2,end,end] = 1;
    d1[2,1,2] = n; d1[2,2,end] = n;
    h3 = MPOHamiltonian(d1);

    #make a teststate to measure expectation values for
    ts1 = InfiniteMPS([pspace],[Dspace]);
    ts2 = InfiniteMPS([pspace,pspace],[Dspace,Dspace]);

    e1 = expectation_value(ts1,h1);
    e2 = expectation_value(ts1,h2);

    h1 = 2*h1-[1];
    @test e1[1]*2-1 ≈ expectation_value(ts1,h1)[1] atol=1e-10

    h1 = h1 + h2;

    @test e1[1]*2+e2[1]-1 ≈ expectation_value(ts1,h1)[1] atol=1e-10

    h1 = repeat(h1,2);

    e1 = expectation_value(ts2,h1);
    e3 = expectation_value(ts2,h3);

    @test e1+e3 ≈ expectation_value(ts2,h1+h3) atol=1e-10

    h4 = h1+h3;
    h4 = h4*h4;
    @test real(sum(expectation_value(ts2,h4)))>=0;
end

@timedtestset "DenseMPO"  for ham in (nonsym_ising_ham(),su2_xxx_ham(spin=1))
    physical_space = ham.pspaces[1];
    ou = oneunit(physical_space);

    ts = InfiniteMPS([physical_space],[ou⊕physical_space]);

    W = convert(DenseMPO,make_time_mpo(ham,1im*0.5,WII()));

    @test abs(dot(W*(W*ts),(W*W)*ts))≈1.0 atol=1e-10
end

println("------------------------------------")
println("|     Algorithms                   |")
println("------------------------------------")

@timedtestset "find_groundstate $(ind)" for (ind,(state,alg,ham)) in enumerate([
        (InfiniteMPS([𝔹^2],[𝔹^10]),Vumps(tol_galerkin=1e-8,verbose=false),force_planar(nonsym_ising_ham(lambda=2.0))),
        (InfiniteMPS([𝔹^2],[𝔹^10]), GradientGrassmann(tol=1e-8, verbosity=0), force_planar(nonsym_ising_ham(lambda=2.0))),
        (InfiniteMPS([𝔹^2],[𝔹^10]), GradientGrassmann(method=LBFGS(6; gradtol=1e-8, verbosity=0)), force_planar(nonsym_ising_ham(lambda=2.0))),
        (InfiniteMPS([𝔹^2],[𝔹^10]),Idmrg1(tol_galerkin=1e-8,maxiter=400,verbose=false),force_planar(nonsym_ising_ham(lambda=2.0))),
        (InfiniteMPS([𝔹^2,𝔹^2],[𝔹^10,𝔹^10]),Idmrg2(trscheme = truncdim(12),tol_galerkin=1e-8,maxiter=400,verbose=false),force_planar(repeat(nonsym_ising_ham(lambda=2.0),2))),
        (InfiniteMPS([𝔹^2], [𝔹^10]), Vumps(tol_galerkin=1e-5,verbose=false)&GradientGrassmann(tol=1e-8, verbosity=0), force_planar(nonsym_ising_ham(lambda=2.0))),
        (FiniteMPS(rand,ComplexF64,10,𝔹^2,𝔹^10),Dmrg2(verbose=false,trscheme=truncdim(10)),force_planar(nonsym_ising_ham(lambda=2.0))),
        (FiniteMPS(rand,ComplexF64,10,𝔹^2,𝔹^10),Dmrg(verbose=false),force_planar(nonsym_ising_ham(lambda=2.0))),
        (FiniteMPS(rand,ComplexF64,10,𝔹^2,𝔹^10),GradientGrassmann(verbosity=0),force_planar(nonsym_ising_ham(lambda=2.0)))
        ])

    v1 = variance(state,ham);
    (ts,envs,delta) =  find_groundstate(state,ham,alg)
    v2 = variance(ts,ham);

    @test v1 > v2
    @test sum(delta) < 1e-6
    @test v2 < 1e-2 #is the ground state variance relatively low?
end

@timedtestset "timestep $(ind)" for (ind,(state,alg,opp)) in enumerate([
    (FiniteMPS(fill(TensorMap(rand,ComplexF64,𝔹^1*𝔹^2,𝔹^1),5)),Tdvp(),force_planar(nonsym_xxz_ham(spin=1//2))),
    (FiniteMPS(fill(TensorMap(rand,ComplexF64,𝔹^1*𝔹^2,𝔹^1),7)),Tdvp2(),force_planar(nonsym_xxz_ham(spin=1//2))),
    (InfiniteMPS([𝔹^3,𝔹^3],[𝔹^50,𝔹^50]),Tdvp(),force_planar(repeat(nonsym_xxz_ham(spin=1),2)))
    ])

    edens = expectation_value(state,opp);

    (state,_) = timestep(state,opp,rand()/10,alg)

    @test sum(expectation_value(state,opp)) ≈ sum(edens) atol = 1e-2
end

@timedtestset "leading_boundary $(ind)" for (ind,alg) in enumerate([
        Vumps(tol_galerkin=1e-5,verbose=false);
        GradientGrassmann(verbosity=0)])

    mpo = force_planar(nonsym_ising_mpo());
    state = InfiniteMPS([𝔹^2],[𝔹^10]);
    (state,envs) = leading_boundary(state,mpo,alg);
    (state,envs) = changebonds(state,mpo,OptimalExpand(trscheme=truncdim(3)),envs)
    (state,envs) = leading_boundary(state,mpo,alg);

    @test dim(space(state.AL[1,1],1)) == 13
    @test expectation_value(state,envs)[1,1] ≈ 2.5337 atol=1e-3
end

@timedtestset "quasiparticle_excitation" begin
    @timedtestset "infinite (ham)" begin
        th = force_planar(nonsym_xxz_ham())
        ts = InfiniteMPS([𝔹^3],[𝔹^48]);
        (ts,envs,_) = find_groundstate(ts,th,Vumps(maxiter=400,verbose=false));
        (energies,Bs) = excitations(th,QuasiparticleAnsatz(),Float64(pi),ts,envs);
        @test energies[1] ≈ 0.41047925 atol=1e-4
        @test variance(Bs[1],th) < 1e-8
    end
    @timedtestset "infinite (mpo)" begin
        th = nonsym_sixvertex_mpo();
        ts = InfiniteMPS([ℂ^2],[ℂ^10]);
        (ts,envs,_) = leading_boundary(ts,th,Vumps(maxiter=400,verbose=false));
        (energies,Bs) = excitations(th,QuasiparticleAnsatz(),[0.0,Float64(pi/2)],ts,envs,verbose=false);
        @test abs(energies[1])>abs(energies[2]) # has a minima at pi/2
    end

    @timedtestset "finite" begin
        th = force_planar(nonsym_ising_ham())
        ts = InfiniteMPS([𝔹^2],[𝔹^12]);
        (ts,envs,_) = find_groundstate(ts,th,Vumps(maxiter=400,verbose=false));
        (energies,Bs) = excitations(th,QuasiparticleAnsatz(),0.0,ts,envs);
        inf_en = energies[1];

        fin_en = map([30,20,10]) do len
            ts = FiniteMPS(rand,ComplexF64,len,𝔹^2,𝔹^12)
            (ts,envs,_) = find_groundstate(ts,th,Dmrg(verbose=false));

            #find energy with quasiparticle ansatz
            (energies_QP,Bs) = excitations(th,QuasiparticleAnsatz(),ts,envs);
            @test variance(Bs[1],th)<1e-8

            #find energy with normal dmrg
            (energies_dm,_) = excitations(th,FiniteExcited(gsalg=Dmrg(verbose=false)),ts);
            @test energies_dm[1] ≈ energies_QP[1]+sum(expectation_value(ts,th,envs)) atol=1e-4

            energies_QP[1]
        end

        @test issorted(abs.(fin_en.-inf_en))
    end
end

@timedtestset "changebonds $((pspace,Dspace))" for (pspace,Dspace) in [(𝔹^4,𝔹^10),
        (Rep[SU₂](1=>1),Rep[SU₂](0=>10,1=>5,2=>1))]

    @timedtestset "mpo" begin
        #random nn interaction
        nn = TensorMap(rand,ComplexF64,pspace*pspace,pspace*pspace);
        nn += nn';

        mpo1 = periodic_boundary_conditions(convert(DenseMPO,make_time_mpo(MPOHamiltonian(nn),0.1,WII())),10);
        mpo2 = changebonds(mpo1,SvdCut(trscheme = truncdim(5)));

        @test dim(space(mpo2[5],1)) < dim(space(mpo1[5],1))
    end

    @timedtestset "infinite mps" begin
        #random nn interaction
        nn = TensorMap(rand,ComplexF64,pspace*pspace,pspace*pspace);
        nn += nn';

        state = InfiniteMPS([pspace,pspace],[Dspace,Dspace]);

        state_re = changebonds(state,RandExpand(trscheme = truncdim(dim(Dspace)*dim(Dspace))));
        @test dot(state,state_re) ≈ 1 atol=1e-8

        (state_oe,_) = changebonds(state,MPOHamiltonian(nn),OptimalExpand(trscheme = truncdim(dim(Dspace)*dim(Dspace))));
        @test dot(state,state_oe) ≈ 1 atol=1e-8

        (state_vs,_) = changebonds(state,MPOHamiltonian(nn),VumpsSvdCut(trscheme=notrunc()));
        @test dim(virtualspace(state,1)) < dim(virtualspace(state_vs,1))

        state_vs_tr = changebonds(state_vs,SvdCut(trscheme = truncdim(dim(Dspace))));
        @test dim(virtualspace(state_vs_tr,1)) < dim(virtualspace(state_vs,1))
    end

    @timedtestset "finite mps" begin
        #random nn interaction
        nn = TensorMap(rand,ComplexF64,pspace*pspace,pspace*pspace);
        nn += nn';

        state = FiniteMPS(10,pspace,Dspace);

        state_re = changebonds(state,RandExpand(trscheme = truncdim(dim(Dspace)*dim(Dspace))));
        @test dot(state,state_re) ≈ 1 atol=1e-8

        (state_oe,_) = changebonds(state,MPOHamiltonian(nn),OptimalExpand(trscheme = truncdim(dim(Dspace)*dim(Dspace))));
        @test dot(state,state_oe) ≈ 1 atol=1e-8

        state_tr = changebonds(state_oe,SvdCut(trscheme = truncdim(dim(Dspace))));
        @test dim(virtualspace(state_tr,5)) < dim(virtualspace(state_oe,5))
    end

    @timedtestset "MPSMultiline" begin
        o = TensorMap(rand,ComplexF64,pspace*pspace,pspace*pspace);
        mpo = MPOMultiline(o);

        t = TensorMap(rand,ComplexF64,Dspace*pspace,Dspace);
        state = MPSMultiline(fill(t,1,1));

        state_re = changebonds(state,RandExpand(trscheme = truncdim(dim(Dspace)*dim(Dspace))));
        @test dot(state,state_re) ≈ 1 atol=1e-8

        (state_oe,_) = changebonds(state,mpo,OptimalExpand(trscheme = truncdim(dim(Dspace)*dim(Dspace))));
        @test dot(state,state_oe) ≈ 1 atol=1e-8

        state_tr = changebonds(state_oe,SvdCut(trscheme = truncdim(dim(Dspace))));
        @test dim(virtualspace(state_tr,1,1)) < dim(virtualspace(state_oe,1,1))

    end
end

@timedtestset "dynamicaldmrg" begin
    ham = force_planar(nonsym_ising_ham(lambda=4.0));
    (gs,_,_) = find_groundstate(InfiniteMPS([𝔹^2],[𝔹^10]),ham,Vumps(verbose=false));
    window = MPSComoving(gs,copy.([gs.AC[1];[gs.AR[i] for i in 2:10]]),gs);

    szd = force_planar(TensorMap([1 0;0 -1],ℂ^2,ℂ^2));
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

@timedtestset "fidelity susceptibility" begin
    lambs = [1.05,2.0,4.0]

    for l in lambs

        (X,Y,Z) = nonsym_spintensors(1//2).*2;

        @tensor ZZ_data[-1 -2;-3 -4] := Z[-1 -3]*Z[-2 -4]
        @tensor X_data[-1;-2] := X[-1 -2]

        ZZham = MPOHamiltonian(ZZ_data);
        Xham = MPOHamiltonian(X_data);

        th = ZZham + l*Xham;

        ts = InfiniteMPS([ℂ^2],[ℂ^20]);
        (ts,envs,_) = find_groundstate(ts,th,Vumps(maxiter=1000,verbose=false));

        #test if the infinite fid sus approximates the analytical one
        num_sus = fidelity_susceptibility(ts,th,[Xham],envs,maxiter=10);
        ana_sus = abs.(1/(16*l^2*(l^2-1)));
        @test ana_sus ≈ num_sus[1,1] atol=1e-2

        #test if the finite fid sus approximates the analytical one with increasing system size
        fin_en = map([30,20,10]) do len
            ts = FiniteMPS(rand,ComplexF64,len,ℂ^2,ℂ^20)
            (ts,envs,_) = find_groundstate(ts,th,Dmrg(verbose=false));
            num_sus = fidelity_susceptibility(ts,th,[Xham],envs,maxiter=10);
            num_sus[1,1]/len
        end
        @test issorted(abs.(fin_en.-ana_sus))
    end

end

#stub tests
@timedtestset "correlation length / entropy" begin

    st = InfiniteMPS([𝔹^2],[𝔹^10]);
    th = force_planar(nonsym_ising_ham());
    (st,_) = find_groundstate(st,th,Vumps(verbose=false))
    len_crit = correlation_length(st)[1]
    entrop_crit = entropy(st);

    th = force_planar(nonsym_ising_ham(lambda=4));
    (st,_) = find_groundstate(st,th,Vumps(verbose=false))
    len_gapped = correlation_length(st)[1]
    entrop_gapped = entropy(st);

    @test len_crit > len_gapped;
    @test real(entrop_crit) > real(entrop_gapped);
end

@timedtestset "expectation value" begin
    st = InfiniteMPS([ℂ^2],[ℂ^10]);
    th = nonsym_ising_ham(lambda=4);
    (st,_) = find_groundstate(st,th,Vumps(verbose=false))

    sz_mpo =TensorMap([1.0 0;0 -1],ℂ^1*ℂ^2,ℂ^2*ℂ^1)
    sz =TensorMap([1.0 0;0 -1],ℂ^2,ℂ^2)
    @tensor szsz[-1 -2;-3 -4]:=sz[-1 -3]*sz[-2 -4]

    @test isapprox(expectation_value(st, [sz_mpo], 1) ,  expectation_value(st,sz,1) ,atol = 1e-2)
    @test isapprox(expectation_value(st, [sz_mpo,sz_mpo], 1) , expectation_value(st,szsz,1) ,atol = 1e-2)
    @test isapprox(expectation_value(st, [sz_mpo,sz_mpo], 2) , expectation_value(st,szsz,1) ,atol = 1e-2)
end

@timedtestset "approximate" begin
    @timedtestset "mpo * infinite ≈ infinite" begin
        st = InfiniteMPS([𝔹^2,𝔹^2],[𝔹^10,𝔹^10]);
        th = force_planar(repeat(nonsym_ising_ham(lambda=4),2));

        dt = 1e-3;
        sW1 = make_time_mpo(th,dt,WI());
        sW2 = make_time_mpo(th,dt,WII());
        W1 = convert(DenseMPO,sW1);
        W2 = convert(DenseMPO,sW2);


        (st1,_) = approximate(st,(sW1,st),Vumps(verbose=false));
        (st2,_) = approximate(st,(W2,st),Vumps(verbose=false));
        (st3,_) = approximate(st,(W1,st),Idmrg1(verbose=false));
        (st4,_) = approximate(st,(sW2,st),Idmrg2(trscheme=truncdim(20),verbose=false));
        (st5,_) = timestep(st,th,dt,Tdvp());
        st6 = changebonds(W1*st,SvdCut(trscheme=truncdim(10)))

        @test abs(dot(st1,st5)) ≈ 1.0 atol = dt
        @test abs(dot(st3,st5)) ≈ 1.0 atol = dt
        @test abs(dot(st6,st5)) ≈ 1.0 atol = dt
        @test abs(dot(st2,st4)) ≈ 1.0 atol = dt

        nW1 = changebonds(W1,SvdCut(trscheme=truncerr(dt))); #this should be a trivial mpo now
        @test dim(space(nW1.opp[1,1],1)) == 1
    end

    @timedtestset "finitemps1 ≈ finitemps2" for alg in [Dmrg(verbose=false),Dmrg2(verbose=false,trscheme=truncdim(10))]
        a = FiniteMPS(10,ℂ^2,ℂ^10);
        b = FiniteMPS(10,ℂ^2,ℂ^20);

        before = abs(dot(a,b));

        a = first(approximate(a,b,alg));

        after = abs(dot(a,b));

        @test before < after
    end

    @timedtestset "mpo*finitemps1 ≈ finitemps2" for alg in [Dmrg(verbose=false),Dmrg2(verbose=false,trscheme=truncdim(10))]
        a = FiniteMPS(10,ℂ^2,ℂ^10);
        b = FiniteMPS(10,ℂ^2,ℂ^20);
        th = nonsym_ising_ham(lambda = 3);
        smpo = make_time_mpo(th,0.01,WI());

        before = abs(dot(b,b));

        (a,_) = approximate(a,(smpo,b),alg);

        (b,_) = timestep(b,th,-0.01,Tdvp())
        after = abs(dot(a,b));

        @test before ≈ after atol = 0.001
    end
end

@timedtestset "periodic boundary conditions" begin
    len = 10;

    #impose periodic boundary conditions on the hamiltonian (cirkel size 10)
    th = nonsym_ising_ham();
    th = periodic_boundary_conditions(th,len);

    ts = FiniteMPS(len,ℂ^2,ℂ^10);

    (gs,envs) = find_groundstate(ts,th,Dmrg(verbose=false));

    #translation mpo:
    @tensor bulk[-1 -2;-3 -4] := isomorphism(ℂ^2,ℂ^2)[-2,-4]*isomorphism(ℂ^2,ℂ^2)[-1,-3];
    translation = periodic_boundary_conditions(DenseMPO(bulk),len);

    #the groundstate should be translation invariant:
    ut = Tensor(ones,ℂ^1);
    @tensor leftstart[-1 -2;-3] := l_LL(gs)[-1,-3]*conj(ut[-2]);
    v = transfer_left(leftstart,translation[:],[gs.AC[1];gs.AR[2:end]],[gs.AC[1];gs.AR[2:end]])
    expval = @tensor v[1,2,3]*r_RR(gs)[3,1]*ut[2]

    @test expval ≈ 1 atol=1e-5

    (energies,values) = exact_diagonalization(th,which=:SR);
    @test energies[1] ≈ sum(expectation_value(gs,th)) atol=1e-5
end
