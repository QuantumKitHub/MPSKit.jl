println("------------------------------------")
println("|     Algorithms                   |")
println("------------------------------------")

@timedtestset "find_groundstate $(ind)" for (ind, (state, alg, ham)) in
                                            enumerate([(InfiniteMPS([𝔹^2], [𝔹^10]),
                                                        VUMPS(; tol_galerkin=1e-8,
                                                              verbose=false),
                                                        force_planar(transverse_field_ising(;
                                                                                      hx=0.51))),
                                                       (InfiniteMPS([𝔹^2], [𝔹^10]),
                                                        IDMRG1(; tol_galerkin=1e-8,
                                                               maxiter=400, verbose=false),
                                                        force_planar(transverse_field_ising(;
                                                                                      hx=0.51))),
                                                       (InfiniteMPS([𝔹^2, 𝔹^2],
                                                                    [𝔹^10, 𝔹^10]),
                                                        IDMRG2(; trscheme=truncdim(12),
                                                               tol_galerkin=1e-8,
                                                               maxiter=400, verbose=false),
                                                        force_planar(repeat(transverse_field_ising(;
                                                                                             hx=0.51),
                                                                            2))),
                                                       (InfiniteMPS([𝔹^2], [𝔹^10]),
                                                        VUMPS(; tol_galerkin=1e-5,
                                                              verbose=false) &
                                                        GradientGrassmann(; tol=1e-8,
                                                                          verbosity=0),
                                                        force_planar(transverse_field_ising(;
                                                                                      hx=0.51))),
                                                       (FiniteMPS(rand, ComplexF64, 10, 𝔹^2,
                                                                  𝔹^10),
                                                        DMRG2(; verbose=false,
                                                              trscheme=truncdim(10)),
                                                        force_planar(transverse_field_ising(;
                                                                                      hx=0.51))),
                                                       (FiniteMPS(rand, ComplexF64, 10, 𝔹^2,
                                                                  𝔹^10),
                                                        DMRG(; verbose=false),
                                                        force_planar(transverse_field_ising(;
                                                                                      hx=0.51))),
                                                       (FiniteMPS(rand, ComplexF64, 10, 𝔹^2,
                                                                  𝔹^10),
                                                        GradientGrassmann(; verbosity=0),
                                                        force_planar(transverse_field_ising(;
                                                                                      hx=0.51)))])
    v1 = variance(state, ham)
    (ts, envs, delta) = find_groundstate(state, ham, alg)
    v2 = variance(ts, ham)

    @test v1 > v2
    @test sum(delta) < 1e-6
    @test v2 < 1e-2 #is the ground state variance relatively low?
end

@timedtestset "timestep $(ind)" for (ind, (state, alg, opp)) in
                                    enumerate([(FiniteMPS(fill(TensorMap(rand, ComplexF64,
                                                                         𝔹^1 * 𝔹^2, 𝔹^1),
                                                               5)), TDVP(),
                                                force_planar(xxx(; spin=1 // 2))),
                                               (FiniteMPS(fill(TensorMap(rand, ComplexF64,
                                                                         𝔹^1 * 𝔹^2, 𝔹^1),
                                                               7)), TDVP2(),
                                                force_planar(xxx(; spin=1 // 2))),
                                               (InfiniteMPS([𝔹^3, 𝔹^3], [𝔹^50, 𝔹^50]),
                                                TDVP(),
                                                force_planar(repeat(xxx(;
                                                                                   spin=1),
                                                                    2)))])
    envs = environments(state,opp)                                                                
    edens = expectation_value(state, opp, envs)
    dt = rand() / 10

    (state_not, _) = timestep(state, opp, dt, alg, envs)

    @test sum(expectation_value(state_not, opp, envs)) ≈ sum(edens) atol = 1e-4

    # do trivial TimedOperator
    oppt  = TimedOperator(opp)
    envst = environments(state,oppt);
    envs  = environments(state,opp);

    (state_t, _) = timestep(state, oppt, 1., dt, alg, envst)

    @test sum(expectation_value(state_t, oppt, 1., envst)) ≈ sum(expectation_value(state_not, opp, envs)) atol = 1e-4

    # do sum of TimedOperator
    oppt = 0.5*TimedOperator(opp) + 0.5*TimedOperator(opp)
    envst = environments(state,oppt);

    (state_t, _) = timestep(state, oppt, 1., dt, alg, envst)

    @test sum(expectation_value(state_t, oppt, 1., envst)) ≈ sum(expectation_value(state_not, opp, envs)) atol = 1e-4
end

@timedtestset "time evolution windowMPS" begin

    simpleH =  xxx(; spin=1 // 2)
    alg = TDVP()

    # with regular MPOHamiltonian
    H0left   =  simpleH;
    H0middle =  repeat(  simpleH, 20); # (= Hmiddle)
    H0right  =  simpleH;

    Ψ = InfiniteMPS([ℂ^2],[ℂ^20]);
    Ψwindow = WindowMPS(Ψ,20);

    Hwindow_0 = Window(H0left,H0middle,H0right);
    WindEnvs_0 = environments(Ψwindow,Hwindow_0);
    timevolvedwindow_0,WindEnvs_0 = timestep(Ψwindow,Hwindow_0,0.1,alg,WindEnvs_0);

    # with SumOfOperators
    Hleft = SumOfOperators([0.5*simpleH, 0.5*simpleH]) ;
    Hmiddle =  SumOfOperators([0.5*repeat( simpleH, 20),0.5*repeat( simpleH, 20)]) ;
    Hright =  SumOfOperators([0.5*simpleH, 0.5*simpleH]) ;

    Hwindow = Window(Hleft,Hmiddle,Hright);
    WindEnvs = environments(Ψwindow,Hwindow);

    timevolvedwindow,WindEnvs = timestep(Ψwindow,Hwindow,0.1,alg,WindEnvs);

    @test timevolvedwindow.left_gs  ≈ timevolvedwindow_0.left_gs  atol = 1e-10
    @test timevolvedwindow.window   ≈ timevolvedwindow_0.window   atol = 1e-10
    @test timevolvedwindow.right_gs ≈ timevolvedwindow_0.right_gs atol = 1e-10

    # Now with time dependence
    Htleft   =  TimedOperator(simpleH);
    Htmiddle =  TimedOperator(repeat(  simpleH, 20)); # (= Hmiddle)
    Htright  =  TimedOperator(simpleH);

    Hwindow_t = Window(Htleft,Htmiddle,Htright);
    WindEnvs_t = environments(Ψwindow,Hwindow_t);
    timevolvedwindow_t,WindEnvs_t = timestep(Ψwindow,Hwindow_t,0.,0.1,alg,WindEnvs_t);

    @test timevolvedwindow_t.left_gs  ≈ timevolvedwindow_0.left_gs  atol = 1e-10
    @test timevolvedwindow_t.window   ≈ timevolvedwindow_0.window   atol = 1e-12
    @test timevolvedwindow_t.right_gs ≈ timevolvedwindow_0.right_gs atol = 1e-12

    end

@timedtestset "leading_boundary $(ind)" for (ind, alg) in
                                            enumerate([VUMPS(; tol_galerkin=1e-5,
                                                             verbose=false)
                                                       GradientGrassmann(; verbosity=0)])
    mpo = force_planar(nonsym_ising_mpo())
    state = InfiniteMPS([𝔹^2], [𝔹^10])
    (state, envs) = leading_boundary(state, mpo, alg)
    (state, envs) = changebonds(state, mpo, OptimalExpand(; trscheme=truncdim(3)), envs)
    (state, envs) = leading_boundary(state, mpo, alg)

    @test dim(space(state.AL[1, 1], 1)) == 13
    @test expectation_value(state, envs)[1, 1] ≈ 2.5337 atol = 1e-3
end

@timedtestset "quasiparticle_excitation" begin
    @timedtestset "infinite (ham)" begin
        th = repeat(force_planar(xxx()), 2)
        ts = InfiniteMPS([𝔹^3, 𝔹^3], [𝔹^48, 𝔹^48])
        ts, envs, _ = find_groundstate(ts, th; maxiter=400, verbose=false)
        energies, Bs = excitations(th, QuasiparticleAnsatz(), Float64(pi), ts, envs)
        @test energies[1] ≈ 0.41047925 atol = 1e-4
        @test variance(Bs[1], th) < 1e-8
    end
    @timedtestset "infinite (mpo)" begin
        th = repeat(nonsym_sixvertex_mpo(), 2)
        ts = InfiniteMPS([ℂ^2, ℂ^2], [ℂ^10, ℂ^10])
        (ts, envs, _) = leading_boundary(ts, th, VUMPS(; maxiter=400, verbose=false))
        (energies, Bs) = excitations(th, QuasiparticleAnsatz(), [0.0, Float64(pi / 2)], ts,
                                     envs; verbose=false)
        @test abs(energies[1]) > abs(energies[2]) # has a minima at pi/2
    end

    @timedtestset "finite" begin
        th = force_planar(transverse_field_ising())
        ts = InfiniteMPS([𝔹^2], [𝔹^12])
        (ts, envs, _) = find_groundstate(ts, th; maxiter=400, verbose=false)
        (energies, Bs) = excitations(th, QuasiparticleAnsatz(), 0.0, ts, envs)
        inf_en = energies[1]

        fin_en = map([30, 20, 10]) do len
            ts = FiniteMPS(rand, ComplexF64, len, 𝔹^2, 𝔹^12)
            (ts, envs, _) = find_groundstate(ts, th; verbose=false)

            #find energy with quasiparticle ansatz
            (energies_QP, Bs) = excitations(th, QuasiparticleAnsatz(), ts, envs)
            @test variance(Bs[1], th) < 1e-8

            #find energy with normal dmrg
            (energies_dm, _) = excitations(th,
                                           FiniteExcited(;
                                                         gsalg=DMRG(; verbose=false,
                                                                    tol=1e-6)), ts)
            @test energies_dm[1] ≈ energies_QP[1] + sum(expectation_value(ts, th, envs)) atol = 1e-4

            return energies_QP[1]
        end

        @test issorted(abs.(fin_en .- inf_en))
    end
end

@timedtestset "changebonds $((pspace,Dspace))" for (pspace, Dspace) in [(𝔹^4, 𝔹^3),
                                                                        (Rep[SU₂](1 => 1),
                                                                         Rep[SU₂](0 => 2, 1 => 2, 2 => 1))]
    @timedtestset "mpo" begin
        #random nn interaction
        nn = TensorMap(rand, ComplexF64, pspace * pspace, pspace * pspace)
        nn += nn'

        mpo1 = periodic_boundary_conditions(convert(DenseMPO,
                                                    make_time_mpo(MPOHamiltonian(nn), 0.1,
                                                                  WII())), 10)
        mpo2 = changebonds(mpo1, SvdCut(; trscheme=truncdim(5)))

        @test dim(space(mpo2[5], 1)) < dim(space(mpo1[5], 1))
    end

    @timedtestset "infinite mps" begin
        #random nn interaction
        nn = TensorMap(rand, ComplexF64, pspace * pspace, pspace * pspace)
        nn += nn'

        state = InfiniteMPS([pspace, pspace], [Dspace, Dspace])

        state_re = changebonds(state,
                               RandExpand(; trscheme=truncdim(dim(Dspace) * dim(Dspace))))
        @test dot(state, state_re) ≈ 1 atol = 1e-8

        (state_oe, _) = changebonds(state, repeat(MPOHamiltonian(nn), 2),
                                    OptimalExpand(;
                                                  trscheme=truncdim(dim(Dspace) *
                                                                    dim(Dspace))))
        @test dot(state, state_oe) ≈ 1 atol = 1e-8

        (state_vs, _) = changebonds(state, repeat(MPOHamiltonian(nn), 2),
                                    VUMPSSvdCut(; trscheme=notrunc()))
        @test dim(left_virtualspace(state, 1)) < dim(left_virtualspace(state_vs, 1))

        state_vs_tr = changebonds(state_vs, SvdCut(; trscheme=truncdim(dim(Dspace))))
        @test dim(right_virtualspace(state_vs_tr, 1)) < dim(right_virtualspace(state_vs, 1))
    end

    @timedtestset "finite mps" begin
        #random nn interaction
        nn = TensorMap(rand, ComplexF64, pspace * pspace, pspace * pspace)
        nn += nn'

        state = FiniteMPS(10, pspace, Dspace)

        state_re = changebonds(state,
                               RandExpand(; trscheme=truncdim(dim(Dspace) * dim(Dspace))))
        @test dot(state, state_re) ≈ 1 atol = 1e-8

        (state_oe, _) = changebonds(state, MPOHamiltonian(nn),
                                    OptimalExpand(;
                                                  trscheme=truncdim(dim(Dspace) *
                                                                    dim(Dspace))))
        @test dot(state, state_oe) ≈ 1 atol = 1e-8

        state_tr = changebonds(state_oe, SvdCut(; trscheme=truncdim(dim(Dspace))))

        @test dim(left_virtualspace(state_tr, 5)) < dim(right_virtualspace(state_oe, 5))
    end

    @timedtestset "MPSMultiline" begin
        o = TensorMap(rand, ComplexF64, pspace * pspace, pspace * pspace)
        mpo = MPOMultiline(o)

        t = TensorMap(rand, ComplexF64, Dspace * pspace, Dspace)
        state = MPSMultiline(fill(t, 1, 1))

        state_re = changebonds(state,
                               RandExpand(; trscheme=truncdim(dim(Dspace) * dim(Dspace))))
        @test dot(state, state_re) ≈ 1 atol = 1e-8

        (state_oe, _) = changebonds(state, mpo,
                                    OptimalExpand(;
                                                  trscheme=truncdim(dim(Dspace) *
                                                                    dim(Dspace))))
        @test dot(state, state_oe) ≈ 1 atol = 1e-8

        state_tr = changebonds(state_oe, SvdCut(; trscheme=truncdim(dim(Dspace))))

        @test dim(right_virtualspace(state_tr, 1, 1)) <
              dim(left_virtualspace(state_oe, 1, 1))
    end
end

@timedtestset "dynamicaldmrg flavour $(f)" for f in (Jeckelmann(), NaiveInvert())
    ham = force_planar(transverse_field_ising(; hx=4.0))
    (gs, _, _) = find_groundstate(InfiniteMPS([𝔹^2], [𝔹^10]), ham, VUMPS(; verbose=false))
    window = WindowMPS(gs, copy.([gs.AC[1]; [gs.AR[i] for i in 2:10]]), gs)

    szd = force_planar(TensorMap([1 0; 0 -1], ℂ^2, ℂ^2))
    @test expectation_value(gs, szd)[1] ≈ expectation_value(window, szd)[1] atol = 1e-10

    polepos = expectation_value(gs, ham, 10)
    @test polepos ≈ expectation_value(window, ham)[2]

    vals = (-0.5:0.2:0.5) .+ polepos
    eta = 0.3im

    predicted = [1 / (v + eta - polepos) for v in vals]

    data = similar(predicted)
    for (i, v) in enumerate(vals)
        (data[i], _) = propagator(window, v + eta, ham,
                                  DynamicalDMRG(; flavour=f, verbose=false))
    end

    @test data ≈ predicted atol = 1e-8
end

@timedtestset "fidelity susceptibility" begin
    lambs = [1.05, 2.0, 4.0]

    for l in lambs
        (X, Y, Z) = nonsym_spintensors(1 // 2) .* 2

        @tensor ZZ_data[-1 -2; -3 -4] := Z[-1 -3] * Z[-2 -4]
        @tensor X_data[-1; -2] := X[-1 -2]

        ZZham = MPOHamiltonian(ZZ_data)
        Xham = MPOHamiltonian(X_data)

        th = ZZham + l * Xham

        ts = InfiniteMPS([ℂ^2], [ℂ^20])
        (ts, envs, _) = find_groundstate(ts, th, VUMPS(; maxiter=1000, verbose=false))

        #test if the infinite fid sus approximates the analytical one
        num_sus = fidelity_susceptibility(ts, th, [Xham], envs; maxiter=10)
        ana_sus = abs.(1 / (16 * l^2 * (l^2 - 1)))
        @test ana_sus ≈ num_sus[1, 1] atol = 1e-2

        #test if the finite fid sus approximates the analytical one with increasing system size
        fin_en = map([30, 20, 10]) do len
            ts = FiniteMPS(rand, ComplexF64, len, ℂ^2, ℂ^20)
            (ts, envs, _) = find_groundstate(ts, th, DMRG(; verbose=false))
            num_sus = fidelity_susceptibility(ts, th, [Xham], envs; maxiter=10)
            return num_sus[1, 1] / len
        end
        @test issorted(abs.(fin_en .- ana_sus))
    end
end

#stub tests
@timedtestset "correlation length / entropy" begin
    st = InfiniteMPS([𝔹^2], [𝔹^10])
    th = force_planar(transverse_field_ising())
    (st, _) = find_groundstate(st, th, VUMPS(; verbose=false))
    len_crit = correlation_length(st)[1]
    entrop_crit = entropy(st)

    th = force_planar(transverse_field_ising(; hx=4))
    (st, _) = find_groundstate(st, th, VUMPS(; verbose=false))
    len_gapped = correlation_length(st)[1]
    entrop_gapped = entropy(st)

    @test len_crit > len_gapped
    @test real(entrop_crit) > real(entrop_gapped)
end

@timedtestset "expectation value / correlator" begin
    st = InfiniteMPS([ℂ^2], [ℂ^10])
    th = transverse_field_ising(; hx=4)
    (st, _) = find_groundstate(st, th, VUMPS(; verbose=false))

    sz_mpo = TensorMap([1.0 0; 0 -1], ℂ^1 * ℂ^2, ℂ^2 * ℂ^1)
    sz = TensorMap([1.0 0; 0 -1], ℂ^2, ℂ^2)
    id_mpo = TensorMap([1.0 0; 0 1.0], ℂ^1 * ℂ^2, ℂ^2 * ℂ^1)
    @tensor szsz[-1 -2; -3 -4] := sz[-1 -3] * sz[-2 -4]

    @test isapprox(expectation_value(st, [sz_mpo], 1), expectation_value(st, sz, 1),
                   atol=1e-2)
    @test isapprox(expectation_value(st, [sz_mpo, sz_mpo], 1),
                   expectation_value(st, szsz, 1), atol=1e-2)
    @test isapprox(expectation_value(st, [sz_mpo, sz_mpo], 2),
                   expectation_value(st, szsz, 1), atol=1e-2)

    G = correlator(st, sz_mpo, sz_mpo, 1, 2:5)
    G2 = correlator(st, szsz, 1, 3:2:5)
    @test isapprox(G[2], G2[1], atol=1e-2)
    @test isapprox(last(G), last(G2), atol=1e-2)
    @test isapprox(G[1], expectation_value(st, szsz, 1), atol=1e-2)
    @test isapprox(G[2], expectation_value(st, [sz_mpo, id_mpo, sz_mpo], 1), atol=1e-2)
    @test isapprox(first(correlator(st, sz_mpo, sz_mpo, 1, 2)),
                   expectation_value(st, szsz, 1), atol=1e-2)
end

@timedtestset "approximate" begin
    @timedtestset "mpo * infinite ≈ infinite" begin
        st = InfiniteMPS([𝔹^2, 𝔹^2], [𝔹^10, 𝔹^10])
        th = force_planar(repeat(transverse_field_ising(; hx=4), 2))

        dt = 1e-3
        sW1 = make_time_mpo(th, dt, TaylorCluster{3}())
        sW2 = make_time_mpo(th, dt, WII())
        W1 = convert(DenseMPO, sW1)
        W2 = convert(DenseMPO, sW2)

        (st1, _) = approximate(st, (sW1, st), VUMPS(; verbose=false))
        (st2, _) = approximate(st, (W2, st), VUMPS(; verbose=false))
        (st3, _) = approximate(st, (W1, st), IDMRG1(; verbose=false))
        (st4, _) = approximate(st, (sW2, st),
                               IDMRG2(; trscheme=truncdim(20), verbose=false))
        (st5, _) = timestep(st, th, dt, TDVP())
        st6 = changebonds(W1 * st, SvdCut(; trscheme=truncdim(10)))

        @test abs(dot(st1, st5)) ≈ 1.0 atol = dt
        @test abs(dot(st3, st5)) ≈ 1.0 atol = dt
        @test abs(dot(st6, st5)) ≈ 1.0 atol = dt
        @test abs(dot(st2, st4)) ≈ 1.0 atol = dt

        nW1 = changebonds(W1, SvdCut(; trscheme=truncerr(dt))) #this should be a trivial mpo now
        @test dim(space(nW1.opp[1, 1], 1)) == 1
    end

    @timedtestset "finitemps1 ≈ finitemps2" for alg in [DMRG(; verbose=false),
                                                        DMRG2(; verbose=false,
                                                              trscheme=truncdim(10))]
        a = FiniteMPS(10, ℂ^2, ℂ^10)
        b = FiniteMPS(10, ℂ^2, ℂ^20)

        before = abs(dot(a, b))

        a = first(approximate(a, b, alg))

        after = abs(dot(a, b))

        @test before < after
    end

    @timedtestset "mpo * finitemps1 ≈ finitemps2" for alg in [DMRG(; verbose=false),
                                                              DMRG2(; verbose=false,
                                                                    trscheme=truncdim(10))]
        a = FiniteMPS(10, ℂ^2, ℂ^10)
        b = FiniteMPS(10, ℂ^2, ℂ^20)
        th = transverse_field_ising(; hx=3)
        smpo = make_time_mpo(th, 0.01, WI())

        before = abs(dot(b, b))

        (a, _) = approximate(a, (smpo, b), alg)

        (b, _) = timestep(b, th, -0.01, TDVP())
        after = abs(dot(a, b))

        @test before ≈ after atol = 0.001
    end
end

@timedtestset "periodic boundary conditions" begin
    len = 10

    #impose periodic boundary conditions on the hamiltonian (circle size 10)
    th = transverse_field_ising()
    th = periodic_boundary_conditions(th, len)

    ts = FiniteMPS(len, ℂ^2, ℂ^10)

    (gs, envs) = find_groundstate(ts, th, DMRG(; verbose=false))

    #translation mpo:
    @tensor bulk[-1 -2; -3 -4] := isomorphism(ℂ^2, ℂ^2)[-2, -4] *
                                  isomorphism(ℂ^2, ℂ^2)[-1, -3]
    translation = periodic_boundary_conditions(DenseMPO(bulk), len)

    #the groundstate should be translation invariant:
    ut = Tensor(ones, ℂ^1)
    @tensor leftstart[-1 -2; -3] := l_LL(gs)[-1, -3] * conj(ut[-2])
    v = leftstart *
        TransferMatrix([gs.AC[1]; gs.AR[2:end]], translation[:], [gs.AC[1]; gs.AR[2:end]])
    expval = @tensor v[1, 2, 3] * r_RR(gs)[3, 1] * ut[2]

    @test expval ≈ 1 atol = 1e-5

    (energies, values) = exact_diagonalization(th; which=:SR)
    @test energies[1] ≈ sum(expectation_value(gs, th)) atol = 1e-5
end