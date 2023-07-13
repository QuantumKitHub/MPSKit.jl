println("------------------------------------")
println("|     Algorithms                   |")
println("------------------------------------")

@testset "find_groundstate" verbose = true begin
    tol = 1e-6
    verbosity = 0
    infinite_algs = [VUMPS(; tol_galerkin=tol, verbose=verbosity > 0),
                     IDMRG1(; tol_galerkin=tol, verbose=verbosity > 0),
                     IDMRG2(; trscheme=truncdim(12), tol_galerkin=tol,
                            verbose=verbosity > 0),
                     GradientGrassmann(; tol=tol, verbosity=verbosity),
                     VUMPS(; tol_galerkin=100 * tol, verbose=verbosity > 0) &
                     GradientGrassmann(; tol=tol, verbosity=verbosity)]

    H = force_planar(transverse_field_ising(; g=1.1))
    ψ₀ = InfiniteMPS([𝔹^2], [𝔹^10])
    v₀ = variance(ψ₀, H)

end

@testset "timestep" verbose = true begin
    dt = 0.1
    algs = [TDVP(), TDVP2()]

    H = force_planar(heisenberg_XXX(; spin=1 // 2))
    ψ₀ = FiniteMPS(fill(TensorMap(rand, ComplexF64, 𝔹^1 * 𝔹^2, 𝔹^1), 5))
    E₀ = expectation_value(ψ₀, H)

    @testset "Finite $(alg isa TDVP ? "TDVP" : "TDVP2")" for alg in algs
        ψ, envs = timestep(ψ₀, H, dt, alg)
        E = expectation_value(ψ, H, envs)
        @test sum(E₀) ≈ sum(E) atol = 1e-2
    end

    @testset "Finite Trivial TimedOperator $(alg isa TDVP ? "TDVP" : "TDVP2")" for alg in algs
        ψ, envs = timestep(ψ₀, H, dt, alg)
        E = expectation_value(ψ, H, envs)
        
        Ht  = TimedOperator(H)
        (ψt, envst) = timestep(ψ₀, Ht, 0., dt, alg)
        Et = expectation_value(ψt, Ht, 0., envst)
    
        @test sum(E) ≈ sum(Et) atol = 1e-10
    end

    @testset "Finite TimedOperator $(alg isa TDVP ? "TDVP" : "TDVP2")" for alg in algs
        f(t) = 3cos(t); t = 1.

        Ht  = TimedOperator(H,f)
        (ψt, envst) = timestep(ψ₀, Ht, t, dt, alg)
        Et = expectation_value(ψt, Ht, t, envst)

        Hunt = Ht(t)
        (ψ, envs) = timestep(ψ₀, Hunt, dt, alg)
        Eunt = expectation_value(ψ, Hunt, envs)

        Hmult = f(t)*H
        (ψ, envs) = timestep(ψ₀, Hmult, dt, alg)
        Emult = expectation_value(ψ, Hmult, envs)
    
        @test sum(Eunt) ≈ sum(Et) atol = 1e-2
        @test sum(Eunt) ≈ sum(Emult) atol = 1e-2
        @test sum(Et) ≈ sum(Emult) atol = 1e-2
    end

    H = force_planar(heisenberg_XXX(InfiniteChain(2); spin=1))
    ψ₀ = InfiniteMPS([𝔹^3, 𝔹^3], [𝔹^50, 𝔹^50])
    E₀ = expectation_value(ψ₀, H)

    @testset "Infinite TDVP" begin
        ψ, envs = timestep(ψ₀, H, dt, TDVP())
        E = expectation_value(ψ, H, envs)
        @test sum(E₀) ≈ sum(E) atol = 1e-2
    end

    @testset "Finite Trivial TimedOperator TDVP" begin
        ψ, envs = timestep(ψ₀, H, dt, algs[1])
        E = expectation_value(ψ, H, envs)
        
        Ht  = TimedOperator(H)
        (ψt, envst) = timestep(ψ₀, Ht, 0., dt, algs[1])
        Et = expectation_value(ψt, Ht, 0., envst)

        @test sum(E) ≈ sum(Et) atol = 1e-10
        end

    @testset "Infinite TimedOperator TDVP" begin
        f(t) = 3cos(t); t = 1.

        Ht  = TimedOperator(H,f)
        (ψt, envst) = timestep(ψ₀, Ht, t, dt, algs[1])
        Et = expectation_value(ψt, Ht, t, envst)

        Hunt = Ht(t)
        (ψ, envs) = timestep(ψ₀, Hunt, dt, algs[1])
        Eunt = expectation_value(ψ, Hunt, envs)

        Hmult = f(t)*H
        (ψ, envs) = timestep(ψ₀, Hmult, dt, algs[1])
        Emult = expectation_value(ψ, Hmult, envs)
    
        @test sum(Eunt) ≈ sum(Et) atol = 1e-2
        @test sum(Eunt) ≈ sum(Emult) atol = 1e-2
        @test sum(Et) ≈ sum(Emult) atol = 1e-2
    end

    
end

@testset "time evolution windowMPS" begin

    simpleH =  heisenberg_XXX(; spin=1 // 2)
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
    @test timevolvedwindow_t.window   ≈ timevolvedwindow_0.window   atol = 1e-10
    @test timevolvedwindow_t.right_gs ≈ timevolvedwindow_0.right_gs atol = 1e-10

end

@testset "leading_boundary" verbose = true begin
    algs = [VUMPS(; tol_galerkin=1e-5, verbose=false),
            GradientGrassmann(; verbosity=0)]
    mpo = force_planar(classical_ising())

    ψ₀ = InfiniteMPS([𝔹^2], [𝔹^10])
    @testset "Infinite $i" for (i, alg) in enumerate(algs)
        ψ, envs = leading_boundary(ψ₀, mpo, alg)
        ψ, envs = changebonds(ψ, mpo, OptimalExpand(; trscheme=truncdim(3)), envs)
        ψ, envs = leading_boundary(ψ, mpo, alg)

        @test dim(space(ψ.AL[1, 1], 1)) == dim(space(ψ₀.AL[1, 1], 1)) + 3
        @test expectation_value(ψ, envs)[1, 1] ≈ 2.5337 atol = 1e-3
    end
end

@testset "quasiparticle_excitation" verbose = true begin
    @testset "infinite (ham)" begin
        H = force_planar(heisenberg_XXX(InfiniteChain(2)))
        ψ = InfiniteMPS([𝔹^3, 𝔹^3], [𝔹^48, 𝔹^48])
        ψ, envs, _ = find_groundstate(ψ, H; maxiter=400, verbose=false)
        energies, Bs = excitations(H, QuasiparticleAnsatz(), Float64(pi), ψ, envs)
        @test energies[1] ≈ 0.41047925 atol = 1e-4
        @test variance(Bs[1], H) < 1e-8
    end
    @testset "infinite (mpo)" begin
        th = repeat(sixvertex(), 2)
        ts = InfiniteMPS([ℂ^2, ℂ^2], [ℂ^10, ℂ^10])
        (ts, envs, _) = leading_boundary(ts, th, VUMPS(; maxiter=400, verbose=false))
        (energies, Bs) = excitations(th, QuasiparticleAnsatz(), [0.0, Float64(pi / 2)], ts,
                                     envs; verbose=false)
        @test abs(energies[1]) > abs(energies[2]) # has a minima at pi/2
    end

    @testset "finite" begin
        th = force_planar(transverse_field_ising())
        ts = InfiniteMPS([𝔹^2], [𝔹^12])
        (ts, envs, _) = find_groundstate(ts, th; maxiter=400, verbose=false)
        (energies, Bs) = excitations(th, QuasiparticleAnsatz(), 0.0, ts, envs)
        inf_en = energies[1]

        fin_en = map([20, 10]) do len
            ts = FiniteMPS(rand, ComplexF64, len, 𝔹^2, 𝔹^12)
            (ts, envs, _) = find_groundstate(ts, th; verbose=false)

            #find energy with quasiparticle ansatz
            (energies_QP, Bs) = excitations(th, QuasiparticleAnsatz(), ts, envs)
            @test variance(Bs[1], th) < 1e-6

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

@testset "changebonds $((pspace,Dspace))" verbose = true for (pspace, Dspace) in
                                                             [(𝔹^4, 𝔹^3),
                                                              (Rep[SU₂](1 => 1),
                                                               Rep[SU₂](0 => 2, 1 => 2,
                                                                        2 => 1))]
    @testset "mpo" begin
        #random nn interaction
        nn = TensorMap(rand, ComplexF64, pspace * pspace, pspace * pspace)
        nn += nn'

        mpo1 = periodic_boundary_conditions(convert(DenseMPO,
                                                    make_time_mpo(MPOHamiltonian(nn), 0.1,
                                                                  WII())), 10)
        mpo2 = changebonds(mpo1, SvdCut(; trscheme=truncdim(5)))

        @test dim(space(mpo2[5], 1)) < dim(space(mpo1[5], 1))
    end

    @testset "infinite mps" begin
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

    @testset "finite mps" begin
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

    @testset "MPSMultiline" begin
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

@testset "Dynamical DMRG" verbose = true begin
    ham = force_planar(-1.0 * MPOHamiltonian(σᶻᶻ()) + MPOHamiltonian(σˣ()) * 4.0)
    gs, = find_groundstate(InfiniteMPS([𝔹^2], [𝔹^10]), ham, VUMPS(; verbose=false))
    window = WindowMPS(gs, copy.([gs.AC[1]; [gs.AR[i] for i in 2:10]]), gs)

    szd = force_planar(S_z())
    @test expectation_value(gs, szd)[1] ≈ expectation_value(window, szd)[1] atol = 1e-10

    polepos = expectation_value(gs, ham, 10)
    @test polepos ≈ expectation_value(window, ham)[2]

    vals = (-0.5:0.2:0.5) .+ polepos
    eta = 0.3im

    predicted = [1 / (v + eta - polepos) for v in vals]

    @testset "Flavour $f" for f in (Jeckelmann(), NaiveInvert())
        alg = DynamicalDMRG(; flavour=f, verbose=false, tol=1e-8)
        data = map(vals) do v
            result, = propagator(window, v + eta, ham, alg)
            return result
        end
        @test data ≈ predicted atol = 1e-8
    end
end

@testset "fidelity susceptibility" begin
    H_X = MPOHamiltonian(σˣ())
    H_ZZ = MPOHamiltonian(σᶻᶻ())

    hamiltonian(λ) = H_ZZ + λ * H_X
    analytical_susceptibility(λ) = abs(1 / (16 * λ^2 * (λ^2 - 1)))

    for λ in [1.05, 2.0, 4.0]
        H = hamiltonian(λ)
        ψ = InfiniteMPS([ℂ^2], [ℂ^20])
        ψ, envs, = find_groundstate(ψ, H, VUMPS(; maxiter=1000, verbose=false))

        numerical_scusceptibility = fidelity_susceptibility(ψ, H, [H_X], envs; maxiter=10)
        @test numerical_scusceptibility[1, 1] ≈ analytical_susceptibility(λ) atol = 1e-2

        #test if the finite fid sus approximates the analytical one with increasing system size
        fin_en = map([20, 15, 10]) do L
            ψ = FiniteMPS(rand, ComplexF64, L, ℂ^2, ℂ^20)
            ψ, envs, = find_groundstate(ψ, H, DMRG(; verbose=false))
            numerical_scusceptibility = fidelity_susceptibility(ψ, H, [H_X], envs;
                                                                maxiter=10)
            return numerical_scusceptibility[1, 1] / L
        end
        @test issorted(abs.(fin_en .- analytical_susceptibility(λ)))
    end
end

#stub tests
@testset "correlation length / entropy" begin
    st = InfiniteMPS([𝔹^2], [𝔹^10])
    th = force_planar(transverse_field_ising())
    (st, _) = find_groundstate(st, th, VUMPS(; verbose=false))
    len_crit = correlation_length(st)[1]
    entrop_crit = entropy(st)

    th = force_planar(transverse_field_ising(; g=4))
    (st, _) = find_groundstate(st, th, VUMPS(; verbose=false))
    len_gapped = correlation_length(st)[1]
    entrop_gapped = entropy(st)

    @test len_crit > len_gapped
    @test real(entrop_crit) > real(entrop_gapped)
end

@testset "expectation value / correlator" begin
    st = InfiniteMPS([ℂ^2], [ℂ^10])
    th = transverse_field_ising(; g=4)
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

@testset "approximate" verbose = true begin
    @testset "mpo * infinite ≈ infinite" begin
        st = InfiniteMPS([𝔹^2, 𝔹^2], [𝔹^10, 𝔹^10])
        th = force_planar(repeat(transverse_field_ising(; g=4), 2))

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

    finite_algs = [DMRG(; verbose=false), DMRG2(; verbose=false, trscheme=truncdim(10))]
    @testset "finitemps1 ≈ finitemps2" for alg in finite_algs
        a = FiniteMPS(10, ℂ^2, ℂ^10)
        b = FiniteMPS(10, ℂ^2, ℂ^20)

        before = abs(dot(a, b))

        a = first(approximate(a, b, alg))

        after = abs(dot(a, b))

        @test before < after
    end

    @testset "mpo * finitemps1 ≈ finitemps2" for alg in finite_algs
        Ψ₁ = FiniteMPS(10, ℂ^2, ℂ^30)
        Ψ₂ = FiniteMPS(10, ℂ^2, ℂ^25)

        H = transverse_field_ising(; g=3.0)
        τ = 0.1

        expH = make_time_mpo(H, τ, WI())
        Ψ₂, = approximate(Ψ₂, (expH, Ψ₁), alg)
        normalize!(Ψ₂)
        Ψ₂′, = timestep(Ψ₁, H, τ, TDVP())
        @test abs(dot(Ψ₁, Ψ₁)) ≈ abs(dot(Ψ₂, Ψ₂′)) atol = 0.0001
    end
end

@testset "periodic boundary conditions" begin
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