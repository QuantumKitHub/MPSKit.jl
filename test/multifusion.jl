println("
-----------------
|   Multifusion |
-----------------
")

module TestMultifusion

    using ..TestSetup
    using Test, TestExtras
    using MPSKit
    using TensorKit

    I = IsingBimodule

    M = I(1, 2, 0) # σ
    Mop = I(2, 1, 0)
    C0 = I(1, 1, 0) # unit of C
    C1 = I(1, 1, 1)
    D0 = I(2, 2, 0) # unit of D
    D1 = I(2, 2, 1)
    V = Vect[I](M => 1)
    Vop = Vect[I](Mop => 1)
    PD = Vect[I](D0 => 1, D1 => 1)
    PC = Vect[I](C0 => 1, C1 => 1)

    bad_fusions = [(PC, PD), (PD, PC), (V, V), (Vop, Vop), (V, PC), (Vop, PD), (V, PD), (Vop, PC), (V, Vop), (Vop, V)]

    flippy(charge::IsingBimodule) = only(charge ⊗ D1)

    function TFIM_multifusion(::Type{T} = ComplexF64; g = 1.0, L = Inf, twosite = false) where {T <: Number}
        P = Vect[I](D0 => 1, D1 => 1)
        X = zeros(T, P ← P)
        for (s, f) in fusiontrees(X)
            isone(only(f.uncoupled)) ? X[s, f] .= g : X[s, f] .= -g
        end
        ZZ = zeros(T, P^2 ← P^2)
        for (s, f) in fusiontrees(ZZ)
            s.uncoupled == map(flippy, f.uncoupled) ? ZZ[s, f] .= 1 : nothing
        end

        if L == Inf
            lattice = twosite ? PeriodicArray([P, P]) : PeriodicArray([P])
            H₁ = InfiniteMPOHamiltonian(lattice, i => X for i in 1:length(lattice))
            H₂ = InfiniteMPOHamiltonian(lattice, (i, i + 1) => ZZ for i in 1:length(lattice))
        else
            lattice = fill(P, L)
            H₁ = FiniteMPOHamiltonian(lattice, i => X for i in 1:L)
            H₂ = FiniteMPOHamiltonian(lattice, (i, i + 1) => ZZ for i in 1:(L - 1))
        end
        return H₁ + H₂
    end

    @testset "InfiniteMPS construction" begin
        for (P, V) in bad_fusions
            @test_throws ArgumentError("invalid fusion channel") InfiniteMPS([P], [V])
        end
    end

    @testset "FiniteMPS construction" begin
        Prand, Vrand = rand(bad_fusions)
        @test_throws ArgumentError("one of Type IsingBimodule doesn't exist") FiniteMPS(2, Prand, Vrand)

        for (P, V) in bad_fusions
            @test_warn "no fusion channels available at site 2" FiniteMPS(rand(2:100), P, V; left = V, right = V)
        end
    end

    @testset "Exact diagonalization" begin
        H = TFIM_multifusion(; g = 0, L = 4)
        @test_throws ArgumentError("one of Type IsingBimodule doesn't exist") exact_diagonalization(H)

        E, ψ = exact_diagonalization(H; sector = D0) # test that it runs with kwarg
        @test isapprox(E, [-3, -1, 1, 3]; atol = 1.0e-6)
    end

    @testset "Finite systems" begin
        L = 6
        g = 1.5
        H = TFIM_multifusion(; g = g, L = L)
        V = Vect[I](M => 8)
        init = FiniteMPS(L, PD, V; left = Vect[I](M => 1), right = Vect[I](M => 1))
        ψ, envs = find_groundstate(init, H, DMRG())
        E = expectation_value(ψ, H, envs)

        ψ2, envs2 = find_groundstate(init, H, DMRG2(; trscheme = truncbelow(1.0e-6)))
        E2 = expectation_value(ψ2, H, envs2)

        @test isapprox(E, E2; atol = 1.0e-6)

        ED, _ = exact_diagonalization(H; sector = D0)
        @test isapprox(E, first(ED); atol = 1.0e-6)

        @test_throws ArgumentError("one of Type IsingBimodule doesn't exist") excitations(H, QuasiparticleAnsatz(), ψ)
        excE, qp = excitations(H, QuasiparticleAnsatz(), ψ2; sector=C0, num=10) # testing sector kwarg
        @test 0 < variance(qp[1], H) < 1e-8
    end

    # @testset "Infinite systems" begin
        # Multifusion: effectively studying the KW dual in SSB phase
        g = 1 / 4
        H = TFIM_multifusion(; g = g, L = Inf, twosite = true)
        V = Vect[I](M => 48)
        init = InfiniteMPS([PD, PD], [V, V])
        v₀ = variance(init, H)
        tol = 1.0e-10
        ψ, envs, δ = find_groundstate(init, H, IDMRG(;tol=tol,maxiter=400))
        E = expectation_value(ψ, H, envs)
        v = variance(ψ, H)

        ψ2, envs2, δ2 = find_groundstate(init, H, IDMRG2(; tol=tol, trscheme = truncbelow(1.0e-6), maxiter=400))
        E2 = expectation_value(ψ2, H, envs2)
        v2 = variance(ψ2, H)

        ψ3, envs3, δ3 = find_groundstate(init, H, VUMPS(;tol=tol, maxiter=400))
        E3 = expectation_value(ψ3, H, envs3)
        v3 = variance(ψ3, H)

        @test isapprox(E, E2; atol = 1.0e-6)
        @test isapprox(E, E3; atol = 1.0e-6)
        for delta in [δ, δ2, δ3]
            @test delta ≈ 0 atol=1e-3
        end
        for var in [v, v2, v3]
            @test var < v₀
            @test 0 < var < 1e-8
        end

        @test_throws ArgumentError("sectors of $V are non-diagonal") transfer_spectrum(ψ)
        @test first(transfer_spectrum(ψ2; sector = C0)) ≈ 1 # testing sector kwarg
        @test !(abs(first(transfer_spectrum(ψ2; sector = C1))) ≈ 1) # testing injectivity

        @test only(keys(entanglement_spectrum(ψ2))) == M

        momentum = 0
        @test_throws ArgumentError("one of Type IsingBimodule doesn't exist") excitations(H, QuasiparticleAnsatz(), momentum, ψ)
        excC0, qpC0 = excitations(H, QuasiparticleAnsatz(), momentum, ψ3; sector = C0) # testing sector kwarg
        excC1, qpC1 = excitations(H, QuasiparticleAnsatz(), momentum, ψ3; sector = C1)
        @test isapprox(first(excC1), abs(2 * (g - 1)); atol = 1.0e-6) # charged excitation lower in energy
        @test variance(qpC1[1], H) < 1e-8 #TODO: figure out why this is negative

        # diagonal test (M = D): injective GS in symmetric phase
        Hdual = TFIM_multifusion(; g = 1 / g, L = Inf, twosite = true)
        Vdiag = Vect[I](D0 => 24, D1 => 24)
        initdiag = InfiniteMPS([PD, PD], [Vdiag, Vdiag])
        gsdiag, envsdiag = find_groundstate(initdiag, Hdual, VUMPS(;tol=tol, maxiter=400))
        Ediag = expectation_value(gsdiag, Hdual, envsdiag)
        excD0, qpD0 = excitations(Hdual, QuasiparticleAnsatz(), momentum, gsdiag; sector = D0)
        excD1, qpD1 = excitations(Hdual, QuasiparticleAnsatz(), momentum, gsdiag; sector = D1)
        @test isapprox(first(excD1), abs(2 * (1 / g - 1)); atol = 1.0e-6) # charged excitation lower in energy
        @test variance(qpD1[1], Hdual) < 1e-8 # TODO: is this ever negative?

        # comparison to Z2 Ising: injective in symmetric phase
        HZ2 = transverse_field_ising(Z2Irrep; g = 1 / g, L = Inf, twosite = true)
        VZ2 = Z2Space(0 => 24, 1 => 24)
        PZ2 = Z2Space(0 => 1, 1 => 1)
        initZ2 = InfiniteMPS([PZ2, PZ2], [VZ2, VZ2])
        gsZ2, envsZ2 = find_groundstate(initZ2, HZ2, VUMPS(;tol=tol, maxiter=400))
        EZ2 = expectation_value(gsZ2, HZ2, envsZ2)
        @test isapprox(EZ2, Ediag; atol = 1.0e-6)
        excZ2_0, qpZ2_0 = excitations(HZ2, QuasiparticleAnsatz(), momentum, gsZ2; sector = Z2Irrep(0))
        excZ2_1, qpZ2_1 = excitations(HZ2, QuasiparticleAnsatz(), momentum, gsZ2; sector = Z2Irrep(1))
        @test isapprox(first(excZ2_1), first(excD1); atol = 1.0e-5)
        @test variance(qpZ2_1[1], HZ2) < 1e-6 #TODO: and this?
    # end

end # module TestMultifusion
