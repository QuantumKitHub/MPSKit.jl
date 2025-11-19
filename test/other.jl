println("
---------------------
|   Miscellaneous   |
---------------------
")
module TestMiscellaneous

    using ..TestSetup
    using Test, TestExtras
    using MPSKit
    using MPSKit: GeometryStyle, FiniteChainStyle, InfiniteChainStyle, OperatorStyle, MPOStyle,
        HamiltonianStyle
    using TensorKit
    using TensorKit: ℙ
    using Plots
    using Aqua

    @testset "Aqua" begin
        # TODO fix this
        Aqua.test_all(MPSKit; ambiguities = false, piracies = false)
    end

    @testset "plot tests" begin
        ψ = InfiniteMPS([ℙ^2], [ℙ^5])
        @test transferplot(ψ) isa Plots.Plot
        @test entanglementplot(ψ) isa Plots.Plot
    end

    @testset "Old bugs" verbose = true begin
        @testset "IDMRG2 space mismatch" begin
            N = 6
            H = repeat(bilinear_biquadratic_model(SU2Irrep; θ = atan(1 / 3)), N)
            ψ₀ = InfiniteMPS(
                fill(SU2Space(1 => 1), N),
                fill(SU2Space(1 // 2 => 2, 3 // 2 => 1), N)
            )
            alg = IDMRG2(; verbosity = 0, tol = 1.0e-5, trscheme = truncrank(32))

            ψ, envs, δ = find_groundstate(ψ₀, H, alg) # used to error
            @test ψ isa InfiniteMPS
        end

        @testset "NaN entanglement entropy" begin
            ψ = InfiniteMPS([ℂ^2], [ℂ^5])
            ψ = changebonds(ψ, RandExpand(; trscheme = truncrank(2)))
            @test !isnan(sum(entropy(ψ)))
            @test !isnan(sum(entropy(ψ, 2)))
        end

        @testset "changebonds with unitcells" begin
            ψ = InfiniteMPS([ℂ^2, ℂ^2, ℂ^2], [ℂ^2, ℂ^3, ℂ^4])
            H = repeat(transverse_field_ising(), 3)
            ψ1, envs = changebonds(ψ, H, OptimalExpand(; trscheme = truncrank(2)))
            @test ψ1 isa InfiniteMPS
            @test norm(ψ1) ≈ 1

            ψ2 = changebonds(ψ, RandExpand(; trscheme = truncrank(2)))
            @test ψ2 isa InfiniteMPS
            @test norm(ψ2) ≈ 1
        end

        @testset "Stackoverflow with gauging" begin
            ψ = FiniteMPS(10_000, ℂ^2, ℂ^1)
            @test ψ.AR[1] isa MPSKit.MPSTensor
            ψ.AC[1] = -ψ.AR[1] # force invalidation of ALs
            @test ψ.AL[end] isa MPSKit.MPSTensor
        end
    end

    @testset "braille" begin
        # Infinite Hamiltonians and MPOs
        # -------------------------------
        H = transverse_field_ising()
        buffer = IOBuffer()
        braille(buffer, H)
        output = String(take!(buffer))
        check = """
        ... 🭻⎡⠉⢈⎤🭻 ...
             ⎣⠀⢀⎦ 
        """
        @test output == check

        O = make_time_mpo(H, 1.0, TaylorCluster(3, false, false))
        braille(buffer, O)
        output = String(take!(buffer))
        check = """
        ... 🭻⎡⡏⠉⠛⠟⎤🭻 ...
             ⎣⡇⠀⠀⡂⎦ 
        """
        @test output == check

        # Finite Hamiltonians and MPOs
        # ----------------------------
        H = transverse_field_ising(; L = 4)
        braille(buffer, H)
        output = String(take!(buffer))
        check = " ⎡⠉⠈⎤🭻🭻⎡⠉⢈⎤🭻🭻⎡⠉⢈⎤🭻🭻⎡⡁⠀⎤ \n ⎣⠀⠀⎦  ⎣⠀⢀⎦  ⎣⠀⢀⎦  ⎣⡀⠀⎦ \n"
        @test output == check

        O = make_time_mpo(H, 1.0, TaylorCluster(3, false, false))
        braille(buffer, O)
        output = String(take!(buffer))
        check = " ⎡⠉⠉⠉⠉⎤🭻🭻⎡⡏⠉⠛⠟⎤🭻🭻⎡⡏⠉⠛⠟⎤🭻🭻⎡⡇⠀⎤ \n ⎣⠀⠀⠀⠀⎦  ⎣⡇⠀⠀⡂⎦  ⎣⡇⠀⠀⡂⎦  ⎣⡇⠀⎦ \n"
        @test output == check
    end

    @testset "Styles" begin
        @test_throws MethodError OperatorStyle(42)
        @test_throws MethodError OperatorStyle(Float64)
        @test_throws MethodError GeometryStyle("abc")
        @test_throws MethodError GeometryStyle(UInt8)

        @test OperatorStyle(MPO) == MPOStyle()
        @test OperatorStyle(InfiniteMPO) == MPOStyle()
        @test OperatorStyle(HamiltonianStyle()) == HamiltonianStyle()
        @test @constinferred OperatorStyle(MPO, InfiniteMPO, MPO) == MPOStyle()
        @test_throws ErrorException OperatorStyle(MPO, HamiltonianStyle())

        @test GeometryStyle(FiniteMPOHamiltonian) == FiniteChainStyle()
        @test GeometryStyle(InfiniteMPS) == InfiniteChainStyle()
        @test GeometryStyle(FiniteMPS) == FiniteChainStyle()
        @test GeometryStyle(FiniteMPO) == FiniteChainStyle()
        @test GeometryStyle(FiniteMPOHamiltonian) == FiniteChainStyle()
        @test GeometryStyle(InfiniteMPO) == InfiniteChainStyle()
        @test GeometryStyle(InfiniteMPOHamiltonian) == InfiniteChainStyle()

        @test GeometryStyle(GeometryStyle(FiniteMPS)) == GeometryStyle(FiniteMPS)
        @test GeometryStyle(FiniteMPS, FiniteMPO) == FiniteChainStyle()
        @test_throws ErrorException GeometryStyle(FiniteMPS, InfiniteMPO)
        @test @constinferred GeometryStyle(InfiniteMPS, InfiniteMPO, InfiniteMPS) == InfiniteChainStyle()
        @test_throws ErrorException GeometryStyle(FiniteMPS, FiniteMPO, InfiniteMPS)
    end
end
