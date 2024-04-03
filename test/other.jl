println("
---------------------
|   Miscellaneous   |
---------------------
")
module TestMiscellaneous

using ..TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ
using Plots

@testset "plot tests" begin
    ψ = InfiniteMPS([ℙ^2], [ℙ^5])
    @test transferplot(ψ) isa Plots.Plot
    @test entanglementplot(ψ) isa Plots.Plot
end

@testset "Old bugs" verbose = true begin
    @testset "IDMRG2 space mismatch" begin
        N = 6
        H = repeat(bilinear_biquadratic_model(SU2Irrep; θ=atan(1 / 3)), N)
        ψ₀ = InfiniteMPS(fill(SU2Space(1 => 1), N),
                         fill(SU2Space(1 // 2 => 2, 3 // 2 => 1), N))
        alg = IDMRG2(; verbosity=0, tol=1e-5, trscheme=truncdim(32))

        ψ, envs, δ = find_groundstate(ψ₀, H, alg) # used to error
        @test ψ isa InfiniteMPS
    end

    @testset "NaN entanglement entropy" begin
        ψ = InfiniteMPS([ℂ^2], [ℂ^5])
        ψ = changebonds(ψ, RandExpand(; trscheme=truncdim(2)))
        @test !isnan(sum(entropy(ψ)))
        @test !isnan(sum(entropy(ψ, 2)))
    end

    @testset "changebonds with unitcells" begin
        ψ = InfiniteMPS([ℂ^2, ℂ^2, ℂ^2], [ℂ^2, ℂ^3, ℂ^4])
        H = repeat(transverse_field_ising(), 3)
        ψ1, envs = changebonds(ψ, H, OptimalExpand(; trscheme=truncdim(2)))
        @test ψ1 isa InfiniteMPS
        @test norm(ψ1) ≈ 1

        ψ2 = changebonds(ψ, RandExpand(; trscheme=truncdim(2)))
        @test ψ2 isa InfiniteMPS
        @test norm(ψ2) ≈ 1
    end
end

end
