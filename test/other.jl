println("
---------------------
|   Miscellaneous   |
---------------------
")

include("setup.jl")
using Plots

@testset "plot tests" begin
    ts = InfiniteMPS([ℙ^2], [ℙ^5])
    @test transferplot(ts) isa Plots.Plot
    @test entanglementplot(ts) isa Plots.Plot
end

@testset "Old bugs" verbose = true begin
    @testset "IDMRG2 space mismatch" begin
        N = 6
        H = repeat(bilinear_biquadratic_model(SU2Irrep; θ=atan(1 / 3)), N)
        ψ₀ = InfiniteMPS(fill(SU2Space(1 => 1), N), fill(SU2Space(1//2 => 2, 3//2 => 1), N))
        alg = IDMRG2(; verbose=false, tol_galerkin=1e-5, trscheme=truncdim(32))

        ψ, envs, δ = find_groundstate(ψ₀, H, alg) # used to error
        @test ψ isa InfiniteMPS
    end

    @testset "NaN entanglement entropy" begin
        ts = InfiniteMPS([ℂ^2], [ℂ^5])
        ts = changebonds(ts, RandExpand(; trscheme=truncdim(2)))
        @test !isnan(sum(entropy(ts)))
        @test !isnan(sum(entropy(ts, 2)))
    end
end
