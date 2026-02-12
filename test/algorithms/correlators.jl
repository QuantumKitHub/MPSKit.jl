println("
-----------------------------
|   Correlators & Entropy    |
-----------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ

@testset "correlation length / entropy" begin
    ψ = InfiniteMPS([ℙ^2], [ℙ^10])
    H = force_planar(transverse_field_ising())
    ψ, = find_groundstate(ψ, H, VUMPS(; verbosity = 0))
    len_crit = correlation_length(ψ)[1]
    entrop_crit = entropy(ψ)

    H = force_planar(transverse_field_ising(; g = 4))
    ψ, = find_groundstate(ψ, H, VUMPS(; verbosity = 0))
    len_gapped = correlation_length(ψ)[1]
    entrop_gapped = entropy(ψ)

    @test len_crit > len_gapped
    @test real(entrop_crit) > real(entrop_gapped)
end

@testset "expectation value / correlator" begin
    g = 4.0
    ψ = InfiniteMPS(ℂ^2, ℂ^10)
    H = transverse_field_ising(; g)
    ψ, = find_groundstate(ψ, H, VUMPS(; verbosity = 0))

    @test expectation_value(ψ, H) ≈
        expectation_value(ψ, 1 => -2g * S_x()) + expectation_value(ψ, (1, 2) => -4S_z_S_z())
    Z_mpo = MPSKit.add_util_leg(S_z())
    G = correlator(ψ, Z_mpo, Z_mpo, 1, 2:5)
    G2 = correlator(ψ, S_z_S_z(), 1, 3:2:5)
    @test isapprox(G[2], G2[1], atol = 1.0e-2)
    @test isapprox(last(G), last(G2), atol = 1.0e-2)
    @test isapprox(G[1], expectation_value(ψ, (1, 2) => S_z_S_z()), atol = 1.0e-2)
    @test isapprox(G[2], expectation_value(ψ, (1, 3) => S_z_S_z()), atol = 1.0e-2)
end
