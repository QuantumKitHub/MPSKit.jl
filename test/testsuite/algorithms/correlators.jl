using ..TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ
using Adapt

export algorithms_correlators_entropy, algorithms_correlators_expectation_value

function algorithms_correlators_entropy(AT; fast = false)
    return @testset "correlation length / entropy ($AT)" begin
        ψ = adapt(AT, InfiniteMPS([ℙ^2], [ℙ^10]))
        H = adapt(AT, force_planar(transverse_field_ising()))
        ψ, = find_groundstate(ψ, H, VUMPS(; verbosity = 0))
        len_crit = correlation_length(ψ; sector = leftunit(ψ))
        entrop_crit = entropy(ψ)

        H = adapt(AT, force_planar(transverse_field_ising(; g = 4)))
        ψ, = find_groundstate(ψ, H, VUMPS(; verbosity = 0))
        len_gapped = correlation_length(ψ; sector = leftunit(ψ))
        entrop_gapped = entropy(ψ)

        @test len_crit > len_gapped
        @test real(entrop_crit) > real(entrop_gapped)
    end
end

function algorithms_correlators_expectation_value(AT; fast = false)
    return @testset "expectation value / correlator ($AT)" begin
        g = 4.0
        ψ = adapt(AT, InfiniteMPS(ℂ^2, ℂ^10))
        H = adapt(AT, transverse_field_ising(; g))
        ψ, = find_groundstate(ψ, H, VUMPS(; verbosity = 0))

        X = adapt(AT, S_x())
        ZZ = adapt(AT, S_z_S_z())
        @test expectation_value(ψ, H) ≈
            expectation_value(ψ, 1 => -2g * X) + expectation_value(ψ, (1, 2) => -4ZZ)
        Z_mpo = MPSKit.add_util_leg(adapt(AT, S_z()))
        G = correlator(ψ, Z_mpo, Z_mpo, 1, 2:5)
        G2 = correlator(ψ, ZZ, 1, 3:2:5)
        @test isapprox(G[2], G2[1], atol = 1.0e-2)
        @test isapprox(last(G), last(G2), atol = 1.0e-2)
        @test isapprox(G[1], expectation_value(ψ, (1, 2) => ZZ), atol = 1.0e-2)
        @test isapprox(G[2], expectation_value(ψ, (1, 3) => ZZ), atol = 1.0e-2)

        # the sites have to be ordered `i < j`: every `correlator` method funnels its
        # arguments through the range version, so all three signatures have to throw
        @test_throws ArgumentError correlator(ψ, Z_mpo, Z_mpo, 3, 2)
        @test_throws ArgumentError correlator(ψ, Z_mpo, Z_mpo, 3, 1:2)
        @test_throws ArgumentError correlator(ψ, ZZ, 3, 2)
        # equal sites are not allowed either
        @test_throws ArgumentError correlator(ψ, Z_mpo, Z_mpo, 2, 2)
    end
end
