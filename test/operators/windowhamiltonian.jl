println("
---------------------------------
|   WindowMPOHamiltonian tests   |
---------------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ

@testset "WindowMPOHamiltonian" begin
    # a uniform state, represented with a two-site unit cell, so that carving out a window
    # at different offsets exercises the (non-trivial) circshift alignment of the infinite
    # environments while the physics stays translationally invariant.
    H1 = force_planar(transverse_field_ising(; g = 1.5))
    gs1, = find_groundstate(InfiniteMPS([ℙ^2], [ℙ^12]), H1, VUMPS(; verbosity = 0))
    gs = repeat(gs1, 2)
    H = repeat(H1, 2)

    L = 6

    @testset "interval offset (circshift)" begin
        # the window energy must be independent of the offset for a uniform state
        energies = map((1:L, 2:(L + 1), 3:(L + 2))) do interval
            ψ = WindowMPS(gs, interval)
            Hw = WindowMPOHamiltonian(H, interval)
            @test length(Hw) == length(interval)
            return expectation_value(ψ, Hw)
        end
        @test energies[1] ≈ energies[2] atol = 1.0e-8
        @test energies[1] ≈ energies[3] atol = 1.0e-8
    end

    @testset "linear algebra" begin
        ψ = WindowMPS(gs, 1:L)
        Hw = WindowMPOHamiltonian(H, 1:L)
        e = expectation_value(ψ, Hw)

        @test expectation_value(ψ, Hw + Hw) ≈ 2 * e atol = 1.0e-8
        @test expectation_value(ψ, Hw - Hw) ≈ 0 atol = 1.0e-8
        @test expectation_value(ψ, 3 * Hw) ≈ 3 * e atol = 1.0e-8
        @test expectation_value(ψ, Hw * 3) ≈ 3 * e atol = 1.0e-8
        @test expectation_value(ψ, -Hw) ≈ -e atol = 1.0e-8
    end
end
