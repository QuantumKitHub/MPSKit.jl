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

    @testset "distinct summands" begin
        # add two genuinely different Hamiltonians: the ZZ Ising string has a single bulk
        # level while the XY hopping needs two, so the boundary/bulk virtual spaces of the
        # two summands differ and have to be block-diagonalized consistently.
        ψ = WindowMPS(gs1, 1:L)
        Ha = WindowMPOHamiltonian(H1, 1:L)
        Hb = WindowMPOHamiltonian(force_planar(XY_model(; g = 0.7)), 1:L)
        @test right_virtualspace(Ha.finite_ham, 3) != right_virtualspace(Hb.finite_ham, 3)

        ea = expectation_value(ψ, Ha)
        eb = expectation_value(ψ, Hb)
        @test expectation_value(ψ, Ha + Hb) ≈ ea + eb atol = 1.0e-8
        @test expectation_value(ψ, Hb + Ha) ≈ ea + eb atol = 1.0e-8
        @test expectation_value(ψ, Ha - Hb) ≈ ea - eb atol = 1.0e-8
    end
end
