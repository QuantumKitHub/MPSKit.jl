println("
---------------------------------------
|   Approximation tests (infinite)    |
---------------------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ

verbosity_conv = 1

@testset "mpo * infinite ≈ infinite" begin
    verbosity = verbosity_conv
    ψ = InfiniteMPS([ℙ^2, ℙ^2], [ℙ^10, ℙ^10])
    ψ0 = InfiniteMPS([ℙ^2, ℙ^2], [ℙ^12, ℙ^12])

    H = force_planar(repeat(transverse_field_ising(; g = 4), 2))

    dt = 1.0e-3
    sW1 = make_time_mpo(H, dt, TaylorCluster(; N = 3, compression = true, extension = true))
    sW2 = make_time_mpo(H, dt, WII())
    W1 = MPSKit.DenseMPO(sW1)
    W2 = MPSKit.DenseMPO(sW2)

    ψ1, _ = approximate(ψ0, (sW1, ψ), VOMPS(; verbosity))
    # VOMPS runs the AC and C projections of a site concurrently, so the allocator serving them
    # is shared; the serial arm takes the other branch of that choice
    ψ2, _ = with_scheduler(MPSKit.SerialScheduler()) do
        return approximate(ψ0, (W2, ψ), VOMPS(; verbosity))
    end

    ψ3, _ = approximate(ψ0, (W1, ψ), IDMRG(; verbosity))
    ψ4, _ = approximate(ψ0, (sW2, ψ), IDMRG2(; trunc = truncrank(12), verbosity))
    ψ5, _ = timestep(ψ, H, 0.0, dt, TDVP())
    ψ6 = changebonds(W1 * ψ, SvdCut(; trunc = truncrank(12)))

    @test abs(dot(ψ1, ψ5)) ≈ 1.0 atol = dt
    @test abs(dot(ψ3, ψ5)) ≈ 1.0 atol = dt
    @test abs(dot(ψ6, ψ5)) ≈ 1.0 atol = dt
    @test abs(dot(ψ2, ψ4)) ≈ 1.0 atol = dt

    nW1 = changebonds(W1, SvdCut(; trunc = trunctol(; atol = dt))) # this should be a trivial mpo now
    @test dim(space(nW1[1], 1)) == 1
end
