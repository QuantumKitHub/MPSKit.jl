println("
-----------------------------
|   Dynamical DMRG tests     |
-----------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ

verbosity_conv = 1

@testset "Dynamical DMRG" verbose = true begin
    L = 10
    H = force_planar(-transverse_field_ising(; L, g = -4))
    gs, = find_groundstate(FiniteMPS(L, ℙ^2, ℙ^10), H; verbosity = verbosity_conv)
    E₀ = expectation_value(gs, H)

    vals = (-0.5:0.2:0.5) .+ E₀
    eta = 0.3im

    predicted = [1 / (v + eta - E₀) for v in vals]

    @testset "Flavour $f" for f in (Jeckelmann(), NaiveInvert())
        alg = DynamicalDMRG(; flavour = f, verbosity = 0, tol = 1.0e-8)
        data = map(vals) do v
            result, = propagator(gs, v + eta, H, alg)
            return result
        end
        @test data ≈ predicted atol = 1.0e-8
    end
end
