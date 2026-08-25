println("
------------------------------------
|   Dynamical DMRG tests (window)   |
------------------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using TensorKit

verbosity_conv = 1

@testset "Dynamical DMRG WindowMPS" verbose = true begin
    N = 20

    H = transverse_field_ising(g = -4)
    Ω = InfiniteMPS(ComplexSpace(2), ComplexSpace(20))

    (Ω, _) = find_groundstate(Ω, H, VUMPS(verbosity = verbosity_conv))
    XΩ = WindowMPS(Ω, N)
    H_w = WindowMPOHamiltonian(H, 1:N)


    gs_en = expectation_value(XΩ, H_w)
    vals = range(gs_en - 1.0, gs_en + 1.0, length = 5)
    eta = 0.3im
    predicted = [1 / (v + eta - gs_en) for v in vals]


    @testset "Flavour $f" for f in (Jeckelmann(), NaiveInvert())
        alg = DynamicalDMRG(; flavour = f, verbosity = 0, tol = 1.0e-8)
        data = map(vals) do v
            result, = propagator(XΩ, v + eta, H_w, alg)
            return result
        end
        @test data ≈ predicted atol = 1.0e-8
    end
end
