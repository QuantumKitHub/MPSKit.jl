println("
--------------------------------------
|   InfiniteMPOHamiltonian tests     |
--------------------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ

pspaces = PSPACES_TRIPLE
vspaces = VSPACES_TRIPLE
if fast_tests
    pspaces = pspaces[1:1]
    vspaces = vspaces[1:1]
end

@testset "InfiniteMPOHamiltonian $(sectortype(pspace))" for (pspace, Dspace) in zip(pspaces, vspaces)
    # generate a 1-2-3 body interaction
    operators = ntuple(3) do i
        O = rand(ComplexF64, pspace^i, pspace^i)
        return O += O'
    end

    H1 = InfiniteMPOHamiltonian(operators[1])
    H2 = InfiniteMPOHamiltonian(operators[2])
    H3 = repeat(InfiniteMPOHamiltonian(operators[3]), 2)

    # make a teststate to measure expectation values for
    ψ1 = InfiniteMPS([pspace], [Dspace])
    ψ2 = InfiniteMPS([pspace, pspace], [Dspace, Dspace])

    e1 = expectation_value(ψ1, H1)
    e2 = expectation_value(ψ1, H2)

    H1 = 2 * H1 - [1]
    @test e1 * 2 - 1 ≈ expectation_value(ψ1, H1) atol = 1.0e-10

    H1 = H1 + H2

    @test e1 * 2 + e2 - 1 ≈ expectation_value(ψ1, H1) atol = 1.0e-10

    H1 = repeat(H1, 2)

    e1 = expectation_value(ψ2, H1)
    e3 = expectation_value(ψ2, H3)

    @test e1 + e3 ≈ expectation_value(ψ2, H1 + H3) atol = 1.0e-10

    H4 = H1 + H3
    @test real(expectation_value(ψ2, H4)) >= 0

    O1_real = project_hermitian!(randn(Float64, pspace, pspace))
    O2_real = project_hermitian!(randn(Float64, pspace ⊗ pspace, pspace ⊗ pspace))
    Hreal_long_range = InfiniteMPOHamiltonian(O1_real) + InfiniteMPOHamiltonian(O2_real)
    Hcomplex_local = InfiniteMPOHamiltonian(operators[1])

    @test scalartype(Hreal_long_range) == Float64
    @test scalartype(Hcomplex_local) == ComplexF64
    for H in (
            Hreal_long_range + Hcomplex_local, Hcomplex_local + Hreal_long_range,
            Hreal_long_range - Hcomplex_local, Hcomplex_local - Hreal_long_range,
        )
        @test scalartype(H) == ComplexF64
    end
    @test expectation_value(ψ1, Hreal_long_range + Hcomplex_local) ≈
        expectation_value(ψ1, Hreal_long_range) + expectation_value(ψ1, Hcomplex_local) atol = 1.0e-10
end
