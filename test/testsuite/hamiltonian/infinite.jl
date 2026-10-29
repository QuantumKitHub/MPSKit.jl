using ..TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ
using Adapt

export hamiltonian_infinite_mpohamiltonian, hamiltonian_infinite_canonical_forms

function test_spaces(fast)
    pspaces = PSPACES_TRIPLE
    vspaces = VSPACES_TRIPLE
    if fast
        pspaces = pspaces[1:1]
        vspaces = vspaces[1:1]
    end
    return pspaces, vspaces
end

function hamiltonian_infinite_mpohamiltonian(AT; fast = false)
    pspaces, vspaces = test_spaces(fast)
    return @testset "InfiniteMPOHamiltonian ($AT) $(sectortype(pspace))" for (pspace, Dspace) in zip(pspaces, vspaces)
        # generate a 1-2-3 body interaction
        operators = ntuple(3) do i
            O = adapt(AT, rand(ComplexF64, pspace^i, pspace^i))
            return O += O'
        end

        H1 = InfiniteMPOHamiltonian(operators[1])
        H2 = InfiniteMPOHamiltonian(operators[2])
        H3 = repeat(InfiniteMPOHamiltonian(operators[3]), 2)

        # make a teststate to measure expectation values for
        ψ1 = adapt(AT, InfiniteMPS([pspace], [Dspace]))
        ψ2 = adapt(AT, InfiniteMPS([pspace, pspace], [Dspace, Dspace]))

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

        O1_real = project_hermitian!(adapt(AT, randn(Float64, pspace, pspace)))
        O2_real = project_hermitian!(adapt(AT, randn(Float64, pspace ⊗ pspace, pspace ⊗ pspace)))
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
end

function hamiltonian_infinite_canonical_forms(AT; fast = false)
    pspaces, vspaces = test_spaces(fast)
    return @testset "InfiniteMPOHamiltonian canonical forms ($AT) $(sectortype(pspace))" for (pspace, Dspace) in zip(pspaces, vspaces)
        O₂ = adapt(AT, rand(ComplexF64, pspace^2, pspace^2))
        O₂ += O₂'
        O₃ = adapt(AT, rand(ComplexF64, pspace^3, pspace^3))
        O₃ += O₃'
        lattice = PeriodicVector(fill(pspace, 2))
        H = InfiniteMPOHamiltonian(lattice, (i, i + 1) => O₂ for i in 1:2) +
            InfiniteMPOHamiltonian(lattice, (i, i + 1, i + 2) => O₃ for i in 1:2)

        ψ = adapt(AT, InfiniteMPS(rand, ComplexF64, fill(pspace, 2), fill(Dspace, 2)))
        E = expectation_value(ψ, H)

        # the upper-left block `[1 C; 0 A]` has orthonormal columns
        HL, = MPSKit.left_canonicalize!(copy(H))
        @test expectation_value(ψ, HL) ≈ E
        for W in parent(HL)
            V = MPSKit.SparseBlockTensorMap(W)[1:(end - 1), 1, 1, 1:(end - 1)]
            @plansor G[r; r′] := conj(V[l p; p′ r]) * V[l p; p′ r′]
            G = TensorMap(G) / dim(physicalspace(W))
            @test G ≈ id(domain(G))
        end

        # the lower-right block `[A B; 0 1]` has orthonormal rows
        HR, = MPSKit.right_canonicalize!(copy(H))
        @test expectation_value(ψ, HR) ≈ E
        for W in parent(HR)
            V = MPSKit.SparseBlockTensorMap(W)[2:end, 1, 1, 2:end]
            @plansor G[l; l′] := V[l p; p′ r] * conj(V[l′ p; p′ r])
            G = TensorMap(G) / dim(physicalspace(W))
            @test G ≈ id(codomain(G))
        end
    end
end
