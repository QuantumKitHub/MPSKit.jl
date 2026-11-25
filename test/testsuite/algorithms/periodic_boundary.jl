using ..TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ
using Adapt

export algorithms_periodic_boundary

function algorithms_periodic_boundary(AT; fast = false)
    return @testset "periodic boundary conditions ($AT)" begin
        Hs = map(
            H -> adapt(AT, H),
            [transverse_field_ising(), heisenberg_XXX(), classical_ising(), sixvertex()]
        )
        for N in 2:6
            for H in Hs
                TH = convert(TensorMap, periodic_boundary_conditions(H, N))
                @test TH ≈
                    permute(TH, ((vcat(N, 1:(N - 1))...,), (vcat(2N, (N + 1):(2N - 1))...,)))
            end
        end

        # non-self-dual virtual space
        fast || let V = U1Space(0 => 1, 1 => 1), P = U1Space(0 => 1, 1 => 1)
            mpo = adapt(AT, InfiniteMPO([randn(ComplexF64, V ⊗ P ← P ⊗ V)]))
            for N in 2:4
                TH = convert(TensorMap, periodic_boundary_conditions(mpo, N))
                @test TH ≈
                    permute(TH, ((vcat(N, 1:(N - 1))...,), (vcat(2N, (N + 1):(2N - 1))...,)))
            end
        end

        # fermionic tests
        h = f_hopping(Float64, Trivial)
        H = adapt(AT, InfiniteMPOHamiltonian([space(h, 1)], (1, 2) => h))
        for N in 3:5
            H_periodic = periodic_boundary_conditions(H, N)
            terms = [(i, i + 1) => h for i in 1:(N - 1)]
            push!(terms, (1, N) => permute(h, ((2, 1), (4, 3))))
            H_periodic2 = adapt(AT, FiniteMPOHamiltonian(physicalspace(H_periodic), terms))
            @test H_periodic ≈ H_periodic2
        end
    end
end
