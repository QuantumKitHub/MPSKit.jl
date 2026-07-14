println("
------------------------------------
|   Periodic boundary conditions    |
------------------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ

@testset "periodic boundary conditions" begin
    Hs = [transverse_field_ising(), heisenberg_XXX(), classical_ising(), sixvertex()]
    for N in 2:6
        for H in Hs
            TH = convert(TensorMap, periodic_boundary_conditions(H, N))
            @test TH ≈
                permute(TH, ((vcat(N, 1:(N - 1))...,), (vcat(2N, (N + 1):(2N - 1))...,)))
        end
    end

    # non-self-dual virtual space
    let V = U1Space(0 => 1, 1 => 1), P = U1Space(0 => 1, 1 => 1)
        mpo = InfiniteMPO([randn(ComplexF64, V ⊗ P ← P ⊗ V)])
        for N in 2:4
            TH = convert(TensorMap, periodic_boundary_conditions(mpo, N))
            @test TH ≈
                permute(TH, ((vcat(N, 1:(N - 1))...,), (vcat(2N, (N + 1):(2N - 1))...,)))
        end
    end

    # fermionic tests
    h = f_hopping(Float64, Trivial)
    H = InfiniteMPOHamiltonian([space(h, 1)], (1, 2) => h)
    for N in 3:5
        H_periodic = periodic_boundary_conditions(H, N)
        terms = [(i, i + 1) => h for i in 1:(N - 1)]
        push!(terms, (1, N) => permute(h, ((2, 1), (4, 3))))
        H_periodic2 = FiniteMPOHamiltonian(physicalspace(H_periodic), terms)
        @test H_periodic ≈ H_periodic2
    end
end
