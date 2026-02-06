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

    # fermionic tests
    h = f_plus_f_min(Float64, Trivial) + f_min_f_plus(Float64, Trivial)
    H = InfiniteMPOHamiltonian([space(h, 1)], (1, 2) => h)
    for N in 3:5
        H_periodic = periodic_boundary_conditions(H, N)
        terms = [(i, i + 1) => h for i in 1:(N - 1)]
        push!(terms, (1, N) => permute(h, ((2, 1), (4, 3))))
        H_periodic2 = FiniteMPOHamiltonian(physicalspace(H_periodic), terms)
        @test H_periodic ≈ H_periodic2
    end
end
