println("
----------------------------
|   MultilineMPO tests      |
----------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using TensorKit

d = 3
P = ℂ^d

# ψ[i] = |i⟩ (3 rows, 1 column), O[i]|i⟩ = |i + 1⟩ exactly
# ψ[i] is an InfiniteMPS and is periodic in the line index i
# this properly tests row-shift convention: row i of a MultilineMPO maps row i of the network onto row i + 1
ψ = MultilineMPS([product_mps(1), product_mps(2), product_mps(3)])
O = MultilineMPO(
    [
        perm_mpo([2, 3, 1]), # row 1: |1⟩ -> |2⟩
        perm_mpo([1, 3, 2]), # row 2: |2⟩ -> |3⟩
        perm_mpo([3, 2, 1]), # row 3: |3⟩ -> |1⟩
    ]
)

@testset "MultilineMPO * InfiniteMPS" begin
    for i in 1:3
        @test abs(dot(ψ[i + 1], O[i] * ψ[i])) ≈ 1 atol = 1.0e-10
        @test abs(dot(ψ[i], O[i] * ψ[i])) ≈ 0 atol = 1.0e-10
    end

    @test abs(dot(ψ[1], O * ψ[1])) ≈ 1 atol = 1.0e-10
    @test abs(dot(ψ[2], O * ψ[2])) ≈ 1 atol = 1.0e-10 # cyclic period

    # the order matters, here testing reverse order not returning to ψ[1]
    reversed = foldl((st, i) -> O[i] * st, 3:-1:1; init = ψ[1])
    @test abs(dot(ψ[1], reversed)) ≈ 0 atol = 1.0e-10
end

@testset "dominant_eigenvalue" begin
    @test dominant_eigenvalue(ψ, O) ≈ 1 atol = 1.0e-10

    # row-accumulated eigenvalues
    scaled(mpo, c) = InfiniteMPO([c * mpo[i] for i in 1:length(mpo)])
    cs = (2.0, 3.0, 5.0)
    O_rows = MultilineMPO([scaled(O[i], cs[i]) for i in 1:3])
    @test dominant_eigenvalue(ψ, O_rows) ≈ prod(cs) atol = 1.0e-8

    # column-accumulated eigenvalues
    id_mpo = perm_mpo([1, 2, 3]) # identity operator, so anything's a fixed point
    cs = (2.0, 3.0)
    O_cols = MultilineMPO([InfiniteMPO([cs[1] * id_mpo[1], cs[2] * id_mpo[1]])])
    ψ_cols = MultilineMPS([InfiniteMPS([P, P], [ℂ^2, ℂ^2])])
    @test dominant_eigenvalue(ψ_cols, O_cols) ≈ prod(cs) atol = 1.0e-8
end
