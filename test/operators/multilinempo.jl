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
V = ℂ^1

# ψ[i] = |i⟩ (3 rows, 1 column), O[i]|i⟩ = |i + 1⟩ exactly, so ψ is the exact λ = 1 fixed point of O
# this properly tests row-shift convention: row i of a MultilineMPO maps row i of the state onto row i + 1
ψ = MultilineMPS([product_mps(1), product_mps(2), product_mps(3)])
O = MultilineMPO(
    [
        perm_mpo([2, 3, 1]), # row 1: |1⟩ -> |2⟩
        perm_mpo([1, 3, 2]), # row 2: |2⟩ -> |3⟩
        perm_mpo([3, 2, 1]), # row 3: |3⟩ -> |1⟩
    ]
)

@testset "MultilineMPO * MultilineMPS" begin
    for i in 1:3
        @test abs(dot(ψ[i + 1], O[i] * ψ[i])) ≈ 1 atol = 1.0e-10
        @test abs(dot(ψ[i], O[i] * ψ[i])) ≈ 0 atol = 1.0e-10
    end

    Oψ = O * ψ
    @test length(parent(Oψ)) == 3 # not rows * cols lines
    @test size(Oψ) == size(ψ) == (3, 1)
    for i in 1:3
        @test abs(dot(ψ[i], Oψ[i])) ≈ 1 atol = 1.0e-10  # O * ψ reproduces ψ
    end
end

@testset "MultilineMPO * MultilineMPO" begin
    # O[i+1] * O[i] maps row i onto row i+2, O[i] * O[i] does not
    for i in 1:3
        @test abs(dot(ψ[i + 2], (O[i + 1] * O[i]) * ψ[i])) ≈ 1 atol = 1.0e-10
    end
    @test !isfinite(O)
    @test !isfinite(typeof(O))

    OO = O * O
    @test !isfinite(OO)
    @test length(parent(OO)) == 3
    @test size(OO) == (3, 1)
    for i in 1:3
        @test abs(dot(ψ[i + 2], OO[i] * ψ[i])) ≈ 1 atol = 1.0e-10
    end
end
