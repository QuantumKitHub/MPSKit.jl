using ..TestSetup
using Test, TestExtras
using MPSKit
using MPSKit: leading_eigenvalue
using TensorKit
using Adapt

export mpo_multiline_mpo_times_mps, mpo_multiline_leading_eigenvalue,
    mpo_multiline_multirow_fixed_point, mpo_multiline_unsupported_inputs,
    mpo_multiline_line_aliases, mpo_multiline_view_bounds

d = 3
P = ℂ^d

# ψ[i] = |i⟩ (3 rows, 1 column), O[i]|i⟩ = |i + 1⟩ exactly
# ψ[i] is an InfiniteMPS and is periodic in the line index i
# this properly tests row-shift convention: row i of a MultilineMPO maps row i of the network onto row i + 1
function multiline_ψ(AT)
    return adapt(AT, MultilineMPS([product_mps(1), product_mps(2), product_mps(3)]))
end
function multiline_O(AT)
    return adapt(
        AT, MultilineMPO(
            [
                perm_mpo([2, 3, 1]), # row 1: |1⟩ -> |2⟩
                perm_mpo([1, 3, 2]), # row 2: |2⟩ -> |3⟩
                perm_mpo([3, 2, 1]), # row 3: |3⟩ -> |1⟩
            ]
        )
    )
end

function mpo_multiline_mpo_times_mps(AT; fast = false)
    ψ = multiline_ψ(AT)
    O = multiline_O(AT)
    return @testset "MultilineMPO * InfiniteMPS ($AT)" begin
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
end

function mpo_multiline_leading_eigenvalue(AT; fast = false)
    ψ = multiline_ψ(AT)
    O = multiline_O(AT)
    return @testset "leading_eigenvalue ($AT)" begin
        @test leading_eigenvalue(ψ, O) ≈ 1 atol = 1.0e-10

        # row-accumulated eigenvalues
        scaled(mpo, c) = InfiniteMPO([c * mpo[i] for i in 1:length(mpo)])
        cs = (2.0, 3.0, 5.0)
        O_rows = MultilineMPO([scaled(O[i], cs[i]) for i in 1:3])
        @test leading_eigenvalue(ψ, O_rows) ≈ prod(cs) atol = 1.0e-8

        # column-accumulated eigenvalues
        id_mpo = adapt(AT, perm_mpo([1, 2, 3])) # identity operator, so anything's a fixed point
        cs = (2.0, 3.0)
        O_cols = MultilineMPO([InfiniteMPO([cs[1] * id_mpo[1], cs[2] * id_mpo[1]])])
        ψ_cols = adapt(AT, MultilineMPS([InfiniteMPS([P, P], [ℂ^2, ℂ^2])]))
        @test leading_eigenvalue(ψ_cols, O_cols) ≈ prod(cs) atol = 1.0e-8
    end
end

function mpo_multiline_multirow_fixed_point(AT; fast = false)
    return @testset "multi-row fixed point ($AT)" begin
        # stacking the same row twice squares the eigenvalue of the single-row fixed point
        O₁ = adapt(AT, classical_ising(; β = 0.5))
        alg = VUMPS(; tol = 1.0e-10, verbosity = 0)
        ψ₁, = leading_boundary(adapt(AT, InfiniteMPS(ℂ^2, ℂ^10)), O₁, alg)
        λ₁ = leading_eigenvalue(ψ₁, O₁)

        O₂ = MultilineMPO([O₁, O₁])
        ψ₂₀ = adapt(AT, MultilineMPS([InfiniteMPS(ℂ^2, ℂ^10), InfiniteMPS(ℂ^2, ℂ^10)]))
        ψ₂, = leading_boundary(ψ₂₀, O₂, alg)
        @test leading_eigenvalue(ψ₂, O₂) ≈ λ₁^2 rtol = 1.0e-8
    end
end

function mpo_multiline_unsupported_inputs(AT; fast = false)
    ψ = multiline_ψ(AT)
    O = multiline_O(AT)
    return @testset "unsupported inputs ($AT)" begin
        @test_throws MethodError expectation_value(ψ, O)

        # Hamiltonian lines are a valid `MultilineMPO`, but not a transfer operator
        H = adapt(AT, transverse_field_ising())
        ψ_H = adapt(AT, MultilineMPS([InfiniteMPS(ℂ^2, ℂ^4), InfiniteMPS(ℂ^2, ℂ^4)]))
        O_H = MultilineMPO([H, H])
        @test O_H isa MultilineMPO && !(O_H isa InfiniteMultilineMPO)
        @test_throws MethodError expectation_value(ψ_H, O_H)
        @test_throws MethodError leading_eigenvalue(ψ_H, O_H)

        ψ_fin = adapt(AT, MultilineMPS([FiniteMPS(4, P, ℂ^2)]))
        @test_throws MethodError leading_boundary(
            ψ_fin, adapt(AT, MultilineMPO([perm_mpo([1, 2, 3])])), VUMPS()
        )
    end
end

function mpo_multiline_line_aliases(AT; fast = false)
    ψ = multiline_ψ(AT)
    O = multiline_O(AT)
    return @testset "line aliases ($AT)" begin
        @test ψ isa InfiniteMultilineMPS
        @test !(ψ isa FiniteMultilineMPS)
        @test O isa InfiniteMultilineMPO
        ψ_fin = adapt(AT, MultilineMPS([FiniteMPS(4, P, ℂ^2)]))
        @test ψ_fin isa FiniteMultilineMPS && ψ_fin isa MultilineMPS
        O_fin = adapt(AT, MultilineMPO([FiniteMPO(fill(perm_mpo([1, 2, 3])[1], 4))]))
        @test O_fin isa FiniteMultilineMPO && O_fin isa MultilineMPO
    end
end

function mpo_multiline_view_bounds(AT; fast = false)
    ψ = multiline_ψ(AT)
    return @testset "view bounds ($AT)" begin
        # rows are periodic, and so are the columns of infinite lines
        @test checkbounds(Bool, ψ.AC, 0, 0) && checkbounds(Bool, ψ.C, 4, -1)
        ψ_fin = adapt(AT, MultilineMPS([FiniteMPS(4, P, ℂ^2), FiniteMPS(4, P, ℂ^2)]))
        @test checkbounds(Bool, ψ_fin.AC, 3, 4) && !checkbounds(Bool, ψ_fin.AC, 1, 5)
        @test checkbounds(Bool, ψ_fin.C, 1, 0) && !checkbounds(Bool, ψ_fin.C, 1, 5)
    end
end
