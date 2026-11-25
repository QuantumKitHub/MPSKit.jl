using ..TestSetup
using Test, TestExtras
using MPSKit
using MPSKit: GeometryStyle
using TensorKit
using Adapt

export mpo_dense_mpo

function mpo_dense_mpo(AT; fast = false)
    return @testset "DenseMPO ($AT)" for ham in (transverse_field_ising(), heisenberg_XXX(; spin = 1))
        ham = adapt(AT, ham)
        pspace = physicalspace(ham, 1)
        ou = rightunitspace(pspace)

        ψ = adapt(AT, InfiniteMPS([pspace], [ou ⊕ pspace]))

        W = MPSKit.DenseMPO(make_time_mpo(ham, 1im * 0.5, WII()))

        @test GeometryStyle(ψ, W) == GeometryStyle(ψ)
        @test W * (W * ψ) ≈ (W * W) * ψ atol = 1.0e-2 # TODO: there is a normalization issue here
    end
end
