println("
--------------------
|   DenseMPO tests  |
--------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using MPSKit: GeometryStyle
using TensorKit

@testset "DenseMPO" for ham in (transverse_field_ising(), heisenberg_XXX(; spin = 1))
    pspace = physicalspace(ham, 1)
    ou = rightunitspace(pspace)

    ψ = InfiniteMPS([pspace], [ou ⊕ pspace])

    W = MPSKit.DenseMPO(make_time_mpo(ham, 1im * 0.5, WII()))

    @test GeometryStyle(ψ, W) == GeometryStyle(ψ)
    @test W * (W * ψ) ≈ (W * W) * ψ atol = 1.0e-2 # TODO: there is a normalization issue here
end
