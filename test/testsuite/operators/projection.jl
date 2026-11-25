using ..TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ
using Adapt

export operators_projection

function operators_projection(AT; fast = false)
    return @testset "ProjectionOperator ($AT)" begin
        L = 10
        psi = adapt(AT, FiniteMPS(rand, ComplexF64, L, ℙ^2, ℙ^2))
        psi2 = adapt(AT, FiniteMPS(rand, ComplexF64, L, ℙ^2, ℙ^2))
        O = MPSKit.ProjectionOperator(psi)

        @test expectation_value(psi, O) ≈ 1.0
        @test expectation_value(psi2, O) ≈ dot(psi, psi2) * dot(psi2, psi)
    end
end
