println("
----------------------
|   FiniteMPS tests   |
----------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using MPSKit: _transpose_front, _transpose_tail
using MPSKit: GeometryStyle, FiniteChainStyle
using TensorKit
using TensorKit: ℙ
using Adapt

@testset "FiniteMPS ($(sectortype(D)), $elt)" for (D, d, elt) in [
        (ℙ^10, ℙ^2, ComplexF64),
        (
            Rep[SU₂](1 => 1, 0 => 3),
            Rep[SU₂](0 => 1) * Rep[SU₂](0 => 1),
            ComplexF32,
        ),
    ]
    L = rand(3:20)
    ψ = FiniteMPS(rand, elt, L, d, D)

    @test eachindex(IndexLinear(), ψ) == eachindex(ψ)
    @test isfinite(ψ)
    @test isfinite(typeof(ψ))
    @test isfinite(ψ) == isfinite(typeof(ψ))
    @test GeometryStyle(typeof(ψ)) == FiniteChainStyle()
    @test GeometryStyle(ψ) == FiniteChainStyle()
    @test @constinferred physicalspace(ψ) == fill(d, L)
    @test all(x -> x ≾ D, @constinferred left_virtualspace(ψ))
    @test all(x -> x ≾ D, @constinferred right_virtualspace(ψ))

    @test eltype(ψ) == eltype(typeof(ψ))

    ovl = dot(ψ, ψ)

    @test ovl ≈ norm(ψ.AC[1])^2

    for i in 1:length(ψ)
        @test ψ.AC[i] ≈ ψ.AL[i] * ψ.C[i]
        @test ψ.AC[i] ≈ _transpose_front(ψ.C[i - 1] * _transpose_tail(ψ.AR[i]))
    end

    @test elt == scalartype(ψ)

    ψ = ψ * 3
    @test ovl * 9 ≈ norm(ψ)^2
    ψ = 3 * ψ
    @test ovl * 9 * 9 ≈ norm(ψ)^2

    @test norm(2 * ψ + ψ - 3 * ψ) ≈ 0.0 atol = sqrt(eps(real(elt)))
end

@testset "FiniteMPS ($(sectortype(D)), $elt)" for (D, d, elt) in [
        (ℙ^10, ℙ^2, ComplexF64),
        (
            Rep[U₁](-1 => 3, 0 => 3, 1 => 3),
            Rep[U₁](-1 => 1, 0 => 1, 1 => 1),
            ComplexF64,
        ),
    ]
    ψ_small = FiniteMPS(rand, elt, 4, d, D)
    ψ_small2 = FiniteMPS(convert(TensorMap, ψ_small))
    @test dot(ψ_small, ψ_small2) ≈ dot(ψ_small, ψ_small)

    ψ′ = @constinferred complex(ψ_small)
    @test scalartype(ψ′) <: Complex
    if elt <: Complex
        @test ψ_small === ψ′
    else
        @test norm(ψ_small) ≈ norm(ψ′)
        @test complex(convert(TensorMap, ψ_small)) ≈ convert(TensorMap, ψ′)
    end
end

@testset "FiniteMPS center + (slice) indexing" begin
    L = 11
    ψ = FiniteMPS(L, ℂ^2, ℂ^16)

    ψ.AC[6] # moving the center to site 6

    @test ψ.center == 6

    @test ψ[5] == ψ.ALs[5]
    @test ψ[6] == ψ.ACs[6]
    @test ψ[7] == ψ.ARs[7]

    @test ψ[5:7] == [ψ.ALs[5], ψ.ACs[6], ψ.ARs[7]]

    @inferred ψ[5]

    @test_throws BoundsError ψ[0]
    @test_throws BoundsError ψ[L + 1]

    ψ.C[6] = randn(ComplexF64, space(ψ.C[6])) # setting the center between sites 6 and 7
    @test ψ.center == 13 / 2
    @test ψ[5:7] == [ψ.ALs[5], ψ.ACs[6], ψ.ARs[7]]
end

@testset "FiniteMPS copying" begin
    L = 10
    mps1 = FiniteMPS(rand, ComplexF64, L, ℂ^2, ℂ^5)
    mps2 = copy(mps1)

    @test mps1 !== mps2

    # arrays are distinct
    @test mps1.ALs !== mps2.ALs
    @test mps1.ARs !== mps2.ARs
    @test mps1.ACs !== mps2.ACs
    @test mps1.Cs !== mps2.Cs

    # tensors are distinct but equal
    for i in 1:L
        @test (ismissing(mps1.ALs[i]) && ismissing(mps2.ALs[i])) ||
            (mps1.ALs[i] !== mps2.ALs[i] && mps1.ALs[i] == mps2.ALs[i])
        @test (ismissing(mps1.ARs[i]) && ismissing(mps2.ARs[i])) ||
            (mps1.ARs[i] !== mps2.ARs[i] && mps1.ARs[i] == mps2.ARs[i])
        @test (ismissing(mps1.ACs[i]) && ismissing(mps2.ACs[i])) ||
            (mps1.ACs[i] !== mps2.ACs[i] && mps1.ACs[i] == mps2.ACs[i])
        @test (ismissing(mps1.Cs[i]) && ismissing(mps2.Cs[i])) ||
            (mps1.Cs[i] !== mps2.Cs[i] && mps1.Cs[i] == mps2.Cs[i])
    end
    @test (ismissing(mps1.Cs[end]) && ismissing(mps2.Cs[end])) ||
        mps1.Cs[end] !== mps2.Cs[end]
end

@testset "FiniteMPS entropy ($(sectortype(D)), $elt)" for (D, d, elt) in [
        (ℙ^10, ℙ^2, ComplexF64),
        (
            Rep[U₁](-1 => 3, 0 => 3, 1 => 3),
            Rep[U₁](-1 => 1, 0 => 1, 1 => 1),
            ComplexF64,
        ),
    ]
    L = 6
    ψ = FiniteMPS(rand, elt, L, d, D)

    # entropy is non-negative at all sites
    for site in 1:L
        @test real(entropy(ψ, site)) >= 0
    end

    # entropy is consistent with entanglement_spectrum
    for site in 1:L
        @test entropy(ψ, site) ≈ entropy(entanglement_spectrum(ψ, site))
    end

    # entropy is zero at the right boundary (trivial bond)
    @test entropy(ψ, L) ≈ 0 atol = 1.0e-10

    # product state has zero entropy everywhere
    ψ_product = FiniteMPS(rand, elt, L, d, oneunit(D))
    for site in 1:L
        @test entropy(ψ_product, site) ≈ 0 atol = 1.0e-10
    end
end
