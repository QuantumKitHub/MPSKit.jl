println("
--------------------------
|   State arithmetic     |
--------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using MPSKit: _transpose_front, _transpose_tail, left_gauge
using TensorKit
using TensorKit: PlanarTrivial, ℙ

"""
    gauge_error(ψ::FiniteMPS)

Largest relative violation of the defining gauge relations `AL[i] * C[i] == AC[i] == C[i-1] * AR[i]`.
"""
function gauge_error(ψ::FiniteMPS)
    err = zero(real(scalartype(ψ)))
    for i in 1:length(ψ)
        AC = ψ.AC[i]
        err = max(err, norm(ψ.AL[i] * ψ.C[i] - AC) / norm(AC))
        err = max(
            err,
            norm(_transpose_front(ψ.C[i - 1] * _transpose_tail(ψ.AR[i])) - AC) / norm(AC)
        )
    end
    return err
end

"""
    flip_gauge!(ψ::FiniteMPS)

Re-gauge site 1 into an equally valid but different gauge (a sign flip) and materialise `AC[1]`.
This reproduces the cache state left behind by e.g. DMRG, which installs `AL, C` from a truncated
SVD while the lazy gauge sweep re-derives them from a positive QR.
"""
function flip_gauge!(ψ::FiniteMPS)
    AL, C, = left_gauge(ψ.AC[1])
    ψ.AC[1] = (-AL, -C)
    ψ.AC[1]
    return ψ
end

# (virtual space, second virtual space, physical space, scalartype)
const arithmetic_spaces = [
    (ℙ^8, ℙ^5, ℙ^2, ComplexF64),
    (
        Rep[U₁](-1 => 2, 0 => 2, 1 => 2), Rep[U₁](-1 => 1, 0 => 3, 1 => 1),
        Rep[U₁](-1 => 1, 0 => 1, 1 => 1), ComplexF64,
    ),
    # integer spin, so that a trivial-sector state exists for every chain length
    (
        Rep[SU₂](0 => 2, 1 => 2, 2 => 1), Rep[SU₂](0 => 1, 1 => 3),
        Rep[SU₂](1 => 1), ComplexF64,
    ),
]

@testset "lazy gauging is a cache ($(sectortype(D)))" for (D, _, d, elt) in arithmetic_spaces
    for L in (4, 6)
        # sweeping right must not change tensors already handed out
        ψ = flip_gauge!(FiniteMPS(rand, elt, L, d, D))
        n = norm(ψ)
        AL₁, C₀ = copy(ψ.AL[1]), copy(ψ.C[1])
        ψ.AL[2] # triggers a gauge-center move
        @test ψ.AL[1] ≈ AL₁
        @test ψ.C[1] ≈ C₀
        @test norm(ψ) ≈ n
        @test gauge_error(ψ) ≤ sqrt(eps(real(elt)))

        # ... and neither must sweeping back left
        ϕ = flip_gauge!(FiniteMPS(rand, elt, L, d, D))
        ϕ.C[L - 1]
        ARₗ, Cₗ = copy(ϕ.AR[L]), copy(ϕ.C[L - 1])
        ϕ.AR[2]
        @test ϕ.AR[L] ≈ ARₗ
        @test ϕ.C[L - 1] ≈ Cₗ
        @test gauge_error(ϕ) ≤ sqrt(eps(real(elt)))
    end
end

@testset "FiniteMPS ± dense cross-check ($(sectortype(D)), L=$L)" for (D, D₂, d, elt) in
        arithmetic_spaces,
        # `convert(TensorMap, ψ)` compiles per (sectortype, length), so only sweep `L` once.
        # `L = 1` is only swept for the trivial sector: a single spin-1 site carries no singlet.
        L in (sectortype(D) === PlanarTrivial ? (1, 2, 3, 4, 5) : (4,))

    ψ₁ = FiniteMPS(rand, elt, L, d, D)
    ψ₂ = FiniteMPS(rand, elt, L, d, D)
    ψ₃ = FiniteMPS(rand, elt, L, d, D₂) # mismatched bond dimensions
    t₁, t₂, t₃ = convert.(TensorMap, (ψ₁, ψ₂, ψ₃))
    atol = sqrt(eps(real(elt)))

    @test convert(TensorMap, ψ₁ + ψ₂) ≈ t₁ + t₂ atol = atol
    @test convert(TensorMap, ψ₁ - ψ₂) ≈ t₁ - t₂ atol = atol
    @test convert(TensorMap, ψ₁ + ψ₃) ≈ t₁ + t₃ atol = atol
    @test convert(TensorMap, ψ₃ - ψ₁) ≈ t₃ - t₁ atol = atol
    @test norm(ψ₁ + ψ₂) ≈ norm(t₁ + t₂) atol = atol
    @test dot(ψ₁ + ψ₂, ψ₁ - ψ₂) ≈ dot(t₁ + t₂, t₁ - t₂) atol = atol

    # scaling is distributive over addition
    @test convert(TensorMap, 2 * ψ₁ + 3 * ψ₂) ≈ 2 * t₁ + 3 * t₂ atol = atol
end

@testset "FiniteMPS argument checks" begin
    @test_throws DimensionMismatch FiniteMPS(rand, ComplexF64, 4, ℂ^2, ℂ^4) +
        FiniteMPS(rand, ComplexF64, 5, ℂ^2, ℂ^4)
    # mismatched boundary virtual spaces, at length 1 and beyond
    onesite(V₁, V₂) = FiniteMPS(
        rand, ComplexF64, [ℂ^2], ComplexSpace[]; left = V₁, right = V₂, normalize = false
    )
    @test_throws SpaceMismatch onesite(ℂ^3, ℂ^1) + onesite(ℂ^2, ℂ^1)
    @test_throws SpaceMismatch onesite(ℂ^1, ℂ^3) + onesite(ℂ^1, ℂ^2)
end

@testset "FiniteMPS cancellation across gauges ($(sectortype(D)))" for (D, _, d, elt) in
    arithmetic_spaces
    # same vector, different tensor network: re-canonicalised through a dense tensor
    ψ₄ = FiniteMPS(rand, elt, 4, d, D)
    @test norm(ψ₄ - FiniteMPS(convert(TensorMap, ψ₄))) ≈ 0 atol =
        sqrt(eps(real(elt))) * norm(ψ₄)

    for L in 2:6
        ψ = FiniteMPS(rand, elt, L, d, D)
        atol = sqrt(eps(real(elt))) * norm(ψ)

        # same vector, different *gauge history* -- this is issue #473
        χ = flip_gauge!(copy(ψ))
        @test norm(ψ - χ) ≈ 0 atol = atol
        @test norm(χ - ψ) ≈ 0 atol = atol
        @test norm(2 * ψ - χ - ψ) ≈ 0 atol = atol
        @test norm(3 * ψ - χ) ≈ 2 * norm(ψ) atol = atol
    end
end

@testset "eigenstate cancellation, issue #473 (L=$L)" for L in (4, 6)
    # `gs` is an eigenstate of `H`, so `E₀ * gs - H * gs` must be the zero vector, even though the
    # two operands are carried by completely different tensor networks.
    H = transverse_field_ising(; g = 1.0, L = L)
    ψ₀ = FiniteMPS(rand, ComplexF64, L, physicalspace(H, 1), ℂ^16)
    gs, envs, = find_groundstate(ψ₀, H, DMRG(; verbosity = 0))
    E₀ = real(expectation_value(gs, H, envs))
    atol = 1.0e-8 * abs(E₀)

    a = E₀ * gs
    b = H * gs
    @test convert(TensorMap, a) ≈ convert(TensorMap, b) atol = atol # the operands agree

    @test norm(a - b) ≈ 0 atol = atol
    @test norm(convert(TensorMap, a - b)) ≈ 0 atol = atol
    @test norm(b - a) ≈ 0 atol = atol

    # ... independent of whether the operands were gauged before the subtraction
    a′, b′ = E₀ * gs, H * gs
    convert(TensorMap, a′)
    convert(TensorMap, b′)
    @test norm(a′ - b′) ≈ 0 atol = atol

    # a non-trivial linear combination is still correct
    @test norm(3 * (E₀ * gs) - H * gs) ≈ 2 * abs(E₀) * norm(gs) atol = atol
end

@testset "single-site addition" begin
    atol = sqrt(eps(Float64))

    # trivial boundaries: cross-check against the dense state
    ψ₁ = FiniteMPS(rand, ComplexF64, 1, ℂ^3, ℂ^4)
    ψ₂ = FiniteMPS(rand, ComplexF64, 1, ℂ^3, ℂ^4)
    t₁, t₂ = convert(TensorMap, ψ₁), convert(TensorMap, ψ₂)
    @test length(ψ₁ + ψ₂) == 1
    @test convert(TensorMap, ψ₁ + ψ₂) ≈ t₁ + t₂ atol = atol
    @test convert(TensorMap, ψ₁ - ψ₂) ≈ t₁ - t₂ atol = atol
    @test norm(ψ₁ + ψ₂) ≈ norm(t₁ + t₂) atol = atol
    @test norm(ψ₁ - ψ₁) ≈ 0 atol = atol
    @test gauge_error(ψ₁ + ψ₂) ≤ atol

    # non-trivial boundaries, e.g. a single-site window carved out of a larger chain. There is no
    # dense state to compare against, but the whole state *is* the one center tensor.
    onesite(V₁, V₂) = FiniteMPS(
        rand, ComplexF64, [ℂ^2], ComplexSpace[]; left = V₁, right = V₂, normalize = false
    )
    ϕ₁, ϕ₂ = onesite(ℂ^3, ℂ^4), onesite(ℂ^3, ℂ^4)
    AC₁, AC₂ = copy(ϕ₁.AC[1]), copy(ϕ₂.AC[1])
    @test (ϕ₁ + ϕ₂).AC[1] ≈ AC₁ + AC₂ atol = atol
    @test (ϕ₁ - ϕ₂).AC[1] ≈ AC₁ - AC₂ atol = atol
    @test left_virtualspace(ϕ₁ + ϕ₂, 1) == ℂ^3
    @test right_virtualspace(ϕ₁ + ϕ₂, 1) == ℂ^4
    @test norm(ϕ₁ + ϕ₂) ≈ norm(AC₁ + AC₂) atol = atol
    @test gauge_error(ϕ₁ + ϕ₂) ≤ atol

    # the operands are left untouched
    @test ϕ₁.AC[1] ≈ AC₁ && ϕ₂.AC[1] ≈ AC₂

    # ... and the same for a single-site `FiniteMPO`
    o₁, o₂ = rand(ComplexF64, ℂ^3 ← ℂ^3), rand(ComplexF64, ℂ^3 ← ℂ^3)
    O₁, O₂ = FiniteMPO(o₁), FiniteMPO(o₂)
    @test length(O₁) == 1
    # `convert` on a single-site MPO used to contract `mpo[1]` with `mpo[end]`, the same tensor
    @test convert(TensorMap, O₁) ≈ o₁ atol = atol
    @test length(O₁ + O₂) == 1
    @test (O₁ + O₂)[1] ≈ O₁[1] + O₂[1] atol = atol
    @test (O₁ - O₂)[1] ≈ O₁[1] - O₂[1] atol = atol
    @test convert(TensorMap, O₁ + O₂) ≈ o₁ + o₂ atol = atol
    @test convert(TensorMap, O₁ - O₂) ≈ o₁ - o₂ atol = atol

    # scalar types promote at length 1 too
    @test scalartype(
        FiniteMPS(rand, Float64, 1, ℂ^3, ℂ^4) + FiniteMPS(rand, ComplexF64, 1, ℂ^3, ℂ^4)
    ) ==
        ComplexF64
    @test scalartype(FiniteMPO(rand(Float64, ℂ^3 ← ℂ^3)) + FiniteMPO(rand(ComplexF64, ℂ^3 ← ℂ^3))) ==
        ComplexF64
end

@testset "scalar-type promotion" begin
    L = 4

    ψᵣ = FiniteMPS(rand, Float64, L, ℂ^2, ℂ^4)
    ψᶜ = FiniteMPS(rand, ComplexF64, L, ℂ^2, ℂ^4)
    tᵣ, tᶜ = convert(TensorMap, ψᵣ), convert(TensorMap, ψᶜ)

    @test scalartype(ψᵣ + ψᶜ) == ComplexF64
    @test scalartype(ψᶜ + ψᵣ) == ComplexF64
    @test convert(TensorMap, ψᵣ + ψᶜ) ≈ tᵣ + tᶜ
    @test convert(TensorMap, ψᶜ + ψᵣ) ≈ tᶜ + tᵣ
    @test scalartype(ψᵣ - ψᶜ) == ComplexF64
    @test convert(TensorMap, ψᵣ - ψᶜ) ≈ tᵣ - tᶜ
    # both operands real stays real
    @test scalartype(ψᵣ + ψᵣ) == Float64

    Oᵣ = FiniteMPO(rand(Float64, (ℂ^2)^L ← (ℂ^2)^L))
    Oᶜ = FiniteMPO(rand(ComplexF64, (ℂ^2)^L ← (ℂ^2)^L))
    oᵣ, oᶜ = convert(TensorMap, Oᵣ), convert(TensorMap, Oᶜ)

    @test scalartype(Oᵣ + Oᶜ) == ComplexF64
    @test scalartype(Oᶜ + Oᵣ) == ComplexF64
    @test convert(TensorMap, Oᵣ + Oᶜ) ≈ oᵣ + oᶜ
    @test convert(TensorMap, Oᶜ + Oᵣ) ≈ oᶜ + oᵣ
    @test scalartype(Oᵣ + Oᵣ) == Float64
end
