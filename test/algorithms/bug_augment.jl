println("
------------------------------------
|   BUG basis-augmentation tests   |
------------------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using MPSKit: _bug_augment_left, _bug_augment_right, _transpose_tail, _transpose_front,
    left_orth, right_gauge
using TensorKit
using TensorKit: ℙ
using LinearAlgebra: I, norm
using Random

# The augment helpers keep the OLD isometry `U₀` as the leading per-sector block and append the
# component of the evolved candidate `K₁` that is orthogonal to it (no truncation). The four core
# properties (checked below for both sweep directions, trivial + U(1)):
#   1. isometry               `Û' Û ≈ 𝟙`
#   2. old-first, per sector  `M = Û' U₀ ≈ [𝟙; 0]` block-by-block
#   3. range ⊇ old, candidate `Û (Û' U₀) ≈ U₀` and `Û (Û' K₁) ≈ K₁`
#   4. rank growth            `dim(Vr₀) < dim(V̂) ≤ 2·dim(Vr₀)`

# Assert the per-sector "old-first" invariant on the overlap `M` (codomain = augmented bond,
# domain = old bond `V_old`): every sector block must be `[𝟙; 0]`, i.e. the leading
# `dim(V_old, c)` rows are the identity and the rest vanish. Iterating `blocks(M)` visits exactly
# the sectors common to `M`'s (co)domain, so purely-new sectors (absent from `V_old`) are correctly
# skipped here.
function _check_old_first(M, V_old; tol = 1.0e-10)
    for (c, b) in blocks(M)
        r0 = dim(V_old, c)
        @test b[1:r0, :] ≈ I
        if r0 < size(b, 1)
            @test norm(b[(r0 + 1):end, :]) < tol
        end
    end
    return nothing
end

function check_augment_left(U₀, K₁; tol = 1.0e-10)
    Û = _bug_augment_left(U₀, K₁)
    M = Û' * U₀                      # old bond's coordinates in the augmented basis (V̂ ← Vr₀)
    # 1. isometry
    @test Û' * Û ≈ one(Û' * Û)
    # 2. old-first, per sector (M : V̂ ← Vr₀)
    _check_old_first(M, domain(U₀)[1]; tol)
    # 3. range captures both the old basis and the evolved candidate
    @test Û * (Û' * U₀) ≈ U₀
    @test Û * (Û' * K₁) ≈ K₁
    @test Û * M ≈ U₀                 # consistency of the returned overlap
    # 4. rank growth: strictly bigger than the old bond, at most doubled
    r = dim(domain(U₀))
    @test r < dim(domain(Û)) ≤ 2r
    return Û, M
end

function check_augment_right(U₀, K₁; tol = 1.0e-10)
    Û = _bug_augment_right(U₀, K₁)
    ût = _transpose_tail(Û)          # V̂ ← P ⊗ Vr, right-isometric (row space)
    u0t = _transpose_tail(U₀)
    k1t = _transpose_tail(K₁)
    M = ût * u0t'                    # old bond's coordinates in the augmented basis (V̂ ← Vl₀)
    # 1. isometry (right-canonical ⇒ tail has orthonormal rows)
    @test ût * ût' ≈ one(ût * ût')
    # 2. old-first, per sector (M : V̂ ← Vl₀)
    _check_old_first(M, codomain(u0t)[1]; tol)
    # 3. row space captures both the old basis and the evolved candidate
    @test (u0t * ût') * ût ≈ u0t
    @test (k1t * ût') * ût ≈ k1t
    @test M' * ût ≈ u0t              # consistency of the returned overlap
    # 4. rank growth on the left bond
    r = dim(codomain(u0t))
    @test r < dim(codomain(ût)) ≤ 2r
    return Û, M
end

@testset "BUG basis augmentation" verbose = true begin
    # -------------------------------------------------------------------------------------------
    # Trivial (dense) tensors, ComplexF64 to exercise the adjoints.
    # -------------------------------------------------------------------------------------------
    @testset "trivial tensors" begin
        Random.seed!(20260707)
        Vl = ℂ^2
        P = ℂ^3
        Vr = ℂ^2

        # left→right: augment the RIGHT bond (domain of a left-isometry)
        U₀_L, _ = left_orth(randn(ComplexF64, Vl ⊗ P ← Vr))   # Vl⊗P ← Vr, left-isometric
        K₁_L = randn(ComplexF64, Vl ⊗ P ← Vr)                 # generic evolved candidate
        Û_L, _ = check_augment_left(U₀_L, K₁_L)

        # right→left: augment the LEFT bond (codomain of a right-isometry)
        _, U₀_R = right_gauge(randn(ComplexF64, Vl ⊗ P ← Vr))  # V ⊗ P ← Vr, right-isometric
        K₁_R = randn(ComplexF64, Vl ⊗ P ← Vr)
        Û_R, _ = check_augment_right(U₀_R, K₁_R)
    end

    # -------------------------------------------------------------------------------------------
    # U(1)-symmetric tensors: per-sector direct sums, old-first sector-by-sector, and a genuinely
    # new sector introduced by the candidate on the left augment.
    # -------------------------------------------------------------------------------------------
    @testset "U(1) symmetric tensors" begin
        Random.seed!(31415926)

        # left→right, WITH a new sector: `Vr₀` deliberately omits sector 0, which `Vl ⊗ P`
        # contains and the candidate `K₁` populates ⇒ augmentation must add it to `V̂`.
        Vl = U1Space(0 => 1, 1 => 1)
        P = U1Space(0 => 1, 1 => 1)           # Vl⊗P : 0=>1, 1=>2, 2=>1
        Vr0 = U1Space(1 => 1, 2 => 1)         # omits sector 0
        Vr0K = U1Space(0 => 1, 1 => 1, 2 => 1) # candidate populates sector 0

        U₀_L, _ = left_orth(randn(ComplexF64, Vl ⊗ P ← Vr0))
        @test domain(U₀_L)[1] == Vr0          # left_orth kept the full old bond
        K₁_L = randn(ComplexF64, Vl ⊗ P ← Vr0K)
        Û_L, _ = check_augment_left(U₀_L, K₁_L)

        # the augmented bond is a per-sector direct sum that gained sector 0
        V̂_L = domain(Û_L)[1]
        @test !(U1Irrep(0) in sectors(Vr0))
        @test U1Irrep(0) in sectors(V̂_L)
        # every old sector survives with at least its old multiplicity (old-first ⇒ nothing dropped)
        for c in sectors(Vr0)
            @test dim(V̂_L, c) ≥ dim(Vr0, c)
        end

        # right→left augment on a compatible symmetric configuration
        Plr = U1Space(0 => 1, 1 => 1)
        Vrr = U1Space(0 => 1, 1 => 1)
        Vl0 = U1Space(0 => 1, 1 => 1)
        _, U₀_R = right_gauge(randn(ComplexF64, Vl0 ⊗ Plr ← Vrr))
        Vl_K = U1Space(0 => 1, 1 => 1)
        K₁_R = randn(ComplexF64, Vl_K ⊗ Plr ← Vrr)
        Û_R, _ = check_augment_right(U₀_R, K₁_R)
    end
end
