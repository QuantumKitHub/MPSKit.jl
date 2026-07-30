println("
-----------------------------
|      Zipup tests           |
-----------------------------
")

using .TestSetup
using Test
using MPSKit
using TensorKit
using TensorKit: ℙ
using Random

spacelist = [
    (ℙ^4, ℙ^3, 4),
    (Rep[SU₂](1 => 1), Rep[SU₂](0 => 2, 1 => 2, 2 => 1), 8),
]

function _random_mpo_mps(pspace, Dspace; elt = ComplexF64)
    Random.seed!(1357)
    L = 6
    Wspace = Dspace
    Vspaces = [oneunit(Wspace); fill(Wspace, L - 1); oneunit(Wspace)]
    O = FiniteMPO(
        [rand(ComplexF64, Vspaces[i] ⊗ pspace ← pspace ⊗ Vspaces[i + 1]) for i in 1:L]
    )
    ψ = FiniteMPS(rand, elt, L, pspace, Dspace)
    return O, ψ
end

@testset "Finite MPO-MPS zip-up $(spacetype(pspace))" for (pspace, Dspace, _) in spacelist
    O, ψ = _random_mpo_mps(pspace, Dspace)
    O_copy = copy(O)
    ψ_copy = copy(ψ)

    trunc = trunctol(; atol = 1.0e-10)
    ref = changebonds(O * ψ, SvdCut(; trunc); normalize = false)

    # both sweep directions, with and without the zip-down pass
    for left_to_right in (true, false), trunc′ in (trunc, (notrunc(), trunc))
        got, ϵ = approximate((O, ψ), Zipup(; trunc = trunc′, left_to_right))
        @test norm(ref - got) / norm(ref) < 1.0e-10
        @test ϵ < 1.0e-10
    end

    @test norm(ψ - ψ_copy) < 1.0e-12
    @test all(i -> norm(O[i] - O_copy[i]) < 1.0e-12, 1:length(O))
end

@testset "Paeckel two-stage zip-up $(spacetype(pspace)), left_to_right = $left_to_right" for
    (pspace, Dspace, Dcut) in spacelist, left_to_right in (true, false)
    O, ψ = _random_mpo_mps(pspace, Dspace)
    rtol = 1.0e-8
    final_trunc = truncrank(Dcut) & truncerror(; rtol)
    zipup_trunc = truncrank(2Dcut) & truncerror(; rtol = rtol / 10)

    ref_tr = changebonds(O * ψ, SvdCut(; trunc = final_trunc); normalize = false)
    got_one_sweep, ϵ_one_sweep = approximate((O, ψ), Zipup(; trunc = final_trunc, left_to_right))
    got_two_sweep, _ = approximate(
        (O, ψ), Zipup(; trunc = (zipup_trunc, final_trunc), left_to_right)
    )
    @test ϵ_one_sweep > 0

    err_one_sweep = norm(ref_tr - got_one_sweep) / norm(ref_tr)
    err_two_sweep = norm(ref_tr - got_two_sweep) / norm(ref_tr)
    @test err_two_sweep < err_one_sweep / 2
    @test maximum(i -> dim(left_virtualspace(got_one_sweep, i)), 2:length(got_one_sweep)) <= Dcut
    @test maximum(i -> dim(left_virtualspace(got_two_sweep, i)), 2:length(got_two_sweep)) <= Dcut
end

@testset "In-place zip-up $(spacetype(pspace)), left_to_right = $left_to_right" for
    (pspace, Dspace, Dcut) in spacelist, left_to_right in (true, false)
    O, ψ = _random_mpo_mps(pspace, Dspace)
    alg = Zipup(; trunc = (truncrank(2Dcut), truncrank(Dcut)), left_to_right)
    ref, ϵ_ref = approximate((O, ψ), alg)

    # empty destination
    dst = similar(ψ, ComplexF64)
    got, ϵ = approximate!(dst, (O, ψ), alg)
    @test got === dst
    @test norm(ref - got) / norm(ref) < 1.0e-12
    @test ϵ ≈ ϵ_ref

    # a destination with unrelated contents is overwritten entirely
    dst = FiniteMPS(rand, ComplexF64, length(ψ), pspace, oneunit(Dspace) ⊕ Dspace ⊕ Dspace)
    got, ϵ = approximate!(dst, (O, ψ), alg)
    @test norm(ref - got) / norm(ref) < 1.0e-12
    @test ϵ ≈ ϵ_ref

    # the input may serve as its own destination
    got, ϵ = approximate!(ψ, (O, ψ), alg)
    @test got === ψ
    @test norm(ref - got) / norm(ref) < 1.0e-12
    @test ϵ ≈ ϵ_ref
end

@testset "Zip-up with non-trivial boundary spaces $(spacetype(pspace))" for (pspace, Dspace, _) in spacelist
    Random.seed!(1357)
    L = 4
    Vspaces = fill(Dspace, L + 1)
    O = FiniteMPO(
        [rand(ComplexF64, Vspaces[i] ⊗ pspace ← pspace ⊗ Vspaces[i + 1]) for i in 1:L]
    )
    ψ = FiniteMPS(rand, ComplexF64, L, pspace, Dspace; left = Dspace, right = Dspace)

    trunc = trunctol(; atol = 1.0e-12)
    got, _ = approximate((O, ψ), Zipup(; trunc))
    got_two_sweep, _ = approximate((O, ψ), Zipup(; trunc = (notrunc(), trunc)))

    # the boundary virtual spaces of the product are the fused ones, in both variants
    for ψ′ in (got, got_two_sweep)
        @test left_virtualspace(ψ′, 1) == fuse(left_virtualspace(ψ, 1) ⊗ left_virtualspace(O, 1))
        @test right_virtualspace(ψ′, L) == fuse(right_virtualspace(ψ, L) ⊗ right_virtualspace(O, L))
    end
    @test norm(got - got_two_sweep) / norm(got) < 1.0e-10
    @test norm(got) ≈ norm(O * ψ)
end

@testset "Zip-up with non-square MPO $(spacetype(pspace))" for (pspace, Dspace, _) in spacelist
    Random.seed!(1357)
    L = 4
    pspace′ = pspace ⊕ oneunit(pspace) # output physical space, different from the input one
    Vspaces = [oneunit(Dspace); fill(Dspace, L - 1); oneunit(Dspace)]
    O = FiniteMPO(
        [rand(ComplexF64, Vspaces[i] ⊗ pspace′ ← pspace ⊗ Vspaces[i + 1]) for i in 1:L]
    )
    ψ = FiniteMPS(rand, ComplexF64, L, pspace, Dspace)

    trunc = trunctol(; atol = 1.0e-12)
    ref = O * ψ
    got, _ = approximate((O, ψ), Zipup(; trunc))
    got_two_sweep, _ = approximate((O, ψ), Zipup(; trunc = (notrunc(), trunc)))

    @test all(i -> physicalspace(got, i) == pspace′, 1:L)
    @test norm(ref - got) / norm(ref) < 1.0e-10
    @test norm(ref - got_two_sweep) / norm(ref) < 1.0e-10

    # the reverse product is not defined
    @test_throws SpaceMismatch approximate((O, got), Zipup(; trunc))
end

@testset "Zip-up scalar type promotion $(spacetype(pspace))" for (pspace, Dspace, _) in spacelist
    O, ψ = _random_mpo_mps(pspace, Dspace; elt = Float64)
    alg = Zipup(; trunc = trunctol(; atol = 1.0e-10))

    got, _ = approximate((O, ψ), alg)
    @test scalartype(got) === ComplexF64
    ref = changebonds(O * ψ, SvdCut(; trunc = trunctol(; atol = 1.0e-10)); normalize = false)
    @test norm(ref - got) / norm(ref) < 1.0e-10

    # a real destination cannot hold the complex result
    @test_throws ArgumentError approximate!(ψ, (O, ψ), alg)
end
