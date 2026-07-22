println("
-----------------------------
|      Zipper tests          |
-----------------------------
")

using .TestSetup
using Test
using MPSKit
using TensorKit
using TensorKit: ℙ
using Random

spacelist = [(ℙ^4, ℙ^3), (Rep[SU₂](1 => 1), Rep[SU₂](0 => 2, 1 => 2, 2 => 1))]

@testset "Finite MPO-MPS zipper $(spacetype(pspace))" for (pspace, Dspace) in spacelist
    Random.seed!(1357)
    L = 6
    Wspace = Dspace
    Vspaces = [oneunit(Wspace); fill(Wspace, L - 1); oneunit(Wspace)]
    O = FiniteMPO(
        [rand(ComplexF64, Vspaces[i] ⊗ pspace ← pspace ⊗ Vspaces[i + 1]) for i in 1:L]
    )
    ψ = FiniteMPS(rand, ComplexF64, L, pspace, Dspace)
    O_copy = copy(O)
    ψ_copy = copy(ψ)

    trscheme = trunctol(; atol = 1.0e-10)
    ref = changebonds(O * ψ, SvdCut(; trscheme); normalize = false)
    got = approximate((O, ψ), Zipper(; trscheme))

    @test norm(ref - got) / norm(ref) < 1.0e-10
    @test norm(ψ - ψ_copy) < 1.0e-12
    @test all(i -> norm(O[i] - O_copy[i]) < 1.0e-12, 1:length(O))

    Dcut = 4
    got_tr = approximate((O, ψ), Zipper(; trscheme = truncrank(Dcut)))
    @test maximum(i -> dim(left_virtualspace(got_tr, i)), 2:length(got_tr)) <= Dcut
end
