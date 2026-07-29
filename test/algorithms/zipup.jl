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

function _random_mpo_mps(pspace, Dspace)
    Random.seed!(1357)
    L = 6
    Wspace = Dspace
    Vspaces = [oneunit(Wspace); fill(Wspace, L - 1); oneunit(Wspace)]
    O = FiniteMPO(
        [rand(ComplexF64, Vspaces[i] ⊗ pspace ← pspace ⊗ Vspaces[i + 1]) for i in 1:L]
    )
    ψ = FiniteMPS(rand, ComplexF64, L, pspace, Dspace)
    return O, ψ
end

@testset "Finite MPO-MPS zip-up $(spacetype(pspace))" for (pspace, Dspace, _) in spacelist
    O, ψ = _random_mpo_mps(pspace, Dspace)
    O_copy = copy(O)
    ψ_copy = copy(ψ)

    trscheme = trunctol(; atol = 1.0e-10)
    ref = changebonds(O * ψ, SvdCut(; trscheme); normalize = false)
    got = approximate((O, ψ), Zipup(; trscheme))

    @test norm(ref - got) / norm(ref) < 1.0e-10
    @test norm(ψ - ψ_copy) < 1.0e-12
    @test all(i -> norm(O[i] - O_copy[i]) < 1.0e-12, 1:length(O))
end

@testset "Paeckel two-stage zip-up $(spacetype(pspace))" for (pspace, Dspace, Dcut) in spacelist
    O, ψ = _random_mpo_mps(pspace, Dspace)
    rtol = 1.0e-8
    final_trscheme = truncrank(Dcut) & truncerror(; rtol)
    zipup_trscheme = truncrank(2Dcut) & truncerror(; rtol = rtol / 10)

    ref_tr = changebonds(O * ψ, SvdCut(; trscheme = final_trscheme); normalize = false)
    got_one_sweep = approximate((O, ψ), Zipup(; trscheme = final_trscheme))
    alg_zipup = Zipup(; trscheme = zipup_trscheme).alg_zipup
    alg_zipdown = Zipup(; trscheme = final_trscheme).alg_zipup
    got_two_sweep = approximate((O, ψ), Zipup(alg_zipup, alg_zipdown))

    err_one_sweep = norm(ref_tr - got_one_sweep) / norm(ref_tr)
    err_two_sweep = norm(ref_tr - got_two_sweep) / norm(ref_tr)
    @test err_two_sweep < err_one_sweep / 2
    @test maximum(i -> dim(left_virtualspace(got_one_sweep, i)), 2:length(got_one_sweep)) <= Dcut
    @test maximum(i -> dim(left_virtualspace(got_two_sweep, i)), 2:length(got_two_sweep)) <= Dcut
end
