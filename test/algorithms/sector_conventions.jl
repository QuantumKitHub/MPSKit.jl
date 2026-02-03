println("
-----------------------------
|   Sector conventions       |
-----------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ
using LinearAlgebra: eigvals

@testset "Sector conventions" begin
    L = 4
    H = XY_model(U1Irrep; L)

    H_dense = convert(TensorMap, H)
    vals_dense = eigvals(H_dense)
    for v in values(vals_dense)
        sort!(v; by = real)
    end

    tol = 1.0e-18 # tolerance required to separate degenerate eigenvalues
    alg = MPSKit.Defaults.alg_eigsolve(; dynamic_tols = false, tol)

    maxVspaces = MPSKit.max_virtualspaces(physicalspace(H))
    gs, = find_groundstate(
        FiniteMPS(physicalspace(H), maxVspaces[2:(end - 1)]), H; verbosity = 0
    )
    E₀ = expectation_value(gs, H)
    @test E₀ ≈ first(vals_dense[unit(U1Irrep)])

    for (sector, vals) in pairs(vals_dense)
        # ED tests
        num = length(vals)
        E₀s, ψ₀s, info = exact_diagonalization(H; num, sector, alg)
        @test E₀s[1:num] ≈ vals[1:num]
        # this is a trick to make the mps full-rank again, which is not guaranteed by ED
        ψ₀ = changebonds(first(ψ₀s), SvdCut(; trscheme = notrunc()))
        Vspaces = left_virtualspace.(Ref(ψ₀), 1:L)
        push!(Vspaces, right_virtualspace(ψ₀, L))
        @test all(splat(==), zip(Vspaces, MPSKit.max_virtualspaces(ψ₀)))

        # Quasiparticle tests
        Es, Bs = excitations(H, QuasiparticleAnsatz(; tol), gs; sector, num = 1)
        Es = Es .+ E₀
        # first excited state is second eigenvalue if sector is trivial
        @test Es[1] ≈ vals[isunit(sector) ? 2 : 1] atol = 1.0e-8
    end

    # shifted charges tests
    # targeting states with Sz = 1 => vals_shift_dense[0] == vals_dense[1]
    # so effectively shifting the charges by -1
    H_shift = MPSKit.add_physical_charge(H, U1Irrep.([1, 0, 0, 0]))
    H_shift_dense = convert(TensorMap, H_shift)
    vals_shift_dense = eigvals(H_shift_dense)
    for v in values(vals_shift_dense)
        sort!(v; by = real)
    end
    for (sector, vals) in pairs(vals_dense)
        sector′ = only(sector ⊗ U1Irrep(-1))
        @test vals ≈ vals_shift_dense[sector′]
    end
end
