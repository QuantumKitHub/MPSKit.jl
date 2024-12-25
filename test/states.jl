println("
--------------
|   States   |
--------------
")
module TestStates

using ..TestSetup
using Test, TestExtras
using MPSKit
using MPSKit: _transpose_front, _transpose_tail
using TensorKit
using TensorKit: ℙ

@testset "FiniteMPS ($(sectortype(D)), $elt)" for (D, d, elt) in [(ℙ^10, ℙ^2, ComplexF64),
                                                                  (Rep[SU₂](1 => 1, 0 => 3),
                                                                   Rep[SU₂](0 => 1) * Rep[SU₂](0 => 1),
                                                                   ComplexF32)]
    ψ = FiniteMPS(rand, elt, rand(3:20), d, D)

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

@testset "FiniteMPS ($(sectortype(D)), $elt)" for (D, d, elt) in [(ℙ^10, ℙ^2, ComplexF64),
                                                                  (Rep[U₁](-1 => 3, 0 => 3, 1 => 3),
                                                                   Rep[U₁](-1 => 1, 0 => 1, 1 => 1),
                                                                   ComplexF64)]
    ψ_small = FiniteMPS(rand, elt, 4, d, D)
    ψ_small2 = FiniteMPS(convert(TensorMap, ψ_small))
    @test dot(ψ_small, ψ_small2) ≈ dot(ψ_small, ψ_small)
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

@testset "InfiniteMPS ($(sectortype(D)), $elt)" for (D, d, elt) in
                                                    [(ℙ^10, ℙ^2, ComplexF64),
                                                     (Rep[U₁](1 => 3), Rep[U₁](0 => 1),
                                                      ComplexF64)]
    tol = Float64(eps(real(elt)) * 100)

    ψ = InfiniteMPS([rand(elt, D * d, D), rand(elt, D * d, D)]; tol)

    for i in 1:length(ψ)
        @plansor difference[-1 -2; -3] := ψ.AL[i][-1 -2; 1] * ψ.C[i][1; -3] -
                                          ψ.C[i - 1][-1; 1] * ψ.AR[i][1 -2; -3]
        @test norm(difference, Inf) < tol * 10

        @test l_LL(ψ, i) * TransferMatrix(ψ.AL[i], ψ.AL[i]) ≈ l_LL(ψ, i + 1)
        @test l_LR(ψ, i) * TransferMatrix(ψ.AL[i], ψ.AR[i]) ≈ l_LR(ψ, i + 1)
        @test l_RL(ψ, i) * TransferMatrix(ψ.AR[i], ψ.AL[i]) ≈ l_RL(ψ, i + 1)
        @test l_RR(ψ, i) * TransferMatrix(ψ.AR[i], ψ.AR[i]) ≈ l_RR(ψ, i + 1)

        @test TransferMatrix(ψ.AL[i], ψ.AL[i]) * r_LL(ψ, i) ≈ r_LL(ψ, i + 1)
        @test TransferMatrix(ψ.AL[i], ψ.AR[i]) * r_LR(ψ, i) ≈ r_LR(ψ, i + 1)
        @test TransferMatrix(ψ.AR[i], ψ.AL[i]) * r_RL(ψ, i) ≈ r_RL(ψ, i + 1)
        @test TransferMatrix(ψ.AR[i], ψ.AR[i]) * r_RR(ψ, i) ≈ r_RR(ψ, i + 1)
    end
end

@testset "MultilineMPS ($(sectortype(D)), $elt)" for (D, d, elt) in
                                                     [(ℙ^10, ℙ^2, ComplexF64),
                                                      (Rep[U₁](1 => 3), Rep[U₁](0 => 1),
                                                       ComplexF32)]
    tol = Float64(eps(real(elt)) * 100)
    ψ = MultilineMPS([rand(elt, D * d, D) rand(elt, D * d, D)
                      rand(elt, D * d, D) rand(elt, D * d, D)]; tol)

    for i in 1:size(ψ, 1), j in 1:size(ψ, 2)
        @plansor difference[-1 -2; -3] := ψ.AL[i, j][-1 -2; 1] * ψ.C[i, j][1; -3] -
                                          ψ.C[i, j - 1][-1; 1] * ψ.AR[i, j][1 -2; -3]
        @test norm(difference, Inf) < tol * 10

        @test l_LL(ψ, i, j) * TransferMatrix(ψ.AL[i, j], ψ.AL[i, j]) ≈ l_LL(ψ, i, j + 1)
        @test l_LR(ψ, i, j) * TransferMatrix(ψ.AL[i, j], ψ.AR[i, j]) ≈ l_LR(ψ, i, j + 1)
        @test l_RL(ψ, i, j) * TransferMatrix(ψ.AR[i, j], ψ.AL[i, j]) ≈ l_RL(ψ, i, j + 1)
        @test l_RR(ψ, i, j) * TransferMatrix(ψ.AR[i, j], ψ.AR[i, j]) ≈ l_RR(ψ, i, j + 1)

        @test TransferMatrix(ψ.AL[i, j], ψ.AL[i, j]) * r_LL(ψ, i, j) ≈ r_LL(ψ, i, j + 1)
        @test TransferMatrix(ψ.AL[i, j], ψ.AR[i, j]) * r_LR(ψ, i, j) ≈ r_LR(ψ, i, j + 1)
        @test TransferMatrix(ψ.AR[i, j], ψ.AL[i, j]) * r_RL(ψ, i, j) ≈ r_RL(ψ, i, j + 1)
        @test TransferMatrix(ψ.AR[i, j], ψ.AR[i, j]) * r_RR(ψ, i, j) ≈ r_RR(ψ, i, j + 1)
    end
end

@testset "WindowMPS" begin
    g = 8.0
    ham = force_planar(transverse_field_ising(; g))

    # operator for testing expectation_value
    X = S_x(; spin=1 // 2)
    E = TensorMap(ComplexF64[1 0; 0 1], ℂ^2 ← ℂ^2)
    O = force_planar(-(S_zz(; spin=1 // 2) + (g / 2) * (X ⊗ E + E ⊗ X)))

    gs, = find_groundstate(InfiniteMPS([ℙ^2], [ℙ^10]), ham, VUMPS(; verbosity=0))

    # constructor 1 - give it a plain array of tensors
    window_1 = WindowMPS(gs, copy.([gs.AC[1]; [gs.AR[i] for i in 2:10]]), gs)

    # constructor 2 - used to take a "slice" from an infinite mps
    window_2 = WindowMPS(gs, 10)

    # we should logically have that window_1 approximates window_2
    ovl = dot(window_1, window_2)
    @test ovl ≈ 1 atol = 1e-8

    # constructor 3 - random initial tensors
    window = WindowMPS(rand, ComplexF64, 10, ℙ^2, ℙ^10, gs, gs)
    normalize!(window)

    for i in 1:length(window)
        @test window.AC[i] ≈ window.AL[i] * window.C[i]
        @test window.AC[i] ≈
              _transpose_front(window.C[i - 1] * _transpose_tail(window.AR[i]))
    end

    @test norm(window) ≈ 1
    window = window * 3
    @test 9 ≈ norm(window)^2
    window = 3 * window
    @test 9 * 9 ≈ norm(window)^2
    normalize!(window)

    e1 = expectation_value(window, (1, 2) => O)

    window, envs, _ = find_groundstate(window, ham, DMRG(; verbosity=0))

    e2 = expectation_value(window, (1, 2) => O)

    @test real(e2) ≤ real(e1)

    window, envs = timestep(window, ham, 0.1, 0.0, TDVP2(), envs)
    window, envs = timestep(window, ham, 0.1, 0.0, TDVP(), envs)

    e3 = expectation_value(window, (1, 2) => O)

    @test e2 ≈ e3 atol = 1e-4
end

@testset "Quasiparticle state" verbose = true begin
    L = 10
    @testset "Finite" verbose = true for (H, D, d) in
                                         [(force_planar(transverse_field_ising(; L)), ℙ^10,
                                           ℙ^2),
                                          (heisenberg_XXX(SU2Irrep; spin=1, L),
                                           Rep[SU₂](1 => 1, 0 => 3), Rep[SU₂](1 => 1))]
        ψ = FiniteMPS(rand, ComplexF64, L, d, D)
        normalize!(ψ)

        #rand_quasiparticle is a private non-exported function
        ϕ₁ = MPSKit.LeftGaugedQP(rand, ψ)
        ϕ₂ = MPSKit.LeftGaugedQP(rand, ψ)

        @test norm(axpy!(1, ϕ₁, copy(ϕ₂))) ≤ norm(ϕ₁) + norm(ϕ₂)
        @test norm(ϕ₁) * 3 ≈ norm(ϕ₁ * 3)

        normalize!(ϕ₁)

        ϕ₁_f = convert(FiniteMPS, ϕ₁)
        ϕ₂_f = convert(FiniteMPS, ϕ₂)

        @test dot(ϕ₁_f, ϕ₂_f) ≈ dot(ϕ₁, ϕ₂) atol = 1e-5
        @test norm(ϕ₁_f) ≈ norm(ϕ₁) atol = 1e-5

        ev_f = expectation_value(ϕ₁_f, H) - expectation_value(ψ, H)
        ev_q = dot(ϕ₁, effective_excitation_hamiltonian(H, ϕ₁))
        @test ev_f ≈ ev_q atol = 1e-5
    end

    @testset "Infinite" for (th, D, d) in
                            [(force_planar(transverse_field_ising()), ℙ^10, ℙ^2),
                             (heisenberg_XXX(SU2Irrep; spin=1), Rep[SU₂](1 => 3, 0 => 2),
                              Rep[SU₂](1 => 1))]
        period = rand(1:4)
        ψ = InfiniteMPS(fill(d, period), fill(D, period))

        #rand_quasiparticle is a private non-exported function
        ϕ₁ = MPSKit.LeftGaugedQP(rand, ψ)
        ϕ₂ = MPSKit.LeftGaugedQP(rand, ψ)

        @test norm(axpy!(1, ϕ₁, copy(ϕ₂))) ≤ norm(ϕ₁) + norm(ϕ₂)
        @test norm(ϕ₁) * 3 ≈ norm(ϕ₁ * 3)

        @test dot(ϕ₁,
                  convert(MPSKit.LeftGaugedQP, convert(MPSKit.RightGaugedQP, ϕ₁))) ≈
              dot(ϕ₁, ϕ₁) atol = 1e-10
    end
end

end
