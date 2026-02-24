println("
----------------------
|   WindowMPS tests   |
----------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using MPSKit: _transpose_front, _transpose_tail
using TensorKit
using TensorKit: ℙ

@testset "WindowMPS" begin
    g = 8.0
    ham = force_planar(transverse_field_ising(; g))

    # operator for testing expectation_value
    X = S_x(; spin = 1 // 2)
    E = TensorMap(ComplexF64[1 0; 0 1], ℂ^2 ← ℂ^2)
    O = force_planar(-(S_z_S_z(; spin = 1 // 2) + (g / 2) * (X ⊗ E + E ⊗ X)))

    gs, = find_groundstate(InfiniteMPS([ℙ^2], [ℙ^10]), ham, VUMPS(; verbosity = 0))

    # constructor 1 - give it a plain array of tensors
    window_1 = WindowMPS(gs, copy.([gs.AC[1]; [gs.AR[i] for i in 2:10]]), gs)

    # constructor 2 - used to take a "slice" from an infinite mps
    window_2 = WindowMPS(gs, 10)

    @test eltype(window_1) == eltype(typeof(window_1))

    P = @constinferred physicalspace(window_2)
    Vleft = @constinferred left_virtualspace(window_2)
    Vright = @constinferred right_virtualspace(window_2)

    for i in -3:13
        @test physicalspace(window_2, i) == P[i]
        @test left_virtualspace(window_2, i) == Vleft[i]
        @test right_virtualspace(window_2, i) == Vright[i]
    end

    # we should logically have that window_1 approximates window_2
    ovl = dot(window_1, window_2)
    @test ovl ≈ 1 atol = 1.0e-8

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

    e1 = expectation_value(window, (2, 3) => O)

    window, envs, _ = find_groundstate(window, ham, DMRG(; verbosity = 0))

    e2 = expectation_value(window, (2, 3) => O)

    @test real(e2) ≤ real(e1)

    window, envs = timestep(window, ham, 0.1, 0.0, TDVP2(; trscheme = truncrank(20)), envs)
    window, envs = timestep(window, ham, 0.1, 0.0, TDVP(), envs)

    e3 = expectation_value(window, (2, 3) => O)

    @test e2 ≈ e3 atol = 1.0e-4
end
