using ..TestSetup
using Test, TestExtras
using MPSKit
using MPSKit: _transpose_front, _transpose_tail
using TensorKit
using TensorKit: ℙ
using Adapt

# `@constinferred` evaluates a helper at global scope, which is not allowed inside a function
# body, so the testsuite functions use `@testinferred` (none of the calls have constant arguments)

export states_windowmps

function states_windowmps(AT; fast = false)
    return @testset "WindowMPS ($AT)" begin
        g = 8.0
        ham = adapt(AT, force_planar(transverse_field_ising(; g)))

        # operator for testing expectation_value
        X = S_x(; spin = 1 // 2)
        E = TensorMap(ComplexF64[1 0; 0 1], ℂ^2 ← ℂ^2)
        O = adapt(AT, force_planar(-(S_z_S_z(; spin = 1 // 2) + (g / 2) * (X ⊗ E + E ⊗ X))))

        gs, = find_groundstate(adapt(AT, InfiniteMPS([ℙ^2], [ℙ^10])), ham, VUMPS(; verbosity = 0))

        # constructor 1 - give it a plain array of tensors
        window_1 = WindowMPS(gs, copy.([gs.AC[1]; [gs.AR[i] for i in 2:10]]), gs)

        # constructor 2 - used to take a "slice" from an infinite mps
        window_2 = WindowMPS(gs, 10)

        @test eltype(window_1) == eltype(typeof(window_1))

        P = @testinferred physicalspace(window_2)
        Vleft = @testinferred left_virtualspace(window_2)
        Vright = @testinferred right_virtualspace(window_2)

        for i in -3:13
            @test physicalspace(window_2, i) == P[i]
            @test left_virtualspace(window_2, i) == Vleft[i]
            @test right_virtualspace(window_2, i) == Vright[i]
        end

        # we should logically have that window_1 approximates window_2
        ovl = dot(window_1, window_2)
        @test ovl ≈ 1 atol = 1.0e-8

        # constructor 3 - random initial tensors
        # TODO: WindowMPS has no Adapt rule, so the random window tensors are not placed in `AT`
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

        w_ham = WindowMPOHamiltonian(ham, 1:length(window))
        window, envs, _ = find_groundstate(window, w_ham, DMRG(; verbosity = 0))

        e2 = expectation_value(window, (2, 3) => O)

        @test real(e2) ≤ real(e1)

        window, envs = timestep(window, w_ham, 0.1, 0.0, TDVP2(; trunc = truncrank(20)), envs)
        window, envs = timestep(window, w_ham, 0.1, 0.0, TDVP(), envs)

        e3 = expectation_value(window, (2, 3) => O)

        @test e2 ≈ e3 atol = 1.0e-4
    end
end
