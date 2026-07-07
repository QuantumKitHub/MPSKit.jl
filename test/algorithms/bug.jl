println("
-----------------------------
|   BUG time-stepping tests |
-----------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ
using LinearAlgebra: dot, norm
using Random

@testset "BUG time evolution" verbose = true begin
    dt = 0.1
    L = 10

    H = force_planar(heisenberg_XXX(Float64, Trivial; spin = 1 // 2, L))
    ψ = FiniteMPS(rand, Float64, L, ℙ^2, ℙ^4)
    ψ₀, = find_groundstate(ψ, H; verbosity = 0)
    E₀ = expectation_value(ψ₀, H)

    # 1. energy conservation + eigenstate phase
    @testset "energy conservation" begin
        ψ1, envs = timestep(ψ₀, H, 0.0, dt, BUG())
        E1 = expectation_value(ψ1, H, envs)
        @test E₀ ≈ E1 atol = 1.0e-2
        @test dot(ψ1, ψ₀) ≈ exp(im * dt * E₀) atol = 1.0e-4
    end

    # 2. agreement with TDVP over a few real-time steps of a random MPS
    @testset "agreement with TDVP" begin
        Random.seed!(1234)
        ψr = complex(FiniteMPS(rand, Float64, L, ℙ^2, ℙ^4))
        δt = 0.01
        ψ_bug, ψ_tdvp = ψr, ψr
        for k in 0:4
            ψ_bug, = timestep(ψ_bug, H, k * δt, δt, BUG())
            ψ_tdvp, = timestep(ψ_tdvp, H, k * δt, δt, TDVP())
        end
        @test expectation_value(ψ_bug, H) ≈ expectation_value(ψ_tdvp, H) atol = 1.0e-3
        @test abs(dot(ψ_bug, ψ_tdvp)) ≈ 1 atol = 1.0e-3
    end

    # 3. second-order convergence on a small full-rank system (isolates the temporal order)
    @testset "second-order convergence" begin
        Random.seed!(2)
        Lc = 4
        Hc = force_planar(transverse_field_ising(ComplexF64, Trivial; L = Lc))
        ψ_full = FiniteMPS(rand, ComplexF64, Lc, ℙ^2, ℙ^4)   # full-rank: 1,2,4,2,1

        Hmat = convert(TensorMap, Hc)
        ψvec = convert(TensorMap, ψ_full)
        ψvec /= norm(ψvec)

        T = 0.5
        dts = [0.1, 0.05, 0.025]
        errs = map(dts) do δt
            n = round(Int, T / δt)
            ref = exp(-im * Hmat * (n * δt)) * ψvec
            ψ = copy(ψ_full)
            envs = environments(ψ, Hc, ψ)
            for k in 0:(n - 1)
                timestep!(ψ, Hc, k * δt, δt, BUG(), envs)
            end
            ψout = convert(TensorMap, ψ)
            ψout /= norm(ψout)
            return 1 - abs(dot(ψout, ref))
        end

        slopes = [
            (log(errs[i + 1]) - log(errs[i])) / (log(dts[i + 1]) - log(dts[i]))
                for i in 1:(length(dts) - 1)
        ]
        @info "BUG convergence" errs slopes
        for s in slopes
            @test s ≈ 2 atol = 0.3
        end
    end

    # 4. imaginary-time evolution lowers the energy toward the ground state (and, having no
    #    backward substep, stays norm-preserving/stable)
    @testset "imaginary-time lowers energy" begin
        Random.seed!(5)
        ψi = complex(FiniteMPS(rand, Float64, L, ℙ^2, ℙ^4))
        E_start = real(expectation_value(ψi, H))
        E_prev = E_start
        for _ in 1:20
            ψi, = timestep(ψi, H, 0.0, 0.1, BUG(); imaginary_evolution = true)
            E_now = real(expectation_value(ψi, H))
            @test E_now ≤ E_prev + 1.0e-6   # monotone (non-increasing) energy
            E_prev = E_now
        end
        @test E_prev < E_start - 1.0        # substantial lowering toward the ground state
        @test norm(ψi) ≈ 1 atol = 1.0e-6    # imaginary-time BUG renormalizes each step
    end

    # 5. LazySum / MultipliedOperator smoke tests
    @testset "LazySum" begin
        Hlazy = LazySum([3 * H, 1.55 * H, -0.1 * H])
        ψl, envs = timestep(ψ₀, Hlazy, 0.0, dt, BUG())
        E = expectation_value(ψl, Hlazy, envs)
        @test (3 + 1.55 - 0.1) * E₀ ≈ E atol = 1.0e-2
    end

    @testset "TimeDependent LazySum" begin
        Ht = MultipliedOperator(H, t -> 4) + MultipliedOperator(H, 1.45)
        ψa, envsa = timestep(ψ₀, Ht(1.0), 0.0, dt, BUG())
        Ea = expectation_value(ψa, Ht(1.0), envsa)

        ψt, envst = timestep(ψ₀, Ht, 1.0, dt, BUG())
        Et = expectation_value(ψt, Ht(1.0), envst)
        @test Ea ≈ Et atol = 1.0e-8
    end
end
