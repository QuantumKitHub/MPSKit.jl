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

# Charge-sector (symmetric-tensor) coverage for the fixed-rank BUG. These use *genuine*
# symmetric tensors (no `force_planar`), exercising the graded-bond paths flagged in the design
# doc's hsector risk register (H1/H6/H7): the transport-tensor seed `isomorphism(V ← V)`, the
# `@plansor` (co)domain/dual conventions in `_bug_transport_*`, and the adjoints carrying sector
# duals. A fixed-rank step must preserve the total charge and the graded structure of every bond.
@testset "BUG symmetric tensors" verbose = true begin
    dt = 0.1
    L = 6

    # 1. U(1)-symmetric Heisenberg, both in the natural total-Sz = 0 sector and in a fixed nonzero
    #    total-charge (Sz = 1) sector: energy conservation + eigenstate phase + sector preservation.
    @testset "U(1) Heisenberg (total Sz = $label)" for (label, right) in
        (("0", U1Space(0 => 1)), ("1", U1Space(1 => 1)))
        Random.seed!(2718)
        H = heisenberg_XXX(ComplexF64, U1Irrep; spin = 1 // 2, L)
        maxV = MPSKit.max_virtualspaces(physicalspace(H))
        ψ = FiniteMPS(physicalspace(H), maxV[2:(end - 1)]; right)
        ψ₀, = find_groundstate(ψ, H; verbosity = 0)
        E₀ = expectation_value(ψ₀, H)

        Vl₀ = left_virtualspace.(Ref(ψ₀), 1:L)
        Vr₀ = right_virtualspace.(Ref(ψ₀), 1:L)

        ψ1, envs = timestep(ψ₀, H, 0.0, dt, BUG())
        E1 = expectation_value(ψ1, H, envs)

        @test E₀ ≈ E1 atol = 1.0e-2
        @test imag(E1) ≈ 0 atol = 1.0e-8
        @test dot(ψ1, ψ₀) ≈ exp(im * dt * E₀) atol = 1.0e-4
        # the fixed-rank step preserves the graded structure (sector content) of every bond
        @test left_virtualspace.(Ref(ψ1), 1:L) == Vl₀
        @test right_virtualspace.(Ref(ψ1), 1:L) == Vr₀
    end

    # 2. A second symmetry group. Z2 (transverse-field Ising) and SU2 (Heisenberg) both stress the
    #    graded transport tensor; same assertions (energy conservation + eigenstate phase).
    @testset "Z2 transverse-field Ising" begin
        Random.seed!(161803)
        H = transverse_field_ising(ComplexF64, Z2Irrep; g = 1.0, L)
        ψ = FiniteMPS(physicalspace(H), Z2Space(0 => 4, 1 => 4))
        ψ₀, = find_groundstate(ψ, H; verbosity = 0)
        E₀ = expectation_value(ψ₀, H)
        Vl₀ = left_virtualspace.(Ref(ψ₀), 1:L)

        ψ1, envs = timestep(ψ₀, H, 0.0, dt, BUG())
        E1 = expectation_value(ψ1, H, envs)
        @test E₀ ≈ E1 atol = 1.0e-2
        @test dot(ψ1, ψ₀) ≈ exp(im * dt * E₀) atol = 1.0e-4
        @test left_virtualspace.(Ref(ψ1), 1:L) == Vl₀
    end

    @testset "SU(2) Heisenberg" begin
        Random.seed!(577215)
        H = heisenberg_XXX(ComplexF64, SU2Irrep; spin = 1 // 2, L)
        # SU(2) spin-1/2 bonds alternate between integer / half-integer spins, so use the
        # model's own full-rank virtual spaces rather than a hand-picked (integer-only) space.
        maxV = MPSKit.max_virtualspaces(physicalspace(H))
        ψ = FiniteMPS(physicalspace(H), maxV[2:(end - 1)])
        ψ₀, = find_groundstate(ψ, H; verbosity = 0)
        E₀ = expectation_value(ψ₀, H)
        Vl₀ = left_virtualspace.(Ref(ψ₀), 1:L)

        ψ1, envs = timestep(ψ₀, H, 0.0, dt, BUG())
        E1 = expectation_value(ψ1, H, envs)
        @test E₀ ≈ E1 atol = 1.0e-2
        @test dot(ψ1, ψ₀) ≈ exp(im * dt * E₀) atol = 1.0e-4
        @test left_virtualspace.(Ref(ψ1), 1:L) == Vl₀
    end

    # 3. Imaginary-time symmetric evolution lowers the energy while preserving the sector + norm.
    @testset "imaginary-time (U(1))" begin
        Random.seed!(141421)
        H = heisenberg_XXX(ComplexF64, U1Irrep; spin = 1 // 2, L)
        maxV = MPSKit.max_virtualspaces(physicalspace(H))
        ψi = FiniteMPS(physicalspace(H), maxV[2:(end - 1)])
        Vl₀ = left_virtualspace.(Ref(ψi), 1:L)
        Vr₀ = right_virtualspace.(Ref(ψi), 1:L)
        E_start = real(expectation_value(ψi, H))
        E_prev = E_start
        for _ in 1:15
            ψi, = timestep(ψi, H, 0.0, 0.1, BUG(); imaginary_evolution = true)
            E_now = real(expectation_value(ψi, H))
            @test E_now ≤ E_prev + 1.0e-6            # monotone (non-increasing) energy
            E_prev = E_now
        end
        @test E_prev < E_start - 0.5                 # substantial lowering toward the ground state
        @test norm(ψi) ≈ 1 atol = 1.0e-6             # imaginary-time BUG renormalizes each step
        # the sector content of every bond is preserved throughout the imaginary-time sweep
        @test left_virtualspace.(Ref(ψi), 1:L) == Vl₀
        @test right_virtualspace.(Ref(ψi), 1:L) == Vr₀
    end
end
