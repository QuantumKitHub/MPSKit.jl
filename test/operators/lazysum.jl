println("
-----------------------------
|   LazySum / Operators      |
-----------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using MPSKit: _transpose_front, _transpose_tail, C_hamiltonian, AC_hamiltonian, AC2_hamiltonian
using TensorKit
using TensorKit: ℙ
using VectorInterface: One

@testset "General LazySum of $(eltype(Os))" for Os in (
        rand(ComplexF64, rand(1:10)),
        map(i -> rand(ComplexF64, ℂ^13, ℂ^7), 1:rand(1:10)),
        map(i -> rand(ComplexF64, ℂ^1 ⊗ ℂ^2, ℂ^3 ⊗ ℂ^4), 1:rand(1:10)),
    )
    LazyOs = LazySum(Os)

    #test user interface
    summed = sum(Os)

    @test sum(LazyOs) ≈ summed atol = 1 - 08

    LazyOs_added = +(LazyOs, Os...)

    @test sum(LazyOs_added) ≈ 2 * summed atol = 1 - 08
end

@testset "MultipliedOperator of $(typeof(O)) with $(typeof(f))" for (O, f) in
    zip(
        (rand(ComplexF64), rand(ComplexF64, ℂ^13, ℂ^7), rand(ComplexF64, ℂ^1 ⊗ ℂ^2, ℂ^3 ⊗ ℂ^4)),
        (t -> 3t, 1.1, One())
    )
    tmp = MPSKit.MultipliedOperator(O, f)
    if tmp isa TimedOperator
        @test tmp(1.1)() ≈ f(1.1) * O atol = 1 - 08
    elseif tmp isa UntimedOperator
        @test tmp() ≈ f * O atol = 1 - 08
    end
end

@testset "General Time-dependent LazySum of $(eltype(Os))" for Os in (
        rand(ComplexF64, 4),
        fill(rand(ComplexF64, ℂ^13, ℂ^7), 4),
        fill(rand(ComplexF64, ℂ^1 ⊗ ℂ^2, ℂ^3 ⊗ ℂ^4), 4),
    )

    #test user interface
    fs = [t -> 3t, t -> t + 2, 4, 1]
    Ofs = map(zip(fs, Os)) do (f, O)
        if f == 1
            return O
        else
            return MPSKit.MultipliedOperator(O, f)
        end
    end
    LazyOs = LazySum(Ofs)
    summed = sum(zip(fs, Os)) do (f, O)
        if f isa Function
            f(1.1) * O
        else
            f * O
        end
    end

    @test sum(LazyOs(1.1)) ≈ summed atol = 1 - 08

    LazyOs_added = +(LazyOs, Ofs...)

    @test sum(LazyOs_added(1.1)) ≈ 2 * summed atol = 1 - 08
end

pspaces = (ℙ^4, Rep[U₁](0 => 2), Rep[SU₂](1 => 1, 2 => 1))
vspaces = (ℙ^10, Rep[U₁]((0 => 20)), Rep[SU₂](1 => 10, 3 => 5, 5 => 1))

@testset "LazySum of (effective) Hamiltonian $(sectortype(pspace))" for (pspace, Dspace) in
    zip(pspaces, vspaces)
    Os = map(1:3) do i
        O = rand(ComplexF64, pspace^i, pspace^i)
        return O += O'
    end
    fs = [t -> 3t, 2, 1]

    @testset "LazySum FiniteMPOHamiltonian" begin
        L = rand(3:2:20)
        ψ = FiniteMPS(rand, ComplexF64, L, pspace, Dspace)
        lattice = fill(pspace, L)
        Hs = map(enumerate(Os)) do (i, O)
            return FiniteMPOHamiltonian(
                lattice,
                ntuple(x -> x + j, i) => O for j in 0:(L - i)
            )
        end
        summedH = LazySum(Hs)

        envs = map(H -> environments(ψ, H), Hs)
        summed_envs = environments(ψ, summedH)

        expval = sum(zip(Hs, envs)) do (H, env)
            return expectation_value(ψ, H, env)
        end
        expval1 = expectation_value(ψ, sum(summedH))
        expval2 = expectation_value(ψ, summedH, summed_envs)
        expval3 = expectation_value(ψ, summedH)
        @test expval ≈ expval1
        @test expval ≈ expval2
        @test expval ≈ expval3

        # test derivatives
        summedhct = C_hamiltonian(1, ψ, summedH, ψ, summed_envs)
        sum1 = sum(zip(Hs, envs)) do (H, env)
            return C_hamiltonian(1, ψ, H, ψ, env)(ψ.C[1])
        end
        @test summedhct(ψ.C[1], 0.0) ≈ sum1

        summedhct = AC_hamiltonian(1, ψ, summedH, ψ, summed_envs)
        sum2 = sum(zip(Hs, envs)) do (H, env)
            return AC_hamiltonian(1, ψ, H, ψ, env)(ψ.AC[1])
        end
        @test summedhct(ψ.AC[1], 0.0) ≈ sum2

        v = _transpose_front(ψ.AC[1]) * _transpose_tail(ψ.AR[2])
        summedhct = AC2_hamiltonian(1, ψ, summedH, ψ, summed_envs)
        sum3 = sum(zip(Hs, envs)) do (H, env)
            return AC2_hamiltonian(1, ψ, H, ψ, env)(v)
        end
        @test summedhct(v, 0.0) ≈ sum3

        Hts = [MultipliedOperator(Hs[1], fs[1]), MultipliedOperator(Hs[2], fs[2]), Hs[3]]
        summedH = LazySum(Hts)
        t = 1.1
        summedH_at = summedH(t)

        envs = map(H -> environments(ψ, H), Hs)
        summed_envs = environments(ψ, summedH)

        expval = sum(zip(fs, Hs, envs)) do (f, H, env)
            return (f isa Function ? f(t) : f) * expectation_value(ψ, H, env)
        end
        expval1 = expectation_value(ψ, sum(summedH_at))
        expval2 = expectation_value(ψ, summedH_at, summed_envs)
        expval3 = expectation_value(ψ, summedH_at)
        @test expval ≈ expval1
        @test expval ≈ expval2
        @test expval ≈ expval3

        # test derivatives
        summedhct = C_hamiltonian(1, ψ, summedH, ψ, summed_envs)
        sum1 = sum(zip(fs, Hs, envs)) do (f, H, env)
            if f isa Function
                f = f(t)
            end
            return f * C_hamiltonian(1, ψ, H, ψ, env)(ψ.C[1])
        end
        @test summedhct(ψ.C[1], t) ≈ sum1

        summedhct = AC_hamiltonian(1, ψ, summedH, ψ, summed_envs)
        sum2 = sum(zip(fs, Hs, envs)) do (f, H, env)
            if f isa Function
                f = f(t)
            end
            return f * AC_hamiltonian(1, ψ, H, ψ, env)(ψ.AC[1])
        end
        @test summedhct(ψ.AC[1], t) ≈ sum2

        v = _transpose_front(ψ.AC[1]) * _transpose_tail(ψ.AR[2])
        summedhct = AC2_hamiltonian(1, ψ, summedH, ψ, summed_envs)
        sum3 = sum(zip(fs, Hs, envs)) do (f, H, env)
            return (f isa Function ? f(t) : f) * AC2_hamiltonian(1, ψ, H, ψ, env)(v)
        end
        @test summedhct(v, t) ≈ sum3
    end

    @testset "LazySum InfiniteMPOHamiltonian" begin
        ψ = repeat(InfiniteMPS(pspace, Dspace), 2)
        Hs = map(Os) do O
            H = InfiniteMPOHamiltonian(O)
            return repeat(H, 2)
        end
        summedH = LazySum(Hs)
        envs = map(H -> environments(ψ, H), Hs)
        summed_envs = environments(ψ, summedH)

        expval = sum(zip(Hs, envs)) do (H, Env)
            return expectation_value(ψ, H, Env)
        end
        expval1 = expectation_value(ψ, sum(summedH))
        expval2 = expectation_value(ψ, summedH, summed_envs)
        expval3 = expectation_value(ψ, summedH)
        @test expval ≈ expval1
        @test expval ≈ expval2
        @test expval ≈ expval3

        # test derivatives
        summedhct = C_hamiltonian(1, ψ, summedH, ψ, summed_envs)
        sum1 = sum(zip(Hs, envs)) do (H, env)
            return C_hamiltonian(1, ψ, H, ψ, env)(ψ.C[1])
        end
        @test summedhct(ψ.C[1], 0.0) ≈ sum1

        summedhct = AC_hamiltonian(1, ψ, summedH, ψ, summed_envs)
        sum2 = sum(zip(Hs, envs)) do (H, env)
            return AC_hamiltonian(1, ψ, H, ψ, env)(ψ.AC[1])
        end
        @test summedhct(ψ.AC[1], 0.0) ≈ sum2

        v = _transpose_front(ψ.AC[1]) * _transpose_tail(ψ.AR[2])
        summedhct = AC2_hamiltonian(1, ψ, summedH, ψ, summed_envs)
        sum3 = sum(zip(Hs, envs)) do (H, env)
            return AC2_hamiltonian(1, ψ, H, ψ, env)(v)
        end
        @test summedhct(v, 0.0) ≈ sum3

        Hts = [MultipliedOperator(Hs[1], fs[1]), MultipliedOperator(Hs[2], fs[2]), Hs[3]]
        summedH = LazySum(Hts)
        t = 1.1
        summedH_at = summedH(t)

        envs = map(H -> environments(ψ, H), Hs)
        summed_envs = environments(ψ, summedH)

        expval = sum(zip(fs, Hs, envs)) do (f, H, env)
            return (f isa Function ? f(t) : f) * expectation_value(ψ, H, env)
        end
        expval1 = expectation_value(ψ, sum(summedH_at))
        expval2 = expectation_value(ψ, summedH_at, summed_envs)
        expval3 = expectation_value(ψ, summedH_at)
        @test expval ≈ expval1
        @test expval ≈ expval2
        @test expval ≈ expval3

        # test derivatives
        summedhct = C_hamiltonian(1, ψ, summedH, ψ, summed_envs)
        sum1 = sum(zip(fs, Hs, envs)) do (f, H, env)
            if f isa Function
                f = f(t)
            end
            return f * C_hamiltonian(1, ψ, H, ψ, env)(ψ.C[1])
        end
        @test summedhct(ψ.C[1], t) ≈ sum1

        summedhct = AC_hamiltonian(1, ψ, summedH, ψ, summed_envs)
        sum2 = sum(zip(fs, Hs, envs)) do (f, H, env)
            if f isa Function
                f = f(t)
            end
            return f * AC_hamiltonian(1, ψ, H, ψ, env)(ψ.AC[1])
        end
        @test summedhct(ψ.AC[1], t) ≈ sum2

        v = _transpose_front(ψ.AC[1]) * _transpose_tail(ψ.AR[2])
        summedhct = AC2_hamiltonian(1, ψ, summedH, ψ, summed_envs)
        sum3 = sum(zip(fs, Hs, envs)) do (f, H, env)
            return (f isa Function ? f(t) : f) * AC2_hamiltonian(1, ψ, H, ψ, env)(v)
        end
        @test summedhct(v, t) ≈ sum3
    end
end
