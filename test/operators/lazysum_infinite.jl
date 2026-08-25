println("
------------------------------------
|   LazySum / Operators (infinite) |
------------------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using MPSKit: _transpose_front, _transpose_tail, C_hamiltonian, AC_hamiltonian, AC2_hamiltonian
using TensorKit
using TensorKit: ℙ

pspaces = (ℙ^4, Rep[U₁](0 => 2), Rep[SU₂](1 => 1, 2 => 1))
vspaces = (ℙ^10, Rep[U₁]((0 => 20)), Rep[SU₂](1 => 10, 3 => 5, 5 => 1))
if fast_tests
    pspaces = pspaces[1:1]
    vspaces = vspaces[1:1]
end

@testset "LazySum of (effective) Hamiltonian $(sectortype(pspace))" for (pspace, Dspace) in
    zip(pspaces, vspaces)
    Os = map(1:3) do i
        O = rand(ComplexF64, pspace^i, pspace^i)
        return O += O'
    end
    fs = [t -> 3t, 2, 1]

    @testset "LazySum InfiniteMPOHamiltonian" begin
        ψ = repeat(InfiniteMPS(pspace, Dspace), 2)
        Hs = map(Os) do O
            H = InfiniteMPOHamiltonian(O)
            return repeat(H, 2)
        end
        summedH = LazySum(Hs)
        envs = map(H -> environments(ψ, H, ψ), Hs)
        summed_envs = environments(ψ, summedH, ψ)

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

        envs = map(H -> environments(ψ, H, ψ), Hs)
        summed_envs = environments(ψ, summedH, ψ)

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
