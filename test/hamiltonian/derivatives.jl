println("
----------------------------
|   Derivative operators    |
----------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using MPSKit: C_hamiltonian, AC_hamiltonian, AC2_hamiltonian
using MPSKit: _transpose_front, _transpose_tail
using TensorKit
using BlockTensorKit: nonzero_length
using Random

Random.seed!(1234)

# `JordanMPO_AC(2)_Hamiltonian` has two construction paths: a fast one, taken when every
# block already has the storagetype of the environments, and a converting one
# this converting path is reached when `scalartype(H) != scalartype(ψ)` and an on-site D block is passed
# nearest-neighbour models have an ending B block on every site (but the first)

@testset "Jordan block structure of the long-range models" begin
    H = long_range_ising(Float64; L = 4)
    Hi = long_range_ising_infinite(Float64; L = 3)

    # nothing ends before the far end of the (long-range) bond
    @test all(i -> nonzero_length(H[i].B) == 0, 1:3)
    @test nonzero_length(H[4].B) == 1
    @test all(i -> nonzero_length(Hi[i].B) == 0, 1:2)
    @test nonzero_length(Hi[3].B) == 1

    # while the left virtual space is non-trivial everywhere but on the first site of
    # the finite chain, so `E` is present exactly where `B` is not
    @test size(H[1], 1) == 1
    @test all(i -> size(H[i], 1) > 1, 2:4)
    @test all(i -> size(Hi[i], 1) > 1, 1:3)

    # every site carries an on-site term, which is what forces the converting constructor
    @test all(i -> nonzero_length(H[i].D) == 1, 1:4)
    @test all(i -> nonzero_length(Hi[i].D) == 1, 1:3)
end

@testset "MPOHamiltonian derivatives: real operator, complex state" verbose = true begin
    D = 8
    L_inf, L_fin = 3, 4
    ψ_inf = InfiniteMPS(randn, ComplexF64, fill(ℂ^2, L_inf), fill(ℂ^D, L_inf))
    ψ_fin = FiniteMPS(randn, ComplexF64, L_fin, ℂ^2, ℂ^D)
    models = [
        "FiniteMPS, long-range" => (long_range_ising(Float64; L = L_fin), ψ_fin),
        "FiniteMPS, nearest-neighbour" => (transverse_field_ising(Float64; g = 4.0, L = L_fin), ψ_fin),
        "InfiniteMPS, long-range" => (long_range_ising_infinite(Float64; L = L_inf), ψ_inf),
        "InfiniteMPS, nearest-neighbour" => (repeat(transverse_field_ising(Float64; g = 4.0), 3), ψ_inf),
    ]

    @testset "$name" for (name, (H, ψ)) in models
        @test scalartype(H) <: Real
        @test scalartype(ψ) <: Complex

        # `complex(H)` takes the fast path, `H` the converting one: both must agree
        Hc = complex(H)
        @test scalartype(Hc) <: Complex
        envs = environments(ψ, H, ψ)
        envs_c = environments(ψ, Hc, ψ)
        @test expectation_value(ψ, H, envs) ≈ expectation_value(ψ, Hc, envs_c)

        L = length(ψ)
        bonds = isfinite(H) ? (1:(L - 1)) : (1:L)

        @testset "AC" begin
            for site in 1:L
                AC = ψ.AC[site]
                @test AC_hamiltonian(site, ψ, H, ψ, envs)(AC) ≈
                    AC_hamiltonian(site, ψ, Hc, ψ, envs_c)(AC)
            end
        end

        @testset "AC2" begin
            for site in bonds
                AC2 = _transpose_front(ψ.AC[site]) * _transpose_tail(ψ.AR[site + 1])
                @test AC2_hamiltonian(site, ψ, H, ψ, envs)(AC2) ≈
                    AC2_hamiltonian(site, ψ, Hc, ψ, envs_c)(AC2)
            end
        end

        @testset "C" begin
            for site in bonds
                C = ψ.C[site]
                @test C_hamiltonian(site, ψ, H, ψ, envs)(C) ≈
                    C_hamiltonian(site, ψ, Hc, ψ, envs_c)(C)
            end
        end
    end
end

@testset "FiniteMPS derivatives reproduce the expectation value" verbose = true begin
    # in mixed gauge the derivative operators contract everything but the center site(s),
    # so their expectation value is the full energy, independent of where we sit
    D = 8
    L = 10
    models = [
        "long-range" => long_range_ising(Float64; L = L),
        "nearest-neighbour" => transverse_field_ising(Float64; g = 4.0, L = L),
    ]

    @testset "$name" for (name, H) in models
        for T in (Float64, ComplexF64)
            ψ = normalize!(FiniteMPS(randn, T, L, ℂ^2, ℂ^D))
            envs = environments(ψ, H, ψ)
            E = expectation_value(ψ, H, envs)

            for site in 1:L
                AC = ψ.AC[site]
                @test dot(AC, AC_hamiltonian(site, ψ, H, ψ, envs)(AC)) ≈ E
            end
            for site in 1:(L - 1)
                AC2 = _transpose_front(ψ.AC[site]) * _transpose_tail(ψ.AR[site + 1])
                @test dot(AC2, AC2_hamiltonian(site, ψ, H, ψ, envs)(AC2)) ≈ E
            end
        end
    end
end
