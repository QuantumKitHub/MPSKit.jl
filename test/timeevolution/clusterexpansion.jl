using Test
using MPSKit, TensorKit, LinearAlgebra, Random
using TensorKit: ℙ
using TensorKitTensors.SpinOperators: σˣ, σʸ, σᶻ

# Independent dense windows also cover lengths that are not multiples of the unit cell.
function dense_mpo_window(O, start = 1, L = length(O))
    sites = [copy(O[start + i - 1]) for i in 1:L]
    sites[1] = sites[1][1, :, :, :]
    boundary = O isa MPOHamiltonian ? size(sites[end], 4) : 1
    sites[end] = sites[end][:, :, :, boundary]
    window = O isa MPOHamiltonian ? FiniteMPOHamiltonian(sites) : MPSKit.DenseMPO(FiniteMPO(sites))
    return convert(TensorMap, window)
end

function random_test_hamiltonian(rng, Ps; onsite_only = false, dimers = false)
    infinite = Ps isa MPSKit.PeriodicVector
    terms = Pair[]
    for s in eachindex(Ps)
        a = randn!(rng, zeros(ComplexF64, Ps[s] ← Ps[s]))
        push!(terms, s => (a + a') / 2)
        if (infinite || s < length(Ps)) && !onsite_only && (!dimers || isodd(s))
            b = randn!(rng, zeros(ComplexF64, Ps[s] ⊗ Ps[s + 1] ← Ps[s] ⊗ Ps[s + 1]))
            push!(terms, (s, s + 1) => (b + b') / 2)
        end
    end
    return infinite ? InfiniteMPOHamiltonian(Ps, terms...) : FiniteMPOHamiltonian(Ps, terms...)
end

@testset "Nonperturbative cluster expansion" begin
    X, Y, Z = σˣ(), σʸ(), σᶻ()
    P = space(X, 1)
    h = Z ⊗ Z + 0.3 * X ⊗ X + 0.2 * (X ⊗ Y + Y ⊗ X)
    H = InfiniteMPOHamiltonian(PeriodicArray([P]), 1 => 0.7X, (1, 2) => h)

    @testset "Exact clusters and inference, N=$N" for N in 1:5
        U = @inferred MPSKit.make_cluster_expansion_mpo(H, -0.1im, Val(N), 1.0e-12)
        @test U isa InfiniteMPO
        @test dim(left_virtualspace(U, 1)) == sum(4^l for l in 0:(N ÷ 2))
        for L in 1:N
            @test dense_mpo_window(U, 1, L) ≈ exp(-0.1im * dense_mpo_window(H, 1, L)) atol = 1.0e-10
        end
        if N ≥ 3
            residual = @inferred MPSKit.evolution_cluster_residual(H, U, 1, -0.1im, Val(N))
            @inferred MPSKit.solve_center_correction(U, 1, residual, Val(N))
        end
        if N in (3, 4)
            larger = dense_mpo_window(U, 1, N + 1)
            reflected = permute(larger, (Tuple((N + 1):-1:1), Tuple((2N + 2):-1:(N + 2))))
            @test larger ≈ reflected atol = 1.0e-10
            U2 = make_time_mpo(repeat(H, 2), 0.1, ClusterExpansion(N))
            @test dense_mpo_window(U2, 2, N + 1) ≈ larger atol = 1.0e-10
        end
    end

    @testset "Leading error, N=$N" for N in 2:4
        for model in (H, open_boundary_conditions(H, 5))
            errors = [
                norm(
                    dense_mpo_window(make_time_mpo(model, dt, ClusterExpansion(N)), 1, N + 1) -
                        exp(-im * dt * dense_mpo_window(model, 1, N + 1))
                ) for dt in (0.04, 0.02)
            ]
            @test log2(errors[1] / errors[2]) ≥ N - 0.15
        end
    end

    @testset "Finite boundaries and cluster-size cap, L=$L" for L in (1, 2, 3, 6)
        finite = open_boundary_conditions(H, L)
        U = make_time_mpo(finite, 0.1, ClusterExpansion(L + 3; tol = 0.5))
        @test U isa FiniteMPO
        @test dim(left_virtualspace(U, 1)) == dim(right_virtualspace(U, L)) == 1
        @test dense_mpo_window(U) ≈ exp(-0.1im * convert(TensorMap, finite)) atol = 1.0e-10
    end
    @testset "Finite/infinite consistency, N=$N" for N in (2, 3, 4, 5)
        finite = make_time_mpo(open_boundary_conditions(H, 6), 0.1, ClusterExpansion(N))
        infinite = make_time_mpo(H, 0.1, ClusterExpansion(N))
        @test dense_mpo_window(finite) ≈ dense_mpo_window(infinite, 1, 6) atol = 1.0e-10
    end

    @test_throws ArgumentError ClusterExpansion(0)
    @test_throws ArgumentError ClusterExpansion(2; tol = 0)
    @test_throws ArgumentError ClusterExpansion(2; tol = Inf)
    @test ClusterExpansion(; N = 3).N == 3
    empty = FiniteMPOHamiltonian(similar(parent(open_boundary_conditions(H, 2)), 0))
    @test_throws ArgumentError make_time_mpo(empty, 0.1, ClusterExpansion(2))
end

@testset "Mixed physical spaces and boundaries" begin
    rng = MersenneTwister(25)
    lattices = (
        [ℂ^2, ℂ^3], [ℂ^2, ℂ^3, ℂ^2], [ℝ^2, ℝ^3], [ℙ^2, ℙ^3],
        [Rep[U₁](0 => 1, 1 => 1), dual(Rep[U₁](0 => 1, 1 => 1, 2 => 1))],
        [dual(Rep[SU₂](1 // 2 => 1)), Rep[SU₂](1 => 1)],
        [Vect[FermionParity](0 => 1, 1 => 1), Vect[FermionParity](0 => 2, 1 => 1)],
    )
    for lattice in lattices, infinite in (true, false)
        Ps = infinite ? PeriodicArray(lattice) : repeat(lattice, 3)[1:5]
        H = random_test_hamiltonian(rng, Ps)
        for N in (4, 5)
            U = @inferred MPSKit.make_cluster_expansion_mpo(H, -0.1im, Val(N), 1.0e-12)
            @test length(U) == length(H)
            for s in eachindex(Ps)
                @test physicalspace(U, s) == Ps[s]
                (infinite || s < length(H)) && @test right_virtualspace(U, s) == left_virtualspace(U, s + 1)
                width = infinite ? N : min(N, length(H) - s + 1)
                @test dense_mpo_window(U, s, width) ≈ exp(-0.1im * dense_mpo_window(H, s, width)) atol = 1.0e-10
            end
        end
    end
end

@testset "Time steps and rank-deficient environments" begin
    rng = MersenneTwister(24)
    Z = σᶻ()
    P = space(Z, 1)
    commuting = InfiniteMPOHamiltonian(PeriodicArray([P]), (1, 2) => Z ⊗ Z)
    U = make_time_mpo(commuting, 0.1, ClusterExpansion(4))
    @test dense_mpo_window(U, 1, 5) ≈ exp(-0.1im * dense_mpo_window(commuting, 1, 5)) atol = 1.0e-10

    for Ps in (PeriodicArray([ℂ^2, ℂ^3]), [ℂ^2, ℂ^3, ℂ^2, ℂ^3, ℂ^2])
        H = random_test_hamiltonian(rng, Ps)
        onsite = random_test_hamiltonian(rng, Ps; onsite_only = true)
        dimers = random_test_hamiltonian(rng, Ps; dimers = true)
        for (model, dt) in ((onsite, 0.1), (dimers, 0.1), (H, 0.0), (H, -0.1im), (H, -0.1))
            U = make_time_mpo(model, dt, ClusterExpansion(5))
            width = isfinite(model) || model === H && !iszero(dt) ? 5 : 6
            starts = isfinite(model) ? (1:1) : eachindex(Ps)
            for s in starts
                @test dense_mpo_window(U, s, width) ≈ exp(-im * dt * dense_mpo_window(model, s, width)) atol = 1.0e-10
            end
            model === onsite && @test dim(left_virtualspace(U, 1)) == 1
        end
        # The SVD cutoff changes the channels, not the matched exponentials.
        for N in (4, 5)
            U = @inferred MPSKit.make_cluster_expansion_mpo(H, -0.1im, Val(N), 0.5)
            starts = isfinite(H) ? (1:(length(H) - N + 1)) : eachindex(Ps)
            for s in starts
                @test dense_mpo_window(U, s, N) ≈ exp(-0.1im * dense_mpo_window(H, s, N)) atol = 1.0e-10
            end
        end
    end
end

# Reconstruction alone does not ensure the factors retain all environment directions.
@testset "Completed SVD support" begin
    L, R = ℂ^2, ℂ^3
    weak = TensorMap(ComplexF64[1 0 0; 0 1.0e-14 0], L ← R)
    for C in (weak, zeros(ComplexF64, L ← R))
        A, B = @inferred MPSKit.factor_with_complementary_spaces(C, 1.0e-12)
        @test norm(A * B - C) ≤ 1.0e-16 * norm(C)
        @test A * pinv(A; rtol = 1.0e-12) ≈ id(storagetype(C), L) atol = 1.0e-10
        @test pinv(B; rtol = 1.0e-12) * B ≈ id(storagetype(C), R) atol = 1.0e-10
        @test dim(only(domain(A))) == (iszero(norm(C)) ? 5 : 4)
    end
end
