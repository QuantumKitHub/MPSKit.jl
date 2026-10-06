using Test
using MPSKit, TensorKit, LinearAlgebra, Random
using TensorKit: ℙ

# Construct finite windows independently of the cluster-contraction kernels,
# including lengths that are not multiples of the infinite unit cell.
function cluster_dense_window(U, start, L)
    sites = [copy(U[start + i - 1]) for i in 1:L]
    sites[1] = sites[1][1, :, :, :]
    sites[end] = sites[end][:, :, :, 1]
    return convert(TensorMap, MPSKit.DenseMPO(FiniteMPO(sites)))
end

function cluster_exact_window(H, τ, start, L)
    if L == 1
        onsite = H[start][1, 1, 1, end]
        return exp(τ * MPSKit.removeunit(MPSKit.removeunit(onsite, 4), 1))
    end
    sites = [copy(H[start + i - 1]) for i in 1:L]
    sites[1] = sites[1][1, :, :, :]
    sites[end] = sites[end][:, :, :, end]
    return exp(τ * convert(TensorMap, FiniteMPOHamiltonian(sites)))
end

@testset "Nonperturbative cluster expansion" begin
    P = ℂ^2
    X = TensorMap(ComplexF64[0 1; 1 0], P ← P)
    Z = TensorMap(ComplexF64[1 0; 0 -1], P ← P)
    h = Z ⊗ Z + 0.3 * (X ⊗ X)
    H = InfiniteMPOHamiltonian(PeriodicArray([P]), 1 => 0.7X, (1, 2) => h)
    dense(U, L) = convert(TensorMap, MPSKit.DenseMPO(open_boundary_conditions(U, L)))
    exact(H, dt, L) = L == 1 ? exp(-im * dt * 0.7X) :
        exp(-im * dt * convert(TensorMap, open_boundary_conditions(H, L)))

    @testset "Exact clusters, N=$N" for N in 1:5
        U = make_time_mpo(H, 0.1, ClusterExpansion(N))
        @test U isa InfiniteMPO
        @test dim(left_virtualspace(U, 1)) == sum(4^l for l in 0:(N ÷ 2))
        for L in 1:N
            @test dense(U, L) ≈ exact(H, 0.1, L) atol = 1.0e-10
        end
    end

    @testset "Time steps and rank-deficient environments" begin
        onsite = InfiniteMPOHamiltonian(PeriodicArray([P]), 1 => 0.7X)
        commuting = InfiniteMPOHamiltonian(PeriodicArray([P]), (1, 2) => Z ⊗ Z)
        for dt in (0.0, -0.1, -0.1im)
            U = make_time_mpo(H, dt, ClusterExpansion(3))
            @test dense(U, 3) ≈ exact(H, dt, 3) atol = 1.0e-10
        end
        for Hsimple in (onsite, commuting)
            U = make_time_mpo(Hsimple, 0.1, ClusterExpansion(3))
            @test dense(U, 4) ≈ exp(-0.1im * convert(TensorMap, open_boundary_conditions(Hsimple, 4))) atol = 1.0e-10
        end
    end

    @testset "Complex Hamiltonian and reflection symmetry" begin
        Y = TensorMap(ComplexF64[0 -im; im 0], P ← P)
        complex_h = h + 0.2 * (X ⊗ Y + Y ⊗ X)
        complex_H = InfiniteMPOHamiltonian(PeriodicArray([P]), 1 => 0.4Y, (1, 2) => complex_h)
        for N in (3, 4), dt in (0.1, -0.1im)
            U = make_time_mpo(complex_H, dt, ClusterExpansion(N))
            @test dense(U, N) ≈ exp(-im * dt * convert(TensorMap, open_boundary_conditions(complex_H, N))) atol = 1.0e-10
            larger = dense(U, N + 1)
            reflected = permute(larger, (Tuple((N + 1):-1:1), Tuple((2N + 2):-1:(N + 2))))
            @test larger ≈ reflected atol = 1.0e-10
        end
    end

    @testset "Leading error on larger chains" begin
        for N in 2:4
            errors = Float64[]
            for dt in (0.04, 0.02)
                U = make_time_mpo(H, dt, ClusterExpansion(N))
                push!(errors, norm(dense(U, N + 1) - exact(H, dt, N + 1)))
            end
            @test log2(errors[1] / errors[2]) ≥ N - 0.15
        end
    end

    @testset "Symmetry sectors and dual physical spaces" begin
        rng = MersenneTwister(1234)
        spaces = (
            ℝ^2, ℙ^2,
            Rep[U₁](0 => 1, 1 => 1), dual(Rep[U₁](0 => 1, 1 => 1)),
            Rep[SU₂](1 // 2 => 1), dual(Rep[SU₂](1 // 2 => 1)),
            Vect[FermionParity](0 => 1, 1 => 1),
        )
        for P in spaces
            raw1 = randn!(rng, zeros(ComplexF64, P ← P))
            onsite = (raw1 + raw1') / 2
            raw2 = randn!(rng, zeros(ComplexF64, P ⊗ P ← P ⊗ P))
            interaction = (raw2 + raw2') / 2
            symmetric_H = InfiniteMPOHamiltonian(PeriodicArray([P]), 1 => onsite, (1, 2) => interaction)
            U = make_time_mpo(symmetric_H, 0.1, ClusterExpansion(1))
            @test dense(U, 1) ≈ exp(-0.1im * onsite) atol = 1.0e-10
            for N in 2:5
                U = @inferred MPSKit.make_cluster_mpo(symmetric_H, -0.1im, Val(N), 1.0e-12)
                @test physicalspace(U, 1) == P
                @test left_virtualspace(U, 1)[2] == fuse(P ⊗ dual(P))
                exact_cluster = exp(-0.1im * convert(TensorMap, open_boundary_conditions(symmetric_H, N)))
                @test dense(U, N) ≈ exact_cluster atol = 1.0e-10
            end
            residual = @inferred MPSKit.cluster_residual(symmetric_H, U, 1, -0.1im, Val(5))
            center = @inferred MPSKit.cluster_center(U, 1, residual, Val(5))
            @test center isa TensorMap
        end
    end

    @testset "Inference after the cluster-size dispatch" begin
        for n in (Val(4), Val(5))
            U = @inferred MPSKit.make_cluster_mpo(H, -0.1im, n, 1.0e-12)
            residual = @inferred MPSKit.cluster_residual(H, U, 1, -0.1im, n)
            center = @inferred MPSKit.cluster_center(U, 1, residual, n)
            @test center isa TensorMap
        end
    end

    @test_throws ArgumentError ClusterExpansion(0)
    @test_throws ArgumentError ClusterExpansion(2; tol = 0)
    @test_throws ArgumentError ClusterExpansion(2; tol = Inf)
    @test ClusterExpansion(; N = 3).N == 3
    H2 = repeat(H, 2)
    @testset "Repeated unit cell" for N in 2:5
        U = make_time_mpo(H, 0.1, ClusterExpansion(N))
        U2 = make_time_mpo(H2, 0.1, ClusterExpansion(N))
        @test length(U2) == 2
        @test cluster_dense_window(U2, 2, N + 1) ≈ dense(U, N + 1) atol = 1.0e-10
    end
end

@testset "Periodic cluster expansion with mixed physical spaces" begin
    rng = MersenneTwister(25)
    lattices = (
        [ℂ^2, ℂ^3], [ℂ^2, ℂ^3, ℂ^2],
        [ℝ^2, ℝ^3], [ℙ^2, ℙ^3],
        [Rep[U₁](0 => 1, 1 => 1), Rep[U₁](0 => 1, 1 => 1, 2 => 1)],
        [dual(Rep[U₁](0 => 1, 1 => 1)), dual(Rep[U₁](0 => 1, 1 => 1, 2 => 1))],
        [Rep[U₁](0 => 1, 1 => 1), dual(Rep[U₁](0 => 1, 1 => 1, 2 => 1))],
        [Rep[SU₂](1 // 2 => 1), Rep[SU₂](1 => 1)],
        [dual(Rep[SU₂](1 // 2 => 1)), dual(Rep[SU₂](1 => 1))],
        [Vect[FermionParity](0 => 1, 1 => 1), Vect[FermionParity](0 => 2, 1 => 1)],
    )
    for lattice in lattices
        Ps = PeriodicArray(lattice)
        terms = Pair[]
        for s in eachindex(lattice)
            a = randn!(rng, zeros(ComplexF64, Ps[s] ← Ps[s]))
            b = randn!(rng, zeros(ComplexF64, Ps[s] ⊗ Ps[s + 1] ← Ps[s] ⊗ Ps[s + 1]))
            push!(terms, s => (a + a') / 2, (s, s + 1) => (b + b') / 2)
        end
        H = InfiniteMPOHamiltonian(Ps, terms...)
        for N in 1:5
            U = @inferred MPSKit.make_cluster_mpo(H, -0.1im, Val(N), 1.0e-12)
            @test length(U) == length(H)
            for s in eachindex(lattice)
                @test physicalspace(U, s) == Ps[s]
                @test right_virtualspace(U, s) == left_virtualspace(U, s + 1)
                for L in 1:N
                    @test cluster_dense_window(U, s, L) ≈ cluster_exact_window(H, -0.1im, s, L) atol = 1.0e-10
                end
            end
        end
    end
end

@testset "Completed SVD support" begin
    rng = MersenneTwister(22)
    spaces = (
        (ℂ^4, ℂ^9),
        (Rep[U₁](0 => 4, 1 => 1), Rep[U₁](0 => 2, -1 => 2, 1 => 3)),
        (Rep[SU₂](0 => 3, 1 // 2 => 2, 1 => 1), Rep[SU₂](0 => 1, 1 // 2 => 3, 2 => 1)),
        (Vect[FermionParity](0 => 4, 1 => 1), Vect[FermionParity](0 => 2, 1 => 3)),
    )
    for (L, R) in spaces, dualize in (false, true)
        L, R = dualize ? (dual(L), dual(R)) : (L, R)
        raw = randn!(rng, zeros(ComplexF64, L ← R))
        for scale in (1.0, 1.0e-28, 0.0), tol in (1.0e-12, 0.5)
            C = scale * raw
            A, B = @inferred MPSKit.cluster_complete_svd(C, tol)
            @test norm(A * B - C) ≤ 2.0e-12 * max(norm(C), 1)
            @test domain(A) == codomain(B)
            @test A * pinv(A; rtol = 1.0e-12) ≈ id(storagetype(C), L) atol = 1.0e-10
            @test pinv(B; rtol = 1.0e-12) * B ≈ id(storagetype(C), R) atol = 1.0e-10
            if scale == 1 && tol == 1.0e-12
                V = only(domain(A))
                for c in union(sectors(L), sectors(R))
                    @test dim(V, c) == max(dim(L, c), dim(R, c))
                end
            end
        end
    end
    # The SVD rank cutoff must not discard the tiny second singular value.
    C = TensorMap(ComplexF64[1 0 0; 0 1.0e-14 0; 0 0 0], ℂ^3 ← ℂ^3)
    A, B = MPSKit.cluster_complete_svd(C, 1.0e-12)
    @test dim(only(domain(A))) == 5
    @test norm(A * B - C) < 1.0e-16
end

@testset "Mixed-cell edge cases and leading error" begin
    rng = MersenneTwister(24)
    for lattice in ([ℂ^2, ℂ^3], [Rep[SU₂](1 // 2 => 1), Rep[SU₂](1 => 1)])
        Ps = PeriodicArray(lattice)
        onsite_terms = Pair[]
        full_terms = Pair[]
        for s in eachindex(lattice)
            a = randn!(rng, zeros(ComplexF64, Ps[s] ← Ps[s]))
            b = randn!(rng, zeros(ComplexF64, Ps[s] ⊗ Ps[s + 1] ← Ps[s] ⊗ Ps[s + 1]))
            push!(onsite_terms, s => (a + a') / 2)
            push!(full_terms, s => (a + a') / 2, (s, s + 1) => (b + b') / 2)
        end
        onsite = InfiniteMPOHamiltonian(Ps, onsite_terms...)
        H = InfiniteMPOHamiltonian(Ps, full_terms...)
        # Remove the interaction across the cell seam to form independent dimers.
        dimers = InfiniteMPOHamiltonian(Ps, filter(p -> p.first != (2, 3), full_terms)...)
        for (model, dt) in ((onsite, 0.1), (dimers, 0.1), (H, 0.0), (H, -0.1im), (H, -0.1))
            U = make_time_mpo(model, dt, ClusterExpansion(5))
            longest = model === onsite || model === dimers || iszero(dt) ? 6 : 5
            for s in eachindex(lattice), L in 1:longest
                @test cluster_dense_window(U, s, L) ≈ cluster_exact_window(model, -im * dt, s, L) atol = 1.0e-10
            end
        end
        # A large cutoff changes the SVD channels, not the matched exponentials.
        for N in (3, 5), tol in (0.1, 0.5)
            U = @inferred MPSKit.make_cluster_mpo(H, -0.1im, Val(N), tol)
            for s in eachindex(lattice), L in 1:N
                @test cluster_dense_window(U, s, L) ≈ cluster_exact_window(H, -0.1im, s, L) atol = 1.0e-10
            end
        end
        for N in 3:4, s in eachindex(lattice)
            # Keep the SU(2) N=4 errors above the floating-point floor.
            steps = spacetype(Ps[1]) == ComplexSpace || N == 3 ? (0.04, 0.02) : (0.16, 0.08)
            errors = Float64[]
            for dt in steps
                U = make_time_mpo(H, dt, ClusterExpansion(N))
                push!(errors, norm(cluster_dense_window(U, s, N + 1) - cluster_exact_window(H, -im * dt, s, N + 1)))
            end
            @test log2(errors[1] / errors[2]) ≥ N - 0.15
        end
    end
end
