using .TestSetup
using Test, MPSKit, TensorKit, Random

sweep_allocator(ψ) = MPSKit.default_allocator(ψ, MPSKit.SerialScheduler())

_kind(nsites) = nsites == 1 ? MPSKit.AC_hamiltonian : MPSKit.AC2_hamiltonian
_window(ψ, site, nsites) = nsites == 1 ? ψ.AC[site] : MPSKit.AC2(ψ, site)

# Run `sweeps` complete sweeps of local updates, testing at every local window that the effective
# Hamiltonian built from the prepared halves of `envs` acts like an unprepared one built from the
# environments `reference(ψ)` of the current state.
function test_dmrg_windows(alg, ψ, H, envs = environments(ψ, H, ψ); reference = ψ -> environments(ψ, H, ψ), sweeps = 1)
    nsites = alg isa DMRG ? 1 : 2
    f = _kind(nsites)
    allocator = sweep_allocator(ψ)
    for iter in 1:sweeps, (direction, sites) in zip((Val(:right), Val(:left)), MPSKit.sweep_ranges(alg, ψ))
        for pos in sites
            x = _window(ψ, pos, nsites)
            @test f(pos, ψ, H, ψ, envs) * x ≈ f(pos, ψ, H, ψ, reference(ψ); prepare = false) * x
            ψ, = MPSKit.local_update!(
                pos, direction, ψ, H, alg, envs, 1.0, 0.0, 0.0, iter, MPSKit.NoTimerOutput(), allocator
            )
        end
    end
    return envs
end

const X = S_x(ComplexF64, Trivial; spin = 1 // 2)
function dmrg_models(L)
    nearest = transverse_field_ising(; g = 2.0, L)
    longrange = long_range_ising(Float64; g = 2.0, L)
    # three-site terms give non-trivial A·A, C·A and A·B channels
    mixed = nearest + FiniteMPOHamiltonian(
        fill(ℂ^2, L),
        (1, 2, 5) => X ⊗ X ⊗ X, (2, 3, 6) => X ⊗ X ⊗ X,
        (1, 4) => X ⊗ X, (3, 6) => X ⊗ X
    )
    return (; nearest, longrange, mixed)
end

dmrg_algorithms(; kwargs...) = (
    DMRG(; verbosity = 0, kwargs...),
    DMRG2(; verbosity = 0, trunc = truncrank(4), kwargs...),
    DMRG(; verbosity = 0, alg_expand = OptimalExpand(; trunc = truncrank(1)), trunc = truncrank(4), kwargs...),
    DMRG(; verbosity = 0, alg_expand = SketchedExpand(; trunc = truncrank(1)), trunc = truncrank(4), kwargs...),
    DMRG(; verbosity = 0, alg_gauge = DMRG3S(0.1, ExponentialDecay(0.7)), trunc = truncrank(4), kwargs...),
)

@testset "Effective Hamiltonians at every local window" begin
    L = 6
    models = dmrg_models(L)
    @testset "$name / $(nameof(typeof(H))) / $k" for (name, H0) in pairs(models), H in (H0, FiniteMPO(H0)),
            (k, alg) in enumerate(dmrg_algorithms())
        Random.seed!(42)
        test_dmrg_windows(alg, FiniteMPS(randn, ComplexF64, L, ℂ^2, ℂ^2), H)
    end
    @testset "planar $(nameof(typeof(alg)))" for alg in dmrg_algorithms()[1:2]
        H = force_planar(models.mixed)
        test_dmrg_windows(alg, FiniteMPS(randn, ComplexF64, L, TensorKit.ℙ^2, TensorKit.ℙ^3), H)
    end
    @testset "real $(nameof(typeof(alg))), L = $L" for L in (2, 4), alg in dmrg_algorithms()[1:2]
        H = transverse_field_ising(Float64; g = 2.0, L)
        test_dmrg_windows(alg, FiniteMPS(randn, Float64, L, ℂ^2, ℂ^3), H)
    end
    H1 = transverse_field_ising(; g = 2.0, L = 4)
    H2 = long_range_ising(Float64; g = 1.0, L = 4)
    @testset "$(nameof(typeof(H))) $(nameof(typeof(alg)))" for H in (
                LazySum([H1, H2]), LazySum([MultipliedOperator(H1, 0.7), MultipliedOperator(H2, 1.3)]),
            ), alg in dmrg_algorithms()[1:2]
        test_dmrg_windows(alg, FiniteMPS(randn, ComplexF64, 4, ℂ^2, ℂ^3), H)
    end
    @testset "SU2 $(nameof(typeof(alg)))" for alg in (DMRG(; verbosity = 0), DMRG2(; verbosity = 0, trunc = truncrank(8)))
        H = heisenberg_XXX(ComplexF64, SU2Irrep; spin = 1 // 2, L = 4)
        exchange = TestSetup.S_exchange(ComplexF64, SU2Irrep; spin = 1 // 2)
        H += FiniteMPOHamiltonian(fill(physicalspace(H, 1), 4), (1, 3) => exchange, (2, 4) => exchange)
        V = Rep[SU₂](0 => 2, 1 // 2 => 2, 1 => 1)
        test_dmrg_windows(alg, FiniteMPS(randn, ComplexF64, 4, physicalspace(H, 1), V), H)
    end
    @testset "WindowMPS with $(nameof(typeof(H)))" for H in (
            force_planar(transverse_field_ising(; g = 2.0)),
            WindowMPOHamiltonian(force_planar(transverse_field_ising(; g = 2.0)), 1:4),
        )
        Hinf = H isa WindowMPOHamiltonian ? H.left_ham : H
        gs = InfiniteMPS(randn, ComplexF64, [TensorKit.ℙ^2], [TensorKit.ℙ^3])
        ψ = WindowMPS(gs, 4)
        boundaries = environments(ψ, Hinf, ψ)
        op = H isa WindowMPOHamiltonian ? H.finite_ham : H
        initialize(ψ) = MPSKit.initialize_environments(ψ, op, ψ, copy(boundaries.GLs[1]), copy(boundaries.GRs[end]))
        envs = test_dmrg_windows(DMRG(; verbosity = 0), ψ, H, initialize(ψ); reference = initialize)
        @test leftenv(envs, 1, ψ) ≈ boundaries.GLs[1]
        @test rightenv(envs, length(ψ), ψ) ≈ boundaries.GRs[end]
    end
end

@testset "Ground states agree with exact diagonalization" begin
    L = 6
    models = dmrg_models(L)
    @testset "$name / $(nameof(typeof(H))) / $k" for (name, H0) in pairs(models), H in (H0, FiniteMPO(H0)),
            (k, alg) in enumerate(dmrg_algorithms(; maxiter = 20, trunc = truncrank(8)))
        E0 = first(first(exact_diagonalization(H0)))
        Random.seed!(7)
        ψ0 = FiniteMPS(randn, ComplexF64, L, ℂ^2, ℂ^8)
        ψ, envs, = find_groundstate(ψ0, H, alg)
        @test real(expectation_value(ψ, H0)) ≈ E0 atol = 1.0e-5
    end
end

@testset "Different bra and ket" begin
    L = 6
    fermionic = kitaev_model(ComplexF64, Trivial; L)
    su2 = heisenberg_XXX(ComplexF64, SU2Irrep; spin = 1 // 2, L)
    exchange = TestSetup.S_exchange(ComplexF64, SU2Irrep; spin = 1 // 2)
    su2 += FiniteMPOHamiltonian(fill(physicalspace(su2, 1), L), (1, 3) => exchange, (2, 5) => exchange)
    models = (;
        dmrg_models(L)...,
        planar = force_planar(dmrg_models(L).mixed),
        real = transverse_field_ising(Float64; g = 2.0, L),
        fermionic, su2,
    )
    function virtualspaces(H, d)
        I = sectortype(physicalspace(H, 1))
        I === Trivial && return ℂ^d
        I === TensorKit.PlanarTrivial && return TensorKit.ℙ^d
        I === FermionParity && return Vect[FermionParity](0 => d, 1 => d)
        return Rep[SU₂](0 => d, 1 // 2 => d)
    end
    @testset "$name, $nsites sites" for (name, H) in pairs(models), nsites in (1, 2)
        Random.seed!(3)
        T = scalartype(H)
        P = physicalspace(H, 1)
        bra = FiniteMPS(randn, T, L, P, virtualspaces(H, 2))
        ket = FiniteMPS(randn, T, L, P, virtualspaces(H, 3))
        envs = environments(bra, H, ket)
        reference = environments(bra, FiniteMPO(H), ket)
        f = _kind(nsites)
        for site in 1:(L - nsites + 1)
            x = _window(ket, site, nsites)
            y = f(site, bra, FiniteMPO(H), ket, reference; prepare = false) * x
            raw = f(site, bra, H, ket, envs; prepare = false)
            @test raw * x ≈ y
            @test MPSKit.prepare_operator!!(raw) * x ≈ y
            @test f(site, bra, H, ket, envs) * x ≈ y
            @test f(site, bra, FiniteMPO(H), ket, reference) * x ≈ y
        end
    end
    @testset "approximate $(nameof(typeof(alg)))" for alg in (DMRG(; verbosity = 0), DMRG2(; verbosity = 0, trunc = truncrank(16)))
        H = dmrg_models(L).mixed
        Random.seed!(5)
        ϕ = FiniteMPS(randn, ComplexF64, L, ℂ^2, ℂ^2)
        ψ0 = FiniteMPS(randn, ComplexF64, L, ℂ^2, ℂ^8)
        ψ, = approximate(ψ0, (H, ϕ), alg)
        ψmpo, = approximate(ψ0, (FiniteMPO(H), ϕ), alg)
        @test abs(dot(ψ, ψmpo)) ≈ norm(ψ) * norm(ψmpo) atol = 1.0e-8
    end
end

@testset "Effective operators are independent snapshots" begin
    for H in (transverse_field_ising(; g = 2.0, L = 5), long_range_ising(Float64; g = 2.0, L = 5))
        ψ = FiniteMPS(randn, ComplexF64, 5, ℂ^2, ℂ^3)
        allocator = sweep_allocator(ψ)
        envs = environments(ψ, H, ψ)
        retained = []
        for (f, x) in ((MPSKit.AC_hamiltonian, ψ.AC[1]), (MPSKit.AC2_hamiltonian, MPSKit.AC2(ψ, 1)))
            fresh = f(1, ψ, H, ψ, environments(ψ, H, ψ); prepare = false)
            first = f(1, ψ, H, ψ, envs; allocator)
            y = first * x
            push!(retained, (first, copy(x), copy(y)))
            for _ in 1:3
                @test f(1, ψ, H, ψ, envs; allocator) * x ≈ fresh * x
                # The requested matvec allocator is independent of the one the halves were prepared with.
                @test f(1, ψ, H, ψ, envs; allocator = MPSKit.DefaultAllocator()) * x ≈ fresh * x
                raw = f(1, ψ, H, ψ, envs; prepare = false, allocator)
                @test raw * x ≈ fresh * x
                @test MPSKit.prepare_operator!!(raw) * x ≈ fresh * x
                @test first * x ≈ y
            end
        end
        pairs = copy(envs.halves.pairs)
        halves = envs.halves.AC2
        alg = DMRG2(; verbosity = 0, trunc = truncrank(4))
        state = MPSKit.DMRGState(ψ, H, alg, envs, allocator, MPSKit.NoTimerOutput())
        it = MPSKit.IterativeSolver(alg, state)
        state = MPSKit.sweep!(it, state, Val(:right), 1)
        left = copy(halves.left)
        state = MPSKit.sweep!(it, state, Val(:left), 1)
        @test all(a === b for (a, b) in zip(left, halves.left))
        @test all(a === b for (a, b) in zip(pairs, envs.halves.pairs) if !isnothing(a))
        @test all(H * x ≈ y for (H, x, y) in retained)
        right = copy(halves.right)
        MPSKit.sweep!(it, state, Val(:right), 2)
        @test all(a === b for (a, b) in zip(right[1:(end - 2)], halves.right[1:(end - 2)]))
        # The boundary environment is never replaced, and neither is its half.
        @test halves.right[end] === right[end]
    end
end

@testset "Effective operators reuse the prepared halves" begin
    H0 = dmrg_models(6).mixed
    ψ = FiniteMPS(randn, ComplexF64, 6, ℂ^2, ℂ^3)
    @testset "$(nameof(typeof(H))), $nsites sites" for H in (H0, FiniteMPO(H0)), nsites in (1, 2)
        envs = environments(ψ, H, ψ)
        f = _kind(nsites)
        l = @inferred MPSKit.lefthalf(f, envs, 2, ψ)
        r = @inferred MPSKit.righthalf(f, envs, 1 + nsites, ψ)
        Heff = @inferred f(2, ψ, H, ψ, envs)
        if H isa FiniteMPO
            @test Heff.leftenv === l
            @test Heff.rightenv === r
            continue
        end
        continuing = nsites == 1 ? :A : :AA
        @test getfield(Heff, continuing).leftenv === l.continuing
        @test getfield(Heff, continuing).rightenv === r.continuing
        for field in fieldnames(typeof(Heff))
            field in (continuing, :backend, :allocator) && continue
            v = getfield(Heff, field)
            ismissing(v) || @test v === getfield(l.prepared, field) || v === getfield(r.prepared, field)
        end
    end
end

@testset "Halves follow their environments" begin
    L = 6
    H0 = dmrg_models(L).mixed
    @testset "$(nameof(typeof(H))), $nsites sites" for H in (H0, FiniteMPO(H0)), nsites in (1, 2)
        ψ = FiniteMPS(randn, ComplexF64, L, ℂ^2, ℂ^3)
        envs = environments(ψ, H, ψ)
        f = _kind(nsites)
        l, r = MPSKit.lefthalf(f, envs, 4, ψ), MPSKit.righthalf(f, envs, 2, ψ)
        @test MPSKit.lefthalf(f, envs, 4, ψ) === l
        @test MPSKit.righthalf(f, envs, 2, ψ) === r
        # Changing the center invalidates the environments on both sides of it.
        ψ.AC[3] = randn(scalartype(ψ), space(ψ.AC[3]))
        @test MPSKit.lefthalf(f, envs, 4, ψ) !== l
        @test MPSKit.righthalf(f, envs, 2, ψ) !== r
        fresh = environments(ψ, H, ψ)
        for site in (5, 1, 4, 2, 3)
            x = _window(ψ, site, nsites)
            @test f(site, ψ, H, ψ, envs) * x ≈ f(site, ψ, H, ψ, fresh; prepare = false) * x
        end
    end
end

@testset "Halves of each kind are prepared when first queried" begin
    H = transverse_field_ising(; g = 2.0, L = 5)
    for alg in (DMRG(; verbosity = 0), DMRG2(; verbosity = 0, trunc = truncrank(4)))
        ψ = FiniteMPS(randn, ComplexF64, 5, ℂ^2, ℂ^3)
        envs = environments(ψ, H, ψ)
        halves, other = alg isa DMRG ? (envs.halves.AC, envs.halves.AC2) : (envs.halves.AC2, envs.halves.AC)
        state = MPSKit.DMRGState(ψ, H, alg, envs, sweep_allocator(ψ), MPSKit.NoTimerOutput())
        @test isnothing(halves.left[2])
        for site in 1:2
            MPSKit.local_update!(
                site, Val(:right), ψ, H, alg, envs, state.ϵ, 0.0, 0.0, 1, state.timeroutput, state.allocator
            )
            # The window at site 1 only queries the halves at its own boundaries.
            @test isnothing(halves.left[2]) == (site == 1)
        end
        @test all(isnothing, other.left) && all(isnothing, other.right)
        # Nearest-neighbor MPOs have no path through two continuing A blocks.
        alg isa DMRG2 && @test all(isempty, something(envs.halves.pairs[1]).channels)
    end
end

@testset "Environments without prepared halves" begin
    ψ = FiniteMPS(randn, ComplexF64, 4, ℂ^2, ℂ^3)
    @test !MPSKit.has_halves(environments(ψ, ψ))
    @test MPSKit.has_halves(environments(ψ, transverse_field_ising(; L = 4), ψ))
end
