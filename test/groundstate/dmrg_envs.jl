using .TestSetup
using Test, MPSKit, TensorKit, Random, Logging

sweep_allocator(ψ) = MPSKit.default_allocator(ψ, MPSKit.SerialScheduler())

# Run `sweeps` complete sweeps of local updates on `DMRGEnvironments`, testing at every
# local window that the effective Hamiltonian built from its halves acts like one built from the
# environments `reference(ψ)` of the current state.
function test_dmrg_windows(alg, ψ, H, envs = environments(ψ, H, ψ); reference = ψ -> environments(ψ, H, ψ), sweeps = 1)
    nsites = alg isa DMRG ? 1 : 2
    allocator = sweep_allocator(ψ)
    state = MPSKit.DMRGState(ψ, H, alg, envs, allocator, MPSKit.NoTimerOutput())
    dmrg_envs = state.envs
    @test MPSKit.uses_dmrg_environments(dmrg_envs)
    for iter in 1:sweeps, (direction, sites) in zip((Val(:right), Val(:left)), MPSKit.sweep_ranges(alg, ψ))
        for pos in sites
            MPSKit.center_window!(dmrg_envs, ψ, pos, pos + nsites - 1)
            fresh = reference(ψ)
            if nsites == 1
                x = ψ.AC[pos]
                @test MPSKit.AC_hamiltonian(pos, ψ, H, ψ, dmrg_envs) * x ≈ MPSKit.AC_hamiltonian(pos, ψ, H, ψ, fresh) * x
            else
                x = MPSKit.AC2(ψ, pos)
                @test MPSKit.AC2_hamiltonian(pos, ψ, H, ψ, dmrg_envs) * x ≈ MPSKit.AC2_hamiltonian(pos, ψ, H, ψ, fresh) * x
            end
            ψ, = MPSKit.local_update!(
                pos, direction, ψ, H, alg, dmrg_envs, 1.0, 0.0, 0.0, iter, MPSKit.NoTimerOutput(), allocator
            )
        end
    end
    return dmrg_envs
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
        dmrg_envs = test_dmrg_windows(DMRG(; verbosity = 0), ψ, H, initialize(ψ); reference = initialize)
        @test leftenv(dmrg_envs, 1, ψ) ≈ boundaries.GLs[1]
        @test rightenv(dmrg_envs, length(ψ), ψ) ≈ boundaries.GRs[end]
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
        @test envs isa MPSKit.FiniteEnvironments
    end
end

@testset "Effective operators are independent snapshots" begin
    for H in (transverse_field_ising(; g = 2.0, L = 5), long_range_ising(Float64; g = 2.0, L = 5))
        ψ = FiniteMPS(randn, ComplexF64, 5, ℂ^2, ℂ^3)
        allocator = sweep_allocator(ψ)
        dmrg_envs = MPSKit.dmrg_environments(
            ψ, H, environments(ψ, H, ψ); nsites = 2, backend = MPSKit.DefaultBackend(), allocator
        )
        pairs = dmrg_envs.pairs
        retained = []
        for (constructor, x) in ((MPSKit.AC_hamiltonian, ψ.AC[1]), (MPSKit.AC2_hamiltonian, MPSKit.AC2(ψ, 1)))
            fresh = constructor(1, ψ, H, ψ, environments(ψ, H, ψ); allocator)
            first = constructor(1, ψ, H, ψ, dmrg_envs; allocator)
            y = first * x
            push!(retained, (first, copy(x), copy(y)))
            for _ in 1:3
                @test constructor(1, ψ, H, ψ, dmrg_envs; allocator) * x ≈ fresh * x
                # The requested matvec allocator is independent of the environments' allocator.
                @test constructor(1, ψ, H, ψ, dmrg_envs; allocator = MPSKit.DefaultAllocator()) * x ≈ fresh * x
                raw = constructor(1, ψ, H, ψ, dmrg_envs; prepare = false, allocator)
                @test raw * x ≈ fresh * x
                @test MPSKit.prepare_operator!!(raw) * x ≈ fresh * x
                @test first * x ≈ y
            end
        end
        alg = DMRG2(; verbosity = 0, trunc = truncrank(4))
        state = MPSKit.DMRGState(ψ, H, dmrg_envs, 0, 1.0, ones(4), zeros(4), zeros(4), MPSKit.NoTimerOutput(), allocator)
        it = MPSKit.IterativeSolver(alg, state)
        state = MPSKit.sweep!(it, state, Val(:right), 1)
        left = copy(dmrg_envs.lhalves)
        state = MPSKit.sweep!(it, state, Val(:left), 1)
        @test all(a === b for (a, b) in zip(left, dmrg_envs.lhalves))
        @test dmrg_envs.pairs === pairs
        @test all(H * x ≈ y for (H, x, y) in retained)
        right = copy(dmrg_envs.rhalves)
        MPSKit.sweep!(it, state, Val(:right), 2)
        @test all(a === b for (a, b) in zip(right[1:(end - 2)], dmrg_envs.rhalves[1:(end - 2)]))
        # The boundary half is never replaced.
        @test dmrg_envs.rhalves[end] === right[end]
    end
end

@testset "Effective operators reuse the prepared halves" begin
    H0 = dmrg_models(6).mixed
    ψ = FiniteMPS(randn, ComplexF64, 6, ℂ^2, ℂ^3)
    @testset "$(nameof(typeof(H))), $nsites sites" for H in (H0, FiniteMPO(H0)), nsites in (1, 2)
        dmrg_envs = MPSKit.dmrg_environments(
            ψ, H, environments(ψ, H, ψ); nsites, backend = MPSKit.DefaultBackend(), allocator = sweep_allocator(ψ)
        )
        MPSKit.transfer_leftenv!(dmrg_envs, ψ, H, ψ, 2)
        l, r = MPSKit.lefthalf(dmrg_envs, 2), MPSKit.righthalf(dmrg_envs, 1 + nsites)
        Heff = nsites == 1 ? MPSKit.AC_hamiltonian(2, ψ, H, ψ, dmrg_envs) : MPSKit.AC2_hamiltonian(2, ψ, H, ψ, dmrg_envs)
        if H isa FiniteMPO
            @test Heff.leftenv === l
            @test Heff.rightenv === r
            continue
        end
        continuing = nsites == 1 ? :A : :AA
        @test getfield(Heff, continuing).leftenv === l.continuing
        @test getfield(Heff, continuing).rightenv === r.continuing
        for f in fieldnames(typeof(Heff))
            f in (continuing, :backend, :allocator) && continue
            v = getfield(Heff, f)
            ismissing(v) || @test v === getfield(l.prepared, f) || v === getfield(r.prepared, f)
        end
    end
end

@testset "Transfers are independent of the prepared halves" begin
    L = 6
    H0 = dmrg_models(L).mixed
    for H in (H0, FiniteMPO(H0))
        ψ = FiniteMPS(randn, ComplexF64, L, ℂ^2, ℂ^3)
        dmrg_envs = MPSKit.dmrg_environments(
            ψ, H, environments(ψ, H, ψ); nsites = 1,
            backend = MPSKit.DefaultBackend(), allocator = MPSKit.DefaultAllocator(),
        )
        i = 3
        for site in 2:i
            MPSKit.transfer_leftenv!(dmrg_envs, ψ, H, ψ, site)
        end
        # Corrupt the prepared halves: transfers must only read the environments.
        poison(half) = H isa MPOHamiltonian ? typeof(half)(half.prepared, zero(half.continuing)) : zero(half)
        dmrg_envs.lhalves[i] = poison(dmrg_envs.lhalves[i])
        dmrg_envs.rhalves[i] = poison(dmrg_envs.rhalves[i])
        MPSKit.transfer_leftenv!(dmrg_envs, ψ, H, ψ, i + 1)
        MPSKit.transfer_rightenv!(dmrg_envs, ψ, H, ψ, i - 1)
        reference = environments(ψ, H, ψ)
        @test leftenv(dmrg_envs, i + 1, ψ) ≈ leftenv(reference, i + 1, ψ)
        @test rightenv(dmrg_envs, i - 1, ψ) ≈ rightenv(reference, i - 1, ψ)
    end
end

@testset "Finalizers" begin
    H = long_range_ising(Float64; g = 2.0, L = 6)
    E0 = first(first(exact_diagonalization(H)))
    ψ0 = FiniteMPS(randn, ComplexF64, 6, ℂ^2, ℂ^3)
    seen = Int[]
    readonly = function (iter, ψ, H, envs)
        push!(seen, iter)
        # Ordinary read access can materialize environments beyond the sweep center.
        for i in 1:length(ψ)
            leftenv(envs, i, ψ)
            rightenv(envs, i, ψ)
        end
        return ψ, envs
    end
    alg = DMRG2(; verbosity = 0, maxiter = 2, trunc = truncrank(4), finalize = readonly)
    ψ, envs, info = @test_logs min_level = Logging.Warn find_groundstate(ψ0, H, alg)
    ref, = find_groundstate(ψ0, H, DMRG2(; verbosity = 0, maxiter = 2, trunc = truncrank(4)))
    @test seen == collect(1:info.numiter)
    @test abs(dot(ψ, ref)) ≈ norm(ψ) * norm(ref) atol = 1.0e-10
    @test envs isa MPSKit.FiniteEnvironments
    @test all(leftenv(envs, i, ψ) ≈ leftenv(environments(ψ, H, ψ), i, ψ) for i in 1:length(ψ))
    @test all(rightenv(envs, i, ψ) ≈ rightenv(environments(ψ, H, ψ), i, ψ) for i in 1:length(ψ))

    # Returning a new state and environments rebuilds the DMRG environments from them.
    replacing = function (iter, ψ, H, envs)
        ψ′ = copy(ψ)
        return ψ′, environments(ψ′, H, ψ′)
    end
    @testset "$(nameof(typeof(alg)))" for alg in (
            DMRG(; verbosity = 0, maxiter = 20, finalize = replacing),
            DMRG2(; verbosity = 0, maxiter = 20, trunc = truncrank(8), finalize = replacing),
        )
        ψ, envs, = @test_logs (:warn, r"rebuilding the DMRG environments") match_mode = :any find_groundstate(ψ0, H, alg)
        @test real(expectation_value(ψ, H, envs)) ≈ E0 atol = 1.0e-6
    end
end

@testset "Environments advance at the next local window" begin
    H = transverse_field_ising(; g = 2.0, L = 5)
    for alg in (DMRG(; verbosity = 0), DMRG2(; verbosity = 0, trunc = truncrank(4)))
        ψ = FiniteMPS(randn, ComplexF64, 5, ℂ^2, ℂ^3)
        state = MPSKit.DMRGState(ψ, H, alg, environments(ψ, H, ψ), sweep_allocator(ψ), MPSKit.NoTimerOutput())
        dmrg_envs = state.envs
        @test isnothing(dmrg_envs.lhalves[2])
        for site in 1:2
            MPSKit.local_update!(
                site, Val(:right), ψ, H, alg, dmrg_envs,
                state.ϵ, 0.0, 0.0, 1, state.timeroutput, state.allocator
            )
            if site == 1
                # Solving the first window leaves advancement for the next call.
                @test isnothing(dmrg_envs.lhalves[2])
            else
                @test !isnothing(dmrg_envs.lhalves[2])
                @test leftenv(dmrg_envs, 2, ψ) ≈ leftenv(environments(ψ, H, ψ), 2, ψ)
            end
        end
        # Nearest-neighbor MPOs have no path through two continuing A blocks.
        alg isa DMRG2 && @test all(all(isempty, pair.channels) for pair in dmrg_envs.pairs)
    end
end
