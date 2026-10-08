using .TestSetup
using Test, MPSKit, TensorKit, Random

# Run the same complete-sweep iterator with ordinary environments as a reference.
# This exercises the existing contraction path independently of the solve-owned cache.
function dmrg_uncached_state(ψ, H, alg)
    envs = environments(ψ, H, ψ)
    n = MPSKit._num_updates(alg, ψ)
    Tr = real(scalartype(ψ))
    return MPSKit.DMRGState(
        ψ, H, envs, 0, one(Tr), ones(Tr, n), zeros(Tr, n), zeros(n),
        MPSKit.NoTimerOutput(), MPSKit.default_allocator(ψ, MPSKit.SerialScheduler()),
    )
end

@testset "Finite DMRG sweep cache" begin
    L = 6
    nearest = force_planar(transverse_field_ising(; g = 2.0, L))
    longrange = force_planar(long_range_ising(Float64; g = 2.0, L))
    X = force_planar(S_x(ComplexF64, Trivial; spin = 1 // 2))
    mixed = nearest + FiniteMPOHamiltonian(
        fill(TensorKit.ℙ^2, L),
        (1, 2, 5) => X ⊗ X ⊗ X, (2, 3, 6) => X ⊗ X ⊗ X,
        (1, 4) => X ⊗ X, (3, 6) => X ⊗ X
    )
    models = (nearest, longrange, mixed, FiniteMPO(nearest))
    algorithms = (
        DMRG(; verbosity = 0),
        DMRG2(; verbosity = 0, trunc = truncrank(4)),
        DMRG(; verbosity = 0, alg_expand = OptimalExpand(; trunc = truncrank(1)), trunc = truncrank(4)),
        DMRG(; verbosity = 0, alg_expand = SketchedExpand(; trunc = truncrank(1)), trunc = truncrank(4)),
        DMRG(; verbosity = 0, alg_gauge = DMRG3S(0.1, ExponentialDecay(0.7)), trunc = truncrank(4)),
    )
    @testset "Transfers are independent of prepared operator contributions" for H in (mixed, FiniteMPO(mixed))
        ψ = FiniteMPS(randn, ComplexF64, L, TensorKit.ℙ^2, TensorKit.ℙ^3)
        cache = MPSKit.initialize_sweep_cache(
            ψ, H, environments(ψ, H, ψ);
            backend = MPSKit.DefaultBackend(), allocator = MPSKit.DefaultAllocator(),
        )
        i = 3
        for j in 1:(i - 1)
            MPSKit.absorb_site!(cache, ψ, j, Val(:right))
        end
        left = cache.left[i]
        side = left.one_site
        # A transfer must still work if a prepared side is corrupted. The new
        # window's contributions are prepared from its new GL/GR.
        poisoned = if H isa MPOHamiltonian
            @test !ismissing(side.continuing)
            typeof(side)(side.raw, side.prepared, zero(side.continuing))
        else
            zero(side)
        end
        cache.left[i] = typeof(left)(left.environment, poisoned, left.two_site)
        reference = MPSKit.leftenv(environments(ψ, H, ψ), i + 1, ψ)
        MPSKit.absorb_site!(cache, ψ, i, Val(:right))
        @test cache.left[i + 1].environment ≈ reference
        @test cache.environments.GLs[i + 1] === cache.left[i + 1].environment
        @test cache.environments.ldependencies[i] === ψ.AL[i]

        right = cache.right[i]
        if H isa MPOHamiltonian
            side = right.one_site
            @test !ismissing(side.continuing)
            poisoned = typeof(side)(side.raw, side.prepared, zero(side.continuing))
            cache.right[i] = typeof(right)(right.environment, poisoned, right.two_site)
        else
            cache.right[i] = typeof(right)(right.environment, right.one_site, missing)
        end
        reference = MPSKit.rightenv(environments(ψ, H, ψ), i - 1, ψ)
        MPSKit.absorb_site!(cache, ψ, i, Val(:left))
        @test cache.right[i - 1].environment ≈ reference
        @test cache.environments.GRs[i] === cache.right[i - 1].environment
        @test cache.environments.rdependencies[i] === ψ.AR[i]
    end
    @testset "Pair-only records preserve local operator queries" for H in (mixed, FiniteMPO(mixed))
        ψ = FiniteMPS(randn, ComplexF64, L, TensorKit.ℙ^2, TensorKit.ℙ^3)
        cache = MPSKit.initialize_sweep_cache(
            ψ, H, environments(ψ, H, ψ); one_site = false,
            backend = MPSKit.DefaultBackend(), allocator = MPSKit.DefaultAllocator(),
        )
        @test !cache.one_site && cache.two_site
        @test ismissing(cache.left[1].one_site)
        @test all(ismissing(r.one_site) for r in cache.right)
        held = nothing
        for (constructor, x) in ((MPSKit.AC_hamiltonian, ψ.AC[1]), (MPSKit.AC2_hamiltonian, MPSKit.AC2(ψ, 1)))
            reference = constructor(1, ψ, H, ψ, environments(ψ, H, ψ))
            for prepare in (true, false)
                op = constructor(1, ψ, H, ψ, cache; prepare)
                @test op * x ≈ reference * x
            end
            constructor === MPSKit.AC2_hamiltonian && (held = (constructor(1, ψ, H, ψ, cache), copy(x), copy(reference * x)))
        end
        # Optional AC reads must not change the pair-only preparation policy.
        @test ismissing(cache.left[1].one_site)
        alg = DMRG2(; verbosity = 0, trunc = truncrank(4))
        state = MPSKit.DMRGState(ψ, H, cache, 0, 1.0, ones(L - 1), zeros(L - 1), zeros(L - 1), MPSKit.NoTimerOutput(), cache.allocator)
        iterate(MPSKit.IterativeSolver(alg, state))
        @test held[1] * held[2] ≈ held[3]
        @test all(isnothing(r) || ismissing(r.one_site) for r in cache.left)
        @test all(ismissing(r.one_site) for r in cache.right)
    end
    @testset "$(nameof(typeof(H))) / $(nameof(typeof(alg))) / $k" for H in models, (k, alg) in enumerate(algorithms)
        original = [copy(H[i]) for i in 1:L]
        Random.seed!(42)
        ψ0 = FiniteMPS(randn, ComplexF64, L, TensorKit.ℙ^2, TensorKit.ℙ^2)
        ψ = copy(ψ0)
        envs = environments(ψ, H, ψ)
        state = MPSKit.DMRGState(
            ψ, H, alg, envs, MPSKit.default_allocator(ψ, MPSKit.SerialScheduler()),
            MPSKit.NoTimerOutput(),
        )
        @test state.envs isa MPSKit.DMRGSweepCache
        cache = state.envs
        leftboundary = cache.left[1]
        rightboundary = cache.right[end]
        cached = MPSKit.IterativeSolver(alg, state)
        reference = MPSKit.IterativeSolver(alg, dmrg_uncached_state(copy(ψ0), H, alg))
        for sweep in 1:2
            # Match randomized expansion draws in the two paths.
            Random.seed!(100 + sweep)
            iterate(cached)
            Random.seed!(100 + sweep)
            iterate(reference)
            a, b = cached.state, reference.state
            @test abs(dot(a.mps, b.mps)) ≈ norm(a.mps) * norm(b.mps) atol = 1.0e-10
            @test a.local_errors ≈ b.local_errors atol = 1.0e-10
            @test a.truncation_errors ≈ b.truncation_errors atol = 1.0e-10
            @test a.decay_rates ≈ b.decay_rates atol = 1.0e-8
            @test a.iter == b.iter == sweep
            @test cache.left[1] === leftboundary
            @test cache.right[end] === rightboundary
            @test expectation_value(a.mps, H, envs) ≈ expectation_value(b.mps, H) atol = 1.0e-10
        end
        @test MPSKit.unwrap_environments(cache) === envs
        @test all(H[i] ≈ original[i] for i in 1:L)
    end
end

@testset "Cached effective operators are independent snapshots" begin
    for H in (
            force_planar(transverse_field_ising(; g = 2.0, L = 5)),
            force_planar(long_range_ising(Float64; g = 2.0, L = 5)),
        )
        ψ = FiniteMPS(randn, ComplexF64, 5, TensorKit.ℙ^2, TensorKit.ℙ^3)
        allocator = MPSKit.default_allocator(ψ, MPSKit.SerialScheduler())
        cache = MPSKit.initialize_sweep_cache(
            ψ, H, environments(ψ, H, ψ);
            backend = MPSKit.DefaultBackend(), allocator
        )
        data = cache.operator_data
        retained = []
        for (constructor, x) in (
                (MPSKit.AC_hamiltonian, ψ.AC[1]),
                (MPSKit.AC2_hamiltonian, MPSKit.AC2(ψ, 1)),
            )
            fresh = constructor(1, ψ, H, ψ, environments(ψ, H, ψ); allocator)
            first = constructor(1, ψ, H, ψ, cache; allocator)
            y = first * x
            push!(retained, (first, copy(x), copy(y)))
            for _ in 1:3
                repeated = constructor(1, ψ, H, ψ, cache; allocator)
                raw = constructor(1, ψ, H, ψ, cache; prepare = false, allocator)
                @test repeated * x ≈ fresh * x
                # The requested matvec allocator is independent of the cache's allocator.
                @test constructor(1, ψ, H, ψ, cache; allocator = MPSKit.DefaultAllocator()) * x ≈ fresh * x
                @test raw * x ≈ fresh * x
                @test first * x ≈ y
                @test MPSKit.prepare_operator!!(raw) * x ≈ fresh * x
                @test constructor(1, ψ, H, ψ, cache; prepare = false, allocator) * x ≈ fresh * x
            end
        end
        alg = DMRG2(; verbosity = 0, trunc = truncrank(4))
        state = MPSKit.DMRGState(ψ, H, cache, 0, 1.0, ones(4), zeros(4), zeros(4), MPSKit.NoTimerOutput(), allocator)
        it = MPSKit.IterativeSolver(alg, state)
        state = MPSKit.sweep!(it, state, Val(:right), 1)
        left = copy(cache.left)
        state = MPSKit.sweep!(it, state, Val(:left), 1)
        @test all(a === b for (a, b) in zip(left, cache.left))
        @test cache.operator_data === data
        @test all(H * x ≈ y for (H, x, y) in retained)
        right = copy(cache.right)
        MPSKit.sweep!(it, state, Val(:right), 2)
        @test all(a === b for (a, b) in zip(right[1:(end - 2)], cache.right[1:(end - 2)]))
        # The terminal right record is refreshed at reversal; earlier records remain available
        # until their right-to-left update, and the boundary is never replaced.
        @test cache.right[end] === right[end]
    end
end

import MPSKit: TensorOperations
struct DMRGCountingBackend <: TensorOperations.AbstractBackend
    contractions::Base.RefValue{Int}
end
# Count the environment–MPO contraction entry points, including the reference
# preparation routines. This measures the expensive work independently of whether
# TensorKit implements a block with multiplication or an identity/permutation shortcut.
for f in (:_contract_GL_O, :_contract_O_GR, :_prepare_GL_O, :_prepare_O_GR)
    @eval function MPSKit.$f(A, B, backend::DMRGCountingBackend, allocator)
        backend.contractions[] += 1
        return MPSKit.$f(A, B, TensorOperations.DefaultBackend(), allocator)
    end
end
function TensorOperations.tensorcontract!(
        C::AbstractArray, A::AbstractArray, pA::TensorOperations.Index2Tuple, conjA::Bool,
        B::AbstractArray, pB::TensorOperations.Index2Tuple, conjB::Bool,
        pAB::TensorOperations.Index2Tuple, α::Number, β::Number,
        ::DMRGCountingBackend, allocator,
    )
    return TensorOperations.tensorcontract!(
        C, A, pA, conjA, B, pB, conjB, pAB, α, β,
        TensorOperations.DefaultBackend(), allocator
    )
end
function TensorOperations.tensoradd!(
        C::AbstractArray, A::AbstractArray, pA::TensorOperations.Index2Tuple, conjA::Bool,
        α::Number, β::Number, ::DMRGCountingBackend, allocator,
    )
    return TensorOperations.tensoradd!(C, A, pA, conjA, α, β, TensorOperations.DefaultBackend(), allocator)
end
function TensorOperations.tensortrace!(
        C::AbstractArray, A::AbstractArray, p::TensorOperations.Index2Tuple,
        q::TensorOperations.Index2Tuple, conjA::Bool, α::Number, β::Number,
        ::DMRGCountingBackend, allocator,
    )
    return TensorOperations.tensortrace!(C, A, p, q, conjA, α, β, TensorOperations.DefaultBackend(), allocator)
end

@testset "Effective operator assembly reuses environment–MPO contractions" begin
    X = S_x(Float64, Trivial; spin = 1 // 2)
    H0 = FiniteMPOHamiltonian(
        fill(ℂ^2, 6),
        (1, 2, 5) => X ⊗ X ⊗ X, (2, 3, 6) => X ⊗ X ⊗ X
    )
    for H in (H0, FiniteMPO(H0))
        ψ = FiniteMPS(randn, ComplexF64, 6, ℂ^2, ℂ^3)
        counter = Ref(0)
        backend = DMRGCountingBackend(counter)
        allocator = MPSKit.default_allocator(ψ, MPSKit.SerialScheduler())
        cache = MPSKit.initialize_sweep_cache(ψ, H, environments(ψ, H, ψ); backend, allocator)
        MPSKit.absorb_site!(cache, ψ, 1, Val(:right))
        @test counter[] > 0
        before = counter[]
        for _ in 1:3, prepare in (false, true)
            MPSKit.AC_hamiltonian(2, ψ, H, ψ, cache; prepare, backend, allocator)
            MPSKit.AC2_hamiltonian(2, ψ, H, ψ, cache; prepare, backend, allocator)
        end
        @test counter[] == before
        # Check that the instrumentation detects the contractions in the reference path.
        MPSKit.AC2_hamiltonian(2, ψ, H, ψ, environments(ψ, H, ψ); backend, allocator)
        @test counter[] > before
    end
end

@testset "Real states, short chains, and operator wrappers" begin
    for L in (1, 2, 4), T in (Float64, ComplexF64)
        H = L == 1 ? FiniteMPOHamiltonian(
                fill(ℂ^2, L),
                (1,) => -4 * S_x(Float64, Trivial; spin = 1 // 2)
            ) :
            transverse_field_ising(Float64; g = 2.0, L)
        ψ0 = FiniteMPS(randn, T, L, ℂ^2, ℂ^3)
        algorithms = L == 1 ? (DMRG(; verbosity = 0),) :
            (DMRG(; verbosity = 0), DMRG2(; verbosity = 0, trunc = truncrank(4)))
        for alg in algorithms
            ψ = copy(ψ0)
            env = environments(ψ, H, ψ)
            it = MPSKit.IterativeSolver(
                alg, MPSKit.DMRGState(
                    ψ, H, alg, env,
                    MPSKit.default_allocator(ψ, MPSKit.SerialScheduler()), MPSKit.NoTimerOutput()
                )
            )
            ref = MPSKit.IterativeSolver(alg, dmrg_uncached_state(copy(ψ0), H, alg))
            for _ in 1:2
                iterate(it)
                iterate(ref)
            end
            @test abs(dot(it.mps, ref.mps)) ≈ norm(it.mps) * norm(ref.mps) atol = 1.0e-10
            @test it.local_errors ≈ ref.local_errors atol = 1.0e-10
            @test it.truncation_errors ≈ ref.truncation_errors atol = 1.0e-10
            @test scalartype(it.mps) == T
            if alg isa DMRG
                @test all(ismissing(record.two_site) for record in it.envs.left if !isnothing(record))
            end
        end
    end
    H1 = transverse_field_ising(; g = 2.0, L = 4)
    H2 = long_range_ising(Float64; g = 1.0, L = 4)
    for H in (LazySum([H1, H2]), LazySum([MultipliedOperator(H1, 0.7), MultipliedOperator(H2, 1.3)]))
        ψ0 = FiniteMPS(randn, ComplexF64, 4, ℂ^2, ℂ^3)
        for alg in (DMRG(; verbosity = 0), DMRG2(; verbosity = 0, trunc = truncrank(4)))
            ψ = copy(ψ0)
            env = environments(ψ, H, ψ)
            it = MPSKit.IterativeSolver(
                alg, MPSKit.DMRGState(
                    ψ, H, alg, env,
                    MPSKit.default_allocator(ψ, MPSKit.SerialScheduler()), MPSKit.NoTimerOutput()
                )
            )
            ref = MPSKit.IterativeSolver(alg, dmrg_uncached_state(copy(ψ0), H, alg))
            for _ in 1:2
                iterate(it)
                iterate(ref)
            end
            @test abs(dot(it.mps, ref.mps)) ≈ norm(it.mps) * norm(ref.mps) atol = 1.0e-10
            @test it.local_errors ≈ ref.local_errors atol = 1.0e-10
            @test expectation_value(it.mps, H, MPSKit.unwrap_environments(it.envs)) ≈ expectation_value(ref.mps, H) atol = 1.0e-10
        end
    end
end

@testset "DMRG cache with SU2 symmetry" begin
    H = heisenberg_XXX(ComplexF64, SU2Irrep; spin = 1 // 2, L = 4)
    exchange = TestSetup.S_exchange(ComplexF64, SU2Irrep; spin = 1 // 2)
    H += FiniteMPOHamiltonian(
        fill(physicalspace(H, 1), 4),
        (1, 3) => exchange, (2, 4) => exchange
    )
    ψ0 = FiniteMPS(randn, ComplexF64, 4, physicalspace(H, 1), Rep[SU₂](0 => 2, 1 // 2 => 2, 1 => 1))
    for alg in (DMRG(; verbosity = 0), DMRG2(; verbosity = 0, trunc = truncrank(8)))
        ψ = copy(ψ0)
        env = environments(ψ, H, ψ)
        it = MPSKit.IterativeSolver(
            alg, MPSKit.DMRGState(
                ψ, H, alg, env,
                MPSKit.default_allocator(ψ, MPSKit.SerialScheduler()), MPSKit.NoTimerOutput()
            )
        )
        ref = MPSKit.IterativeSolver(alg, dmrg_uncached_state(copy(ψ0), H, alg))
        for _ in 1:2
            iterate(it)
            iterate(ref)
        end
        @test abs(dot(it.mps, ref.mps)) ≈ norm(it.mps) * norm(ref.mps) atol = 1.0e-10
        @test it.local_errors ≈ ref.local_errors atol = 1.0e-10
        @test it.truncation_errors ≈ ref.truncation_errors atol = 1.0e-10
    end
end

@testset "Window DMRG keeps its supplied boundaries" begin
    Hinf = force_planar(transverse_field_ising(; g = 2.0))
    gs = InfiniteMPS(randn, ComplexF64, [TensorKit.ℙ^2], [TensorKit.ℙ^3])
    ψ0 = WindowMPS(gs, 4)
    boundaries = environments(ψ0, Hinf, ψ0)
    for H in (Hinf, WindowMPOHamiltonian(Hinf, 1:4))
        alg = DMRG(; verbosity = 0)
        ψ, ψref = copy(ψ0), copy(ψ0)
        op = H isa WindowMPOHamiltonian ? H.finite_ham : H
        env = MPSKit.initialize_environments(
            ψ, op, ψ,
            copy(boundaries.GLs[1]), copy(boundaries.GRs[end])
        )
        envref = MPSKit.initialize_environments(
            ψref, op, ψref,
            copy(boundaries.GLs[1]), copy(boundaries.GRs[end])
        )
        state = MPSKit.DMRGState(
            ψ, H, alg, env,
            MPSKit.default_allocator(ψ, MPSKit.SerialScheduler()), MPSKit.NoTimerOutput()
        )
        n = MPSKit._num_updates(alg, ψref)
        reference = MPSKit.DMRGState(
            ψref, H, envref, 0, 1.0, ones(n), zeros(n), zeros(n),
            MPSKit.NoTimerOutput(), MPSKit.default_allocator(ψref, MPSKit.SerialScheduler())
        )
        it, ref = MPSKit.IterativeSolver(alg, state), MPSKit.IterativeSolver(alg, reference)
        for _ in 1:2
            iterate(it)
            iterate(ref)
        end
        @test abs(dot(it.mps, ref.mps)) ≈ norm(it.mps) * norm(ref.mps) atol = 1.0e-10
        @test it.local_errors ≈ ref.local_errors atol = 1.0e-10
        @test it.truncation_errors ≈ ref.truncation_errors atol = 1.0e-10
        @test state.envs.left[1].environment ≈ boundaries.GLs[1]
        @test state.envs.right[end].environment ≈ boundaries.GRs[end]
    end
end

@testset "Read-only finalizers and returned environments" begin
    H = long_range_ising(Float64; g = 2.0, L = 6)
    ψ0 = FiniteMPS(randn, ComplexF64, 6, ℂ^2, ℂ^3)
    seen = Int[]
    finalize = function (iter, ψ, H, envs)
        push!(seen, iter)
        # Ordinary read access can materialize environments beyond the sweep center.
        for i in 1:length(ψ)
            leftenv(envs, i, ψ)
            rightenv(envs, i, ψ)
        end
        return ψ, envs
    end
    alg = DMRG2(; verbosity = 0, maxiter = 2, trunc = truncrank(4), finalize)
    ψ, envs, info = find_groundstate(ψ0, H, alg)
    ref, _, refinfo = find_groundstate(
        ψ0, H,
        DMRG2(; verbosity = 0, maxiter = 2, trunc = truncrank(4))
    )
    @test seen == collect(1:info.numiter)
    @test info.numiter == refinfo.numiter
    @test abs(dot(ψ, ref)) ≈ norm(ψ) * norm(ref) atol = 1.0e-10
    @test envs isa MPSKit.FiniteEnvironments
    @test all(leftenv(envs, i, ψ) ≈ leftenv(environments(ψ, H, ψ), i, ψ) for i in 1:length(ψ))
    @test all(rightenv(envs, i, ψ) ≈ rightenv(environments(ψ, H, ψ), i, ψ) for i in 1:length(ψ))
end

@testset "Environment records advance at the next local window" begin
    H = transverse_field_ising(; g = 2.0, L = 5)
    for alg in (DMRG(; verbosity = 0), DMRG2(; verbosity = 0, trunc = truncrank(4)))
        ψ = FiniteMPS(randn, ComplexF64, 5, ℂ^2, ℂ^3)
        state = MPSKit.DMRGState(
            ψ, H, alg, environments(ψ, H, ψ),
            MPSKit.default_allocator(ψ, MPSKit.SerialScheduler()), MPSKit.NoTimerOutput()
        )
        cache = state.envs
        @test isnothing(cache.left[2])
        for site in 1:2
            MPSKit.local_update!(
                site, Val(:right), ψ, H, alg, cache,
                state.ϵ, 0.0, 0.0, 1, state.timeroutput, state.allocator
            )
            if site == 1
                # Solving the first window leaves advancement for the next call.
                @test isnothing(cache.left[2])
            else
                @test !isnothing(cache.left[2])
                @test leftenv(cache, 2, ψ) ≈ leftenv(environments(ψ, H, ψ), 2, ψ)
            end
        end
        # Nearest-neighbor MPOs have no path through two continuing A blocks.
        @test all(all(isempty, pair.channels) for pair in cache.operator_data.pairs)
    end
end
