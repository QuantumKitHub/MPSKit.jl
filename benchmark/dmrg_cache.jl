# Standalone, dependency-light cache benchmark. Run through jld with --project=test:
# eval --scratch 'include("benchmark/dmrg_cache.jl"); benchmark_dmrg_cache(); nothing'
# Configuration: DMRG_BENCH_SAMPLES (5), DMRG_BENCH_DIMS (16,32),
# DMRG_BENCH_OUTPUT (benchmark/results/dmrg_cache.csv).
using MPSKit, TensorKit, LinearAlgebra, Random, Statistics, Printf

function product_term(indices, factors, coefficient)
    tensors = [MPSKit.add_util_leg(A) for A in factors]
    tensors[end] = coefficient * tensors[end]
    return collect(indices) => FiniteMPO(tensors)
end

function cache_benchmark_models(; chemistry_sites = 12)
    rng = MersenneTwister(20261006)
    V = TensorKit.ℙ^2
    X = TensorMap(vec([0.0 1.0; 1.0 0.0]), V ← V)
    Z = TensorMap(vec([1.0 0.0; 0.0 -1.0]), V ← V)
    identity = TensorMap(vec(Matrix{Float64}(I, 2, 2)), V ← V)
    creation = TensorMap(vec([0.0 0.0; 1.0 0.0]), V ← V)
    annihilation = copy(creation')
    number = creation * annihilation

    L = 24
    nearest_terms = [product_term((i,), (X,), -1.3) for i in 1:L]
    append!(nearest_terms, [product_term((i, i + 1), (Z, Z), -1.0) for i in 1:(L - 1)])
    nearest = FiniteMPOHamiltonian(fill(V, L), nearest_terms)

    # Many independent long-range strings, with tensor-valued continuing blocks.
    supports = Set{NTuple{4, Int}}()
    while length(supports) < 150
        push!(supports, Tuple(sort(randperm(rng, L)[1:4])))
    end
    sparse_terms = copy(nearest_terms)
    for support in sort!(collect(supports))
        push!(sparse_terms, product_term(support, (X, Z, Z, X), 0.2 * randn(rng)))
    end
    sparse = FiniteMPOHamiltonian(fill(V, L), sparse_terms)

    # Spinless-orbital surrogate: all-pairs hopping/density interactions and
    # c†_i c†_j c_k c_l + h.c. for every i < j < k < l, with seeded random integrals.
    # Jordan-Wigner strings are multiplied explicitly, retaining their signs.
    # This exercises chemistry-like four-orbital terms, not molecular integrals
    # or a specialized complementary-operator quantum-chemistry MPO.
    Lq = chemistry_sites
    chemistry_terms = [product_term((i,), (number,), randn(rng)) for i in 1:Lq]
    function fermionic_term(operators, coefficient)
        factors = [copy(identity) for _ in 1:Lq]
        for (site, dagger) in operators
            for j in 1:(site - 1)
                factors[j] = factors[j] * Z
            end
            factors[site] = factors[site] * (dagger ? creation : annihilation)
        end
        support = findall(A -> A != identity, factors)
        return product_term(support, factors[support], coefficient)
    end
    for i in 1:Lq, j in (i + 1):Lq
        t = randn(rng) / sqrt(Lq)
        push!(chemistry_terms, fermionic_term(((i, true), (j, false)), t))
        push!(chemistry_terms, fermionic_term(((j, true), (i, false)), t))
        push!(chemistry_terms, product_term((i, j), (number, number), randn(rng) / Lq))
    end
    for i in 1:Lq, j in (i + 1):Lq, k in (j + 1):Lq, l in (k + 1):Lq
        v = 0.2 * randn(rng) / Lq
        push!(chemistry_terms, fermionic_term(((i, true), (j, true), (k, false), (l, false)), v))
        push!(chemistry_terms, fermionic_term(((l, true), (k, true), (j, false), (i, false)), v))
    end
    chemistry = FiniteMPOHamiltonian(fill(V, Lq), chemistry_terms)
    return (
        (name = "nearest_neighbor", H = nearest, terms = length(nearest_terms)),
        (name = "large_sparse", H = sparse, terms = length(sparse_terms)),
        (name = "chemistry_like", H = chemistry, terms = length(chemistry_terms)),
    )
end

function benchmark_iterator(ψ, H, alg, cached; timer = MPSKit.NoTimerOutput())
    envs = environments(ψ, H, ψ)
    allocator = MPSKit.default_allocator(ψ, MPSKit.SerialScheduler())
    state = if cached
        MPSKit.DMRGState(ψ, H, alg, envs, allocator, timer)
    else
        # Identical complete-sweep iterator, bypassing only the solve-owned cache.
        n = MPSKit._num_updates(alg, ψ)
        MPSKit.DMRGState(ψ, H, envs, 0, 1.0, ones(n), zeros(n), zeros(n), timer, allocator)
    end
    return MPSKit.IterativeSolver(alg, state)
end

function sweep_sample(ψ0, H, alg, cached)
    ψ = copy(ψ0) # Excluded from all timings.
    GC.gc()
    setup = @timed benchmark_iterator(ψ, H, alg, cached)
    it = setup.value
    first = @timed iterate(it)
    second = @timed iterate(it)
    return (; setup, first, second, it)
end

function benchmark_dmrg_cache()
    samples = parse(Int, get(ENV, "DMRG_BENCH_SAMPLES", "5"))
    dims = parse.(Int, split(get(ENV, "DMRG_BENCH_DIMS", "16,32"), ','))
    output = get(ENV, "DMRG_BENCH_OUTPUT", joinpath(@__DIR__, "results", "dmrg_cache.csv"))
    old_blas = BLAS.get_num_threads()
    old_scheduler = MPSKit.Defaults.scheduler[]
    BLAS.set_num_threads(1)
    MPSKit.Defaults.scheduler[] = MPSKit.SerialScheduler()
    rows = []
    function record(model, χ, algorithm, path, phase, sample, measurement)
        return push!(
            rows, (
                model.name, length(model.H), model.terms,
                maximum(dim(right_virtualspace(W)) for W in model.H),
                sum(MPSKit.nonzero_length(W.tensors) + length(W.scalars) for W in model.H),
                sum(size(W, 1) * size(W, 4) for W in model.H),
                χ, algorithm, path, phase, sample,
                measurement.time, measurement.bytes, measurement.gctime,
            )
        )
    end
    try
        println("Julia ", VERSION, "; Julia threads=", Threads.nthreads(), "; BLAS threads=1")
        println("CPU: ", Sys.cpu_info()[1].model, "; samples=", samples)
        println("Building seeded benchmark operators...")
        flush(stdout)
        for model in cache_benchmark_models()
            H = model.H
            println(
                model.name, ": L=", length(H), ", terms=", model.terms,
                ", max MPO bond=", maximum(dim(right_virtualspace(W)) for W in H)
            )
            flush(stdout)
            for χ in dims, algorithm in ("DMRG", "DMRG2")
                Random.seed!(17)
                ψ0 = FiniteMPS(randn, Float64, length(H), TensorKit.ℙ^2, TensorKit.ℙ^χ)
                eigsolve = (; adaptive = false, dynamic_tols = false, tol = 1.0e-8, krylovdim = 20, maxiter = 4)
                alg = algorithm == "DMRG" ? DMRG(; verbosity = 0, alg_eigsolve = eigsolve) :
                    DMRG2(; verbosity = 0, alg_eigsolve = eigsolve, trunc = truncrank(χ))
                # Warm both code paths, verify both sweeps yield the same state/errors.
                uncached = sweep_sample(ψ0, H, alg, false)
                cached = sweep_sample(ψ0, H, alg, true)
                a, b = uncached.it.state, cached.it.state
                overlap = abs(dot(a.mps, b.mps)) / (norm(a.mps) * norm(b.mps))
                @assert isapprox(overlap, 1; atol = 1.0e-8)
                @assert isapprox(a.local_errors, b.local_errors; atol = 1.0e-7)
                for sample in 1:samples
                    for use_cache in (isodd(sample) ? (false, true) : (true, false))
                        result = sweep_sample(ψ0, H, alg, use_cache)
                        for phase in (:setup, :first, :second)
                            record(model, χ, algorithm, use_cache ? "cached" : "ordinary", string(phase), sample, getproperty(result, phase))
                        end
                    end
                end
                for path in ("ordinary", "cached")
                    selected = filter(r -> r[1] == model.name && r[7] == χ && r[8] == algorithm && r[9] == path, rows)
                    times = [median(r[12] for r in selected if r[10] == phase) for phase in ("setup", "first", "second")]
                    @printf(
                        "  %s χ=%d %s: setup %.3fs, first %.3fs, second %.3fs (overlap %.12f)\n",
                        algorithm, χ, path, times..., overlap
                    )
                end
                flush(stdout)
            end
            # Save completed models even if a later model fails or is interrupted.
            mkpath(dirname(output))
            open(output, "w") do io
                println(io, "model,sites,terms,max_mpo_bond,nonzero_blocks,possible_blocks,mps_bond,algorithm,path,phase,sample,seconds,allocated_bytes,gc_seconds")
                for row in rows
                    println(io, join(row, ','))
                end
            end
        end
        println("Raw results: ", output)
    finally
        BLAS.set_num_threads(old_blas)
        MPSKit.Defaults.scheduler[] = old_scheduler
    end
    return rows
end
