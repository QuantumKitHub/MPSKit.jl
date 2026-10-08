# jld --project=test eval --scratch:
# include("benchmark/dmrg_pair_preparation.jl"); benchmark_dmrg_pair_preparation(); nothing
include("dmrg_operator_profile.jl")
include("dmrg_hybrid_profile.jl")

# All paths use the same solver and transfers. `both` explicitly restores the
# previous preparation policy; `pairs` uses the new plain-DMRG2 default.
function preparation_iterator(ψ, H, alg, mode; timer = MPSKit.NoTimerOutput(), wrapped = false)
    if mode == "both"
        envs = environments(ψ, H, ψ)
        allocator = MPSKit.default_allocator(ψ, MPSKit.SerialScheduler())
        cache = MPSKit.initialize_sweep_cache(ψ, H, envs; alg.backend, allocator, one_site = true, two_site = true)
        n = length(ψ) - 1
        state = MPSKit.DMRGState(ψ, H, cache, 0, 1.0, ones(n), zeros(n), zeros(n), timer, allocator)
        it = MPSKit.IterativeSolver(alg, state)
    else
        it = benchmark_iterator(ψ, H, alg, mode == "pairs"; timer)
    end
    if wrapped
        s = it.state
        state = MPSKit.DMRGState(
            s.mps, s.operator, ProfiledDMRGEnvironments(s.envs, timer), s.iter, s.ϵ,
            s.local_errors, s.truncation_errors, s.decay_rates, timer, s.allocator,
        )
        it = MPSKit.IterativeSolver(alg, state)
    end
    return it
end

function preparation_sample(ψ0, H, alg, mode; wrapped = false, diagnostics = false)
    ψ = copy(ψ0)
    timer = MPSKit.TimerOutput()
    GC.gc()
    setup = @timed preparation_iterator(ψ, H, alg, mode; timer, wrapped)
    it = setup.value
    first = @timed iterate(it)
    wrapped && GC.gc()
    tm = parentmodule(typeof(timer))
    tm.reset_timer!(timer)
    counter = SweepMatvecCounter(0, 0)
    second = if diagnostics
        with_logger(counter) do
            @timed iterate(it)
        end
    else
        @timed iterate(it)
    end
    return (; setup, first, second, it, timer, counter)
end

function benchmark_dmrg_pair_preparation(;
        χ = 64, samples = 3, continuous_sweeps = 8, sample_rate = 0.01,
        output_prefix = joinpath(@__DIR__, "results", "dmrg_pair_preparation"),
    )
    old_blas, old_scheduler = BLAS.get_num_threads(), MPSKit.Defaults.scheduler[]
    BLAS.set_num_threads(1)
    MPSKit.Defaults.scheduler[] = MPSKit.SerialScheduler()
    rows, stages, agreements, construction, continuous, allocations = [], [], [], [], [], []
    mkpath(dirname(output_prefix))
    try
        model = last(cache_benchmark_models(; chemistry_sites = 14))
        H = model.H
        Random.seed!(17)
        ψ0 = FiniteMPS(randn, Float64, length(H), TensorKit.ℙ^2, TensorKit.ℙ^χ)
        eigsolve = (; adaptive = false, dynamic_tols = false, tol = 1.0e-8, krylovdim = 20, maxiter = 4)
        alg = DMRG2(; verbosity = 0, alg_eigsolve = eigsolve, trunc = truncrank(χ))
        modes = ("ordinary", "both", "pairs")
        warmed = map(modes) do mode
            println("Warming ", mode)
            flush(stdout)
            preparation_sample(ψ0, H, alg, mode; diagnostics = true)
        end
        reference = first(warmed)
        for (mode, result) in zip(modes, warmed)
            overlap = abs(dot(reference.it.state.mps, result.it.state.mps)) /
                (norm(reference.it.state.mps) * norm(result.it.state.mps))
            error = maximum(abs.(reference.it.state.local_errors - result.it.state.local_errors))
            @assert isapprox(overlap, 1; atol = 1.0e-8)
            @assert error < 1.0e-7
            @assert reference.counter.matvecs == result.counter.matvecs
            push!(agreements, (mode, overlap, error, result.counter.matvecs))
        end
        warmed = reference = result = nothing
        tm = parentmodule(MPSKit.TimerOutput)
        orders = (modes, reverse(modes), ("both", "ordinary", "pairs"))
        for sample in 1:samples, mode in orders[mod1(sample, length(orders))]
            result = preparation_sample(ψ0, H, alg, mode)
            for phase in (:setup, :first, :second)
                t = getproperty(result, phase)
                push!(rows, (mode, sample, phase, t.time, t.gctime, t.bytes))
            end
            sweep = result.timer["sweep"]
            for stage in ("advance_env", "AC2_hamiltonian", "AC2_eigsolve", "gauge")
                t = sweep[stage]
                push!(stages, (mode, sample, stage, tm.time(t) / 1.0e9, tm.allocated(t)))
            end
            println("Timed sample ", sample, " ", mode)
            flush(stdout)
            result = nothing
        end
        for sample in 1:samples, mode in (isodd(sample) ? ("both", "pairs") : ("pairs", "both"))
            # Separate stage diagnostics, with GC before the measured sweep.
            reference = preparation_sample(ψ0, H, alg, mode; diagnostics = true)
            result = preparation_sample(ψ0, H, alg, mode; wrapped = true, diagnostics = true)
            @assert abs(dot(reference.it.state.mps, result.it.state.mps)) ≈ norm(reference.it.state.mps) * norm(result.it.state.mps)
            @assert reference.it.state.local_errors ≈ result.it.state.local_errors
            @assert reference.counter.matvecs == result.counter.matvecs
            sweep = result.timer["sweep"]
            for (parent, stage) in (
                    ("advance_env", "environment transfers"), ("advance_env", "one-site preparation"),
                    ("advance_env", "two-site preparation"), ("AC2_hamiltonian", "effective assembly"),
                )
                t = sweep[parent][stage]
                push!(construction, (mode, sample, stage, tm.time(t) / 1.0e9, tm.allocated(t), result.second.time, result.second.gctime))
            end
            reference = result = nothing
        end
        reference = nothing
        for mode in modes
            GC.gc()
            it = preparation_iterator(copy(ψ0), H, alg, mode)
            for sweep in 1:continuous_sweeps
                t = @timed iterate(it)
                push!(continuous, (mode, sweep, t.time, t.gctime, t.bytes))
                t = nothing
            end
            if isnothing(reference)
                reference = copy(it.state.mps)
            else
                @assert abs(dot(reference, it.state.mps)) ≈ norm(reference) * norm(it.state.mps)
            end
            it = nothing
        end
        for mode in ("both", "pairs")
            GC.gc()
            it = preparation_iterator(copy(ψ0), H, alg, mode)
            iterate(it)
            GC.gc()
            Profile.Allocs.clear()
            Profile.Allocs.@profile sample_rate = sample_rate iterate(it)
            append!(allocations, allocation_profile_rows(model.name, "DMRG2", mode, sample_rate))
            Profile.Allocs.clear()
            it = nothing
        end
    finally
        Profile.Allocs.clear()
        BLAS.set_num_threads(old_blas)
        MPSKit.Defaults.scheduler[] = old_scheduler
    end
    write_profile_rows(output_prefix * ".csv", ("mode", "sample", "phase", "seconds", "gc_seconds", "allocated_bytes"), rows)
    write_profile_rows(output_prefix * "_stages.csv", ("mode", "sample", "stage", "seconds", "allocated_bytes"), stages)
    write_profile_rows(output_prefix * "_agreement.csv", ("mode", "overlap", "max_local_error_difference", "eigensolver_matvecs"), agreements)
    write_profile_rows(output_prefix * "_construction.csv", ("mode", "sample", "stage", "seconds", "allocated_bytes", "sweep_seconds", "gc_seconds"), construction)
    write_profile_rows(output_prefix * "_continuous.csv", ("mode", "sweep", "seconds", "gc_seconds", "allocated_bytes"), continuous)
    write_profile_rows(output_prefix * "_allocations.csv", ("model", "algorithm", "path", "stage", "owner_file", "owner_line", "owner_function", "leaf_file", "leaf_line", "leaf_function", "type", "sampled_allocations", "sampled_bytes", "sample_rate"), allocations)
    return output_prefix
end
