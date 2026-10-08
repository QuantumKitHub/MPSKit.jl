# Run via jld --project=test eval --scratch:
# include("benchmark/dmrg_cache_scaling.jl"); benchmark_dmrg_cache_scaling(); nothing
include("dmrg_cache.jl")
using Profile, Logging

mutable struct SweepMatvecCounter <: AbstractLogger
    updates::Int
    matvecs::Int
end
Logging.min_enabled_level(::SweepMatvecCounter) = Logging.Debug
Logging.shouldlog(::SweepMatvecCounter, level, mod, group, id) = mod === MPSKit
Logging.catch_exceptions(::SweepMatvecCounter) = false
function Logging.handle_message(logger::SweepMatvecCounter, level, message, mod, group, id, file, line; kwargs...)
    if haskey(kwargs, :numops)
        logger.updates += 1
        logger.matvecs += kwargs[:numops]
    elseif level >= Logging.Warn
        Logging.handle_message(ConsoleLogger(stderr), level, message, mod, group, id, file, line; kwargs...)
    end
    return nothing
end

max_mps_bond(ψ) = maximum(dim(right_virtualspace(ψ, i)) for i in 1:(length(ψ) - 1))

function scaling_sample(ψ0, H, alg, cached; cpu_io = nothing, sample_cpu::Bool = true)
    ψ = copy(ψ0)
    actual_setup = max_mps_bond(ψ)
    timer = MPSKit.TimerOutput()
    GC.gc()
    setup = @timed benchmark_iterator(ψ, H, alg, cached; timer)
    it = setup.value
    counter = SweepMatvecCounter(0, 0)
    first = if isnothing(cpu_io)
        @timed iterate(it)
    else
        # Warm the debug-counter path before enabling CPU sampling.
        with_logger(counter) do
            @timed iterate(it)
        end
    end
    actual_first = max_mps_bond(ψ)
    timer_module = parentmodule(typeof(timer))
    timer_module.reset_timer!(timer)
    # No extra GC between the two sweeps: match the original timing protocol.
    counter.updates = counter.matvecs = 0
    second = if isnothing(cpu_io)
        @timed iterate(it)
    else
        with_logger(counter) do
            if sample_cpu
                Profile.clear()
                @timed Profile.@profile iterate(it)
            else
                @timed iterate(it)
            end
        end
    end
    actual_second = max_mps_bond(ψ)
    # Inclusive stage times: children are not added to their parent again.
    sweep = timer["sweep"]
    labels = alg isa DMRG ? ("advance_env", "AC_eigsolve", "gauge") :
        ("advance_env", "AC2_hamiltonian", "AC2_eigsolve", "gauge")
    stages = [
        (
            stage = label, seconds = timer_module.time(sweep[label]) / 1.0e9,
            bytes = timer_module.allocated(sweep[label]), calls = timer_module.ncalls(sweep[label]),
        )
            for label in labels
    ]
    push!(
        stages, (
            stage = "other", seconds = (timer_module.time(sweep) / 1.0e9) - sum(s.seconds for s in stages),
            bytes = timer_module.allocated(sweep) - sum(s.bytes for s in stages), calls = 1,
        )
    )
    if !isnothing(cpu_io)
        println(
            cpu_io, "Local updates=", counter.updates, "; eigensolver matvecs=", counter.matvecs,
            "; actual maximum MPS bond=", actual_second
        )
        MPSKit.print_timer(cpu_io, timer)
        if sample_cpu
            println(cpu_io, "\nCPU samples (5 ms interval; inclusive flat counts):")
            Profile.print(cpu_io; format = :flat, sortedby = :count, mincount = 5, C = false)
        else
            println(cpu_io, "\nCPU sampling disabled; warm-up timers and solver counts only.")
        end
        println(cpu_io)
        flush(cpu_io)
    end
    return (;
        setup, first, second, it, stages, counter,
        actual_setup, actual_first, actual_second,
    )
end

function save_scaling_rows(path, header, rows)
    mkpath(dirname(path))
    return open(path, "w") do io
        println(io, header)
        for row in rows
            println(io, join(row, ','))
        end
    end
end

function benchmark_dmrg_cache_scaling(;
        dims = (64, 128), samples = 3,
        algorithms = ("DMRG", "DMRG2"),
        sample_cpu::Bool = true,
        output_prefix = joinpath(@__DIR__, "results", "dmrg_cache_scaling"),
        models = nothing,
    )
    old_blas = BLAS.get_num_threads()
    old_scheduler = MPSKit.Defaults.scheduler[]
    BLAS.set_num_threads(1)
    MPSKit.Defaults.scheduler[] = MPSKit.SerialScheduler()
    sample_cpu && Profile.init(; n = 10^7, delay = 0.005)
    rows, stage_rows, agreements = [], [], []
    try
        models = if isnothing(models)
            original = cache_benchmark_models()
            larger = last(cache_benchmark_models(; chemistry_sites = 14))
            (original..., merge(larger, (; name = "chemistry_like_14")))
        else
            models
        end
        mkpath(dirname(output_prefix))
        open(output_prefix * "_profile.txt", "w") do cpu_io
            println(
                cpu_io, "Julia ", VERSION, "; CPU ", Sys.cpu_info()[1].model,
                "; Julia threads=", Threads.nthreads(), "; BLAS threads=1; CPU sampling=", sample_cpu
            )
            for model in models
                H = model.H
                # The 12-site model saturates at 64. Add χ=32 on the new 14-site
                # model to distinguish increasing bond dimension from a bigger MPO.
                model_dims = model.name == "chemistry_like_14" ? unique((32, dims...)) : dims
                for χ in model_dims
                    χ > 2^(length(H) ÷ 2) && continue
                    for algorithm in algorithms
                        @printf("%s L=%d %s χ=%d: warming%s...\n", model.name, length(H), algorithm, χ, sample_cpu ? " and CPU profiling" : "")
                        flush(stdout)
                        Random.seed!(17)
                        ψ0 = FiniteMPS(randn, Float64, length(H), TensorKit.ℙ^2, TensorKit.ℙ^χ)
                        eigsolve = (; adaptive = false, dynamic_tols = false, tol = 1.0e-8, krylovdim = 20, maxiter = 4)
                        alg = algorithm == "DMRG" ? DMRG(; verbosity = 0, alg_eigsolve = eigsolve) :
                            DMRG2(; verbosity = 0, alg_eigsolve = eigsolve, trunc = truncrank(χ))
                        warmed = map((false, true)) do cached
                            println(
                                cpu_io, "\n", model.name, " L=", length(H), " ", algorithm,
                                " requested χ=", χ, " ", cached ? "cached" : "ordinary"
                            )
                            scaling_sample(ψ0, H, alg, cached; cpu_io, sample_cpu)
                        end
                        a, b = warmed[1].it.state, warmed[2].it.state
                        overlap = abs(dot(a.mps, b.mps)) / (norm(a.mps) * norm(b.mps))
                        error_difference = maximum(abs, a.local_errors - b.local_errors)
                        @assert isapprox(overlap, 1; atol = 1.0e-8)
                        @assert isapprox(a.local_errors, b.local_errors; atol = 1.0e-7)
                        @assert warmed[1].counter.updates == warmed[2].counter.updates
                        push!(
                            agreements, (
                                model.name, length(H), χ, algorithm, overlap,
                                error_difference, warmed[1].counter.updates,
                                warmed[1].counter.matvecs, warmed[2].counter.matvecs,
                            )
                        )
                        # Keep only scalar diagnostics while timing fresh independent runs.
                        warmed = a = b = nothing
                        for sample in 1:samples
                            for cached in (isodd(sample) ? (false, true) : (true, false))
                                result = scaling_sample(ψ0, H, alg, cached)
                                path = cached ? "cached" : "ordinary"
                                for phase in (:setup, :first, :second)
                                    measurement = getproperty(result, phase)
                                    actual = getproperty(result, Symbol("actual_", phase))
                                    push!(
                                        rows, (
                                            model.name, length(H), model.terms,
                                            maximum(dim(right_virtualspace(W)) for W in H),
                                            χ, actual, algorithm, path, string(phase), sample,
                                            measurement.time, measurement.bytes, measurement.gctime,
                                        )
                                    )
                                end
                                for s in result.stages
                                    push!(
                                        stage_rows, (
                                            model.name, length(H), χ, result.actual_second,
                                            algorithm, path, sample, s.stage, s.seconds, s.bytes, s.calls,
                                        )
                                    )
                                end
                                result = nothing
                            end
                            @printf("  completed sample %d/%d\n", sample, samples)
                            flush(stdout)
                        end
                        save_scaling_rows(
                            output_prefix * ".csv",
                            "model,sites,terms,max_mpo_bond,requested_mps_bond,actual_mps_bond,algorithm,path,phase,sample,seconds,allocated_bytes,gc_seconds", rows
                        )
                        save_scaling_rows(
                            output_prefix * "_stages.csv",
                            "model,sites,requested_mps_bond,actual_mps_bond,algorithm,path,sample,stage,seconds,allocated_bytes,calls", stage_rows
                        )
                        save_scaling_rows(
                            output_prefix * "_agreement.csv",
                            "model,sites,requested_mps_bond,algorithm,overlap,max_local_error_difference,updates,ordinary_eigensolver_matvecs,cached_eigensolver_matvecs", agreements
                        )
                        for path in ("ordinary", "cached")
                            selected = filter(r -> r[1] == model.name && r[5] == χ && r[7] == algorithm && r[8] == path && r[9] == "second", rows)
                            @printf(
                                "  %s: second sweep %.3fs, %.1f MiB allocated, actual χ=%d\n",
                                path, median(r[11] for r in selected), median(r[12] for r in selected) / 2.0^20, selected[1][6]
                            )
                        end
                        flush(stdout)
                    end
                end
            end
        end
    finally
        BLAS.set_num_threads(old_blas)
        MPSKit.Defaults.scheduler[] = old_scheduler
    end
    profile_path = output_prefix * "_profile.txt"
    write(profile_path, rstrip(read(profile_path, String)) * "\n")
    println("Scaling results: ", output_prefix)
    return (; rows, stage_rows, agreements)
end
