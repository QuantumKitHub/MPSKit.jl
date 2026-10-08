# Run through jld --project=test eval --scratch:
# include("benchmark/dmrg_hybrid_profile.jl"); benchmark_dmrg_hybrid_profile(); nothing
isdefined(@__MODULE__, :benchmark_dmrg_cache_scaling) || include("dmrg_cache_scaling.jl")

csv_cell(x) = "\"" * replace(string(x), '"' => "\"\"") * "\""
function write_profile_rows(path, header, rows)
    return open(path, "w") do io
        println(io, join(header, ','))
        for row in rows
            println(io, join(csv_cell.(row), ','))
        end
    end
end

# Attribute each stack once. These are inclusive algorithm stages, not leaf costs.
function profile_stage(frames; include_gc = true)
    names = string.(getproperty.(frames, :func))
    include_gc && any(names) do n
        any(s -> occursin(s, n), ("gc_collect", "gc_mark", "gc_sweep", "gc_wait", "gc_scan", "gc_queue"))
    end && return "GC"
    any(n -> occursin("prepare_left_AC", n) || occursin("prepare_right_AC", n) || occursin("_AC_ingredients", n), names) && return "operator preparation"
    any(n -> occursin("transfer_left", n) || occursin("transfer_right", n), names) && return "environment transfers"
    any(n -> occursin("AC_hamiltonian", n) || occursin("AC2_hamiltonian", n), names) && return "effective assembly"
    any(n -> occursin("eigsolve", n), names) && return "eigensolve"
    any(n -> occursin("gauge", n), names) && return "gauge"
    any(n -> occursin("calc_galerkin", n), names) && return "Galerkin error"
    return "other"
end

function profile_site(frames; local_only = true)
    frame = findfirst(frames) do f
        file = string(f.file)
        endswith(file, ".jl") && occursin("/src/", file) && (!local_only || occursin("MPSKit", file))
    end
    isnothing(frame) && return ("unknown", 0, "unknown")
    f = frames[frame]
    return (string(f.file), f.line, string(f.func))
end

# A higher sampling rate in the chemistry cases resolves library allocation sites
# without making instrumentation part of the complete-sweep timing experiment.
function benchmark_chemistry_allocations(;
        χ = 64, sample_rate = 0.02,
        output = joinpath(@__DIR__, "results", "dmrg_hybrid_chemistry_allocations.csv"),
    )
    old_blas, old_scheduler = BLAS.get_num_threads(), MPSKit.Defaults.scheduler[]
    BLAS.set_num_threads(1)
    MPSKit.Defaults.scheduler[] = MPSKit.SerialScheduler()
    rows = []
    try
        model = last(cache_benchmark_models(; chemistry_sites = 14))
        for algorithm in ("DMRG", "DMRG2"), cached in (false, true)
            Random.seed!(17)
            ψ0 = FiniteMPS(randn, Float64, length(model.H), TensorKit.ℙ^2, TensorKit.ℙ^χ)
            eigsolve = (; adaptive = false, dynamic_tols = false, tol = 1.0e-8, krylovdim = 20, maxiter = 4)
            alg = algorithm == "DMRG" ? DMRG(; verbosity = 0, alg_eigsolve = eigsolve) :
                DMRG2(; verbosity = 0, alg_eigsolve = eigsolve, trunc = truncrank(χ))
            GC.gc()
            it = benchmark_iterator(ψ0, model.H, alg, cached)
            iterate(it)
            GC.gc()
            Profile.Allocs.clear()
            Profile.Allocs.@profile sample_rate = sample_rate iterate(it)
            path = cached ? "hybrid" : "ordinary"
            append!(rows, allocation_profile_rows(model.name, algorithm, path, sample_rate))
            Profile.Allocs.clear()
            it = nothing
            println("Detailed allocation profile: ", algorithm, " ", path)
            flush(stdout)
        end
    finally
        Profile.Allocs.clear()
        BLAS.set_num_threads(old_blas)
        MPSKit.Defaults.scheduler[] = old_scheduler
    end
    mkpath(dirname(output))
    write_profile_rows(output, ("model", "algorithm", "path", "stage", "owner_file", "owner_line", "owner_function", "leaf_file", "leaf_line", "leaf_function", "type", "sampled_allocations", "sampled_bytes", "sample_rate"), rows)
    return output
end

function cpu_profile_rows(model, algorithm, path)
    data = Profile.fetch(; include_meta = false)
    dict = Profile.getdict(data)
    groups = Dict{Tuple{String, String, Int, String}, Int}()
    frames = Base.StackTraces.StackFrame[]
    for ip in data
        if ip == 0
            if !isempty(frames)
                key = (profile_stage(frames), profile_site(frames)...)
                groups[key] = get(groups, key, 0) + 1
                empty!(frames)
            end
        else
            append!(frames, get(dict, ip, Base.StackTraces.StackFrame[]))
        end
    end
    # The daemon has an idle I/O thread. Only stacks with an MPSKit owner belong
    # to the numerical sweep; including idle stacks would dilute every share.
    return [
        (model, algorithm, path, k..., n)
            for (k, n) in sort!(collect(groups); by = last, rev = true) if k[2] != "unknown"
    ]
end

function allocation_profile_rows(model, algorithm, path, sample_rate)
    groups = Dict{Tuple, Tuple{Int, Int}}()
    for a in Profile.Allocs.fetch().allocs
        key = (
            profile_stage(a.stacktrace; include_gc = false), profile_site(a.stacktrace)...,
            profile_site(a.stacktrace; local_only = false)..., string(a.type),
        )
        count, bytes = get(groups, key, (0, 0))
        groups[key] = (count + 1, bytes + a.size)
    end
    return [
        (model, algorithm, path, k..., n, bytes, sample_rate)
            for (k, (n, bytes)) in sort!(collect(groups); by = x -> last(x)[2], rev = true)
    ]
end

"""
Profile unwrapped production sweeps for all three operator classes and both algorithms.
Repeated second-sweep measurements have no sampling instrumentation. CPU and allocation
profiles use separate fresh iterators, warmed by a first sweep, with GC before the second.
Allocation samples are uniform by allocation event, not by bytes; stage byte totals in
the timing CSV are exact cumulative allocations. A separate chemistry run measures a
sequence of sweeps without forcing GC between them.
"""
function benchmark_dmrg_hybrid_profile(;
        χ = 64, samples = 3, sample_rate = 0.002, continuous_sweeps = 8,
        output_prefix = joinpath(@__DIR__, "results", "dmrg_hybrid_profile"),
    )
    old_blas, old_scheduler = BLAS.get_num_threads(), MPSKit.Defaults.scheduler[]
    BLAS.set_num_threads(1)
    MPSKit.Defaults.scheduler[] = MPSKit.SerialScheduler()
    cpu_rows, allocation_rows, continuous_rows = [], [], []
    mkpath(dirname(output_prefix))
    try
        models = cache_benchmark_models(; chemistry_sites = 14)
        # Preserve the established complete-sweep timing protocol and ordinary controls.
        benchmark_dmrg_cache_scaling(; models, dims = (χ,), samples, sample_cpu = false, output_prefix)
        Profile.init(; n = 10^7, delay = 0.0005)
        open(output_prefix * "_cpu.txt", "w") do io
            for model in models, algorithm in ("DMRG", "DMRG2"), cached in (false, true)
                Random.seed!(17)
                ψ0 = FiniteMPS(randn, Float64, length(model.H), TensorKit.ℙ^2, TensorKit.ℙ^χ)
                eigsolve = (; adaptive = false, dynamic_tols = false, tol = 1.0e-8, krylovdim = 20, maxiter = 4)
                alg = algorithm == "DMRG" ? DMRG(; verbosity = 0, alg_eigsolve = eigsolve) :
                    DMRG2(; verbosity = 0, alg_eigsolve = eigsolve, trunc = truncrank(χ))
                path = cached ? "hybrid" : "ordinary"
                it = benchmark_iterator(copy(ψ0), model.H, alg, cached)
                iterate(it)
                GC.gc()
                Profile.clear()
                Profile.@profile iterate(it)
                append!(cpu_rows, cpu_profile_rows(model.name, algorithm, path))
                println(io, model.name, " ", algorithm, " ", path)
                Profile.print(io; format = :flat, C = true, sortedby = :count, mincount = 3)
                println(io)
                flush(io)
                it = nothing
                GC.gc()
                it = benchmark_iterator(copy(ψ0), model.H, alg, cached)
                iterate(it)
                GC.gc()
                Profile.Allocs.clear()
                Profile.Allocs.@profile sample_rate = sample_rate iterate(it)
                append!(allocation_rows, allocation_profile_rows(model.name, algorithm, path, sample_rate))
                Profile.Allocs.clear()
                it = nothing
                println("CPU and allocation profiles: ", model.name, " ", algorithm, " ", path)
                flush(stdout)
            end
        end
        model = last(models)
        reference = nothing
        for cached in (false, true)
            Random.seed!(17)
            ψ0 = FiniteMPS(randn, Float64, length(model.H), TensorKit.ℙ^2, TensorKit.ℙ^χ)
            eigsolve = (; adaptive = false, dynamic_tols = false, tol = 1.0e-8, krylovdim = 20, maxiter = 4)
            alg = DMRG2(; verbosity = 0, alg_eigsolve = eigsolve, trunc = truncrank(χ))
            GC.gc()
            it = benchmark_iterator(ψ0, model.H, alg, cached)
            for sweep in 1:continuous_sweeps
                t = @timed iterate(it)
                push!(continuous_rows, (cached ? "hybrid" : "ordinary", sweep, t.time, t.gctime, t.bytes))
                t = nothing
            end
            if cached
                @assert abs(dot(reference, it.state.mps)) ≈ norm(reference) * norm(it.state.mps)
            else
                reference = copy(it.state.mps)
            end
            it = nothing
        end
    finally
        Profile.Allocs.clear()
        BLAS.set_num_threads(old_blas)
        MPSKit.Defaults.scheduler[] = old_scheduler
    end
    write_profile_rows(output_prefix * "_cpu.csv", ("model", "algorithm", "path", "stage", "file", "line", "function", "samples"), cpu_rows)
    write_profile_rows(output_prefix * "_allocations.csv", ("model", "algorithm", "path", "stage", "owner_file", "owner_line", "owner_function", "leaf_file", "leaf_line", "leaf_function", "type", "sampled_allocations", "sampled_bytes", "sample_rate"), allocation_rows)
    write_profile_rows(output_prefix * "_continuous.csv", ("path", "sweep", "seconds", "gc_seconds", "allocated_bytes"), continuous_rows)
    println("Hybrid profiles: ", output_prefix)
    return output_prefix
end
