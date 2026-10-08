# Compare frozen main/latest checkouts using the same manifest and serialized inputs.
# Load once in each daemon; execute timing requests sequentially in alternating order.
include("dmrg_cache.jl")
using Serialization, Pkg, Logging

mutable struct MainSweepRunner{S, O, A, E, T, B}
    mps::S
    operator::O
    alg::A
    envs::E
    iter::Int
    error::Float64
    local_errors::Vector{Float64}
    truncation_errors::Vector{Float64}
    decay_rates::Vector{Float64}
    timer::T
    allocator::B
end

mutable struct ComparisonCounter <: AbstractLogger
    updates::Int
    matvecs::Int
end
Logging.min_enabled_level(::ComparisonCounter) = Logging.Debug
Logging.shouldlog(::ComparisonCounter, level, mod, group, id) = mod === MPSKit && group === :dmrg
Logging.catch_exceptions(::ComparisonCounter) = false
function Logging.handle_message(counter::ComparisonCounter, level, message, mod, group, id, file, line; kwargs...)
    if haskey(kwargs, :numops)
        counter.updates += 1
        counter.matvecs += kwargs[:numops]
    elseif level >= Logging.Warn
        Logging.handle_message(ConsoleLogger(stderr), level, message, mod, group, id, file, line; kwargs...)
    end
    return nothing
end

# Reproduce main's complete-sweep loop, calling its production local_update!.
# The adapter is checked against main's public find_groundstate! before measuring it.
function comparison_sweep!(runner::MainSweepRunner)
    (; mps, operator, alg, envs, timer, allocator) = runner
    runner.iter += 1
    fwd, bwd = MPSKit._sweep_ranges(alg, mps)
    MPSKit.@timeit timer "sweep" begin
        for (direction, sites) in ((Val(:right), fwd), (Val(:left), bwd)), pos in sites
            mps, runner.local_errors[pos], runner.truncation_errors[pos], runner.decay_rates[pos] =
                MPSKit.local_update!(
                pos, direction, mps, operator, alg, envs,
                runner.error, runner.truncation_errors[pos], runner.decay_rates[pos],
                runner.iter, timer, allocator,
            )
            runner.error = maximum(runner.local_errors)
        end
    end
    mps, envs = alg.finalize(runner.iter, mps, operator, envs)
    @assert mps === runner.mps && envs === runner.envs
    return runner
end
comparison_sweep!(runner) = (iterate(runner); runner)
comparison_mps(runner::MainSweepRunner) = runner.mps
comparison_mps(runner) = runner.state.mps
comparison_errors(runner::MainSweepRunner) = runner.local_errors
comparison_errors(runner) = runner.state.local_errors
comparison_error(runner::MainSweepRunner) = runner.error
comparison_error(runner) = runner.state.ϵ
comparison_bond(ψ) = maximum(dim(right_virtualspace(ψ, i)) for i in 1:(length(ψ) - 1))

function comparison_alg(χ; sweeps = 4)
    return DMRG2(;
        verbosity = 0, maxiter = sweeps, tol = 0.0, trunc = truncrank(χ),
        alg_eigsolve = (; adaptive = false, dynamic_tols = false, tol = 1.0e-8, krylovdim = 20, maxiter = 4),
    )
end
function comparison_runner(ψ, H, alg; timer = MPSKit.NoTimerOutput())
    envs = environments(ψ, H, ψ)
    allocator = MPSKit.default_allocator(ψ, MPSKit.SerialScheduler())
    if isdefined(MPSKit, :DMRGState)
        state = MPSKit.DMRGState(ψ, H, alg, envs, allocator, timer)
        return MPSKit.IterativeSolver(alg, state)
    end
    n = MPSKit._num_updates(alg, ψ)
    return MainSweepRunner(ψ, H, alg, envs, 0, 1.0, ones(n), zeros(n), zeros(n), timer, allocator)
end
function comparison_inputs(path; sites = 18, dims = (128, 256))
    model = last(cache_benchmark_models(; chemistry_sites = sites))
    states = Dict(
        map(dims) do χ
            Random.seed!(17)
            χ => FiniteMPS(randn, Float64, sites, TensorKit.ℙ^2, TensorKit.ℙ^χ)
        end
    )
    serialize(path, (; H = model.H, states, terms = model.terms))
    println("Input: sites=", sites, "; terms=", model.terms, "; MPO bond=", maximum(dim(right_virtualspace(W)) for W in model.H))
    println("Initial MPS bonds: ", [(χ, comparison_bond(ψ)) for (χ, ψ) in states])
    return nothing
end
function comparison_inventory(path)
    open(path, "w") do io
        println(io, "uuid,name,version,tree_hash,source")
        for (uuid, dep) in sort!(collect(Pkg.dependencies()); by = x -> string(first(x)))
            println(io, join((uuid, dep.name, dep.version, dep.tree_hash, dep.source), ','))
        end
    end
    open(path * ".txt", "w") do io
        println(io, "Julia: ", VERSION, "; CPU: ", Sys.cpu_info()[1].model)
        println(io, "Numerical Julia threads: ", Threads.nthreads(), "; BLAS threads: ", BLAS.get_num_threads())
        println(io, "BLAS: ", BLAS.get_config())
        println(io, "MPSKit source: ", pathof(MPSKit))
    end
    return nothing
end
function comparison_save_rows(path, header, rows)
    open(path, "w") do io
        println(io, header)
        for row in rows
            println(io, join(row, ','))
        end
    end
    return nothing
end

# Warm the complete measurement helpers on an inexpensive input with identical
# argument and runner types, including their logger and TimerOutput paths.
function comparison_prime(data, prefix)
    model = last(cache_benchmark_models(; chemistry_sites = 6))
    Random.seed!(17)
    small = (; H = model.H, states = Dict(4 => FiniteMPS(randn, Float64, 6, TensorKit.ℙ^2, TensorKit.ℙ^4)), terms = model.terms)
    @assert typeof(small) === typeof(data)
    comparison_continuous(small, 4, prefix * "_continuous"; sweeps = 4)
    comparison_stages(small, 4, prefix * "_stages")
    comparison_public_sample(small, 4, prefix * "_public")
    return nothing
end

function comparison_warm(data, χ, prefix)
    alg = comparison_alg(χ; sweeps = 2)
    ψ = copy(data.states[χ])
    GC.gc()
    warm = @timed find_groundstate!(ψ, data.H, alg)
    _, _, info = warm.value
    @assert 1 <= info.numiter <= 2
    runner = comparison_runner(copy(data.states[χ]), data.H, alg)
    counter = ComparisonCounter(0, 0)
    with_logger(counter) do
        for _ in 1:info.numiter
            comparison_sweep!(runner)
        end
    end
    ψ2 = comparison_mps(runner)
    overlap = abs(dot(ψ, ψ2)) / (norm(ψ) * norm(ψ2))
    @assert isapprox(overlap, 1; atol = 1.0e-8)
    @assert isapprox(info.galerkin, comparison_error(runner); atol = 1.0e-7)
    @assert comparison_bond(ψ) == χ
    serialize(prefix * "_warm.jls", (; mps = ψ2, errors = comparison_errors(runner), updates = counter.updates, matvecs = counter.matvecs))
    println("Warm χ=", χ, ": public solve ", warm.time, " s for ", info.numiter, " sweeps; adapter overlap=", overlap, "; updates=", counter.updates, "; matvecs=", counter.matvecs)
    return nothing
end

function comparison_public_sample(data, χ, prefix; sweeps = 4)
    ψ = copy(data.states[χ])
    alg = comparison_alg(χ; sweeps)
    GC.gc()
    timed = @timed find_groundstate!(ψ, data.H, alg)
    _, _, info = timed.value
    @assert 1 <= info.numiter <= sweeps
    @assert comparison_bond(ψ) == χ
    comparison_save_rows(prefix * ".csv", "seconds,gc_seconds,allocated_bytes,sweeps,mps_bond,galerkin,compile_seconds,recompile_seconds", [(timed.time, timed.gctime, timed.bytes, info.numiter, comparison_bond(ψ), info.galerkin, timed.compile_time, timed.recompile_time)])
    serialize(prefix * ".jls", (; mps = ψ, galerkin = info.galerkin))
    println("Solve χ=", χ, ": ", timed.time, " s; GC=", timed.gctime, "; MiB=", timed.bytes / 2^20)
    return nothing
end

function comparison_continuous(data, χ, prefix; sweeps = 6)
    ψ = copy(data.states[χ])
    alg = comparison_alg(χ)
    GC.gc()
    setup = @timed comparison_runner(ψ, data.H, alg)
    runner = setup.value
    counter = ComparisonCounter(0, 0)
    rows = [(0, setup.time, setup.gctime, setup.bytes, 0, 0, setup.compile_time, setup.recompile_time)]
    for sweep in 1:sweeps
        counter.updates = counter.matvecs = 0
        timed = with_logger(counter) do
            @timed comparison_sweep!(runner)
        end
        push!(rows, (sweep, timed.time, timed.gctime, timed.bytes, counter.updates, counter.matvecs, timed.compile_time, timed.recompile_time))
    end
    @assert comparison_bond(ψ) == χ
    comparison_save_rows(prefix * ".csv", "sweep,seconds,gc_seconds,allocated_bytes,updates,matvecs,compile_seconds,recompile_seconds", rows)
    serialize(prefix * ".jls", (; mps = comparison_mps(runner), errors = comparison_errors(runner)))
    println("Continuous χ=", χ, ": ", sum(r[2] for r in rows), " s including setup")
    return nothing
end

function comparison_stages(data, χ, prefix)
    timer = MPSKit.TimerOutput()
    runner = comparison_runner(copy(data.states[χ]), data.H, comparison_alg(χ); timer)
    first_counter = ComparisonCounter(0, 0)
    with_logger(first_counter) do
        comparison_sweep!(runner)
    end
    GC.gc()
    tm = parentmodule(typeof(timer))
    tm.reset_timer!(timer)
    counter = ComparisonCounter(0, 0)
    timed = with_logger(counter) do
        @timed comparison_sweep!(runner)
    end
    stage_rows = []
    sweep = timer["sweep"]
    for stage in ("advance_env", "AC2_hamiltonian", "AC2_eigsolve", "gauge")
        if haskey(sweep, stage)
            t = sweep[stage]
            push!(stage_rows, (stage, tm.time(t) / 1.0e9, tm.allocated(t), tm.gctime(t) / 1.0e9))
        else
            push!(stage_rows, (stage, 0.0, 0, 0.0))
        end
    end
    push!(stage_rows, ("other", timed.time - sum(r[2] for r in stage_rows), timed.bytes - sum(r[3] for r in stage_rows), timed.gctime - sum(r[4] for r in stage_rows)))
    comparison_save_rows(prefix * ".csv", "stage,seconds,allocated_bytes,gc_seconds", stage_rows)
    comparison_save_rows(prefix * "_compilation.csv", "compile_seconds,recompile_seconds", [(timed.compile_time, timed.recompile_time)])
    serialize(prefix * ".jls", (; mps = comparison_mps(runner), errors = comparison_errors(runner), updates = counter.updates, matvecs = counter.matvecs, first_updates = first_counter.updates, first_matvecs = first_counter.matvecs))
    println("Stages χ=", χ, ": updates=", counter.updates, "; matvecs=", counter.matvecs)
    return nothing
end

"""Check final states, local errors, and counted solver work across the two snapshots."""
function validate_comparison_files(root; samples = 3, continuous_reps = 2)
    rows = []
    for χ in (128, 256), kind in ("solve", "continuous", "stages")
        reps = kind == "solve" ? (1:samples) : kind == "continuous" ? (1:continuous_reps) : (1:1)
        for rep in reps
            suffix = kind == "stages" ? "stages" : kind * "_" * string(rep)
            a = deserialize(joinpath(root, "main_$(χ)_$(suffix).jls"))
            b = deserialize(joinpath(root, "latest_$(χ)_$(suffix).jls"))
            overlap = abs(dot(a.mps, b.mps)) / (norm(a.mps) * norm(b.mps))
            @assert isapprox(overlap, 1; atol = 1.0e-8)
            difference = if hasproperty(a, :errors)
                maximum(abs.(a.errors - b.errors))
            else
                abs(a.galerkin - b.galerkin)
            end
            @assert difference < 1.0e-7
            if kind == "continuous" && rep > 1
                left = split.(readlines(joinpath(root, "main_$(χ)_$(suffix).csv")), ',')
                right = split.(readlines(joinpath(root, "latest_$(χ)_$(suffix).csv")), ',')
                @assert left[1] == right[1] && length(left) == length(right)
                for column in ("updates", "matvecs", "compile_seconds", "recompile_seconds")
                    index = findfirst(==(column), left[1])
                    @assert index !== nothing
                    @assert all(a[index] == b[index] for (a, b) in zip(left[2:end], right[2:end]))
                    if endswith(column, "seconds")
                        @assert all(parse(Float64, row[index]) == 0 for row in left[2:end])
                    end
                end
            end
            updates = matvecs = first_matvecs = 0
            if hasproperty(a, :updates)
                @assert a.updates == b.updates && a.matvecs == b.matvecs
                @assert a.first_updates == b.first_updates && a.first_matvecs == b.first_matvecs
                updates, matvecs, first_matvecs = a.updates, a.matvecs, a.first_matvecs
            end
            push!(rows, (χ, kind, rep, overlap, difference, updates, first_matvecs, matvecs))
        end
    end
    comparison_save_rows(joinpath(root, "agreement.csv"), "mps_bond,kind,sample,overlap,max_error_difference,updates,first_sweep_matvecs,second_sweep_matvecs", rows)
    println("Validated ", length(rows), " main/latest state pairs")
    return nothing
end
