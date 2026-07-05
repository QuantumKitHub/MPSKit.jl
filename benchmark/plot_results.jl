using JSON
using DataFrames
using CairoMakie
using Statistics

# Loading in the data
# -------------------
resultdir = joinpath(@__DIR__, "results")

# ============================================================================
# Legacy AC2-contraction / timestep regression plots (BenchmarkTools-based,
# `benchmark/MPSKitBenchmarks/`, see `benchmark/benchmarks.jl`). Unrelated to the
# time-to-accuracy suites below; only runs if its expected result files exist.
# ============================================================================
function plot_legacy_regression_results(resultdir)
    result_files = Dict(
        "main" => joinpath(resultdir, "results_MPSKit@main.json"),
        "dirty" => joinpath(resultdir, "results_MPSKit@dirty.json")
    )

    df = let df = DataFrame(
        :version => String[], :model => String[], :symmetry => String[],
        :D => Int[], :V => Int[], :memory => Tuple{Int, Int}[], :allocs => Tuple{Int, Int}[], :times => Tuple{Vector{Int}, Vector{Int}}[]
    )
    for (version, result_file) in pairs(result_files)
        result = JSON.parsefile(result_file)
        for (model, model_res) in result.data.derivatives.data.AC2_contraction.data
            for (symmetry, sym_res) in model_res.data
                for (DV, contract_bench) in sym_res.data
                    prep_bench = result.data.derivatives.data.AC2_preparation.data[model].data[symmetry].data[DV]
                    D, V = eval(Meta.parse(DV))::Tuple{Int, Int}
                    push!(
                        df,
                        (
                            version, model, symmetry, D, V,
                            (prep_bench.memory, contract_bench.memory),
                            (prep_bench.allocs, contract_bench.allocs),
                            (collect(Int, prep_bench.times), collect(Int, contract_bench.times)),
                        )
                    )
                end
            end
        end
    end
    df
end

df_prep = let df = DataFrame(
        :version => String[], :model => String[], :symmetry => String[],
        :D => Int[], :V => Int[], :memory => Int[], :allocs => Int[], :times => Vector{Int}[]
    )
    for (version, result_file) in pairs(result_files)
        result = JSON.parsefile(result_file)
        for (model, model_res) in result.data.derivatives.data.AC2_preparation.data
            for (symmetry, sym_res) in model_res.data
                for (DV, bench) in sym_res.data
                    D, V = eval(Meta.parse(DV))::Tuple{Int, Int}

                    push!(
                        df,
                        (version, model, symmetry, D, V, bench.memory, bench.allocs, collect(Int, bench.times))
                    )
                end
            end
        end
    end
    df
end

# Plotting the results
# --------------------
fontsize = 20
estimator = median

function plot_result(df, num_applications, choice = :times)
    f = Figure(; size = (1400, 1400))
    models = ["heisenberg_nn", "heisenberg_nnn", "heisenberg_cylinder", "heisenberg_coulomb"]
    symmetries = ["Trivial", "Irrep[U₁]", "Irrep[SU₂]"]


    df_model = groupby(df, [:model, :symmetry])
    for row in eachindex(models), col in eachindex(symmetries)
        df_data = get(df_model, (; model = models[row], symmetry = symmetries[col]), nothing)
        ylabel_ = choice === :times ? "Δt (μs)" : string(choice)
        ax = Axis(f[row, col], xscale = log10, xlabel = "D", ylabel = ylabel_, yscale = log10)
        @assert !isnothing(df_data)
        for (k, v) in pairs(groupby(df_data, :version))
            Ds = v[!, :D]
            if choice === :times
                times_prep = estimator.(first.(v[!, :times])) ./ 1.0e3
                times_contract = estimator.(last.(v[!, :times])) ./ 1.0e3
                data = times_prep .+ (num_applications .* times_contract)
            else
                allocs_prep = first.(v[!, choice]) ./ 1.0e3
                allocs_contract = last.(v[!, choice]) ./ 1.0e3
                data = allocs_prep .+ (num_applications .* allocs_contract)
            end
            I = sortperm(Ds)
            scatterlines!(ax, Ds[I], data[I]; label = "$(k.version)")
        end
        axislegend(ax, position = :lt)
    end

    Label(f[0, 0], "times"; fontsize)
    for (row, model) in enumerate(models)
        Label(f[row, 0], model; rotation = pi / 2, fontsize, tellheight = false, tellwidth = false)
    end
    for (col, symmetry) in enumerate(symmetries)
        Label(f[0, col], symmetry; fontsize, tellheight = false, tellwidth = false)
    end

    return f
end
    for choice in (:allocs, :memory, :times), n in [1, 3, 10]
        f = plot_result(df, n, choice)
        save(joinpath(resultdir, "bench_$(choice)_$n.png"), f)
        save(joinpath(resultdir, "bench_$(choice)_$n.svg"), f)
    end
    return nothing
end

if isfile(joinpath(resultdir, "results_MPSKit@main.json")) && isfile(joinpath(resultdir, "results_MPSKit@dirty.json"))
    plot_legacy_regression_results(resultdir)
else
    @info "Skipping legacy AC2-contraction regression plots: expected result files not found in $resultdir (these come from the separate BenchmarkTools workflow in benchmark/benchmarks.jl, not from benchmark/run.jl)."
end

# ============================================================================
# Suite 1-2 time-to-accuracy plots (docs/IMPROVEMENT_PLAN.md §4.2, items 1-2).
# Reads the JSON result files produced by `benchmark/run.jl` (one file per suite run,
# named `suite1_dmrg_trivial_*.json` / `suite2_dmrg_u1_*.json`) and, for each, plots
# |E - E_best| vs wall time on log-log axes, one line per χ. `E_best` is the lowest
# energy observed anywhere in that result file's batch of χ runs (methodology guardrail
# §4.3: never publish a "ground truth" energy the suite itself did not produce).
# ============================================================================

"""
    plot_suite_trajectories(result_file; title) -> Figure

Plot the energy-error-vs-walltime trajectories (one line per χ) stored in a suite 1/2
JSON result file.
"""
function plot_suite_trajectories(result_file::AbstractString; title::AbstractString = basename(result_file))
    result = JSON.parsefile(result_file)
    best_energy = result["best_energy"]
    trials = result["trials"]

    f = Figure(; size = (700, 500))
    ax = Axis(
        f[1, 1];
        xlabel = "wall time (s)", ylabel = "|E - E_best|",
        xscale = log10, yscale = log10, title = title
    )
    for trial in sort(trials; by = t -> t["chi_actual"])
        energies = Float64.(trial["energies"])
        walltimes = Float64.(trial["walltimes"])
        err = max.(abs.(energies .- best_energy), eps(Float64))
        # the last point typically has err == 0 (it *is* the best energy in its own run,
        # or ties another run's) and would vanish on a log scale; keep it visible at eps.
        scatterlines!(ax, walltimes, err; label = "χ = $(trial["chi_actual"])")
    end
    axislegend(ax, position = :rt)
    return f
end

"""
    plot_all_suite_results(resultsdir = joinpath(@__DIR__, "results"))

For each of suite 1 and suite 2, find the most recent result file in `resultsdir` and
save a `<name>.png` / `<name>.svg` trajectory plot next to it. No-ops for suites with no
result files yet (e.g. before `benchmark/run.jl` has been run).
"""
function plot_all_suite_results(resultsdir::AbstractString = joinpath(@__DIR__, "results"))
    suites = (
        ("suite1_dmrg_trivial", "Suite 1: finite DMRG, no symmetry"),
        ("suite2_dmrg_u1", "Suite 2: finite DMRG, U(1) symmetry"),
    )
    for (prefix, label) in suites
        files = filter(readdir(resultsdir)) do fname
            startswith(fname, prefix) && endswith(fname, ".json")
        end
        if isempty(files)
            @info "No result files found for $label (expected $(prefix)_*.json in $resultsdir); skipping."
            continue
        end
        file = joinpath(resultsdir, sort(files)[end]) # most recent by the embedded timestamp
        f = plot_suite_trajectories(file; title = label)
        outbase = joinpath(resultsdir, replace(basename(file), ".json" => ""))
        save(outbase * ".png", f)
        save(outbase * ".svg", f)
        println("Wrote plot for $label -> $(outbase).{png,svg}")
    end
    return nothing
end

plot_all_suite_results(resultdir)

# ============================================================================
# Cross-library comparison plots and extractions (investigation layer).
# MPSKit result files have no filename prefix; ITensorMPS files carry `itensor_`.
# Every function below no-ops with an @info when one side is missing, so this script can
# always be run on a partially populated results/ directory.
# ============================================================================

"""
    latest_result(resultsdir, prefix) -> Union{String, Nothing}

Path of the most recent result file starting with `prefix` (exact prefix match, so
`"suite1"` does not match `"itensor_suite1"`), or `nothing`.
"""
function latest_result(resultsdir::AbstractString, prefix::AbstractString)
    files = filter(readdir(resultsdir)) do fname
        startswith(fname, prefix) && endswith(fname, ".json")
    end
    return isempty(files) ? nothing : joinpath(resultsdir, sort(files)[end])
end

"""
    time_to_accuracy(trial, e_ref; rtol = 1e-8) -> Union{Float64, Nothing}

First recorded wall time (s) at which the trial's energy satisfies
`E - e_ref <= rtol * |e_ref|`, or `nothing` if the trajectory never gets there.
Signed difference, not absolute: a variational energy below `e_ref` (better than the
reference) also counts as converged.
"""
function time_to_accuracy(trial, e_ref::Float64; rtol::Float64 = 1.0e-8)
    energies = Float64.(trial["energies"])
    walltimes = Float64.(trial["walltimes"])
    idx = findfirst(e -> e - e_ref <= rtol * abs(e_ref), energies)
    return isnothing(idx) ? nothing : walltimes[idx]
end

"""
    compare_dmrg_suite(resultsdir, prefix, label; rtol = 1e-8)

For one DMRG suite (`prefix` ∈ {"suite1_dmrg_trivial", "suite2_dmrg_u1"}): overlay the
energy-error trajectories of the latest MPSKit and ITensorMPS runs (E_ref = best energy
across BOTH files — §4.3: the reference is produced by the suite itself), print the
sanity-gate energy differences and the time-to-accuracy table, and save the figure.
"""
function compare_dmrg_suite(resultsdir::AbstractString, prefix::AbstractString, label::AbstractString; rtol::Float64 = 1.0e-8)
    mpskit_file = latest_result(resultsdir, prefix)
    itensor_file = latest_result(resultsdir, "itensor_" * prefix)
    if isnothing(mpskit_file) || isnothing(itensor_file)
        @info "Skipping $label comparison: need both a MPSKit and an ITensorMPS result file in $resultsdir."
        return nothing
    end
    sides = ("MPSKit" => JSON.parsefile(mpskit_file), "ITensorMPS" => JSON.parsefile(itensor_file))
    e_ref = minimum(Float64(result["best_energy"]) for (_, result) in sides)

    f = Figure(; size = (800, 550))
    ax = Axis(
        f[1, 1];
        xlabel = "wall time (s)", ylabel = "E - E_ref",
        xscale = log10, yscale = log10, title = label
    )
    linestyles = Dict("MPSKit" => :solid, "ITensorMPS" => :dash)
    for (library, result) in sides
        for trial in sort(result["trials"]; by = t -> t["chi_target"])
            energies = Float64.(trial["energies"])
            walltimes = Float64.(trial["walltimes"])
            err = max.(energies .- e_ref, eps(Float64))
            scatterlines!(
                ax, walltimes, err;
                linestyle = linestyles[library],
                label = "$library χ = $(trial["chi_target"])"
            )
        end
    end
    axislegend(ax; position = :rt, nbanks = 2, labelsize = 10)
    outbase = joinpath(resultsdir, "compare_" * prefix)
    save(outbase * ".png", f)
    save(outbase * ".svg", f)
    println("Wrote comparison plot for $label -> $(outbase).{png,svg}")

    # sanity gate + time-to-accuracy table (stdout is the deliverable here)
    println("\n$label — sanity gate and time to E - E_ref <= $rtol * |E_ref| (E_ref = $e_ref):")
    println("  χ | final E (MPSKit) | final E (ITensorMPS) | ΔE | t_acc MPSKit (s) | t_acc ITensorMPS (s)")
    mpskit_trials = Dict(t["chi_target"] => t for t in sides[1][2]["trials"])
    itensor_trials = Dict(t["chi_target"] => t for t in sides[2][2]["trials"])
    for χ in sort(collect(keys(mpskit_trials)))
        haskey(itensor_trials, χ) || continue
        tm, ti = mpskit_trials[χ], itensor_trials[χ]
        em, ei = Float64(last(tm["energies"])), Float64(last(ti["energies"]))
        tam = time_to_accuracy(tm, e_ref; rtol)
        tai = time_to_accuracy(ti, e_ref; rtol)
        fmt(x) = isnothing(x) ? "not reached" : string(round(x; sigdigits = 4))
        println("  $χ | $em | $ei | $(abs(em - ei)) | $(fmt(tam)) | $(fmt(tai))")
    end
    return nothing
end

"""
    compare_tdvp_suite(resultsdir; prefix = "suite5_tdvp")

Suite 5: overlay the ⟨Sz⟩(t) trajectories of both libraries at matched χ (the TDVP
sanity gate — visible divergence means a protocol mismatch) and plot the throughput
(seconds of wall time per unit physical time, measure phase) vs χ.
"""
function compare_tdvp_suite(resultsdir::AbstractString; prefix::AbstractString = "suite5_tdvp")
    mpskit_file = latest_result(resultsdir, prefix)
    itensor_file = latest_result(resultsdir, "itensor_" * prefix)
    if isnothing(mpskit_file) || isnothing(itensor_file)
        @info "Skipping TDVP comparison: need both a MPSKit and an ITensorMPS suite-5 result file in $resultsdir."
        return nothing
    end
    sides = ("MPSKit" => JSON.parsefile(mpskit_file), "ITensorMPS" => JSON.parsefile(itensor_file))

    f = Figure(; size = (1100, 500))
    ax1 = Axis(f[1, 1]; xlabel = "t", ylabel = "⟨Sz⟩ mid-chain", title = "Suite 5 sanity gate: observable trajectories")
    ax2 = Axis(
        f[1, 2];
        xlabel = "χ", ylabel = "wall seconds per unit time",
        xscale = log2, yscale = log10, title = "Suite 5: TDVP throughput (measure phase)"
    )
    linestyles = Dict("MPSKit" => :solid, "ITensorMPS" => :dash)
    for (library, result) in sides
        chis = Float64[]
        thoughputs = Float64[]
        for trial in sort(result["trials"]; by = t -> t["chi_target"])
            traj = vcat(trial["trajectory_grow"], trial["trajectory_measure"])
            ts = [Float64(r["t"]) for r in traj]
            szs = [Float64(r["sz_mid"]) for r in traj]
            lines!(ax1, ts, szs; linestyle = linestyles[library], label = "$library χ = $(trial["chi_target"])")
            push!(chis, Float64(trial["chi_target"]))
            push!(thoughputs, Float64(trial["seconds_per_unit_time"]))
        end
        scatterlines!(ax2, chis, thoughputs; linestyle = linestyles[library], label = library)
    end
    axislegend(ax1; position = :rt, nbanks = 2, labelsize = 10)
    axislegend(ax2; position = :lt)
    outbase = joinpath(resultsdir, "compare_suite5_tdvp")
    save(outbase * ".png", f)
    save(outbase * ".svg", f)
    println("Wrote TDVP comparison plot -> $(outbase).{png,svg}")
    return nothing
end

"""
    compare_thread_scaling(resultsdir)

Suite 7: collect ALL suite-7 result files (one per (library, julia-threads, blas-threads)
grid point, latest per point), and plot the speedup of the total workload wall time
relative to that library's own (1, 1) baseline.
"""
function compare_thread_scaling(resultsdir::AbstractString)
    entries = Dict{Tuple{String, Int, Int}, Any}()   # (library, nj, nb) => result, latest wins
    for fname in sort(filter(f -> occursin("suite7_threads", f) && endswith(f, ".json"), readdir(resultsdir)))
        result = JSON.parsefile(joinpath(resultsdir, fname))
        library = get(result, "library", "MPSKit")
        entries[(library, Int(result["nthreads_julia"]), Int(result["nthreads_blas"]))] = result
    end
    if isempty(entries)
        @info "Skipping thread-scaling plot: no suite-7 result files in $resultsdir."
        return nothing
    end

    total_wall(result) = Float64(last(result["trials"][1]["walltimes"]))

    f = Figure(; size = (900, 500))
    ax = Axis(
        f[1, 1];
        xlabel = "(julia threads, blas threads)", ylabel = "speedup vs (1,1)",
        title = "Suite 7: thread scaling of the suite-1 workload"
    )
    grid = sort(unique([(nj, nb) for (_, nj, nb) in keys(entries)]))
    ax.xticks = (1:length(grid), ["($nj,$nb)" for (nj, nb) in grid])
    for library in sort(unique([lib for (lib, _, _) in keys(entries)]))
        haskey(entries, (library, 1, 1)) || begin
            @info "Thread-scaling: no (1,1) baseline for $library; skipping its speedup line."
            continue
        end
        baseline = total_wall(entries[(library, 1, 1)])
        xs = Int[]
        ys = Float64[]
        for (i, (nj, nb)) in enumerate(grid)
            haskey(entries, (library, nj, nb)) || continue
            push!(xs, i)
            push!(ys, baseline / total_wall(entries[(library, nj, nb)]))
        end
        scatterlines!(ax, xs, ys; label = library)
    end
    hlines!(ax, [1.0]; color = :gray, linestyle = :dot)
    axislegend(ax; position = :lt)
    outbase = joinpath(resultsdir, "compare_suite7_threads")
    save(outbase * ".png", f)
    save(outbase * ".svg", f)
    println("Wrote thread-scaling plot -> $(outbase).{png,svg}")
    return nothing
end

compare_dmrg_suite(resultdir, "suite1_dmrg_trivial", "Suite 1: finite two-site DMRG, no symmetry")
compare_dmrg_suite(resultdir, "suite2_dmrg_u1", "Suite 2: finite two-site DMRG, U(1) symmetry")
compare_tdvp_suite(resultdir)
compare_thread_scaling(resultdir)
