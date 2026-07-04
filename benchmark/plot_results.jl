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
