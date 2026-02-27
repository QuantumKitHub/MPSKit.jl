using JSON
using DataFrames
using CairoMakie
using Statistics

# Loading in the data
# -------------------
resultdir = joinpath(@__DIR__, "results")

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
