using JSON
using DataFrames
using CairoMakie
using Statistics

# Loading in the data
# -------------------
resultdir = joinpath(@__DIR__, "results")
resultfile(i) = "results_MPSKit@bench$i.json"

df_contract = let df = DataFrame(
        :version => Int[], :model => String[], :symmetry => String[],
        :D => Int[], :V => Int[], :memory => Int[], :allocs => Int[], :times => Vector{Int}[]
    )

    for version in 0:3
        result = JSON.parsefile(joinpath(resultdir, resultfile(version)))
        for (model, model_res) in result.data.derivatives.data.AC2_contraction.data
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
df_prep = let df = DataFrame(
        :version => Int[], :model => String[], :symmetry => String[],
        :D => Int[], :V => Int[], :memory => Int[], :allocs => Int[], :times => Vector{Int}[]
    )

    for version in 0:3
        result = JSON.parsefile(joinpath(resultdir, resultfile(version)))
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

f_times = let f = Figure(; size = (1400, 1400))
    models = ["heisenberg_nn", "heisenberg_nnn", "heisenberg_cylinder", "heisenberg_coulomb"]
    symmetries = ["Trivial", "Irrep[U₁]", "Irrep[SU₂]"]


    df_model = groupby(df_contract, [:model, :symmetry])
    for row in eachindex(models), col in eachindex(symmetries)
        df_data = get(df_model, (; model = models[row], symmetry = symmetries[col]), nothing)
        ax = Axis(f[row, col], xscale = log10, xlabel = "D", ylabel = "Δt (μs)", yscale = log10)
        @assert !isnothing(df_data)
        for (k, v) in pairs(groupby(df_data, :version))
            Ds = v[!, :D]
            times = estimator.(v[!, :times]) ./ 1.0e3
            I = sortperm(Ds)
            scatterlines!(ax, Ds[I], times[I]; label = "v$(k.version)")
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

    f
end
save(joinpath(resultdir, "bench_times.png"), f_times)

f_times_relative = let f = Figure(; size = (1400, 1400))
    models = ["heisenberg_nn", "heisenberg_nnn", "heisenberg_cylinder", "heisenberg_coulomb"]
    symmetries = ["Trivial", "Irrep[U₁]", "Irrep[SU₂]"]


    df_model = groupby(df_contract, [:model, :symmetry])
    for row in eachindex(models), col in eachindex(symmetries)
        df_data = get(df_model, (; model = models[row], symmetry = symmetries[col]), nothing)
        ax = Axis(f[row, col], xscale = log10, xlabel = "D", ylabel = "Δt / Δt₀")
        hlines!([1], color = :red)
        @assert !isnothing(df_data)

        df_v = groupby(df_data, :version)

        v = get(df_v, (; version = 0), nothing)
        Ds = v[!, :D]
        times = estimator.(v[!, :times])
        I = sortperm(Ds)
        times₀ = times[I]

        for (k, v) in pairs(groupby(df_data, :version))
            k.version == 0 && continue
            Ds = v[!, :D]
            I = sortperm(Ds)
            times = estimator.(v[!, :times])[I]
            scatterlines!(ax, Ds[I], times ./ times₀; label = "v$(k.version)")
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

    f
end
save(joinpath(resultdir, "bench_times_relative.png"), f_times_relative)

f_allocs = let f = Figure(; size = (1400, 1400))
    models = ["heisenberg_nn", "heisenberg_nnn", "heisenberg_cylinder", "heisenberg_coulomb"]
    symmetries = ["Trivial", "Irrep[U₁]", "Irrep[SU₂]"]


    df_model = groupby(df_contract, [:model, :symmetry])
    for row in eachindex(models), col in eachindex(symmetries)
        df_data = get(df_model, (; model = models[row], symmetry = symmetries[col]), nothing)
        ax = Axis(f[row, col], xscale = log10, xlabel = "D", ylabel = "allocs", yscale = log10)
        @assert !isnothing(df_data)
        for (k, v) in pairs(groupby(df_data, :version))
            Ds = v[!, :D]
            allocs = estimator.(v[!, :allocs])
            I = sortperm(Ds)
            scatterlines!(ax, Ds[I], allocs[I]; label = "v$(k.version)")
        end
        axislegend(ax, position = :lt)
    end

    Label(f[0, 0], "allocs"; fontsize)
    for (row, model) in enumerate(models)
        Label(f[row, 0], model; rotation = pi / 2, fontsize, tellheight = false, tellwidth = false)
    end
    for (col, symmetry) in enumerate(symmetries)
        Label(f[0, col], symmetry; fontsize, tellheight = false, tellwidth = false)
    end

    f
end
save(joinpath(resultdir, "bench_allocs.png"), f_allocs)

f_memory = let f = Figure(; size = (1400, 1400))
    models = ["heisenberg_nn", "heisenberg_nnn", "heisenberg_cylinder", "heisenberg_coulomb"]
    symmetries = ["Trivial", "Irrep[U₁]", "Irrep[SU₂]"]


    df_model = groupby(df_contract, [:model, :symmetry])
    for row in eachindex(models), col in eachindex(symmetries)
        df_data = get(df_model, (; model = models[row], symmetry = symmetries[col]), nothing)
        ax = Axis(f[row, col], xscale = log10, xlabel = "D", ylabel = "memory (KiB)", yscale = log10)
        @assert !isnothing(df_data)
        for (k, v) in pairs(groupby(df_data, :version))
            Ds = v[!, :D]
            memory = estimator.(v[!, :memory]) ./ (2^10)
            I = sortperm(Ds)
            scatterlines!(ax, Ds[I], memory[I]; label = "v$(k.version)")
        end
        axislegend(ax, position = :lt)
    end

    Label(f[0, 0], "memory"; fontsize)
    for (row, model) in enumerate(models)
        Label(f[row, 0], model; rotation = pi / 2, fontsize, tellheight = false, tellwidth = false)
    end
    for (col, symmetry) in enumerate(symmetries)
        Label(f[0, col], symmetry; fontsize, tellheight = false, tellwidth = false)
    end

    f
end
save(joinpath(resultdir, "bench_memory.png"), f_allocs)

f_memory_relative = let f = Figure(; size = (1400, 1400))
    models = ["heisenberg_nn", "heisenberg_nnn", "heisenberg_cylinder", "heisenberg_coulomb"]
    symmetries = ["Trivial", "Irrep[U₁]", "Irrep[SU₂]"]


    df_model = groupby(df_contract, [:model, :symmetry])
    for row in eachindex(models), col in eachindex(symmetries)
        df_data = get(df_model, (; model = models[row], symmetry = symmetries[col]), nothing)
        ax = Axis(f[row, col], xscale = log10, xlabel = "D", ylabel = "memory / memory₀")
        hlines!([1], color = :red)
        @assert !isnothing(df_data)

        df_v = groupby(df_data, :version)

        v = get(df_v, (; version = 0), nothing)
        Ds = v[!, :D]
        times = estimator.(v[!, :memory])
        I = sortperm(Ds)
        times₀ = times[I]

        for (k, v) in pairs(groupby(df_data, :version))
            k.version == 0 && continue
            Ds = v[!, :D]
            I = sortperm(Ds)
            times = estimator.(v[!, :memory])[I]
            scatterlines!(ax, Ds[I], times ./ times₀; label = "v$(k.version)")
        end
        axislegend(ax, position = :lt)
    end

    Label(f[0, 0], "memory (relative)"; fontsize)
    for (row, model) in enumerate(models)
        Label(f[row, 0], model; rotation = pi / 2, fontsize, tellheight = false, tellwidth = false)
    end
    for (col, symmetry) in enumerate(symmetries)
        Label(f[0, col], symmetry; fontsize, tellheight = false, tellwidth = false)
    end

    f
end
save(joinpath(resultdir, "bench_memory_relative.png"), f_memory_relative)


# Including preparation times
# ---------------------------
for n_applications in [3, 10, 30]
    f_times_relative = let f = Figure(; size = (1400, 1400))
        models = ["heisenberg_nn", "heisenberg_nnn", "heisenberg_cylinder", "heisenberg_coulomb"]
        symmetries = ["Trivial", "Irrep[U₁]", "Irrep[SU₂]"]


        df_model = groupby(df_contract, [:model, :symmetry])
        dfp_model = groupby(df_prep, [:model, :symmetry])
        for row in eachindex(models), col in eachindex(symmetries)
            df_data = get(df_model, (; model = models[row], symmetry = symmetries[col]), nothing)
            dfp_data = get(dfp_model, (; model = models[row], symmetry = symmetries[col]), nothing)
            ax = Axis(f[row, col], xscale = log10, xlabel = "D", ylabel = "Δt / Δt₀")
            hlines!([1], color = :red)
            @assert !isnothing(df_data) && !isnothing(dfp_data)

            df_v = groupby(df_data, :version)
            dfp_v = groupby(dfp_data, :version)

            v = get(df_v, (; version = 0), nothing)
            Ds = v[!, :D]
            times = estimator.(v[!, :times])
            I = sortperm(Ds)
            times₀ = n_applications .* times[I]

            vp = get(dfp_v, (; version = 0), nothing)
            Ds = vp[!, :D]
            times = estimator.(vp[!, :times])
            I = sortperm(Ds)
            times₀ .+= times[I]

            df_data_v = groupby(dfp_data, :version)
            for (k, v) in pairs(groupby(df_data, :version))
                k.version == 0 && continue
                Ds = v[!, :D]
                I = sortperm(Ds)
                times = n_applications .* estimator.(v[!, :times])[I]

                vp = get(df_data_v, (; k.version), nothing)
                @assert !isnothing(vp)
                Ds = vp[!, :D]
                I = sortperm(Ds)
                times .+= estimator.(vp[!, :times][I])

                scatterlines!(ax, Ds[I], times ./ times₀; label = "v$(k.version)")
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

        f
    end
    save(joinpath(resultdir, "bench_prep_times_relative_n=$n_applications.png"), f_times_relative)
end
