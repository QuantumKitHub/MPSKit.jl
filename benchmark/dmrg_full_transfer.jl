# Compare the current full-MPO-transfer cache with ordinary environments, then
# separate chemistry preparation from transfers. Run through jld --project=test:
# include("benchmark/dmrg_full_transfer.jl"); benchmark_dmrg_full_transfer(); nothing
include("dmrg_operator_profile.jl")

function benchmark_dmrg_full_transfer(;
        χ = 64, samples = 3, chemistry_sites = 14,
        output_prefix = joinpath(@__DIR__, "results", "dmrg_full_transfer_after"),
    )
    models = cache_benchmark_models(; chemistry_sites)
    benchmark_dmrg_cache_scaling(;
        models, dims = (χ,), samples, sample_cpu = false, output_prefix,
    )
    benchmark_dmrg_operator_construction(;
        cases = ((chemistry_sites, χ),), samples,
        output_prefix = output_prefix * "_construction",
    )
    return output_prefix
end
