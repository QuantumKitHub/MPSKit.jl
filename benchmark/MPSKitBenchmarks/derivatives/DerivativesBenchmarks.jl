module DerivativesBenchmarks

export AC2Spec

using BenchmarkTools
using TOML
using TensorKit
using BlockTensorKit
using MPSKit
using ..BenchUtils
import ..BenchUtils: tomlify, untomlify

const SUITE = BenchmarkGroup()

const allparams = Dict(
    "heisenberg_NN" => TOML.parsefile(joinpath(@__DIR__, "heisenberg_NN_specs.toml"))
)

include("AC2_benchmarks.jl")

T = Float64

suite_init = addgroup!(SUITE, "AC2_preparation")
suite_apply = addgroup!(SUITE, "AC2_contraction")

for (model, params) in allparams
    g_prep = addgroup!(suite_init, model)
    g_contract = addgroup!(suite_apply, model)
    specs = untomlify.(AC2Spec, params["specs"])

    for spec in specs
        name = benchname(spec)
        g_prep[name] = preparation_benchmark(spec; T)
        g_contract[name] = contraction_benchmark(spec; T)
    end
end

end
