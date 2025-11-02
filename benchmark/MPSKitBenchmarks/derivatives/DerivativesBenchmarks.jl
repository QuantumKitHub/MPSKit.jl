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
    "heisenberg_NN" => TOML.parsefile(joinpath(@__DIR__, "heisenberg_NN.toml"))
)

include("AC2_benchmarks.jl")

T = Float64

suite_init = addgroup!(SUITE, "AC2_preparation")
suite_apply = addgroup!(SUITE, "AC2_contraction")

for (model, params) in allparams
    g_prep = addgroup!(suite_init, model)
    g_contract = addgroup!(suite_apply, model)
    for (symmetry, specs) in params
        g_prep_sym = addgroup!(g_prep, symmetry)
        g_contract_sym = addgroup!(g_contract, symmetry)
        for spec_dict in specs
            spec = untomlify(AC2Spec, spec_dict)
            name = benchname(spec)
            g_prep_sym[name] = preparation_benchmark(spec; T)
            g_contract_sym[name] = contraction_benchmark(spec; T)
        end
    end
end

end
