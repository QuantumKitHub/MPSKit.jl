# Load benchmark code
include("MPSKitBenchmarks/MPSKitBenchmarks.jl")
const SUITE = MPSKitBenchmarks.SUITE

# Populate benchmarks
# Detect if user supplied extra arguments to load only specific modules
# e.g. julia benchmarks.jl --modules=linalg,tensornetworks
modules_pattern = r"(?:--modules=)(\w+)"
arg_id = findfirst(contains(modules_pattern), ARGS)
if isnothing(arg_id)
    MPSKitBenchmarks.loadall!()
else
    modules = split(only(match(modules_pattern, ARGS[arg_id]).captures[1]), ",")
    for m in modules
        MPSKitBenchmarks.load!(m)
    end
end
