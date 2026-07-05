# Diagnostic: profile WHERE the time goes in one MPSKit DMRG2 sweep of the suite-1
# workload. MPSKit-side only — this is an investigation tool, not a benchmark; nothing it
# prints is a comparable number.
#
# Usage:
#   julia --project=benchmark benchmark/profile_sweep.jl [--chi=256] [--N=100] [--maxdepth=12]
#
# Runs one warm-up sweep (JIT), then profiles a single additional sweep and prints
#   1. a flat hot-function list (depth-independent; the primary view),
#   2. a culled profile tree (deep by default: the interesting frames sit ~10 levels
#      below the script's include scaffolding), and
#   3. the @timed totals (wall seconds, GC seconds, GiB allocated) for that sweep.
# Interpreting: the hot leaves are typically BLAS (gemm), the truncated SVD, and
# KrylovKit's eigsolve; how much sits above them in MPSKit's own environment/derivative
# code is the actionable part.

using LinearAlgebra
using Profile
using Random

const HERE = @__DIR__

function parse_flag(args, name, default)
    idx = findfirst(a -> startswith(a, "--$name="), args)
    return isnothing(idx) ? default : parse(Int, split(args[idx], "=", limit = 2)[2])
end

chi = parse_flag(ARGS, "chi", 256)
N = parse_flag(ARGS, "N", 100)
maxdepth = parse_flag(ARGS, "maxdepth", 30)
BLAS.set_num_threads(parse_flag(ARGS, "blas-threads", 1))

include(joinpath(HERE, "suites", "common.jl"))
using .BenchCommon
using MPSKit
using MPSKitModels
using TensorKit

println("profile_sweep: N = $N, χ = $chi, julia threads = $(Threads.nthreads()), blas threads = $(BLAS.get_num_threads())")

H = heisenberg_XXX(Float64, Trivial, FiniteChain(N); J = 1.0, spin = 1)
Random.seed!(1234)
ψ₀ = FiniteMPS(Float64, physicalspace(H), ℂ^chi)

# one unprofiled sweep: JIT-compile everything and settle the state at the fixed point of
# the fixed-χ protocol, so the profiled sweep is a representative steady-state sweep
res = dmrg2_trajectory(ψ₀, H; nsweeps = 1, χ = chi)
ψ = res.ψ

Profile.clear()
stats = @timed Profile.@profile dmrg2_trajectory(ψ, H; nsweeps = 1, χ = chi)

nsamples = length(Profile.fetch())
println("\n=== hot functions, flat view (mincount = 1% of samples) ===")
Profile.print(; format = :flat, sortedby = :count, mincount = max(1, nsamples ÷ 100))

println("\n=== profile tree (mincount = 2% of samples, maxdepth = $maxdepth, noisefloor = 2) ===")
Profile.print(; format = :tree, mincount = max(1, nsamples ÷ 50), maxdepth = maxdepth, noisefloor = 2)

println("\n=== one-sweep totals (χ = $chi) ===")
println("wall seconds: ", round(stats.time; digits = 3))
println("GC seconds:   ", round(stats.gctime; digits = 3), " (", round(100 * stats.gctime / stats.time; digits = 1), "%)")
println("allocated:    ", round(stats.bytes / 2^30; digits = 3), " GiB")
