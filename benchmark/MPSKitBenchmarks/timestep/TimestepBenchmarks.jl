module TimestepBenchmarks

using BenchmarkTools
using MPSKit
using MPSKitModels

const SUITE = BenchmarkGroup()
const dt = 0.05

const HAMILTONIANS = (
    "tfim" => transverse_field_ising(),
    "heis_xxx" => heisenberg_XXX(),
)

# Steady-state benchmark: time `make_time_mpo` after compilation has settled.
# Keep the matrix small — we want a stable baseline, not an exhaustive sweep.
let g_top = addgroup!(SUITE, "make_time_mpo")
    for (Hname, H) in HAMILTONIANS
        g = addgroup!(g_top, Hname)
        for N in (1, 2, 3)
            alg = TaylorCluster(; N)
            g["N=$N"] = @benchmarkable make_time_mpo($H, $dt, $alg)
        end
        g["WI"] = @benchmarkable make_time_mpo($H, $dt, $(MPSKit.WI))
    end
end

end # module
