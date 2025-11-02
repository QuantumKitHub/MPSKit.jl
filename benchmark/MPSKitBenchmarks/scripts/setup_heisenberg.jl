using TensorKit
using MPSKit
using MPSKit.BlockTensorKit: nonzero_keys
using MPSKit.BlockTensorKit
using MPSKitModels
using TOML

include(joinpath(@__DIR__, "..", "MPSKitBenchmarks.jl"))
using .MPSKitBenchmarks
# MPSKitBenchmarks.load!("derivatives")
using .MPSKitBenchmarks.DerivativesBenchmarks: AC2Spec
using .MPSKitBenchmarks.BenchUtils: tomlify

# Utility functions
# ----------------
# setup "product state" -> ⨂^N |↑↓ - ↓↑⟩
function initial_state(H)
    @assert iseven(length(H))
    @assert allequal(physicalspace(H))

    pspace = physicalspace(H, 1)
    A = rand(oneunit(pspace) ⊗ pspace^2 ← oneunit(pspace))
    As = MPSKit.decompose_localmps(A)
    return FiniteMPS(repeat(As, length(H) ÷ 2))
end

function generate_spaces(H, alg; D_min = 2, D_steps = 5)
    # compute maximal spaces
    psi_init = initial_state(H)
    psi, = find_groundstate(psi_init, H, alg)

    D_max = maximum(dim, left_virtualspace(psi))
    Ds = round.(Int, logrange(D_min, D_max, D_steps))

    return map(Ds) do D
        mps = changebonds(psi, SvdCut(; trscheme = truncrank(D)))
        return AC2Spec(mps, H)
    end
end

# Parameters
# ----------
T = Float64
spin = 1

D_max = 10_000

D_steps = 10
D_min = 3


# FiniteChain
# -----------
L = 100
lattice = FiniteChain(L)

symmetry = SU2Irrep
alg = DMRG2(;
    maxiter = 10, tol = 1.0e-12,
    alg_eigsolve = (; tol = 1.0e-5, dynamic_tols = false, maxiter = 3),
    trscheme = truncrank(D_max)
)

H = heisenberg_XXX(T, symmetry, lattice; spin);
specs_su2 = generate_spaces(H, alg; D_min, D_steps)

specs_triv = filter!(convert.(AC2Spec{ComplexSpace}, specs_su2)) do spec
    dim(spec.mps_virtualspaces[1]) < 1000
end

specs_u1 = filter!(convert.(AC2Spec{U1Space}, specs_su2)) do spec
    dim(spec.mps_virtualspaces[1]) < 5000
end

specs = vcat(specs_triv, specs_u1, specs_su2)

output_file = joinpath(@__DIR__, "heisenberg_NN_specs.toml")
open(output_file, "w") do io
    TOML.print(io, Dict("specs" => tomlify.(specs)))
end
@info("Spaces written to $output_file")
