using TensorKit
using TensorKit: type_repr
using MPSKit
using BlockTensorKit
using BlockTensorKit: nonzero_keys
using MPSKitModels
using TOML

include(joinpath(@__DIR__, "..", "MPSKitBenchmarks.jl"))
using .MPSKitBenchmarks
# MPSKitBenchmarks.load!("derivatives")
using .MPSKitBenchmarks.DerivativesBenchmarks: AC2Spec
using .MPSKitBenchmarks.BenchUtils: tomlify

# needed to make parsing correct
BlockTensorKit.SUMSPACE_SHOW_LIMIT[] = typemax(Int)

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


# NN
# --
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

output_file = joinpath(@__DIR__, "..", "derivatives", "heisenberg_nn.toml")
open(output_file, "w") do io
    TOML.print(
        io, Dict(
            type_repr(Trivial) => tomlify.(specs_triv),
            type_repr(U1Irrep) => tomlify.(specs_u1),
            type_repr(SU2Irrep) => tomlify.(specs_su2)
        )
    )
end
@info("Spaces written to $output_file")

# NNN
# ---
L = 100
lattice = FiniteChain(L)

symmetry = SU2Irrep
alg = DMRG2(;
    maxiter = 10, tol = 1.0e-12,
    alg_eigsolve = (; tol = 1.0e-5, dynamic_tols = false, maxiter = 3),
    trscheme = truncrank(D_max)
)

SS = S_exchange(T, symmetry; spin)
H = @mpoham sum(next_nearest_neighbours(lattice)) do (i, j)
    return SS{i, j}
end
H = heisenberg_XXX(T, symmetry, lattice; spin);
specs_su2 = generate_spaces(H, alg; D_min, D_steps)

specs_triv = filter!(convert.(AC2Spec{ComplexSpace}, specs_su2)) do spec
    dim(spec.mps_virtualspaces[1]) < 1000
end

specs_u1 = filter!(convert.(AC2Spec{U1Space}, specs_su2)) do spec
    dim(spec.mps_virtualspaces[1]) < 5000
end

output_file = joinpath(@__DIR__, "..", "derivatives", "heisenberg_nnn.toml")
open(output_file, "w") do io
    TOML.print(
        io, Dict(
            type_repr(Trivial) => tomlify.(specs_triv),
            type_repr(U1Irrep) => tomlify.(specs_u1),
            type_repr(SU2Irrep) => tomlify.(specs_su2)
        )
    )
end
@info("Spaces written to $output_file")

# FiniteCylinder
# --------------
rows = 6
cols = 12
lattice = FiniteCylinder(rows, rows * cols)

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

output_file = joinpath(@__DIR__, "..", "derivatives", "heisenberg_cylinder.toml")
open(output_file, "w") do io
    TOML.print(
        io, Dict(
            type_repr(Trivial) => tomlify.(specs_triv),
            type_repr(U1Irrep) => tomlify.(specs_u1),
            type_repr(SU2Irrep) => tomlify.(specs_su2)
        )
    )
end
@info("Spaces written to $output_file")

# Coulomb
# -------
L = 30
symmetry = SU2Irrep
SS = S_exchange(T, symmetry; spin)
lattice = fill(space(SS, 1), L)
terms = []
for i in 1:(L - 1), j in (i + 1):L
    push!(terms, (i, j) => SS / abs(i - j))
end
H = FiniteMPOHamiltonian(lattice, terms...);
H = changebonds(H, SvdCut(; trscheme = truncrank(500)));

D_max = 1_000
symmetry = SU2Irrep
alg = DMRG2(;
    maxiter = 10, tol = 1.0e-12,
    alg_eigsolve = (; tol = 1.0e-5, dynamic_tols = false, maxiter = 3),
    trscheme = truncrank(D_max)
)

specs_su2 = generate_spaces(H, alg; D_min, D_steps)

specs_triv = filter!(convert.(AC2Spec{ComplexSpace}, specs_su2)) do spec
    dim(spec.mps_virtualspaces[1]) < 500
end

specs_u1 = filter!(convert.(AC2Spec{U1Space}, specs_su2)) do spec
    dim(spec.mps_virtualspaces[1]) < 500
end

output_file = joinpath(@__DIR__, "..", "derivatives", "heisenberg_coulomb.toml")
open(output_file, "w") do io
    TOML.print(
        io, Dict(
            type_repr(Trivial) => tomlify.(specs_triv),
            type_repr(U1Irrep) => tomlify.(specs_u1),
            type_repr(SU2Irrep) => tomlify.(specs_su2)
        )
    )
end
@info("Spaces written to $output_file")
