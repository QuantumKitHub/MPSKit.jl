# Suite 2 (docs/IMPROVEMENT_PLAN.md §4.2, item 2): same recording protocol as suite 1,
# with U(1) (Sz-conservation) symmetry enforced on the spin-1 Heisenberg chain, and
# two-site DMRG (`DMRG2`) instead of single-site so the bond dimension is distributed
# across symmetry sectors automatically (see the sector-allocation note below).
#
# API verified against the installed MPSKitModels source
# (`~/.julia/packages/MPSKitModels/*/src/models/hamiltonians.jl` and
# `.../src/operators/spinoperators.jl`):
#   heisenberg_XXX(elt, symmetry::Type{<:Sector}, lattice; J, spin)
# and for `symmetry = U1Irrep`, `S_z` builds the physical space as
#   U1Space(v => 1 for v in U1Irrep.((-spin):spin))
# i.e. one U(1) charge sector per Sz eigenvalue. We don't hardcode this: we build `H`
# first and read its physical space back via `physicalspace(H)` (exported by MPSKit),
# so the site charges always agree with whatever MPSKitModels actually generated.
module Suite2DMRGU1

using MPSKit
using MPSKitModels
using TensorKit
using Random
using Dates

include(joinpath(@__DIR__, "common.jl"))
using .BenchCommon

# Sector allocation: unlike suite 1 (fixed full-χ space, single-site DMRG), this suite
# uses two-site DMRG (`DMRG2` with `trscheme = truncrank(χ)`), which redistributes the
# bond dimension across U(1) sectors automatically at every two-site update. This mirrors
# ITensor's behavior, whose two-site `dmrg` likewise chooses the per-sector block sizes
# itself; a hand-picked static split (tried first) was demonstrably suboptimal
# (−26.4027 vs ITensor's −26.8188 at χ = 8 on the N = 20 smoke check). The initial state
# only needs a small sector-diverse seed for the growth to start from; the boundary
# virtual spaces stay at the default trivial charge, targeting the total-Sz = 0 sector.
u1_seedspace() = U1Space(q => 1 for q in -1:1)

"""
    run(; N, chis, nsweeps, seed = 1234, J = 1.0, spin = 1, resultsdir = BenchCommon.results_dir())

Run the suite-2 protocol (U(1)-symmetric two-site DMRG) and write a timestamped JSON
result file to `resultsdir`. Returns the path to the written file.
"""
function run(;
        N::Int, chis::AbstractVector{<:Int}, nsweeps::Int,
        seed::Int = 1234, J::Real = 1.0, spin::Real = 1,
        resultsdir::AbstractString = BenchCommon.results_dir()
    )
    H = heisenberg_XXX(ComplexF64, U1Irrep, FiniteChain(N); J = J, spin = spin)
    pspaces = physicalspace(H)

    # warmup: run the full pipeline once at a tiny size so JIT compilation does not
    # pollute the first timed trajectory (methodology guardrail §4.3: wall times must
    # reflect the algorithm, not the compiler)
    let Hw = heisenberg_XXX(ComplexF64, U1Irrep, FiniteChain(6); J = J, spin = spin)
        dmrg2_trajectory(FiniteMPS(physicalspace(Hw), u1_seedspace()), Hw; nsweeps = 2, χ = 4)
    end

    trials = Vector{Dict{String, Any}}()
    for χ in chis
        Random.seed!(seed)
        ψ₀ = FiniteMPS(pspaces, u1_seedspace())
        elapsed = @elapsed result = dmrg2_trajectory(ψ₀, H; nsweeps = nsweeps, χ = χ)
        chi_actual = maximum(dim(left_virtualspace(result.ψ, n)) for n in 2:N)

        @info "suite 2: χ = $χ done" chi_actual final_energy = last(result.energies) total_time = elapsed final_galerkin_error = result.ϵ

        push!(
            trials, Dict{String, Any}(
                "chi_target" => χ,
                "chi_actual" => chi_actual,
                "energies" => result.energies,
                "walltimes" => result.walltimes,
                "final_galerkin_error" => result.ϵ,
            )
        )
    end

    best_energy = minimum(minimum(t["energies"]) for t in trials)

    data = collect_metadata()
    data["suite"] = "2-dmrg-u1"
    data["description"] = "finite two-site DMRG time-to-accuracy, spin-1 Heisenberg chain, U(1) symmetry"
    data["model"] = "heisenberg_XXX"
    data["symmetry"] = "U1Irrep"
    data["algorithm"] = "DMRG2(trscheme = truncrank(chi))"
    data["N"] = N
    data["J"] = J
    data["spin"] = spin
    data["nsweeps"] = nsweeps
    data["seed"] = seed
    data["chi_schedule"] = collect(chis)
    data["best_energy"] = best_energy
    data["trials"] = trials

    timestamp = Dates.format(Dates.now(), "yyyymmdd-HHMMSS")
    path = joinpath(resultsdir, "suite2_dmrg_u1_N$(N)_$(timestamp).json")
    write_results(path, data)
    return path
end

end # module
