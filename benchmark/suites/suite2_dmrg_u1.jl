# Suite 2 (docs/IMPROVEMENT_PLAN.md §4.2, item 2): same protocol as suite 1, with U(1)
# (Sz-conservation) symmetry enforced on the spin-1 Heisenberg chain.
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

# REVIEW: choice of U(1) virtual-space shape. The ground state of the antiferromagnetic
# spin-1 Heisenberg chain lives in the total-Sz = 0 sector, so the boundary (leftmost /
# rightmost) virtual spaces are left at the default trivial charge (`unitspace`, via
# `FiniteMPS`'s `left`/`right` keywords), which is standard and not in question. What *is*
# a judgment call is the interior bond structure: how many U(1) sectors `-qmax:qmax` to
# allow and how to spread the requested bond dimension χ across them (here: uniformly).
# This has not been checked against a known-good Sz-sector weight distribution for this
# model, and a poor choice could make the U(1) run converge slower than an equally-sized
# unconstrained run purely from a bad sector split rather than from symmetry overhead.
# `qmax = 4` is a reasonable guess (spin-1 chains rarely need |Sz| > a handful of units in
# the bulk truncated space) but is otherwise arbitrary.
function u1_virtualspace(χ::Int; qmax::Int = 4)
    qs = (-qmax):qmax
    d = max(1, cld(χ, length(qs)))
    return U1Space(q => d for q in qs)
end

"""
    run(; N, chis, nsweeps, seed = 1234, J = 1.0, spin = 1, qmax = 4, resultsdir = BenchCommon.results_dir())

Run the suite-2 protocol (U(1)-symmetric DMRG) and write a timestamped JSON result file
to `resultsdir`. Returns the path to the written file.
"""
function run(;
        N::Int, chis::AbstractVector{<:Int}, nsweeps::Int,
        seed::Int = 1234, J::Real = 1.0, spin::Real = 1, qmax::Int = 4,
        resultsdir::AbstractString = BenchCommon.results_dir()
    )
    H = heisenberg_XXX(ComplexF64, U1Irrep, FiniteChain(N); J = J, spin = spin)
    pspaces = physicalspace(H)

    # warmup: run the full pipeline once at a tiny size so JIT compilation does not
    # pollute the first timed trajectory (methodology guardrail §4.3: wall times must
    # reflect the algorithm, not the compiler)
    let Hw = heisenberg_XXX(ComplexF64, U1Irrep, FiniteChain(6); J = J, spin = spin)
        dmrg_trajectory(FiniteMPS(physicalspace(Hw), u1_virtualspace(4; qmax = 2)), Hw; nsweeps = 2)
    end

    trials = Vector{Dict{String, Any}}()
    for χ in chis
        Random.seed!(seed)
        vspace = u1_virtualspace(χ; qmax = qmax)
        ψ₀ = FiniteMPS(pspaces, vspace)
        elapsed = @elapsed result = dmrg_trajectory(ψ₀, H; nsweeps = nsweeps)
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
    data["description"] = "finite DMRG time-to-accuracy, spin-1 Heisenberg chain, U(1) symmetry"
    data["model"] = "heisenberg_XXX"
    data["symmetry"] = "U1Irrep"
    data["N"] = N
    data["J"] = J
    data["spin"] = spin
    data["nsweeps"] = nsweeps
    data["seed"] = seed
    data["qmax"] = qmax
    data["chi_schedule"] = collect(chis)
    data["best_energy"] = best_energy
    data["trials"] = trials

    timestamp = Dates.format(Dates.now(), "yyyymmdd-HHMMSS")
    path = joinpath(resultsdir, "suite2_dmrg_u1_N$(N)_$(timestamp).json")
    write_results(path, data)
    return path
end

end # module
