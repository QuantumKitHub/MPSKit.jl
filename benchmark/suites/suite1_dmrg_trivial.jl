# Suite 1 (docs/IMPROVEMENT_PLAN.md §4.2, item 1): finite DMRG time-to-accuracy for the
# spin-1 Heisenberg chain, no symmetry.
#
# Protocol: for each χ in a schedule, build a random full-χ `FiniteMPS` with (plain,
# ungraded) virtual space ℂ^χ and run two-site DMRG (`DMRG2`, `trscheme = truncrank(χ)`)
# for a fixed number of sweeps, recording the energy and elapsed wall time after every
# sweep (see `BenchCommon.dmrg2_trajectory`). Every two-site update truncates back to at
# most χ states, so each schedule point genuinely probes that bond dimension.
#
# Two-site is the matched update scheme: ITensorMPS exposes only two-site `dmrg`, so a
# single-site comparison does not exist and the earlier single-site variant of this suite
# was dropped (maintainer decision, 2026-07-05). `BenchCommon.dmrg_trajectory` (single
# site) is kept for MPSKit-internal diagnostics only.
#
# Element type is Float64 on BOTH sides — the Hamiltonian is real-symmetric, real
# arithmetic is each library's best case, and matching it removes a 2-4x BLAS confound
# (maintainer decision, 2026-07-05).
module Suite1DMRGTrivial

using MPSKit
using MPSKitModels
using TensorKit
using Random
using Dates

include(joinpath(@__DIR__, "common.jl"))
using .BenchCommon

"""
    run(; N, chis, nsweeps, seed = 1234, J = 1.0, spin = 1, resultsdir = BenchCommon.results_dir())

Run the suite-1 protocol and write a timestamped JSON result file to `resultsdir`.
Returns the path to the written file.
"""
function run(;
        N::Int, chis::AbstractVector{<:Int}, nsweeps::Int,
        seed::Int = 1234, J::Real = 1.0, spin::Real = 1,
        resultsdir::AbstractString = BenchCommon.results_dir()
    )
    H = heisenberg_XXX(Float64, Trivial, FiniteChain(N); J = J, spin = spin)
    pspaces = physicalspace(H)

    # warmup: run the full pipeline once at a tiny size so JIT compilation does not
    # pollute the first timed trajectory (methodology guardrail §4.3: wall times must
    # reflect the algorithm, not the compiler)
    let Hw = heisenberg_XXX(Float64, Trivial, FiniteChain(6); J = J, spin = spin)
        dmrg2_trajectory(FiniteMPS(Float64, physicalspace(Hw), ℂ^4), Hw; nsweeps = 2, χ = 4)
    end

    trials = Vector{Dict{String, Any}}()
    for χ in chis
        Random.seed!(seed)
        ψ₀ = FiniteMPS(Float64, pspaces, ℂ^χ)
        elapsed = @elapsed result = dmrg2_trajectory(ψ₀, H; nsweeps = nsweeps, χ = χ)
        chi_actual = maximum(dim(left_virtualspace(result.ψ, n)) for n in 2:N)

        @info "suite 1: χ = $χ done" chi_actual final_energy = last(result.energies) total_time = elapsed final_galerkin_error = result.ϵ

        push!(
            trials, Dict{String, Any}(
                "chi_target" => χ,
                "chi_actual" => chi_actual,
                "energies" => result.energies,
                "walltimes" => result.walltimes,
                "gctimes" => result.gctimes,
                "allocd_bytes" => result.allocd,
                "final_galerkin_error" => result.ϵ,
            )
        )
    end

    best_energy = minimum(minimum(t["energies"]) for t in trials)

    data = collect_metadata()
    data["suite"] = "1-dmrg-trivial"
    data["description"] = "finite two-site DMRG time-to-accuracy, spin-1 Heisenberg chain, no symmetry"
    data["model"] = "heisenberg_XXX"
    data["symmetry"] = "Trivial"
    data["algorithm"] = "DMRG2(trscheme = truncrank(chi))"
    data["eltype"] = "Float64"
    data["N"] = N
    data["J"] = J
    data["spin"] = spin
    data["nsweeps"] = nsweeps
    data["seed"] = seed
    data["chi_schedule"] = collect(chis)
    data["best_energy"] = best_energy
    data["trials"] = trials

    timestamp = Dates.format(Dates.now(), "yyyymmdd-HHMMSS")
    path = joinpath(resultsdir, "suite1_dmrg_trivial_N$(N)_$(timestamp).json")
    write_results(path, data)
    return path
end

end # module
