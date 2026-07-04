# Suite 2 (docs/IMPROVEMENT_PLAN.md §4.2, item 2), ITensorMPS.jl side: same protocol as
# suite 1, with U(1) (Sz-conservation) symmetry. Mirrors
# `benchmark/suites/suite2_dmrg_u1.jl` on the MPSKit side (which uses `U1Irrep`).
#
# Idiom is the OFFICIAL ITensorMPS QN-DMRG tutorial pattern
# (https://docs.itensor.org/ITensorMPS/stable/tutorials/QN_DMRG.html):
#     sites = siteinds("S=1", N; conserve_sz = true)
#     H = MPO(os, sites)                             # same OpSum as suite 1
#     state = [isodd(n) ? "Up" : "Dn" for n in 1:N]  # fixes total Sz = 0 for even N
#     psi0 = random_mps(sites, state; linkdims = χ)  # random full-χ MPS in the Sz = 0 sector
# `conserve_sz = true` is the precise U(1) analogue of MPSKit's `U1Irrep` Sz conservation
# (`conserve_qns = true`, used in the tutorial's title example, is equivalent for spin
# sites — for "S=1" the only conserved quantum number is total Sz). The tutorial itself
# uses the alternating "Up"/"Dn" state array for S=1 sites, so this is the documented idiom.
#
# PARITY / SECTOR NOTE: the AF spin-1 Heisenberg ground state lives in total Sz = 0, which
# the alternating state selects (`flux(psi0) == QN("Sz", 0)`; logged below). Unlike the
# MPSKit side — which must *hand-pick* how to spread χ across U(1) charge sectors (its
# `u1_virtualspace`/`qmax` REVIEW note) — ITensor's `random_mps(sites, state; linkdims)`
# distributes the bond dimension across sectors automatically for the given flux. So the
# per-sector split here is ITensor's own default, chosen by the library, not by us; this is
# the fair, idiomatic choice and cannot be called a strawman.
module ITensorSuite2DMRGU1

using ITensors
using ITensorMPS
using Random
using Dates

include(joinpath(@__DIR__, "common.jl"))
using .ITensorBenchCommon

"""
    heisenberg_opsum(N; J) -> OpSum

Spin-1 Heisenberg `OpSum`, H = J·Σ_j S_j·S_{j+1} (identical to suite 1; the OpSum is
symmetry-agnostic, symmetry lives in the site indices).
"""
function heisenberg_opsum(N::Int; J::Real = 1.0)
    os = OpSum()
    for j in 1:(N - 1)
        os += J, "Sz", j, "Sz", j + 1
        os += J / 2, "S+", j, "S-", j + 1
        os += J / 2, "S-", j, "S+", j + 1
    end
    return os
end

"""
    neel_state(N) -> Vector{String}

Alternating "Up"/"Dn" product state (official QN-DMRG tutorial idiom) fixing total Sz = 0
for even `N`.
"""
neel_state(N::Int) = [isodd(n) ? "Up" : "Dn" for n in 1:N]

"""
    run(; N, chis, nsweeps, seed = 1234, J = 1.0, resultsdir = results_dir())

Run the suite-2 (U(1)-symmetric) protocol with ITensorMPS and write a timestamped JSON
result file (shared schema, `"library" => "ITensorMPS"`) to `resultsdir`. Returns the path.
"""
function run(;
        N::Int, chis::AbstractVector{<:Int}, nsweeps::Int,
        seed::Int = 1234, J::Real = 1.0,
        resultsdir::AbstractString = results_dir(),
    )
    isodd(N) && error("suite 2 assumes even N so the Néel state has total Sz = 0 (got N = $N)")

    sites = siteinds("S=1", N; conserve_sz = true)   # https://docs.itensor.org/ITensorMPS/stable/tutorials/QN_DMRG.html
    H = MPO(heisenberg_opsum(N; J = J), sites)

    # warmup (see suite 1); tiny even chain in the Sz = 0 sector.
    let Nw = 6, sw = siteinds("S=1", Nw; conserve_sz = true)
        Hw = MPO(heisenberg_opsum(Nw; J = J), sw)
        dmrg_trajectory(Hw, random_mps(sw, neel_state(Nw); linkdims = 4), 4; nsweeps = 2)
    end

    trials = Vector{Dict{String, Any}}()
    for χ in chis
        Random.seed!(seed)
        # random full-χ MPS with total Sz fixed by `state` (parity with the MPSKit
        # U(1) FiniteMPS in the Sz = 0 sector). random_mps distributes χ across the
        # allowed U(1) sectors itself. https://docs.itensor.org/ITensorMPS/stable/MPSandMPO.html
        ψ₀ = random_mps(sites, neel_state(N); linkdims = χ)
        @assert flux(ψ₀) == QN("Sz", 0) "initial MPS not in the total-Sz = 0 sector: flux = $(flux(ψ₀))"
        elapsed = @elapsed result = dmrg_trajectory(H, ψ₀, χ; nsweeps = nsweeps)
        chi_actual = maxlinkdim(result.psi)

        @info "suite 2 (ITensorMPS): χ = $χ done" chi_actual final_energy = last(result.energies) total_time = elapsed

        push!(
            trials, Dict{String, Any}(
                "chi_target" => χ,
                "chi_actual" => chi_actual,
                "energies" => result.energies,
                "walltimes" => result.walltimes,
                "final_galerkin_error" => nothing,
            )
        )
    end

    best_energy = minimum(minimum(t["energies"]) for t in trials)

    data = collect_metadata()
    data["suite"] = "2-dmrg-u1"
    data["description"] = "finite DMRG time-to-accuracy, spin-1 Heisenberg chain, U(1) symmetry (ITensorMPS)"
    data["model"] = "heisenberg_XXX"
    data["symmetry"] = "U1 (conserve_sz)"
    data["N"] = N
    data["J"] = J
    data["spin"] = 1
    data["nsweeps"] = nsweeps
    data["seed"] = seed
    data["chi_schedule"] = collect(chis)
    data["best_energy"] = best_energy
    data["trials"] = trials

    timestamp = Dates.format(Dates.now(), "yyyymmdd-HHMMSS")
    path = joinpath(resultsdir, "itensor_suite2_dmrg_u1_N$(N)_$(timestamp).json")
    write_results(path, data)
    return path
end

end # module
