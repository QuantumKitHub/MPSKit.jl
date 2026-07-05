# Suite 1 (docs/IMPROVEMENT_PLAN.md §4.2, item 1), ITensorMPS.jl side: finite DMRG
# time-to-accuracy for the spin-1 Heisenberg chain, NO symmetry. Mirrors
# `benchmark/suites/suite1_dmrg_trivial.jl` on the MPSKit side.
#
# Model / idiom is the OFFICIAL ITensorMPS DMRG tutorial pattern
# (https://docs.itensor.org/ITensorMPS/stable/tutorials/DMRG.html), verbatim except for the
# J prefactor:
#     sites = siteinds("S=1", N)
#     os += "Sz",j,"Sz",j+1;  os += 1/2,"S+",j,"S-",j+1;  os += 1/2,"S-",j,"S+",j+1
#     H = MPO(os, sites)
#     psi0 = random_mps(sites; linkdims = χ)
# This builds H = J * Σ_j S_i·S_j, identical to MPSKit's
# `heisenberg_XXX(Float64, Trivial, FiniteChain(N); J, spin = 1)`.
#
# PARITY NOTE (element type): both sides use Float64. `random_mps` "by default has element
# type Float64" (documented idiom, correct for this real-symmetric Hamiltonian), and the
# MPSKit side passes Float64 explicitly for exact parity (maintainer decision, 2026-07-05 —
# previously MPSKit ran its ComplexF64 default, a 2-4x BLAS handicap).
#
# PARITY NOTE (update scheme): both sides use two-site DMRG — ITensor's default `dmrg`
# here, `DMRG2(trscheme = truncrank(χ))` on the MPSKit side. ITensorMPS exposes no
# single-site `dmrg`, so this is the only matched comparison.
module ITensorSuite1DMRGTrivial

using ITensors
using ITensorMPS
using Random
using Dates

include(joinpath(@__DIR__, "common.jl"))
using .ITensorBenchCommon

"""
    heisenberg_opsum(N; J) -> OpSum

Spin-1 Heisenberg `OpSum`, H = J·Σ_j S_j·S_{j+1}, built with the official Sz / S+ / S-
decomposition from the ITensorMPS DMRG tutorial.
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
    run(; N, chis, nsweeps, seed = 1234, J = 1.0, resultsdir = results_dir())

Run the suite-1 protocol with ITensorMPS and write a timestamped JSON result file (shared
schema, `"library" => "ITensorMPS"`) to `resultsdir`. Returns the path to the written file.
"""
function run(;
        N::Int, chis::AbstractVector{<:Int}, nsweeps::Int,
        seed::Int = 1234, J::Real = 1.0,
        resultsdir::AbstractString = results_dir(),
    )
    sites = siteinds("S=1", N)                       # https://docs.itensor.org/ITensorMPS/stable/tutorials/DMRG.html
    H = MPO(heisenberg_opsum(N; J = J), sites)

    # warmup: JIT-compile the whole pipeline once at a tiny size so compilation does not
    # pollute the first timed trajectory (methodology guardrail §4.3), matching MPSKit's
    # warmup at N = 6, χ = 4.
    let Nw = 6, sw = siteinds("S=1", Nw)
        Hw = MPO(heisenberg_opsum(Nw; J = J), sw)
        dmrg_trajectory(Hw, random_mps(sw; linkdims = 4), 4; nsweeps = 2)
    end

    trials = Vector{Dict{String, Any}}()
    for χ in chis
        Random.seed!(seed)                           # reseed per χ, as the MPSKit side does
        ψ₀ = random_mps(sites; linkdims = χ)         # random full-χ start (parity with FiniteMPS(pspaces, ℂ^χ))
        elapsed = @elapsed result = dmrg_trajectory(H, ψ₀, χ; nsweeps = nsweeps)
        chi_actual = maxlinkdim(result.psi)          # https://docs.itensor.org/ITensorMPS/stable/MPSandMPO.html

        @info "suite 1 (ITensorMPS): χ = $χ done" chi_actual final_energy = last(result.energies) total_time = elapsed

        push!(
            trials, Dict{String, Any}(
                "chi_target" => χ,
                "chi_actual" => chi_actual,
                "energies" => result.energies,
                "walltimes" => result.walltimes,
                "gctimes" => result.gctimes,
                "allocd_bytes" => result.allocd,
                # ITensor has no direct analogue of MPSKit's Galerkin (subspace) error;
                # kept as null so the schema matches and plotting code stays shared.
                "final_galerkin_error" => nothing,
            )
        )
    end

    best_energy = minimum(minimum(t["energies"]) for t in trials)

    data = collect_metadata()
    data["suite"] = "1-dmrg-trivial"
    data["description"] = "finite two-site DMRG time-to-accuracy, spin-1 Heisenberg chain, no symmetry (ITensorMPS)"
    data["model"] = "heisenberg_XXX"
    data["symmetry"] = "Trivial"
    data["algorithm"] = "dmrg (two-site), maxdim = mindim = chi, cutoff = 0, noise = 0"
    data["eltype"] = "Float64"
    data["N"] = N
    data["J"] = J
    data["spin"] = 1
    data["nsweeps"] = nsweeps
    data["seed"] = seed
    data["chi_schedule"] = collect(chis)
    data["best_energy"] = best_energy
    data["trials"] = trials

    timestamp = Dates.format(Dates.now(), "yyyymmdd-HHMMSS")
    path = joinpath(resultsdir, "itensor_suite1_dmrg_trivial_N$(N)_$(timestamp).json")
    write_results(path, data)
    return path
end

end # module
