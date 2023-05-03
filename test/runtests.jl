using MPSKit, TensorKit, Test, OptimKit, MPSKitModels, TestExtras, Plots
using MPSKit: _transpose_tail, _transpose_front, @plansor

using TensorOperations
TensorOperations.disable_cache()

include("planarspace.jl")
include("states.jl")
include("operators.jl")
include("algorithms.jl")


@timedtestset "plot 'tests'" begin
    ts = InfiniteMPS([𝔹^2], [𝔹^5])
    @test transferplot(ts) isa Plots.Plot
    @test entanglementplot(ts) isa Plots.Plot
end

println("------------------------------------")
println("|     Old bugs                     |")
println("------------------------------------")

@timedtestset "IDMRG2 space mismatch" begin
    ## Hamiltonian ##

    function heisenberg_interaction(spin)
        physical = Rep[SU₂](spin => 1)
        adjoint = Rep[SU₂](1 => 1)

        Sl = TensorMap(ones, ComplexF64, physical, adjoint ⊗ physical)
        Sr = TensorMap(ones, ComplexF64, adjoint ⊗ physical, physical)

        return @tensor H[-1 -2; -3 -4] := Sl[-1; 1 -3] * Sr[1 -2; -4] * (spin * (spin + 1))
    end

    H_heis = heisenberg_interaction(1)
    @tensor H_aklt[-1 -2; -3 -4] := H_heis[-1 -2; -3 -4] +
                                    1 / 3 * H_heis[-1 -2; 1 2] * H_heis[1 2; -3 -4]
    ##
    H = MPOHamiltonian(H_aklt)
    N = 6
    H = repeat(H, N)
    ψ₀ = InfiniteMPS(fill(SU2Space(1 => 1), N), fill(SU2Space(1 // 2 => 2, 3 // 2 => 1), N))
    alg = IDMRG2(; verbose=false, tol_galerkin=1e-5, trscheme=truncdim(32))

    ψ, envs, δ = find_groundstate(ψ₀, H, alg) # used to error
    @test ψ isa InfiniteMPS
end

@testset "NaN entanglement entropy" begin
    ts = InfiniteMPS([ℂ^2], [ℂ^5])
    ts = changebonds(ts, RandExpand(; trscheme=truncdim(2)))
    @test !isnan(sum(entropy(ts)))
    @test !isnan(sum(entropy(ts, 2)))
end
