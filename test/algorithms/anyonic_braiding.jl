println("
------------------------------------
|   Anyonic braiding consistency    |
------------------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using BlockTensorKit
using MPSKit: AC_hamiltonian, AC2_hamiltonian, AC2
using TensorKit
using KrylovKit: eigsolve
using LinearAlgebra
using Random

# when reinterpreting an MPO as an MPS ρ, there are contractions which
# braid the ancilla leg past the operator's virtual leg
# this turned out to be incorrect in several places
# the tests below check that the braiding is consistent with the dense trace for finite systems,
# and with the MPO-level overlap transfer for infinite systems

@testset "density-matrix sandwich (Fibonacci)" begin
    V = Vect[FibonacciAnyon](:I => 1, :τ => 1)
    ρ = randn(ComplexF64, V ⊗ V, V ⊗ V)
    O = randn(ComplexF64, V ⊗ V, V ⊗ V)
    ρ_mps = convert(FiniteMPS, FiniteMPO(ρ))

    ref = tr(ρ' * O * ρ)
    @test dot(FiniteMPO(ρ), FiniteMPO(O) * FiniteMPO(ρ)) ≈ ref
    @test dot(ρ_mps, FiniteMPO(O), ρ_mps) ≈ ref
end

@testset "expectation values with density matrices (Fibonacci)" begin
    V = Vect[FibonacciAnyon](:I => 1, :τ => 1)
    L = 4
    h = randn(ComplexF64, V ⊗ V, V ⊗ V)
    h = h + h'
    H = FiniteMPOHamiltonian(fill(V, L), [(i, i + 1) => h for i in 1:(L - 1)])
    ρ = make_time_mpo(H, 0.1, TaylorCluster(; N = 2); imaginary_evolution = true)
    ρd, Hd = convert(TensorMap, ρ), convert(TensorMap, H)
    ref = tr(ρd' * Hd * ρd) / tr(ρd' * ρd)

    # expectation_value here attempts to braid ancilla leg with mpo virtual leg,
    # which is a sumspace/plain pair and thus fails
    @test_broken expectation_value(ρ, H) ≈ ref

    # free workaround: flatten jordan tensors into plain mpo tensors
    # don't convert to tensormap, this is an exponential cost in L
    H_dense = FiniteMPO([W isa AbstractBlockTensorMap ? TensorMap(W) : W for W in parent(H)])
    @test expectation_value(ρ, H_dense) ≈ ref

    # these require the densification in _mpo_to_mps (mpo.jl#90)
    @test expectation_value(ρ, FiniteMPO(Hd)) ≈ ref
    @test expectation_value(ρ, (1, 2) => h) + expectation_value(ρ, (2, 3) => h) +
        expectation_value(ρ, (3, 4) => h) ≈ ref
end

@testset "braiding consistency in derivative operators (Fibonacci)" begin
    V = Vect[FibonacciAnyon](:I => 1, :τ => 1)
    L = 3
    T = ComplexF64
    ρ, O = FiniteMPO(randn(T, V^L, V^L)), FiniteMPO(randn(T, V^L, V^L))
    ρ_mps = convert(FiniteMPS, ρ)
    exact = convert(FiniteMPS, O * ρ)
    ψ0 = FiniteMPS(randn, T, fill(V ⊗ V', L), Vect[FibonacciAnyon](:I => 24, :τ => 24))
    fid(ψ) = abs(dot(ψ, exact)) / (norm(ψ) * norm(exact))

    # unprepared AC reached by one-site approximate
    ψ1, = approximate(ψ0, (O, ρ_mps), DMRG(; tol = 1.0e-10, maxiter = 50)) # simply doesn't converge without the fix
    @test fid(ψ1) ≈ 1 atol = 1.0e-6

    # unprepared AC2 reached by two-site approximate
    ψ2, = approximate(ψ0, (O, ρ_mps), DMRG2(; tol = 1.0e-10, maxiter = 50, trunc = truncrank(64))) # simply doesn't converge without the fix
    @test fid(ψ2) ≈ 1 atol = 1.0e-6

    # prepared AC, reached by any eigensolver
    # τ' fix on mpo_derivatives.jl#252 allows the eigensolver to converge at all,
    # even though the eigenproblem is degenerate over the ancilla leg
    A = randn(T, V^L, V^L)
    Oh = FiniteMPO(A + A')
    ψ3, _, ϵ3 = find_groundstate(ρ_mps, Oh, DMRG(; tol = 1.0e-10, maxiter = 50))
    @test ϵ3 ≤ 1.0e-10

    # prepared AC2, reached by two-site TDVP
    Ohd = convert(TensorMap, Oh)
    ρd = randn(T, V^L, V^L)
    ρ_mps2 = convert(FiniteMPS, FiniteMPO(ρd))
    dt = 0.02
    ρ_exact_d = exp(-im * dt * Ohd) * ρd
    target = convert(FiniteMPS, FiniteMPO(ρ_exact_d))
    ψt, = timestep(ρ_mps2, Oh, 0.0, dt, TDVP2(; trunc = truncrank(64)))
    tdvp2_fid = abs(dot(ψt, target)) / (norm(ψt) * norm(target))
    @test tdvp2_fid ≈ 1 atol = 1.0e-6 # fidelity was previously high, but not ≈ 1
end

const braid_spaces = (
    ℂ^2,
    Vect[Z2Irrep](0 => 1, 1 => 1),
    Vect[FermionParity](0 => 1, 1 => 1),
    Vect[FibonacciAnyon](:I => 1, :τ => 1),
)

# build L-site operator that's the identity everywhere except at site i
# where it's op with physical space V
_embed(L, op, i) = foldl(⊗, (ntuple(_ -> id(space(op, 1)), i - 1)..., op, ntuple(_ -> id(space(op, 1)), L - i - numout(op) + 1)...))

@testset "density-matrix braiding, finite: $(sectortype(V))" for V in braid_spaces
    L = 3
    Random.seed!(4321)
    ρd = randn(ComplexF64, V^L, V^L)
    Od = randn(ComplexF64, V^L, V^L)
    ρ, O = FiniteMPO(ρd), FiniteMPO(Od)
    ψ = convert(FiniteMPS, ρ)
    ref = tr(ρd' * Od * ρd)
    nrm = tr(ρd' * ρd)

    # both routes to <ρ|O|ρ> must reproduce the dense trace
    @test dot(ρ, O * ρ) ≈ ref
    @test dot(ψ, O, ψ) ≈ ref

    # each local derivative operator must reproduce the same scalar
    envs = environments(ψ, O, ψ)
    for i in 1:L, prepare in (false, true)
        h = AC_hamiltonian(i, ψ, O, ψ, envs; prepare)
        @test dot(ψ.AC[i], h * ψ.AC[i]) ≈ ref
    end
    for i in 1:(L - 1)
        x = AC2(ψ, i)
        h = AC2_hamiltonian(i, ψ, O, ψ, envs; prepare = false)
        @test dot(x, h * x) ≈ ref
    end

    # The prepared two-site derivative is a separate contraction whose crossings
    # are encoded in `braid` levels rather than τ, along with fused legs
    for i in 1:(L - 1)
        x = AC2(ψ, i)
        h = AC2_hamiltonian(i, ψ, O, ψ, envs; prepare = true)
        @test dot(x, h * x) ≈ ref
    end

    # local observables measured on a density matrix
    o1 = randn(ComplexF64, V, V)
    o1 = o1 + o1'
    @test expectation_value(ρ, 2 => o1) ≈ tr(ρd' * _embed(L, o1, 2) * ρd) / nrm
    o2 = randn(ComplexF64, V ⊗ V, V ⊗ V)
    o2 = o2 + o2'
    for i in 1:(L - 1)
        @test expectation_value(ρ, (i, i + 1) => o2) ≈ tr(ρd' * _embed(L, o2, i) * ρd) / nrm
    end
end

# overlap per unit cell of two infinite MPOs as leading eigenvalue of
# TM that contracts both physical legs directly
# contains no braiding tensor -> independent reference for the MPS-MPO-MPS sandwich
function mpo_overlap(mpo1, mpo2)
    N = length(mpo1)
    function step(v)
        for i in 1:N
            @plansor w[-1; -2] := v[1; 2] * conj(mpo1[i][1 3; 4 -1]) * mpo2[i][2 3; 4 -2]
            v = w
        end
        return v
    end
    v0 = randn(ComplexF64, space(mpo1[N], 4)' ← space(mpo2[N], 4)')
    vals, = eigsolve(step, v0, 1, :LM; krylovdim = 30)
    return vals[1]
end

@testset "density-matrix braiding, infinite: $(sectortype(V))" for V in braid_spaces
    Vv = V isa ComplexSpace ? ℂ^3 : V ⊕ V
    for N in (1, 2)
        Random.seed!(2024)
        ρ = InfiniteMPO([randn(ComplexF64, Vv ⊗ V ← V ⊗ Vv) for _ in 1:N])
        O = InfiniteMPO([randn(ComplexF64, V ⊗ V ← V ⊗ V) for _ in 1:N])
        ref = mpo_overlap(ρ, O * ρ) / mpo_overlap(ρ, ρ)
        ψ = convert(InfiniteMPS, ρ)
        @test dot(ψ, O, ψ) ≈ ref
    end
end
