using TestEnv;
TestEnv.activate()
include("setup.jl")

using .TestSetup

using Test, TestExtras
using MPSKit
using MPSKit: fuse_mul_mpo
using TensorKit
using TensorKit: ℙ
using BlockTensorKit

verbosity_full = 5
verbosity_conv = 1
tol = 1e-8
g = 4.0
D = 6

H_ref = transverse_field_ising(; g)
H2 = InfiniteMPOHamiltonian(map(SparseBlockTensorMap, parent(H_ref)))

ψ = InfiniteMPS(ℂ^2, ℂ^D)
v₀ = variance(ψ, H_ref)

for unit_cell_size in [1, 3]
    ψ = unit_cell_size == 1 ? InfiniteMPS(ℂ^2, ℂ^D) : repeat(ψ, unit_cell_size)
    H = repeat(H_ref, unit_cell_size)
    H3 = repeat(H2, unit_cell_size)

    # test logging
    ψ, envs, δ = find_groundstate(ψ, H,
                                  VUMPS(; tol, verbosity=verbosity_full, maxiter=2))

    ψ, envs, δ = find_groundstate(ψ, H, VUMPS(; tol, verbosity=verbosity_conv))
    v = variance(ψ, H, envs)
    v2 = expectation_value(ψ, H3)
    expectation_value(ψ, H, envs)

    # test using low variance
    @test sum(δ) ≈ 0 atol = 1e-3
    @test v < v₀
    @test v < 1e-2
end

for unit_cell_size in [1, 3]
    ψ = unit_cell_size == 1 ? InfiniteMPS(ℂ^2, ℂ^D) : repeat(ψ, unit_cell_size)
    H = repeat(H_ref, unit_cell_size)

    # test logging
    ψ, envs, δ = find_groundstate(ψ, H,
                                  IDMRG(; tol, verbosity=verbosity_full, maxiter=2))

    ψ, envs, δ = find_groundstate(ψ, H, IDMRG(; tol, verbosity=verbosity_conv))
    v = variance(ψ, H, envs)

    # test using low variance
    @test sum(δ) ≈ 0 atol = 1e-3
    @test v < v₀
    @test v < 1e-2
end

begin
    ψ = repeat(InfiniteMPS(2, D), 2)
    H = repeat(H_ref, 2)

    trscheme = truncbelow(1e-8)

    # test logging
    ψ, envs, δ = find_groundstate(ψ, H,
                                  IDMRG2(; tol, verbosity=verbosity_full, maxiter=2,
                                         trscheme))

    ψ, envs, δ = find_groundstate(ψ, H,
                                  IDMRG2(; tol, verbosity=verbosity_conv, trscheme))
    v = variance(ψ, H, envs)

    # test using low variance
    @test sum(δ) ≈ 0 atol = 1e-3
    @test v < v₀
    @test v < 1e-2
end

for unit_cell_size in [1, 3]
    ψ = unit_cell_size == 1 ? InfiniteMPS(ℂ^2, ℂ^D) : repeat(ψ, unit_cell_size)
    H = repeat(H_ref, unit_cell_size)

    # test logging
    ψ, envs, δ = find_groundstate(ψ, H,
                                  GradientGrassmann(; tol, verbosity=verbosity_full,
                                                    maxiter=2))

    ψ, envs, δ = find_groundstate(ψ, H,
                                  GradientGrassmann(; tol, verbosity=verbosity_conv))
    v = variance(ψ, H, envs)

    # test using low variance
    @test sum(δ) ≈ 0 atol = 1e-3
    @test v < v₀
    @test v < 1e-2
end

for unit_cell_size in [1, 3]
    ψ = unit_cell_size == 1 ? InfiniteMPS(ℂ^2, ℂ^D) : repeat(ψ, unit_cell_size)
    H = repeat(H_ref, unit_cell_size)

    alg = VUMPS(; tol=100 * tol, verbosity=verbosity_conv, maxiter=10) &
          GradientGrassmann(; tol, verbosity=verbosity_conv, maxiter=50)
    ψ, envs, δ = find_groundstate(ψ, H, alg)

    v = variance(ψ, H, envs)

    # test using low variance
    @test sum(δ) ≈ 0 atol = 1e-3
    @test v < v₀
    @test v < 1e-2
end

########################

tol = 1e-8
g = 4.0
D = 6
L = 10

H = transverse_field_ising(; g, L)

H + H

begin
    ψ₀ = FiniteMPS(randn, ComplexF64, L, ℂ^2, ℂ^D)
    v₀ = variance(ψ₀, H)

    # test logging
    ψ, envs, δ = find_groundstate(ψ₀, H,
                                  DMRG(; verbosity=verbosity_full, maxiter=2))

    ψ, envs, δ = find_groundstate(ψ, H,
                                  DMRG(; verbosity=verbosity_conv, maxiter=10),
                                  envs)
    v = variance(ψ, H)

    # test using low variance
    @test sum(δ) ≈ 0 atol = 1e-3
    @test v < v₀ && v < 1e-2
end

begin
    ψ₀ = FiniteMPS(randn, ComplexF64, 10, ℂ^2, ℂ^D)
    v₀ = variance(ψ₀, H)
    trscheme = truncdim(floor(Int, D * 1.5))
    # test logging
    ψ, envs, δ = find_groundstate(ψ₀, H,
                                  DMRG2(; verbosity=verbosity_full, maxiter=2,
                                        trscheme))

    ψ, envs, δ = find_groundstate(ψ, H,
                                  DMRG2(; verbosity=verbosity_conv, maxiter=10,
                                        trscheme), envs)
    v = variance(ψ, H)

    # test using low variance
    @test sum(δ) ≈ 0 atol = 1e-3

    @test v < v₀ && v < 1e-2
end

begin
    ψ₀ = FiniteMPS(randn, ComplexF64, 10, ℂ^2, ℂ^D)
    v₀ = variance(ψ₀, H)

    # test logging
    ψ, envs, δ = find_groundstate(ψ₀, H,
                                  GradientGrassmann(; verbosity=verbosity_full,
                                                    maxiter=2))

    ψ, envs, δ = find_groundstate(ψ, H,
                                  GradientGrassmann(; verbosity=verbosity_conv,
                                                    maxiter=50),
                                  envs)
    v = variance(ψ, H)

    # test using low variance
    @test sum(δ) ≈ 0 atol = 1e-3
    @test v < v₀ && v < 1e-2
end

L = 3
T = ComplexF64
V = ℂ^2
lattice = fill(V, L)
O₁ = rand(T, V, V)
E = id(storagetype(O₁), domain(O₁))
O₂ = rand(T, V^2 ← V^2)

H1 = FiniteMPOHamiltonian(lattice, i => O₁ for i in 1:L)
H2 = FiniteMPOHamiltonian(lattice, (i, i + 1) => O₂ for i in 1:(L - 1))
H3 = FiniteMPOHamiltonian(lattice, 1 => O₁, (2, 3) => O₂, (1, 3) => O₂)

# check if constructor works by converting back to tensormap
H1_tm = convert(TensorMap, H1)
operators = vcat(fill(E, L - 1), O₁)
@test H1_tm ≈ mapreduce(+, 1:L) do i
    return reduce(⊗, circshift(operators, i))
end
operators = vcat(fill(E, L - 2), O₂)
@test convert(TensorMap, H2) ≈ mapreduce(+, 1:(L - 1)) do i
    return reduce(⊗, circshift(operators, i))
end
@test convert(TensorMap, H3) ≈
      O₁ ⊗ E ⊗ E + E ⊗ O₂ + permute(O₂ ⊗ E, ((1, 3, 2), (4, 6, 5)))

# check if adding terms on the same site works
single_terms = Iterators.flatten(Iterators.repeated((i => O₁ / 2 for i in 1:L), 2))
H4 = FiniteMPOHamiltonian(lattice, single_terms)
@test H4 ≈ H1 atol = 1e-6
double_terms = Iterators.flatten(Iterators.repeated(((i, i + 1) => O₂ / 2
                                                     for i in 1:(L - 1)), 2))
H5 = FiniteMPOHamiltonian(lattice, double_terms)
@test H5 ≈ H2 atol = 1e-6

# test linear algebra
@test H1 ≈
      FiniteMPOHamiltonian(lattice, 1 => O₁) +
      FiniteMPOHamiltonian(lattice, 2 => O₁) +
      FiniteMPOHamiltonian(lattice, 3 => O₁)
@test 0.8 * H1 + 0.2 * H1 ≈ H1 atol = 1e-6
@test convert(TensorMap, H1 + H2) ≈ convert(TensorMap, H1) + convert(TensorMap, H2) atol = 1e-6

# test dot and application
state = rand(T, prod(lattice))
mps = FiniteMPS(state)

@test convert(TensorMap, H1 * mps) ≈ H1_tm * state
@test H1 * state ≈ H1_tm * state
@test dot(mps, H2, mps) ≈ dot(mps, H2 * mps)

# test constructor from dictionary with mixed linear and Cartesian lattice indices as keys
grid = square = fill(V, 3, 3)

local_operators = Dict((I,) => O₁ for I in eachindex(grid))
I_vertical = CartesianIndex(1, 0)
vertical_operators = Dict((I, I + I_vertical) => O₂
                          for I in eachindex(IndexCartesian(), square)
                          if I[1] < size(square, 1))
operators = merge(local_operators, vertical_operators)
H4 = FiniteMPOHamiltonian(grid, operators)

@test H4 ≈
      FiniteMPOHamiltonian(grid, local_operators) +
      FiniteMPOHamiltonian(grid, vertical_operators)

pspace = ℂ^2
Dspace = ℂ^D
Os = map(1:3) do i
    O = rand(ComplexF64, pspace^i, pspace^i)
    return O += O'
end
fs = [t -> 1t, 1, 1]

L = 5
ψ = FiniteMPS(rand, ComplexF64, L, pspace, Dspace)
lattice = fill(pspace, L)
Hs = map(enumerate(Os)) do (i, O)
    return FiniteMPOHamiltonian(lattice,
                                ntuple(x -> x + j, i) => O for j in 0:(L - i))
end
summedH = LazySum(Hs)

envs = map(H -> environments(ψ, H), Hs)
summed_envs = environments(ψ, summedH)

expval = sum(zip(Hs, envs)) do (H, env)
    return expectation_value(ψ, H, env)
end
expval1 = expectation_value(ψ, sum(summedH))
expval2 = expectation_value(ψ, summedH, summed_envs)
expval3 = expectation_value(ψ, summedH)
@test expval ≈ expval1
@test expval ≈ expval2
@test expval ≈ expval3

# test derivatives
summedhct = MPSKit.∂∂C(1, ψ, summedH, summed_envs)
sum1 = sum(zip(Hs, envs)) do (H, env)
    return MPSKit.∂∂C(1, ψ, H, env)(ψ.C[1])
end
@test summedhct(ψ.C[1], 0.0) ≈ sum1

summedhct = MPSKit.∂∂AC(1, ψ, summedH, summed_envs)
sum2 = sum(zip(Hs, envs)) do (H, env)
    return MPSKit.∂∂AC(1, ψ, H, env)(ψ.AC[1])
end
@test summedhct(ψ.AC[1], 0.0) ≈ sum2

v = _transpose_front(ψ.AC[1]) * _transpose_tail(ψ.AR[2])
summedhct = MPSKit.∂∂AC2(1, ψ, summedH, summed_envs)
sum3 = sum(zip(Hs, envs)) do (H, env)
    return MPSKit.∂∂AC2(1, ψ, H, env)(v)
end
@test summedhct(v, 0.0) ≈ sum3

Hts = [MultipliedOperator(Hs[1], fs[1]), MultipliedOperator(Hs[2], fs[2]), Hs[3]]
summedH = LazySum(Hts)
t = 1.1
summedH_at = summedH(t)
@which summedH(t)

envs = map(H -> environments(ψ, H), Hs)
summed_envs = environments(ψ, summedH)

expval = sum(zip(fs, Hs, envs)) do (f, H, env)
    return (f isa Function ? f(t) : f) * expectation_value(ψ, H, env)
end
expval1 = expectation_value(ψ, sum(summedH_at))
expval2 = expectation_value(ψ, summedH_at, summed_envs)
expval3 = expectation_value(ψ, summedH_at)
@test expval ≈ expval1
@test expval ≈ expval2
@test expval ≈ expval3

# test derivatives
summedhct = MPSKit.∂∂C(1, ψ, summedH, summed_envs)
sum1 = sum(zip(fs, Hs, envs)) do (f, H, env)
    if f isa Function
        f = f(t)
    end
    return f * MPSKit.∂∂C(1, ψ, H, env)(ψ.C[1])
end
@test summedhct(ψ.C[1], t) ≈ sum1

summedhct = MPSKit.∂∂AC(1, ψ, summedH, summed_envs)
sum2 = sum(zip(fs, Hs, envs)) do (f, H, env)
    if f isa Function
        f = f(t)
    end
    return f * MPSKit.∂∂AC(1, ψ, H, env)(ψ.AC[1])
end
@test summedhct(ψ.AC[1], t) ≈ sum2

v = _transpose_front(ψ.AC[1]) * _transpose_tail(ψ.AR[2])
summedhct = MPSKit.∂∂AC2(1, ψ, summedH, summed_envs)
sum3 = sum(zip(fs, Hs, envs)) do (f, H, env)
    return (f isa Function ? f(t) : f) * MPSKit.∂∂AC2(1, ψ, H, env)(v)
end
@test summedhct(v, t) ≈ sum3
# ------------------------------------------------------
dt = 0.1
algs = [TDVP(), TDVP2()]
L = 10

H = (TestSetup.heisenberg_XXX(; spin=1 // 2, L))
ψ₀ = FiniteMPS(L, ℂ^2, ℂ^1)
E₀ = expectation_value(ψ₀, H)

@testset "Finite $(alg isa TDVP ? "TDVP" : "TDVP2")" for alg in algs
    ψ, envs = timestep(ψ₀, H, 0.0, dt, alg)
    E = expectation_value(ψ, H, envs)
    @test E₀ ≈ E atol = 1e-2
end

Hlazy = LazySum([3 * H, 1.55 * H, -0.1 * H])

@testset "Finite LazySum $(alg isa TDVP ? "TDVP" : "TDVP2")" for alg in algs
    ψ, envs = timestep(ψ₀, Hlazy, 0.0, dt, alg)
    E = expectation_value(ψ, Hlazy, envs)
    @test (3 + 1.55 - 0.1) * E₀ ≈ E atol = 1e-2
end

Ht = MultipliedOperator(H, t -> 4) + MultipliedOperator(H, 1.45);

@testset "Finite TimeDependent LazySum $(alg isa TDVP ? "TDVP" : "TDVP2")" for alg in
                                                                               algs
    ψ, envs = timestep(ψ₀, Ht(1.0), 0.0, dt, alg)
    E = expectation_value(ψ, Ht(1.0), envs)

    ψt, envst = timestep(ψ₀, Ht, 1.0, dt, alg)
    Et = expectation_value(ψt, Ht(1.0), envst)
    @test E ≈ Et atol = 1e-8
end

H = repeat(TestSetup.heisenberg_XXX(; spin=1), 2)
ψ₀ = InfiniteMPS([ℂ^3, ℂ^3], [ℂ^50, ℂ^50])
E₀ = expectation_value(ψ₀, H)

@testset "Infinite TDVP" begin
    ψ, envs = timestep(ψ₀, H, 0.0, dt, TDVP())
    E = expectation_value(ψ, H, envs)
    @test E₀ ≈ E atol = 1e-2
end

Hlazy = LazySum([3 * deepcopy(H), 1.55 * deepcopy(H), -0.1 * deepcopy(H)])

@testset "Infinite LazySum TDVP" begin
    ψ, envs = timestep(ψ₀, Hlazy, 0.0, dt, TDVP())
    E = expectation_value(ψ, Hlazy, envs)
    @test (3 + 1.55 - 0.1) * E₀ ≈ E atol = 1e-2
end

Ht = MultipliedOperator(H, t -> 4) + MultipliedOperator(H, 1.45)

@testset "Infinite TimeDependent LazySum" begin
    ψ, envs = timestep(ψ₀, Ht(1.0), 0.0, dt, TDVP())
    E = expectation_value(ψ, Ht(1.0), envs)

    ψt, envst = timestep(ψ₀, Ht, 1.0, dt, TDVP())
    Et = expectation_value(ψt, Ht(1.0), envst)
    @test E ≈ Et atol = 1e-8
end
