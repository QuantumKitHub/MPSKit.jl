using MPSKit, MPSKitModels, TensorKit
using Random
using Test

Random.seed!(123456)
L = 11
T = ComplexF64
chain = FiniteChain(L)
symmetry = Trivial
spin = 1
J = 1

H = heisenberg_XXX(symmetry, chain; J, spin)

physical_space = ℂ^3
virtual_space = ℂ^1
# A = zeros(ComplexF64, virtual_space ⊗ physical_space ← virtual_space)
# A.data .= [1, 0.1, -10.0]
# rand!(A.data)

psi = FiniteMPS(rand, T, L, physical_space, virtual_space)
# psi = FiniteMPS(fill(A, L))
gs, envs, delta = find_groundstate(psi, H, DMRG(; verbosity = 0));

Sx = S_x(T, symmetry; spin = 1)

# E_x = map(eachindex(psi)) do i
#     expectation_value(psi, i => Sx)
# end
# E_xx = map(eachindex(psi)) do i
#     expectation_value(psi, (i,) => Sx^2)
# end
# E_xx2 = map(eachindex(psi)) do i
#     Sx_normalized = add(Sx, id(space(Sx, 1)), -E_x[i])
#     expectation_value(psi, i => Sx_normalized^2)
# end

# E_xx .- E_x .^ 2
# E_xx2
# test for random state D = 1
for i in 1:length(chain)
    # i = length(chain) ÷ 2
    E_x = expectation_value(psi, i => Sx)
    E_xx = expectation_value(psi, (i,) => Sx^2)
    E_xx2 = expectation_value(psi, (i,) => add(Sx, id(space(Sx, 1)), -E_x)^2)

    # @info i E_x E_xx E_xx - E_x^2 E_xx2
    @test E_xx - E_x^2 ≈ E_xx2
end

# test for groundstate D = 1
gs, envs, delta = find_groundstate(psi, H, DMRG(; verbosity = 0));
for i in 1:length(chain)
    # i = length(chain) ÷ 2
    E_x = expectation_value(gs, i => Sx)
    E_xx = expectation_value(gs, (i,) => Sx^2)
    E_xx2 = expectation_value(gs, (i,) => add(Sx, id(space(Sx, 1)), -E_x)^2)

    # @info i E_x E_xx E_xx - E_x^2 E_xx2
    @test E_xx - E_x^2 ≈ E_xx2
end

virtual_space = ℂ^15
psi = FiniteMPS(rand, T, L, physical_space, virtual_space)
for i in 1:length(chain)
    # i = length(chain) ÷ 2
    E_x = expectation_value(psi, i => Sx)
    E_xx = expectation_value(psi, (i,) => Sx^2)
    E_xx2 = expectation_value(psi, (i,) => add(Sx, id(space(Sx, 1)), -E_x)^2)

    # @info i E_x E_xx E_xx - E_x^2 E_xx2
    @test E_xx - E_x^2 ≈ E_xx2
end

gs, envs, delta = find_groundstate(psi, H, DMRG(; verbosity = 0));
for i in 1:length(chain)
    # i = length(chain) ÷ 2
    E_x = expectation_value(gs, i => Sx)
    E_xx = expectation_value(gs, (i,) => Sx^2)
    E_xx2 = expectation_value(gs, (i,) => add(Sx, id(space(Sx, 1)), -E_x)^2)

    # @info i E_x E_xx E_xx - E_x^2 E_xx2
    @test E_xx - E_x^2 ≈ E_xx2
end

println()
#=
┌ Info: 1
│   E_x = 0.9283027892120569
│   E_xx = 0.9873406410374899
│   E_xx - E_x ^ 2 = 0.1255945725786053
└   E_xx2 = 0.1255945725786054
┌ Info: 2
│   E_x = 0.5958931587232229
│   E_xx = 0.8842424256151229
│   E_xx - E_x ^ 2 = 0.5291537690019827
└   E_xx2 = 0.5291537690019827
┌ Info: 3
│   E_x = 0.9119512168846627
│   E_xx = 0.9410385403620483
│   E_xx - E_x ^ 2 = 0.10938351838463112
└   E_xx2 = 0.10938351838463076
┌ Info: 4
│   E_x = 0.8475876798883178
│   E_xx = 0.9998527656967889
│   E_xx - E_x ^ 2 = 0.2814478905983274
└   E_xx2 = 0.28144789059832703
┌ Info: 5
│   E_x = 0.5411573880001049
│   E_xx = 0.5994538976943021
│   E_xx - E_x ^ 2 = 0.306602579107206
└   E_xx2 = 0.306602579107206
┌ Info: 6
│   E_x = 0.99952372893415
│   E_xx = 0.9996661378739403
│   E_xx - E_x ^ 2 = 0.0006184531715122121
└   E_xx2 = 0.0006184531715116068
┌ Info: 7
│   E_x = 0.7484424712641702
│   E_xx = 0.7491543238199131
│   E_xx - E_x ^ 2 = 0.18898819102789488
└   E_xx2 = 0.18898819102789452
┌ Info: 8
│   E_x = 0.9525317843713124
│   E_xx = 0.9548671515022985
│   E_xx - E_x ^ 2 = 0.047550351264702195
└   E_xx2 = 0.04755035126470154
┌ Info: 9
│   E_x = 0.5371423149073694
│   E_xx = 0.728001367170992
│   E_xx - E_x ^ 2 = 0.43947950070694436
└   E_xx2 = 0.43947950070694425
┌ Info: 10
│   E_x = 0.9289084036689698
│   E_xx = 0.9917741739555096
│   E_xx - E_x ^ 2 = 0.12890335154867594
└   E_xx2 = 0.12890335154867597
┌ Info: 11
│   E_x = 0.748255413094344
│   E_xx = 0.789444809740308
│   E_xx - E_x ^ 2 = 0.22955864651532065
└   E_xx2 = 0.2295586465153205
=#
