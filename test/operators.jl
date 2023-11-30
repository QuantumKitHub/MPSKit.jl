println("
-----------------
|   Operators   |
-----------------
")

include("setup.jl")

pspaces = (ℙ^4, Rep[U₁](0 => 2), Rep[SU₂](1 => 1))
vspaces = (ℙ^10, Rep[U₁]((0 => 20)), Rep[SU₂](1//2 => 10, 3//2 => 5, 5//2 => 1))

@testset "MPOHamiltonian $(sectortype(pspace))" for (pspace, Dspace) in
                                                    zip(pspaces, vspaces)
    #generate a 1-2-3 body interaction
    n = TensorMap(rand, ComplexF64, pspace, pspace)
    n += n'
    nn = TensorMap(rand, ComplexF64, pspace * pspace, pspace * pspace)
    nn += nn'
    nnn = TensorMap(rand, ComplexF64, pspace * pspace * pspace, pspace * pspace * pspace)
    nnn += nnn'

    #can you pass in a proper mpo?
    identity = complex(isomorphism(oneunit(pspace) * pspace, pspace * oneunit(pspace)))
    mpoified = MPSKit.decompose_localmpo(MPSKit.add_util_leg(nnn))
    d3 = Array{Union{Missing,typeof(identity)},3}(missing, 1, 4, 4)
    d3[1, 1, 1] = identity
    d3[1, end, end] = identity
    d3[1, 1, 2] = mpoified[1]
    d3[1, 2, 3] = mpoified[2]
    d3[1, 3, 4] = mpoified[3]
    h1 = MPOHamiltonian(d3)

    #¢an you pass in the actual hamiltonian?
    h2 = MPOHamiltonian(nn)

    #can you generate a hamiltonian using only onsite interactions?
    d1 = Array{Any,3}(missing, 2, 3, 3)
    d1[1, 1, 1] = 1
    d1[1, end, end] = 1
    d1[1, 1, 2] = n
    d1[1, 2, end] = n
    d1[2, 1, 1] = 1
    d1[2, end, end] = 1
    d1[2, 1, 2] = n
    d1[2, 2, end] = n
    h3 = MPOHamiltonian(d1)

    #make a teststate to measure expectation values for
    ts1 = InfiniteMPS([pspace], [Dspace])
    ts2 = InfiniteMPS([pspace, pspace], [Dspace, Dspace])

    e1 = expectation_value(ts1, h1)
    e2 = expectation_value(ts1, h2)

    h1 = 2 * h1 - [1]
    @test e1[1] * 2 - 1 ≈ expectation_value(ts1, h1)[1] atol = 1e-10

    h1 = h1 + h2

    @test e1[1] * 2 + e2[1] - 1 ≈ expectation_value(ts1, h1)[1] atol = 1e-10

    h1 = repeat(h1, 2)

    e1 = expectation_value(ts2, h1)
    e3 = expectation_value(ts2, h3)

    @test e1 + e3 ≈ expectation_value(ts2, h1 + h3) atol = 1e-10

    h4 = h1 + h3
    h4 = h4 * h4
    @test real(sum(expectation_value(ts2, h4))) >= 0
end

@testset "General LazySum of $(eltype(Os))" for Os in (
    rand(ComplexF64, rand(1:10)),
    map(i -> TensorMap(rand, ComplexF64, ℂ^13, ℂ^7), 1:rand(1:10)),
    map(i -> TensorMap(rand, ComplexF64, ℂ^1⊗ℂ^2,ℂ^3⊗ℂ^4), 1:rand(1:10)),
)
    LazyOs = LazySum(Os)

    #test user interface
    summed = sum(Os)

    @test sum(LazyOs) ≈ summed atol = 1 - 08

    LazyOs_added = +(LazyOs, Os...)

    @test sum(LazyOs_added) ≈ 2 * summed atol = 1 - 08
end

@testset "DenseMPO" for ham in (transverse_field_ising(), heisenberg_XXX(; spin=1))
    physical_space = ham.pspaces[1]
    ou = oneunit(physical_space)

    ts = InfiniteMPS([physical_space], [ou ⊕ physical_space])

    W = convert(DenseMPO, make_time_mpo(ham, 1im * 0.5, WII()))

    @test abs(dot(W * (W * ts), (W * W) * ts)) ≈ 1.0 atol = 1e-10
end

pspaces = (ℙ^4, Rep[U₁](0 => 2), Rep[SU₂](1 => 1, 2 => 1))
vspaces = (ℙ^10, Rep[U₁]((0 => 20)), Rep[SU₂](1 => 10, 3 => 5, 5 => 1))

@testset "LazySum of (effective) Hamiltonian $(sectortype(pspace))" for (pspace, Dspace) in
                                                                        zip(
    pspaces, vspaces
)
    n = TensorMap(rand, ComplexF64, pspace, pspace)
    n += n'
    nn = TensorMap(rand, ComplexF64, pspace * pspace, pspace * pspace)
    nn += nn'
    nnn = TensorMap(rand, ComplexF64, pspace * pspace * pspace, pspace * pspace * pspace)
    nnn += nnn'

    H1 = repeat(MPOHamiltonian(n), 2)
    H2 = repeat(MPOHamiltonian(nn), 2)
    H3 = repeat(MPOHamiltonian(nnn), 2)
    Hs = [H1, H2, H3]
    summedH = LazySum(Hs)

    Ψs = [
        FiniteMPS(rand, ComplexF64, rand(3:2:20), pspace, Dspace),
        InfiniteMPS([
            TensorMap(rand, ComplexF64, Dspace * pspace, Dspace),
            TensorMap(rand, ComplexF64, Dspace * pspace, Dspace),
        ]),
    ]

    @testset "LazySum $(Ψ isa FiniteMPS ? "F" : "Inf")initeMPS" for Ψ in Ψs
        Envs = map(H -> environments(Ψ, H), Hs)
        summedEnvs = environments(Ψ, summedH)

        expval = sum(zip(Hs, Envs)) do (H, Env)
            expectation_value(Ψ, H, Env)
        end
        expval1 = expectation_value(Ψ, sum(summedH))
        expval2 = expectation_value(Ψ, summedH, summedEnvs)
        expval3 = expectation_value(Ψ, summedH)
        @test expval ≈ expval1
        @test expval ≈ expval2
        @test expval ≈ expval3

        # test derivatives
        summedhct = MPSKit.∂∂C(1, Ψ, summedH, summedEnvs)
        sum1 = sum(zip(Hs, Envs)) do (H, env)
            MPSKit.∂∂C(1, Ψ, H, env)(Ψ.CR[1])
        end
        @test summedhct(Ψ.CR[1]) ≈ sum1

        summedhct = MPSKit.∂∂AC(1, Ψ, summedH, summedEnvs)
        sum2 = sum(zip(Hs, Envs)) do (H, env)
            MPSKit.∂∂AC(1, Ψ, H, env)(Ψ.AC[1])
        end
        @test summedhct(Ψ.AC[1]) ≈ sum2

        v = MPSKit._transpose_front(Ψ.AC[1]) * MPSKit._transpose_tail(Ψ.AR[2])
        summedhct = MPSKit.∂∂AC2(1, Ψ, summedH, summedEnvs)
        sum3 = sum(zip(Hs, Envs)) do (H, env)
            MPSKit.∂∂AC2(1, Ψ, H, env)(v)
        end
        @test summedhct(v) ≈ sum3
    end
end
