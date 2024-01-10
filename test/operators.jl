println("
-----------------
|   Operators   |
-----------------
")

include("setup.jl")

pspaces = (ℙ^4, Rep[U₁](0 => 2), Rep[SU₂](1 => 1))
vspaces = (ℙ^10, Rep[U₁]((0 => 20)), Rep[SU₂](1 // 2 => 10, 3 // 2 => 5, 5 // 2 => 1))

@testset "MPOHamiltonian $(sectortype(pspace))" for (pspace, Dspace) in zip(pspaces,
                                                                            vspaces)
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
    ψ1 = InfiniteMPS([pspace], [Dspace])
    ψ2 = InfiniteMPS([pspace, pspace], [Dspace, Dspace])

    e1 = expectation_value(ψ1, h1)
    e2 = expectation_value(ψ1, h2)

    h1 = 2 * h1 - [1]
    @test e1[1] * 2 - 1 ≈ expectation_value(ψ1, h1)[1] atol = 1e-10

    h1 = h1 + h2

    @test e1[1] * 2 + e2[1] - 1 ≈ expectation_value(ψ1, h1)[1] atol = 1e-10

    h1 = repeat(h1, 2)

    e1 = expectation_value(ψ2, h1)
    e3 = expectation_value(ψ2, h3)

    @test e1 + e3 ≈ expectation_value(ψ2, h1 + h3) atol = 1e-10

    h4 = h1 + h3
    h4 = h4 * h4
    @test real(sum(expectation_value(ψ2, h4))) >= 0
end

@testset "General LazySum of $(eltype(Os))" for Os in (rand(ComplexF64, rand(1:10)),
                                                       map(i -> TensorMap(rand, ComplexF64,
                                                                          ℂ^13, ℂ^7),
                                                           1:rand(1:10)),
                                                       map(i -> TensorMap(rand, ComplexF64,
                                                                          ℂ^1 ⊗ ℂ^2,
                                                                          ℂ^3 ⊗ ℂ^4),
                                                           1:rand(1:10)))
    LazyOs = LazySum(Os)

    #test user interface
    summed = sum(Os)

    @test sum(LazyOs) ≈ summed atol = 1 - 08

    LazyOs_added = +(LazyOs, Os...)

    @test sum(LazyOs_added) ≈ 2 * summed atol = 1 - 08
end

@testset "MulitpliedOperator of $(typeof(O)) with $(typeof(f))" for (O, f) in
                                                                    zip((rand(ComplexF64),
                                                                         TensorMap(rand,
                                                                                   ComplexF64,
                                                                                   ℂ^13,
                                                                                   ℂ^7),
                                                                         TensorMap(rand,
                                                                                   ComplexF64,
                                                                                   ℂ^1 ⊗
                                                                                   ℂ^2,
                                                                                   ℂ^3 ⊗
                                                                                   ℂ^4)),
                                                                        (t -> 3t, 1.1,
                                                                         One()))
    tmp = MPSKit.MultipliedOperator(O, f)
    if tmp isa TimedOperator
        @test tmp(1.1)() ≈ f(1.1) * O atol = 1 - 08
    elseif tmp isa UntimedOperator
        @test tmp() ≈ f * O atol = 1 - 08
    end
end

@testset "General Time-dependent LazySum of $(eltype(Os))" for Os in (rand(ComplexF64, 4),
                                                                      fill(TensorMap(rand,
                                                                                     ComplexF64,
                                                                                     ℂ^13, ℂ^7),
                                                                           4),
                                                                      fill(TensorMap(rand,
                                                                                     ComplexF64,
                                                                                     ℂ^1 ⊗ ℂ^2,
                                                                                     ℂ^3 ⊗ ℂ^4),
                                                                           4))

    #test user interface
    fs = [t -> 3t, t -> t + 2, 4, 1]
    Ofs = map(zip(fs, Os)) do (f, O)
        if f == 1
            return O
        else
            return MPSKit.MultipliedOperator(O, f)
        end
    end
    LazyOs = LazySum(Ofs)
    summed = sum(zip(fs, Os)) do (f, O)
        if f isa Function
            f(1.1) * O
        else
            f * O
        end
    end

    @test sum(LazyOs(1.1)) ≈ summed atol = 1 - 08

    LazyOs_added = +(LazyOs, Ofs...)

    @test sum(LazyOs_added(1.1)) ≈ 2 * summed atol = 1 - 08
end

@testset "DenseMPO" for ham in (transverse_field_ising(), heisenberg_XXX(; spin=1))
    physical_space = ham.pspaces[1]
    ou = oneunit(physical_space)

    ψ = InfiniteMPS([physical_space], [ou ⊕ physical_space])

    W = convert(DenseMPO, make_time_mpo(ham, 1im * 0.5, WII()))

    @test abs(dot(W * (W * ψ), (W * W) * ψ)) ≈ 1.0 atol = 1e-10
end

pspaces = (ℙ^4, Rep[U₁](0 => 2), Rep[SU₂](1 => 1, 2 => 1))
vspaces = (ℙ^10, Rep[U₁]((0 => 20)), Rep[SU₂](1 => 10, 3 => 5, 5 => 1))

@testset "LazySum of (effective) Hamiltonian $(sectortype(pspace))" for (pspace, Dspace) in
                                                                        zip(pspaces,
                                                                            vspaces)
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

    ψs = [FiniteMPS(rand, ComplexF64, rand(3:2:20), pspace, Dspace),
          InfiniteMPS([TensorMap(rand, ComplexF64, Dspace * pspace, Dspace),
                       TensorMap(rand, ComplexF64, Dspace * pspace, Dspace)])]

    @testset "LazySum $(ψ isa FiniteMPS ? "F" : "Inf")initeMPS" for ψ in ψs
        Envs = map(H -> environments(ψ, H), Hs)
        summedEnvs = environments(ψ, summedH)

        expval = sum(zip(Hs, Envs)) do (H, Env)
            return expectation_value(ψ, H, Env)
        end
        expval1 = expectation_value(ψ, sum(summedH))
        expval2 = expectation_value(ψ, summedH, summedEnvs)
        expval3 = expectation_value(ψ, summedH)
        @test expval ≈ expval1
        @test expval ≈ expval2
        @test expval ≈ expval3

        # test derivatives
        summedhct = MPSKit.∂∂C(1, ψ, summedH, summedEnvs)
        sum1 = sum(zip(Hs, Envs)) do (H, env)
            return MPSKit.∂∂C(1, ψ, H, env)(ψ.CR[1])
        end
        @test summedhct(ψ.CR[1], 0.0) ≈ sum1

        summedhct = MPSKit.∂∂AC(1, ψ, summedH, summedEnvs)
        sum2 = sum(zip(Hs, Envs)) do (H, env)
            return MPSKit.∂∂AC(1, ψ, H, env)(ψ.AC[1])
        end
        @test summedhct(ψ.AC[1], 0.0) ≈ sum2

        v = MPSKit._transpose_front(ψ.AC[1]) * MPSKit._transpose_tail(ψ.AR[2])
        summedhct = MPSKit.∂∂AC2(1, ψ, summedH, summedEnvs)
        sum3 = sum(zip(Hs, Envs)) do (H, env)
            return MPSKit.∂∂AC2(1, ψ, H, env)(v)
        end
        @test summedhct(v, 0.0) ≈ sum3
    end

    fs = [t -> 3t, 2, 1]
    Hts = [MultipliedOperator(H1, fs[1]), MultipliedOperator(H2, fs[2]), H3]
    summedH = LazySum(Hts)
    t = 1.1
    summedH_at = summedH(t)

    @testset "Time-dependent LazySum $(ψ isa FiniteMPS ? "F" : "Inf")initeMPS" for ψ in ψs
        Envs = map(H -> environments(ψ, H), Hs)
        summedEnvs = environments(ψ, summedH)

        expval = sum(zip(fs, Hs, Envs)) do (f, H, Env)
            if f isa Function
                f = f(t)
            end
            return f * expectation_value(ψ, H, Env)
        end
        expval1 = expectation_value(ψ, sum(summedH_at))
        expval2 = expectation_value(ψ, summedH_at, summedEnvs)
        expval3 = expectation_value(ψ, summedH_at)
        @test expval ≈ expval1
        @test expval ≈ expval2
        @test expval ≈ expval3

        # test derivatives
        summedhct = MPSKit.∂∂C(1, ψ, summedH, summedEnvs)
        sum1 = sum(zip(fs, Hs, Envs)) do (f, H, env)
            if f isa Function
                f = f(t)
            end
            return f * MPSKit.∂∂C(1, ψ, H, env)(ψ.CR[1])
        end
        @test summedhct(ψ.CR[1], t) ≈ sum1

        summedhct = MPSKit.∂∂AC(1, ψ, summedH, summedEnvs)
        sum2 = sum(zip(fs, Hs, Envs)) do (f, H, env)
            if f isa Function
                f = f(t)
            end
            return f * MPSKit.∂∂AC(1, ψ, H, env)(ψ.AC[1])
        end
        @test summedhct(ψ.AC[1], t) ≈ sum2

        v = MPSKit._transpose_front(ψ.AC[1]) * MPSKit._transpose_tail(ψ.AR[2])
        summedhct = MPSKit.∂∂AC2(1, ψ, summedH, summedEnvs)
        sum3 = sum(zip(fs, Hs, Envs)) do (f, H, env)
            if f isa Function
                f = f(t)
            end
            return f * MPSKit.∂∂AC2(1, ψ, H, env)(v)
        end
        @test summedhct(v, t) ≈ sum3
    end
end
