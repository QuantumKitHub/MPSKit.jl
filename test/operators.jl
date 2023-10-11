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

@testset "DenseMPO" for ham in (transverse_field_ising(), heisenberg_XXX(; spin=1))
    physical_space = ham.pspaces[1]
    ou = oneunit(physical_space)

    ts = InfiniteMPS([physical_space], [ou ⊕ physical_space])

    W = convert(DenseMPO, make_time_mpo(ham, 1im * 0.5, WII()))

    @test abs(dot(W * (W * ts), (W * W) * ts)) ≈ 1.0 atol = 1e-10
end

@testset "Timed/SumOf Operators $(sectortype(pspace))" for (pspace, Dspace) in [
    (ℙ^4, ℙ^10),
    (Rep[U₁](0 => 2), Rep[U₁]((0 => 20))),
    (Rep[SU₂](1 => 1), Rep[SU₂](1//2 => 10, 3//2 => 5, 5//2 => 1)),
]
    O = TensorMap(rand, ComplexF64, Dspace * pspace, Dspace * pspace)
    f(t) = 3 * exp(t)

    timedO = TimedOperator(O, f)

    @test timedO(0.5) == f(0.5) * O

    # SumOfOperators
    Os = map(i -> TensorMap(rand, ComplexF64, Dspace * pspace, Dspace * pspace), 1:4)
    fs = [t -> 3 + t, t -> 7 * t, t -> 2 * cos(t), t -> t^2]

    # different ways of constructing SumOfOperators
    SummedOs = SumOfOperators(Os, fs) #direct construction
    SummedOs2 = sum(map((O, f) -> TimedOperator(O, f), Os, fs)) # sum the different timedoperators using defined +

    @test SummedOs(0.5) == sum(map((O, f) -> f(0.5) * O, Os, fs))
    @test SummedOs(1.0) == SummedOs2(1.0)
end

@testset "Timed/SumOf (effective) Hamiltonian $(sectortype(pspace))" for (
    pspace, Dspace, HDspace
) in [
    (ℙ^4, ℙ^10, ℙ^2),
    (Rep[U₁](0 => 2), Rep[U₁]((0 => 20)), Rep[U₁]((0 => 4))),
    (Rep[SU₂](0 => 2), Rep[SU₂](1 => 1, 0 => 3), Rep[SU₂](0 => 1)),
]
    Os = map(
        (D1, D2) -> TensorMap(rand, ComplexF64, D1 * pspace, pspace * D2),
        [oneunit(HDspace), HDspace],
        [HDspace, oneunit(HDspace)],
    )
    H = repeat(MPOHamiltonian(Os), 2)
    f(t) = 3 * exp(0.1 * t)
    Ht = TimedOperator(H, f)

    Ψs = [
        FiniteMPS(rand, ComplexF64, rand(3:20), pspace, Dspace),
        InfiniteMPS([
            TensorMap(rand, ComplexF64, Dspace * pspace, Dspace),
            TensorMap(rand, ComplexF64, Dspace * pspace, Dspace),
        ]),
    ]

    @testset "TimedOperator $(Ψ isa InfiniteMPS ? "InfiniteMPS" : "FiniteMPS")" for Ψ in Ψs
        envs = environments(Ψ, H)
        envs2 = environments(Ψ, Ht(3.0))
        envst = environments(Ψ, Ht)

        @test envs.opp.data == envst.opp.data #check that env are the same, time-dep sits elsewhere
        @test f(3.0) .* expectation_value(Ψ, H) ≈ expectation_value(Ψ, Ht(3.0))


        ## time-dependence of derivatives
        hc = MPSKit.∂∂C(1, Ψ, H, envs)
        hct = MPSKit.∂∂C(1, Ψ, Ht, envst)

        @test norm(hct(Ψ.CR[1], 3.0) - f(3.0) * hc(Ψ.CR[1])) < 1e-5

        hac = MPSKit.∂∂AC(1, Ψ, H, envs)
        hact = MPSKit.∂∂AC(1, Ψ, Ht, envst)

        @test norm(hact(Ψ.AC[1], 3.0) - f(3.0) * hac(Ψ.AC[1])) < 1e-5

        hac2 = MPSKit.∂∂AC2(1, Ψ, H, envs)
        hac2t = MPSKit.∂∂AC2(1, Ψ, Ht, envst)

        v = MPSKit._transpose_front(Ψ.AC[1]) * MPSKit._transpose_tail(Ψ.AR[2])

        @test norm(hac2t(v, 3.0) - f(3.0) * hac2(v)) < 1e-5
    end

    ##########################
    #tests for sumofoperators
    ##########################

    Os = map(
        (D1, D2) -> TensorMap(rand, ComplexF64, D1 * pspace, pspace * D2),
        [oneunit(HDspace), HDspace, HDspace, HDspace],
        [HDspace, HDspace, HDspace, oneunit(HDspace)],
    )
    Hs = repeat.(map(i -> MPOHamiltonian([Os[1:i]..., Os[end]]), 1:(length(Os) - 1)), 2)
    fs = [t -> 3 + t, t -> 7 * t, t -> 2 * cos(t), t -> t^2]
    summedH = SumOfOperators(Hs, fs)

    @testset "Timed Sum $(Ψ isa FiniteMPS ? "F" : "Inf")initeMPS" for Ψ in Ψs
        Envs = map(H -> environments(Ψ, H), Hs)
        summedEnvs = environments(Ψ, summedH)
    
        expval1 = sum(zip(Hs, fs)) do (H, f)
            f(5.0) * expectation_value(Ψ, H)
        end
        expval2 = expectation_value(Ψ, summedH(5.0))
        @test expval1 ≈ expval2 

        # test derivatives
        summedhct = MPSKit.∂∂C(1, Ψ, summedH, summedEnvs)
        sum1 = sum(zip(Hs, fs, Envs)) do (H, f, env)
            f(5.0) * MPSKit.∂∂C(1, Ψ, H, env)(Ψ.CR[1])
        end
        @test summedhct(Ψ.CR[1], 5.0) ≈ sum1

        summedhct = MPSKit.∂∂AC(1, Ψ, summedH, summedEnvs)
        sum2 = sum(zip(Hs, fs, Envs)) do (H, f, env)
            f(5.0) * MPSKit.∂∂AC(1, Ψ, H, env)(Ψ.AC[1])
        end
        @test summedhct(Ψ.AC[1], 5.0) ≈ sum2

        v = MPSKit._transpose_front(Ψ.AC[1]) * MPSKit._transpose_tail(Ψ.AR[2])
        summedhct = MPSKit.∂∂AC2(1, Ψ, summedH, summedEnvs)
        sum3 = sum(zip(Hs, fs, Envs)) do (H, f, env)
            f(5.0) * MPSKit.∂∂AC2(1, Ψ, H, env)(v)
        end
        @test summedhct(v, 5.0) ≈ sum3
    end

    # finally test in case SumOfOperators contains non-timed operators
    fs = [3, 5.0, 10, 1]
    summedH = SumOfOperators(Hs, fs)

    @testset "Untimed Sum $(Ψ isa FiniteMPS ? "F" : "Inf")initeMPS" for Ψ in Ψs
        Envs = map(H -> environments(Ψ, H), Hs)
        summedEnvs = environments(Ψ, summedH)
        
        expval1 = sum(zip(Hs, fs)) do (H, f)
            f * expectation_value(Ψ, H)
        end
        expval2 = expectation_value(Ψ, summedH(5.0))
        @test expval1 ≈ expval2

        # test derivatives
        summedhct = MPSKit.∂∂C(1, Ψ, summedH, summedEnvs)
        sum1 = sum(zip(Hs, fs, Envs)) do (H, f, env)
            f * MPSKit.∂∂C(1, Ψ, H, env)(Ψ.CR[1])
        end
        @test summedhct(Ψ.CR[1], 5.0) ≈ sum1

        summedhct = MPSKit.∂∂AC(1, Ψ, summedH, summedEnvs)
        sum2 = sum(zip(Hs, fs, Envs)) do (H, f, env)
            f * MPSKit.∂∂AC(1, Ψ, H, env)(Ψ.AC[1])
        end
        @test summedhct(Ψ.AC[1], 5.0) ≈ sum2

        v = MPSKit._transpose_front(Ψ.AC[1]) * MPSKit._transpose_tail(Ψ.AR[2])
        summedhct = MPSKit.∂∂AC2(1, Ψ, summedH, summedEnvs)
        sum3 = sum(zip(Hs, fs, Envs)) do (H, f, env)
            f * MPSKit.∂∂AC2(1, Ψ, H, env)(v)
        end
        @test summedhct(v, 5.0) ≈ sum3
    end
end
