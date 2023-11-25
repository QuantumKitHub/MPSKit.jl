println("
-----------------
|   Operators   |
-----------------
")

include("setup.jl")

pspaces = (ℙ^2, Rep[U₁](0 => 2), Rep[SU₂](1 => 1))
vspaces = (ℙ^4, Rep[U₁]((0 => 10)), Rep[SU₂](0 => 5, 1 => 2, 2 => 1))

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

@testset "Simple MultipliedOperator/LazySum" begin

    a = 1.4678
    O = 1.1
    Ou = UntimedOperator(O,a)
    Ot = TimedOperator(O,t->(1+a)*t)
    Os = LazySum([Ou,Ot,O])
    Osu = LazySum([Ou,Ou,O])

    #test user interface
    @test Ou() == a*O
    @test Ot(3.5)() == 3.5*(1+a)*O
    @test Os(3.5)() == Ou()+Ot(3.5)()+O
    @test ConvertOperator(Os,3.5) == Ou()+ConvertOperator(Ot,3.5)+O
    @test Osu() == Ou()+Ou()+O

    @test applicable(Ou,)
    @test !applicable(Ou,1)
    @test !applicable(Ot,)
    @test applicable(Ot,1)
end

@testset "Timed/LazySum (effective) Hamiltonian $(sectortype(pspace))" for (pspace, Dspace) in
                                                                         zip(
    pspaces, vspaces
)
    n = TensorMap(rand, ComplexF64, pspace, pspace)
    n += n'
    nn = TensorMap(rand, ComplexF64, pspace * pspace, pspace * pspace)
    nn += nn'
    nnn = TensorMap(rand, ComplexF64, pspace * pspace * pspace, pspace * pspace * pspace)
    nnn += nnn'

    # Os = map(
    #     (D1, D2) -> TensorMap(rand, ComplexF64, D1 * pspace, pspace * D2),
    #     [oneunit(HDspace), HDspace],
    #     [HDspace, oneunit(HDspace)],
    # )
    H = repeat(MPOHamiltonian(nn), 2)
    f(t) = 3 * exp(0.1 * t)
    Ht = TimedOperator(H, f)
    Hu = UntimedOperator(H,f(3.0))

    Ψs = [
        FiniteMPS(rand, ComplexF64, rand(3:2:20), pspace, Dspace),
        InfiniteMPS([
            TensorMap(rand, ComplexF64, Dspace * pspace, Dspace),
            TensorMap(rand, ComplexF64, Dspace * pspace, Dspace),
        ]),
    ]

    @testset "UntimedOperator $(Ψ isa FiniteMPS ? "F" : "Inf")initeMPS" for Ψ in Ψs
        envs = environments(Ψ, H)
        envs2 = environments(Ψ, Hu())
        envsu = environments(Ψ, Hu)

        @test envs.opp.data == envsu.opp.data #check that env are the same
        expresult = f(3.0) .* expectation_value(Ψ, H, envs)
        @test expresult ≈ expectation_value(Ψ, Hu(), envs2)
        @test expresult ≈ expectation_value(Ψ, Hu, envsu)
        @test expresult ≈ expectation_value(Ψ, Hu())
        @test expresult ≈ expectation_value(Ψ, Hu)

        ## derivatives
        hc = MPSKit.∂∂C(1, Ψ, H, envs)
        hcu = MPSKit.∂∂C(1, Ψ, Hu, envsu)

        @test hcu(Ψ.CR[1], 3.0) ≈ f(3.0) * hc(Ψ.CR[1]) atol = 1e-5

        hac = MPSKit.∂∂AC(1, Ψ, H, envs)
        hacu = MPSKit.∂∂AC(1, Ψ, Hu, envsu)

        @test hacu(Ψ.AC[1], 3.0) ≈ f(3.0) * hac(Ψ.AC[1]) atol = 1e-5

        hac2 = MPSKit.∂∂AC2(1, Ψ, H, envs)
        hac2u = MPSKit.∂∂AC2(1, Ψ, Hu, envsu)

        v = MPSKit._transpose_front(Ψ.AC[1]) * MPSKit._transpose_tail(Ψ.AR[2])

        @test hac2u(v, 3.0) ≈ f(3.0) * hac2(v) atol = 1e-5
    end

    @testset "TimedOperator $(Ψ isa FiniteMPS ? "F" : "Inf")initeMPS" for Ψ in Ψs
        envs = environments(Ψ, H)
        envs2 = environments(Ψ, Ht(3.0))
        envst = environments(Ψ, Ht)

        @test envs.opp.data == envst.opp.data #check that env are the same, time-dep sits elsewhere
        expresult = f(3.0) .* expectation_value(Ψ, H, envs)
        @test expresult ≈ expectation_value(Ψ, Ht(3.0), envst)
        @test expresult ≈ expectation_value(Ψ, Ht(3.0), envs2)
        @test expresult ≈ expectation_value(Ψ, Ht(3.0))

        ## time-dependence of derivatives
        hc = MPSKit.∂∂C(1, Ψ, H, envs)
        hct = MPSKit.∂∂C(1, Ψ, Ht, envst)

        @test hct(Ψ.CR[1], 3.0) ≈ f(3.0) * hc(Ψ.CR[1]) atol = 1e-5

        hac = MPSKit.∂∂AC(1, Ψ, H, envs)
        hact = MPSKit.∂∂AC(1, Ψ, Ht, envst)

        @test hact(Ψ.AC[1], 3.0) ≈ f(3.0) * hac(Ψ.AC[1]) atol = 1e-5

        hac2 = MPSKit.∂∂AC2(1, Ψ, H, envs)
        hac2t = MPSKit.∂∂AC2(1, Ψ, Ht, envst)

        v = MPSKit._transpose_front(Ψ.AC[1]) * MPSKit._transpose_tail(Ψ.AR[2])

        @test hac2t(v, 3.0) ≈ f(3.0) * hac2(v) atol = 1e-5
    end

    ##########################
    #tests for LazySum
    ##########################

    # Os = map(
    #     (D1, D2) -> TensorMap(rand, ComplexF64, D1 * pspace, pspace * D2),
    #     [oneunit(HDspace), HDspace, HDspace, HDspace],
    #     [HDspace, HDspace, HDspace, oneunit(HDspace)],
    # )

    fs = [t -> 3 + t, t -> 7 * t, t -> 2 * cos(t), t -> t^2]
    Hs = repeat.(map(i -> MPOHamiltonian(iseven(i) ? nn : nnn), 1:length(fs)), 2)
    summedH = LazySum(Hs, fs)

    @testset "Timed Sum $(Ψ isa FiniteMPS ? "F" : "Inf")initeMPS" for Ψ in Ψs
        Envs = map(H -> environments(Ψ, H), Hs)
        summedEnvs = environments(Ψ, summedH)
        summedEnvs2 = environments(Ψ, summedH(5.0))

        #@test envs.opp.data == summedEnvs.opp.data #check that env are the same, time-dep sits elsewhere

        expval = sum(zip(Hs, fs, Envs)) do (H, f, Env)
            f(5.0) * expectation_value(Ψ, H, Env)
        end
        expval1 = expectation_value(Ψ, summedH(5.0), summedEnvs)
        expval2 = expectation_value(Ψ, summedH(5.0))
        expval3 = expectation_value(Ψ, summedH(5.0), summedEnvs2)
        @test expval ≈ expval1
        @test expval ≈ expval2
        @test expval ≈ expval3

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

    # finally test in case LazySum contains non-timed operators
    fs = [3, 5.0, 10, 1]
    summedH = LazySum(Hs, fs)

    @testset "Untimed Sum $(Ψ isa FiniteMPS ? "F" : "Inf")initeMPS" for Ψ in Ψs
        Envs = map(H -> environments(Ψ, H), Hs)
        summedEnvs = environments(Ψ, summedH)

        expval = sum(zip(Hs, fs, Envs)) do (H, f, Env)
            f * expectation_value(Ψ, H, Env)
        end
        expval1 = expectation_value(Ψ, summedH())
        expval2 = expectation_value(Ψ, summedH, summedEnvs)
        expval3 = expectation_value(Ψ, summedH)
        @test expval ≈ expval1
        @test expval ≈ expval2
        @test expval ≈ expval3

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

    Hs      = [Hs[1],Hs[2],Hs[1]]
    fs      = [1,5,3.5*2]
    summedH = Hs[1] + UntimedOperator(Hs[2],5) + TimedOperator(Hs[3],t->3.5*t)

    @testset "Mixed Sum $(Ψ isa FiniteMPS ? "F" : "Inf")initeMPS" for Ψ in Ψs
        Envs = map(H -> environments(Ψ, H), summedH)
        summedEnvs = environments(Ψ, summedH)

        expval = sum(zip(Hs, fs, Envs)) do (H, f, Env)
            f * expectation_value(Ψ, H, Env)
        end
        expval1 = expectation_value(Ψ, summedH(2)())
        expval2 = expectation_value(Ψ, summedH(2), summedEnvs)
        expval3 = expectation_value(Ψ, summedH(2))
        @test expval ≈ expval1
        @test expval ≈ expval2
        @test expval ≈ expval3

        # test derivatives
        summedhct = MPSKit.∂∂C(1, Ψ, summedH, summedEnvs)
        sum1 = sum(zip(Hs, fs, Envs)) do (H, f, env)
            f * MPSKit.∂∂C(1, Ψ, H, env)(Ψ.CR[1])
        end
        @test summedhct(Ψ.CR[1], 2.0) ≈ sum1

        summedhct = MPSKit.∂∂AC(1, Ψ, summedH, summedEnvs)
        sum2 = sum(zip(Hs, fs, Envs)) do (H, f, env)
            f * MPSKit.∂∂AC(1, Ψ, H, env)(Ψ.AC[1])
        end
        @test summedhct(Ψ.AC[1], 2.0) ≈ sum2

        v = MPSKit._transpose_front(Ψ.AC[1]) * MPSKit._transpose_tail(Ψ.AR[2])
        summedhct = MPSKit.∂∂AC2(1, Ψ, summedH, summedEnvs)
        sum3 = sum(zip(Hs, fs, Envs)) do (H, f, env)
            f * MPSKit.∂∂AC2(1, Ψ, H, env)(v)
        end
        @test summedhct(v, 2.0) ≈ sum3
    end
end
