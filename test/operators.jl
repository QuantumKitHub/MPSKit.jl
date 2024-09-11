println("
-----------------
|   Operators   |
-----------------
")
module TestOperators

using ..TestSetup
using Test, TestExtras
using MPSKit
using MPSKit: _transpose_front, _transpose_tail
using TensorKit
using TensorKit: ℙ
using VectorInterface: One

pspaces = (ℙ^4, Rep[U₁](0 => 2), Rep[SU₂](1 => 1))
vspaces = (ℙ^10, Rep[U₁]((0 => 20)), Rep[SU₂](1 // 2 => 10, 3 // 2 => 5, 5 // 2 => 1))

@testset "FiniteMPO" begin
    # start from random operators
    L = 4
    O₁ = rand(ComplexF64, (ℂ^2)^L, (ℂ^2)^L)
    O₂ = rand(ComplexF64, space(O₁))

    # create MPO and convert it back to see if it is the same
    mpo₁ = FiniteMPO(O₁) # type-unstable for now!
    mpo₂ = FiniteMPO(O₂)
    @test convert(TensorMap, mpo₁) ≈ O₁
    @test convert(TensorMap, -mpo₂) ≈ -O₂

    # test scalar multiplication
    α = rand(ComplexF64)
    @test convert(TensorMap, α * mpo₁) ≈ α * O₁
    @test convert(TensorMap, mpo₁ * α) ≈ O₁ * α

    # test addition and multiplication
    @test convert(TensorMap, mpo₁ + mpo₂) ≈ O₁ + O₂
    @test convert(TensorMap, mpo₁ * mpo₂) ≈ O₁ * O₂

    # test application to a state
    ψ₁ = rand(ComplexF64, domain(O₁))
    mps₁ = FiniteMPS(ψ₁)

    @test convert(TensorMap, mpo₁ * mps₁) ≈ O₁ * ψ₁

    @test dot(mps₁, mpo₁, mps₁) ≈ dot(ψ₁, O₁, ψ₁)
    @test dot(mps₁, mpo₁, mps₁) ≈ dot(mps₁, mpo₁ * mps₁)
    # test conversion to and from mps
    mpomps₁ = convert(FiniteMPS, mpo₁)
    mpompsmpo₁ = convert(FiniteMPO, mpomps₁)

    @test convert(FiniteMPO, mpomps₁) ≈ mpo₁ rtol = 1e-6

    @test dot(mpomps₁, mpomps₁) ≈ dot(mpo₁, mpo₁)
end

@testset "Finite MPOHamiltonian" begin
    L = 3
    lattice = fill(ℂ^2, L)
    O₁ = rand(ComplexF64, ℂ^2, ℂ^2)
    E = id(Matrix{ComplexF64}, domain(O₁))
    O₂ = rand(ComplexF64, ℂ^2 * ℂ^2, ℂ^2 * ℂ^2)

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

    # test linear algebra
    @test H1 ≈
          FiniteMPOHamiltonian(lattice, 1 => O₁) + FiniteMPOHamiltonian(lattice, 2 => O₁) +
          FiniteMPOHamiltonian(lattice, 3 => O₁)
    @test 0.8 * H1 + 0.2 * H1 ≈ H1 atol = 1e-6
    @test convert(TensorMap, H1 + H2) ≈ convert(TensorMap, H1) + convert(TensorMap, H2) atol = 1e-6

    # test dot and application
    state = rand(ComplexF64, prod(lattice))
    mps = FiniteMPS(state)

    @test convert(TensorMap, H1 * mps) ≈ H1_tm * state
    @test dot(mps, H2, mps) ≈ dot(mps, H2 * mps)

    # test constructor from dictionary with mixed linear and Cartesian lattice indices as keys
    grid = square = fill(ℂ^2, 3, 3)

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
end

@testset "InfiniteMPOHamiltonian $(sectortype(pspace))" for (pspace, Dspace) in
                                                            zip(pspaces,
                                                                vspaces)
    # generate a 1-2-3 body interaction
    operators = ntuple(3) do i
        O = rand(ComplexF64, pspace^i, pspace^i)
        return O += O'
    end

    H1 = InfiniteMPOHamiltonian(operators[1])
    H2 = InfiniteMPOHamiltonian(operators[2])
    H3 = repeat(InfiniteMPOHamiltonian(operators[3]), 2)

    # can you pass in a proper mpo?
    # TODO: fix this!
    # identity = complex(isomorphism(oneunit(pspace) * pspace, pspace * oneunit(pspace)))
    # mpoified = MPSKit.decompose_localmpo(MPSKit.add_util_leg(nnn))
    # d3 = Array{Union{Missing,typeof(identity)},3}(missing, 1, 4, 4)
    # d3[1, 1, 1] = identity
    # d3[1, end, end] = identity
    # d3[1, 1, 2] = mpoified[1]
    # d3[1, 2, 3] = mpoified[2]
    # d3[1, 3, 4] = mpoified[3]
    # h1 = MPOHamiltonian(d3)

    # make a teststate to measure expectation values for
    ψ1 = InfiniteMPS([pspace], [Dspace])
    ψ2 = InfiniteMPS([pspace, pspace], [Dspace, Dspace])

    e1 = expectation_value(ψ1, H1)
    e2 = expectation_value(ψ1, H2)

    H1 = 2 * H1 - [1]
    @test e1 * 2 - 1 ≈ expectation_value(ψ1, H1) atol = 1e-10

    H1 = H1 + H2

    @test e1 * 2 + e2 - 1 ≈ expectation_value(ψ1, H1) atol = 1e-10

    H1 = repeat(H1, 2)

    e1 = expectation_value(ψ2, H1)
    e3 = expectation_value(ψ2, H3)

    @test e1 + e3 ≈ expectation_value(ψ2, H1 + H3) atol = 1e-10

    H4 = H1 + H3
    h4 = H4 * H4
    @test real(expectation_value(ψ2, H4)) >= 0
end

@testset "General LazySum of $(eltype(Os))" for Os in (rand(ComplexF64, rand(1:10)),
                                                       map(i -> rand(ComplexF64,
                                                                     ℂ^13, ℂ^7),
                                                           1:rand(1:10)),
                                                       map(i -> rand(ComplexF64,
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

@testset "MultipliedOperator of $(typeof(O)) with $(typeof(f))" for (O, f) in
                                                                    zip((rand(ComplexF64),
                                                                         rand(ComplexF64,
                                                                              ℂ^13, ℂ^7),
                                                                         rand(ComplexF64,
                                                                              ℂ^1 ⊗ ℂ^2,
                                                                              ℂ^3 ⊗ ℂ^4)),
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
                                                                      fill(rand(ComplexF64,
                                                                                ℂ^13, ℂ^7),
                                                                           4),
                                                                      fill(rand(ComplexF64,
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
    pspace = physicalspace(ham, 1)
    ou = oneunit(pspace)

    ψ = InfiniteMPS([pspace], [ou ⊕ pspace])

    W = DenseMPO(make_time_mpo(ham, 1im * 0.5, WII()))

    @test W * (W * ψ) ≈ (W * W) * ψ atol = 1e-3
end

pspaces = (ℙ^4, Rep[U₁](0 => 2), Rep[SU₂](1 => 1, 2 => 1))
vspaces = (ℙ^10, Rep[U₁]((0 => 20)), Rep[SU₂](1 => 10, 3 => 5, 5 => 1))

@testset "LazySum of (effective) Hamiltonian $(sectortype(pspace))" for (pspace, Dspace) in
                                                                        zip(pspaces,
                                                                            vspaces)
    # n = rand(ComplexF64, pspace, pspace)
    # n += n'
    # nn = rand(ComplexF64, pspace * pspace, pspace * pspace)
    # nn += nn'
    # nnn = rand(ComplexF64, pspace * pspace * pspace, pspace * pspace * pspace)
    # nnn += nnn'
    Os = map(1:3) do i
        O = rand(ComplexF64, pspace^i, pspace^i)
        return O += O'
    end

    fs = [t -> 3t, 2, 1]


    # H1 = repeat(MPOHamiltonian(n), 2)
    # H2 = repeat(MPOHamiltonian(nn), 2)
    # H3 = repeat(MPOHamiltonian(nnn), 2)
    # Hs = [H1, H2, H3]
    # summedH = LazySum(Hs)

    @testset "LazySum FiniteMPOHamiltonian" begin
        L = rand(3:2:20)
        ψ = FiniteMPS(rand, ComplexF64, L, pspace, Dspace)
        lattice = fill(pspace, L)
        Hs = map(FiniteMPOHamiltonian, Os)
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
            return MPSKit.∂∂C(1, ψ, H, env)(ψ.CR[1])
        end
        @test summedhct(ψ.CR[1], 0.0) ≈ sum1

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
            return f * MPSKit.∂∂C(1, ψ, H, env)(ψ.CR[1])
        end
        @test summedhct(ψ.CR[1], t) ≈ sum1

        summedhct = MPSKit.∂∂AC(1, ψ, summedH, summed_envs)
        sum2 = sum(zip(fs, Hs, Envs)) do (f, H, env)
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
    end

    @testset "LazySum InfiniteMPOHamiltonian" begin
        ψ = repeat(InfiniteMPS(pspace, Dspace), 2)
        Hs = map(Os) do O
            H = InfiniteMPOHamiltonian(O)
            return repeat(H, 2)
        end
        envs = map(H -> environments(ψ, H), Hs)
        summed_envs = environments(ψ, summedH)

        expval = sum(zip(Hs, envs)) do (H, Env)
            return expectation_value(ψ, H, Env)
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
            return MPSKit.∂∂C(1, ψ, H, env)(ψ.CR[1])
        end
        @test summedhct(ψ.CR[1], 0.0) ≈ sum1

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
            return f * MPSKit.∂∂C(1, ψ, H, env)(ψ.CR[1])
        end
        @test summedhct(ψ.CR[1], t) ≈ sum1

        summedhct = MPSKit.∂∂AC(1, ψ, summedH, summed_envs)
        sum2 = sum(zip(fs, Hs, Envs)) do (f, H, env)
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
    end
end

end
