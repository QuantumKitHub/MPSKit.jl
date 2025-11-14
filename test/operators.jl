println("
-----------------
|   Operators   |
-----------------
")
module TestOperators

    using ..TestSetup
    using Test, TestExtras
    using MPSKit
    using MPSKit: _transpose_front, _transpose_tail, C_hamiltonian, AC_hamiltonian,
        AC2_hamiltonian
    using MPSKit: GeometryStyle, FiniteStyle, InfiniteStyle, OperatorStyle, MPOStyle,
        HamiltonianStyle
    using TensorKit
    using TensorKit: ℙ
    using VectorInterface: One

    pspaces = (ℙ^4, Rep[U₁](0 => 2), Rep[SU₂](1 => 1))
    vspaces = (ℙ^10, Rep[U₁]((0 => 20)), Rep[SU₂](1 // 2 => 10, 3 // 2 => 5, 5 // 2 => 1))

    @testset "FiniteMPO" begin
        # start from random operators
        L = 4
        T = ComplexF64

        for V in (ℂ^2, U1Space(0 => 1, 1 => 1))
            O₁ = rand(T, V^L, V^L)
            O₂ = rand(T, space(O₁))
            O₃ = rand(real(T), space(O₁))

            # create MPO and convert it back to see if it is the same
            mpo₁ = FiniteMPO(O₁) # type-unstable for now!
            mpo₂ = FiniteMPO(O₂)
            mpo₃ = FiniteMPO(O₃)

            @test GeometryStyle(mpo₁) == FiniteStyle()
            @test OperatorStyle(mpo₁) == MPOStyle()


            @test @constinferred physicalspace(mpo₁) == fill(V, L)
            Vleft = @constinferred left_virtualspace(mpo₁)
            Vright = @constinferred right_virtualspace(mpo₂)
            for i in 1:L
                @test Vleft[i] == left_virtualspace(mpo₁, i)
                @test Vright[i] == right_virtualspace(mpo₁, i)
            end

            @test convert(TensorMap, mpo₁) ≈ O₁
            @test convert(TensorMap, -mpo₂) ≈ -O₂
            @test convert(TensorMap, @constinferred complex(mpo₃)) ≈ complex(O₃)

            # test scalar multiplication
            α = rand(T)
            @test convert(TensorMap, α * mpo₁) ≈ α * O₁
            @test convert(TensorMap, mpo₁ * α) ≈ O₁ * α
            @test α * mpo₃ ≈ α * complex(mpo₃) atol = 1.0e-6

            # test addition and multiplication
            @test convert(TensorMap, mpo₁ + mpo₂) ≈ O₁ + O₂
            @test convert(TensorMap, mpo₁ + mpo₃) ≈ O₁ + O₃
            @test convert(TensorMap, mpo₁ * mpo₂) ≈ O₁ * O₂
            @test convert(TensorMap, mpo₁ * mpo₃) ≈ O₁ * O₃

            # test application to a state
            ψ₁ = rand(T, domain(O₁))
            ψ₂ = rand(real(T), domain(O₂))
            mps₁ = FiniteMPS(ψ₁)
            mps₂ = FiniteMPS(ψ₂)

            @test convert(TensorMap, mpo₁ * mps₁) ≈ O₁ * ψ₁
            @test mpo₁ * ψ₁ ≈ O₁ * ψ₁
            @test convert(TensorMap, mpo₃ * mps₁) ≈ O₃ * ψ₁
            @test mpo₃ * ψ₁ ≈ O₃ * ψ₁
            @test convert(TensorMap, mpo₁ * mps₂) ≈ O₁ * ψ₂
            @test mpo₁ * ψ₂ ≈ O₁ * ψ₂

            @test dot(mps₁, mpo₁, mps₁) ≈ dot(ψ₁, O₁, ψ₁)
            @test dot(mps₁, mpo₁, mps₁) ≈ dot(mps₁, mpo₁ * mps₁)
            # test conversion to and from mps
            mpomps₁ = convert(FiniteMPS, mpo₁)
            mpompsmpo₁ = convert(FiniteMPO, mpomps₁)

            @test convert(FiniteMPO, mpomps₁) ≈ mpo₁ rtol = 1.0e-6

            @test dot(mpomps₁, mpomps₁) ≈ dot(mpo₁, mpo₁)
        end
    end

    @testset "InfiniteMPO" begin
        P = ℂ^2
        T = Float64

        H1 = randn(T, P ← P)
        H1 += H1'
        H = InfiniteMPO([H1])

        @test GeometryStyle(H) == InfiniteStyle()
        @test OperatorStyle(H) == MPOStyle()
    end

    @testset "MPOHamiltonian constructors" begin
        P = ℂ^2
        T = Float64

        H1 = randn(T, P ← P)
        H1 += H1'
        D = FiniteMPO(H1)[1]

        H2 = randn(T, P^2 ← P^2)
        H2 += H2'
        C, B = FiniteMPO(H2)[1:2]

        Elt = Union{Missing, typeof(D), scalartype(D)}
        Wmid = Elt[1.0 C D; 0.0 0.0 B; 0.0 0.0 1.0]
        Wleft = Wmid[1:1, :]
        Wright = Wmid[:, end:end]

        # Finite
        Ws = [Wleft, Wmid, Wmid, Wright]
        H = FiniteMPOHamiltonian(
            fill(P, 4), [(i,) => H1 for i in 1:4]..., [(i, i + 1) => H2 for i in 1:3]...
        )
        H′ = FiniteMPOHamiltonian(Ws)
        @test H ≈ H′

        H′ = FiniteMPOHamiltonian(map(Base.Fix1(collect, Any), Ws)) # without type info
        @test H ≈ H′

        @test GeometryStyle(H) == FiniteStyle()
        @test OperatorStyle(H) == HamiltonianStyle()

        # Infinite
        Ws = [Wmid]
        H = InfiniteMPOHamiltonian(
            fill(P, 1), [(i,) => H1 for i in 1:1]..., [(i, i + 1) => H2 for i in 1:1]...
        )
        H′ = InfiniteMPOHamiltonian(Ws)
        @test all(parent(H) .≈ parent(H′))

        H′ = InfiniteMPOHamiltonian(map(Base.Fix1(collect, Any), Ws)) # without type info
        @test all(parent(H) .≈ parent(H′))

        @test GeometryStyle(H) == InfiniteStyle()
        @test OperatorStyle(H) == HamiltonianStyle()
    end

    @testset "Finite MPOHamiltonian" begin
        L = 3
        T = ComplexF64
        for T in (Float64, ComplexF64), V in (ℂ^2, U1Space(-1 => 1, 0 => 1, 1 => 1))
            lattice = fill(V, L)
            O₁ = randn(T, V, V)
            O₁ += O₁'
            E = id(storagetype(O₁), domain(O₁))
            O₂ = randn(T, V^2 ← V^2)
            O₂ += O₂'

            H1 = FiniteMPOHamiltonian(lattice, i => O₁ for i in 1:L)
            H2 = FiniteMPOHamiltonian(lattice, (i, i + 1) => O₂ for i in 1:(L - 1))
            H3 = FiniteMPOHamiltonian(lattice, 1 => O₁, (2, 3) => O₂, (1, 3) => O₂)

            @test scalartype(H1) == scalartype(H2) == scalartype(H3) == T
            if !(T <: Complex)
                for H in (H1, H2, H3)
                    Hc = @constinferred complex(H)
                    @test scalartype(Hc) == complex(T)
                    # cannot define `real(H)`, so only test elementwise
                    for (Wc, W) in zip(parent(Hc), parent(H))
                        Wr = @constinferred real(Wc)
                        @test scalartype(Wr) == T
                        @test Wr ≈ W atol = 1.0e-6
                    end
                end
            end

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
            @test H4 ≈ H1 atol = 1.0e-6
            double_terms = Iterators.flatten(
                Iterators.repeated(((i, i + 1) => O₂ / 2 for i in 1:(L - 1)), 2)
            )
            H5 = FiniteMPOHamiltonian(lattice, double_terms)
            @test H5 ≈ H2 atol = 1.0e-6

            # test linear algebra
            @test H1 ≈
                FiniteMPOHamiltonian(lattice, 1 => O₁) +
                FiniteMPOHamiltonian(lattice, 2 => O₁) +
                FiniteMPOHamiltonian(lattice, 3 => O₁)
            @test 0.8 * H1 + 0.2 * H1 ≈ H1 atol = 1.0e-6
            @test convert(TensorMap, H1 + H2) ≈ convert(TensorMap, H1) + convert(TensorMap, H2) atol = 1.0e-6
            H1_trunc = changebonds(H1, SvdCut(; trscheme = truncrank(0)))
            @test H1_trunc ≈ H1
            @test all(left_virtualspace(H1_trunc) .== left_virtualspace(H1))

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
            vertical_operators = Dict(
                (I, I + I_vertical) => O₂ for I in eachindex(IndexCartesian(), square) if I[1] < size(square, 1)
            )
            I_horizontal = CartesianIndex(0, 1)
            horizontal_operators = Dict(
                (I, I + I_horizontal) => O₂ for I in eachindex(IndexCartesian(), square) if I[2] < size(square, 1)
            )
            operators = merge(local_operators, vertical_operators, horizontal_operators)
            H4 = FiniteMPOHamiltonian(grid, operators)

            @test H4 ≈
                FiniteMPOHamiltonian(grid, local_operators) +
                FiniteMPOHamiltonian(grid, vertical_operators) +
                FiniteMPOHamiltonian(grid, horizontal_operators) atol = 1.0e-4

            H5 = changebonds(H4 / 3 + 2H4 / 3, SvdCut(; trscheme = trunctol(; atol = 1.0e-12)))
            psi = FiniteMPS(physicalspace(H5), V ⊕ oneunit(V))
            @test expectation_value(psi, H4) ≈ expectation_value(psi, H5)
        end
    end

    @testset "Finite MPOHamiltonian repeated indices" begin
        X = randn(ComplexF64, ℂ^2, ℂ^2)
        X += X'
        Y = randn(ComplexF64, ℂ^2, ℂ^2)
        Y += Y'
        L = 4
        chain = fill(space(X, 1), 4)

        H1 = FiniteMPOHamiltonian(chain, (1,) => (X * X * Y * Y))
        H2 = FiniteMPOHamiltonian(chain, (1, 1, 1, 1) => (X ⊗ X ⊗ Y ⊗ Y))
        @test convert(TensorMap, H1) ≈ convert(TensorMap, H2)

        H1 = FiniteMPOHamiltonian(chain, (1, 2) => ((X * Y) ⊗ (X * Y)))
        H2 = FiniteMPOHamiltonian(chain, (1, 2, 1, 2) => (X ⊗ X ⊗ Y ⊗ Y))
        @test convert(TensorMap, H1) ≈ convert(TensorMap, H2)

        H1 = FiniteMPOHamiltonian(chain, (1, 2) => ((X * X * Y) ⊗ Y))
        H2 = FiniteMPOHamiltonian(chain, (1, 1, 1, 2) => (X ⊗ X ⊗ Y ⊗ Y))
        @test convert(TensorMap, H1) ≈ convert(TensorMap, H2)

        H1 = FiniteMPOHamiltonian(chain, (1, 2) => ((X * Y * Y) ⊗ X))
        H2 = FiniteMPOHamiltonian(chain, (1, 2, 1, 1) => (X ⊗ X ⊗ Y ⊗ Y))
        @test convert(TensorMap, H1) ≈ convert(TensorMap, H2)

        H1 = FiniteMPOHamiltonian(chain, (1, 2, 3) => FiniteMPO((X * X) ⊗ Y ⊗ Y))
        H2 = FiniteMPOHamiltonian(chain, (1, 1, 2, 3) => FiniteMPO(X ⊗ X ⊗ Y ⊗ Y))
        @test convert(TensorMap, H1) ≈ convert(TensorMap, H2)

        H1 = FiniteMPOHamiltonian(chain, (1, 2, 3) => FiniteMPO((Y * Y) ⊗ X ⊗ X))
        H2 = FiniteMPOHamiltonian(chain, (2, 3, 1, 1) => FiniteMPO(X ⊗ X ⊗ Y ⊗ Y))
        @test convert(TensorMap, H1) ≈ convert(TensorMap, H2)

        H1 = FiniteMPOHamiltonian(chain, (1, 2, 3) => FiniteMPO(X ⊗ (X * Y) ⊗ Y))
        H2 = FiniteMPOHamiltonian(chain, (1, 2, 2, 3) => FiniteMPO(X ⊗ X ⊗ Y ⊗ Y))
        @test convert(TensorMap, H1) ≈ convert(TensorMap, H2)
    end

    @testset "InfiniteMPOHamiltonian $(sectortype(pspace))" for (pspace, Dspace) in zip(pspaces, vspaces)
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
        @test e1 * 2 - 1 ≈ expectation_value(ψ1, H1) atol = 1.0e-10

        H1 = H1 + H2

        @test e1 * 2 + e2 - 1 ≈ expectation_value(ψ1, H1) atol = 1.0e-10

        H1 = repeat(H1, 2)

        e1 = expectation_value(ψ2, H1)
        e3 = expectation_value(ψ2, H3)

        @test e1 + e3 ≈ expectation_value(ψ2, H1 + H3) atol = 1.0e-10

        H4 = H1 + H3
        h4 = H4 * H4
        @test real(expectation_value(ψ2, H4)) >= 0
    end

    @testset "General LazySum of $(eltype(Os))" for Os in (
            rand(ComplexF64, rand(1:10)),
            map(i -> rand(ComplexF64, ℂ^13, ℂ^7), 1:rand(1:10)),
            map(i -> rand(ComplexF64, ℂ^1 ⊗ ℂ^2, ℂ^3 ⊗ ℂ^4), 1:rand(1:10)),
        )
        LazyOs = LazySum(Os)

        #test user interface
        summed = sum(Os)

        @test sum(LazyOs) ≈ summed atol = 1 - 08

        LazyOs_added = +(LazyOs, Os...)

        @test sum(LazyOs_added) ≈ 2 * summed atol = 1 - 08
    end

    @testset "MultipliedOperator of $(typeof(O)) with $(typeof(f))" for (O, f) in
        zip(
            (rand(ComplexF64), rand(ComplexF64, ℂ^13, ℂ^7), rand(ComplexF64, ℂ^1 ⊗ ℂ^2, ℂ^3 ⊗ ℂ^4)),
            (t -> 3t, 1.1, One())
        )
        tmp = MPSKit.MultipliedOperator(O, f)
        if tmp isa TimedOperator
            @test tmp(1.1)() ≈ f(1.1) * O atol = 1 - 08
        elseif tmp isa UntimedOperator
            @test tmp() ≈ f * O atol = 1 - 08
        end
    end
    @testset "General Time-dependent LazySum of $(eltype(Os))" for Os in (
            rand(ComplexF64, 4),
            fill(rand(ComplexF64, ℂ^13, ℂ^7), 4),
            fill(rand(ComplexF64, ℂ^1 ⊗ ℂ^2, ℂ^3 ⊗ ℂ^4), 4),
        )

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

    @testset "DenseMPO" for ham in (transverse_field_ising(), heisenberg_XXX(; spin = 1))
        pspace = physicalspace(ham, 1)
        ou = oneunit(pspace)

        ψ = InfiniteMPS([pspace], [ou ⊕ pspace])

        W = MPSKit.DenseMPO(make_time_mpo(ham, 1im * 0.5, WII()))
        @test W * (W * ψ) ≈ (W * W) * ψ atol = 1.0e-2 # TODO: there is a normalization issue here
    end

    pspaces = (ℙ^4, Rep[U₁](0 => 2), Rep[SU₂](1 => 1, 2 => 1))
    vspaces = (ℙ^10, Rep[U₁]((0 => 20)), Rep[SU₂](1 => 10, 3 => 5, 5 => 1))

    @testset "LazySum of (effective) Hamiltonian $(sectortype(pspace))" for (pspace, Dspace) in
        zip(pspaces, vspaces)
        Os = map(1:3) do i
            O = rand(ComplexF64, pspace^i, pspace^i)
            return O += O'
        end
        fs = [t -> 3t, 2, 1]

        @testset "LazySum FiniteMPOHamiltonian" begin
            L = rand(3:2:20)
            ψ = FiniteMPS(rand, ComplexF64, L, pspace, Dspace)
            lattice = fill(pspace, L)
            Hs = map(enumerate(Os)) do (i, O)
                return FiniteMPOHamiltonian(
                    lattice,
                    ntuple(x -> x + j, i) => O for j in 0:(L - i)
                )
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
            summedhct = C_hamiltonian(1, ψ, summedH, ψ, summed_envs)
            sum1 = sum(zip(Hs, envs)) do (H, env)
                return C_hamiltonian(1, ψ, H, ψ, env)(ψ.C[1])
            end
            @test summedhct(ψ.C[1], 0.0) ≈ sum1

            summedhct = AC_hamiltonian(1, ψ, summedH, ψ, summed_envs)
            sum2 = sum(zip(Hs, envs)) do (H, env)
                return AC_hamiltonian(1, ψ, H, ψ, env)(ψ.AC[1])
            end
            @test summedhct(ψ.AC[1], 0.0) ≈ sum2

            v = _transpose_front(ψ.AC[1]) * _transpose_tail(ψ.AR[2])
            summedhct = AC2_hamiltonian(1, ψ, summedH, ψ, summed_envs)
            sum3 = sum(zip(Hs, envs)) do (H, env)
                return AC2_hamiltonian(1, ψ, H, ψ, env)(v)
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
            summedhct = C_hamiltonian(1, ψ, summedH, ψ, summed_envs)
            sum1 = sum(zip(fs, Hs, envs)) do (f, H, env)
                if f isa Function
                    f = f(t)
                end
                return f * C_hamiltonian(1, ψ, H, ψ, env)(ψ.C[1])
            end
            @test summedhct(ψ.C[1], t) ≈ sum1

            summedhct = AC_hamiltonian(1, ψ, summedH, ψ, summed_envs)
            sum2 = sum(zip(fs, Hs, envs)) do (f, H, env)
                if f isa Function
                    f = f(t)
                end
                return f * AC_hamiltonian(1, ψ, H, ψ, env)(ψ.AC[1])
            end
            @test summedhct(ψ.AC[1], t) ≈ sum2

            v = _transpose_front(ψ.AC[1]) * _transpose_tail(ψ.AR[2])
            summedhct = AC2_hamiltonian(1, ψ, summedH, ψ, summed_envs)
            sum3 = sum(zip(fs, Hs, envs)) do (f, H, env)
                return (f isa Function ? f(t) : f) * AC2_hamiltonian(1, ψ, H, ψ, env)(v)
            end
            @test summedhct(v, t) ≈ sum3
        end

        @testset "LazySum InfiniteMPOHamiltonian" begin
            ψ = repeat(InfiniteMPS(pspace, Dspace), 2)
            Hs = map(Os) do O
                H = InfiniteMPOHamiltonian(O)
                return repeat(H, 2)
            end
            summedH = LazySum(Hs)
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
            summedhct = C_hamiltonian(1, ψ, summedH, ψ, summed_envs)
            sum1 = sum(zip(Hs, envs)) do (H, env)
                return C_hamiltonian(1, ψ, H, ψ, env)(ψ.C[1])
            end
            @test summedhct(ψ.C[1], 0.0) ≈ sum1

            summedhct = AC_hamiltonian(1, ψ, summedH, ψ, summed_envs)
            sum2 = sum(zip(Hs, envs)) do (H, env)
                return AC_hamiltonian(1, ψ, H, ψ, env)(ψ.AC[1])
            end
            @test summedhct(ψ.AC[1], 0.0) ≈ sum2

            v = _transpose_front(ψ.AC[1]) * _transpose_tail(ψ.AR[2])
            summedhct = AC2_hamiltonian(1, ψ, summedH, ψ, summed_envs)
            sum3 = sum(zip(Hs, envs)) do (H, env)
                return AC2_hamiltonian(1, ψ, H, ψ, env)(v)
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
            summedhct = C_hamiltonian(1, ψ, summedH, ψ, summed_envs)
            sum1 = sum(zip(fs, Hs, envs)) do (f, H, env)
                if f isa Function
                    f = f(t)
                end
                return f * C_hamiltonian(1, ψ, H, ψ, env)(ψ.C[1])
            end
            @test summedhct(ψ.C[1], t) ≈ sum1

            summedhct = AC_hamiltonian(1, ψ, summedH, ψ, summed_envs)
            sum2 = sum(zip(fs, Hs, envs)) do (f, H, env)
                if f isa Function
                    f = f(t)
                end
                return f * AC_hamiltonian(1, ψ, H, ψ, env)(ψ.AC[1])
            end
            @test summedhct(ψ.AC[1], t) ≈ sum2

            v = _transpose_front(ψ.AC[1]) * _transpose_tail(ψ.AR[2])
            summedhct = AC2_hamiltonian(1, ψ, summedH, ψ, summed_envs)
            sum3 = sum(zip(fs, Hs, envs)) do (f, H, env)
                return (f isa Function ? f(t) : f) * AC2_hamiltonian(1, ψ, H, ψ, env)(v)
            end
            @test summedhct(v, t) ≈ sum3
        end
    end

    @testset "ProjectionOperator" begin
        L = 10
        psi = FiniteMPS(rand, ComplexF64, L, ℙ^2, ℙ^2)
        psi2 = FiniteMPS(rand, ComplexF64, L, ℙ^2, ℙ^2)
        O = MPSKit.ProjectionOperator(psi)

        @test expectation_value(psi, O) ≈ 1.0
        @test expectation_value(psi2, O) ≈ dot(psi, psi2) * dot(psi2, psi)
    end

    @testset "MPO copy behaviour" begin
        # testset that checks the fix for issue #288
        H = transverse_field_ising()
        O = make_time_mpo(H, 0.1, TaylorCluster(2, true, true))
        FO = open_boundary_conditions(O, 4)
        FH = open_boundary_conditions(H, 4)

        # check if the copy of the MPO is the same type as the original
        @test typeof(copy(O)) == typeof(O)
        @test typeof(copy(FO)) == typeof(FO)
        @test typeof(copy(H)) == typeof(H)
        @test typeof(copy(FH)) == typeof(FH)
    end

end
