println("
----------------------------
|   MPOHamiltonian tests    |
----------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using MPSKit: GeometryStyle, FiniteChainStyle, InfiniteChainStyle, OperatorStyle, HamiltonianStyle
using TensorKit
using TensorKit: ℙ
using Adapt

pspaces = (ℙ^4, Rep[U₁](0 => 2), Rep[SU₂](1 => 1))
vspaces = (ℙ^10, Rep[U₁]((0 => 20)), Rep[SU₂](1 // 2 => 10, 3 // 2 => 5, 5 // 2 => 1))

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

    @test isfinite(H)
    @test isfinite(typeof(H))
    @test GeometryStyle(typeof(H)) == FiniteChainStyle()
    @test GeometryStyle(H) == FiniteChainStyle()
    @test OperatorStyle(typeof(H)) == HamiltonianStyle()
    @test OperatorStyle(H) == HamiltonianStyle()
    @test OperatorStyle(H, H′) == OperatorStyle(H)

    # Infinite
    Ws = [Wmid]
    H = InfiniteMPOHamiltonian(
        fill(P, 1), [(i,) => H1 for i in 1:1]..., [(i, i + 1) => H2 for i in 1:1]...
    )
    H′ = InfiniteMPOHamiltonian(Ws)
    @test all(parent(H) .≈ parent(H′))

    H′ = InfiniteMPOHamiltonian(map(Base.Fix1(collect, Any), Ws)) # without type info
    @test all(parent(H) .≈ parent(H′))

    @test !isfinite(H)
    @test !isfinite(typeof(H))
    @test GeometryStyle(typeof(H)) == InfiniteChainStyle()
    @test GeometryStyle(H) == InfiniteChainStyle()
    @test OperatorStyle(typeof(H)) == HamiltonianStyle()
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
        psi = FiniteMPS(physicalspace(H5), V ⊕ rightunitspace(V))
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
