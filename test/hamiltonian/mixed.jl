println("
-----------------------------------------------
|   MPOHamiltonian tests (mixed/constructors) |
-----------------------------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using MPSKit: GeometryStyle, FiniteChainStyle, InfiniteChainStyle, OperatorStyle, HamiltonianStyle
using TensorKit
using TensorKit: ℙ
using Adapt

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
    @test TensorKit.storagetype(H) == Vector{T}
    @test TensorKit.storagetype(typeof(H)) == Vector{T}

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
    @test TensorKit.storagetype(H′) == Vector{T}
    @test TensorKit.storagetype(typeof(H′)) == Vector{T}
end

adapt_Vs = fast_tests ? (ℂ^2,) : (ℂ^2, U1Space(-1 => 1, 0 => 1, 1 => 1))
Ts = fast_tests ? (Float64,) : (Float64, ComplexF64)
@testset "Adapt" for V in adapt_Vs
    h = rand(Float32, V^2 ← V^2)
    h += h'

    L = 4
    H1 = FiniteMPOHamiltonian(
        fill(V, L),
        ((i, i + 1) => h for i in 1:(L - 1))...,
        ((i, i + 2) => h for i in 1:(L - 2))...,
        ((i, i + 3) => h for i in 1:(L - 3))...,
    )
    mps1 = FiniteMPS(physicalspace(H1), oneunit(V))

    for T in Ts
        H2 = if VERSION <= v"1.12"
            adapt(Vector{T}, H1)
        else
            @testinferred adapt(Vector{T}, H1)
        end
        @test H2 isa FiniteMPOHamiltonian
        @test scalartype(H2) == T
        @test storagetype(H2) == Vector{T}
        @test expectation_value(mps1, H1) ≈ expectation_value(mps1, H2)
    end

    H3 = InfiniteMPOHamiltonian(fill(V, L), (1, 2) => h, (1, 3) => h, (1, 4) => h)
    mps2 = InfiniteMPS(physicalspace(H3), [oneunit(V)])
    for T in Ts
        H4 = if VERSION <= v"1.12"
            # this is type unstable for LTS for some reason
            adapt(Vector{T}, H3)
        else
            @testinferred adapt(Vector{T}, H3)
        end
        @test H4 isa InfiniteMPOHamiltonian
        @test scalartype(H4) == T
        @test storagetype(H4) == Vector{T}
        @test storagetype(typeof(H4)) == Vector{T}
        @test expectation_value(mps2, H3) ≈ expectation_value(mps2, H4)
    end
end

@testset "MPOHamiltonian channel sharing" begin
    # Terms that start out identically share a virtual channel, which is what keeps the bond
    # dimension of a long-range interaction linear -- instead of quadratic -- in its range.
    # Correctness is checked against the same Hamiltonian assembled by summing single-term
    # Hamiltonians, which cannot share anything and is thus independent of how channels are
    # assigned.
    Z = TensorMap(Float64[1 0; 0 -1], ℂ^2, ℂ^2)
    X = TensorMap(Float64[0 1; 1 0], ℂ^2, ℂ^2)

    infinite_ev(lattice, terms) = let L = length(lattice)
        ψ = InfiniteMPS(collect(lattice), fill(ℂ^8, L))
        H = InfiniteMPOHamiltonian(lattice, terms...)
        Href = sum(InfiniteMPOHamiltonian(lattice, t) for t in terms)
        (H, sum(expectation_value(ψ, H)), sum(expectation_value(ψ, Href)))
    end

    @testset "power-law couplings, unit cell L=$L" for L in (1, 2, 3)
        R = 5
        lattice = fill(ℂ^2, L)
        terms = Pair[(i, i + d) => (1 / d^3) * (Z ⊗ Z) for i in 1:L for d in 1:R]
        append!(terms, [(i,) => 0.7 * X for i in 1:L])
        H, e, eref = infinite_ev(lattice, terms)
        @test e ≈ eref
        @test all(i -> dim(left_virtualspace(H, i)) ≤ R + 2, 1:L)
    end

    @testset "proportional terms are merged" begin
        lattice = fill(ℂ^2, 1)
        H = InfiniteMPOHamiltonian(lattice, (1, 2) => Z ⊗ Z, (1, 2) => 0.5 * (Z ⊗ Z))
        Href = InfiniteMPOHamiltonian(lattice, (1, 2) => 1.5 * (Z ⊗ Z))
        ψ = InfiniteMPS([ℂ^2], [ℂ^8])
        @test sum(expectation_value(ψ, H)) ≈ sum(expectation_value(ψ, Href))
        @test dim(left_virtualspace(H, 1)) == dim(left_virtualspace(Href, 1))
    end

    @testset "negative prefactors" begin
        lattice = fill(ℂ^2, 1)
        terms = Pair[
            (1, 2) => Z ⊗ Z, (1, 3) => -2.0 * (Z ⊗ Z), (1, 4) => 0.25 * (Z ⊗ Z),
        ]
        _, e, eref = infinite_ev(lattice, terms)
        @test e ≈ eref
    end

    @testset "mixed operators and ranges" begin
        lattice = fill(ℂ^2, 2)
        terms = Pair[
            (1, 2) => Z ⊗ Z, (1, 3) => Z ⊗ X, (1, 4) => 3.0 * (Z ⊗ Z),
            (2, 3, 4) => X ⊗ Z ⊗ X, (1,) => 0.7 * X,
        ]
        _, e, eref = infinite_ev(lattice, terms)
        @test e ≈ eref
    end

    @testset "finite chain" begin
        L = 6
        lattice = fill(ℂ^2, L)
        terms = Pair[
            (i, j) => (1 / (j - i)^3) * (Z ⊗ Z) for i in 1:L for j in (i + 1):L
        ]
        push!(terms, (1,) => 0.7 * X)
        H = FiniteMPOHamiltonian(lattice, terms...)
        Href = sum(FiniteMPOHamiltonian(lattice, t) for t in terms)
        @test convert(TensorMap, H) ≈ convert(TensorMap, Href)
        @test maximum(i -> dim(left_virtualspace(H, i)), 1:L) <
            maximum(i -> dim(left_virtualspace(Href, i)), 1:L)
    end

    @testset "symmetric tensors $(sectortype(pspace))" for (pspace, Dspace) in
        zip(pspaces, vspaces)
        O = rand(ComplexF64, pspace^2, pspace^2)
        O += O'
        terms = Pair[(1, 1 + d) => (1 / d^3) * O for d in 1:4]
        lattice = fill(pspace, 1)
        H = InfiniteMPOHamiltonian(lattice, terms...)
        Href = sum(InfiniteMPOHamiltonian(lattice, t) for t in terms)
        ψ = InfiniteMPS([pspace], [Dspace])
        @test sum(expectation_value(ψ, H)) ≈ sum(expectation_value(ψ, Href))
        @test dim(left_virtualspace(H, 1)) < dim(left_virtualspace(Href, 1))
    end
end
