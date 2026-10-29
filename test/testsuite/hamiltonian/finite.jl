using ..TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ
using Adapt

export hamiltonian_finite_mpohamiltonian, hamiltonian_finite_canonicalization,
    hamiltonian_finite_repeated_indices

function hamiltonian_finite_mpohamiltonian(AT; fast = false)
    return @testset "Finite MPOHamiltonian ($AT)" begin
        L = 3
        T = ComplexF64
        mpoham_finite_Vs = fast ? (ℂ^2,) : (ℂ^2, U1Space(-1 => 1, 0 => 1, 1 => 1))
        Ts = fast ? (Float64,) : (Float64, ComplexF64)
        for (T, V) in zip(Ts, mpoham_finite_Vs)
            lattice = fill(V, L)
            O₁ = adapt(AT, randn(T, V, V))
            O₁ += O₁'
            E = id(storagetype(O₁), domain(O₁))
            O₂ = adapt(AT, randn(T, V^2 ← V^2))
            O₂ += O₂'

            H1 = FiniteMPOHamiltonian(lattice, i => O₁ for i in 1:L)
            H2 = FiniteMPOHamiltonian(lattice, (i, i + 1) => O₂ for i in 1:(L - 1))
            H3 = FiniteMPOHamiltonian(lattice, 1 => O₁, (2, 3) => O₂, (1, 3) => O₂)

            @test scalartype(H1) == scalartype(H2) == scalartype(H3) == T
            if !(T <: Complex)
                for H in (H1, H2, H3)
                    # `@constinferred` cannot be used inside a function; `@testinferred` is
                    # equivalent here since there are no constant arguments
                    Hc = @testinferred complex(H)
                    @test scalartype(Hc) == complex(T)
                    # cannot define `real(H)`, so only test elementwise
                    for (Wc, W) in zip(parent(Hc), parent(H))
                        Wr = @testinferred real(Wc)
                        @test scalartype(Wr) == T
                        @test Wr ≈ W atol = 1.0e-6
                    end
                end
            end

            # A one-site Hamiltonian must not be contracted with itself.
            Hsingle = FiniteMPOHamiltonian([V], 1 => O₁)
            @test convert(TensorMap, Hsingle) ≈ O₁

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
            H1_trunc = changebonds(H1, SvdCut(; trunc = truncrank(0)))
            @test H1_trunc ≈ H1
            @test all(left_virtualspace(H1_trunc) .== left_virtualspace(H1))

            # test dot and application
            state = adapt(AT, rand(T, prod(lattice)))
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

            H5 = changebonds(H4 / 3 + 2H4 / 3, SvdCut(; trunc = trunctol(; atol = 1.0e-12)))
            psi = adapt(AT, FiniteMPS(physicalspace(H5), V ⊕ rightunitspace(V)))
            @test expectation_value(psi, H4) ≈ expectation_value(psi, H5)
        end
    end
end

function hamiltonian_finite_canonicalization(AT; fast = false)
    return @testset "Finite MPOHamiltonian canonicalization ($AT)" begin
        T = ComplexF64
        for V in (ℂ^2, U1Space(-1 => 1, 0 => 1, 1 => 1), Rep[SU₂](1 // 2 => 1))
            O₁ = adapt(AT, randn(T, V, V))
            O₁ += O₁'
            O₂ = adapt(AT, randn(T, V^2 ← V^2))
            O₂ += O₂'
            E = id(storagetype(O₁), domain(O₁))

            # regression test for https://github.com/QuantumKitHub/MPSKit.jl/issues/522
            # bulk block of the middle site is empty on the left but not on the right
            H_right = FiniteMPOHamiltonian(fill(V, 3), (1,) => O₁, (2, 3) => O₂)
            H_right_tm = O₁ ⊗ E ⊗ E + E ⊗ O₂
            H, = MPSKit.right_canonicalize!(copy(H_right))
            @test convert(TensorMap, H) ≈ H_right_tm

            # bulk block of the middle site is empty on the right but not on the left
            H_left = FiniteMPOHamiltonian(fill(V, 3), (1, 2) => O₂, (3,) => O₁)
            H_left_tm = O₂ ⊗ E + E ⊗ E ⊗ O₁
            H, = MPSKit.left_canonicalize!(copy(H_left))
            @test convert(TensorMap, H) ≈ H_left_tm

            L = 5
            H5 = FiniteMPOHamiltonian(fill(V, L), (1, 2) => O₂, (3,) => O₁, (4, 5) => O₂)
            H5_tm = O₂ ⊗ E ⊗ E ⊗ E + E ⊗ E ⊗ O₁ ⊗ E ⊗ E + E ⊗ E ⊗ E ⊗ O₂
            H, = MPSKit.right_canonicalize!(copy(H5))
            @test convert(TensorMap, H) ≈ H5_tm
            H, = MPSKit.left_canonicalize!(H)
            @test convert(TensorMap, H) ≈ H5_tm

            H5_trunc = changebonds(H5, SvdCut(; trunc = truncrank(16)))
            @test convert(TensorMap, H5_trunc) ≈ H5_tm

            # canonical forms, also when `C`/`B` have identity components and the blocks are rank-deficient
            L = 4
            H = FiniteMPOHamiltonian(fill(V, L), (i, i + 1) => O₂ for i in 1:(L - 1))
            H_tm = convert(TensorMap, H)

            HL, Ls = MPSKit.left_canonicalize!(copy(H))
            @test length(Ls) == L - 1
            @test all(i -> codomain(Ls[i])[1] == right_virtualspace(HL[i].A), 1:(L - 1))
            @test convert(TensorMap, HL) ≈ H_tm
            for W in HL[2:(L - 1)]
                Vb = MPSKit.SparseBlockTensorMap(W)[1:(end - 1), 1, 1, 1:(end - 1)]
                @plansor G[r; r′] := conj(Vb[l p; p′ r]) * Vb[l p; p′ r′]
                G = TensorMap(G) / dim(physicalspace(W))
                @test G ≈ id(domain(G))
            end

            HR, Rs = MPSKit.right_canonicalize!(copy(H))
            @test length(Rs) == L - 1
            @test all(i -> domain(Rs[i])[1] == left_virtualspace(HR[i + 1].A), 1:(L - 1))
            @test convert(TensorMap, HR) ≈ H_tm
            for W in HR[2:(L - 1)]
                Vb = MPSKit.SparseBlockTensorMap(W)[2:end, 1, 1, 2:end]
                @plansor G[l; l′] := Vb[l p; p′ r] * conj(Vb[l′ p; p′ r])
                G = TensorMap(G) / dim(physicalspace(W))
                @test G ≈ id(codomain(G))
            end
        end
    end
end

function hamiltonian_finite_repeated_indices(AT; fast = false)
    return @testset "Finite MPOHamiltonian repeated indices ($AT)" begin
        X = adapt(AT, randn(ComplexF64, ℂ^2, ℂ^2))
        X += X'
        Y = adapt(AT, randn(ComplexF64, ℂ^2, ℂ^2))
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

        # a caller-supplied MPO is only copied when the indices actually need canonicalising,
        # so check that neither path writes back into it
        for inds in ((1, 2, 3), (1, 2, 2, 3), (2, 3, 1, 1))
            mpo = FiniteMPO(X ⊗ X ⊗ Y ⊗ Y)
            length(inds) == 3 && (mpo = FiniteMPO((X * X) ⊗ Y ⊗ Y))
            before = map(copy, parent(mpo))
            FiniteMPOHamiltonian(chain, inds => mpo)
            @test all(splat(≈), zip(parent(mpo), before))
        end
    end
end
