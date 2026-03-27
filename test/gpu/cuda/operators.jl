using .TestSetup
using Test, TestExtras
using MPSKit
using MPSKit: GeometryStyle, FiniteChainStyle, InfiniteChainStyle, OperatorStyle, MPOStyle
using TensorKit
using MatrixAlgebraKit
using TensorKit: ℙ, tensormaptype, TensorMapWithStorage
using Adapt, CUDA, cuTENSOR, CUDA.CUBLAS

@testset "CuFiniteMPO" for V in (ℂ^2, U1Space(0 => 1, 1 => 1))
    # start from random operators
    L = 4
    T = ComplexF64

    O₁ = rand(T, V^L, V^L)
    O₂ = rand(T, space(O₁))
    O₃ = rand(real(T), space(O₁))

    dO₁ = adapt(CuArray, O₁)
    dO₂ = adapt(CuArray, O₂)
    dO₃ = adapt(CuArray, O₃)

    mpo₁ = adapt(CuVector{T, CUDA.DeviceMemory}, FiniteMPO(O₁))
    mpo₂ = adapt(CuVector{T, CUDA.DeviceMemory}, FiniteMPO(O₂))
    mpo₃ = adapt(CuVector{T, CUDA.DeviceMemory}, FiniteMPO(O₃))

    @test isfinite(mpo₁)
    @test isfinite(typeof(mpo₁))
    @test GeometryStyle(typeof(mpo₁)) == FiniteChainStyle()
    @test GeometryStyle(mpo₁) == FiniteChainStyle()
    @test OperatorStyle(typeof(mpo₁)) == MPOStyle()
    @test TensorKit.storagetype(mpo₁) == CuVector{T, CUDA.DeviceMemory}
    @test TensorKit.storagetype(mpo₂) == CuVector{T, CUDA.DeviceMemory}
    @test TensorKit.storagetype(mpo₃) == CuVector{T, CUDA.DeviceMemory}

    @test @constinferred physicalspace(mpo₁) == fill(V, L)
    Vleft = @constinferred left_virtualspace(mpo₁)
    Vright = @constinferred right_virtualspace(mpo₂)
    for i in 1:L
        @test Vleft[i] == left_virtualspace(mpo₁, i)
        @test Vright[i] == right_virtualspace(mpo₁, i)
    end

    TM = TensorMap
    @test convert(TM, mpo₁) ≈ dO₁
    @test convert(TM, -mpo₂) ≈ -dO₂
    @test convert(TM, @constinferred complex(mpo₃)) ≈ complex(dO₃)

    # test scalar multiplication
    α = rand(T)
    @test convert(TM, α * mpo₁) ≈ α * dO₁
    @test convert(TM, mpo₁ * α) ≈ dO₁ * α
    @test α * mpo₃ ≈ α * complex(mpo₃) atol = 1.0e-6

    # test addition and multiplication
    @test convert(TM, mpo₁ + mpo₂) ≈ dO₁ + dO₂
    @test convert(TM, mpo₁ + mpo₃) ≈ dO₁ + dO₃
    @test convert(TM, mpo₁ * mpo₂) ≈ dO₁ * dO₂
    @test convert(TM, mpo₁ * mpo₃) ≈ dO₁ * dO₃

    # test application to a state
    ψ₁ = adapt(CuArray, rand(T, domain(O₁)))
    #ψ₂ = adapt(CuArray, rand(real(T), domain(O₂))) # not allowed due to cuTENSOR
    mps₁ = adapt(CuArray, FiniteMPS(ψ₁))
    #mps₂ = adapt(CuArray, FiniteMPS(ψ₂))

    @test @constinferred GeometryStyle(mps₁, mpo₁, mps₁) == GeometryStyle(mps₁)

    @test convert(TM, mpo₁ * mps₁) ≈ dO₁ * ψ₁
    @test mpo₁ * ψ₁ ≈ dO₁ * ψ₁
    @test convert(TM, mpo₃ * mps₁) ≈ dO₃ * ψ₁
    @test mpo₃ * ψ₁ ≈ dO₃ * ψ₁
    #@test convert(TM, mpo₁ * mps₂) ≈ dO₁ * ψ₂
    #@test mpo₁ * ψ₂ ≈ dO₁ * ψ₂

    @test dot(mps₁, mpo₁, mps₁) ≈ dot(ψ₁, dO₁, ψ₁)
    @test dot(mps₁, mpo₁, mps₁) ≈ dot(mps₁, mpo₁ * mps₁)
    # test conversion to and from mps
    mpomps₁ = convert(FiniteMPS, mpo₁)
    mpompsmpo₁ = convert(FiniteMPO, mpomps₁)

    @test convert(FiniteMPO, mpomps₁) ≈ mpo₁ rtol = 1.0e-6

    @test dot(mpomps₁, mpomps₁) ≈ dot(mpo₁, mpo₁)
end

@testset "Finite CuMPOHamiltonian" begin
    L = 3
    T = ComplexF64
    for T in (Float64, ComplexF64), V in (ℂ^2, U1Space(-1 => 1, 0 => 1, 1 => 1))
        lattice = fill(V, L)
        O₁ = randn(T, V, V)
        O₁ += O₁'
        E = id(storagetype(O₁), domain(O₁))
        O₂ = randn(T, V^2 ← V^2)
        O₂ += O₂'

        H1 = adapt(CuVector{T, CUDA.DeviceMemory}, FiniteMPOHamiltonian(lattice, i => O₁ for i in 1:L))
        H2 = adapt(CuVector{T, CUDA.DeviceMemory}, FiniteMPOHamiltonian(lattice, (i, i + 1) => O₂ for i in 1:(L - 1)))
        H3 = adapt(CuVector{T, CUDA.DeviceMemory}, FiniteMPOHamiltonian(lattice, 1 => O₁, (2, 3) => O₂, (1, 3) => O₂))
        @test TensorKit.storagetype(H1) == CuVector{T, CUDA.DeviceMemory}
        @test TensorKit.storagetype(H2) == CuVector{T, CUDA.DeviceMemory}
        @test TensorKit.storagetype(H3) == CuVector{T, CUDA.DeviceMemory}

        @test scalartype(H1) == scalartype(H2) == scalartype(H3) == T
        if !(T <: Complex)
            for H in (H1, H2, H3)
                Hc = @constinferred complex(H)
                @test scalartype(Hc) == complex(T)
                @test TensorKit.storagetype(Hc) == CuVector{complex(T), CUDA.DeviceMemory}
            end
        end

        # check if constructor works by converting back to tensormap
        #= H1_tm = convert(TensorMap, H1)
        operators = vcat(fill(E, L - 1), O₁)
        @test H1_tm ≈ mapreduce(+, 1:L) do i
            return reduce(⊗, circshift(operators, i))
        end
        operators = vcat(fill(E, L - 2), O₂)
        @test convert(TensorMap, H2) ≈ mapreduce(+, 1:(L - 1)) do i
            return reduce(⊗, circshift(operators, i))
        end
        @test convert(TensorMap, H3) ≈
            O₁ ⊗ E ⊗ E + E ⊗ O₂ + permute(O₂ ⊗ E, ((1, 3, 2), (4, 6, 5))) =# # needs a fix in BlockTensorKit

        # check if adding terms on the same site works
        single_terms = Iterators.flatten(Iterators.repeated((i => O₁ / 2 for i in 1:L), 2))
        H4 = adapt(CuArray, FiniteMPOHamiltonian(lattice, single_terms))
        @test TensorKit.storagetype(H4) == CuVector{T, CUDA.DeviceMemory}
        @test H4 ≈ H1 atol = 1.0e-6
        double_terms = Iterators.flatten(
            Iterators.repeated(((i, i + 1) => O₂ / 2 for i in 1:(L - 1)), 2)
        )
        H5 = adapt(CuArray, FiniteMPOHamiltonian(lattice, double_terms))
        @test TensorKit.storagetype(H5) == CuVector{T, CUDA.DeviceMemory}
        @test H5 ≈ H2 atol = 1.0e-6

        # test linear algebra
        @test H1 ≈
            adapt(CuArray, FiniteMPOHamiltonian(lattice, 1 => O₁)) +
            adapt(CuArray, FiniteMPOHamiltonian(lattice, 2 => O₁)) +
            adapt(CuArray, FiniteMPOHamiltonian(lattice, 3 => O₁))
        @test TensorKit.storagetype(H1) == CuVector{T, CUDA.DeviceMemory}
        #@test 0.8 * H1 + 0.2 * H1 ≈ H1 atol = 1.0e-6 # broken due to JordanMPOTensorMap
        #=@test convert(TensorMap, H1 + H2) ≈ convert(TensorMap, H1) + convert(TensorMap, H2) atol = 1.0e-6
        H1_trunc = changebonds(H1, SvdCut(; trscheme = truncrank(0)))
        @test H1_trunc ≈ H1
        @test all(left_virtualspace(H1_trunc) .== left_virtualspace(H1))=# # needs fix in BlockTensorKit

        # test dot and application
        state = rand(T, prod(lattice))
        mps = adapt(CuArray, FiniteMPS(state))

        #=@test convert(TensorMap, H1 * mps) ≈ H1_tm * state # needs fix in BlockTensorKit
        @test H1 * state ≈ H1_tm * state
        @test dot(mps, H2, mps) ≈ dot(mps, H2 * mps)=#

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
        H4 = adapt(CuArray, FiniteMPOHamiltonian(grid, operators))
        @test TensorKit.storagetype(H4) == CuVector{T, CUDA.DeviceMemory}

        @test H4 ≈
            adapt(CuArray, FiniteMPOHamiltonian(grid, local_operators)) +
            adapt(CuArray, FiniteMPOHamiltonian(grid, vertical_operators)) +
            adapt(CuArray, FiniteMPOHamiltonian(grid, horizontal_operators)) atol = 1.0e-4
        @test TensorKit.storagetype(H4) == CuVector{T, CUDA.DeviceMemory}

        #H4′= H4 / 3 + 2H4 / 3
        #@test TensorKit.storagetype(H4′) == CuVector{T, CUDA.DeviceMemory}
        #H5 = changebonds(H4′, SvdCut(; trscheme = trunctol(; atol = 1.0e-12)))
        #@test TensorKit.storagetype(H5) == CuVector{T, CUDA.DeviceMemory} # more problems with arithmetic operations...
        #psi = adapt(CuArray, FiniteMPS(physicalspace(H5), V ⊕ rightunitspace(V)))
        #@test expectation_value(psi, H4) ≈ expectation_value(psi, H5)
    end
end

@testset "CuInfiniteMPOHamiltonian $(sectortype(pspace))" for (pspace, Dspace) in zip(pspaces, vspaces)
    # generate a 1-2-3 body interaction
    operators = ntuple(3) do i
        O = rand(ComplexF64, pspace^i, pspace^i)
        return O += O'
    end

    H1 = adapt(CuVector{ComplexF64, CUDA.DeviceMemory}, InfiniteMPOHamiltonian(operators[1]))
    H2 = adapt(CuVector{ComplexF64, CUDA.DeviceMemory}, InfiniteMPOHamiltonian(operators[2]))
    H3 = adapt(CuVector{ComplexF64, CUDA.DeviceMemory}, repeat(InfiniteMPOHamiltonian(operators[3]), 2))

    @test TensorKit.storagetype(H1) == CuVector{ComplexF64, CUDA.DeviceMemory}
    @test TensorKit.storagetype(H2) == CuVector{ComplexF64, CUDA.DeviceMemory}
    @test TensorKit.storagetype(H3) == CuVector{ComplexF64, CUDA.DeviceMemory}
    # make a teststate to measure expectation values for
    ψ1 = adapt(CuVector{ComplexF64, CUDA.DeviceMemory}, InfiniteMPS([pspace], [Dspace]))
    ψ2 = adapt(CuVector{ComplexF64, CUDA.DeviceMemory}, InfiniteMPS([pspace, pspace], [Dspace, Dspace]))
    @test TensorKit.storagetype(ψ1) == CuVector{ComplexF64, CUDA.DeviceMemory}
    @test TensorKit.storagetype(ψ2) == CuVector{ComplexF64, CUDA.DeviceMemory}

    #=
    e1 = expectation_value(ψ1, H1)
    e2 = expectation_value(ψ1, H2)
    =# # broken due to BraidingTensor

    # H1 = 2 * H1 - CuArray([1]) # scalar indexing
    # @test TensorKit.storagetype(H1) == CuVector{ComplexF64, CUDA.DeviceMemory}
    # @test e1 * 2 - 1 ≈ expectation_value(ψ1, H1) atol = 1.0e-10 # broken due to BraidingTensor

    H1 = H1 + H2
    @test TensorKit.storagetype(H1) == CuVector{ComplexF64, CUDA.DeviceMemory}

    # @test e1 * 2 + e2 - 1 ≈ expectation_value(ψ1, H1) atol = 1.0e-10 # broken due to BraidingTensor

    H1 = repeat(H1, 2)
    @test TensorKit.storagetype(H1) == CuVector{ComplexF64, CUDA.DeviceMemory}

    #=e1 = expectation_value(ψ2, H1)
    e3 = expectation_value(ψ2, H3)

    @test e1 + e3 ≈ expectation_value(ψ2, H1 + H3) atol = 1.0e-10=# # broken due to BraidingTensor

    #H4 = H1 + H3 # broken due to BraidingTensor
    #@test TensorKit.storagetype(H4) == CuVector{ComplexF64, CUDA.DeviceMemory}
    #h4 = H4 * H4
    #@test TensorKit.storagetype(h4) == CuVector{ComplexF64, CUDA.DeviceMemory}
    #@test real(expectation_value(ψ2, H4)) >= 0 # broken due to BraidingTensor
end
