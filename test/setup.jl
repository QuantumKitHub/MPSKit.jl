# Planar stuff
# ----------------------------
module TestSetup

# imports
using MPSKit
using MPSKit: JordanMPOTensor
using TensorKit
using TensorKit: PlanarTrivial, ℙ, BraidingTensor
using BlockTensorKit
using LinearAlgebra: Diagonal
using Combinatorics: permutations

# exports
export S_xx, S_yy, S_zz, S_x, S_y, S_z
export c_plusmin, c_minplus, c_number
export force_planar
export symm_mul_mpo
export transverse_field_ising, heisenberg_XXX, bilinear_biquadratic_model, XY_model,
       kitaev_model
export classical_ising, finite_classical_ising, sixvertex

# using TensorOperations

force_planar(x::Number) = x

force_planar(c::Sector) = PlanarTrivial() ⊠ c

# convert spaces
force_planar(V::Union{CartesianSpace,ComplexSpace}) = ℙ^dim(V)
function force_planar(V::GradedSpace)
    return Vect[PlanarTrivial ⊠ sectortype(V)](force_planar(c) => dim(V, c)
                                               for c in sectors(V))
end
force_planar(V::SumSpace) = SumSpace(map(force_planar, V.spaces))
force_planar(V::ProductSpace) = ProductSpace(map(force_planar, V.spaces))
force_planar(V::HomSpace) = force_planar(codomain(V)) ← force_planar(domain(V))

# convert tensors
function force_planar(tsrc::AbstractTensorMap)
    V′ = force_planar(space(tsrc))
    tdst = similar(tsrc, V′)
    for (c, b) in blocks(tsrc)
        c′ = force_planar(c)
        copyto!(block(tdst, c′), b)
    end
    return tdst
end
function force_planar(tsrc::TensorMap)
    return TensorMap{eltype(tsrc)}(copy(tsrc.data), force_planar(space(tsrc)))
end
function force_planar(x::BraidingTensor)
    return BraidingTensor{scalartype(x)}(force_planar(space(x)))
end
function force_planar(x::BlockTensorMap)
    data = map(force_planar, x.data)
    return BlockTensorMap{eltype(data)}(data, force_planar(space(x)))
end
function force_planar(x::SparseBlockTensorMap)
    data = Dict(I => force_planar(v) for (I, v) in pairs(x.data))
    return SparseBlockTensorMap{valtype(data)}(data, force_planar(space(x)))
end
function force_planar(W::JordanMPOTensor)
    V = force_planar(space(W))
    TW = MPSKit.jordanmpotensortype(eltype(V[1]), scalartype(W))
    dst = TW(undef, V)

    for t in (:A, :B, :C, :D)
        for (I, v) in nonzero_pairs(getproperty(W, t))
            getproperty(dst, t)[I] = force_planar(v)
        end
    end
    return dst
end
force_planar(mpo::MPOHamiltonian) = MPOHamiltonian(map(force_planar, parent(mpo)))
force_planar(mpo::MPO) = MPO(map(force_planar, parent(mpo)))

# sum of all permutations: {Os...}
function symm_mul_mpo(Os::MPSKit.MPOTensor...)
    N! = factorial(length(Os))
    return sum(permutations(Os)) do os
        return foldl(MPSKit.fuse_mul_mpo, os)
    end / N!
end

# Toy models
# ----------------------------

function S_x(::Type{Trivial}=Trivial, ::Type{T}=ComplexF64; spin=1 // 2) where {T<:Number}
    return if spin == 1 // 2
        TensorMap(T[0 1; 1 0], ℂ^2 ← ℂ^2)
    elseif spin == 1
        TensorMap(T[0 1 0; 1 0 1; 0 1 0], ℂ^3 ← ℂ^3) / sqrt(2)
    else
        throw(ArgumentError("spin $spin not supported"))
    end
end
function S_y(::Type{Trivial}=Trivial, ::Type{T}=ComplexF64; spin=1 // 2) where {T<:Number}
    return if spin == 1 // 2
        TensorMap(T[0 -im; im 0], ℂ^2 ← ℂ^2)
    elseif spin == 1
        TensorMap(T[0 -im 0; im 0 -im; 0 im 0], ℂ^3 ← ℂ^3) / sqrt(2)
    else
        throw(ArgumentError("spin $spin not supported"))
    end
end
function S_z(::Type{Trivial}=Trivial, ::Type{T}=ComplexF64; spin=1 // 2) where {T<:Number}
    return if spin == 1 // 2
        TensorMap(T[1 0; 0 -1], ℂ^2 ← ℂ^2)
    elseif spin == 1
        TensorMap(T[1 0 0; 0 0 0; 0 0 -1], ℂ^3 ← ℂ^3)
    else
        throw(ArgumentError("spin $spin not supported"))
    end
end
function S_xx(::Type{Trivial}=Trivial, ::Type{T}=ComplexF64; spin=1 // 2) where {T<:Number}
    return S_x(Trivial, T; spin) ⊗ S_x(Trivial, T; spin)
end
function S_yy(::Type{Trivial}=Trivial, ::Type{T}=ComplexF64; spin=1 // 2) where {T<:Number}
    return S_y(Trivial, T; spin) ⊗ S_y(Trivial, T; spin)
end
function S_zz(::Type{Trivial}=Trivial, ::Type{T}=ComplexF64; spin=1 // 2) where {T<:Number}
    return S_z(Trivial, T; spin) ⊗ S_z(Trivial, T; spin)
end

function transverse_field_ising(; g=1.0, L=Inf)
    X = S_x(; spin=1 // 2)
    ZZ = S_zz(; spin=1 // 2)
    E = TensorMap(ComplexF64[1 0; 0 1], ℂ^2 ← ℂ^2)

    # lattice = L == Inf ? PeriodicVector([ℂ^2]) : fill(ℂ^2, L)
    if L == Inf
        lattice = PeriodicArray([ℂ^2])
        return InfiniteMPOHamiltonian(lattice,
                                      (i, i + 1) => -(ZZ + (g / 2) * (X ⊗ E + E ⊗ X))
                                      for i in 1:1)
        # return MPOHamiltonian(-ZZ - (g / 2) * (X ⊗ E + E ⊗ X))
    else
        lattice = fill(ℂ^2, L)
        return FiniteMPOHamiltonian(lattice,
                                    (i, i + 1) => -(ZZ + (g / 2) * (X ⊗ E + E ⊗ X))
                                    for i in 1:(L - 1)) #+
        # FiniteMPOHamiltonian(lattice, (i,) => -g * X for i in 1:L)
    end

    H = S_zz(; spin=1 // 2) + (g / 2) * (X ⊗ E + E ⊗ X)
    return if L == Inf
        MPOHamiltonian(H)
    else
        FiniteMPOHamiltonian(fill(ℂ^2, L), (i, i + 1) => H for i in 1:(L - 1))
    end
    return MPOHamiltonian(-H)
end

function heisenberg_XXX(::Type{SU2Irrep}; spin=1, L=Inf)
    h = ones(ComplexF64, SU2Space(spin => 1)^2 ← SU2Space(spin => 1)^2)
    for (c, b) in blocks(h)
        S = (dim(c) - 1) / 2
        b .= S * (S + 1) / 2 - spin * (spin + 1)
    end
    scale!(h, 4)

    if L == Inf
        lattice = PeriodicArray([space(h, 1)])
        return InfiniteMPOHamiltonian(lattice, (i, i + 1) => h for i in 1:1)
    else
        lattice = fill(space(h, 1), L)
        return FiniteMPOHamiltonian(lattice, (i, i + 1) => h for i in 1:(L - 1))
    end
end

function heisenberg_XXX(; spin=1, L=Inf)
    h = ones(ComplexF64, SU2Space(spin => 1)^2 ← SU2Space(spin => 1)^2)
    for (c, b) in blocks(h)
        S = (dim(c) - 1) / 2
        b .= S * (S + 1) / 2 - spin * (spin + 1)
    end
    A = convert(Array, h)
    d = convert(Int, 2 * spin + 1)
    h′ = TensorMap(A, (ℂ^d)^2 ← (ℂ^d)^2)

    if L == Inf
        lattice = PeriodicArray([space(h′, 1)])
        return InfiniteMPOHamiltonian(lattice, (i, i + 1) => h′ for i in 1:1)
    else
        lattice = fill(space(h′, 1), L)
        return FiniteMPOHamiltonian(lattice, (i, i + 1) => h′ for i in 1:(L - 1))
    end
end

function XY_model(::Type{U1Irrep}; g=1 / 2, L=Inf)
    h = zeros(ComplexF64,
              U1Space(-1 // 2 => 1, 1 // 2 => 1)^2 ← U1Space(-1 // 2 => 1, 1 // 2 => 1)^2)
    h[U1Irrep.((-1 // 2, 1 // 2, -1 // 2, 1 // 2))] .= 1
    h[U1Irrep.((1 // 2, -1 // 2, 1 // 2, -1 // 2))] .= 1
    Sz = zeros(ComplexF64, space(h, 1) ← space(h, 1))
    Sz[U1Irrep.((-1 // 2, 1 // 2))] .= -1 // 2
    Sz[U1Irrep.((1 // 2, -1 // 2))] .= 1 // 2
    if L == Inf
        lattice = PeriodicArray([space(h, 1)])
        H = InfiniteMPOHamiltonian(lattice, (i, i + 1) => -h for i in 1:1)
        iszero(g) && return H
        return H + g * InfiniteMPOHamiltonian(lattice, (i,) => -Sz for i in 1:1)
    else
        lattice = fill(space(h, 1), L)
        H = FiniteMPOHamiltonian(lattice, (i, i + 1) => -h for i in 1:(L - 1))
        iszero(g) && return H
        return H + g * FiniteMPOHamiltonian(lattice, (i,) => -Sz for i in 1:L)
    end
end

function bilinear_biquadratic_model(::Type{SU2Irrep}; θ=atan(1 / 3), L=Inf)
    h1 = ones(ComplexF64, SU2Space(1 => 1)^2 ← SU2Space(1 => 1)^2)
    for (c, b) in blocks(h1)
        S = (dim(c) - 1) / 2
        b .= S * (S + 1) / 2 - 1 * (1 + 1)
    end
    h2 = h1^2
    h = cos(θ) * h1 + sin(θ) * h2
    if L == Inf
        lattice = PeriodicArray([space(h2, 1)])
        return InfiniteMPOHamiltonian(lattice, (i, i + 1) => h for i in 1:1)
    else
        lattice = fill(space(h2, 1), L)
        return FiniteMPOHamiltonian(lattice, (i, i + 1) => h for i in 1:(L - 1))
    end
end

function c_plusmin()
    P = Vect[FermionParity](0 => 1, 1 => 1)
    t = zeros(ComplexF64, P^2 ← P^2)
    I = sectortype(P)
    t[(I(1), I(0), dual(I(0)), dual(I(1)))] .= 1
    return t
end

function c_minplus()
    P = Vect[FermionParity](0 => 1, 1 => 1)
    t = zeros(ComplexF64, P^2 ← P^2)
    I = sectortype(t)
    t[(I(0), I(1), dual(I(1)), dual(I(0)))] .= 1
    return t
end

function c_plusplus()
    P = Vect[FermionParity](0 => 1, 1 => 1)
    t = zeros(ComplexF64, P^2 ← P^2)
    I = sectortype(t)
    t[(I(1), I(1), dual(I(0)), dual(I(0)))] .= 1
    return t
end

function c_minmin()
    P = Vect[FermionParity](0 => 1, 1 => 1)
    t = zeros(ComplexF64, P^2 ← P^2)
    I = sectortype(t)
    t[(I(0), I(0), dual(I(1)), dual(I(1)))] .= 1
    return t
end

function c_number()
    P = Vect[FermionParity](0 => 1, 1 => 1)
    t = zeros(ComplexF64, P ← P)
    block(t, fℤ₂(1)) .= 1
    return t
end

function kitaev_model(; t=1.0, mu=1.0, Delta=1.0, L=Inf)
    TB = scale!(c_plusmin() + c_minplus(), -t / 2)     # tight-binding term
    SC = scale!(c_plusplus() + c_minmin(), Delta / 2)  # superconducting term
    CP = scale!(c_number(), -mu)                       # chemical potential term

    if L == Inf
        lattice = PeriodicArray([space(TB, 1)])
        return InfiniteMPOHamiltonian(lattice, (1, 2) => TB + SC, (1,) => CP)
    else
        lattice = fill(space(TB, 1), L)
        terms = Iterators.flatten((((i, i + 1) => TB + SC for i in 1:(L - 1)),
                                   (i,) => CP for i in 1:L))
        return FiniteMPOHamiltonian(lattice, terms)
    end
end

function ising_bond_tensor(β)
    t = [exp(β) exp(-β); exp(-β) exp(β)]
    r = eigen(t)
    nt = r.vectors * sqrt(Diagonal(r.values)) * r.vectors
    return nt
end

function classical_ising(; β=log(1 + sqrt(2)) / 2, L=Inf)
    nt = ising_bond_tensor(β)

    δbulk = zeros(ComplexF64, (2, 2, 2, 2))
    δbulk[1, 1, 1, 1] = 1
    δbulk[2, 2, 2, 2] = 1
    @tensor obulk[-1 -2; -3 -4] := δbulk[1 2; 3 4] * nt[-1; 1] * nt[-2; 2] * nt[-3; 3] *
                                   nt[-4; 4]
    Obulk = TensorMap(obulk, ℂ^2 * ℂ^2, ℂ^2 * ℂ^2)

    L == Inf && return InfiniteMPO([Obulk])

    δleft = zeros(ComplexF64, (1, 2, 2, 2))
    δleft[1, 1, 1, 1] = 1
    δleft[1, 2, 2, 2] = 1
    @tensor oleft[-1 -2; -3 -4] := δleft[-1 1; 2 3] * nt[-2; 1] * nt[-3; 2] * nt[-4; 3]
    Oleft = TensorMap(oleft, ℂ^1 * ℂ^2, ℂ^2 * ℂ^2)

    δright = zeros(ComplexF64, (2, 2, 2, 1))
    δright[1, 1, 1, 1] = 1
    δright[2, 2, 2, 1] = 1
    @tensor oright[-1 -2; -3 -4] := δright[1 2; 3 -4] * nt[-1; 1] * nt[-2; 2] * nt[-3; 3]
    Oright = TensorMap(oright, ℂ^2 * ℂ^2, ℂ^2 * ℂ^1)

    return FiniteMPO([Oleft, fill(Obulk, L - 2)..., Oright])
end

function finite_classical_ising(N)
    return classical_ising(; L=N)
    β = log(1 + sqrt(2)) / 2
    nt = ising_bond_tensor(β)

    # bulk
    O = zeros(ComplexF64, (2, 2, 2, 2))
    O[1, 1, 1, 1] = 1
    O[2, 2, 2, 2] = 1
    @tensor o[-1 -2; -3 -4] := O[1 2; 3 4] * nt[-1; 1] * nt[-2; 2] * nt[-3; 3] * nt[-4; 4]
    Obulk = TensorMap(o, ℂ^2 * ℂ^2, ℂ^2 * ℂ^2)

    # left
    OL = zeros(ComplexF64, (1, 2, 2, 2))
    OL[1, 1, 1, 1] = 1
    OL[1, 2, 2, 2] = 1
    @tensor oL[-1 -2; -3 -4] := OL[-1 1; 2 3] * nt[-2; 1] * nt[-3; 2] * nt[-4; 3]
    Oleft = TensorMap(oL, ℂ^1 * ℂ^2, ℂ^2 * ℂ^2)

    # right
    OR = zeros(ComplexF64, (2, 2, 2, 1))
    OR[1, 1, 1, 1] = 1
    OR[2, 2, 2, 1] = 1
    @tensor oR[-1 -2; -3 -4] := OR[1 2; 3 -4] * nt[-1; 1] * nt[-2; 2] * nt[-3; 3]
    Oright = TensorMap(oR, ℂ^2 * ℂ^2, ℂ^2 * ℂ^1)

    return DenseMPO([Oleft, fill(Obulk, N - 2)..., Oright])
end

function sixvertex(; a=1.0, b=1.0, c=1.0)
    d = ComplexF64[a 0 0 0
                   0 c b 0
                   0 b c 0
                   0 0 0 a]
    return InfiniteMPO([permute(TensorMap(d, ℂ^2 ⊗ ℂ^2, ℂ^2 ⊗ ℂ^2), ((1, 2), (4, 3)))])
end

end
