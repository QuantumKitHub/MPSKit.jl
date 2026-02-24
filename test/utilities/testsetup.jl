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
using TensorKitTensors.SpinOperators: S_x, S_y, S_z, S_x_S_x, S_y_S_y, S_z_S_z, S_exchange, S_plus_S_min, S_min_S_plus
using TensorKitTensors.FermionOperators: f_plus_f_min, f_min_f_plus, f_plus_f_plus, f_min_f_min, f_num

# exports
export S_x, S_y, S_z
export S_x_S_x, S_y_S_y, S_z_S_z
export f_plus_f_min, f_min_f_plus, f_num
export force_planar
export symm_mul_mpo
export transverse_field_ising, heisenberg_XXX, bilinear_biquadratic_model, XY_model,
    kitaev_model
export classical_ising_tensors, classical_ising, sixvertex

# using TensorOperations

force_planar(x::Number) = x

force_planar(c::Sector) = PlanarTrivial() ⊠ c

# convert spaces
force_planar(V::Union{CartesianSpace, ComplexSpace}) = ℙ^dim(V)
function force_planar(V::GradedSpace)
    return Vect[PlanarTrivial ⊠ sectortype(V)](force_planar(c) => dim(V, c) for c in sectors(V))
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

function transverse_field_ising(
        T::Type{<:Number} = ComplexF64, sym::Type{<:Sector} = Trivial;
        g = 1.0, L = Inf
    )
    X = S_x(T, sym; spin = 1 // 2) * 2
    ZZ = S_z_S_z(T, sym; spin = 1 // 2) * 4

    if L == Inf
        lattice = PeriodicArray([space(X, 1)])
        H₁ = InfiniteMPOHamiltonian(lattice, i => -g * X for i in 1:1)
        H₂ = InfiniteMPOHamiltonian(lattice, (i, i + 1) => -ZZ for i in 1:1)
    else
        lattice = fill(space(X, 1), L)
        H₁ = FiniteMPOHamiltonian(lattice, i => -g * X for i in 1:L)
        H₂ = FiniteMPOHamiltonian(lattice, (i, i + 1) => -ZZ for i in 1:(L - 1))
    end
    return H₁ + H₂
end

function heisenberg_XXX(
        T::Type{<:Number} = ComplexF64, sym::Type{<:Sector} = Trivial;
        spin = 1, L = Inf
    )
    h = S_exchange(ComplexF64, sym; spin)

    if L == Inf
        lattice = PeriodicArray([space(h, 1)])
        return InfiniteMPOHamiltonian(lattice, (i, i + 1) => h for i in 1:1)
    else
        lattice = fill(space(h, 1), L)
        return FiniteMPOHamiltonian(lattice, (i, i + 1) => h for i in 1:(L - 1))
    end
end

function XY_model(
        T::Type{<:Number} = ComplexF64, sym::Type{<:Sector} = Trivial;
        g = 1 / 2, L = Inf
    )
    spin = 1 // 2
    h = S_plus_S_min(T, sym; spin) + S_min_S_plus(T, sym; spin)
    Z = S_z(T, sym; spin)

    if L == Inf
        lattice = PeriodicArray([space(h, 1)])
        H = InfiniteMPOHamiltonian(lattice, (i, i + 1) => -h for i in 1:1)
        iszero(g) && return H
        return H + g * InfiniteMPOHamiltonian(lattice, (i,) => -Z for i in 1:1)
    else
        lattice = fill(space(h, 1), L)
        H = FiniteMPOHamiltonian(lattice, (i, i + 1) => -h for i in 1:(L - 1))
        iszero(g) && return H
        return H + g * FiniteMPOHamiltonian(lattice, (i,) => -Z for i in 1:L)
    end
end

function bilinear_biquadratic_model(
        T::Type{<:Number} = ComplexF64, sym::Type{<:Sector} = Trivial;
        θ = atan(1 / 3), L = Inf
    )
    h1 = S_exchange(T, sym; spin = 1)
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

function kitaev_model(
        T::Type{<:Number} = ComplexF64, sym::Type{<:Sector} = Trivial;
        t = 1.0, mu = 1.0, Delta = 1.0, L = Inf
    )
    TB = scale!(f_plus_f_min(T, sym) + f_min_f_plus(T, sym), -t / 2)     # tight-binding term
    SC = scale!(f_plus_f_plus(T, sym) + f_min_f_min(T, sym), Delta / 2)  # superconducting term
    CP = scale!(f_num(T, sym), -mu)                       # chemical potential term

    if L == Inf
        lattice = PeriodicArray([space(TB, 1)])
        return InfiniteMPOHamiltonian(lattice, (1, 2) => TB + SC, (1,) => CP)
    else
        lattice = fill(space(TB, 1), L)
        onsite_terms = ((i,) => CP for i in 1:L)
        twosite_terms = ((i, i + 1) => TB + SC for i in 1:(L - 1))
        terms = Iterators.flatten(twosite_terms, onsite_terms)
        return FiniteMPOHamiltonian(lattice, terms)
    end
end

function ising_bond_tensor(β)
    J = 1.0
    K = β * J

    # Boltzmann weights
    t = ComplexF64[exp(K) exp(-K); exp(-K) exp(K)]
    r = eigen(t)
    nt = r.vectors * sqrt(Diagonal(r.values)) * r.vectors
    return nt
end

function classical_ising_tensors(beta)
    nt = ising_bond_tensor(beta)
    J = 1.0

    # local partition function tensor
    δbulk = zeros(ComplexF64, (2, 2, 2, 2))
    δbulk[1, 1, 1, 1] = 1
    δbulk[2, 2, 2, 2] = 1
    @tensor obulk[-1 -2; -3 -4] := δbulk[1 2; 3 4] * nt[-1; 1] * nt[-2; 2] * nt[-3; 3] *
        nt[-4; 4]

    # magnetization tensor
    M = copy(δbulk)
    M[2, 2, 2, 2] *= -1
    @tensor m[-1 -2; -3 -4] := M[1 2; 3 4] * nt[-1; 1] * nt[-2; 2] * nt[-3; 3] * nt[-4; 4]

    # bond interaction tensor and energy-per-site tensor
    e = ComplexF64[-J J; J -J] .* nt
    @tensor e_hor[-1 -2; -3 -4] :=
        δbulk[1 2; 3 4] * nt[-1; 1] * nt[-2; 2] * nt[-3; 3] * e[-4; 4]
    @tensor e_vert[-1 -2; -3 -4] :=
        δbulk[1 2; 3 4] * nt[-1; 1] * nt[-2; 2] * e[-3; 3] * nt[-4; 4]
    e = e_hor + e_vert

    # fixed tensor map space for all three
    TMS = ℂ^2 ⊗ ℂ^2 ← ℂ^2 ⊗ ℂ^2

    return TensorMap(obulk, TMS), TensorMap(m, TMS), TensorMap(e, TMS)
end;

function classical_ising(; β = log(1 + sqrt(2)) / 2, L = Inf)
    Obulk, _, _ = classical_ising_tensors(β)

    L == Inf && return InfiniteMPO([Obulk])

    nt = ising_bond_tensor(β)
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

function sixvertex(; a = 1.0, b = 1.0, c = 1.0)
    d = ComplexF64[
        a 0 0 0
        0 c b 0
        0 b c 0
        0 0 0 a
    ]
    return InfiniteMPO([permute(TensorMap(d, ℂ^2 ⊗ ℂ^2, ℂ^2 ⊗ ℂ^2), ((1, 2), (4, 3)))])
end

end
