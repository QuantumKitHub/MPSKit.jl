# Planar stuff
# ----------------------------
module TestSetup

# imports
using MPSKit
using TensorKit
using TensorKit: PlanarTrivial, ℙ
using LinearAlgebra: Diagonal

# exports
export S_xx, S_yy, S_zz, S_x, S_y, S_z
export force_planar
export transverse_field_ising, heisenberg_XXX, bilinear_biquadratic_model
export classical_ising, finite_classical_ising, sixvertex

# using TensorOperations

force_planar(x::Number) = x
function force_planar(x::AbstractTensorMap)
    cod = reduce(*, map(i -> ℙ^dim(space(x, i)), codomainind(x)))
    dom = reduce(*, map(i -> ℙ^dim(space(x, i)), domainind(x)))
    t = TensorMap(zeros, scalartype(x), cod ← dom)
    copyto!(blocks(t)[PlanarTrivial()], convert(Array, x))
    return t
end
function force_planar(mpo::MPOHamiltonian)
    L = mpo.period
    V = mpo.odim
    return MPOHamiltonian(map(Iterators.product(1:L, 1:V, 1:V)) do (i, j, k)
                              return force_planar(mpo.Os[i, j, k])
                          end)
end
force_planar(mpo::DenseMPO) = DenseMPO(force_planar.(mpo.opp))

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

function transverse_field_ising(; g=1.0)
    X = S_x(; spin=1 // 2)
    E = TensorMap(ComplexF64[1 0; 0 1], ℂ^2 ← ℂ^2)
    H = S_zz(; spin=1 // 2) + (g / 2) * (X ⊗ E + E ⊗ X)
    return MPOHamiltonian(-H)
end

function heisenberg_XXX(::Type{SU2Irrep}; spin=1)
    H = TensorMap(ones, ComplexF64, SU2Space(spin => 1)^2 ← SU2Space(spin => 1)^2)
    for (c, b) in blocks(H)
        S = (dim(c) - 1) / 2
        b .= S * (S + 1) / 2 - spin * (spin + 1)
    end
    return MPOHamiltonian(H * 4)
end

function heisenberg_XXX(; spin=1)
    H = TensorMap(ones, ComplexF64, SU2Space(spin => 1)^2 ← SU2Space(spin => 1)^2)
    for (c, b) in blocks(H)
        S = (dim(c) - 1) / 2
        b .= S * (S + 1) / 2 - spin * (spin + 1)
    end
    A = convert(Array, H)
    d = convert(Int, 2 * spin + 1)
    H′ = TensorMap(A, (ℂ^d)^2 ← (ℂ^d)^2)
    return MPOHamiltonian(H′)
end

function bilinear_biquadratic_model(::Type{SU2Irrep}; θ=atan(1 / 3))
    H1 = TensorMap(ones, ComplexF64, SU2Space(1 => 1)^2 ← SU2Space(1 => 1)^2)
    for (c, b) in blocks(H1)
        S = (dim(c) - 1) / 2
        b .= S * (S + 1) / 2 - 1 * (1 + 1)
    end
    H2 = H1 * H1
    H = cos(θ) * H1 + sin(θ) * H2
    return MPOHamiltonian(H)
end

function ising_bond_tensor(β)
    t = [exp(β) exp(-β); exp(-β) exp(β)]
    r = eigen(t)
    nt = r.vectors * sqrt(Diagonal(r.values)) * r.vectors
    return nt
end

function classical_ising()
    β = log(1 + sqrt(2)) / 2
    nt = ising_bond_tensor(β)
    O = zeros(ComplexF64, (2, 2, 2, 2))
    O[1, 1, 1, 1] = 1
    O[2, 2, 2, 2] = 1

    @tensor o[-1 -2; -3 -4] := O[1 2; 3 4] * nt[-1; 1] * nt[-2; 2] * nt[-3; 3] * nt[-4; 4]
    return DenseMPO(TensorMap(o, ℂ^2 * ℂ^2, ℂ^2 * ℂ^2))
end

function finite_classical_ising(N)
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
    return DenseMPO(permute(TensorMap(d, ℂ^2 ⊗ ℂ^2, ℂ^2 ⊗ ℂ^2), ((1, 2), (4, 3))))
end

end
