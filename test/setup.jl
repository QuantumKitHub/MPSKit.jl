# Planar stuff
# ----------------------------

using Test, TestExtras
using MPSKit
using MPSKit: _transpose_tail, _transpose_front
using TensorKit
using TensorKit: PlanarTrivial, ℙ

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
    return MPOHamiltonian(
        map(Iterators.product(1:(mpo.period), 1:(mpo.odim), 1:(mpo.odim))) do (i, j, k)
            force_planar(mpo.Os[i, j, k])
        end,
    )
end
force_planar(mpo::DenseMPO) = DenseMPO(force_planar.(mpo.opp))

# Toy models
# ----------------------------
using LinearAlgebra: Diagonal

function transverse_field_ising(; g=1.0)
    X = TensorMap(ComplexF64[0 1; 1 0], ℂ^2 ← ℂ^2)
    Z = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2 ← ℂ^2)
    E = TensorMap(ComplexF64[1 0; 0 1], ℂ^2 ← ℂ^2)
    H = Z ⊗ Z + (g / 2) * (X ⊗ E + E ⊗ X)
    return MPOHamiltonian(H)
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

function classical_ising()
    β = log(1 + sqrt(2)) / 2
    t = [exp(β) exp(-β); exp(-β) exp(β)]
    r = eigen(t)
    nt = r.vectors * sqrt(Diagonal(r.values)) * r.vectors
    O = zeros(ComplexF64, (2, 2, 2, 2))
    O[1, 1, 1, 1] = 1
    O[2, 2, 2, 2] = 1

    @tensor o[-1 -2; -3 -4] := O[1 2; 3 4] * nt[-1; 1] * nt[-2; 2] * nt[-3; 3] * nt[-4; 4]
    return DenseMPO(TensorMap(o, ℂ^2 * ℂ^2, ℂ^2 * ℂ^2))
end

function sixvertex(; a=1.0, b=1.0, c=1.0)
    d = ComplexF64[
        a 0 0 0
        0 c b 0
        0 b c 0
        0 0 0 a
    ]
    return DenseMPO(permute(TensorMap(d, ℂ^2 ⊗ ℂ^2, ℂ^2 ⊗ ℂ^2), ((1, 2), (4, 3))))
end
