"""
    struct MPODerivativeOperator{L,O<:Tuple,R}

Effective local operator obtained from taking the partial derivative of an MPS-MPO-MPS sandwich.
"""
struct MPODerivativeOperator{L, O <: Tuple, R} <: DerivativeOperator
    leftenv::L
    operators::O
    rightenv::R
end

struct MPOContractedDerivativeOperator{L, R, N} <: DerivativeOperator
    leftenv::L
    rightenv::R
end

Base.length(H::MPODerivativeOperator) = length(H.operators)
Base.length(::MPOContractedDerivativeOperator{L, R, N}) where {L, R, N} = N

const MPO_C_Hamiltonian{L, R} = MPODerivativeOperator{L, Tuple{}, R}
MPO_C_Hamiltonian(GL, GR) = MPODerivativeOperator(GL, (), GR)

const MPO_AC_Hamiltonian{L, O, R} = MPODerivativeOperator{L, Tuple{O}, R}
MPO_AC_Hamiltonian(GL, O, GR) = MPODerivativeOperator(GL, (O,), GR)

const MPO_AC2_Hamiltonian{L, O₁, O₂, R} = MPODerivativeOperator{L, Tuple{O₁, O₂}, R}
MPO_AC2_Hamiltonian(GL, O1, O2, GR) = MPODerivativeOperator(GL, (O1, O2), GR)

const MPO_Contracted_AC_Hamiltonian{L, R} = MPOContractedDerivativeOperator{L, R, 1}
MPO_Contracted_AC_Hamiltonian(GL::L, GR::R) where {L, R} = MPOContractedDerivativeOperator{L,R,1}(GL,  GR)

const MPO_Contracted_AC2_Hamiltonian{L, R} = MPOContractedDerivativeOperator{L, R, 2}
MPO_Contracted_AC2_Hamiltonian(GL::L, GR::R) where {L, R} = MPOContractedDerivativeOperator{L,R,2}(GL, GR)

# Constructors
# ------------
const _HAM_MPS_TYPES = Union{
    FiniteMPS{<:MPSTensor},
    WindowMPS{<:MPSTensor},
    InfiniteMPS{<:MPSTensor},
}

function C_hamiltonian(site::Int, below, operator, above, envs)
    return MPO_C_Hamiltonian(leftenv(envs, site + 1, below), rightenv(envs, site, below))
end
function AC_hamiltonian(site::Int, below, operator, above, envs)
    O = isnothing(operator) ? nothing : operator[site]
    return MPO_AC_Hamiltonian(leftenv(envs, site, below), O, rightenv(envs, site, below))
end
function AC2_hamiltonian(site::Int, below, operator, above, envs)
    O1, O2 = isnothing(operator) ? (nothing, nothing) : (operator[site], operator[site + 1])
    return MPO_AC2_Hamiltonian(
        leftenv(envs, site, below), O1, O2, rightenv(envs, site + 1, below)
    )
end
function AC_hamiltonian(site::Int, below::_HAM_MPS_TYPES, operator::MPO{<:MPOTensor}, above::_HAM_MPS_TYPES, envs)
    O = operator[site]
    GL = leftenv(envs, site, below)
    return AC_hamiltonian(GL, O, rightenv(envs, site, below))
end
function AC_hamiltonian(GL::MPSTensor, O::MPOTensor, GR::MPSTensor)
    @plansor GLW[-1 -2 -3; -4 -5] ≔ GL[-1 1; -4] * O[1 -2; -5 -3]
    return MPO_Contracted_AC_Hamiltonian(GLW, GR)
end
function AC2_hamiltonian(site::Int, below::_HAM_MPS_TYPES, operator::MPO{<:MPOTensor}, above::_HAM_MPS_TYPES, envs)
    O1 = operator[site]
    O2 = operator[site + 1]
    GL = leftenv(envs, site, below)
    GR = rightenv(envs, site + 1, below)
    return AC2_hamiltonian(GL, O1, O2, GR)
end
function AC2_hamiltonian(GL::MPSTensor, O1::MPOTensor, O2::MPOTensor, GR::MPSTensor)
    @plansor GLW[-1 -2 -3; -4 -5] ≔ GL[-1 1; -4] * O1[1 -2; -5 -3]
    @plansor GWR[-1 -2 -3; -4 -5] ≔ O2[-3 -5; -2 1] * GR[-1 1; -4]
    return MPO_Contracted_AC2_Hamiltonian(GLW, GWR)
end

# Properties
# ----------
function TensorKit.domain(H::MPODerivativeOperator)
    V_l = right_virtualspace(H.leftenv)
    V_r = left_virtualspace(H.rightenv)
    V_o = prod(physicalspace, H.O; init = one(V_l))
    return V_l ⊗ V_o ⊗ V_r
end
function TensorKit.codomain(H::MPODerivativeOperator)
    V_l = left_virtualspace(H.leftenv)
    V_r = right_virtualspace(H.rightenv)
    V_o = prod(physicalspace, H.O; init = one(V_l))
    return V_l ⊗ V_o ⊗ V_r
end
function TensorKit.domain(H::MPOContractedDerivativeOperator)
    V_l = right_virtualspace(H.leftenv)
    V_r = left_virtualspace(H.rightenv)
    ## TODO: How to deal with the H.O here?
    V_o = prod(physicalspace, H.O; init = one(V_l))
    return V_l ⊗ V_o ⊗ V_r
end
function TensorKit.codomain(H::MPOContractedDerivativeOperator)
    V_l = left_virtualspace(H.leftenv)
    V_r = right_virtualspace(H.rightenv)
    ## TODO: How to deal with the H.O here?
    V_o = prod(physicalspace, H.O; init = one(V_l))
    return V_l ⊗ V_o ⊗ V_r
end

# Actions
# -------
function (h::MPO_C_Hamiltonian{<:MPSBondTensor, <:MPSBondTensor})(x::MPSBondTensor)
    @plansor y[-1; -2] ≔ h.leftenv[-1; 1] * x[1; 2] * h.rightenv[2; -2]
    return y isa AbstractBlockTensorMap ? only(y) : y
end
function (h::MPO_C_Hamiltonian{<:MPSTensor, <:MPSTensor})(x::MPSBondTensor)
    @plansor y[-1; -2] ≔ h.leftenv[-1 3; 1] * x[1; 2] * h.rightenv[2 3; -2]
    return y isa AbstractBlockTensorMap ? only(y) : y
end

function (h::MPO_AC_Hamiltonian{<:MPSTensor, <:MPOTensor, <:MPSTensor})(x::MPSTensor)
    @plansor y[-1 -2; -3] ≔ h.leftenv[-1 5; 4] * x[4 2; 1] * h.operators[1][5 -2; 2 3] *
        h.rightenv[1 3; -3]
    return y isa AbstractBlockTensorMap ? only(y) : y
end
function (h::MPO_AC_Hamiltonian{<:MPSTensor, <:Number, <:MPSTensor})(x::MPSTensor)
    @plansor y[-1 -2; -3] ≔ (
        h.leftenv[-1 5; 4] * x[4 6; 1] * τ[6 5; 7 -2] * h.rightenv[1 7; -3]
    ) * only(h.operators)
    return y isa AbstractBlockTensorMap ? only(y) : y
end
function (h::MPO_AC_Hamiltonian{<:MPSBondTensor, Nothing, <:MPSBondTensor})(x::MPSTensor)
    return @plansor y[-1 -2; -3] ≔ h.leftenv[-1; 2] * x[2 -2; 1] * h.rightenv[1; -3]
end
function (h::MPO_AC_Hamiltonian{<:MPSTensor, <:MPOTensor, <:MPSTensor})(
        x::GenericMPSTensor{<:Any, 3}
    )
    @plansor y[-1 -2 -3; -4] ≔ h.leftenv[-1 7; 6] * x[6 4 2; 1] *
        h.operators[1][7 -2; 4 5] * τ[5 -3; 2 3] * h.rightenv[1 3; -4]
    return y isa AbstractBlockTensorMap ? only(y) : y
end

function (h::MPO_AC2_Hamiltonian{<:MPSBondTensor, Nothing, Nothing, <:MPSBondTensor})(
        x::MPOTensor
    )
    @plansor y[-1 -2; -3 -4] ≔ h.leftenv[-1; 1] * x[1 -2; 2 -4] * h.rightenv[2 -3]
    return y isa AbstractBlockTensorMap ? only(y) : y
end
function (h::MPO_AC2_Hamiltonian{<:MPSTensor, <:MPOTensor, <:MPOTensor, <:MPSTensor})(
        x::MPOTensor
    )
    @plansor y[-1 -2; -3 -4] ≔ h.leftenv[-1 7; 6] * x[6 5; 1 3] *
        h.operators[1][7 -2; 5 4] * h.operators[2][4 -4; 3 2] * h.rightenv[1 2; -3]
    return y isa AbstractBlockTensorMap ? only(y) : y
end
function (h::MPO_AC2_Hamiltonian{<:MPSTensor, <:MPOTensor, <:MPOTensor, <:MPSTensor})(
        x::AbstractTensorMap{<:Any, <:Any, 3, 3}
    )
    @plansor y[-1 -2 -3; -4 -5 -6] ≔ h.leftenv[-1 11; 10] * x[10 8 6; 1 2 4] *
        h.rightenv[1 3; -4] * h.operators[1][11 -2; 8 9] * τ[9 -3; 6 7] *
        h.operators[2][7 -6; 4 5] * τ[5 -5; 2 3]
    return y isa AbstractBlockTensorMap ? only(y) : y
end
function (h::MPO_Contracted_AC_Hamiltonian)(
        x::AbstractTensorMap{<:Any, <:Any, 2, 1}
    )
    @plansor y[-1 -2; -3] ≔ h.leftenv[-1 -2 3; 1 2] * x[1 2; 4] * h.rightenv[4 3; -3]
    return y isa AbstractBlockTensorMap ? only(y) : y
end
function (h::MPO_Contracted_AC2_Hamiltonian)(
        x::MPOTensor
    )
    @plansor y[-1 -2; -3 -4] ≔ h.leftenv[-1 -2 5; 1 2] * x[1 2; 3 4] * h.rightenv[3 4 5; -3 -4]
    return y isa AbstractBlockTensorMap ? only(y) : y
end