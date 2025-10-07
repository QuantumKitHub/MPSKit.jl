"""
    struct MPODerivativeOperator{L,O<:Tuple,R}

Effective local operator obtained from taking the partial derivative of an MPS-MPO-MPS sandwich.
"""
struct MPODerivativeOperator{L, R, O} <: DerivativeOperator
    leftenv::L
    rightenv::R
end

Base.length(H::MPODerivativeOperator) = length(H.operators)

const MPO_C_Hamiltonian{L, R} = MPODerivativeOperator{L, R, 0}
MPO_C_Hamiltonian(GL::L, GR::R) where {L, R} = MPODerivativeOperator{L,R,0}(GL, GR)

const MPO_AC_Hamiltonian{L, R, O} = MPODerivativeOperator{L, R, O}
MPO_AC_Hamiltonian(GL::L, ::O, GR::R) where {L, R, O} = MPODerivativeOperator{L,R,O}(GL, GR)

const MPO_AC2_Hamiltonian{L, R} = MPODerivativeOperator{L, R, 2}
MPO_AC2_Hamiltonian(GL::L, ::O₁, ::O₂, GR::R) where {L, R, O₁, O₂} = MPODerivativeOperator{L,R,Tuple{O₁,O₂}}(GL, GR)


# Constructors
# ------------
function C_hamiltonian(site::Int, below, operator, above, envs)
    return MPO_C_Hamiltonian(leftenv(envs, site + 1, below), rightenv(envs, site, below))
end
function AC_hamiltonian(site::Int, below, operator<:MPOTensor, above, envs)
    O = operator[site]
    L = leftenv(envs, site, below)
    @plansor L[-1 -2 -3; -4 -5] ≔ L[-1 1; -4] * O[1 -2; -5 -3]
    R = rightenv(envs, site, below)
    R = permute(R, (2, 1), (3,))
    return MPO_AC_Hamiltonian(L, O, R)
end
function AC_hamiltonian(site::Int, below, operator, above, envs)
    O = isnothing(operator) ? nothing : operator[site]
    return MPO_AC_Hamiltonian(leftenv(envs, site, below), O, rightenv(envs, site, below))
end
function AC2_hamiltonian(site::Int, below, operator<:MPOTensor, above, envs)
    O1, O2 = operator[site], operator[site + 1]
    L = leftenv(envs, site, below)
    @tensor GLW[-1 -2 -3; -4 -5] ≔ L[-1 1; -4] * O1[1 -2; -5 -3]
    @tensor GWR[-1 -2 -3; -4 -5] ≔ O2[-1 -5; -3 1] * R[-2 1; -4]
    return MPO_AC2_Hamiltonian(L, O1, O2, R)
end
function AC2_hamiltonian(site::Int, below, operator, above, envs)
    O1, O2 = isnothing(operator) ? (nothing, nothing) : (operator[site], operator[site + 1])
    return MPO_AC2_Hamiltonian(
        leftenv(envs, site, below), O1, O2, rightenv(envs, site + 1, below)
    )
end

# Properties
# ----------
function TensorKit.domain(H::MPODerivativeOperator{L, R, O}) where {L, R, O}
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
