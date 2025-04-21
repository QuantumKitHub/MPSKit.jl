"""
    struct MPODerivativeOperator{L,O<:Tuple,R}

Effective local operator obtained from taking the partial derivative of an MPS-MPO-MPS sandwich.
"""
struct MPODerivativeOperator{L,O<:Tuple,R} <: DerivativeOperator
    leftenv::L
    operators::O
    rightenv::R
end

Base.length(H::MPODerivativeOperator) = length(H.operators)

const MPO_C_Hamiltonian{L,R} = MPODerivativeOperator{L,Tuple{},R}
MPO_C_Hamiltonian(GL, GR) = MPODerivativeOperator(GL, (), GR)

const MPO_AC_Hamiltonian{L,O,R} = MPODerivativeOperator{L,Tuple{O},R}
MPO_AC_Hamiltonian(GL, O, GR) = MPODerivativeOperator(GL, (O,), GR)

const MPO_AC2_Hamiltonian{L,O₁,O₂,R} = MPODerivativeOperator{L,Tuple{O₁,O₂},R}
MPO_AC2_Hamiltonian(GL, O1, O2, GR) = MPODerivativeOperator(GL, (O1, O2), GR)

# Constructors
# ------------
function C_hamiltonian(site::Int, below, operator, above, envs)
    return MPO_C_Hamiltonian(leftenv(envs, site + 1, below), rightenv(envs, site, below))
end
function AC_hamiltonian(site::Int, below, operator, above, envs)
    O = isnothing(operator) ? nothing : operator[site]
    return MPO_AC_Hamiltonian(leftenv(envs, site, below), O, rightenv(envs, site, below))
end
function AC2_hamiltonian(site::Int, below, operator, above, envs)
    O1, O2 = isnothing(operator) ? (nothing, nothing) : (operator[site], operator[site + 1])
    return MPO_AC2_Hamiltonian(leftenv(envs, site, below), O1, O2,
                               rightenv(envs, site + 1, below))
end

# Properties
# ----------
function TensorKit.domain(H::MPODerivativeOperator)
    V_l = right_virtualspace(H.leftenv)
    V_r = left_virtualspace(H.rightenv)
    V_o = prod(physicalspace, H.O; init=one(V_l))
    return V_l ⊗ V_o ⊗ V_r
end
function TensorKit.codomain(H::MPODerivativeOperator)
    V_l = left_virtualspace(H.leftenv)
    V_r = right_virtualspace(H.rightenv)
    V_o = prod(physicalspace, H.O; init=one(V_l))
    return V_l ⊗ V_o ⊗ V_r
end

# Actions
# -------
function (h::MPO_C_Hamiltonian{<:MPSBondTensor,<:MPSBondTensor})(x::MPSBondTensor)
    @plansor y[-1; -2] ≔ h.leftenv[-1; 1] * x[1; 2] * h.rightenv[2; -2]
    return y isa AbstractBlockTensorMap ? only(y) : y
end
function (h::MPO_C_Hamiltonian{<:MPSTensor,<:MPSTensor})(x::MPSBondTensor)
    @plansor y[-1; -2] ≔ h.leftenv[-1 3; 1] * x[1; 2] * h.rightenv[2 3; -2]
    return y isa AbstractBlockTensorMap ? only(y) : y
end

function (h::MPO_AC_Hamiltonian{<:MPSTensor,<:MPOTensor,<:MPSTensor})(x::MPSTensor)
    @plansor y[-1 -2; -3] ≔ h.leftenv[-1 5; 4] * x[4 2; 1] *
                            h.operators[1][5 -2; 2 3] * h.rightenv[1 3; -3]
    return y isa AbstractBlockTensorMap ? only(y) : y
end
function (h::MPO_AC_Hamiltonian{<:MPSTensor,<:Number,<:MPSTensor})(x::MPSTensor)
    @plansor y[-1 -2; -3] ≔ (h.leftenv[-1 5; 4] * x[4 6; 1] * τ[6 5; 7 -2] *
                             h.rightenv[1 7; -3]) * only(h.operators)
    return y isa AbstractBlockTensorMap ? only(y) : y
end
function (h::MPO_AC_Hamiltonian{<:MPSBondTensor,Nothing,<:MPSBondTensor})(x::MPSTensor)
    @plansor y[-1 -2; -3] ≔ h.leftenv[-1; 2] * x[2 -2; 1] * h.rightenv[1; -3]
end
function (h::MPO_AC_Hamiltonian{<:MPSTensor,<:MPOTensor,<:MPSTensor})(x::GenericMPSTensor{<:Any,
                                                                                          3})
    @plansor y[-1 -2 -3; -4] ≔ h.leftenv[-1 7; 6] * x[6 4 2; 1] *
                               h.operators[1][7 -2; 4 5] * τ[5 -3; 2 3] *
                               h.rightenv[1 3; -4]
    return y isa AbstractBlockTensorMap ? only(y) : y
end

function (h::MPO_AC2_Hamiltonian{<:MPSBondTensor,Nothing,Nothing,<:MPSBondTensor})(x::MPOTensor)
    @plansor y[-1 -2; -3 -4] ≔ h.leftenv[-1; 1] * x[1 -2; 2 -4] * h.rightenv[2 -3]
    return y isa AbstractBlockTensorMap ? only(y) : y
end
function (h::MPO_AC2_Hamiltonian{<:MPSTensor,<:MPOTensor,<:MPOTensor,<:MPSTensor})(x::MPOTensor)
    @plansor y[-1 -2; -3 -4] ≔ h.leftenv[-1 7; 6] * x[6 5; 1 3] *
                               h.operators[1][7 -2; 5 4] * h.operators[2][4 -4; 3 2] *
                               h.rightenv[1 2; -3]
    return y isa AbstractBlockTensorMap ? only(y) : y
end
function (h::MPO_AC2_Hamiltonian{<:MPSTensor,<:MPOTensor,<:MPOTensor,<:MPSTensor})(x::AbstractTensorMap{<:Any,
                                                                                              <:Any,
                                                                                              3,
                                                                                              3})
    @plansor y[-1 -2 -3; -4 -5 -6] ≔ h.leftenv[-1 11; 10] * x[10 8 6; 1 2 4] *
                                     h.rightenv[1 3; -4] *
                                     h.operators[1][11 -2; 8 9] * τ[9 -3; 6 7] *
                                     h.operators[2][7 -6; 4 5] * τ[5 -5; 2 3]
    return y isa AbstractBlockTensorMap ? only(y) : y
end
