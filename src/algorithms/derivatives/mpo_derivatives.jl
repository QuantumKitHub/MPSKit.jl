"""
    struct MPO_∂∂ACN{L,O<:Tuple,R}

Effective local operator obtained from taking the partial derivative of an MPS-MPO-MPS sandwich.
"""
struct MPO_∂∂ACN{L,O<:Tuple,R} <: DerivativeOperator
    leftenv::L
    operators::O
    rightenv::R
end

Base.length(H::MPO_∂∂ACN) = length(H.operators)

const MPO_∂∂C{L,R} = MPO_∂∂ACN{L,Tuple{},R}
MPO_∂∂C(GL, GR) = MPO_∂∂ACN(GL, (), GR)

const MPO_∂∂AC{L,O,R} = MPO_∂∂ACN{L,Tuple{O},R}
MPO_∂∂AC(GL, O, GR) = MPO_∂∂ACN(GL, (O,), GR)

const MPO_∂∂AC2{L,O₁,O₂,R} = MPO_∂∂ACN{L,Tuple{O₁,O₂},R}
MPO_∂∂AC2(GL, O1, O2, GR) = MPO_∂∂ACN(GL, (O1, O2), GR)

# Constructors
# ------------
function ∂∂C(pos::Int, mps, operator, envs)
    return MPO_∂∂C(leftenv(envs, pos + 1, mps), rightenv(envs, pos, mps))
end
function ∂∂AC(pos::Int, mps, operator, envs)
    O = isnothing(operator) ? nothing : operator[pos]
    return MPO_∂∂AC(leftenv(envs, pos, mps), O, rightenv(envs, pos, mps))
end
function ∂∂AC2(pos::Int, mps, operator, envs)
    O1, O2 = isnothing(operator) ? (nothing, nothing) : (operator[pos], operator[pos + 1])
    return MPO_∂∂AC2(leftenv(envs, pos, mps), O1, O2, rightenv(envs, pos + 1, mps))
end

# Properties
# ----------
function TensorKit.domain(H::MPO_∂∂ACN)
    V_l = right_virtualspace(H.leftenv)
    V_r = left_virtualspace(H.rightenv)
    V_o = prod(physicalspace, H.O; init=one(V_l))
    return V_l ⊗ V_o ⊗ V_r
end
function TensorKit.codomain(H::MPO_∂∂ACN)
    V_l = left_virtualspace(H.leftenv)
    V_r = right_virtualspace(H.rightenv)
    V_o = prod(physicalspace, H.O; init=one(V_l))
    return V_l ⊗ V_o ⊗ V_r
end

# Actions
# -------
function (h::MPO_∂∂C{<:MPSBondTensor,<:MPSBondTensor})(x::MPSBondTensor)
    @plansor y[-1; -2] ≔ h.leftenv[-1; 1] * x[1; 2] * h.rightenv[2; -2]
    return y isa AbstractBlockTensorMap ? only(y) : y
end
function (h::MPO_∂∂C{<:MPSTensor,<:MPSTensor})(x::MPSBondTensor)
    @plansor y[-1; -2] ≔ h.leftenv[-1 3; 1] * x[1; 2] * h.rightenv[2 3; -2]
    return y isa AbstractBlockTensorMap ? only(y) : y
end

function (h::MPO_∂∂AC{<:MPSTensor,<:MPOTensor,<:MPSTensor})(x::MPSTensor)
    @plansor y[-1 -2; -3] ≔ h.leftenv[-1 5; 4] * x[4 2; 1] *
                            h.operators[1][5 -2; 2 3] * h.rightenv[1 3; -3]
    return y isa AbstractBlockTensorMap ? only(y) : y
end
function (h::MPO_∂∂AC{<:MPSTensor,<:Number,<:MPSTensor})(x::MPSTensor)
    @plansor y[-1 -2; -3] ≔ (h.leftenv[-1 5; 4] * x[4 6; 1] * τ[6 5; 7 -2] *
                             h.rightenv[1 7; -3]) * only(h.operators)
    return y isa AbstractBlockTensorMap ? only(y) : y
end
function (h::MPO_∂∂AC{<:MPSBondTensor,Nothing,<:MPSBondTensor})(x::MPSTensor)
    @plansor y[-1 -2; -3] ≔ h.leftenv[-1; 2] * x[2 -2; 1] * h.rightenv[1; -3]
end
function (h::MPO_∂∂AC{<:MPSTensor,<:MPOTensor,<:MPSTensor})(x::GenericMPSTensor{<:Any,3})
    @plansor y[-1 -2 -3; -4] ≔ h.leftenv[-1 7; 6] * x[6 4 2; 1] *
                               h.operators[1][7 -2; 4 5] * τ[5 -3; 2 3] *
                               h.rightenv[1 3; -4]
    return y isa AbstractBlockTensorMap ? only(y) : y
end

function (h::MPO_∂∂AC2{<:MPSBondTensor,Nothing,Nothing,<:MPSBondTensor})(x::MPOTensor)
    @plansor y[-1 -2; -3 -4] ≔ h.leftenv[-1; 1] * x[1 -2; 2 -4] * h.rightenv[2 -3]
    return y isa AbstractBlockTensorMap ? only(y) : y
end
function (h::MPO_∂∂AC2{<:MPSTensor,<:MPOTensor,<:MPOTensor,<:MPSTensor})(x::MPOTensor)
    @plansor y[-1 -2; -3 -4] ≔ h.leftenv[-1 7; 6] * x[6 5; 1 3] *
                               h.operators[1][7 -2; 5 4] * h.operators[2][4 -4; 3 2] *
                               h.rightenv[1 2; -3]
    return y isa AbstractBlockTensorMap ? only(y) : y
end
function (h::MPO_∂∂AC2{<:MPSTensor,<:MPOTensor,<:MPOTensor,<:MPSTensor})(x::AbstractTensorMap{<:Any,
                                                                                              <:Any,
                                                                                              3,
                                                                                              3})
    @plansor y[-1 -2 -3; -4 -5 -6] ≔ h.leftenv[-1 11; 10] * x[10 8 6; 1 2 4] *
                                     h.rightenv[1 3; -4] *
                                     h.operators[1][11 -2; 8 9] * τ[9 -3; 6 7] *
                                     h.operators[2][7 -6; 4 5] * τ[5 -5; 2 3]
    return y isa AbstractBlockTensorMap ? only(y) : y
end
