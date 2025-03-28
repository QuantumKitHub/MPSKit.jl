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
(h::MPO_∂∂C)(x) = ∂C(x, h.leftenv, h.rightenv)
(h::MPO_∂∂AC)(x) = ∂AC(x, h.operators[1], h.leftenv, h.rightenv)
(h::MPO_∂∂AC2)(x) = ∂AC2(x, h.operators[1], h.operators[2], h.leftenv, h.rightenv)
