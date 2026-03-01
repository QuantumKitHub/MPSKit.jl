"""
    struct MPODerivativeOperator{L,O<:Tuple,R}

Effective local operator obtained from taking the partial derivative of an MPS-MPO-MPS sandwich.
"""
struct MPODerivativeOperator{L, O <: Tuple, R} <: DerivativeOperator
    leftenv::L
    operators::O
    rightenv::R
end

Base.length(H::MPODerivativeOperator) = length(H.operators)

const MPO_C_Hamiltonian{L, R} = MPODerivativeOperator{L, Tuple{}, R}
MPO_C_Hamiltonian(GL, GR) = MPODerivativeOperator(GL, (), GR)

const MPO_AC_Hamiltonian{L, O, R} = MPODerivativeOperator{L, Tuple{O}, R}
MPO_AC_Hamiltonian(GL, O, GR) = MPODerivativeOperator(GL, (O,), GR)

const MPO_AC2_Hamiltonian{L, O₁, O₂, R} = MPODerivativeOperator{L, Tuple{O₁, O₂}, R}
MPO_AC2_Hamiltonian(GL, O1, O2, GR) = MPODerivativeOperator(GL, (O1, O2), GR)

# Constructors
# ------------
function C_hamiltonian(site::Int, below, operator, above, envs; prepare::Bool = true)
    H_C = MPO_C_Hamiltonian(leftenv(envs, site + 1, below), rightenv(envs, site, below))
    return prepare ? prepare_operator!!(H_C) : H_C
end
function AC_hamiltonian(site::Int, below, operator, above, envs; prepare::Bool = true)
    O = isnothing(operator) ? nothing : operator[site]
    H_AC = MPO_AC_Hamiltonian(leftenv(envs, site, below), O, rightenv(envs, site, below))
    return prepare ? prepare_operator!!(H_AC) : H_AC
end
function AC2_hamiltonian(site::Int, below, operator, above, envs; prepare::Bool = true)
    O1, O2 = isnothing(operator) ? (nothing, nothing) : (operator[site], operator[site + 1])
    H_AC2 = MPO_AC2_Hamiltonian(
        leftenv(envs, site, below), O1, O2, rightenv(envs, site + 1, below)
    )
    return prepare ? prepare_operator!!(H_AC2) : H_AC2
end

# Properties
# ----------
function TensorKit.domain(H::MPODerivativeOperator)
    V_l = right_virtualspace(H.leftenv)
    V_r = left_virtualspace(H.rightenv)
    V_o = prod(physicalspace, H.operators; init = one(V_l))
    return V_l ⊗ V_o ⊗ V_r
end
function TensorKit.codomain(H::MPODerivativeOperator)
    V_l = left_virtualspace(H.leftenv)
    V_r = right_virtualspace(H.rightenv)
    V_o = prod(physicalspace, H.operators; init = one(V_l))
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

# prepared operators
# ------------------
struct PrecomputedDerivative{
        T <: Number, S <: ElementarySpace, N₁, N₂, N₃, N₄,
        T1 <: AbstractTensorMap{T, S, N₁, N₂}, T2 <: AbstractTensorMap{T, S, N₃, N₄},
        B <: AbstractBackend, A,
    } <: DerivativeOperator
    leftenv::T1
    rightenv::T2
    backend::B
    allocator::A
end

const PrecomputedCDerivative{T, S} = PrecomputedDerivative{T, S, 1, 2, 2, 1}
const PrecomputedACDerivative{T, S} = PrecomputedDerivative{T, S, 2, 3, 2, 1}
const PrecomputedAC2Derivative{T, S} = PrecomputedDerivative{T, S, 2, 3, 3, 2}

VectorInterface.scalartype(::Type{<:PrecomputedDerivative{T}}) where {T} = T

function prepare_operator!!(
        H::MPO_C_Hamiltonian{<:MPSTensor, <:MPSTensor},
        backend::AbstractBackend, allocator
    )
    leftenv = _transpose_tail(H.leftenv isa TensorMap ? H.leftenv : TensorMap(H.leftenv))
    rightenv = H.rightenv isa TensorMap ? H.rightenv : TensorMap(H.rightenv)
    return PrecomputedDerivative(leftenv, rightenv, backend, allocator)
end
function prepare_operator!!(
        H::MPO_AC_Hamiltonian{<:MPSTensor, <:MPOTensor, <:MPSTensor},
        backend::AbstractBackend, allocator
    )
    @plansor backend = backend allocator = allocator begin
        GL_O[-1 -2; -4 -5 -3] := H.leftenv[-1 1; -4] * H.operators[1][1 -2; -5 -3]
    end
    leftenv = GL_O isa TensorMap ? GL_O : TensorMap(GL_O)
    rightenv = H.rightenv isa TensorMap ? H.rightenv : TensorMap(H.rightenv)

    return PrecomputedDerivative(leftenv, rightenv, backend, allocator)
end

function prepare_operator!!(
        H::MPO_AC2_Hamiltonian{<:MPSTensor, <:MPOTensor, <:MPOTensor, <:MPSTensor},
        backend::AbstractBackend, allocator
    )
    @plansor backend = backend allocator = allocator begin
        GL_O[-1 -2; -4 -5 -3] := H.leftenv[-1 1; -4] * H.operators[1][1 -2; -5 -3]
        O_GR[-1 -2 -3; -4 -5] := H.operators[2][-3 -5; -2 1] * H.rightenv[-1 1; -4]
    end

    leftenv = GL_O isa TensorMap ? GL_O : TensorMap(GL_O)
    rightenv = O_GR isa TensorMap ? O_GR : TensorMap(O_GR)
    return PrecomputedDerivative(leftenv, rightenv, backend, allocator)
end

function (H::PrecomputedCDerivative)(x::MPSBondTensor)
    backend, allocator = H.backend, H.allocator
    L, R = H.leftenv, H.rightenv
    cp = allocator_checkpoint!(allocator)

    TC = TensorOperations.promote_contract(scalartype(x), scalartype(H))
    xR = TensorOperations.tensoralloc_contract(
        TC, x, ((1,), (2,)), false, R, ((1,), (2, 3)), false, ((1, 2), (3,)), Val(true), allocator
    )
    mul_front!(xR, x, R, One(), Zero(), backend, allocator)
    LxR = L * xR

    TensorOperations.tensorfree!(xR, allocator)
    allocator_reset!(allocator, cp)

    return LxR
end
function (H::PrecomputedACDerivative)(x::MPSTensor)
    backend, allocator = H.backend, H.allocator
    L, R = H.leftenv, H.rightenv

    L_fused = fuse_legs(L, 2, 2)
    x_fused = fuse_legs(x, 2, 1)
    LxR_fused = PrecomputedDerivative(L_fused, R, backend, allocator)(x_fused)

    return TensorMap{scalartype(LxR_fused)}(LxR_fused.data, codomain(L) ← domain(R))
end
function (H::PrecomputedAC2Derivative)(x::MPOTensor)
    backend, allocator = H.backend, H.allocator
    L, R = H.leftenv, H.rightenv

    L_fused = fuse_legs(L, 2, 2)
    x_fused = fuse_legs(x, 2, 2)
    R_fused = fuse_legs(R, 2, 2)
    LxR_fused = PrecomputedDerivative(L_fused, R_fused, backend, allocator)(x_fused)

    return TensorMap{scalartype(LxR_fused)}(LxR_fused.data, codomain(L) ← domain(R))
end

# TODO: these contractions are annoying and could be better if the input structure was different
# TODO: allocator things
function (H::PrecomputedACDerivative)(x::AbstractTensorMap{<:Any, <:Any, 3, 1})
    backend, allocator = H.backend, H.allocator
    L, R = H.leftenv, H.rightenv

    @plansor backend = backend allocator = allocator begin
        y[-1 -2 -3; -4] ≔
            L[-1 -2; 4 5 2] * x[4 5 3; 1] * τ[2 -3; 3 6] * R[1 6; -4]
    end
    return y
end
function (H::PrecomputedAC2Derivative)(x::AbstractTensorMap{<:Any, <:Any, 3, 3})
    backend, allocator = H.backend, H.allocator
    L, R = H.leftenv, H.rightenv

    x_braided = braid(x, ((5, 3, 1, 2), (4, 6)), (1, 2, 3, 4, 5, 6))
    @plansor backend = backend allocator = allocator begin
        y_braided[-5 -3 -1 -2; -4 -6] ≔
            L[-1 -2; 3 4 5] * x_braided[-5 -3 3 4; 1 2] * R[1 2 5; -4 -6]
    end
    return braid(y_braided, ((3, 4, 2), (5, 1, 6)), (5, 3, 1, 2, 4, 6))
end

const _ToPrepare = Union{
    MPO_C_Hamiltonian{<:MPSTensor, <:MPSTensor},
    MPO_AC_Hamiltonian{<:MPSTensor, <:MPOTensor, <:MPSTensor},
    MPO_AC2_Hamiltonian{<:MPSTensor, <:MPOTensor, <:MPOTensor, <:MPSTensor},
}

function prepare_operator!!(H::Multiline{<:_ToPrepare}, backend::AbstractBackend, allocator)
    return Multiline(map(x -> prepare_operator!!(x, backend, allocator), parent(H)))
end
