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
        T <: Number, S <: ElementarySpace, M <: DenseVector{T},
        N₁, N₂, N₃, N₄, B <: AbstractBackend, A,
    } <: DerivativeOperator
    leftenv::TensorMap{T, S, N₁, N₂, M}
    rightenv::TensorMap{T, S, N₃, N₄, M}
    backend::B
    allocator::A
end
function PrecomputedDerivative(L::AbstractTensorMap, R::AbstractTensorMap, backend, allocator)
    S = TensorKit.check_spacetype(L, R)
    T = TensorOperations.promote_contract(scalartype(L), scalartype(R))
    M = TensorKit.promote_storagetype(T, L, R)
    return PrecomputedDerivative{T, S, M, numout(L), numin(L), numout(R), numin(R), typeof(backend), typeof(allocator)}(L, R, backend, allocator)
end

const PrecomputedCDerivative{T, S, M, B, A} = PrecomputedDerivative{T, S, M, 1, 2, 2, 1, B, A}
const PrecomputedACDerivative{T, S, M, B, A} = PrecomputedDerivative{T, S, M, 2, 2, 2, 1, B, A}
const PrecomputedAC2Derivative{T, S, M, B, A} = PrecomputedDerivative{T, S, M, 2, 2, 2, 2, B, A}

VectorInterface.scalartype(::Type{<:PrecomputedDerivative{T}}) where {T} = T
TensorKit.storagetype(::Type{<:PrecomputedDerivative{T, S, M}}) where {T, S, M} = M

Base.@assume_effects :foldable function prepared_operator_type(
        ::Type{MPO_C_Hamiltonian{L, R}}, ::Type{B}, ::Type{A}
    ) where {L, R, B, A}
    T = TensorOperations.promote_contract(scalartype(L), scalartype(R))
    S = TensorKit.check_spacetype(L, R)
    M = TensorKit.promote_storagetype(T, L, R)
    return PrecomputedCDerivative{T, S, M, B, A}
end
Base.@assume_effects :foldable function prepared_operator_type(
        ::Type{MPO_AC_Hamiltonian{L, O, R}}, ::Type{B}, ::Type{A}
    ) where {L, O, R, B, A}
    T = TensorOperations.promote_contract(scalartype(L), scalartype(O), scalartype(R))
    S = TensorKit.check_spacetype(L, O, R)
    M = TensorKit.promote_storagetype(T, L, O, R)
    return PrecomputedACDerivative{T, S, M, B, A}
end
Base.@assume_effects :foldable function prepared_operator_type(
        ::Type{MPO_AC2_Hamiltonian{L, O₁, O₂, R}}, ::Type{B}, ::Type{A}
    ) where {L, O₁, O₂, R, B, A}
    T = TensorOperations.promote_contract(scalartype(L), scalartype(O₁), scalartype(O₂), scalartype(R))
    S = TensorKit.check_spacetype(L, O₁, O₂, R)
    M = TensorKit.promote_storagetype(T, L, O₁, O₂, R)
    return PrecomputedAC2Derivative{T, S, M, B, A}
end

function prepare_operator!!(
        H::MPO_C_Hamiltonian{<:MPSTensor, <:MPSTensor},
        backend::AbstractBackend, allocator
    )
    leftenv = _transpose_tail(H.leftenv isa TensorMap ? H.leftenv : TensorMap(H.leftenv))
    rightenv = H.rightenv isa TensorMap ? H.rightenv : TensorMap(H.rightenv)
    return prepared_operator_type(typeof(H), typeof(backend), typeof(allocator))(
        leftenv, rightenv, backend, allocator
    )
end
function prepare_operator!!(
        H::MPO_AC_Hamiltonian{<:MPSTensor, <:MPOTensor, <:MPSTensor},
        backend::AbstractBackend, allocator
    )
    @plansor backend = backend allocator = allocator begin
        GL_O[-1 -2 -3; -4 -5] := H.leftenv[-1 1; -4] * H.operators[1][1 -2; -5 -3]
    end
    leftenv = GL_O isa TensorMap ? GL_O : TensorMap(GL_O)
    leftenv = repartition(fuse_legs(leftenv, 1, 2), 2, 2)
    rightenv = H.rightenv isa TensorMap ? H.rightenv : TensorMap(H.rightenv)

    return prepared_operator_type(typeof(H), typeof(backend), typeof(allocator))(
        leftenv, rightenv, backend, allocator
    )
end

function prepare_operator!!(
        H::MPO_AC2_Hamiltonian{<:MPSTensor, <:MPOTensor, <:MPOTensor, <:MPSTensor},
        backend::AbstractBackend, allocator
    )
    @plansor backend = backend allocator = allocator begin
        GL_O[-1 -2 -3; -4 -5] := H.leftenv[-1 1; -4] * H.operators[1][1 -2; -5 -3]
        O_GR[-1 -2; -4 -5 -3] := H.operators[2][-3 -5; -2 1] * H.rightenv[-1 1; -4]
    end
    leftenv = GL_O isa TensorMap ? GL_O : TensorMap(GL_O)
    leftenv = repartition(fuse_legs(leftenv, 1, 2), 2, 2)

    rightenv = O_GR isa TensorMap ? O_GR : TensorMap(O_GR)
    rightenv = repartition(fuse_legs(rightenv, 2, 1), 2, 2)
    return prepared_operator_type(typeof(H), typeof(backend), typeof(allocator))(
        leftenv, rightenv, backend, allocator
    )
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

    L_fused = fuse_legs(L, 2, 1)
    x_fused = fuse_legs(x, 2, 1)
    LxR_fused = PrecomputedDerivative(L_fused, R, backend, allocator)(x_fused)

    return TensorMap{scalartype(LxR_fused)}(LxR_fused.data, codomain(L) ← domain(R))
end
function (H::PrecomputedAC2Derivative)(x::MPOTensor)
    backend, allocator = H.backend, H.allocator
    L, R = H.leftenv, H.rightenv

    L_fused = fuse_legs(L, 2, 1)
    x_fused = fuse_legs(x, 2, 2)
    R_fused = fuse_legs(R, 1, 2)
    LxR_fused = PrecomputedDerivative(L_fused, R_fused, backend, allocator)(x_fused)

    return TensorMap{scalartype(LxR_fused)}(LxR_fused.data, codomain(L) ← domain(R))
end

# TODO: these contractions are annoying and could be better if the input structure was different
# TODO: allocator things
function (H::PrecomputedACDerivative)(x::AbstractTensorMap{<:Any, <:Any, 3, 1})
    backend, allocator = H.backend, H.allocator
    L, R = H.leftenv, H.rightenv

    @plansor backend = backend allocator = allocator begin
        xR[-1 -2; -4 -5 -3] := x[-1 -2 3; 1] * R[1 2; -4] * τ[2 3; -5 -3]
    end
    xR_fused = fuse_legs(xR, 2, 1)
    @plansor backend = backend allocator = allocator begin
        y[-1 -2 -3; -4] := L[-1 -2; 1 2] * xR_fused[1; -4 -3 2]
    end
    return y
end
function (H::PrecomputedAC2Derivative)(x::AbstractTensorMap{<:Any, <:Any, 3, 3})
    backend, allocator = H.backend, H.allocator
    L, R = H.leftenv, H.rightenv

    x_braided = fuse_legs(braid(x, ((1, 2, 3, 5), (4, 6)), (1, 2, 3, 4, 5, 6)), 1, 2)

    @plansor backend = backend allocator = allocator begin
        xR[-1 -2; -4 -6 -7 -5 -3] := x_braided[-1 -2 -3 -5; 1] * R[1 -7; -4 -6]
    end
    @notensor xR_braided = braid(fuse_legs(xR, 2, 1), ((1, 4), (2, 3, 5, 6)), (1, 4, 6, 7, 5, 3))
    @plansor backend = backend allocator = allocator begin
        y_braided[-1 -2; -4 -6 -5 -3] := L[-1 -2; 1 2] * xR_braided[1 2; -4 -6 -5 -3]
    end
    return braid(y_braided, ((1, 2, 6), (3, 5, 4)), (1, 2, 4, 5, 5, 3))
end

const _ToPrepare = Union{
    MPO_C_Hamiltonian{<:MPSTensor, <:MPSTensor},
    MPO_AC_Hamiltonian{<:MPSTensor, <:MPOTensor, <:MPSTensor},
    MPO_AC2_Hamiltonian{<:MPSTensor, <:MPOTensor, <:MPOTensor, <:MPSTensor},
}

function prepare_operator!!(H::Multiline{<:_ToPrepare}, backend::AbstractBackend, allocator)
    return Multiline(map(x -> prepare_operator!!(x, backend, allocator), parent(H)))
end
