"""
    abstract type AbstractEnvironments end

Abstract supertype for all environment types.
"""
abstract type AbstractEnvironments end

# Allocating tensors
# ------------------

# TODO: fix the fucking left/right virtualspace bullshit
# TODO: storagetype stuff
function allocate_GL(bra::AbstractMPS, mpo::AbstractMPO, ket::AbstractMPS, i::Int)
    T = Base.promote_type(scalartype(bra), scalartype(mpo), scalartype(ket))
    V = left_virtualspace(bra, i - 1) ⊗ left_virtualspace(mpo, i)' ←
        left_virtualspace(ket, i - 1)
    if V isa BlockTensorKit.TensorMapSumSpace
        return BlockTensorMap{T}(undef, V)
    else
        return TensorMap{T}(undef, V)
    end
end

function allocate_GR(bra::AbstractMPS, mpo::AbstractMPO, ket::AbstractMPS, i::Int)
    T = Base.promote_type(scalartype(bra), scalartype(mpo), scalartype(ket))
    V = right_virtualspace(ket, i) ⊗ right_virtualspace(mpo, i)' ←
        right_virtualspace(bra, i)
    if V isa BlockTensorKit.TensorMapSumSpace
        return BlockTensorMap{T}(undef, V)
    else
        return TensorMap{T}(undef, V)
    end
end

function allocate_GBL(bra::QP, mpo::AbstractMPO, ket::QP, i::Int)
    T = Base.promote_type(scalartype(bra), scalartype(mpo), scalartype(ket))
    V = left_virtualspace(bra.left_gs, i - 1) ⊗ left_virtualspace(mpo, i)' ←
        auxiliaryspace(ket)' ⊗ left_virtualspace(ket.left_gs, i - 1)
    if V isa BlockTensorKit.TensorMapSumSpace
        return BlockTensorMap{T}(undef, V)
    else
        return TensorMap{T}(undef, V)
    end
end

function allocate_GBR(bra::QP, mpo::AbstractMPO, ket::QP, i::Int)
    T = Base.promote_type(scalartype(bra), scalartype(mpo), scalartype(ket))
    V = right_virtualspace(ket.right_gs, i) ⊗ right_virtualspace(mpo, i)' ←
        auxiliaryspace(ket)' ⊗ right_virtualspace(bra.right_gs, i)
    if V isa BlockTensorKit.TensorMapSumSpace
        return BlockTensorMap{T}(undef, V)
    else
        return TensorMap{T}(undef, V)
    end
end
