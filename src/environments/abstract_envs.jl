"""
    abstract type AbstractEnvironments end

Abstract supertype for all environment types.
"""
abstract type AbstractMPSEnvironments end

# Locking
# -------
Base.lock(f::Function, envs::AbstractMPSEnvironments) = lock(f, envs.lock)
Base.lock(envs::AbstractMPSEnvironments) = lock(envs.lock)
Base.unlock(envs::AbstractMPSEnvironments) = unlock(envs.lock);

# Allocating tensors
# ------------------

function allocate_GL(bra::AbstractMPS, mpo::AbstractMPO, ket::AbstractMPS, i::Int)
    T = Base.promote_type(scalartype(bra), scalartype(mpo), scalartype(ket))
    V = left_virtualspace(bra, i) ⊗ left_virtualspace(mpo, i)' ←
        left_virtualspace(ket, i)
    if V isa BlockTensorKit.TensorMapSumSpace
        TT = blocktensormaptype(spacetype(bra), numout(V), numin(V), T)
    else
        TT = TensorMap{T}
    end
    return TT(undef, V)
end

function allocate_GR(bra::AbstractMPS, mpo::AbstractMPO, ket::AbstractMPS, i::Int)
    T = Base.promote_type(scalartype(bra), scalartype(mpo), scalartype(ket))
    V = right_virtualspace(ket, i) ⊗ right_virtualspace(mpo, i) ←
        right_virtualspace(bra, i)
    if V isa BlockTensorKit.TensorMapSumSpace
        TT = blocktensormaptype(spacetype(bra), numout(V), numin(V), T)
    else
        TT = TensorMap{T}
    end
    return TT(undef, V)
end

function allocate_GBL(bra::QP, mpo::AbstractMPO, ket::QP, i::Int)
    T = Base.promote_type(scalartype(bra), scalartype(mpo), scalartype(ket))
    V = left_virtualspace(bra, i) ⊗ left_virtualspace(mpo, i)' ←
        auxiliaryspace(ket)' ⊗ left_virtualspace(ket, i)
    if V isa BlockTensorKit.TensorMapSumSpace
        TT = blocktensormaptype(spacetype(bra), numout(V), numin(V), T)
    else
        TT = TensorMap{T}
    end
    return TT(undef, V)
end

function allocate_GBR(bra::QP, mpo::AbstractMPO, ket::QP, i::Int)
    T = Base.promote_type(scalartype(bra), scalartype(mpo), scalartype(ket))
    V = right_virtualspace(ket, i) ⊗ right_virtualspace(mpo, i) ←
        auxiliaryspace(ket)' ⊗ right_virtualspace(bra, i)
    if V isa BlockTensorKit.TensorMapSumSpace
        TT = blocktensormaptype(spacetype(bra), numout(V), numin(V), T)
    else
        TT = TensorMap{T}
    end
    return TT(undef, V)
end
