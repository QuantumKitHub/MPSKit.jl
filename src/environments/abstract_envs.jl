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
    M = TensorKit.promote_storagetype(T, eltype(mpo), eltype(bra), eltype(ket))
    S = TensorKit.check_spacetype(bra, mpo, ket)
    V = left_virtualspace(bra, i) ⊗ left_virtualspace(mpo, i)' ←
        left_virtualspace(ket, i)
    if V isa BlockTensorKit.TensorMapSumSpace
        TT = blocktensormaptype(S, numout(V), numin(V), M)
    else
        TT = tensormaptype(S, numout(V), numin(V), M)
    end
    return TT(undef, V)
end

function allocate_GR(bra::AbstractMPS, mpo::AbstractMPO, ket::AbstractMPS, i::Int)
    T = Base.promote_type(scalartype(bra), scalartype(mpo), scalartype(ket))
    M = TensorKit.promote_storagetype(T, eltype(mpo), eltype(bra), eltype(ket))
    S = TensorKit.check_spacetype(bra, mpo, ket)
    V = right_virtualspace(ket, i) ⊗ right_virtualspace(mpo, i) ←
        right_virtualspace(bra, i)
    if V isa BlockTensorKit.TensorMapSumSpace
        TT = blocktensormaptype(S, numout(V), numin(V), M)
    else
        TT = tensormaptype(S, numout(V), numin(V), M)
    end
    return TT(undef, V)
end

function allocate_GBL(bra::QP, mpo::AbstractMPO, ket::QP, i::Int)
    T = Base.promote_type(scalartype(bra), scalartype(mpo), scalartype(ket))
    M = TensorKit.promote_storagetype(T, eltype(mpo), eltype(bra), eltype(ket))
    S = TensorKit.check_spacetype(bra, mpo, ket)
    V = left_virtualspace(bra.left_gs, i) ⊗ left_virtualspace(mpo, i)' ←
        auxiliaryspace(ket)' ⊗ left_virtualspace(ket.right_gs, i)
    if V isa BlockTensorKit.TensorMapSumSpace
        TT = blocktensormaptype(S, numout(V), numin(V), M)
    else
        TT = tensormaptype(S, numout(V), numin(V), M)
    end
    return TT(undef, V)
end

function allocate_GBR(bra::QP, mpo::AbstractMPO, ket::QP, i::Int)
    T = Base.promote_type(scalartype(bra), scalartype(mpo), scalartype(ket))
    M = TensorKit.promote_storagetype(T, eltype(mpo), eltype(bra), eltype(ket))
    S = TensorKit.check_spacetype(bra, mpo, ket)
    V = right_virtualspace(ket.left_gs, i) ⊗ right_virtualspace(mpo, i) ←
        auxiliaryspace(ket)' ⊗ right_virtualspace(bra.right_gs, i)
    if V isa BlockTensorKit.TensorMapSumSpace
        TT = blocktensormaptype(S, numout(V), numin(V), M)
    else
        TT = tensormaptype(S, numout(V), numin(V), M)
    end
    return TT(undef, V)
end

# Environment algorithms
# ----------------------
"""
    environment_alg(below, operator, above; kwargs...)

Determine an appropriate algorithm for computing the environments, based on the given `kwargs...`.
"""
function environment_alg(
        ::Union{InfiniteMPS, MultilineMPS}, ::Union{InfiniteMPO, MultilineMPO},
        ::Union{InfiniteMPS, MultilineMPS};
        tol = Defaults.tol, maxiter = Defaults.maxiter, krylovdim = Defaults.krylovdim,
        verbosity = Defaults.VERBOSE_NONE, eager = true
    )
    return Arnoldi(; tol, maxiter, krylovdim, verbosity, eager)
end
function environment_alg(
        below, ::InfiniteMPOHamiltonian, above;
        tol = Defaults.tol, maxiter = Defaults.maxiter, krylovdim = Defaults.krylovdim,
        verbosity = Defaults.VERBOSE_NONE
    )
    max_krylovdim = ceil(Int, dim(left_virtualspace(above, 1)) * dim(left_virtualspace(below, 1)))
    return GMRES(; tol, maxiter, krylovdim = min(max_krylovdim, krylovdim), verbosity)
end
function environment_alg(
        ::Union{InfiniteQP, MultilineQP}, ::Union{InfiniteMPO, MultilineMPO},
        ::Union{InfiniteQP, MultilineQP};
        tol = Defaults.tol, maxiter = Defaults.maxiter, krylovdim = Defaults.krylovdim,
        verbosity = Defaults.VERBOSE_NONE
    )
    return GMRES(; tol, maxiter, krylovdim, verbosity)
end
