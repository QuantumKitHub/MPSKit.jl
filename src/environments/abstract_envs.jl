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

# Abstract Infinite Environments
# ------------------------------
"""
    AbstractInfiniteEnvironments <: AbstractEnvironments

Abstract supertype for infinite environment managers.
"""
abstract type AbstractInfiniteEnvironments <: AbstractMPSEnvironments end

leftenv(envs, pos::CartesianIndex, state) = leftenv(envs, Tuple(pos)..., state)
rightenv(envs, pos::CartesianIndex, state) = rightenv(envs, Tuple(pos)..., state)

# recalculate logic
# -----------------
function check_recalculate!(envs::AbstractInfiniteEnvironments, state)
    # check if dependency got updated - cheap test to avoid having to lock
    if !check_dependency(envs, state)
        # acquire lock and check again (might have updated while waiting)
        lock(envs) do
            return check_dependency(envs, state) || recalculate!(envs, state)
        end
    end
    return envs
end

function recalculate!(envs::AbstractInfiniteEnvironments, state; tol=envs.solver.tol)
    # check if the virtual spaces have changed and reallocate if necessary
    if !issamespace(envs, state)
        envs.leftenvs, envs.rightenvs = initialize_environments(state, envs.operator)
    end

    solver = envs.solver
    envs.solver = solver.tol == tol ? solver : @set solver.tol = tol
    envs.dependency = state

    @sync begin
        Threads.@spawn compute_leftenv!(envs)
        Threads.@spawn compute_rightenv!(envs)
    end
    normalize!(envs)

    return envs
end
