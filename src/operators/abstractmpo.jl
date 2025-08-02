# Matrix Product Operators
# ========================
"""
    abstract type AbstractMPO{O} <: AbstractVector{O} end

Abstract supertype for Matrix Product Operators (MPOs).
"""
abstract type AbstractMPO{O} <: AbstractVector{O} end

# useful union types
const SparseMPO{O <: SparseBlockTensorMap} = AbstractMPO{O}
Base.isfinite(O::AbstractMPO) = isfinite(typeof(O))

# By default, define things in terms of parent
Base.size(mpo::AbstractMPO, args...) = size(parent(mpo), args...)
Base.length(mpo::AbstractMPO) = length(parent(mpo))
eachsite(mpo::AbstractMPO) = eachindex(mpo)

@inline Base.getindex(mpo::AbstractMPO, i::Int) = getindex(parent(mpo), i)
@inline function Base.setindex!(mpo::AbstractMPO, value, i::Int)
    setindex!(parent(mpo), value, i)
    return mpo
end

# Properties
# ----------
left_virtualspace(mpo::AbstractMPO, site::Int) = left_virtualspace(mpo[site])
left_virtualspace(mpo::AbstractMPO) = map(Base.Fix1(left_virtualspace, mpo), eachsite(mpo))
right_virtualspace(mpo::AbstractMPO, site::Int) = right_virtualspace(mpo[site])
right_virtualspace(mpo::AbstractMPO) = map(Base.Fix1(right_virtualspace, mpo), eachsite(mpo))
physicalspace(mpo::AbstractMPO, site::Int) = physicalspace(mpo[site])
physicalspace(mpo::AbstractMPO) = map(Base.Fix1(physicalspace, mpo), eachsite(mpo))

for ftype in (:spacetype, :sectortype, :storagetype)
    @eval TensorKit.$ftype(mpo::AbstractMPO) = $ftype(typeof(mpo))
    @eval TensorKit.$ftype(::Type{MPO}) where {MPO <: AbstractMPO} = $ftype(eltype(MPO))
end

# Utility functions
# -----------------
remove_orphans!(mpo::AbstractMPO; tol = eps(real(scalartype(mpo)))^(3 / 4)) = mpo
function remove_orphans!(mpo::SparseMPO; tol = eps(real(scalartype(mpo)))^(3 / 4))
    droptol!.(mpo, tol)

    if isfinite(mpo)
        # Forward sweep
        # col j on site i empty -> remove row j on site i + 1
        for i in 1:(length(mpo) - 1)
            mask = filter(1:size(mpo[i], 4)) do j
                return j ∈ getindex.(nonzero_keys(mpo[i]), 4)
            end
            mpo[i] = mpo[i][:, :, :, mask]
            mpo[i + 1] = mpo[i + 1][mask, :, :, :]
        end

        # Backward sweep
        # row j on site i empty -> remove col j on site i - 1
        for i in length(mpo):-1:2
            mask = filter(1:size(mpo[i], 1)) do j
                return j ∈ getindex.(nonzero_keys(mpo[i]), 1)
            end
            mpo[i] = mpo[i][mask, :, :, :]
            mpo[i - 1] = mpo[i - 1][:, :, :, mask]
        end
    else
        # drop dead starts/ends
        changed = true
        while changed
            changed = false
            for i in 1:length(mpo)
                # slice empty columns on right or empty rows on left
                mask = filter(1:size(mpo[i], 4)) do j
                    return j ∈ getindex.(nonzero_keys(mpo[i]), 4) &&
                        j ∈ getindex.(nonzero_keys(mpo[i + 1]), 1)
                end
                changed |= length(mask) == size(mpo[i], 4)
                mpo[i] = mpo[i][:, :, :, mask]
                mpo[i + 1] = mpo[i + 1][mask, :, :, :]
            end
        end
    end

    return mpo
end

# Linear Algebra
# --------------
Base.:+(mpo::AbstractMPO) = scale(mpo, One())
Base.:-(mpo::AbstractMPO) = scale(mpo, -1)
Base.:-(mpo1::AbstractMPO, mpo2::AbstractMPO) = mpo1 + (-mpo2)

Base.:*(α::Number, mpo::AbstractMPO) = scale(mpo, α)
Base.:*(mpo::AbstractMPO, α::Number) = scale(mpo, α)
Base.:/(mpo::AbstractMPO, α::Number) = scale(mpo, inv(α))
Base.:\(α::Number, mpo::AbstractMPO) = scale(mpo, inv(α))

function VectorInterface.scale(mpo::AbstractMPO, α::Number)
    T = VectorInterface.promote_scale(scalartype(mpo), scalartype(α))
    dst = similar(mpo, T)
    return scale!(dst, mpo, α)
end

LinearAlgebra.norm(mpo::AbstractMPO) = sqrt(abs(dot(mpo, mpo)))

function Base.:(^)(a::AbstractMPO, n::Int)
    n >= 1 || throw(DomainError(n, "n should be a positive integer"))
    return Base.power_by_squaring(a, n)
end

Base.conj(mpo::AbstractMPO) = conj!(copy(mpo))
function Base.conj!(mpo::AbstractMPO)
    for i in 1:length(mpo)
        mpo[i] = _conj_mpo(mpo[i])
    end
    return mpo
end

function _conj_mpo(O::MPOTensor)
    return @plansor O′[-1 -2; -3 -4] := conj(O[-1 -3; -2 -4])
end

# Kernels
# -------
# TODO: diagram

function _fuse_mpo_mpo(O1::MPOTensor, O2::MPOTensor, Fₗ, Fᵣ)
    return if O1 isa BraidingTensor && O2 isa BraidingTensor
        # shouldn't happen
        T = promote_type(scalartype(O1), scalartype(O2))
        V = fuse(left_virtualspace(O2) ⊗ left_virtualspace(O1)) ⊗ physicalspace(O1) ←
            physicalspace(O2) ⊗ fuse(right_virtualspace(O2) ⊗ right_virtualspace(O1))
        return BraidingTensor{T}(V)
    elseif O1 isa BraidingTensor
        @plansor O′[-1 -2; -3 -4] := Fₗ[-1; 1 2] * O2[1 3; -3 5] *
            τ[2 -2; 3 4] * conj(Fᵣ[-4; 5 4])
    elseif O2 isa BraidingTensor
        @plansor O′[-1 -2; -3 -4] := Fₗ[-1; 1 2] * τ[1 3; -3 5] *
            O1[2 -2; 3 4] * conj(Fᵣ[-4; 5 4])
    else
        @plansor O′[-1 -2; -3 -4] := Fₗ[-1; 1 2] * O2[1 3; -3 5] *
            O1[2 -2; 3 4] * conj(Fᵣ[-4; 5 4])
    end
end

"""
    fuse_mul_mpo(O1, O2)

Compute the mpo tensor that arises from multiplying MPOs.
"""
function fuse_mul_mpo(O1, O2)
    T = promote_type(scalartype(O1), scalartype(O2))
    F_left = fuser(T, left_virtualspace(O2), left_virtualspace(O1))
    F_right = fuser(T, right_virtualspace(O2), right_virtualspace(O1))
    return _fuse_mpo_mpo(O1, O2, F_left, F_right)
end
function fuse_mul_mpo(O1::BraidingTensor, O2::BraidingTensor)
    T = promote_type(scalartype(O1), scalartype(O2))
    V = fuse(left_virtualspace(O2) ⊗ left_virtualspace(O1)) ⊗ physicalspace(O1) ←
        physicalspace(O2) ⊗ fuse(right_virtualspace(O2) ⊗ right_virtualspace(O1))
    return BraidingTensor{T}(V)
end
function fuse_mul_mpo(
        O1::AbstractBlockTensorMap{T₁, S, 2, 2}, O2::AbstractBlockTensorMap{T₂, S, 2, 2}
    ) where {T₁, T₂, S}
    TT = promote_type((eltype(O1)), eltype((O2)))
    V = fuse(left_virtualspace(O2) ⊗ left_virtualspace(O1)) ⊗ physicalspace(O1) ←
        physicalspace(O2) ⊗ fuse(right_virtualspace(O2) ⊗ right_virtualspace(O1))
    if BlockTensorKit.issparse(O1) && BlockTensorKit.issparse(O2)
        O = SparseBlockTensorMap{TT}(undef, V)
    else
        O = BlockTensorMap{TT}(undef, V)
    end
    cartesian_inds = reshape(
        CartesianIndices(O),
        size(O2, 1), size(O1, 1), size(O, 2), size(O, 3), size(O2, 4), size(O1, 4)
    )
    for (I, o2) in nonzero_pairs(O2), (J, o1) in nonzero_pairs(O1)
        K = cartesian_inds[I[1], J[1], I[2], I[3], I[4], J[4]]
        O[K] = fuse_mul_mpo(o1, o2)
    end
    return O
end

function add_physical_charge(O::MPOTensor, charge::Sector)
    sectortype(O) === typeof(charge) || throw(SectorMismatch())
    auxspace = Vect[typeof(charge)](charge => 1)'
    F = fuser(scalartype(O), physicalspace(O), auxspace)
    @plansor O_charged[-1 -2; -3 -4] := F[-2; 1 2] *
        O[-1 1; 4 3] * τ[3 2; 5 -4] * conj(F[-3; 4 5])
    return O_charged
end
function add_physical_charge(O::BraidingTensor, charge::Sector)
    sectortype(O) === typeof(charge) || throw(SectorMismatch())
    auxspace = Vect[typeof(charge)](charge => 1)'
    V = left_virtualspace(O) ⊗ fuse(physicalspace(O), auxspace) ←
        fuse(physicalspace(O), auxspace) ⊗ right_virtualspace(O)
    return BraidingTensor{scalartype(O)}(V)
end
function add_physical_charge(O::AbstractBlockTensorMap{<:Any, <:Any, 2, 2}, charge::Sector)
    sectortype(O) == typeof(charge) || throw(SectorMismatch())
    auxspace = Vect[typeof(charge)](charge => 1)'
    Odst = similar(
        O,
        left_virtualspace(O) ⊗ fuse(physicalspace(O), auxspace) ←
            fuse(physicalspace(O), auxspace) ⊗ right_virtualspace(O)
    )
    for (I, v) in nonzero_pairs(O)
        Odst[I] = add_physical_charge(v, charge)
    end
    return Odst
end

# Contractions
# ------------
# This function usually does not require to be specified for many N, so @generated function is fine?
@generated function _instantiate_finitempo(
        L::AbstractTensorMap{<:Any, S, 1, 2},
        O::NTuple{N, MPOTensor{S}},
        R::AbstractTensorMap{<:Any, S, 2, 1}
    ) where {N, S}
    sites = N + 2
    t_out = tensorexpr(:T, -(1:sites), -(1:sites) .- sites)
    t_left = tensorexpr(:L, -1, (-1 - sites, 1))
    t_mid = ntuple(N) do n
        return tensorexpr(:(O[$n]), (n, -n - 1), (-n - sites - 1, n + 1))
    end
    t_right = tensorexpr(:R, (sites - 1, -sites), -2sites)
    ex = :(@plansor $t_out ≔ *($t_left, $t_right, $(t_mid...)))
    return macroexpand(@__MODULE__, ex)
end

@generated function _apply_finitempo(
        x::AbstractTensorMap{<:Any, S, M, A},
        L::AbstractTensorMap{<:Any, S, 1, 2},
        O::NTuple{N, MPOTensor{S}},
        R::AbstractTensorMap{<:Any, S, 2, 1}
    ) where {N, M, S, A}
    M == N + 2 || throw(ArgumentError("Incompatible number of spaces"))
    t_out = tensorexpr(:y, -(1:M), -(1:A) .- M)
    t_in = tensorexpr(:x, 1:2:(2M - 1), -(1:A) .- M)
    t_left = tensorexpr(:L, -1, (1, 2))
    t_mid = ntuple(N) do n
        return tensorexpr(:(O[$n]), (2n, -n - 1), (2n + 1, 2n + 2))
    end
    t_right = tensorexpr(:R, (2N + 2, -M), 2N + 3)
    ex = :(@plansor $t_out ≔ *($t_in, $t_left, $t_right, $(t_mid...)))
    return macroexpand(@__MODULE__, ex)
end
