"""
    JordanMPOTensor{E,S,TA,TB,TC,TD} <: AbstractBlockTensorMap{E,S,2,2}

A tensor map that represents the upper triangular block form of a matrix product operator (MPO).

```math
\\begin{pmatrix}
1 & C & D \\\\
0 & A & B \\\\
0 & 0 & 1
\\end{pmatrix}
```
"""
struct JordanMPOTensor{
        E, S,
        TA <: AbstractTensorMap{E, S, 2, 2},
        TB <: AbstractTensorMap{E, S, 2, 1},
        TC <: AbstractTensorMap{E, S, 1, 2},
        TD <: AbstractTensorMap{E, S, 1, 1},
    } <: AbstractBlockTensorMap{E, S, 2, 2}
    V::TensorMapSumSpace{S, 2, 2}
    A::SparseBlockTensorMap{TA, E, S, 2, 2, 4}
    B::SparseBlockTensorMap{TB, E, S, 2, 1, 3}
    C::SparseBlockTensorMap{TC, E, S, 1, 2, 3}
    D::SparseBlockTensorMap{TD, E, S, 1, 1, 2}
    # uninitialized constructor
    function JordanMPOTensor{E, S, TA, TB, TC, TD}(
            ::UndefInitializer, V::TensorMapSumSpace{S, 2, 2}
        ) where {E, S, TA, TB, TC, TD}
        allVs = eachspace(V)

        # Note that this is a bit of a hack using end to get the last index:
        # it should be 1 or end depending on this being an "edge" tensor or a "bulk" tensor
        VA = space(allVs[2:(end - 1), 1, 1, 2:(end - 1)])
        A = SparseBlockTensorMap{TA}(undef, VA)

        VB = removeunit(space(allVs[2:(end - 1), 1, 1, end]), 4)
        B = SparseBlockTensorMap{TB}(undef, VB)

        VC = removeunit(space(allVs[1, 1, 1, 2:(end - 1)]), 1)
        C = SparseBlockTensorMap{TC}(undef, VC)

        VD = removeunit(removeunit(space(allVs[1, 1, 1, end:end]), 4), 1)
        D = SparseBlockTensorMap{TD}(undef, VD)

        return new{E, S, TA, TB, TC, TD}(V, A, B, C, D)
    end

    # constructor from data
    function JordanMPOTensor{E, S, TA, TB, TC, TD}(
            V::TensorMapSumSpace,
            A::SparseBlockTensorMap{TA, E, S, 2, 2},
            B::SparseBlockTensorMap{TB, E, S, 2, 1},
            C::SparseBlockTensorMap{TC, E, S, 1, 2},
            D::SparseBlockTensorMap{TD, E, S, 1, 1}
        ) where {E, S, TA, TB, TC, TD}
        return new{E, S, TA, TB, TC, TD}(V, A, B, C, D)
    end
end

const JordanMPOTensorMap{T, S, A <: DenseVector{T}} = JordanMPOTensor{
    T, S,
    Union{TensorMap{T, S, 2, 2, A}, BraidingTensor{T, S}},
    TensorMap{T, S, 2, 1, A},
    TensorMap{T, S, 1, 2, A},
    TensorMap{T, S, 1, 1, A},
}

function JordanMPOTensor{E, S}(::UndefInitializer, V::TensorMapSumSpace{S}) where {E, S}
    return jordanmpotensortype(S, E)(undef, V)
end
function JordanMPOTensor{E}(::UndefInitializer, V::TensorMapSumSpace{S}) where {E, S}
    return JordanMPOTensor{E, S}(undef, V)
end

function JordanMPOTensor(
        V::TensorMapSumSpace{S, 2, 2},
        A::SparseBlockTensorMap{TA, E, S, 2, 2}, B::SparseBlockTensorMap{TB, E, S, 2, 1},
        C::SparseBlockTensorMap{TC, E, S, 1, 2}, D::SparseBlockTensorMap{TD, E, S, 1, 1}
    ) where {E, S, TA, TB, TC, TD}
    allVs = eachspace(V)
    VA = space(allVs[2:(end - 1), 1, 1, 2:(end - 1)])
    VA == space(A) || throw(SpaceMismatch(lazy"A-block has incompatible spaces:\n$VA\n$(space(A))"))

    VB = removeunit(space(allVs[2:(end - 1), 1, 1, end]), 4)
    VB == space(B) || throw(SpaceMismatch(lazy"B-block has incompatible spaces:\n$VB\n$(space(B))"))

    VC = removeunit(space(allVs[1, 1, 1, 2:(end - 1)]), 1)
    VC == space(C) || throw(SpaceMismatch(lazy"C-block has incompatible spaces:\n$VC\n$(space(C))"))

    VD = removeunit(removeunit(space(allVs[1, 1, 1, end:end]), 4), 1)
    VD == space(D) || throw(SpaceMismatch(lazy"D-block has incompatible spaces:\n$VD\n$(space(D))"))

    return JordanMPOTensor{E, S, TA, TB, TC, TD}(V, A, B, C, D)
end
function JordanMPOTensor(
        V::TensorMapSumSpace{S, 2, 2},
        A::AbstractTensorMap{E, S, 2, 2}, B::AbstractTensorMap{E, S, 2, 1},
        C::AbstractTensorMap{E, S, 1, 2}, D::AbstractTensorMap{E, S, 1, 1}
    ) where {E, S}
    return JordanMPOTensor(
        V,
        A isa SparseBlockTensorMap ? A : SparseBlockTensorMap(A),
        B isa SparseBlockTensorMap ? B : SparseBlockTensorMap(B),
        C isa SparseBlockTensorMap ? C : SparseBlockTensorMap(C),
        D isa SparseBlockTensorMap ? D : SparseBlockTensorMap(D)
    )
end

function JordanMPOTensor(W::SparseBlockTensorMap{TT, E, S, 2, 2}) where {TT, E, S}
    @assert W[1, 1, 1, 1] isa BraidingTensor && W[end, 1, 1, end] isa BraidingTensor
    # @assert all(I -> I[1] ≤ I[4], nonzero_keys(W))

    A = W[2:(end - 1), 1, 1, 2:(end - 1)]
    B = W[2:(end - 1), 1, 1, end]
    C = W[1, 1, 1, 2:(end - 1)]
    D = W[1, 1, 1, end:end] # ensure still blocktensor to allow for sparse

    return JordanMPOTensor(
        space(W), A, removeunit(B, 4), removeunit(C, 1), removeunit(removeunit(D, 4), 1)
    )
end

function jordanmpotensortype(::Type{S}, ::Type{TorA}) where {S <: VectorSpace, TorA}
    TA = Union{tensormaptype(S, 2, 2, TorA), BraidingTensor{scalartype(TorA), S}}
    TB = tensormaptype(S, 2, 1, TorA)
    TC = tensormaptype(S, 1, 2, TorA)
    TD = tensormaptype(S, 1, 1, TorA)
    return JordanMPOTensor{scalartype(TorA), S, TA, TB, TC, TD}
end
function jordanmpotensortype(::Type{O}) where {O <: MPOTensor}
    return jordanmpotensortype(spacetype(O), scalartype(O))
end

function Base.similar(W::JordanMPOTensor, ::Type{T}) where {T <: Number}
    return JordanMPOTensor{T}(undef, space(W))
end

# Properties
# ----------
TensorKit.space(W::JordanMPOTensor) = W.V
Base.eltype(::Type{JordanMPOTensor{E, S, TA, TB, TC, TD}}) where {E, S, TA, TB, TC, TD} = TA

function Base.haskey(W::JordanMPOTensor, I::CartesianIndex{4})
    Base.checkbounds(W, I.I...)
    # only has braiding tensors if sizes are large enough
    sz = size(W)
    (
        sz[1] > 1 && I == CartesianIndex(1, 1, 1, 1) ||
            sz[4] > 1 && I == CartesianIndex(sz[1], 1, 1, sz[4])
    ) && return true

    row, col = I.I[1], I.I[4]

    if row == 1 && col == sz[4]
        return haskey(W.D, CartesianIndex(1, 1))
    elseif row == 1
        return haskey(W.C, CartesianIndex(1, 1, col - 1))
    elseif col == sz[4]
        return haskey(W.B, CartesianIndex(row - 1, 1, 1))
    elseif 1 < row < sz[1] && 1 < col < sz[4]
        return haskey(W.A, CartesianIndex(row - 1, 1, 1, col - 1))
    else
        return false
    end
end

Base.parent(W::JordanMPOTensor) = parent(SparseBlockTensorMap(W))

BlockTensorKit.issparse(W::JordanMPOTensor) = true

# Converters
# ----------
function BlockTensorKit.SparseBlockTensorMap(W::JordanMPOTensor)
    τ = BraidingTensor{scalartype(W)}(eachspace(W)[1])
    W′ = SparseBlockTensorMap{AbstractTensorMap{scalartype(W), spacetype(W), 2, 2}}(
        undef_blocks, space(W)
    )
    if size(W, 4) > 1
        W′[1, 1, 1, 1] = τ
    end
    if size(W, 1) > 1
        W′[end, 1, 1, end] = τ
    end

    Ia = CartesianIndex(1, 0, 0, 1)
    for (I, v) in nonzero_pairs(W.A)
        W′[I + Ia] = v
    end
    Ib = CartesianIndex(1, 0, 0)
    for (I, v) in nonzero_pairs(W.B)
        W′[I + Ib, size(W′, 4)] = insertrightunit(v, 3)
    end
    Ic = CartesianIndex(0, 0, 1)
    for (I, v) in nonzero_pairs(W.C)
        W′[1, I + Ic] = insertleftunit(v, 1)
    end
    if nonzero_length(W.D) > 0
        W′[1, 1, 1, end] = insertrightunit(insertleftunit(only(W.D), 1), 3)
    end

    return W′
end

for f in (:real, :complex)
    @eval function Base.$f(W::JordanMPOTensor)
        E = $f(scalartype(W))
        W′ = JordanMPOTensor{E}(undef, space(W))
        for (I, v) in nonzero_pairs(W.A)
            W′.A[I] = $f(v)
        end
        for (I, v) in nonzero_pairs(W.B)
            W′.B[I] = $f(v)
        end
        for (I, v) in nonzero_pairs(W.C)
            W′.C[I] = $f(v)
        end
        for (I, v) in nonzero_pairs(W.D)
            W′.D[I] = $f(v)
        end
        return W′
    end
end

# Indexing
# --------

@inline Base.getindex(W::JordanMPOTensor, I::CartesianIndex{4}) = W[I.I...]
@propagate_inbounds function Base.getindex(W::JordanMPOTensor, I::Vararg{Int, 4})
    @assert I[2] == I[3] == 1
    i = I[1]
    j = I[4]
    if (size(W, 4) > 1 && i == 1 && j == 1) ||
            (size(W, 1) > 1 && i == size(W, 1) && j == size(W, 4))
        return BraidingTensor{scalartype(W)}(eachspace(W)[1])
    elseif i == 1 && j == size(W, 4)
        return insertrightunit(insertleftunit(only(W.D), 1), 3)
    elseif i == 1
        return insertleftunit(W.C[1, 1, j - 1], 1)
    elseif j == size(W, 4)
        return insertrightunit(W.B[i - 1, 1, 1], 3)
    elseif 1 < i < size(W, 1) && 1 < j < size(W, 4)
        return W.A[i - 1, 1, 1, j - 1]
    else
        return zeros(scalartype(W), eachspace(W)[i, 1, 1, j])
    end
end

@inline function Base.setindex!(W::JordanMPOTensor, v::MPOTensor, I::CartesianIndex{4})
    return setindex!(W, v, I.I...)
end
@propagate_inbounds function Base.setindex!(
        W::JordanMPOTensor, v::MPOTensor, I::Vararg{Int, 4}
    )
    @assert I[2] == I[3] == 1
    i = I[1]
    j = I[4]
    if i == 1 && j == size(W, 4)
        W.D[1] = removeunit(removeunit(v, 4), 1)
    elseif i == 1 && 1 < j < size(W, 4)
        W.C[1, 1, j - 1] = removeunit(v, 1)
    elseif j == size(W, 4) && 1 < i < size(W, 1)
        W.B[i - 1, 1, 1] = removeunit(v, 4)
    elseif 1 < i < size(W, 1) && 1 < j < size(W, 4)
        W.A[i - 1, 1, 1, j - 1] = v
    elseif (size(W, 4) > 1 && i == 1 && j == 1) ||
            (size(W, 1) > 1 && i == size(W, 1) && j == size(W, 4))
        v isa BraidingTensor || throw(ArgumentError("Cannot set BraidingTensor"))
    else
        throw(ArgumentError("Cannot set index ($i, 1, 1, $j)"))
    end
    return W
end
@inline function Base.setindex!(W::JordanMPOTensor, v::MPOTensor, I::Int)
    return setindex!(W, v, CartesianIndices(W)[I])
end

# Sparse functionality
# --------------------
function BlockTensorKit.nonzero_keys(W::JordanMPOTensor)
    nrows = size(W, 1)
    ncols = size(W, 4)
    p = CartesianIndex{4}[]
    ncols > 1 && push!(p, CartesianIndex(1, 1, 1, 1))
    nrows > 1 && push!(p, CartesianIndex(nrows, 1, 1, ncols))

    Ia = CartesianIndex(1, 0, 0, 1)
    for I in nonzero_keys(W.A)
        push!(p, I + Ia)
    end

    Ib = CartesianIndex(1, 0, 0)
    for I in nonzero_keys(W.B)
        push!(p, CartesianIndex((I + Ib).I..., ncols))
    end

    Ic = CartesianIndex(0, 0, 1)
    for I in nonzero_keys(W.C)
        push!(p, CartesianIndex(1, (I + Ic).I...))
    end

    for I in nonzero_keys(W.D)
        push!(p, CartesianIndex(1, 1, 1, ncols))
    end

    return p
end
function BlockTensorKit.nonzero_values(W::JordanMPOTensor)
    return Iterators.map(I -> W[I], nonzero_keys(W))
end
function BlockTensorKit.nonzero_pairs(W::JordanMPOTensor)
    return Iterators.map(I -> I => W[I], nonzero_keys(W))
end
function BlockTensorKit.nonzero_length(W::JordanMPOTensor)
    return nonzero_length(W.A) + nonzero_length(W.B) + nonzero_length(W.C) +
        nonzero_length(W.D) + Int(size(W, 1) > 1) + Int(size(W, 4) > 1)
end

# linalg
# ------
# do we want this?
function Base.:+(W1::JordanMPOTensor, W2::JordanMPOTensor)
    return SparseBlockTensorMap(W1) + SparseBlockTensorMap(W2)
end
function Base.:-(W1::JordanMPOTensor, W2::JordanMPOTensor)
    return SparseBlockTensorMap(W1) - SparseBlockTensorMap(W2)
end

function fuse_mul_mpo(O1::JordanMPOTensor, O2::JordanMPOTensor)
    TT = promote_type((eltype(O1)), eltype((O2)))
    V = fuse(left_virtualspace(O2) ⊗ left_virtualspace(O1)) ⊗ physicalspace(O1) ←
        physicalspace(O2) ⊗ fuse(right_virtualspace(O2) ⊗ right_virtualspace(O1))
    O = jordanmpotensortype(TT)(undef, V)
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

function _conj_mpo(W::JordanMPOTensor)
    V = left_virtualspace(W)' ⊗ physicalspace(W) ← physicalspace(W) ⊗ right_virtualspace(W)'
    A = _conj_mpo(W.A)
    @plansor B[-1 -2; -3] ≔ conj(W.B[-1 -3; -2])
    @plansor C[-1; -2 -3] ≔ conj(W.C[-2; -1 -3])
    D = copy(adjoint(W.D))
    return JordanMPOTensor(V, A, B, C, D)
end

function add_physical_charge(O::JordanMPOTensor, charge::Sector)
    sectortype(O) == typeof(charge) || throw(SectorMismatch())
    auxspace = Vect[typeof(charge)](charge => 1)'
    Vdst = left_virtualspace(O) ⊗
        fuse(physicalspace(O), auxspace) ←
        fuse(physicalspace(O), auxspace) ⊗ right_virtualspace(O)
    Odst = JordanMPOTensor{scalartype(O)}(undef, Vdst)
    for (I, v) in nonzero_pairs(O)
        Odst[I] = add_physical_charge(v, charge)
    end
    return Odst
end

# Utility
# -------
function Base.copy(W::JordanMPOTensor)
    return JordanMPOTensor(W.V, copy(W.A), copy(W.B), copy(W.C), copy(W.D))
end
function Base.copy!(Wdst::JordanMPOTensor, Wsrc::JordanMPOTensor)
    space(Wdst) == space(Wsrc) || throw(SpaceMismatch())
    copy!(Wdst.A, Wsrc.A)
    copy!(Wdst.B, Wsrc.B)
    copy!(Wdst.C, Wsrc.C)
    copy!(Wdst.D, Wsrc.D)
    return Wdst
end

# Avoid falling back to `norm(W1 - W2)` which has to convert to SparseBlockTensorMap
function Base.isapprox(W1::JordanMPOTensor, W2::JordanMPOTensor; kwargs...)
    return isapprox(W1.A, W2.A; kwargs...) &&
        isapprox(W1.B, W2.B; kwargs...) &&
        isapprox(W1.C, W2.C; kwargs...) &&
        isapprox(W1.D, W2.D; kwargs...)
end

function Base.showarg(io::IO, W::JordanMPOTensor, toplevel::Bool)
    !toplevel && print(io, "::")
    print(io, TensorKit.type_repr(typeof(W)))
    return nothing
end

function TensorKit.type_repr(::Type{<:JordanMPOTensor{E, S}}) where {E, S}
    return "JordanMPOTensor{$E, " * TensorKit.type_repr(S) * ", …}"
end
