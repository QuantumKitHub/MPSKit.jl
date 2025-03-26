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
struct JordanMPOTensor{E,S,
                       TA<:AbstractTensorMap{E,S,2,2},
                       TB<:AbstractTensorMap{E,S,2,1},
                       TC<:AbstractTensorMap{E,S,1,2},
                       TD<:AbstractTensorMap{E,S,1,1}} <: AbstractBlockTensorMap{E,S,2,2}
    V::TensorMapSumSpace{S,2,2}
    A::SparseBlockTensorMap{TA,E,S,2,2,4}
    B::SparseBlockTensorMap{TB,E,S,2,1,3}
    C::SparseBlockTensorMap{TC,E,S,1,2,3}
    D::SparseBlockTensorMap{TD,E,S,1,1,2}
    # uninitialized constructor
    function JordanMPOTensor{E,S,TA,TB,TC,TD}(::UndefInitializer,
                                              V::TensorMapSumSpace{S,2,2}) where {E,S,TA,TB,
                                                                                  TC,TD}
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

        return new{E,S,TA,TB,TC,TD}(V, A, B, C, D)
    end

    # constructor from data
    function JordanMPOTensor{E,S,TA,TB,TC,TD}(V::TensorMapSumSpace,
                                              A::SparseBlockTensorMap{TA,E,S,2,2},
                                              B::SparseBlockTensorMap{TB,E,S,2,1},
                                              C::SparseBlockTensorMap{TC,E,S,1,2},
                                              D::SparseBlockTensorMap{TD,E,S,1,1}) where {E,
                                                                                          S,
                                                                                          TA,
                                                                                          TB,
                                                                                          TC,
                                                                                          TD}
        # TODO: add space and size checks
        return new{E,S,TA,TB,TC,TD}(V, A, B, C, D)
    end
end

function JordanMPOTensor{E,S}(::UndefInitializer, V::TensorMapSumSpace{S}) where {E,S}
    return jordanmpotensortype(S, E)(undef, V)
end
function JordanMPOTensor{E}(::UndefInitializer, V::TensorMapSumSpace{S}) where {E,S}
    return JordanMPOTensor{E,S}(undef, V)
end

function JordanMPOTensor(V::TensorMapSumSpace{S,2,2},
                         A::SparseBlockTensorMap{TA,E,S,2,2},
                         B::SparseBlockTensorMap{TB,E,S,2,1},
                         C::SparseBlockTensorMap{TC,E,S,1,2},
                         D::SparseBlockTensorMap{TD,E,S,1,1}) where {E,S,TA,TB,TC,TD}
    return JordanMPOTensor{E,S,TA,TB,TC,TD}(V, A, B, C, D)
end

function JordanMPOTensor(W::SparseBlockTensorMap{TT,E,S,2,2}) where {TT,E,S}
    @assert W[1, 1, 1, 1] isa BraidingTensor && W[end, 1, 1, end] isa BraidingTensor
    # @assert all(I -> I[1] ≤ I[4], nonzero_keys(W))

    A = W[2:(end - 1), 1, 1, 2:(end - 1)]
    B = W[2:(end - 1), 1, 1, end]
    C = W[1, 1, 1, 2:(end - 1)]
    D = W[1, 1, 1, end:end] # ensure still blocktensor to allow for sparse

    return JordanMPOTensor(space(W),
                           A,
                           removeunit(B, 4),
                           removeunit(C, 1),
                           removeunit(removeunit(D, 4), 1))
end

function jordanmpotensortype(::Type{S}, ::Type{E}) where {S<:VectorSpace,E<:Number}
    TA = Union{tensormaptype(S, 2, 2, E),BraidingTensor{E,S}}
    TB = tensormaptype(S, 2, 1, E)
    TC = tensormaptype(S, 1, 2, E)
    TD = tensormaptype(S, 1, 1, E)
    return JordanMPOTensor{E,S,TA,TB,TC,TD}
end

function Base.similar(W::JordanMPOTensor, ::Type{T}) where {T<:Number}
    return JordanMPOTensor{T}(undef, space(W))
end

# Properties
# ----------
TensorKit.space(W::JordanMPOTensor) = W.V

Base.size(W::JordanMPOTensor) = size(eachspace(W))
Base.size(W::JordanMPOTensor, i::Int) = i ≤ 4 ? size(W)[i] : 1
Base.length(W::JordanMPOTensor) = prod(size(W))
Base.axes(W::JordanMPOTensor) = map(Base.OneTo, size(W))

Base.firstindex(W::JordanMPOTensor) = 1
Base.firstindex(::JordanMPOTensor, i::Int) = i ≤ 4 ? 1 : 1
Base.lastindex(W::JordanMPOTensor) = prod(size(W))
Base.lastindex(W::JordanMPOTensor, i::Int) = i ≤ 4 ? size(W, i) : 1

Base.CartesianIndices(W::JordanMPOTensor) = CartesianIndices(size(W))
Base.eachindex(W::JordanMPOTensor) = eachindex(size(W))

Base.eltype(W::JordanMPOTensor) = eltype(typeof(W))
Base.eltype(::Type{JordanMPOTensor{E,S,TA,TB,TC,TD}}) where {E,S,TA,TB,TC,TD} = TA

function Base.haskey(W::JordanMPOTensor, I::CartesianIndex{4})
    Base.checkbounds(W, I.I...)
    # only has braiding tensors if sizes are large enough
    sz = size(W)
    (sz[1] > 1 && I == CartesianIndex(1, 1, 1, 1) ||
     sz[4] > 1 && I == CartesianIndex(sz[1], 1, 1, sz[4])) && return true

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

# TODO: avoid this slow fallback wherever possible:
Base.parent(W::JordanMPOTensor) = (parent(SparseBlockTensorMap(W)))

BlockTensorKit.issparse(W::JordanMPOTensor) = true

# Converters
# ----------
function SparseBlockTensorMap(W::JordanMPOTensor)
    τ = BraidingTensor{scalartype(W)}(eachspace(W)[1])
    W′ = SparseBlockTensorMap{AbstractTensorMap{scalartype(W),spacetype(W),2,2}}(undef_blocks,
                                                                                 space(W))
    if size(W, 1) > 1
        W′[1, 1, 1, 1] = τ
    end
    if size(W, 4) > 1
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
    W′[1, 1, 1, end] = insertrightunit(insertleftunit(only(W.D), 1), 3)

    return W′
end

# Indexing
# --------
@inline function Base.checkbounds(W::JordanMPOTensor, I...)
    checkbounds(Bool, W, I...) || Base.throw_boundserror(W, I)
    return nothing
end
@inline function Base.checkbounds(::Type{Bool}, W::JordanMPOTensor, I...)
    return Base.checkbounds_indices(Bool, axes(W), I)
end

@inline Base.getindex(W::JordanMPOTensor, I::CartesianIndex{4}) = W[I.I...]
@propagate_inbounds function Base.getindex(W::JordanMPOTensor, I::Vararg{Int,4})
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

# TODO: this is definitely suboptimal, but might not matter?
@propagate_inbounds function Base.getindex(W::JordanMPOTensor,
                                           inds::Vararg{BlockTensorKit.SliceIndex,4})
    V = space(eachspace(W)[inds...])
    dst = similar(W, V)

    Rsrc = CartesianIndices(W)[inds...]
    Rdst = CartesianIndices(dst)

    for I in nonzero_keys(W)
        j = @something findfirst(==(I), Rsrc) continue
        dst[Rdst[j]] = W[I]
    end

    return dst
end
@propagate_inbounds function Base.getindex(W::JordanMPOTensor,
                                           inds::Vararg{BlockTensorKit.Strided.SliceIndex,
                                                        4})
    V = space(eachspace(W)[inds...])
    dst = similar(W, V)

    # prevent discarding of singleton dimensions
    inds2 = map(inds) do ind
        return ind isa Int ? (ind:ind) : ind
    end
    Rsrc = CartesianIndices(W)[inds2...]
    Rdst = CartesianIndices(dst)

    for I in nonzero_keys(W)
        j = @something findfirst(==(I), Rsrc) continue
        dst[Rdst[j]] = W[I]
    end

    return dst
end

@inline function Base.setindex!(W::JordanMPOTensor, v::MPOTensor, I::CartesianIndex{4})
    return setindex!(W, v, I.I...)
end
@propagate_inbounds function Base.setindex!(W::JordanMPOTensor, v::MPOTensor,
                                            I::Vararg{Int,4})
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

# Utility
# -------
# Avoid falling back to `norm(W1 - W2)` which has to convert to SparseBlockTensorMap
function Base.isapprox(W1::JordanMPOTensor, W2::JordanMPOTensor; kwargs...)
    return isapprox(W1.A, W2.A; kwargs...) &&
           isapprox(W1.B, W2.B; kwargs...) &&
           isapprox(W1.C, W2.C; kwargs...) &&
           isapprox(W1.D, W2.D; kwargs...)
end

function Base.summary(io::IO, W::JordanMPOTensor)
    szstring = Base.dims2string(size(W))
    TT = eltype(W)
    typeinfo = get(io, :typeinfo, Any)
    if typeinfo <: typeof(W) || typeinfo <: TT
        typestring = ""
    else
        typestring = "{$TT}"
    end
    V = space(W)
    return print(io, "$szstring JordanMPOTensor$typestring($V)")
end
