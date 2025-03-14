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
    A::SparseBlockTensorMap{TA,E,S,2,2,4}
    B::SparseBlockTensorMap{TB,E,S,2,1,3}
    C::SparseBlockTensorMap{TC,E,S,1,2,3}
    D::SparseBlockTensorMap{TD,E,S,1,1,2}

    # uninitialized constructor
    function JordanMPOTensor{E,S,TA,TB,TC,TD}(::UndefInitializer,
                                              V::TensorMapSumSpace{S,2,2}) where {E,S,TA,TB,
                                                                                  TC,TD}
        allVs = eachspace(V)

        VA = space(allVs[2:(end - 1), 1, 1, 2:(end - 1)])
        A = SparseBlockTensorMap{TA}(undef, VA)

        VB = removeunit(space(allVs[2:(end - 1), 1, 1, end]), 4)
        B = SparseBlockTensorMap{TB}(undef, VB)

        VC = removeunit(space(allVs[1, 1, 1, 2:(end - 1)]), 1)
        C = SparseBlockTensorMap{TC}(undef, VC)

        VD = removeunit(removeunit(space(allVs[1, 1, 1, end:end]), 4), 1)
        D = SparseBlockTensorMap{TD}(undef, VD)

        return new{E,S,TA,TB,TC,TD}(A, B, C, D)
    end

    # constructor from data
    function JordanMPOTensor{E,S,TA,TB,TC,TD}(A::SparseBlockTensorMap{TA,E,S,2,2},
                                              B::SparseBlockTensorMap{TB,E,S,2,1},
                                              C::SparseBlockTensorMap{TC,E,S,1,2},
                                              D::SparseBlockTensorMap{TD,E,S,1,1}) where {E,
                                                                                          S,
                                                                                          TA,
                                                                                          TB,
                                                                                          TC,
                                                                                          TD}
        # TODO: add space and size checks
        return new{E,S,TA,TB,TC,TD}(A, B, C, D)
    end
end

function JordanMPOTensor{E,S}(::UndefInitializer,
                              V::TensorMapSumSpace{S}) where {E,S}
    TA = tensormaptype(S, 2, 2, E)
    TB = tensormaptype(S, 2, 1, E)
    TC = tensormaptype(S, 1, 2, E)
    TD = tensormaptype(S, 1, 1, E)
    return JordanMPOTensor{E,S,TA,TB,TC,TD}(undef, V)
end
function JordanMPOTensor{E}(::UndefInitializer, V::TensorMapSumSpace{S}) where {E,S}
    return JordanMPOTensor{E,S}(undef, V)
end

function JordanMPOTensor(A::SparseBlockTensorMap{TA,E,S,2,2},
                         B::SparseBlockTensorMap{TB,E,S,2,1},
                         C::SparseBlockTensorMap{TC,E,S,1,2},
                         D::SparseBlockTensorMap{TD,E,S,1,1}) where {E,S,TA,TB,TC,TD}
    return JordanMPOTensor{E,S,TA,TB,TC,TD}(A, B, C, D)
end

function JordanMPOTensor(W::SparseBlockTensorMap{TT,E,S,2,2}) where {TT,E,S}
    @assert W[1, 1, 1, 1] isa BraidingTensor && W[end, 1, 1, end] isa BraidingTensor
    # @assert all(I -> I[1] ≤ I[4], nonzero_keys(W))

    A = W[2:(end - 1), 1, 1, 2:(end - 1)]
    B = W[2:(end - 1), 1, 1, end]
    C = W[1, 1, 1, 2:(end - 1)]
    D = W[1, 1, 1, end:end] # ensure still blocktensor to allow for sparse

    return JordanMPOTensor(A,
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

# Properties
# ----------
function TensorKit.space(W::JordanMPOTensor)
    V_triv = oneunit(spacetype(W.A))
    V_left = BlockTensorKit.oplus(V_triv, left_virtualspace(W.A), V_triv)
    V_right = BlockTensorKit.oplus(V_triv, right_virtualspace(W.A), V_triv)
    P = physicalspace(W.A) # == physicalspace(W.B) == physicalspace(W.C) == physicalspace(W.D)
    return V_left ⊗ P ← P ⊗ V_right
end

Base.size(W::JordanMPOTensor) = size(W.A) .+ (2, 0, 0, 2)
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
    I == CartesianIndex(1, 1, 1, 1) ||
        I == CartesianIndex(size(W, 1), 1, 1, size(W, 4)) && return true

    row, col = I.I[1], I.I[4]

    if row == 1 && col == size(W, 4)
        return haskey(W.D, CartesianIndex(1, 1))
    elseif row == 1
        return haskey(W.C, CartesianIndex(1, 1, col - 1))
    elseif col == size(W, 4)
        return haskey(W.B, CartesianIndex(row - 1, 1, 1))
    elseif 1 < row < size(W, 1) && 1 < col < size(W, 4)
        return haskey(W.A, CartesianIndex(row - 1, 1, 1, col - 1))
    else
        return false
    end
end

# TODO: avoid this slow fallback wherever possible:
Base.parent(W::JordanMPOTensor) = (error(); parent(SparseBlockTensorMap(W)))

BlockTensorKit.issparse(W::JordanMPOTensor) = true

# Converters
# ----------
function SparseBlockTensorMap(W::JordanMPOTensor)
    τ = BraidingTensor{scalartype(W)}(eachspace(W)[1])
    W′ = SparseBlockTensorMap{Union{eltype(W.A),typeof(τ)}}(undef_blocks, space(W))
    W′[1, 1, 1, 1] = τ
    W′[end, 1, 1, end] = τ

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

@inline function Base.getindex(W::JordanMPOTensor, I::Vararg{Int,4})
    @assert I[2] == I[3] == 1
    return W[I[1], I[4]]
end
@inline Base.getindex(W::JordanMPOTensor, I::CartesianIndex{4}) = W[I.I...]
@propagate_inbounds function Base.getindex(W::JordanMPOTensor, i::Int, j::Int)
    if (i == 1 && j == 1) || (i == size(W, 1) && j == size(W, 4))
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

    Rsrc = CartesianIndices(W)[inds...]
    Rdst = CartesianIndices(dst)

    for I in nonzero_keys(W)
        j = @something findfirst(==(I), Rsrc) continue
        dst[Rdst[j]] = W[I]
    end

    return dst
end

@inline function Base.setindex!(W::JordanMPOTensor, v::MPOTensor, I::Vararg{Int,4})
    @assert I[2] == I[3] == 1
    W[I[1], I[4]] = v
    return W
end
@inline function Base.setindex!(W::JordanMPOTensor, v::MPOTensor, I::CartesianIndex{4})
    return setindex!(W, v, I.I...)
end
@propagate_inbounds function Base.setindex!(W::JordanMPOTensor, v::MPOTensor, i::Int,
                                            j::Int)
    if (i == 1 && j == 1) || (i == size(W, 1) && j == size(W, 4))
        throw(ArgumentError("Cannot set BraidingTensor"))
    elseif i == 1 && j == size(W, 4)
        W.D = removeunit(removeunit(v, 1), 4)
    elseif i == 1
        W.C[1, 1, j - 1] = removeunit(v, 1)
    elseif j == size(W, 4)
        W.B[i - 1, 1, 1] = removeunit(v, 4)
    elseif 1 < i < size(W, 1) && 1 < j < size(W, 4)
        W.A[i - 1, 1, 1, j - 1] = v
    else
        throw(ArgumentError("Cannot set index ($i, 1, 1, $j)"))
    end
    return W
end

# Sparse functionality
# --------------------
function BlockTensorKit.nonzero_keys(W::JordanMPOTensor)
    Ia = CartesianIndex(1, 0, 0, 1)
    pA = [I + Ia for I in nonzero_keys(W.A)]

    Ib = CartesianIndex(1, 0, 0)
    ncols = size(W, 4)
    pB = [CartesianIndex((I + Ib).I..., ncols) for I in nonzero_keys(W.B)]

    Ic = CartesianIndex(0, 0, 1)
    pC = [CartesianIndex(1, (I + Ic).I...) for I in nonzero_keys(W.C)]

    pD = [CartesianIndex(1, 1, 1, ncols) for I in nonzero_keys(W.D)]

    pτ = (CartesianIndex(1, 1, 1, 1), CartesianIndex(1, 1, 1, ncols))
    return Iterators.flatten((pτ, pA, pB, pC, pD))
end
function BlockTensorKit.nonzero_values(W::JordanMPOTensor)
    return Iterators.map(I -> W[I], nonzero_keys(W))
end
function BlockTensorKit.nonzero_pairs(W::JordanMPOTensor)
    return Iterators.map(I -> I => W[I], nonzero_keys(W))
end
function BlockTensorKit.nonzero_length(W::JordanMPOTensor)
    return nonzero_length(W.A) + nonzero_length(W.B) + nonzero_length(W.C) +
           nonzero_length(W.D) + 2
end

# Utility
# -------
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
