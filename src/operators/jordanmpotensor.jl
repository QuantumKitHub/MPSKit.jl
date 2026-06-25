"""
    JordanMPOTensor{T,S,A} <: AbstractBlockTensorMap{T,S,2,2}

A tensor map that represents the upper triangular block form of a matrix product operator (MPO).

```math
\\begin{pmatrix}
1 & C & D \\\\
0 & A & B \\\\
0 & 0 & 1
\\end{pmatrix}
```

The genuine operators are stored in `tensors` over the full virtual space, while the
scalar multiples of the identity (in particular the diagonal `1`s) are kept separately in
`scalars`, keyed by their `(row, col)` virtual position.
"""
struct JordanMPOTensor{
        T <: Number, S, A <: DenseVector{T},
    } <: AbstractBlockTensorMap{T, S, 2, 2}
    tensors::SparseBlockTensorMap{TensorMap{T, S, 2, 2, A}, T, S, 2, 2, 4}
    scalars::Dict{CartesianIndex{2}, T}

    # constructor from fields
    function JordanMPOTensor{T, S, A}(
            tensors::SparseBlockTensorMap{TensorMap{T, S, 2, 2, A}, T, S, 2, 2, 4},
            scalars::Dict{CartesianIndex{2}, T}
        ) where {T, S, A}
        return new{T, S, A}(tensors, scalars)
    end

    # uninitialized constructor
    function JordanMPOTensor{T, S, A}(
            ::UndefInitializer, V::TensorMapSumSpace{S, 2, 2}
        ) where {T, S, A}
        tensors = SparseBlockTensorMap{tensormaptype(S, 2, 2, A)}(undef, V)
        scalars = Dict{CartesianIndex{2}, T}()
        rows, cols = size(tensors, 1), size(tensors, 4)
        cols > 1 && (scalars[CartesianIndex(1, 1)] = one(T))
        rows > 1 && (scalars[CartesianIndex(rows, cols)] = one(T))
        return new{T, S, A}(tensors, scalars)
    end
end

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

    W = jordanmpotensortype(S, storagetype(A))(undef, V)
    cols = size(W, 4)
    for (I, v) in nonzero_pairs(A)
        W[I[1] + 1, 1, 1, I[4] + 1] = v
    end
    for (I, v) in nonzero_pairs(B)
        W[I[1] + 1, 1, 1, cols] = insertrightunit(v, 3)
    end
    for (I, v) in nonzero_pairs(C)
        W[1, 1, 1, I[3] + 1] = insertleftunit(v, 1)
    end
    if nonzero_length(D) > 0
        W[1, 1, 1, cols] = insertrightunit(insertleftunit(only(D), 1), 3)
    end
    return W
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

function jordanmpotensortype(::Type{S}, ::Type{E}) where {S <: VectorSpace, E}
    TT = tensormaptype(S, 2, 2, E)
    return JordanMPOTensor{scalartype(TT), S, storagetype(TT)}
end
function jordanmpotensortype(::Type{O}) where {O <: AbstractTensorMap}
    return jordanmpotensortype(spacetype(O), storagetype(O))
end
function Base.similar(W::JordanMPOTensor, ::Type{T}) where {T <: Number}
    TE = TensorKit.similarstoragetype(TensorKit.storagetype(W), T)
    return jordanmpotensortype(spacetype(W), TE)(undef, space(W))
end

# Properties
# ----------
TensorKit.space(W::JordanMPOTensor) = space(getfield(W, :tensors))
function Base.eltype(::Type{<:JordanMPOTensor{T, S, A}}) where {T, S, A}
    return tensormaptype(S, 2, 2, A)
end

function Base.getproperty(W::JordanMPOTensor, sym::Symbol)
    sym === :A && return _jordan_A(W)
    sym === :B && return _jordan_B(W)
    sym === :C && return _jordan_C(W)
    sym === :D && return _jordan_D(W)
    return getfield(W, sym)
end

# reduced-leg blocks reconstructed from `tensors` and `scalars`
function _jordan_A(W::JordanMPOTensor)
    rows, cols = size(W, 1), size(W, 4)
    Tτ = BraidingTensor{scalartype(W), spacetype(W), storagetype(W)}
    TA = tensormaptype(spacetype(W), 2, 2, storagetype(W))
    tensors = getfield(W, :tensors)
    VA = space(eachspace(tensors)[2:(end - 1), 1, 1, 2:(end - 1)])
    A = SparseBlockTensorMap{Union{TA, Tτ}}(undef, VA)
    for (I, v) in nonzero_pairs(tensors)
        (1 < I[1] < rows && 1 < I[4] < cols) || continue
        A[I[1] - 1, 1, 1, I[4] - 1] = v
    end
    for (K, c) in getfield(W, :scalars)
        i, j = K[1], K[2]
        (1 < i < rows && 1 < j < cols) || continue
        τ = Tτ(eachspace(A)[i - 1, 1, 1, j - 1])
        A[i - 1, 1, 1, j - 1] = isone(c) ? τ : scale!(TensorMap(τ), c)
    end
    return A
end
function _jordan_B(W::JordanMPOTensor)
    rows, cols = size(W, 1), size(W, 4)
    TB = tensormaptype(spacetype(W), 2, 1, storagetype(W))
    VB = removeunit(space(eachspace(getfield(W, :tensors))[2:(end - 1), 1, 1, end]), 4)
    B = SparseBlockTensorMap{TB}(undef, VB)
    for I in nonzero_keys(W)
        (1 < I[1] < rows && I[4] == cols) || continue
        B[I[1] - 1, 1, 1] = removeunit(W[I], 4)
    end
    return B
end
function _jordan_C(W::JordanMPOTensor)
    cols = size(W, 4)
    TC = tensormaptype(spacetype(W), 1, 2, storagetype(W))
    VC = removeunit(space(eachspace(getfield(W, :tensors))[1, 1, 1, 2:(end - 1)]), 1)
    C = SparseBlockTensorMap{TC}(undef, VC)
    for I in nonzero_keys(W)
        (I[1] == 1 && 1 < I[4] < cols) || continue
        C[1, 1, I[4] - 1] = removeunit(W[I], 1)
    end
    return C
end
function _jordan_D(W::JordanMPOTensor)
    cols = size(W, 4)
    TD = tensormaptype(spacetype(W), 1, 1, storagetype(W))
    VD = removeunit(removeunit(space(eachspace(getfield(W, :tensors))[1, 1, 1, end:end]), 4), 1)
    D = SparseBlockTensorMap{TD}(undef, VD)
    K = CartesianIndex(1, 1, 1, cols)
    if haskey(W, K)
        D[1, 1] = removeunit(removeunit(W[K], 4), 1)
    end
    return D
end

function Base.haskey(W::JordanMPOTensor, I::CartesianIndex{4})
    Base.checkbounds(W, I.I...)
    return haskey(getfield(W, :scalars), CartesianIndex(I[1], I[4])) ||
        haskey(getfield(W, :tensors), I)
end

Base.parent(W::JordanMPOTensor) = parent(SparseBlockTensorMap(W))

BlockTensorKit.issparse(W::JordanMPOTensor) = true

# Converters
# ----------
function BlockTensorKit.SparseBlockTensorMap(W::JordanMPOTensor)
    W′ = SparseBlockTensorMap{AbstractTensorMap{scalartype(W), spacetype(W), 2, 2}}(
        undef_blocks, space(W)
    )
    for (I, v) in nonzero_pairs(getfield(W, :tensors))
        W′[I] = v
    end
    for (K, c) in getfield(W, :scalars)
        τ = BraidingTensor{scalartype(W), spacetype(W), storagetype(W)}(
            eachspace(W)[K[1], 1, 1, K[2]]
        )
        W′[K[1], 1, 1, K[2]] = isone(c) ? τ : scale!(TensorMap(τ), c)
    end
    return W′
end

for f in (:real, :complex)
    @eval function Base.$f(W::JordanMPOTensor)
        W′ = similar(W, $f(scalartype(W)))
        for (I, v) in nonzero_pairs(getfield(W, :tensors))
            getfield(W′, :tensors)[I] = $f(v)
        end
        empty!(getfield(W′, :scalars))
        for (K, c) in getfield(W, :scalars)
            getfield(W′, :scalars)[K] = $f(c)
        end
        return W′
    end
end

# Indexing
# --------

@inline Base.getindex(W::JordanMPOTensor, I::CartesianIndex{4}) = W[I.I...]
@propagate_inbounds function Base.getindex(W::JordanMPOTensor, I::Vararg{Int, 4})
    @assert I[2] == I[3] == 1
    i, j = I[1], I[4]
    c = get(getfield(W, :scalars), CartesianIndex(i, j), nothing)
    if c !== nothing
        τ = BraidingTensor{scalartype(W), spacetype(W), storagetype(W)}(eachspace(W)[i, 1, 1, j])
        return isone(c) ? TensorMap(τ) : scale!(TensorMap(τ), c)
    elseif haskey(getfield(W, :tensors), CartesianIndex(i, 1, 1, j))
        return getfield(W, :tensors)[i, 1, 1, j]
    else
        return zeros(storagetype(W), eachspace(W)[i, 1, 1, j])
    end
end

@inline function Base.setindex!(W::JordanMPOTensor, v::MPOTensor, I::CartesianIndex{4})
    return setindex!(W, v, I.I...)
end
@propagate_inbounds function Base.setindex!(
        W::JordanMPOTensor, v::MPOTensor, I::Vararg{Int, 4}
    )
    @assert I[2] == I[3] == 1
    i, j = I[1], I[4]
    tensors, scalars = getfield(W, :tensors), getfield(W, :scalars)
    if v isa BraidingTensor
        haskey(tensors, CartesianIndex(i, 1, 1, j)) && delete!(tensors, CartesianIndex(i, 1, 1, j))
        scalars[CartesianIndex(i, j)] = one(scalartype(W))
    else
        delete!(scalars, CartesianIndex(i, j))
        tensors[i, 1, 1, j] = v
    end
    return W
end
@inline function Base.setindex!(W::JordanMPOTensor, v::MPOTensor, I::Int)
    return setindex!(W, v, CartesianIndices(W)[I])
end

# Sparse functionality
# --------------------
function BlockTensorKit.nonzero_keys(W::JordanMPOTensor)
    p = collect(CartesianIndex{4}, nonzero_keys(getfield(W, :tensors)))
    for K in keys(getfield(W, :scalars))
        push!(p, CartesianIndex(K[1], 1, 1, K[2]))
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
    return nonzero_length(getfield(W, :tensors)) + length(getfield(W, :scalars))
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
    for (I, v) in nonzero_pairs(getfield(O, :tensors))
        getfield(Odst, :tensors)[I] = add_physical_charge(v, charge)
    end
    merge!(getfield(Odst, :scalars), getfield(O, :scalars))
    return Odst
end

# Utility
# -------
function Base.copy(W::JordanMPOTensor)
    return typeof(W)(copy(getfield(W, :tensors)), copy(getfield(W, :scalars)))
end
function Base.copy!(Wdst::JordanMPOTensor, Wsrc::JordanMPOTensor)
    space(Wdst) == space(Wsrc) || throw(SpaceMismatch())
    copy!(getfield(Wdst, :tensors), getfield(Wsrc, :tensors))
    empty!(getfield(Wdst, :scalars))
    merge!(getfield(Wdst, :scalars), getfield(Wsrc, :scalars))
    return Wdst
end

# Avoid falling back to `norm(W1 - W2)` which has to convert to SparseBlockTensorMap
function Base.isapprox(W1::JordanMPOTensor, W2::JordanMPOTensor; kwargs...)
    isapprox(getfield(W1, :tensors), getfield(W2, :tensors); kwargs...) || return false
    s1, s2 = getfield(W1, :scalars), getfield(W2, :scalars)
    keys(s1) == keys(s2) || return false
    return all(k -> isapprox(s1[k], s2[k]; kwargs...), keys(s1))
end

function Base.showarg(io::IO, W::JordanMPOTensor, toplevel::Bool)
    !toplevel && print(io, "::")
    print(io, TensorKit.type_repr(typeof(W)))
    return nothing
end

function TensorKit.type_repr(::Type{<:JordanMPOTensor{E, S}}) where {E, S}
    return "JordanMPOTensor{$E, " * TensorKit.type_repr(S) * ", …}"
end
