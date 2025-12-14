function _transpose_front(t::AbstractTensorMap; copy::Bool = false)
    return repartition(t, numind(t) - 1, 1; copy)
end
function _transpose_tail(t::AbstractTensorMap; copy::Bool = false) # make TensorMap{S,1,N₁+N₂-1}
    return repartition(t, 1, numind(t) - 1; copy)
end
function _transpose_as(t1::AbstractTensorMap, t2::AbstractTensorMap; copy::Bool = false)
    return repartition(t1, numout(t2), numin(t2); copy)
end

_mul_front(C, A) = matrix_contract(A, C, 1; transpose = true) # _transpose_front(C * _transpose_tail(A))
_mul_tail(A, C) = matrix_contract(A, C, numind(A)) # A * C

function _similar_tail(A::AbstractTensorMap)
    cod = _firstspace(A)
    dom = ⊗(dual(_lastspace(A)), dual.(space.(Ref(A), reverse(2:(numind(A) - 1))))...)
    return similar(A, cod ← dom)
end

_firstspace(t::AbstractTensorMap) = space(t, 1)
_lastspace(t::AbstractTensorMap) = space(t, numind(t))

#given a Hamiltonian with unit legs on the side, decompose it using svds to form a "localmpo"
function decompose_localmpo(
        inpmpo::AbstractTensorMap{T, PS, N, N}, trunc = trunctol(; atol = eps(real(T))^(3 / 4))
    ) where {T, PS, N}
    N == 2 && return [inpmpo]

    leftind = (N + 1, 1, 2)
    rightind = (ntuple(x -> x + N + 1, N - 1)..., reverse(ntuple(x -> x + 2, N - 2))...)
    V, C = left_orth!(transpose(inpmpo, (leftind, rightind); copy = true); trunc)

    A = transpose(V, ((2, 3), (1, 4)))
    B = transpose(C, ((1, reverse(ntuple(x -> x + N, N - 2))...), ntuple(x -> x + 1, N - 1)))
    return [A; decompose_localmpo(B)]
end

# given a state with util legs on the side, decompose using svds to form an array of mpstensors
function decompose_localmps(
        state::AbstractTensorMap{T, PS, N, 1}, trunc = trunctol(; atol = eps(real(T))^(3 / 4))
    ) where {T, PS, N}
    N == 2 && return [state]

    leftind = (1, 2)
    rightind = reverse(ntuple(x -> x + 2, N - 1))
    A, C = left_orth!(transpose(state, (leftind, rightind); copy = true); trunc)
    B = _transpose_front(C)
    return [A; decompose_localmps(B)]
end

"""
    add_util_leg(tensor::AbstractTensorMap{T, S, N1, N2}) where {T, S, N1, N2}
        -> AbstractTensorMap{T, S, N1+1, N2+1}

Add trivial one-dimensional utility spaces with trivial sector to the left and right of a
given tensor map, i.e. as the first space of the codomain and the last space of the domain.
"""
function add_util_leg(tensor::AbstractTensorMap{T, S, N1, N2}) where {T, S, N1, N2}
    return insertrightunit(insertleftunit(tensor, 1), numind(tensor) + 1)
end

function union_split(a::AbstractArray)
    T = reduce((a, b) -> Union{a, b}, typeof.(a))
    nA = similar(a, T)
    return copy!(nA, a)
end
union_types(x::Union) = (x.a, union_types(x.b)...)
union_types(x::Type) = (x,)

function _embedders(spaces)
    totalspace = reduce(⊕, spaces)

    maps = [isometry(totalspace, first(spaces))]
    restmap = left_null(first(maps))

    for sp in spaces[2:end]
        cm = isometry(domain(restmap), sp)

        push!(maps, restmap * cm)
        restmap = restmap * left_null(cm)
    end

    return maps
end

#=
map every element in the tensormap to dfun(E)
allows us to create random tensormaps for any storagetype
=#
function fill_data!(a::TensorMap, dfun)
    for (k, v) in blocks(a)
        map!(x -> dfun(typeof(x)), v, v)
    end

    return a
end
randomize!(a::TensorMap) = fill_data!(a, randn)
function randomize!(a::AbstractBlockTensorMap)
    for t in nonzero_values(a)
        randomize!(t)
    end
    return a
end

_totuple(t) = t isa Tuple ? t : (t isa Symbol ? tuple(t) : Tuple(t))

"""
    tensorexpr(name, ind_out, [ind_in])

Generates expressions for use within [`@tensor`](@extref TensorOperations.@tensor) environments
of the form `name[ind_out...; ind_in]`.
"""
tensorexpr(name, inds) = Expr(:ref, name, _totuple(inds)...)
function tensorexpr(name, indout, indin)
    return Expr(
        :typed_vcat, name, Expr(:row, _totuple(indout)...), Expr(:row, _totuple(indin)...)
    )
end

function check_length(a, b...)
    L = length(a)
    all(==(L), length.(b)) || throw(ArgumentError("lengths must match"))
    return L
end

function fuser(::Type{T}, V1::S, V2::S) where {T, S <: IndexSpace}
    return isomorphism(T, fuse(V1 ⊗ V2), V1 ⊗ V2)
end

"""
    check_unambiguous_braiding(::Type{Bool}, V::VectorSpace)::Bool
    check_unambiguous_braiding(V::VectorSpace)

Verify that the braiding of a vector space is unambiguous. This is the case if the braiding
is symmetric or if all sectors are trivial. The signature with `Type{Bool}` is used to check
while the signature without is used to throw an error if the braiding is ambiguous.
"""
function check_unambiguous_braiding(::Type{Bool}, V::VectorSpace)
    I = sectortype(V)
    BraidingStyle(I) isa SymmetricBraiding && return true
    return all(isone, sectors(V))
end
function check_unambiguous_braiding(V::VectorSpace)
    return check_unambiguous_braiding(Bool, V) ||
        throw(ArgumentError("cannot unambiguously braid $V"))
end

"""
    matrix_contract(
        A::AbstractTensorMap, B::AbstractTensorMap{T, S, 1, 1}, i::Int,
        α::Number = One(),
        backend::AbstractBackend = DefaultBackend(), allocator = DefaultAllocator();
        transpose::Bool = false
    )

Compute the tensor contraction `α * A * B`, where the (1, 1) - tensor `B` is attached to index `i` of `A`.
Whenever `transpose = true`, this contraction (lazily) uses `transpose(B)` instead.

See also [`matrix_contract!`](@ref).
"""
function matrix_contract(
        A::AbstractTensorMap, B::AbstractTensorMap{<:Any, <:Any, 1, 1}, i::Int,
        α::Number = One(), backend::AbstractBackend = DefaultBackend(), allocator = DefaultAllocator();
        transpose::Bool = false
    )
    if i <= numout(A)
        cod = ProductSpace(TT.setindex(codomain(A).spaces, space(B, transpose ? 1 : 2), i))
        dom = domain(A)
    else
        cod = codomain(A)
        dom = ProductSpace(TT.setindex(domain(A).spaces, space(B, transpose ? 1 : 2)', i - numout(A)))
    end
    T = TensorOperations.promote_contract(scalartype(A), scalartype(B), scalartype(α))
    C = similar(A, T, cod ← dom)
    return matrix_contract!(C, A, B, i, α, Zero(), backend, allocator; transpose)
end

"""
    matrix_contract!(
        C::AbstractTensorMap, A::AbstractTensorMap, B::AbstractTensorMap{T, S, 1, 1}, i::Int,
        α::Number = One(), β::Number = Zero(),
        backend::AbstractBackend = DefaultBackend(), allocator = DefaultAllocator();
        transpose::Bool = false
    )

Compute the tensor contraction `C ← β * C + α * A * B`, where the (1, 1) - tensor `B` is attached to index `i` of `A`,
and the result is added into `C`. Whenever `transpose = true`, this contraction (lazily) uses `transpose(B)` instead.

See also [`matrix_contract`](@ref).
"""
function matrix_contract!(
        C::AbstractTensorMap, A::AbstractTensorMap, B::AbstractTensorMap{<:Any, <:Any, 1, 1}, i::Int,
        α::Number = One(), β::Number = Zero(),
        backend::AbstractBackend = DefaultBackend(), allocator = DefaultAllocator();
        transpose::Bool = false
    )

    @boundscheck for k in 1:numind(C)
        numin(C) == numin(A) && numout(C) == numout(A) || throw(ArgumentError("Invalid number of dimensions"))
        if k == i
            space(C, k) == space(B, transpose ? 1 : 2) || throw(SpaceMismatch())
            space(A, k) == space(B, transpose ? 2 : 1)' || throw(SpaceMismatch())
        else
            space(C, k) == space(A, k) || throw(SpaceMismatch())
        end
    end

    N, N₁ = numind(C), numout(C)
    pA = (TT.deleteat(ntuple(identity, N), i), (i,))
    pB = transpose ? ((2,), (1,)) : ((1,), (2,))
    pAB = TensorKit._canonicalize(TT.insertafter(ntuple(identity, N - 1), i - 1, (N,)), C)

    Bblocks = blocks(B)
    for ((f₁, f₂), c) in subblocks(C)
        uncoupled_i = i <= N₁ ? f₁.uncoupled[i] : f₂.uncoupled[i - N₁]
        transpose && (uncoupled_i = dual(uncoupled_i))
        if TensorKit.hasblock(B, uncoupled_i)
            a = A[f₁, f₂]
            b = Bblocks[uncoupled_i]
            TensorOperations.tensorcontract!(c, a, pA, false, b, pB, false, pAB, α, β, backend, allocator)
        else
            scale!(c, β)
        end
    end

    return C
end

@inline fuse_legs(x::TensorMap, N₁::Int, N₂::Int) = fuse_legs(x, Val(N₁), Val(N₂))
function fuse_legs(x::TensorMap, ::Val{N₁}, ::Val{N₂}) where {N₁, N₂}
    ((0 <= N₁ <= numout(x)) && (0 <= N₂ <= numin(x))) || throw(ArgumentError("invalid fusing scheme"))
    init = one(spacetype(x))

    cod = if N₁ > 1
        cod_spaces = codomain(x).spaces
        fuse(prod(TT.getindices(cod_spaces, ntuple(identity, N₁)))) ⊗
            prod(TT.getindices(cod_spaces, ntuple(i -> i + N₁, numout(x) - N₁)); init)
    else
        codomain(x)
    end

    dom = if N₂ > 1
        dom_spaces = domain(x).spaces
        dom = fuse(prod(TT.getindices(dom_spaces, ntuple(identity, N₂)); init)) ⊗
            prod(TT.getindices(domain(x).spaces, ntuple(i -> i + N₂, numin(x) - N₂)); init)
    else
        domain(x)
    end

    return TensorMap{scalartype(x)}(x.data, cod ← dom)
end
