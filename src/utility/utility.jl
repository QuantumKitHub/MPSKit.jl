function _transpose_front(t::AbstractTensorMap) # make TensorMap{S,N₁+N₂-1,1}
    return repartition(t, numind(t) - 1, 1)
end
function _transpose_tail(t::AbstractTensorMap) # make TensorMap{S,1,N₁+N₂-1}
    return repartition(t, 1, numind(t) - 1)
end
function _transpose_as(t1::AbstractTensorMap, t2::AbstractTensorMap)
    return repartition(t1, numout(t2), numin(t2))
end

_mul_front(C, A) = _transpose_front(C * _transpose_tail(A))
_mul_tail(A, C) = A * C

function _similar_tail(A::AbstractTensorMap)
    cod = _firstspace(A)
    dom = ⊗(dual(_lastspace(A)), dual.(space.(Ref(A), reverse(2:(numind(A) - 1))))...)
    return similar(A, cod ← dom)
end

_firstspace(t::AbstractTensorMap) = space(t, 1)
_lastspace(t::AbstractTensorMap) = space(t, numind(t))

#given a hamiltonian with unit legs on the side, decompose it using svds to form a "localmpo"
function decompose_localmpo(
        inpmpo::AbstractTensorMap{T, PS, N, N}, trunc = truncbelow(Defaults.tol)
    ) where {T, PS, N}
    N == 2 && return [inpmpo]

    leftind = (N + 1, 1, 2)
    rightind = (ntuple(x -> x + N + 1, N - 1)..., reverse(ntuple(x -> x + 2, N - 2))...)
    U, S, V = tsvd(transpose(inpmpo, (leftind, rightind)); trunc = trunc)

    A = transpose(U * S, ((2, 3), (1, 4)))
    B = transpose(
        V,
        ((1, reverse(ntuple(x -> x + N, N - 2))...), ntuple(x -> x + 1, N - 1))
    )
    return [A; decompose_localmpo(B)]
end

# given a state with util legs on the side, decompose using svds to form an array of mpstensors
function decompose_localmps(
        state::AbstractTensorMap{T, PS, N, 1}, trunc = truncbelow(Defaults.tol)
    ) where {T, PS, N}
    N == 2 && return [state]

    leftind = (1, 2)
    rightind = reverse(ntuple(x -> x + 2, N - 1))
    U, S, V = tsvd(transpose(state, (leftind, rightind)); trunc = trunc)

    A = U * S
    B = _transpose_front(V)
    return [A; decompose_localmps(B)]
end

"""
    add_util_leg(tensor::AbstractTensorMap{S,N1,N2}) where {S,N1,N2}
        -> AbstractTensorMap{S,N1+1,N2+1}

Add trivial one-dimensional utility spaces with trivial sector to the left and right of a
given tensor map, i.e. as the first space of the codomain and the last space of the domain.
"""
function add_util_leg(tensor::AbstractTensorMap{T, S, N1, N2}) where {T, S, N1, N2}
    ou = oneunit(_firstspace(tensor))

    util_front = isomorphism(storagetype(tensor), ou * codomain(tensor), codomain(tensor))
    util_back = isomorphism(storagetype(tensor), domain(tensor), domain(tensor) * ou)

    return util_front * tensor * util_back
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
    restmap = leftnull(first(maps))

    for sp in spaces[2:end]
        cm = isometry(domain(restmap), sp)

        push!(maps, restmap * cm)
        restmap = restmap * leftnull(cm)
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

_totuple(t) = t isa Tuple ? t : tuple(t)

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
