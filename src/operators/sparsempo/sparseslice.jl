# this object represents a sparse mpo at a single position
"""
    SparseMPOSlice{S,T,E} <: AbstractArray{T,2}

A view of a sparse MPO at a single position.

# Fields
- `Os::AbstractMatrix{Union{T,E}}`: matrix of operators.
- `domspaces::AbstractVector{S}`: list of left virtual spaces.
- `imspaces::AbstractVector{S}`: list of right virtual spaces.
- `pspace::S`: physical space.
"""
struct SparseMPOSlice{S,T,E,A<:AbstractMatrix{Union{E,T}},B<:AbstractVector{Union{S}}} <:
       AbstractMatrix{T}
    Os::A
    domspaces::B
    imspaces::B
    pspace::S
    function SparseMPOSlice(Os::AbstractMatrix{Union{E,T}},
                            domspaces::B, imspaces::B,
                            pspace::S) where {S<:IndexSpace,T<:AbstractTensorMap{S},
                                              E<:Number,B<:AbstractVector{S}}
        sz1, sz2 = size(Os)
        sz1 == length(imspaces) || throw(ArgumentError("imspaces must have length $sz1"))
        sz2 == length(domspaces) || throw(ArgumentError("domspaces must have length $sz2"))
        return new{S,T,E,typeof(Os),B}(Os, domspaces, imspaces, pspace)
    end
end

function Base.getproperty(x::SparseMPOSlice, s::Symbol)
    if s == :odim
        return size(x, 1)
    else
        return getfield(x, s)
    end
end

#methods it must extend to be an abstractarray
Base.size(sl::SparseMPOSlice) = size(sl.Os)

function Base.getindex(x::SparseMPOSlice{S,T,E}, a::Int, b::Int)::T where {S,T,E}
    a <= x.odim && b <= x.odim || throw(BoundsError(x, [a, b]))
    if x.Os[a, b] isa E
        if x.Os[a, b] == zero(E)
            return fill_data!(TensorMap(x -> storagetype(T)(undef, x),
                                        x.domspaces[a] * x.pspace,
                                        x.pspace * x.imspaces[b]'), zero)
        else
            F = isomorphism(storagetype(T), x.domspaces[a] * x.pspace,
                            x.imspaces[b]' * x.pspace)
            return @plansor temp[-1 -2; -3 -4] := (x.Os[a, b] * F)[-1 -2; 1 2] *
                                                  τ[1 2; -3 -4]
        end
    else
        return x.Os[a, b]
    end
end

function Base.setindex!(x::SparseMPOSlice{S,T,E}, v::T, a::Int, b::Int) where {S,T,E}
    a <= x.odim && b <= x.odim || throw(BoundsError(x, [a, b]))
    (ii, scal) = isid(v)

    if ii
        x.Os[a, b] = scal ≈ one(scal) ? one(scal) : scal
    elseif v ≈ zero(v)
        x.Os[a, b] = zero(E)
    else
        x.Os[a, b] = v
    end

    return x
end

#utility methods
function Base.keys(x::SparseMPOSlice)
    return Iterators.filter(a -> contains(x, a[1], a[2]), product(1:(x.odim), 1:(x.odim)))
end
function Base.keys(x::SparseMPOSlice, ::Colon, t::Int)
    return Iterators.filter(a -> contains(x, a, t), 1:(x.odim))
end
function Base.keys(x::SparseMPOSlice, t::Int, ::Colon)
    return Iterators.filter(a -> contains(x, t, a), 1:(x.odim))
end

opkeys(x::SparseMPOSlice) = Iterators.filter(a -> !isscal(x, a[1], a[2]), keys(x));
scalkeys(x::SparseMPOSlice) = Iterators.filter(a -> isscal(x, a[1], a[2]), keys(x));

function opkeys(x::SparseMPOSlice, ::Colon, a::Int)
    return Iterators.filter(t -> contains(x, t, a) && !isscal(x, t, a), 1:(x.odim))
end;
function opkeys(x::SparseMPOSlice, a::Int, ::Colon)
    return Iterators.filter(t -> contains(x, a, t) && !isscal(x, a, t), 1:(x.odim))
end;

function scalkeys(x::SparseMPOSlice, ::Colon, a::Int)
    return Iterators.filter(t -> isscal(x, t, a), 1:(x.odim))
end;
function scalkeys(x::SparseMPOSlice, a::Int, ::Colon)
    return Iterators.filter(t -> isscal(x, a, t), 1:(x.odim))
end;

function Base.contains(x::SparseMPOSlice{S,T,E}, a::Int, b::Int) where {S,T,E}
    return !(x.Os[a, b] == zero(E))
end
function isscal(x::SparseMPOSlice{S,T,E}, a::Int, b::Int) where {S,T,E}
    return x.Os[a, b] isa E && contains(x, a, b)
end
