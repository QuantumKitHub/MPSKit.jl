# this object represents a sparse mpo at a single position
struct SparseMPOSlice{S,T,E} <: AbstractArray{T,2}
    Os::SubArray{Union{T,E}, 2, PeriodicArray{Union{T,E}, 3}, Tuple{Int64, Base.Slice{Base.OneTo{Int64}}, Base.Slice{Base.OneTo{Int64}}}, false}
    domspaces::SubArray{S, 1, PeriodicArray{S, 2}, Tuple{Int64, Base.Slice{Base.OneTo{Int64}}}, false}
    imspaces::SubArray{S, 1, PeriodicArray{S, 2}, Tuple{Int64, Base.Slice{Base.OneTo{Int64}}}, false}
    pspace::S
end

function Base.getproperty(x::SparseMPOSlice,s::Symbol)
    if s == :odim
        return size(x,1)
    else
        return getfield(x,s)
    end
end

#methods it must extend to be an abstractarray
Base.size(sl::SparseMPOSlice) = size(sl.Os)

function Base.getindex(x::SparseMPOSlice{S,T,E},a::Int,b::Int)::T where {S,T,E}
    a <= x.odim && b <= x.odim || throw(BoundsError(x,[a,b]))
    if x.Os[a,b] isa E
        if x.Os[a,b] == zero(E)
            return TensorMap(zeros,E,x.domspaces[a]*x.pspace,x.pspace*x.imspaces[b]')
        else
            return @plansor temp[-1 -2;-3 -4] := (x.Os[a,b]*isomorphism(Matrix{E},x.domspaces[a]*x.pspace,x.imspaces[b]'*x.pspace))[-1 -2;1 2]*τ[1 2;-3 -4]
        end
    else
        return x.Os[a,b]
    end
end

function Base.setindex!(x::SparseMPOSlice{S,T,E},v::T,a::Int,b::Int)  where {S,T,E}
    a <= x.odim && b <= x.odim || throw(BoundsError(x,[a,b]))

    (ii,scal) = isid(v);

    if ii
        x.Os[a,b] = scal
    elseif v ≈ zero(v)
        x.Os[a,b] = zero(E)
    else
        x.Os[a,b] = v;
    end

    return x
end

#utility methods
Base.keys(x::SparseMPOSlice) = Iterators.filter(a->contains(x,a[1],a[2]),product(1:x.odim,1:x.odim))
opkeys(x::SparseMPOSlice) = Iterators.filter(a-> !isscal(x,a[1],a[2]),keys(x));
scalkeys(x::SparseMPOSlice) = Iterators.filter(a-> isscal(x,a[1],a[2]),keys(x));
Base.contains(x::SparseMPOSlice{S,T,E},a::Int,b::Int) where {S,T,E} = !(x.Os[a,b] == zero(E))
isscal(x::SparseMPOSlice{S,T,E},a::Int,b::Int) where {S,T,E} = x.Os[a,b] isa E && contains(x,a,b)
