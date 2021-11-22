# this object represents a sparse mpo at a single position
# you can then do things like transfer_left(vector,ham[5],st.AL[5],st.AR[5])

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
            return x.Os[a,b]*isomorphism(Matrix{E},x.domspaces[a]*x.pspace,x.pspace*x.imspaces[b]')
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

#the usual mpoham transfer
function transfer_left(vec::AbstractVector{V},ham::SparseMPOSlice,A::V,Ab::V=A) where V<:MPSTensor
    toret = [TensorMap(zeros,eltype(A),_lastspace(Ab)'*ham.imspaces[i],_lastspace(A)') for i in 1:ham.odim]::Vector{V}

    @sync for k in 1:ham.odim
        @Threads.spawn toret[k] = foldxt(+, 1:ham.odim |>
            Filter(j->contains(ham,j,k)) |>
            Map() do j
                if isscal(ham,j,k)
                    ham.Os[j,k]*transfer_left(vec[j],A,Ab)
                else
                    transfer_left(vec[j],ham[j,k],A,Ab)
                end
            end,init=toret[k]);
    end

    return toret
end
function transfer_right(vec::AbstractVector{V},ham::SparseMPOSlice,A::V,Ab::V=A) where V<:MPSTensor
    toret = [TensorMap(zeros,eltype(A),_firstspace(A)*ham.domspaces[i],_firstspace(Ab)) for i in 1:ham.odim]::Vector{V}

    @sync for j in 1:ham.odim
        @Threads.spawn toret[j] = foldxt(+, 1:ham.odim |>
            Filter(k->contains(ham,j,k)) |>
            Map() do k
                if isscal(ham,j,k)
                    ham.Os[j,k]*transfer_right(vec[k],A,Ab)
                else
                    transfer_right(vec[k],ham[j,k],A,Ab)
                end
            end,init=toret[j]);
    end
    return toret
end
