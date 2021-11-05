# this object represents a sparse mpo at a single position
# you can then do things like transfer_left(vector,ham[5],st.AL[5],st.AR[5])

#=
Os::PeriodicArray{Union{E,T},3}

domspaces::PeriodicArray{S,2}
pspaces::PeriodicArray{S,1}
=#
struct SparseMPOSlice{S,T,E} <: AbstractArray{T,2}
    Os::Matrix{Union{E,T}}
    domspaces::Vector{S}
    imspaces::Vector{S}
    pspace::S
end

function Base.getproperty(x::SparseMPOSlice,s::Symbol)
    if s == :odim
        return size(x,1)
    else
        return getfield(x,s)
    end
end

#created here
Base.getindex(x::SparseMPO{S,T,E},a::Int,b::Colon,c::Colon) where {S,T,E} = SparseMPOSlice{S,T,E}(convert(Matrix{Union{E,T}},x.Os[a,:,:]),x.domspaces[a,:],x.imspaces[a,:],x.pspaces[a]);
Base.getindex(x::SparseMPO,a::Colon,b::Colon,c::Colon) = x;
Base.getindex(x::SparseMPO,a::UnitRange,b::Colon,c::Colon) = map(t->x[t,:,:],a);

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
        @Threads.spawn toret[k] = foldl(+, 1:ham.odim |>
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
        @Threads.spawn toret[j] = foldl(+, 1:ham.odim |>
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
