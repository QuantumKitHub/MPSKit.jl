"
    MPOHamiltonian

    represents a general periodic quantum hamiltonian

    really just a sparsempo, with some garantuees on its structure
"
struct MPOHamiltonian{S,T<:MPOTensor,E<:Number}<:Hamiltonian
    data :: SparseMPO{S,T,E}
end

#default constructor
MPOHamiltonian(x::AbstractArray{<:Any,3}) = MPOHamiltonian(SparseMPO(x))

#allow passing in regular tensormaps
MPOHamiltonian(t::TensorMap) = MPOHamiltonian(decompose_localmpo(add_util_leg(t)));

#-------------

function Base.getproperty(h::MPOHamiltonian,f::Symbol)
    if f==:odim
        return size(h.domspaces,2)
    elseif f==:period
        return size(h.pspaces,1)
    elseif f==:imspaces
        return PeriodicArray(circshift(adjoint.(h.domspaces),(-1,0)))
    else
        return getfield(h,f)
    end
end

#a very simple utility constructor; given our "localmpo", constructs a mpohamiltonian
function MPOHamiltonian(x::Array{T,1}) where T<:MPOTensor{Sp} where Sp
    domspaces = [_firstspace.(x);_lastspace(x[end])']
    pspaces = [space(x[1],2)]

    nOs = PeriodicArray{Union{eltype(T),T}}(fill(zero(eltype(T)),1,length(x)+1,length(x)+1))

    for (i,t) in enumerate(x)
        nOs[1,i,i+1]=t
    end

    nOs[1,1,1] = one(eltype(T));
    nOs[1,end,end] = one(eltype(T));

    ndomspace = PeriodicArray{Sp,2}(undef,1,length(x)+1);
    ndomspace[1,:] = domspaces[:]

    return MPOHamiltonian{Sp,T,eltype(T)}(nOs,ndomspace,PeriodicArray(pspaces))
end

function _envsetypes(d::Tuple)
    a = Base.first(d);
    b = Base.tail(d);

    if a <: MPOTensor
        return spacetype(a),a,eltype(a);
    elseif a <: MPSBondTensor
        return spacetype(a),tensormaptype(spacetype(a),2,2,eltype(a)),eltype(a)
    else
        @assert !isempty(b)
        return _envsetypes(b);
    end
end

#utility functions for finite mpo
function Base.getindex(x::MPOHamiltonian{S,T,E},a::Int,b::Int,c::Int)::T where {S,T,E}
    b <= x.odim && c <= x.odim || throw(BoundsError(x,[a,b,c]))
    if x.Os[a,b,c] isa E
        if x.Os[a,b,c] == zero(E)
            return TensorMap(zeros,E,x.domspaces[a,b]*x.pspaces[a],x.pspaces[a]*x.imspaces[a,c]')
        else
            return x.Os[a,b,c]*isomorphism(Matrix{E},x.domspaces[a,b]*x.pspaces[a],x.pspaces[a]*x.imspaces[a,c]')
        end
    else
        return x.Os[a,b,c]
    end
end

function Base.setindex!(x::MPOHamiltonian{S,T,E},v::T,a::Int,b::Int,c::Int)  where {S,T,E}
    b <= x.odim && c <= x.odim || throw(BoundsError(x,[a,b,c]))

    (ii,scal) = isid(v);

    if ii
        x.Os[a,b,c] = scal
    elseif v ≈ zero(v)
        x.Os[a,b,c] = zero(E)
    else
        x.Os[a,b,c] = v;
    end

    return x
end
Base.getindex(x::MPOHamiltonian,a::Colon,b::Int,c::Int) = [x[i,b,c] for i in 1:x.period];

Base.eltype(x::MPOHamiltonian) = typeof(x[1,1,1])
Base.size(x::MPOHamiltonian) = (x.period,x.odim,x.odim)
Base.size(x::MPOHamiltonian,i) = size(x)[i]

Base.keys(x::MPOHamiltonian) = Iterators.filter(a->contains(x,a[1],a[2],a[3]),product(1:x.period,1:x.odim,1:x.odim))
Base.keys(x::MPOHamiltonian,i::Int) = Iterators.filter(a->contains(x,i,a[1],a[2]),product(1:x.odim,1:x.odim))

opkeys(x::MPOHamiltonian) = Iterators.filter(a-> !isscal(x,a[1],a[2],a[3]),keys(x));
opkeys(x::MPOHamiltonian,i::Int) = Iterators.filter(a-> !isscal(x,i,a[1],a[2]),keys(x,i));

scalkeys(x::MPOHamiltonian) = Iterators.filter(a-> isscal(x,a[1],a[2],a[3]),keys(x));
scalkeys(x::MPOHamiltonian,i::Int) = Iterators.filter(a-> isscal(x,i,a[1],a[2]),keys(x,i));

Base.contains(x::MPOHamiltonian{S,T,E},a::Int,b::Int,c::Int) where {S,T,E} = !(x.Os[a,b,c] == zero(E))
isscal(x::MPOHamiltonian{S,T,E},a::Int,b::Int,c::Int) where {S,T,E} = x.Os[a,b,c] isa E && contains(x,a,b,c)

"
checks if ham[:,i,i] = 1 for every i
"
isid(ham::MPOHamiltonian{S,T,E},i::Int) where {S,T,E}= reduce((a,b) -> a && isscal(ham,b,i,i) && abs(ham.Os[b,i,i]-one(E))<1e-14,1:ham.period,init=true)

"
to be valid in the thermodynamic limit, these hamiltonians need to have a peculiar structure
"
function sanitycheck(ham::MPOHamiltonian)
    for i in 1:ham.period

        @assert isid(ham[i,1,1])[1]
        @assert isid(ham[i,ham.odim,ham.odim])[1]

        for j in 1:ham.odim, k in 1:(j-1)
            contains(ham,i,j,k) && return false
        end
    end

    return true
end

include("linalg.jl")
include("hamslice.jl")
