"
    SparseMPO - used to represent both time evolution mpos and hamiltonians
"
struct SparseMPO{S,T<:MPOTensor,E<:Number} <: AbstractArray{T,3}
    Os::PeriodicArray{Union{E,T},3}

    domspaces::PeriodicArray{S,2}
    pspaces::PeriodicArray{S,1}
end

function Base.getproperty(h::SparseMPO,f::Symbol)
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

#=
allow passing in
        - non strictly typed matrices
        - missing fields
        - 2leg tensors
        - only mpo tensors
=#

# bit of a helper - accept non strict typed data
SparseMPO(x::AbstractArray{Any,3}) = SparseMPO(union_split(x));

#another helper - artificially create a union and reuse next constructor
SparseMPO(x::AbstractArray{T,3}) where T<: TensorMap = SparseMPO(convert(AbstractArray{Union{T,eltype(T)},3},x));

function SparseMPO(x::AbstractArray{T,3}) where T<:Union{A} where A
    (Sp,M,E) = _envsetypes(union_types(T));

    nx = similar(x,Union{E,M});

    for (i,t) in enumerate(x)
        if t isa MPSBondTensor
            nx[i] = add_util_leg(t)
        elseif ismissing(t)
            nx[i] = zero(E)
        elseif t isa Number
            nx[i] = convert(E,t);
        else
            nx[i] = t;
        end
    end

    SparseMPO(nx);
end

#default constructor
function SparseMPO(x::AbstractArray{Union{E,M},3}) where {M<:MPOTensor,E<:Number}
    (period,numrows,numcols) = size(x);

    Sp = spacetype(M);
    E == eltype(M) || throw(ArgumentError("scalar type should match mpo eltype $E ≠ $(eltype(M))"))
    numrows == numcols || throw(ArgumentError("mpos have to be square"))

    domspaces = PeriodicArray{Union{Missing,Sp}}(missing,period,numrows);
    pspaces = PeriodicArray{Union{Missing,Sp}}(missing,period)

    isused = fill(false,period,numrows,numcols);
    isstopped = false
    while !isstopped
        isstopped = true;

        for i = 1:period, j in 1:numrows, k in 1:numcols
            isused[i,j,k] && continue;

            if x[i,j,k] isa MPOTensor
                isused[i,j,k] = true;
                isstopped = false;

                #asign spaces when possible
                dom = _firstspace(x[i,j,k]);im = _lastspace(x[i,j,k]);p = space(x[i,j,k],2)

                ismissing(pspaces[i]) && (pspaces[i] = p);
                pspaces[i] != p && throw(ArgumentError("physical space for $((i,j,k)) incompatible : $(pspaces[i]) ≠ $(p)"))

                ismissing(domspaces[i,j]) && (domspaces[i,j] = dom)
                domspaces[i,j] != dom && throw(ArgumentError("Domspace for $((i,j,k)) incompatible : $(domspaces[i,j]) ≠ $(dom)"))

                ismissing(domspaces[i+1,k]) && (domspaces[i+1,k] = im')
                domspaces[i+1,k] != im' && throw(ArgumentError("Imspace for $((i,j,k)) incompatible : $(domspaces[i+1,k]) ≠ $(im')"))

                #if it's zero -> store zero
                #if it's the identity -> store identity
                if x[i,j,k] ≈ zero(x[i,j,k])
                    x[i,j,k] = zero(E) #the element is zero/missing
                else
                    ii,sc = isid(x[i,j,k])

                    if ii #the tensor is actually proportional to the identity operator -> store this knowledge
                        x[i,j,k] = sc
                    end
                end
            elseif x[i,j,k] != zero(E)
                if !ismissing(domspaces[i,j])
                    isused[i,j,k] = true;
                    isstopped = false;

                    ismissing(domspaces[i+1,k]) && (domspaces[i+1,k] = domspaces[i,j])
                    domspaces[i+1,k] != domspaces[i,j] && throw(ArgumentError("Identity incompatible at $((i,j,k)) : $(domspaces[i+1,k]) ≠ $(domspaces[i,j])"))
                elseif !ismissing(domspaces[i+1,k])
                    isused[i,j,k] = true;
                    isstopped = false;

                    ismissing(domspaces[i,j]) && (domspaces[i,j] = domspaces[i+1,k])
                    domspaces[i+1,k] != domspaces[i,j] && throw(ArgumentError("Identity incompatible at $((i,j,k)) : $(domspaces[i+1,k]) ≠ $(domspaces[i,j])"))
                end

            else
                isused[i,j,k] = true;
            end
        end
    end

    sum(ismissing.(pspaces)) == 0 || throw(ArgumentError("Not all physical spaces were assigned"))
    sum(ismissing.(domspaces)) == 0 || @warn "failed to deduce all domspaces"
    f_domspaces = map(x-> ismissing(x) ? oneunit(Sp) : x,domspaces) #missing domspaces => oneunit ; should also not happen

    ndomspaces = PeriodicArray{Sp}(f_domspaces)
    npspaces = PeriodicArray{Sp}(pspaces)

    return SparseMPO{Sp,M,E}(PeriodicArray(copy(x)),ndomspaces,npspaces)
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


# mandatory methods to implement for abstractarray
Base.size(x::SparseMPO) = size(x.Os);

#utility functions for finite mpo
function Base.getindex(x::SparseMPO{S,T,E},a,b,c)::T where {S,T,E}
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

function Base.setindex!(x::SparseMPO{S,T,E},v::T,a,b,c)  where {S,T,E}
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

Base.keys(x::SparseMPO) = Iterators.filter(a->contains(x,a[1],a[2],a[3]),product(1:x.period,1:x.odim,1:x.odim))
Base.keys(x::SparseMPO,i::Int) = Iterators.filter(a->contains(x,i,a[1],a[2]),product(1:x.odim,1:x.odim))

opkeys(x::SparseMPO) = Iterators.filter(a-> !isscal(x,a[1],a[2],a[3]),keys(x));
opkeys(x::SparseMPO,i::Int) = Iterators.filter(a-> !isscal(x,i,a[1],a[2]),keys(x,i));

scalkeys(x::SparseMPO) = Iterators.filter(a-> isscal(x,a[1],a[2],a[3]),keys(x));
scalkeys(x::SparseMPO,i::Int) = Iterators.filter(a-> isscal(x,i,a[1],a[2]),keys(x,i));

Base.contains(x::SparseMPO{S,T,E},a::Int,b::Int,c::Int) where {S,T,E} = !(x.Os[a,b,c] == zero(E))
isscal(x::SparseMPO{S,T,E},a::Int,b::Int,c::Int) where {S,T,E} = x.Os[a,b,c] isa E && contains(x,a,b,c)

"
checks if ham[:,i,i] = 1 for every i
"
isid(ham::SparseMPO{S,T,E},i::Int) where {S,T,E}= reduce((a,b) -> a && isscal(ham,b,i,i) && abs(ham.Os[b,i,i]-one(E))<1e-14,1:ham.period,init=true)

"
checks if the given 4leg tensor is the identity (needed for infinite mpo hamiltonians)
"
function isid(x::MPOTensor;tol=Defaults.tolgauge)
    (_firstspace(x) == _lastspace(x)' && space(x,2) == space(x,3)') || return false,zero(eltype(x));
    _can_unambiguously_braid(_firstspace(x)) || return false,zero(eltype(x));

    id = isomorphism(Matrix{eltype(x)},codomain(x),domain(x))
    scal = dot(id,x)/dot(id,id)
    diff = x-scal*id

    return norm(diff)<tol,scal
end

include("linalg.jl")
include("sparseslice.jl")
