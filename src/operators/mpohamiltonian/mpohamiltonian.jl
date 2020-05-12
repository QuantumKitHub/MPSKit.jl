#=
    Os can be either
        - a nonzero scalar
        - a tensormap
        - zero

    if it is a scalar; implicitly we have the identity operator at that site - so we can optimize the contraction
    if it is zero then there is nothing to optimize, the result is zero anyway
    if it's a tensormap, then we do the contraction

    In a later stage we can perhaps support nonzero scalar -> scalar*isometry - though the usecases for that seem very limited

    The principles are that you can write code without knowing anything about the sparse structure, and it should just work (potentially a bit slower)

    Unhappy about this design because :
        - the constructor is a mess
        - need to know about the internals to use efficiently
        - it's annoying to initialize

    An alternative is to store 2 fields;
        - the dense blocks (so store the zero opperators)
        - an informative (isscal,value)
=#
"
    MPOHamiltonian

    represents a general periodic quantum hamiltonian
"
struct MPOHamiltonian{S,T<:MPOTensor,E<:Number}<:Hamiltonian
    Os::PeriodicArray{Union{E,T},3}

    domspaces::PeriodicArray{S,2}
    pspaces::PeriodicArray{S,1}
end

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

#allow passing in 2leg tensors
MPOHamiltonian(x::M) where M <: AbstractArray{T,3} where T<: Union{<:MPSBondTensor,E} where E =
    MPOHamiltonian(map(t-> isa(t,MPSBondTensor) ? permute(add_util_leg(t),(1,2),(4,3)) : t,x))

#allow passing in only tensors
MPOHamiltonian(x::M) where M <: AbstractArray{T,3} where T <: MPOTensor =
    MPOHamiltonian(Array{Union{eltype(T),T}}(x[:,:,:]))

#default constructor
function MPOHamiltonian(x::T) where T <:AbstractArray{Union{oE,M},3} where M<:MPOTensor{Sp} where {Sp,oE<:Number}
    E = eltype(M)#there is a bug in julia at the moment, this is a 'workaround'
    #https://github.com/JuliaLang/julia/issues/35853

    (period,numrows,numcols) = size(x);

    E == eltype(M) || throw(ArgumentError("scalar type should match mpo eltype $E ≠ $(eltype(M))"))
    numrows == numcols || throw(ArgumentError("mpos have to be square"))

    domspaces = PeriodicArray{Union{Missing,Sp}}(missing,period,numrows);
    pspaces = PeriodicArray{Union{Missing,Sp}}(missing,period)

    for (i,j,k) in Iterators.product(1:period,1:numrows,1:numcols)
        if x[i,j,k] isa MPOTensor
            #asign spaces when possible
            dom = space(x[i,j,k],1);im = space(x[i,j,k],3);p = space(x[i,j,k],2)

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
        end
    end

    sum(ismissing.(pspaces)) == 0 || throw(ArgumentError("Not all physical spaces were assigned"))
    f_domspaces = map(x-> ismissing(x) ? oneunit(Sp) : x,domspaces) #missing domspaces => oneunit ; should also not happen

    ndomspaces = PeriodicArray{Sp}(f_domspaces)
    npspaces = PeriodicArray{Sp}(pspaces)

    return MPOHamiltonian{Sp,M,E}(PeriodicArray(x[:,:,:]),ndomspaces,npspaces)
end


#allow passing in regular tensormaps
MPOHamiltonian(t::TensorMap) = MPOHamiltonian(decompose_localmpo(add_util_leg(t)));

#a very simple utility constructor; given our "localmpo", constructs a mpohamiltonian
function MPOHamiltonian(x::Array{T,1}) where T<:MPOTensor{Sp} where Sp
    domspaces = Sp[space(y,1) for y in x]
    push!(domspaces,space(x[end],3)')

    pspaces=[space(x[1],2)]

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

#utility functions for finite mpo
function Base.getindex(x::MPOHamiltonian{S,T,E},a::Int,b::Int,c::Int) where {S,T,E}
    b <= x.odim && c <= x.odim || throw(BoundsError(x,[a,b,c]))
    if x.Os[a,b,c] isa E
        if x.Os[a,b,c] == zero(E)
            return TensorMap(zeros,E,x.domspaces[a,b]*x.pspaces[a],x.imspaces[a,c]'*x.pspaces[a])::T
        else
            return x.Os[a,b,c]*isomorphism(Matrix{E},x.domspaces[a,b]*x.pspaces[a],x.imspaces[a,c]'*x.pspaces[a])::T
        end
    else
        return x.Os[a,b,c]::T
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
Base.eltype(x::MPOHamiltonian) = typeof(x[1,1,1])
Base.size(x::MPOHamiltonian) = (x.period,x.odim,x.odim)
Base.size(x::MPOHamiltonian,i) = size(x)[i]

keys(x::MPOHamiltonian) = Iterators.filter(a->contains(x,a[1],a[2],a[3]),Iterators.product(1:x.period,1:x.odim,1:x.odim))
keys(x::MPOHamiltonian,i::Int) = Iterators.filter(a->contains(x,i,a[1],a[2]),Iterators.product(1:x.odim,1:x.odim))

opkeys(x::MPOHamiltonian) = Iterators.filter(a-> !isscal(x,a[1],a[2],a[3]),keys(x));
opkeys(x::MPOHamiltonian,i::Int) = Iterators.filter(a-> !isscal(x,i,a[1],a[2]),keys(x,i));

scalkeys(x::MPOHamiltonian) = Iterators.filter(a-> isscal(x,a[1],a[2],a[3]),keys(x));
scalkeys(x::MPOHamiltonian,i::Int) = Iterators.filter(a-> isscal(x,i,a[1],a[2]),keys(x,i));

contains(x::MPOHamiltonian{S,T,E},a::Int,b::Int,c::Int) where {S,T,E} = !(x.Os[a,b,c] == zero(E))
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

        for j in 1:ham.odim
            for k in 1:(j-1)
                if contains(ham,i,j,k)
                    return false
                end
            end
        end
    end

    return true
end

"
checks if the given 4leg tensor is the identity (needed for infinite mpo hamiltonians)
"
function isid(x::MPOTensor)
    cod = space(x,1)*space(x,2);
    dom = space(x,3)'*space(x,4)';

    #would like to have an 'isisomorphic'
    for c in union(blocksectors(cod), blocksectors(dom))
        blockdim(cod, c) == blockdim(dom, c) || return false,0.0;
    end

    id = isomorphism(Matrix{eltype(x)},cod,dom)
    scal = dot(id,x)/dot(id,id)
    diff = x-scal*id

    scal = (scal ≈ 0.0) ? 0.0 : scal #shouldn't be necessary (and I don't think it is)

    return norm(diff)<1e-14,scal
end

include("utility.jl")
