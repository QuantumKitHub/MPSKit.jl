"
    MPOHamiltonian

    represents a general periodic quantum hamiltonian

    really just a sparsempo, with some garantuees on its structure
"
struct MPOHamiltonian{S,T<:MPOTensor,E<:Number}
    data :: SparseMPO{S,T,E}
end

#default constructor
MPOHamiltonian(x::AbstractArray{<:Any,3}) = MPOHamiltonian(SparseMPO(x))

#allow passing in regular tensormaps
MPOHamiltonian(t::TensorMap) = MPOHamiltonian(decompose_localmpo(add_util_leg(t)));

#a very simple utility constructor; given our "localmpo", constructs a mpohamiltonian
function MPOHamiltonian(x::Array{T,1}) where T<:MPOTensor{Sp} where Sp
    nOs = PeriodicArray{Union{eltype(T),T}}(fill(zero(eltype(T)),1,length(x)+1,length(x)+1))

    for (i,t) in enumerate(x)
        nOs[1,i,i+1]=t
    end

    nOs[1,1,1] = one(eltype(T));
    nOs[1,end,end] = one(eltype(T));


    return MPOHamiltonian(SparseMPO(nOs))
end

function Base.getproperty(h::MPOHamiltonian,f::Symbol)
    if f in (:odim,:period,:imspaces,:domspaces,:Os,:pspaces)
        return getproperty(h.data,f)
    else
        return getfield(h,f)
    end
end

Base.getindex(x::MPOHamiltonian,a) = x.data[a,:,:];
Base.getindex(x::MPOHamiltonian,a,b,c) = x.data[a,b,c];
Base.setindex!(x::MPOHamiltonian,v,a,b,c) = setindex!(x.data,v,a,b,c);

Base.eltype(x::MPOHamiltonian) = eltype(x.data);
Base.size(x::MPOHamiltonian) = (x.period,x.odim,x.odim)
Base.size(x::MPOHamiltonian,i) = size(x)[i]

Base.keys(x::MPOHamiltonian) = keys(x.data)
Base.keys(x::MPOHamiltonian,i::Int) = keys(x.data,i)

opkeys(x::MPOHamiltonian) = opkeys(x.data);
opkeys(x::MPOHamiltonian,i::Int) = opkeys(x.data,i);

scalkeys(x::MPOHamiltonian) = scalkeys(x.data);
scalkeys(x::MPOHamiltonian,i::Int) = scalkeys(x.data,i);

Base.contains(x::MPOHamiltonian,a,b,c) = contains(x.data,a,b,c)
isscal(x::MPOHamiltonian,a,b,c) = isscal(x.data,a,b,c)

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
