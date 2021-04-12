# this object represents a slice of the hamiltonian at a single position.
# you can then do things like transfer_left(vector,ham[5],st.AL[5],st.AR[5])

struct MPOHamSlice{O<:MPOHamiltonian}
    ham::O
    i::Int
end

function Base.getproperty(x::MPOHamSlice,s::Symbol)
    if s == :imspaces
        return x.ham.imspaces[x.i,:]
    elseif s == :domspaces
        return x.ham.domspaces[x.i,:]
    elseif s == :odim
        return x.ham.odim
    elseif s == :Os
        return x.ham.Os[x.i,:,:]
    else
        return getfield(x,s)
    end
end

Base.getindex(x::MPOHamiltonian,a::AbstractVector{Int}) = [x[i] for i in a];
Base.getindex(x::MPOHamiltonian,a::Int) = MPOHamSlice(x,a);

Base.getindex(x::MPOHamSlice,args...) = Base.getindex(x.ham,x.i,args...);
isscal(x::MPOHamSlice,j,k) = isscal(x.ham,x.i,j,k);
Base.keys(x::MPOHamSlice) = keys(x.ham,x.i);
scalkeys(x::MPOHamSlice) = scalkeys(x.ham,x.i);
opkeys(x::MPOHamSlice) = opkeys(x.ham,x.i);
Base.contains(x::MPOHamSlice,j,k) = contains(x.ham,x.i,j,k);

#the usual mpoham transfer
function transfer_left(vec::Vector{V},ham::MPOHamSlice,A::V,Ab::V=A) where V<:MPSTensor
    toreturn = [TensorMap(zeros,eltype(A),_lastspace(Ab)'*ham.imspaces[i],_lastspace(A)') for i in 1:ham.odim]::Vector{V}

    for (j,k) in keys(ham)
        if isscal(ham,j,k)
            toreturn[k]+=ham.Os[j,k]*transfer_left(vec[j],A,Ab)
        else
            v = transfer_left(vec[j],ham[j,k],A,Ab)
            toreturn[k]+=transfer_left(vec[j],ham[j,k],A,Ab)
        end
    end

    return toreturn
end
function transfer_right(vec::Vector{V},ham::MPOHamSlice,A::V,Ab::V=A) where V<:MPSTensor
    toreturn = [TensorMap(zeros,eltype(A),_firstspace(A)*ham.domspaces[i],_firstspace(Ab)) for i in 1:ham.odim]::Vector{V}

    for (j,k) in keys(ham)
        if isscal(ham,j,k)
            toreturn[j]+=ham.Os[j,k]*transfer_right(vec[k],A,Ab)
        else
            toreturn[j]+=transfer_right(vec[k],ham[j,k],A,Ab)
        end
    end

    return toreturn
end
