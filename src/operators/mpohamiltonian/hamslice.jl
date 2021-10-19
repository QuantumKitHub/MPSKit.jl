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
function transfer_right(vec::Vector{V},ham::MPOHamSlice,A::V,Ab::V=A) where V<:MPSTensor
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
