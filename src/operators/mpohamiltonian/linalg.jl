#addition / substraction
function Base.:+(a::MPOHamiltonian,e::AbstractVector)
    length(e) == a.period || throw(ArgumentError("periodicity should match $(a.period) ≠ $(length(e))"))

    nOs = copy(a.data) # we don't want our addition to change different copies of the original hamiltonian

    for c = 1:a.period
        nOs[c,1,end] += e[c]*isomorphism(storagetype(nOs[c,1,end]),codomain(nOs[c,1,end]),domain(nOs[c,1,end]))
    end

    return MPOHamiltonian(nOs)
end
Base.:-(e::AbstractVector,a::MPOHamiltonian) = -1.0*a + e
Base.:+(e::AbstractVector,a::MPOHamiltonian) = a + e
Base.:-(a::MPOHamiltonian,e::AbstractVector) = a + (-e)

function Base.:+(a::MPOHamiltonian{S,T,E},b::MPOHamiltonian{S,T,E}) where {S,T,E}
    a.period == b.period || throw(ArgumentError("periodicity should match $(a.period) ≠ $(b.period)"))
    @assert sanitycheck(a)
    @assert sanitycheck(b)

    nodim = a.odim+b.odim-2;
    nOs = PeriodicArray{Union{E,T},3}(fill(zero(E),a.period,nodim,nodim))

    for pos in 1:a.period
        for (i,j) in keys(a,pos)
            #A block
            if(i<a.odim && j<a.odim)
                nOs[pos,i,j] = a[pos,i,j]
            end

            #right side
            if(i<a.odim && j==a.odim)
                nOs[pos,i,nodim] = a[pos,i,j]
            end
        end

        for (i,j) in keys(b,pos)

            #upper Bs
            if(i==1 && j>1)
                if nOs[pos,1,a.odim+j-2] isa T
                    nOs[pos,1,a.odim+j-2] += b[pos,i,j]
                else
                    nOs[pos,1,a.odim+j-2] = b[pos,i,j]
                end
            end

            #B block
            if(i>1 && j>1)
                nOs[pos,a.odim+i-2,a.odim+j-2] = b[pos,i,j]
            end
        end
    end

    MPOHamiltonian(SparseMPO(nOs))
end
Base.:-(a::MPOHamiltonian,b::MPOHamiltonian) = a+(-1.0*b)

#multiplication
Base.:*(b::Number,a::MPOHamiltonian)=a*b
function Base.:*(a::MPOHamiltonian,b::Number)
    nOs = copy(a.data);

    for i=1:a.period,j = 1:(a.odim-1)
        nOs[i,j,a.odim]*=b;
    end
    MPOHamiltonian(nOs);
end

Base.:*(b::MPOHamiltonian,a::MPOHamiltonian) = MPOHamiltonian(b.data*a.data);
Base.repeat(x::MPOHamiltonian,n::Int) = MPOHamiltonian(repeat(x.data,n));
Base.conj(a::MPOHamiltonian) = MPOHamiltonian(conj(a.data))
