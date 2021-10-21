#addition / substraction
function Base.:+(a::MPOHamiltonian{S,T,E},e::AbstractArray{V,1}) where {S,T,E,V}
    length(e) == a.period || throw(ArgumentError("periodicity should match $(a.period) ≠ $(length(e))"))

    nOs = copy(a.Os) # we don't want our addition to change different copies of the original hamiltonian

    for c = 1:a.period
        if nOs[c,1,end] isa E
            nOs[c,1,end] += e[c]
        else
            nOs[c,1,end] += e[c]*isomorphism(storagetype(nOs[c,1,end]),codomain(nOs[c,1,end]),domain(nOs[c,1,end]))
        end
    end

    return MPOHamiltonian{S,T,E}(nOs,a.domspaces,a.pspaces)
end
Base.:-(e::Array{T,1},a::MPOHamiltonian) where T = -1.0*a + e
Base.:+(e::Array{T,1},a::MPOHamiltonian) where T = a + e
Base.:-(a::MPOHamiltonian,e::AbstractArray{T,1}) where T = a + (-e)

function Base.:+(a::MPOHamiltonian{S,T,E},b::MPOHamiltonian{S,T,E}) where {S,T,E}
    a.period == b.period || throw(ArgumentError("periodicity should match $(a.period) ≠ $(b.period)"))
    @assert sanitycheck(a)
    @assert sanitycheck(b)

    #new odim;domspaces,imspaces (pspaces necessarily stays the same)
    nodim = a.odim+b.odim-2
    ndomspaces = PeriodicArray{S,2}(undef,a.period,nodim)

    nOs = PeriodicArray{Union{E,T},3}(fill(zero(E),a.period,nodim,nodim))

    for pos in 1:a.period
        ndomspaces[pos,1:(a.odim-1)] = a.domspaces[pos,1:a.odim-1]
        ndomspaces[pos,a.odim:a.odim+b.odim-2] = b.domspaces[pos,2:b.odim]

        for (i,j) in keys(a,pos)
            #A block
            if(i<a.odim && j<a.odim)
                nOs[pos,i,j] = a.Os[pos,i,j]
            end

            #right side
            if(i<a.odim && j==a.odim)
                nOs[pos,i,nodim] = a.Os[pos,i,j]
            end
        end

        for (i,j) in keys(b,pos)

            #upper Bs
            if(i==1 && j>1)
                if nOs[pos,1,a.odim+j-2] isa T
                    nOs[pos,1,a.odim+j-2] += b[pos,i,j]
                else
                    if b.Os[pos,i,j] isa T

                        #one doesn't exist when dom != codom; hacky workaround
                        #nOs[pos,1,a.odim+j-2] = nOs[pos,1,a.odim+j-2]*one(b[pos,i,j])+b[pos,i,j]
                        if nOs[pos,1,a.odim+j-2] == zero(E)
                            nOs[pos,1,a.odim+j-2] = b[pos,i,j]
                        else
                            t_b = isomorphism(storagetype(b[pos,i,j]),codomain(b[pos,i,j]),domain(b[pos,i,j]));
                            nOs[pos,1,a.odim+j-2] = nOs[pos,1,a.odim+j-2]*t_b+b[pos,i,j]
                        end

                    else
                        nOs[pos,1,a.odim+j-2] += b.Os[pos,i,j]
                    end
                end
            end

            #B block
            if(i>1 && j>1)
                nOs[pos,a.odim+i-2,a.odim+j-2] = b.Os[pos,i,j]
            end
        end
    end


    return MPOHamiltonian{S,T,E}(nOs,ndomspaces,a.pspaces)
end
Base.:-(a::MPOHamiltonian,b::MPOHamiltonian) = a+(-1.0*b)

#multiplication
Base.:*(b::Number,a::MPOHamiltonian)=a*b
function Base.:*(a::MPOHamiltonian{S,T,E},b::Number) where {S,T,E}
    nOs = copy(a.Os)

    for i=1:a.period,j = 1:(a.odim-1)
        nOs[i,j,a.odim]*=b;
    end

    return MPOHamiltonian{S,T,E}(nOs,a.domspaces,a.pspaces)
end

function Base.:*(b::MPOHamiltonian{S,T,E},a::MPOHamiltonian{S,T,E}) where {S,T,E}
    nodim = a.odim*b.odim
    indmap = LinearIndices((a.odim,b.odim))
    nOs = PeriodicArray{Union{E,T},3}(fill(zero(E),a.period,nodim,nodim))

    fusers = PeriodicArray(map(product(1:a.period,1:a.odim,1:b.odim)) do (pos,i,j)
        isomorphism(fuse(a.domspaces[pos,i]*b.domspaces[pos,j]),a.domspaces[pos,i]*b.domspaces[pos,j])
    end)

    ndomspaces = PeriodicArray{S,2}(undef,a.period,nodim)
    for pos = 1:a.period,i in 1:a.odim, j = 1:b.odim
        ndomspaces[pos,indmap[i,j]] = codomain(fusers[pos,i,j])
    end

    for pos = 1:a.period,
        (i,j) in keys(a,pos),
        (k,l) in keys(b,pos)

        if isscal(a,pos,i,j) && isscal(b,pos,k,l)
            nOs[pos,indmap[i,k],indmap[j,l]] = a.Os[pos,i,j]*b.Os[pos,k,l]
        else
            @plansor nOs[pos,indmap[i,k],indmap[j,l]][-1 -2;-3 -4] :=
                fusers[pos,i,k][-1;1 2]*conj(fusers[pos+1,j,l][-4;3 4])*a[pos,i,j][1 5;-3 3]*b[pos,k,l][2 -2;5 4]
        end
    end

    return MPOHamiltonian{S,T,E}(nOs,ndomspaces,a.pspaces)
end

#without the copy, we get side effects when repeating + setindex
Base.repeat(x::MPOHamiltonian{S,T,E},n::Int) where {S,T,E} =
    MPOHamiltonian{S,T,E}(repeat(x.Os,n,1,1),repeat(x.domspaces,n,1),repeat(x.pspaces,n))


function Base.conj(a::MPOHamiltonian)
    b = copy(a.Os)

    for (i,j,k) in keys(a)
        @plansor b[i,j,k][-1 -2;-3 -4]:=conj(a[i,j,k][-1 -3;-2 -4])
    end

    MPOHamiltonian(b)
end
