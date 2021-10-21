# scalar multiplication - I'm not sure this one is necessary or makes sense
Base.:*(b::Number,a::SparseMPO)=a*b
function Base.:*(a::SparseMPO{S,T,E},b::Number) where {S,T,E}
    nOs = copy(a.Os)
    nOs[i,j,a.odim].*=b;
    return SparseMPO{S,T,E}(nOs,a.domspaces,a.pspaces)
end

function Base.:*(b::SparseMPO{S,T,E},a::SparseMPO{S,T,E}) where {S,T,E}
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

    return SparseMPO{S,T,E}(nOs,ndomspaces,a.pspaces)
end

#without the copy, we get side effects when repeating + setindex
Base.repeat(x::SparseMPO{S,T,E},n::Int) where {S,T,E} =
    SparseMPO{S,T,E}(repeat(x.Os,n,1,1),repeat(x.domspaces,n,1),repeat(x.pspaces,n))


function Base.conj(a::SparseMPO)
    b = copy(a.Os)

    for (i,j,k) in keys(a)
        @plansor b[i,j,k][-1 -2;-3 -4]:=conj(a[i,j,k][-1 -3;-2 -4])
    end

    SparseMPO(b)
end
