#=
    Implements mpohamiltonian algebra (addition,subtraction,multiplication)
    other actions on mpohamiltonian objects
    the mpohamiltonian transfers
=#

#addition / substraction
function Base.:+(a::MPOHamiltonian{S,T,E},e::AbstractArray{V,1}) where {S,T,E,V}

    @assert length(e) == a.period

    nOs = deepcopy(a.Os) # we don't want our addition to change different copies of the original hamiltonian

    for c = 1:a.period
        if nOs[c,1,end] isa E
            nOs[c,1,end] += e[c]
        else
            @tensor nOs[c,1,a.odim][-1 -2;-3 -4]:=a[c,1,a.odim][-1,-2,-3,-4]+(e[c]*one(a[c,1,a.odim]))[-1,-2,-3,-4]
        end
    end

    return MPOHamiltonian{S,T,E}(nOs,a.domspaces,a.pspaces)
end
Base.:-(e::Array{T,1},a::MPOHamiltonian) where T = -1.0*a + e
Base.:+(e::Array{T,1},a::MPOHamiltonian) where T = a + e
Base.:-(a::MPOHamiltonian,e::AbstractArray{T,1}) where T = a + (-e)

function Base.:+(a::MPOHamiltonian{S,T,E},b::MPOHamiltonian{S,T,E}) where {S,T,E}
    @assert a.period == b.period
    @assert sanitycheck(a)
    @assert sanitycheck(b)

    #new odim;domspaces,imspaces (pspaces necessarily stays the same)
    nodim = a.odim+b.odim-2
    ndomspaces = PeriodicArray{S,2}(undef,a.period,nodim)

    nOs = PeriodicArray{Union{E,T},3}(fill(zero(E),a.period,nodim,nodim))

    for pos in 1:a.period
        ndomspace = Vector{S}(undef,nodim)

        #not entirely sure :3
        ndomspace[1:(a.odim-1)]=a.domspaces[pos,1:a.odim-1]
        ndomspace[a.odim:a.odim+b.odim-2]=b.domspaces[pos,2:b.odim]
        ndomspaces[pos,:] = ndomspace[:]

        for (i,j) in keys(a,pos)
            #A block
            if(i<a.odim && j<a.odim)
                nOs[pos,i,j]=a.Os[pos,i,j]
            end

            #right side
            if(i<a.odim && j==a.odim)
                nOs[pos,i,nodim]=a.Os[pos,i,j]
            end
        end

        for (i,j) in keys(b,pos)

            #upper Bs
            if(i==1 && j>1)
                if nOs[pos,1,a.odim+j-2] isa T
                    nOs[pos,1,a.odim+j-2] += b[pos,i,j]
                else
                    if b.Os[pos,i,j] isa T
                        if nOs[pos,1,a.odim+j-2] == zero(E)
                            nOs[pos,1,a.odim+j-2] = b[pos,i,j] #one doesn't exist when dom != codom; hacky workaround
                        else
                            nOs[pos,1,a.odim+j-2] = nOs[pos,1,a.odim+j-2]*one(b[pos,i,j])+b[pos,i,j]
                        end
                    else
                        nOs[pos,1,a.odim+j-2] += b.Os[pos,i,j]
                    end
                end
            end

            #B block
            if(i>1 && j>1)
                nOs[pos,a.odim+i-2,a.odim+j-2]=b.Os[pos,i,j]
            end
        end
    end


    return MPOHamiltonian{S,T,E}(nOs,ndomspaces,a.pspaces)
end
Base.:-(a::MPOHamiltonian,b::MPOHamiltonian) = a+(-1.0*b)

#multiplication
Base.:*(b::Number,a::MPOHamiltonian)=a*b
function Base.:*(a::MPOHamiltonian{S,T,E},b::Number) where {S,T,E}
    nOs=deepcopy(a.Os)

    for i=1:a.period
        for j=1:(a.odim-1)
            if(contains(a,i,j,a.odim))
                nOs[i,j,a.odim]*=b
            end
        end
    end

    return MPOHamiltonian{S,T,E}(nOs,a.domspaces,a.pspaces)
end

#this is the index-map used in the ham x ham multiplication function (also needed somewhere else)
#i think julia has a build in for this, but it got renamed somewhere (linearindices?)
multmap(a::MPOHamiltonian,b::MPOHamiltonian) = (i,j)->(i-1)*b.odim+j
function Base.:*(b::MPOHamiltonian{S,T,E},a::MPOHamiltonian{S,T,E}) where {S,T,E}
    nodim=a.odim*b.odim

    indmap=multmap(a,b)

    nOs = PeriodicArray{Union{E,T},3}(fill(zero(E),a.period,nodim,nodim))

    ndomspaces = PeriodicArray{S,2}(undef,a.period,nodim)

    for pos=1:a.period
        ndomspace=Array{eltype(a.domspaces[pos]),1}(undef,nodim)

        for i in 1:a.odim
            for j in 1:b.odim
                ndomspace[indmap(i,j)]=fuse(a.domspaces[pos,i]*b.domspaces[pos,j])
            end
        end
        ndomspaces[pos,:]=ndomspace[:]
    end

    for pos=1:a.period
        for (i,j) in keys(a,pos)
            for (k,l) in keys(b,pos)
                if isscal(a,pos,i,j) && isscal(b,pos,k,l)
                    nOs[pos,indmap(i,k),indmap(j,l)] = a.Os[pos,i,j]*b.Os[pos,k,l]
                else
                    @tensor newopp[-1 -2;-3 -4 -5 -6]:=a[pos,i,j][-1,1,-4,-6]*b[pos,k,l][-2,-3,-5,1]
                    newopp=TensorMap(newopp.data,ndomspaces[pos,indmap(i,k)],domain(newopp))
                    newopp=permute(newopp,(1,2,5),(3,4))
                    newopp=TensorMap(newopp.data,codomain(newopp),ndomspaces[pos+1,indmap(j,l)])
                    newopp=permute(newopp,(1,2),(4,3))

                    nOs[pos,indmap(i,k),indmap(j,l)]=newopp
                end
            end
        end
    end

    return MPOHamiltonian{S,T,E}(nOs,ndomspaces,a.pspaces)
end

#without the copy, we get side effects when repeating + setindex
Base.repeat(x::MPOHamiltonian{S,T,E},n::Int) where {S,T,E} = 
    MPOHamiltonian{S,T,E}(
                                            repeat(x.Os,n,1,1),
                                            repeat(x.domspaces,n,1),
                                            repeat(x.pspaces,n))

#transpo = false => inplace conjugate
#transpo = true => flip physical legs
function Base.conj(a::MPOHamiltonian;transpo=false)
    b = deepcopy(a.Os)

    for (i,j,k) in keys(a)
        b[i,j,k] = @tensor temp[-1 -2;-3 -4]:=conj(a[i,j,k][-1,-2,-3,-4])
        if transpo
            b[i,j,k]=permute(b[i,j,k],(1,4),(3,2))
        end
    end

    MPOHamiltonian(b)
end

#needed this; perhaps move to tensorkit?
TensorKit.fuse(f::T) where T<: VectorSpace = f

#the usual mpoham transfer
function transfer_left(vec::Array{V,1},ham::MPOHamiltonian,pos::Int,A::V,Ab::V=A) where V<:MPSTensor
    toreturn = Array{V,1}(undef,length(vec));
    assigned = [false for i in 1:ham.odim]

    for (j,k) in keys(ham,pos)
        if assigned[k]
            if isscal(ham,pos,j,k)
                toreturn[k]+=ham.Os[pos,j,k]*transfer_left(vec[j],A,Ab)
            else
                toreturn[k]+=transfer_left(vec[j],ham[pos,j,k],A,Ab)
            end
        else
            if isscal(ham,pos,j,k)
                toreturn[k]=ham.Os[pos,j,k]*transfer_left(vec[j],A,Ab)
            else
                toreturn[k]=transfer_left(vec[j],ham[pos,j,k],A,Ab)
            end
            assigned[k]=true
        end
    end


    for k in 1:ham.odim
        if !assigned[k]
            #prefereably this never happens, because it's a wasted step
            #it's also avoideable with a little bit more code
            toreturn[k]=transfer_left(vec[1],ham[pos,1,k],A,Ab)
        end
    end

    return toreturn
end
function transfer_right(vec::Array{V,1},ham::MPOHamiltonian,pos::Int,A::V,Ab::V=A) where V<:MPSTensor
    toreturn = Array{V,1}(undef,length(vec));
    assigned = [false for i in 1:ham.odim]

    for (j,k) in keys(ham,pos)
        if assigned[j]
            if isscal(ham,pos,j,k)
                toreturn[j]+=ham.Os[pos,j,k]*transfer_right(vec[k],A,Ab)
            else
                toreturn[j]+=transfer_right(vec[k],ham[pos,j,k],A,Ab)
            end

        else
            if isscal(ham,pos,j,k)
                toreturn[j]=ham.Os[pos,j,k]*transfer_right(vec[k],A,Ab)
            else
                toreturn[j]=transfer_right(vec[k],ham[pos,j,k],A,Ab)
            end
            assigned[j]=true
        end
    end

    for j in 1:ham.odim
        if !assigned[j]
            toreturn[j]=transfer_right(vec[1],ham[pos,j,1],A,Ab)
        end
    end

    return toreturn
end
