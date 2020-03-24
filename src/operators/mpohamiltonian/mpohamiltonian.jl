#=
    An mpo hamiltonian h is
        - a sparse collection of dense mpo's
        - often contains the identity on te diagonal (maybe up to a constant)

    when we query h[i,j,k], the following logic is followed
        j == k?
            h.scalars[i,j] assigned?
                identity * h.scalars[i,j]
            elseif h.Os[i,j,k] assigned?
                h.Os[i,j,k]
            else
                zeros
        else
            h.Os[i,j,k] assigned?
                h.Os[i,j,k]
            else
                zeros


    we make the distinction between non-idenity fields and identity field for two reasons
        - speed (can optimize contractions)
        - requires vastly different approaches when doing thermodynamic limit (rescaling)

    both h.scalars and h.Os are periodic, allowing us to represent both place dependent and periodic hamiltonians

    h.domspaces and h.pspaces are needed when h.Os[i,j,k] is unassigned, to know what kind of zero-mpo we need to return

    the convention is that we start left with [1,0,0,0,...,0]; right [0,0,....,0,1]

    I didn't want to use union{T,E} because identity is impossible away from diagonal
=#
"
    MPOHamiltonian

    represents a general periodic quantum hamiltonian
"
struct MPOHamiltonian{S,T<:MPOTensor,E<:Number}<:Hamiltonian
    scalars::PeriodicArray{Array{Union{Missing,E},1},1}
    Os::PeriodicArray{Array{Union{Missing,T},2},1}

    domspaces::PeriodicArray{Array{S,1},1}
    pspaces::PeriodicArray{S,1}
end

function Base.getproperty(h::MPOHamiltonian,f::Symbol)
    if f==:odim
        return length(h.domspaces[1])::Int
    elseif f==:period
        return size(h.pspaces,1)
    elseif f==:imspaces
        return circshift(PeriodicArray([adjoint.(d) for d in h.domspaces.data]),-1)
    else
        return getfield(h,f)
    end
end

#dense representation of mpohamiltonian -> the actual mpohamiltonian
function MPOHamiltonian(ox::Array{T,3}) where T<:Union{Missing,M} where M<:MPOTensor
    x = fillmissing(ox);

    len = size(x,1);E = eltype(M);
    @assert size(x,2)==size(x,3)

    #Os and scalars
    tOs = Matrix{Union{Missing,T}}[Matrix{Union{Missing,M}}(missing,size(x,2),size(x,3)) for i in 1:len]
    tSs = Vector{Union{Missing,E}}[Vector{Union{Missing,E}}(missing,size(x,2)) for i in 1:len]

    for (i,j,k) in Iterators.product(1:size(x,1),1:size(x,2),1:size(x,3))
        if norm(x[i,j,k])>1e-12
            ii,sc = isid(x[i,j,k]) #is identity; if so scalar = sc
            if ii && j==k
                tSs[i][j] = sc
            else
                tOs[i][j,k] = x[i,j,k]
            end

        end
    end

    pspaces=[space(x[i,1,1],2) for i in 1:len]
    domspaces=[[space(y,1) for y in x[i,:,1]] for i in 1:len]

    return MPOHamiltonian(PeriodicArray(tSs),PeriodicArray(tOs),PeriodicArray(domspaces),PeriodicArray(pspaces))
end

#allow passing in 2leg mpos
function MPOHamiltonian(x::Array{T,3}) where T<:Union{Missing,<:MPSBondTensor}
    MPOHamiltonian(map(t-> ismissing(t) ? t : permute(add_util_leg(t),(1,2),(4,3)),x))
end

#allow passing in regular tensormaps
MPOHamiltonian(t::TensorMap) = MPOHamiltonian(decompose_localmpo(add_util_leg(t)));

#a very simple utility constructor; given our "localmpo", constructs a mpohamiltonian
function MPOHamiltonian(x::Array{T,1}) where T<:MPOTensor
    domspaces=[space(y,1) for y in x]
    push!(domspaces,space(x[end],3)')

    pspaces=[space(x[1],2)]

    nOs=Array{Union{Missing,T},2}(missing,length(x)+1,length(x)+1)
    for (i,t) in enumerate(x)
        nOs[i,i+1]=t
    end

    nSs = Array{Union{Missing,eltype(T)},1}(missing,length(x)+1)
    nSs[1] = 1
    nSs[end] = 1

    return MPOHamiltonian(PeriodicArray([nSs]),PeriodicArray([nOs]),PeriodicArray([domspaces]),PeriodicArray(pspaces))
end

#utility functions for finite mpo
function Base.getindex(x::MPOHamiltonian{S,T,E},a::Int,b::Int,c::Int) where {S,T,E}
    if b == c && !ismissing(x.scalars[a][b])
        return x.scalars[a][b]*isomorphism(Matrix{eltype(T)},x.domspaces[a][b]*x.pspaces[a],x.imspaces[a][c]'*x.pspaces[a])::T
    elseif !ismissing(x.Os[a][b,c])
        return x.Os[a][b,c]::T
    else
        return TensorMap(zeros,eltype(T),x.domspaces[a][b]*x.pspaces[a],x.imspaces[a][c]'*x.pspaces[a])::T
    end
end

function Base.setindex!(x::MPOHamiltonian{S,T,E},v::T,a::Int,b::Int,c::Int)  where {S,T,E}
    (ii,scal) = isid(v);

    if ii && b==c
        x.scalars[a][b] = scal
    else
        x.Os[a][b,c] = v;
    end

    return x
end
Base.eltype(x::MPOHamiltonian) = typeof(x[1,1,1])
Base.size(x::MPOHamiltonian) = (x.period,x.odim,x.odim)
Base.size(x::MPOHamiltonian,i) = size(x)[i]

keys(x::MPOHamiltonian) = Iterators.filter(a->contains(x,a[1],a[2],a[3]),Iterators.product(1:x.period,1:x.odim,1:x.odim))
keys(x::MPOHamiltonian,i::Int) = Iterators.filter(a->contains(x,i,a[1],a[2]),Iterators.product(1:x.odim,1:x.odim))

opkeys(x::MPOHamiltonian) = Iterators.filter(a-> !(a[2] == a[3] && isscal(x,a[1],a[2])),keys(x));
opkeys(x::MPOHamiltonian,i::Int) = Iterators.filter(a-> !(a[1] == a[2] && isscal(x,i,a[2])),keys(x,i));

scalkeys(x::MPOHamiltonian) = Iterators.filter(a-> (a[2] == a[3] && isscal(x,a[1],a[2])),keys(x));
scalkeys(x::MPOHamiltonian,i::Int) = Iterators.filter(a-> (a[1] == a[2] && isscal(x,i,a[2])),keys(x,i));

contains(x::MPOHamiltonian,a::Int,b::Int,c::Int) = !ismissing(x.Os[a][b,c]) || (b==c && !ismissing(x.scalars[a][b]))
isscal(x::MPOHamiltonian,a::Int,b::Int) = !ismissing(x.scalars[a][b])

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
    return norm(diff)<1e-14,scal
end

"
checks if ham[:,i,i] = 1 for every i
"
isid(ham::MPOHamiltonian,i::Int) = reduce((a,b) -> a && isscal(ham,b,i) && abs(ham.scalars[b][i]-1)<1e-14,1:ham.period,init=true)

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

#when there are missing values in an input mpo, we will fill them in with 0s
function fillmissing(x::Array{T,3}) where T<:Union{Missing,M} where M<:MPOTensor{Sp} where Sp
    @assert size(x,2) == size(x,3);

    #fill in Domspaces and pspaces
    Domspaces = Array{Union{Missing,Sp},2}(missing,size(x,1),size(x,2))
    pspaces = Array{Union{Missing,Sp},1}(missing,size(x,1))
    for (i,j,k) in Iterators.product(1:size(x,1),1:size(x,2),1:size(x,3))
        if !ismissing(x[i,j,k])
            dom = space(x[i,j,k],1)
            im = space(x[i,j,k],3)
            p = space(x[i,j,k],2)

            if ismissing(pspaces[i])
                pspaces[i] = p;
            elseif pspaces[i] != p
                println("physical space for $((i,j,k)) incompatible")
                println("$(pspaces[i]) ≠ $(p)")
                @assert false
            end

            if ismissing(Domspaces[i,j])
                Domspaces[i,j] = dom
            elseif Domspaces[i,j] != dom
                println("Domspace for $((i,j,k)) incompatible")
                println("$(Domspaces[i,j]) ≠ $(dom)")
                @assert false
            end

            if ismissing(Domspaces[mod1(i+1,end),k])
                Domspaces[mod1(i+1,end),k] = im'
            elseif Domspaces[mod1(i+1,end),k] != im'
                println("Imspace for $((i,j,k)) incompatible")
                println("$(Domspaces[mod1(i+1,end),k]) ≠ $(im')")
                @assert false
            end
        end
    end

    #otherwise x[n,:,:] is empty somewhere
    @assert sum(ismissing.(pspaces))==0
    Domspaces = map(x-> ismissing(x) ? oneunit(Sp) : x,Domspaces) #missing domspaces => oneunit

    nx = Array{T,3}(undef,size(x,1),size(x,2),size(x,3)) # the filled in version of x
    for (i,j,k) in Iterators.product(1:size(x,1),1:size(x,2),1:size(x,3))
        if ismissing(x[i,j,k])
            nx[i,j,k] = TensorMap(zeros,eltype(M),Domspaces[i,j]*pspaces[i],Domspaces[mod1(i+1,end),k]*pspaces[i])
        else
            nx[i,j,k] = x[i,j,k]
        end
    end
    return nx
end

include("utility.jl")
