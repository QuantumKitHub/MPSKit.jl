#=
    Implements mpohamiltonian algebra (addition,subtraction,multiplication)
    other actions on mpohamiltonian objects
    the mpohamiltonian transfers
=#

#addition / substraction
function Base.:+(a::MpoHamiltonian,e::AbstractArray{T,1}) where T
    @assert length(e)==a.period
    nOs=copy(a.Os)

    for c=1:a.period
        @tensor nOs[c][1,a.odim][-1 -2;-3 -4]:=a[c,1,a.odim][-1,-2,-3,-4]+(e[c]*one(a[c,1,a.odim]))[-1,-2,-3,-4]
    end

    return MpoHamiltonian(a.scalars,nOs,a.domspaces,a.pspaces)
end
Base.:-(e::Array{T,1},a::MpoHamiltonian) where T = -1.0*a+e
Base.:+(e::Array{T,1},a::MpoHamiltonian) where T = a+e
Base.:-(a::MpoHamiltonian,e::AbstractArray{T,1}) where T = a+(-e)
function Base.:+(a::MpoHamiltonian{S,T,E},b::MpoHamiltonian{S,T,E}) where {S,T,E}
    @assert a.period == b.period
    @assert sanitycheck(a)
    @assert sanitycheck(b)

    #new odim;domspaces,imspaces (pspaces necessarily stays the same)
    nodim=a.odim+b.odim-2
    ndomspaces=similar(a.domspaces)
    nOs=[Array{Union{Missing,T},2}(missing,nodim,nodim) for i in 1:a.period]
    nSs=[Array{Union{Missing,E},1}(missing,nodim) for i in 1:a.period]

    for pos in 1:a.period
        ndomspace=Array{eltype(a.domspaces[pos]),1}(undef,nodim)

        #not entirely sure :3
        ndomspace[1:(a.odim-1)]=a.domspaces[pos][1:a.odim-1]
        ndomspace[a.odim:a.odim+b.odim-2]=b.domspaces[pos][2:b.odim]
        ndomspaces[pos]=ndomspace


        for (i,j) in keys(a,pos)
            #A block
            if(i<a.odim && j<a.odim)
                if i == j && isscal(a,pos,j)
                    nSs[pos][i] = a.scalars[pos][i]
                else
                    nOs[pos][i,j]=a.Os[pos][i,j]
                end
            end

            #right side
            if(i<a.odim && j==a.odim)
                nOs[pos][i,nodim]=a.Os[pos][i,j]
            end
        end

        for (i,j) in keys(b,pos)

            #upper Bs
            if(i==1 && j>1)
                if(j==b.odim && isassigned(nOs[pos],1,nodim))
                    nOs[pos][1,nodim]+=b.Os[pos][i,j]
                else
                    nOs[pos][1,a.odim+j-2]=b.Os[pos][i,j]
                end
            end

            #B block
            if(i>1 && j>1)
                if i == j && isscal(b,pos,i)
                    nSs[pos][a.odim+i-2]=b.scalars[pos][i]
                else
                    nOs[pos][a.odim+i-2,a.odim+j-2]=b.Os[pos][i,j]
                end
            end
        end
    end


    return MpoHamiltonian(Periodic(nSs),Periodic(nOs),ndomspaces,a.pspaces)
end
Base.:-(a::MpoHamiltonian,b::MpoHamiltonian) = a+(-1.0*b)

#multiplication
Base.:*(b::Number,a::MpoHamiltonian)=a*b
function Base.:*(a::MpoHamiltonian,b::Number)
    nOs=deepcopy(a.Os)

    for i=1:a.period
        for j=1:(a.odim-1)
            if(contains(a,i,j,a.odim))
                nOs[i][j,a.odim]*=b
            end
        end
    end

    return MpoHamiltonian(a.scalars,nOs,a.domspaces,a.pspaces)
end

#this is the index-map used in the ham x ham multiplication function (also needed somewhere else)
#i think julia has a build in for this, but it got renamed somewhere (linearindices?)
multmap(a::MpoHamiltonian,b::MpoHamiltonian) = (i,j)->(i-1)*b.odim+j
function Base.:*(a::MpoHamiltonian{S,T,E},b::MpoHamiltonian{S,T,E}) where {S,T,E}
    nodim=a.odim*b.odim

    indmap=multmap(a,b)

    nOs=[Array{Union{Missing,T},2}(missing,nodim,nodim) for i in 1:a.period]
    nSs=[Array{Union{Missing,E},1}(missing,nodim) for i in 1:a.period]
    ndomspaces=similar(a.domspaces)

    for pos=1:a.period
        ndomspace=Array{eltype(a.domspaces[pos]),1}(undef,nodim)

        for i in 1:a.odim
            for j in 1:b.odim
                ndomspace[indmap(i,j)]=fuse(a.domspaces[pos][i]*b.domspaces[pos][j])
            end
        end
        ndomspaces[pos]=ndomspace
    end

    for pos=1:a.period
        for (i,j) in keys(a,pos)
            for (k,l) in keys(b,pos)
                if i == j && k==l && isscal(a,pos,i) && isscal(b,pos,k)
                    nSs[pos][indmap(i,k)] = a.scalars[pos][i]*b.scalars[pos][k]
                else
                    @tensor newopp[-1 -2;-3 -4 -5 -6]:=a[pos,i,j][-1,1,-4,-6]*b[pos,k,l][-2,-3,-5,1]
                    newopp=TensorMap(newopp.data,ndomspaces[pos][indmap(i,k)],domain(newopp))
                    newopp=permute(newopp,(1,2,5),(3,4))
                    newopp=TensorMap(newopp.data,codomain(newopp),ndomspaces[pos+1][indmap(j,l)])
                    newopp=permute(newopp,(1,2),(4,3))

                    nOs[pos][indmap(i,k),indmap(j,l)]=newopp
                end
            end
        end
    end

    return MpoHamiltonian(Periodic(nSs),Periodic(nOs),ndomspaces,a.pspaces)
end

#without the copy, we get side effects when repeating + setindex
Base.repeat(x::MpoHamiltonian,n::Int) = MpoHamiltonian(
                                            Periodic(copy.(repeat(x.scalars,n))),
                                            Periodic(copy.(repeat(x.Os,n))),
                                            repeat(x.domspaces,n),
                                            repeat(x.pspaces,n))

#transpo = false => inplace conjugate
#transpo = true => flip physical legs
function Base.conj(a::MpoHamiltonian;transpo=false)
    b = Array{Union{Missing,eltype(a)},3}(missing,a.period,a.odim,a.odim)

    for (i,j,k) in keys(a)
        b[i,j,k] = @tensor temp[-1 -2;-3 -4]:=conj(a[i,j,k][-1,-2,-3,-4])
        if transpo
            b[i,j,k]=permute(b[i,j,k],(1,4),(3,2))
        end
    end

    MpoHamiltonian(b)
end

#needed this; perhaps move to tensorkit?
TensorKit.fuse(f::T) where T<: VectorSpace = f

#the usual mpoham transfer
mps_apply_transfer_left(vec::Array{V,1},ham::MpoHamiltonian,pos::Int,A::V,Ab::V=A) where V<:MpsType = mps_apply_transfer_left(V,vec,ham,pos,A,Ab)
mps_apply_transfer_right(vec::Array{V,1},ham::MpoHamiltonian,pos::Int,A::V,Ab::V=A) where V<:MpsType = mps_apply_transfer_right(V,vec,ham,pos,A,Ab)

#A is an excitation tensor; with an excitation leg
mps_apply_transfer_left(vec::Array{V,1},ham::MpoHamiltonian,pos::Int,A::M,Ab::V=A) where V<:MpsType where M <:MpoType = mps_apply_transfer_left(M,vec,ham,pos,A,Ab)
mps_apply_transfer_right(vec::Array{V,1},ham::MpoHamiltonian,pos::Int,A::M,Ab::V=A) where V<:MpsType where M <:MpoType = mps_apply_transfer_right(M,vec,ham,pos,A,Ab)

#v has an extra excitation leg
mps_apply_transfer_left(vec::Array{V,1},ham::MpoHamiltonian,pos::Int,A::M,Ab::M=A) where V<:MpoType where M <:MpsType = mps_apply_transfer_left(V,vec,ham,pos,A,Ab)
mps_apply_transfer_right(vec::Array{V,1},ham::MpoHamiltonian,pos::Int,A::M,Ab::M=A) where V<:MpoType where M <:MpsType = mps_apply_transfer_right(V,vec,ham,pos,A,Ab)

function mps_apply_transfer_left(RetType,vec,ham::MpoHamiltonian,pos,A,Ab=A)
    toreturn = Array{RetType,1}(undef,length(vec));
    assigned = [false for i in 1:ham.odim]

    for (j,k) in keys(ham,pos)
        if assigned[k]
            if j==k && isscal(ham,pos,j)
                toreturn[k]+=ham.scalars[pos][j]*mps_apply_transfer_left(vec[j],[A],[Ab])
            else
                toreturn[k]+=mps_apply_transfer_left(vec[j],ham[pos,j,k],A,Ab)
            end
        else
            if j==k && isscal(ham,pos,j)
                toreturn[k]=ham.scalars[pos][j]*mps_apply_transfer_left(vec[j],[A],[Ab])
            else
                toreturn[k]=mps_apply_transfer_left(vec[j],ham[pos,j,k],A,Ab)
            end
            assigned[k]=true
        end
    end


    for k in 1:ham.odim
        if !assigned[k]
            #prefereably this never happens, because it's a wasted step
            #it's also avoideable with a little bit more code
            toreturn[k]=mps_apply_transfer_left(vec[1],ham[pos,1,k],A,Ab)
        end
    end

    return toreturn
end
function mps_apply_transfer_right(RetType,vec,ham::MpoHamiltonian,pos,A,Ab=A)
    toreturn = Array{RetType,1}(undef,length(vec));
    assigned = [false for i in 1:ham.odim]

    for (j,k) in keys(ham,pos)
        if assigned[j]
            if j==k && isscal(ham,pos,j)
                toreturn[j]+=ham.scalars[pos][j]*mps_apply_transfer_right(vec[k],[A],[Ab])
            else
                toreturn[j]+=mps_apply_transfer_right(vec[k],ham[pos,j,k],A,Ab)
            end

        else
            if j==k && isscal(ham,pos,j)
                toreturn[j]=ham.scalars[pos][j]*mps_apply_transfer_right(vec[k],[A],[Ab])
            else
                toreturn[j]=mps_apply_transfer_right(vec[k],ham[pos,j,k],A,Ab)
            end
            assigned[j]=true
        end
    end

    for j in 1:ham.odim
        if !assigned[j]
            toreturn[j]=mps_apply_transfer_right(vec[1],ham[pos,j,1],A,Ab)
        end
    end

    return toreturn
end

#=
"turn the mpo hamiltonian into a full mpo"
function full(th :: MpoHamiltonian) #completely and utterly untested
    Os=[]

    #if you want to put this mpoham on a finite lattice, you want it to end with a trivial leg
    #this is the trivial leg you should use
    starters=[]
    stoppers=[]

    for i in 1:th.period

        #preliminaries
        totdom=reduce((a,b)-> fuse(a,b),th.domspaces[i])
        ou = oneunit(totdom)
        matrspace = reduce(⊕,[ou for i in 1:th.odim])

        #the current block
        curO = TensorMap(zeros,fuse(matrspace,totdom)*th.pspaces[i],fuse(matrspace,totdom)*th.pspaces[i])

        #add the blocks in curO
        for j in 1:th.odim
            front_top = fuse(reduce(*,th.domspaces[i][1:j],init=one(ou)))
            front_bot = fuse(reduce(*,th.domspaces[i][j+1:end],init=one(ou)))

            for k in 1:th.odim
                back_top = fuse(reduce(*,th.imspaces[i][1:k],init=one(ou)))
                back_bot = fuse(reduce(*,th.imspaces[i][k+1:end],init=one(ou)))

                frontback = TensorMap(I,front_top*front_bot,back_top'*back_bot')

                #embed the curent block in the total domain - image space
                @tensor cblock[-1 -2 -3;-4 -5 -6 -7 -8] := th[i,j,k][-2,-4,-6,-8]*frontback[-1,-3,-7,-5]
                cblock = TensorMap(cblock.data,fuse(codomain(cblock)),domain(cblock))
                cblock = permute(cblock,(1,2,6),(3,4,5))
                cblock = TensorMap(cblock.data,codomain(cblock),fuse(domain(cblock)))
                cblock = permute(cblock,(1,2),(4,3))

                #embed it in the matrix space
                embedder = TensorMap(matrspace,matrspace) do dims

                    @assert dims[1] == dims[2]
                    @assert length(dims)==2
                    @assert dims[1] == th.odim #if this isn't true, then I don't understand what's happening

                    tor = zeros(dims)
                    tor[j,k] = 1
                    return tor
                end

                @tensor cblock[-1 -2;-3 -4 -5 -6] := embedder[-1,-5]*cblock[-2,-3,-4,-6]
                cblock = TensorMap(cblock.data,fuse(codomain(cblock)),domain(cblock))
                cblock = permute(cblock,(1,2,5),(3,4))
                cblock = TensorMap(cblock.data,codomain(cblock),fuse(domain(cblock)))
                cblock = permute(cblock,(1,2),(4,3))

                curO += cblock
            end
        end
        push!(Os,curO)


        #make the starter
        leftutil = TensorMap(I,ou,totdom)
        leftmat = Tensor(matrspace) do x
            tor=zeros(x);tor[1,1] = 1;tor
        end
        @tensor leftstart[-1;-2 -3] := leftmat[-2]*leftutil[-1,-3]
        leftstart = TensorMap(leftstart.data,fuse(codomain(leftstart)),fuse(domain(leftstart)))
        push!(starters,leftstart)


        #make the stopper
        rightutil = TensorMap(I,totdom,ou)
        rightmat = Tensor(matrspace) do x
            tor=zeros(x);tor[end,1] = 1;tor
        end
        @tensor rightstop[-1 -2;-3] := rightmat[-1]*rightutil[-2,-3]
        rightstop = TensorMap(rightstop.data,fuse(codomain(rightstop)),fuse(domain(rightstop)))
        push!(stoppers,rightstop)

    end
    Periodic(Os),Periodic(starters),Periodic(circshift(stoppers,-1))
end
=#
