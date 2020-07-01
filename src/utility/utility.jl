function _permute_front(t::AbstractTensorMap) # make TensorMap{S,N₁+N₂-1,1}
    I = TensorKit.allind(t) # = (1:N₁+N₂...,)
    if BraidingStyle(sectortype(t)) isa SymmetricBraiding
        permute(t, Base.front(I), (I[end],))
    else
        levels = I
        braid(t, levels, Base.front(I), (I[end],))
    end
end
function _permute_tail(t::AbstractTensorMap) # make TensorMap{S,1,N₁+N₂-1}
    I = TensorKit.allind(t) # = (1:N₁+N₂...,)
    if BraidingStyle(sectortype(t)) isa SymmetricBraiding
        permute(t, (I[1],), Base.tail(I))
    else
        levels = I
        braid(t, levels, (I[1],), Base.tail(I))
    end
end
function _permute_as(t1::AbstractTensorMap, t2::AbstractTensorMap)
    if BraidingStyle(sectortype(t1)) isa SymmetricBraiding
        permute(t1, TensorKit.codomainind(t2), TensorKit.domainind(t2))
    else
        levels = allind(t1)
        braid(t1, TensorKit.codomainind(t2), TensorKit.domainind(t2))
    end
end
_firstspace(t::AbstractTensorMap) = space(t, 1)
_lastspace(t::AbstractTensorMap) = space(t, numind(t))

"
    Returns spin operators Sx,Sy,Sz,Id for spin s
"
function spinmatrices(s::Union{Rational{Int},Int})
    N = Int(2*s)

    Sx=zeros(Defaults.eltype,N+1,N+1)
    Sy=zeros(Defaults.eltype,N+1,N+1)
    Sz=zeros(Defaults.eltype,N+1,N+1)

    for row=1:(N+1)
        for col=1:(N+1)
            term=sqrt((s+1)*(row+col-1)-row*col)/2.0

            if (row+1==col)
                Sx[row,col]+=term
                Sy[row,col]-=1im*term
            end

            if(row==col+1)
                Sx[row,col]+=term
                Sy[row,col]+=1im*term
            end

            if(row==col)
                Sz[row,col]+=s+1-row
            end

        end
    end
    return Sx,Sy,Sz,one(Sx)
end

function nonsym_spintensors(s)
    (Sxd,Syd,Szd) = spinmatrices(s)
    sp = ComplexSpace(size(Sxd,1))

    Sx = TensorMap(Sxd,sp,sp);
    Sy = TensorMap(Syd,sp,sp);
    Sz = TensorMap(Szd,sp,sp);

    return Sx,Sy,Sz,one(Sx)
end

"""
bosonic creation anihilation operators with a cutoff
cutoff = maximal number of bosons at one location
"""
function nonsym_bosonictensors(cutoff::Int)
    creadat = zeros(Defaults.eltype,cutoff+1,cutoff+1);

    for i in 1:cutoff
        creadat[i+1,i] = sqrt(i);
    end

    a⁺ = TensorMap(creadat,ℂ^(cutoff+1),ℂ^(cutoff+1));
    a⁻ = TensorMap(collect(creadat'),ℂ^(cutoff+1),ℂ^(cutoff+1));
    return (a⁺,a⁻)
end
#given a hamiltonian with unit legs on the side, decompose it using svds to form a "localmpo"
function decompose_localmpo(inpmpo::AbstractTensorMap{PS,N1,N2}) where {PS,N1,N2}
    numind=N1+N2
    if(numind==4)
        return [permute(inpmpo,(1,2),(4,3))]
    end

    leftind=(1,2,Int(numind/2+1))
    otherind=(ntuple(x->x+2,Val{Int((N1+N2)/2)-2}())..., ntuple(x->x+Int(numind/2+1),Val{Int((N1+N2)/2)-1}())...)

    (U,S,V) = tsvd(inpmpo,leftind,otherind)

    T=U*S

    T=permute(T,(1,2),(4,3))


    return [T;decompose_localmpo(V)]
end

function add_util_leg(tensor::AbstractTensorMap{S,N1,N2}) where {S,N1,N2}
    #ntuple(x->x,Val{3+4}())

    util=Tensor(ones,eltype(tensor),oneunit(space(tensor,1)))
    tensor1=util*permute(tensor,(),ntuple(x->x,Val{N1+N2}()))
    return permute(tensor1,ntuple(x->x,Val{N1+N2+1}()),())*util'
end
