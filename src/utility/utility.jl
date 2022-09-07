function _transpose_front(t::AbstractTensorMap) # make TensorMap{S,N₁+N₂-1,1}
    I1 = TensorKit.codomainind(t)
    I2 = TensorKit.domainind(t)
    transpose(t, (I1..., reverse(Base.tail(I2))...), (I2[1],))
end
function _transpose_tail(t::AbstractTensorMap) # make TensorMap{S,1,N₁+N₂-1}
    I1 = TensorKit.codomainind(t)
    I2 = TensorKit.domainind(t)
    transpose(t, (I1[1],), (I2..., reverse(Base.tail(I1))...))
end
function _transpose_as(t1::AbstractTensorMap, t2::AbstractTensorMap{S,N1,N2}) where {S,N1,N2}
    I1 = (TensorKit.codomainind(t1)...,reverse(TensorKit.domainind(t1))...);

    A = ntuple(x->I1[x],N1);
    B = ntuple(x->I1[x+N1],N2);

    transpose(t1, A,B)
end

_firstspace(t::AbstractTensorMap) = space(t, 1)
_lastspace(t::AbstractTensorMap) = space(t, numind(t))

#given a hamiltonian with unit legs on the side, decompose it using svds to form a "localmpo"
function decompose_localmpo(inpmpo::AbstractTensorMap{PS,N,N},trunc = truncbelow(Defaults.tol)) where {PS,N}
    N == 2 && return [transpose(inpmpo,(1,2),(3,4))]

    leftind = (N+1,1,2)
    rightind = (ntuple(x->x+N+1,N-1)...,reverse(ntuple(x->x+2,N-2))...);
    (U,S,V) = tsvd(transpose(inpmpo,leftind,rightind),trunc = trunc)

    A = transpose(U,(2,3),(1,4));
    B = transpose(S*V,(1,reverse(ntuple(x->x+N,N-2))...),ntuple(x->x+1,N-1))
    return [A;decompose_localmpo(B)]
end

function add_util_leg(tensor::AbstractTensorMap{S,N1,N2}) where {S,N1,N2}
    ou = oneunit(_firstspace(tensor));

    util_front = isomorphism(storagetype(tensor),ou*codomain(tensor),codomain(tensor));
    util_back = isomorphism(storagetype(tensor),domain(tensor),domain(tensor)*ou);

    return util_front*tensor*util_back
end

function union_split(a::AbstractArray)
    T = reduce((a,b)->Union{a,b},typeof.(a))
    nA = similar(a,T);
    copy!(nA,a)
end
union_types(x::Union) = (x.a, union_types(x.b)...)
union_types(x::Type) = (x,)

function _embedders(spaces)
    totalspace = reduce(⊕,spaces);

    maps = [isometry(totalspace,first(spaces))];
    restmap = leftnull(first(maps));

    for sp in spaces[2:end]
        cm = isometry(domain(restmap),sp);

        push!(maps,restmap*cm);
        restmap = restmap*leftnull(cm);
    end

    maps
end

function _can_unambiguously_braid(sp::VectorSpace)
    s = sectortype(sp);

    #either the braidingstyle is bosonic
    BraidingStyle(s) isa Bosonic && return true

    #or there is only one irrep ocurring - the trivial one
    for sect in sectors(sp)
        sect == one(sect) || return false
    end
    return true
end

#needed this; perhaps move to tensorkit?
TensorKit.fuse(f::T) where T<: VectorSpace = f


function safe_xlogx(t::AbstractTensorMap,eps = eps(real(eltype(t))))
    (U,S,V) = tsvd(t,alg = SVD(), trunc = truncbelow(eps));
    U*S*log(S)*V
end
