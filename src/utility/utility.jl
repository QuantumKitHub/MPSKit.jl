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

"""
Take the L2 Tikhonov regularised inverse of a matrix `m`.

The regularisation parameter is the larger of `delta` (the optional argument that defaults
to zero) and square root of machine epsilon. The inverse is done using an SVD.
"""
function reginv(m, delta=zero(eltype(m)))
    delta = max(abs(delta), sqrt(eps(real(float(one(eltype(m)))))))
    U, S, Vdg = tsvd(m)
    Sinv = inv(real(sqrt(S^2 + delta^2*one(S))))
    minv = Vdg' * Sinv * U'
    return minv
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

"
checks if the given 4leg tensor is the identity (needed for infinite mpo hamiltonians)
"
function isid(x::MPOTensor;tol=Defaults.tolgauge)
    (_firstspace(x) == _lastspace(x)' && space(x,2) == space(x,3)') || return false,zero(eltype(x));
    _can_unambiguously_braid(_firstspace(x)) || return false,zero(eltype(x));

    id = isomorphism(Matrix{eltype(x)},codomain(x),domain(x))
    scal = dot(id,x)/dot(id,id)
    diff = x-scal*id

    return norm(diff)<tol,scal
end

#needed this; perhaps move to tensorkit?
TensorKit.fuse(f::T) where T<: VectorSpace = f
