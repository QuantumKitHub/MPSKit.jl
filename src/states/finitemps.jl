"""
    mutable struct FiniteMPS{A<:GenericMPSTensor,B<:MPSBondTensor} <: AbstractMPS

Represents a finite matrix product state
"""
mutable struct FiniteMPS{A<:GenericMPSTensor,B<:MPSBondTensor} <: AbstractMPS
    site_tensors::Vector{A}
    bond_tensors::Vector{Union{Missing,B}} #contains CL

    #gaugedpos[1] = last left gauged tensor
    #gaugedpos[2] = last right gauged tensor
    gaugedpos::Tuple{Int,Int}

    function FiniteMPS{A,B}(site_tensors::Vector{A},bond_tensors::Vector{Union{Missing,B}},
                            gaugedpos::Tuple{Int,Int}) where {A<:GenericMPSTensor,B<:MPSBondTensor}

        # checking the site tensor spaces
        _firstspace(site_tensors[1]) == oneunit(_firstspace(site_tensors[1])) ||
            throw(SectorMismatch("Leftmost virtual index of MPS should be trivial."))

        N = length(site_tensors)
        for n = 1:N-1
            dual(_lastspace(site_tensors[n])) == _firstspace(site_tensors[n+1]) ||
                throw(SectorMismatch("Non-matching virtual index on bond $n."))
        end
        dim(_lastspace(site_tensors[N])) == 1 ||
            throw(SectorMismatch("Rightmost virtual index should be one-dimensional."))

        # checking the bond tensor spaces
        length(bond_tensors) == N+1 ||
            throw(DimensionMismatch("bond_tensors incorrect length"))

        for n = 1:N
            ismissing(bond_tensors[n]) ||
                dual(_lastspace(bond_tensors[n])) == _firstspace(space_tensors[n]) ||
                throw(SectorMismatch("bond tensor doesn't fit on space tensor"))

            ismissing(bond_tensors[n+1]) ||
                dual(_firstspace(bond_tensors[n+1])) == _lastspace(space_tensors[n]) ||
                throw(SectorMismatch("bond tensor doesn't fit on space tensor"))
        end

        return new{A,B}(site_tensors, bond_tensors, gaugedpos)
    end
end

#this should not be necessary ...
bond_type(t::GenericMPSTensor) = typeof(TensorMap(rand,eltype(t),space(t,1),space(t,1)))

# allow construction with only site_tensors missing
function FiniteMPS(site_tensors::Vector{A},
            bond_tensors::Vector{Union{Missing,B}} = Vector{Union{Missing,bond_type(site_tensors[1])}}(missing,length(site_tensors)+1),
            gaugedpos::Tuple{Int,Int} = (0,length(site_tensors)+1)) where {A<:GenericMPSTensor, B <: MPSBondTensor}
    FiniteMPS{A,B}(site_tensors, bond_tensors, gaugedpos)
end

# allow construction with one large tensorkit space
function FiniteMPS(f, P::ProductSpace, args...; kwargs...)
    return FiniteMPS(f, collect(P), args...; kwargs...)
end

# allow construction given only a physical space and length
function FiniteMPS(f, N::Int, V::VectorSpace, args...; kwargs...)
    return FiniteMPS(f, fill(V, N), args...; kwargs...)
end

function FiniteMPS(f, physspaces::Vector{<:Union{S,CompositeSpace{S}}}, maxvirtspace::S;
                    left::S = oneunit(maxvirtspace),
                    right::S = oneunit(maxvirtspace)) where {S<:ElementarySpace}
    N = length(physspaces)
    virtspaces = Vector{S}(undef, N+1)
    virtspaces[1] = left
    for k = 2:N
        virtspaces[k] = min(fuse(virtspaces[k-1], fuse(physspaces[k])), maxvirtspace)
    end
    virtspaces[N+1] = right
    for k = N:-1:2
        virtspaces[k] = min(virtspaces[k], fuse(virtspaces[k+1], flip(fuse(physspaces[k]))))
    end
    return FiniteMPS(f, physspaces, virtspaces)
end
function FiniteMPS(f,
                    physspaces::Vector{<:Union{S,CompositeSpace{S}}},
                    virtspaces::Vector{S}) where {S<:ElementarySpace}
    N = length(physspaces)
    length(virtspaces) == N+1 || throw(DimensionMismatch())

    tensors = [TensorMap(f, virtspaces[n] ⊗ physspaces[n], virtspaces[n+1]) for n=1:N]

    return FiniteMPS(tensors)
end

Base.copy(psi::FiniteMPS) = FiniteMPS(deepcopy(psi.site_tensors), deepcopy(psi.bond_tensors), psi.gaugedpos)

function Base.getproperty(psi::FiniteMPS,prop::Symbol)
    if prop == :AL
        return ALView(psi)
    elseif prop == :AR
        return ARView(psi)
    elseif prop == :AC
        return ACView(psi)
    elseif prop == :CR
        return CRView(psi)
    else
        return getfield(psi,prop)
    end
end

Base.length(psi::FiniteMPS) = length(psi.site_tensors)
Base.size(psi::FiniteMPS, i...) = size(psi.site_tensors, i...)

#conflicted if this is actually true
Base.eltype(st::FiniteMPS{A}) where {A<:GenericMPSTensor} = A
Base.eltype(::Type{FiniteMPS{A}}) where {A<:GenericMPSTensor} = A

Base.similar(psi::FiniteMPS) = FiniteMPS(similar.(psi.site_tensors),similar.(psi.bond_tensors))

TensorKit.space(psi::FiniteMPS{<:MPSTensor}, n::Integer) = space(psi.site_tensors[n], 2)
function TensorKit.space(psi::FiniteMPS{<:GenericMPSTensor}, n::Integer)
    t = psi.site_tensors[n]
    S = spacetype(t)
    return ProductSpace{S}(space.(Ref(t), Base.front(Base.tail(TensorKit.allind(t)))))
end

virtualspace(psi::FiniteMPS, n::Integer) =
    n < length(psi) ? _firstspace(psi.site_tensors[n+1]) : dual(_lastspace(psi.site_tensors[n]))

r_RR(state::FiniteMPS{T}) where T = isomorphism(Matrix{eltype(T)},domain(state.site_tensors[end]),domain(state.site_tensors[end]))
l_LL(state::FiniteMPS{T}) where T = isomorphism(Matrix{eltype(T)},space(state.site_tensors[1],1),space(state.site_tensors[1],1))

# Gauging left and right
"""
    function leftorth!(psi::FiniteMPS, n = length(psi); normalize = true, alg = QRpos())

Bring all MPS tensors of `psi` up to and including site `n` into left orthonormal form, using
the orthogonal factorization algorithm `alg`
"""
function TensorKit.leftorth!(psi::FiniteMPS, n::Integer = length(psi);
                    alg::OrthogonalFactorizationAlgorithm = Polar(),
                    normalize::Bool = true)
    @assert 1 <= n <= length(psi)

    while first(psi.gaugedpos) < n
        k = first(psi.gaugedpos) + 1

        if !ismissing(psi.bond_tensors[k])
            C = psi.bond_tensors[k];
            psi.bond_tensors[k] = missing;
            psi.site_tensors[k] = _permute_front(C*_permute_tail(psi.site_tensors[k+1]))
        end

        psi.site_tensors[k], psi.bond_tensors[k+1] = leftorth!(psi.site_tensors[k]; alg = alg)

        psi.gaugedpos = (k,last(psi.gaugedpos))
    end
    return normalize ? normalize!(psi) : psi
end
function TensorKit.rightorth!(psi::FiniteMPS, n::Integer = 1;
                    alg::OrthogonalFactorizationAlgorithm = Polar(),
                    normalize::Bool = true)
    @assert 1 <= n <= length(psi)
    while last(psi.gaugedpos) > n
        k = last(psi.gaugedpos) - 1

        if !ismissing(psi.bond_tensors[k+1])
            C = psi.bond_tensors[k+1];
            psi.bond_tensors[k+1] = missing;
            psi.site_tensors[k] = psi.site_tensors[k]*C
        end

        C, AR = rightorth!(_permute_tail(psi.site_tensors[k]); alg = alg)

        psi.site_tensors[k] = _permute_front(AR)
        psi.bond_tensors[k] = C;

        psi.gaugedpos = (first(psi.gaugedpos), k)
    end

    return normalize ? normalize!(psi) : psi
end

# Linear algebra methods
#------------------------
#=

This code is close to working, but the resulting finitemps is non injective
We need injectivity to make sure that psi.AL is uniquely defined
We need psi.AL to be uniquely defined to make sure that the caches work
Therefore, perhaps this shouldn't be included?

function Base.:+(psi1::FiniteMPS, psi2::FiniteMPS)
    length(psi1) == length(psi2) || throw(DimensionMismatch())

    N = length(psi1)
    for k = 1:N
        space(psi1, k) == space(psi2, k) ||
            throw(SpaceMismatch("Non-matching physical space on site $k."))
    end
    virtualspace(psi1, 0) == virtualspace(psi2, 0) ||
        throw(SpaceMismatch("Non-matching left virtual space."))
    virtualspace(psi1, N) == virtualspace(psi2, N) ||
        throw(SpaceMismatch("Non-matching right virtual space."))

    k = 1 # firstindex(psi1)
    t1 = psi1.A[k]
    t2 = psi2.A[k]
    V1 = domain(t1)[1]
    V2 = domain(t2)[1]
    w1 = isometry(storagetype(psi1.A[1]), V1 ⊕ V2, V1)
    w2 = leftnull(w1)
    @assert domain(w2) == ⊗(V2)
    t = t1*w1' + t2*w2'
    tensors = similar(psi1.tensors, typeof(t))
    tensors[1] = t
    for k = 2:N-1
        t1 = _permute_front(w1*_permute_tail(psi1.A[k]))
        t2 = _permute_front(w2*_permute_tail(psi2.A[k]))
        V1 = domain(t1)[1]
        V2 = domain(t2)[1]
        w1 = isometry(storagetype(psi1.A[1]), V1 ⊕ V2, V1)
        w2 = leftnull(w1)
        @assert domain(w2) == ⊗(V2)
        tensors[k] = t1*w1' + t2*w2'
    end
    k = N
    t1 = _permute_front(w1*_permute_tail(psi1.A[k]))
    t2 = _permute_front(w2*_permute_tail(psi2.A[k]))
    tensors[k] = t1 + t2
    return FiniteMPS(tensors)
end
=#

function Base.:*(psi::FiniteMPS, a::Number)
    psi′ = FiniteMPS(psi.site_tensors .* one(a) , psi.bond_tensors .* one(a), psi.gaugedpos)
    return rmul!(psi′, a)
end

function Base.:*(a::Number, psi::FiniteMPS)
    psi′ = FiniteMPS(one(a) .* psi.site_tensors, psi.bond_tensors .* one(a), psi.gaugedpos)
    return lmul!(a, psi′)
end

function TensorKit.lmul!(a::Number, psi::FiniteMPS)
    if first(psi.gaugedpos) + 1 == last(psi.gaugedpos)
        #every tensor is either left or right gauged => the bond tensor is centergauged
        @assert !ismissing(psi.bond_tensors[first(psi.gaugedpos)+1])

        lmul!(a, psi.bond_tensors[first(psi.gaugedpos)+1])
    else
        lmul!(a, psi.site_tensors[first(psi.gaugedpos)+1])
    end

    return psi
end

function TensorKit.rmul!(a::Number, psi::FiniteMPS)
    if first(psi.gaugedpos) + 1 == last(psi.gaugedpos)
        #every tensor is either left or right gauged => the bond tensor is centergauged
        @assert !ismissing(psi.bond_tensors[first(psi.gaugedpos)+1])

        rmul!(a, psi.bond_tensors[first(psi.gaugedpos)+1])
    else
        rmul!(a, psi.site_tensors[first(psi.gaugedpos)+1])
    end

    return psi
end

function TensorKit.dot(psi1::FiniteMPS, psi2::FiniteMPS)
    #todo : rewrite this without having to gauge
    length(psi1) == length(psi2) || throw(ArgumentError("MPS with different length"))

    return tr(_permute_front(psi1.AC[1])' * _permute_front(psi2.AC[1]))
end

function TensorKit.norm(psi::FiniteMPS)
    #todo : rewrite this without having to gauge

    if first(psi.gaugedpos) == length(psi)
        #everything is left gauged
        #the bond tensor should be center gauged
        @assert !ismissing(psi.bond_tensors[first(psi.gaugedpos)+1])
        return norm(psi.bond_tensors[first(psi.gaugedpos)+1])
    elseif last(psi.gaugedpos) == 1
        #everything is right gauged
        #the bond tensor should be center gauged
        @assert !ismissing(psi.bond_tensors[1])
        return norm(psi.bond_tensors[1])
    else
        return norm(psi.AC[first(psi.gaugedpos)+1])
    end
end

TensorKit.normalize!(psi::FiniteMPS) = rmul!(psi, 1/norm(psi))
TensorKit.normalize(psi::FiniteMPS) = normalize!(copy(psi))


"
    A FiniteMPS/MPO starts and ends with a bond dimension 1 leg
    Its bond dimension also can't grow faster then pspaces (resp. pspaces^2)
    This function calculates the maximal achievable bond dimension
"
function max_Ds(f::FiniteMPS{G}) where G<:GenericMPSTensor{S,N} where {S,N}
    Ds = [1 for v in 1:length(f)+1];
    for i in 1:length(f)
        Ds[i+1] = Ds[i]*prod(map(x->dim(space(f.site_tensors[i],x)),ntuple(x->x+1,Val{N-1}())))
    end

    Ds[end] = 1;
    for i in length(f):-1:1
        Ds[i] = min(Ds[i],Ds[i+1]*prod(map(x->dim(space(f.site_tensors[i],x)),ntuple(x->x+1,Val{N-1}()))))
    end
    Ds
end
