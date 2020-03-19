"""
    mutable struct FiniteMPS{A<:GenericMPSTensor} <: AbstractMPS

Represents a finite matrix product state
"""
mutable struct FiniteMPS{A<:GenericMPSTensor} <: AbstractMPS
    tensors::Vector{A}
    centerpos::UnitRange{Int} # range of tensors which are not left or right normalized
    function FiniteMPS{A}(tensors::Vector{A},
                            centerpos::UnitRange{Int}) where {A<:GenericMPSTensor}
        _firstspace(tensors[1]) == oneunit(_firstspace(tensors[1])) ||
            throw(SectorMismatch("Leftmost virtual index of MPS should be trivial."))
        N = length(tensors)
        for n = 1:N-1
            dual(_lastspace(tensors[n])) == _firstspace(tensors[n+1]) ||
                throw(SectorMismatch("Non-matching virtual index on bond $n."))
        end
        dim(_lastspace(tensors[N])) == 1 ||
            throw(SectorMismatch("Rightmost virtual index should be one-dimensional."))
        return new{A}(tensors, centerpos)
    end
end
FiniteMPS(tensors::Vector{A},
            centerpos::UnitRange{Int} = 1:length(tensors)) where {A<:GenericMPSTensor} =
    FiniteMPS{A}(tensors, centerpos)

function FiniteMPS(f, P::ProductSpace, args...; kwargs...)
    return FiniteMPS(f, collect(P), args...; kwargs...)
end

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
    if f == randisometry
        return FiniteMPS(tensors, N:N)
    else
        return FiniteMPS(tensors, 1:N)
    end
end

Base.copy(psi::FiniteMPS) = FiniteMPS(map(copy, psi.tensors), psi.centerpos)

function Base.getproperty(psi::FiniteMPS,prop::Symbol)
    if prop == :AL
        return ALView(psi)
    elseif prop == :AR
        return ARView(psi)
    elseif prop == :AC
        return ACView(psi)
    elseif prop == :CR
        return CRView(psi)
    elseif prop == :A
        return AView(psi)
    else
        return getfield(psi,prop)
    end
end

Base.length(psi::FiniteMPS) = length(psi.tensors)
Base.size(psi::FiniteMPS, i...) = size(psi.tensors, i...)
Base.eltype(::Type{FiniteMPS{A}}) where {A<:GenericMPSTensor} = A
Base.similar(psi::FiniteMPS) = FiniteMPS(similar.(psi.tensors))

TensorKit.space(psi::FiniteMPS{<:MPSTensor}, n::Integer) = space(psi.A[n], 2)
function TensorKit.space(psi::FiniteMPS{<:GenericMPSTensor}, n::Integer)
    t = psi.A[n]
    S = spacetype(t)
    return ProductSpace{S}(space.(Ref(t), Base.front(Base.tail(TensorKit.allind(t)))))
end

virtualspace(psi::FiniteMPS, n::Integer) =
    n < length(psi) ? _firstspace(psi.A[n+1]) : dual(_lastspace(psi.A[n]))

r_RR(state::FiniteMPS{T}) where T = isomorphism(Matrix{eltype(T)},domain(state.A[end]),domain(state.A[end]))
l_LL(state::FiniteMPS{T}) where T = isomorphism(Matrix{eltype(T)},space(state.A[1],1),space(state.A[1],1))

# Gauging left and right
"""
    function leftorth!(psi::FiniteMPS, n = length(psi); normalize = true, alg = QRpos())

Bring all MPS tensors of `psi` to the left of site `n` into left orthonormal form, using
the orthogonal factorization algorithm `alg`
"""
function TensorKit.leftorth!(psi::FiniteMPS, n::Integer = length(psi);
                    alg::OrthogonalFactorizationAlgorithm = QRpos(),
                    normalize::Bool = true)
    @assert 1 <= n <= length(psi)
    while first(psi.centerpos) < n
        k = first(psi.centerpos)
        AL, C = leftorth!(psi.tensors[k]; alg = alg)
        psi.tensors[k] = AL
        psi.tensors[k+1] = _permute_front(C*_permute_tail(psi.tensors[k+1]))
        psi.centerpos = k+1:max(k+1,last(psi.centerpos))
    end
    return normalize ? normalize!(psi) : psi
end
function TensorKit.rightorth!(psi::FiniteMPS, n::Integer = 1;
                    alg::OrthogonalFactorizationAlgorithm = LQpos(),
                    normalize::Bool = true)
    @assert 1 <= n <= length(psi)
    while last(psi.centerpos) > n
        k = last(psi.centerpos)
        C, AR = rightorth!(_permute_tail(psi.tensors[k]); alg = alg)
        psi.tensors[k] = _permute_front(AR)
        psi.tensors[k-1] = psi.tensors[k-1]*C
        psi.centerpos = min(first(psi.centerpos), k-1):k-1
    end
    return normalize ? normalize!(psi) : psi
end

# Linear algebra methods
#------------------------
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

function Base.:*(psi::FiniteMPS, a::Number)
    psi′ = FiniteMPS(psi.tensors .* one(a) , psi.centerpos)
    return rmul!(psi′, a)
end

function Base.:*(a::Number, psi::FiniteMPS)
    psi′ = FiniteMPS(one(a) .* psi.tensors, psi.centerpos)
    return lmul!(a, psi′)
end

function TensorKit.lmul!(a::Number, psi::FiniteMPS)
    lmul!(a, psi.tensors[first(psi.centerpos)])
    return psi
end

function TensorKit.rmul!(psi::FiniteMPS, a::Number)
    rmul!(psi.tensors[first(psi.centerpos)], a)
    return psi
end

function TensorKit.dot(psi1::FiniteMPS, psi2::FiniteMPS)
    length(psi1) == length(psi2) || throw(ArgumentError("MPS with different length"))

    ρL = _permute_front(psi1.A[1])' * _permute_front(psi2.A[1])
    for k in 2:length(psi1)
        ρL = transfer_left(ρL, psi2.A[k], psi1.A[k])
    end
    return tr(ρL)
end

function TensorKit.norm(psi::FiniteMPS)
    k = first(psi.centerpos)
    if k == last(psi.centerpos)
        return norm(psi.A[k])
    else
        _, C = leftorth(psi.A[k])
        k += 1
        while k <= last(psi.centerpos)
            _, C = leftorth!(_permute_front(C * _permute_tail(psi.A[k])))
            k += 1
        end
        return norm(C)
    end
end

TensorKit.normalize!(psi::FiniteMPS) = rmul!(psi, 1/norm(psi))
TensorKit.normalize(psi::FiniteMPS) = normalize!(copy(psi))


"
    A FiniteMPS/MPO starts and ends with a bond dimension 1 leg
    It's bond dimension also can't grow faster then pspaces (resp. pspaces^2)
    This function calculates the maximal achievable bond dimension
"
function max_Ds(f::FiniteMPS{G}) where G<:GenericMPSTensor{S,N} where {S,N}
    Ds = [1 for v in 1:length(f)+1];
    for i in 1:length(f)
        Ds[i+1] = Ds[i]*prod(map(x->dim(space(f.A[i],x)),ntuple(x->x+1,Val{N-1}())))
    end

    Ds[end] = 1;
    for i in length(f):-1:1
        Ds[i] = min(Ds[i],Ds[i+1]*prod(map(x->dim(space(f.A[i],x)),ntuple(x->x+1,Val{N-1}()))))
    end
    Ds
end
