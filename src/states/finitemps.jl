"
    FiniteMPS(data::Array)

    finite one dimensional mps
    algorithms usually assume a right-orthormalized input
"
struct FiniteMPS{A<:GenericMPSTensor} <: AbstractMPS
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

Base.@propagate_inbounds Base.getindex(psi::FiniteMPS, args...) =
    getindex(psi.tensors, args...)
Base.@propagate_inbounds Base.setindex!(psi::FiniteMPS, args...) =
    setindex!(psi.tensors, args...)

Base.length(psi::FiniteMPS) = length(psi.tensors)
Base.size(psi::FiniteMPS, i...) = size(psi.tensors, i...)
Base.firstindex(psi::FiniteMPS, i...) = firstindex(psi.tensors, i...)
Base.lastindex(psi::FiniteMPS, i...) = lastindex(psi.tensors, i...)

Base.iterate(psi::FiniteMPS, i...) = iterate(psi.tensors, i...)
Base.IteratorSize(::Type{<:FiniteMPS}) = Base.HasShape{1}()
Base.IteratorEltype(::Type{<:FiniteMPS}) = Base.HasEltype()

Base.eltype(::Type{FiniteMPS{A}}) where {A<:GenericMPSTensor} = A
Base.similar(psi::FiniteMPS) = FiniteMPS(similar.(psi.tensors))

TensorKit.space(psi::FiniteMPS{<:MPSTensor}, n::Integer) = space(psi[n], 2)
function TensorKit.space(psi::FiniteMPS{<:GenericMPSTensor}, n::Integer)
    t = psi[n]
    S = spacetype(t)
    return ProductSpace{S}(space.(Ref(t), Base.front(Base.tail(TensorKit.allind(t)))))
end

virtualspace(psi::FiniteMPS, n::Integer) =
    n < length(psi) ? _firstspace(psi[n+1]) : dual(_lastspace(psi[n]))

r_RR(state::FiniteMPS{T}) where T = isomorphism(Matrix{eltype(T)},domain(state[end]),domain(state[end]))
l_LL(state::FiniteMPS{T}) where T = isomorphism(Matrix{eltype(T)},space(state[1],1),space(state[1],1))

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
    t1 = psi1[k]
    t2 = psi2[k]
    V1 = domain(t1)[1]
    V2 = domain(t2)[1]
    w1 = isometry(storagetype(psi1[1]), V1 ⊕ V2, V1)
    w2 = leftnull(w1)
    @assert domain(w2) == ⊗(V2)
    t = t1*w1' + t2*w2'
    tensors = similar(psi1.tensors, typeof(t))
    tensors[1] = t
    for k = 2:N-1
        t1 = _permute_front(w1*_permute_tail(psi1[k]))
        t2 = _permute_front(w2*_permute_tail(psi2[k]))
        V1 = domain(t1)[1]
        V2 = domain(t2)[1]
        w1 = isometry(storagetype(psi1[1]), V1 ⊕ V2, V1)
        w2 = leftnull(w1)
        @assert domain(w2) == ⊗(V2)
        tensors[k] = t1*w1' + t2*w2'
    end
    k = N
    t1 = _permute_front(w1*_permute_tail(psi1[k]))
    t2 = _permute_front(w2*_permute_tail(psi2[k]))
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

    ρL = _permute_front(psi1[1])' * _permute_front(psi2[1])
    for k in 2:length(psi1)
        ρL = transfer_left(ρL, psi2[k], psi1[k])
    end
    return tr(ρL)
end

TensorKit.norm(psi::FiniteMPS) = sqrt(real(dot(psi,psi)))

TensorKit.normalize!(psi::FiniteMPS) = rmul!(psi, 1/norm(psi))
TensorKit.normalize(psi::FiniteMPS) = normalize!(copy(psi))
