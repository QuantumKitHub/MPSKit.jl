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

Base.copy(psi::FiniteMPS) = FiniteMPS(map(copy, psi.tensors), copy(psi.normalizations))

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

"
    take the sum of 2 finite mpses
"
function Base.:+(v1::FiniteMPS{T},v2::FiniteMPS{T}) where T #untested and quite horrible code, but not sure how to make it nice
    @assert length(v1)==length(v2)

    ou = oneunit(space(v1[1],1));

    m1 = TensorMap(rand,eltype(T),ou,ou⊕ou);
    (_,m1) = rightorth(m1);
    m2 = rightnull(m1);

    pm1 = m1+m2;

    tot = similar(v1);

    for i = 1:length(v1)
        nm1 = TensorMap(rand,eltype(T),space(v1[i],3)',space(v1[i],3)⊕space(v2[i],3));
        (_,nm1) = rightorth(nm1);
        nm2 = rightnull(nm1);

        @tensor t[-1 -2;-3] := conj(m1[1,-1])*v1[i][1,-2,2]*nm1[2,-3]
        @tensor t[-1 -2;-3] += conj(m2[1,-1])*v2[i][1,-2,2]*nm2[2,-3]

        tot[i] = t;

        m1 = nm1;
        m2 = nm2;
    end

    pm2 = m1+m2;

    @tensor tot[1][-1 -2;-3] := pm1[-1,1]*tot[1][1,-2,-3]
    @tensor tot[end][-1 -2;-3] := tot[end][-1,-2,1]*conj(pm2[-3,1])

    return tot
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
