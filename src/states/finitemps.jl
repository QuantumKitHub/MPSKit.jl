"""
    FiniteMPS{A<:GenericMPSTensor,B<:MPSBondTensor} <: AbstractFiniteMPS

Type that represents a finite Matrix Product State.

## Fields
- `ALs` -- left-gauged MPS tensors
- `ARs` -- right-gauged MPS tensors
- `ACs` -- center-gauged MPS tensors
- `CLs` -- gauge tensors

Where each is entry can be a tensor or `missing`.

## Notes
By convention, we have that:
- `AL[i] * CL[i+1]` = `AC[i]` = `CL[i] * AR[i]`
- `AL[i]' * AL[i] = 1`
- `AR[i] * AR[i]' = 1`

---

## Constructors
    FiniteMPS([f, eltype], physicalspaces::Vector{<:Union{S, CompositeSpace{S}},
              virtualspaces::Vector{<:Union{S, CompositeSpace{S}};
              normalize=true) where {S<:ElementarySpace}
    FiniteMPS([f, eltype], physicalspaces::Vector{<:Union{S,CompositeSpace{S}}},
              maxvirtualspace::S;
              normalize=true, left=oneunit(S), right=oneunit(S)) where {S<:ElementarySpace}
    FiniteMPS([f, eltype], N::Int, physicalspace::Union{S,CompositeSpace{S}},
              maxvirtualspace::S;
              normalize=true, left=oneunit(S), right=oneunit(S)) where {S<:ElementarySpace}
    FiniteMPS(As::Vector{<:GenericMPSTensor}; normalize=false, overwrite=false)

Construct an MPS via a specification of physical and virtual spaces, or from a list of
tensors `As`. All cases reduce to the latter.

### Arguments
- `As::Vector{<:GenericMPSTensor}`: vector of site tensors

- `f::Function=rand`: initializer function for tensor data
- `eltype::Type{<:Number}=ComplexF64`: scalar type of tensors

- `physicalspaces::Vector{<:Union{S, CompositeSpace{S}}`: list of physical spaces
- `N::Int`: number of sites
- `physicalspace::Union{S,CompositeSpace{S}}`: local physical space

- `virtualspaces::Vector{<:Union{S, CompositeSpace{S}}`: list of virtual spaces
- `maxvirtualspace::S`: maximum virtual space

### Keywords
- `normalize`: normalize the constructed state
- `overwrite=false`: overwrite the given input tensors
- `left=oneunit(S)`: left-most virtual space
- `right=oneunit(S)`: right-most virtual space
"""
struct FiniteMPS{A<:GenericMPSTensor,B<:MPSBondTensor} <: AbstractFiniteMPS
    ALs::Vector{Union{Missing,A}}
    ARs::Vector{Union{Missing,A}}
    ACs::Vector{Union{Missing,A}}
    CLs::Vector{Union{Missing,B}}

    function FiniteMPS(ALs::Vector{Union{Missing,A}}, ARs::Vector{Union{Missing,A}},
                       ACs::Vector{Union{Missing,A}},
                       CLs::Vector{Union{Missing,B}}) where
             {A<:GenericMPSTensor,B<:MPSBondTensor}
        length(ACs) == length(CLs) - 1 == length(ALs) == length(ARs) ||
            throw(DimensionMismatch("length mismatch of tensors"))
        sum(ismissing.(ACs)) + sum(ismissing.(CLs)) < length(ACs) + length(CLs) ||
            throw(ArgumentError("at least one AC/CL should not be missing"))

        S = spacetype(A)
        left_virt_spaces = Vector{Union{Missing,S}}(missing, length(CLs))
        right_virt_spaces = Vector{Union{Missing,S}}(missing, length(CLs))

        for (i, tup) in enumerate(zip(ALs, ARs, ACs))
            non_missing = filter(!ismissing, tup)
            isempty(non_missing) && throw(ArgumentError("missing site tensor"))
            (al, ar, ac) = tup

            if !ismissing(al)
                !ismissing(left_virt_spaces[i]) &&
                    (left_virt_spaces[i] == _firstspace(al) ||
                     throw(SpaceMismatch("Virtual space of AL on site $(i) doesn't match")))

                left_virt_spaces[i + 1] = _lastspace(al)'
                left_virt_spaces[i] = _firstspace(al)
            end

            if !ismissing(ar)
                !ismissing(right_virt_spaces[i]) &&
                    (right_virt_spaces[i] == _firstspace(ar) ||
                     throw(SpaceMismatch("Virtual space of AR on site $(i) doesn't match")))

                right_virt_spaces[i + 1] = _lastspace(ar)'
                right_virt_spaces[i] = _firstspace(ar)
            end

            if !ismissing(ac)
                !ismissing(left_virt_spaces[i]) &&
                    (left_virt_spaces[i] == _firstspace(ac) ||
                     throw(SpaceMismatch("Left virtual space of AC on site $(i) doesn't match")))
                !ismissing(right_virt_spaces[i + 1]) &&
                    (right_virt_spaces[i + 1] == _lastspace(ac)' ||
                     throw(SpaceMismatch("Right virtual space of AC on site $(i) doesn't match")))

                right_virt_spaces[i + 1] = _lastspace(ac)'
                left_virt_spaces[i] = _firstspace(ac)
            end
        end

        for (i, c) in enumerate(CLs)
            ismissing(c) && continue
            !ismissing(left_virt_spaces[i]) && (left_virt_spaces[i] == _firstspace(c) ||
                                                throw(SpaceMismatch("Left virtual space of CL on site $(i) doesn't match")))
            !ismissing(right_virt_spaces[i]) && (right_virt_spaces[i] == _lastspace(c)' ||
                                                 throw(SpaceMismatch("Right virtual space of CL on site $(i) doesn't match")))
        end

        return new{A,B}(ALs, ARs, ACs, CLs)
    end
end

function Base.getproperty(Ψ::FiniteMPS, prop::Symbol)
    if prop == :AL
        return ALView(Ψ)
    elseif prop == :AR
        return ARView(Ψ)
    elseif prop == :AC
        return ACView(Ψ)
    elseif prop == :CR
        return CRView(Ψ)
    else
        return getfield(Ψ, prop)
    end
end

#===========================================================================================
Constructors
===========================================================================================#

function FiniteMPS(As::Vector{<:GenericMPSTensor}; normalize=false, overwrite=false)
    # TODO: copying the input vector is probably not necessary, as we are constructing new 
    # vectors anyways, maybe deprecate `overwrite`.
    As = overwrite ? As : copy(As)
    N = length(As)
    for i in 1:(N - 1)
        As[i], C = leftorth(As[i]; alg=QRpos())
        normalize && normalize!(C)
        As[i + 1] = _transpose_front(C * _transpose_tail(As[i + 1]))
    end

    As[end], C = leftorth(As[end]; alg=QRpos())
    normalize && normalize!(C)

    A = eltype(As)
    B = typeof(C)

    CLs = Vector{Union{Missing,B}}(missing, N + 1)
    ALs = Vector{Union{Missing,A}}(missing, N)
    ARs = Vector{Union{Missing,A}}(missing, N)
    ACs = Vector{Union{Missing,A}}(missing, N)

    ALs .= As
    CLs[end] = C

    return FiniteMPS(ALs, ARs, ACs, CLs)
end

function FiniteMPS(f, elt, physspaces::Vector{<:Union{S,CompositeSpace{S}}},
                   virtspaces::Vector{S}; normalize=true) where {S<:ElementarySpace}
    N = length(physspaces)
    length(virtspaces) == N + 1 || throw(DimensionMismatch("length mismatch of spaces"))
    tensors = MPSTensor.(f, elt, physspaces, virtspaces[1:end-1], virtspaces[2:end])
    return FiniteMPS(tensors; normalize=normalize, overwrite=true)
end
function FiniteMPS(physspaces::Vector{<:Union{S,CompositeSpace{S}}}, virtspaces::Vector{S};
                   kwargs...) where {S<:ElementarySpace}
    return FiniteMPS(rand, Defaults.eltype, physspaces, virtspaces; kwargs...)
end

function FiniteMPS(f, elt, physspaces::Vector{<:Union{S,CompositeSpace{S}}},
                   maxvirtspace::S; left::S=oneunit(S),
                   right::S=oneunit(S), kwargs...) where {S<:ElementarySpace}
    N = length(physspaces)
    virtspaces = Vector{S}(undef, N + 1)
    virtspaces[1] = left
    for k in 2:N
        virtspaces[k] = infimum(fuse(virtspaces[k - 1], fuse(physspaces[k])), maxvirtspace)
        dim(virtspaces[k]) > 0 || @warn "no fusion channels available"
    end
    virtspaces[N + 1] = right

    for k in N:-1:2
        virtspaces[k] = infimum(virtspaces[k],
                                fuse(virtspaces[k + 1], dual(fuse(physspaces[k]))))
        dim(virtspaces[k]) > 0 || @warn "no fusion channels available"
    end

    return FiniteMPS(f, elt, physspaces, virtspaces; kwargs...)
end
function FiniteMPS(physspaces::Vector{<:Union{S,CompositeSpace{S}}}, maxvirtspace::S;
                   kwargs...) where {S<:ElementarySpace}
    return FiniteMPS(rand, Defaults.eltype, physspaces, maxvirtspace; kwargs...)
end

FiniteMPS(P::ProductSpace, args...; kwargs...) = FiniteMPS(collect(P), args...; kwargs...)
function FiniteMPS(f, elt, P::ProductSpace, args...; kwargs...)
    return FiniteMPS(f, elt, collect(P), args...; kwargs...)
end

function FiniteMPS(N::Int, V::VectorSpace, args...; kwargs...)
    return FiniteMPS(fill(V, N), args...; kwargs...)
end
function FiniteMPS(f, elt, N::Int, V::VectorSpace, args...; kwargs...)
    return FiniteMPS(f, elt, fill(V, N), args...; kwargs...)
end

#===========================================================================================
Utility
===========================================================================================#

Base.size(Ψ::FiniteMPS, args...) = size(Ψ.ALs, args...)
Base.length(Ψ::FiniteMPS) = length(Ψ.ALs)
Base.eltype(Ψtype::Type{<:FiniteMPS}) = site_type(Ψtype) # this might not be true
Base.copy(Ψ::FiniteMPS) = FiniteMPS(copy(Ψ.ALs), copy(Ψ.ARs), copy(Ψ.ACs), copy(Ψ.CLs))
function Base.similar(Ψ::FiniteMPS)
    return FiniteMPS(similar(Ψ.ALs), similar(Ψ.ARs), similar(Ψ.ACs), similar(Ψ.CLs))
end

function Base.convert(TType::Type{<:AbstractTensorMap}, Ψ::FiniteMPS)
    T = foldl(Ψ.AR[2:end]; init=first(Ψ.AC)) do x, y
        return _transpose_front(x * _transpose_tail(y))
    end
    return convert(TType, T)
end

site_type(::Type{<:FiniteMPS{A}}) where {A} = A
bond_type(::Type{<:FiniteMPS{<:Any,B}}) where {B} = B

function left_virtualspace(Ψ::FiniteMPS, n::Integer)
    if n > 0 && !ismissing(Ψ.ALs[n])
        dual(_lastspace(Ψ.ALs[n]))
    elseif n < length(Ψ.ALs) && !ismissing(Ψ.ALs[n + 1])
        _firstspace(Ψ.ALs[n + 1])
    else
        _firstspace(Ψ.CR[n])
    end
end
function right_virtualspace(Ψ::FiniteMPS, n::Integer)
    if n > 0 && !ismissing(Ψ.ARs[n])
        dual(_lastspace(Ψ.ARs[n]))
    elseif n < length(Ψ.ARs) && !ismissing(Ψ.ARs[n + 1])
        _firstspace(Ψ.ARs[n + 1])
    else
        dual(_lastspace(Ψ.CR[n]))
    end
end

function physicalspace(Ψ::FiniteMPS{<:GenericMPSTensor{<:Any,N}}, n::Integer) where {N}
    N == 1 && return ProductSpace{spacetype(Ψ)}()
    A = !ismissing(Ψ.ALs[n]) ? Ψ.ALs[n] : 
        !ismissing(Ψ.ARs[n]) ? Ψ.ARs[n] : Ψ.AC[n] # should never reach last case?
        
    if N == 2
        return space(A, 2)
    else
        return ProductSpace{spacetype(Ψ),N - 1}(space.(Ref(A),
                                                Base.front(Base.tail(TensorKit.allind(A)))))
    end
end

TensorKit.space(Ψ::FiniteMPS{<:MPSTensor}, n::Integer) = space(Ψ.AC[n], 2)
function TensorKit.space(Ψ::FiniteMPS{<:GenericMPSTensor}, n::Integer)
    t = Ψ.AC[n]
    S = spacetype(t)
    return ProductSpace{S}(space.(Ref(t), Base.front(Base.tail(TensorKit.allind(t)))))
end

"""
    max_Ds(Ψ::FiniteMPS) -> Vector{Float64}

Compute the dimension of the maximal virtual space at a given site.
"""
function max_Ds(Ψ::FiniteMPS)
    N = length(Ψ)
    physicaldims = dim.(space.(Ref(Ψ), 1:N))
    D_left = cumprod(vcat(dim(left_virtualspace(Ψ, 1)), physicaldims))
    D_right = cumprod(vcat(dim(right_virtualspace(Ψ, N)), reverse(physicaldims)))
    return min.(D_left, D_right)
end

#===========================================================================================
Linear Algebra
===========================================================================================#

#=
No support yet for converting the scalar type, also no in-place operations
=#
Base.:*(Ψ::FiniteMPS, a::Number) = rmul!(copy(Ψ), a)
Base.:*(a::Number, Ψ::FiniteMPS) = lmul!(a, copy(Ψ))

function Base.:+(Ψ₁::FiniteMPS{A}, Ψ₂::FiniteMPS{A}) where {A}
    length(Ψ₁) == length(Ψ₂) || throw(DimensionMismatch())
    N = length(Ψ₁)
    for k in 1:N
        space(Ψ₁, k) == space(Ψ₂, k) ||
            throw(SpaceMismatch("Non-matching physical space on site $k."))
    end
    left_virtualspace(Ψ₁, 0) == left_virtualspace(Ψ₂, 0) ||
        throw(SpaceMismatch("Non-matching left virtual space."))
    right_virtualspace(Ψ₁, N) == right_virtualspace(Ψ₂, N) ||
        throw(SpaceMismatch("Non-matching right virtual space."))

    tensors = A[]

    k = 1 # firstindex(Ψ₁)
    t1 = Ψ₁.AL[k]
    t2 = Ψ₂.AL[k]
    V1 = domain(t1)[1]
    V2 = domain(t2)[1]
    w1 = isometry(storagetype(A), V1 ⊕ V2, V1)
    w2 = leftnull(w1)
    @assert domain(w2) == ⊗(V2)

    push!(tensors, t1 * w1' + t2 * w2')
    for k in 2:(N - 1)
        t1 = _transpose_front(w1 * _transpose_tail(Ψ₁.AL[k]))
        t2 = _transpose_front(w2 * _transpose_tail(Ψ₂.AL[k]))
        V1 = domain(t1)[1]
        V2 = domain(t2)[1]
        w1 = isometry(storagetype(A), V1 ⊕ V2, V1)
        w2 = leftnull(w1)
        @assert domain(w2) == ⊗(V2)
        push!(tensors, t1 * w1' + t2 * w2')
    end
    k = N
    t1 = _transpose_front(w1 * _transpose_tail(Ψ₁.AC[k]))
    t2 = _transpose_front(w2 * _transpose_tail(Ψ₂.AC[k]))
    push!(tensors, t1 + t2)
    return FiniteMPS(tensors)
end
Base.:-(Ψ₁::FiniteMPS, Ψ₂::FiniteMPS) = Ψ₁ + (-1 * Ψ₂)

function TensorKit.lmul!(a::Number, Ψ::FiniteMPS)
    Ψ.ACs .*= a
    Ψ.CLs .*= a
    return Ψ
end

function TensorKit.rmul!(Ψ::FiniteMPS, a::Number)
    Ψ.ACs .*= a
    Ψ.CLs .*= a
    return Ψ
end

function TensorKit.dot(Ψ₁::FiniteMPS, Ψ₂::FiniteMPS)
    #todo : rewrite this without having to gauge
    length(Ψ₁) == length(Ψ₂) || throw(ArgumentError("MPS with different length"))
    ρr = TransferMatrix(Ψ₂.AR[2:end], Ψ₁.AR[2:end]) * r_RR(Ψ₂)
    return tr(_transpose_front(Ψ₁.AC[1])' * _transpose_front(Ψ₂.AC[1]) * ρr)
end

#todo : rewrite this without having to gauge
TensorKit.norm(Ψ::FiniteMPS) = norm(Ψ.AC[1])
TensorKit.normalize!(Ψ::FiniteMPS) = rmul!(Ψ, 1 / norm(Ψ))
TensorKit.normalize(Ψ::FiniteMPS) = normalize!(copy(Ψ))

#===========================================================================================
Fixedpoints
===========================================================================================#

function r_RR(Ψ::FiniteMPS{T}) where {T}
    return isomorphism(storagetype(T), domain(Ψ.AR[end]), domain(Ψ.AR[end]))
end
function l_LL(Ψ::FiniteMPS{T}) where {T}
    return isomorphism(storagetype(T), space(Ψ.AL[1], 1), space(Ψ.AL[1], 1))
end
