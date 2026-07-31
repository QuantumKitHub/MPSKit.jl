"""
    FiniteMPS{A<:GenericMPSTensor,B<:MPSBondTensor} <: AbstractFiniteMPS

Type that represents a finite Matrix Product State.

## Properties
- `AL` -- left-gauged MPS tensors
- `AR` -- right-gauged MPS tensors
- `AC` -- center-gauged MPS tensors
- `C` -- gauge tensors
- `center` -- location of the gauge center

The center property returns `center::HalfInt` that indicates the location of the MPS center:
- `isinteger(center)` → `center` is a whole number and indicates the location of the first `AC` tensor present in the underlying `ψ.ACs` field.
- `ishalfodd(center)` → `center` is a half-odd-integer, meaning that there are no `AC` tensors, and indicating between which sites the bond tensor lives.

e.g `mps.center = 7/2` means that the bond tensor is to the right of the 3rd site and can be accessed via `mps.C[3]`.

## Notes
By convention, we have that:
- `AL[i] * C[i]` = `AC[i]` = `C[i-1] * AR[i]`
- `AL[i]' * AL[i] = 1`
- `AR[i] * AR[i]' = 1`

---

## Constructors
    FiniteMPS([f, eltype], physicalspaces::Vector{<:Union{S,CompositeSpace{S}}},
              maxvirtualspaces::Union{S,Vector{S}};
              normalize=true, left=unitspace(S), right=unitspace(S)) where {S<:ElementarySpace}
    FiniteMPS([f, eltype], N::Int, physicalspace::Union{S,CompositeSpace{S}},
              maxvirtualspaces::Union{S,Vector{S}};
              normalize=true, left=unitspace(S), right=unitspace(S)) where {S<:ElementarySpace}
    FiniteMPS(As::Vector{<:GenericMPSTensor}; normalize=false, overwrite=false)

Construct an MPS via a specification of physical and virtual spaces, or from a list of
tensors `As`. All cases reduce to the latter. In particular, a state with a non-trivial
total charge can be constructed by passing a non-trivially charged vector space as the
`left` or `right` virtual spaces.

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
- `normalize=true`: normalize the constructed state
- `overwrite=false`: overwrite the given input tensors
- `left=unitspace(S)`: left-most virtual space
- `right=unitspace(S)`: right-most virtual space
"""
struct FiniteMPS{A <: GenericMPSTensor, B <: MPSBondTensor} <: AbstractFiniteMPS
    ALs::Vector{Union{Missing, A}}
    ARs::Vector{Union{Missing, A}}
    ACs::Vector{Union{Missing, A}}
    Cs::Vector{Union{Missing, B}}
    function FiniteMPS{A, B}(
            ALs::Vector{Union{Missing, A}}, ARs::Vector{Union{Missing, A}},
            ACs::Vector{Union{Missing, A}}, Cs::Vector{Union{Missing, B}}
        ) where {A <: GenericMPSTensor, B <: MPSBondTensor}
        return new{A, B}(ALs, ARs, ACs, Cs)
    end
    function FiniteMPS(
            ALs::Vector{MA}, ARs::Vector{MA},
            ACs::Vector{MA},
            Cs::Vector{MB}
        ) where {MA <: Union{GenericMPSTensor, Missing}, MB <: Union{MPSBondTensor, Missing}}
        A = _not_missing_type(MA)
        B = _not_missing_type(MB)
        length(ACs) == length(Cs) - 1 == length(ALs) == length(ARs) ||
            throw(DimensionMismatch("length mismatch of tensors"))
        sum(ismissing.(ACs)) + sum(ismissing.(Cs)) < length(ACs) + length(Cs) ||
            throw(ArgumentError("at least one AC/C should not be missing"))

        S = spacetype(A)
        left_virt_spaces = Vector{Union{Missing, S}}(missing, length(Cs))
        right_virt_spaces = Vector{Union{Missing, S}}(missing, length(Cs))

        for (i, tup) in enumerate(zip(ALs, ARs, ACs))
            non_missing = filter(!ismissing, tup)
            isempty(non_missing) && throw(ArgumentError("missing site tensor"))
            (al, ar, ac) = tup

            if !ismissing(al)
                !ismissing(left_virt_spaces[i]) &&
                    (
                    left_virt_spaces[i] == _firstspace(al) ||
                        throw(SpaceMismatch("Virtual space of AL on site $(i) doesn't match"))
                )

                left_virt_spaces[i + 1] = _lastspace(al)'
                left_virt_spaces[i] = _firstspace(al)
            end

            if !ismissing(ar)
                !ismissing(right_virt_spaces[i]) &&
                    (
                    right_virt_spaces[i] == _firstspace(ar) ||
                        throw(SpaceMismatch("Virtual space of AR on site $(i) doesn't match"))
                )

                right_virt_spaces[i + 1] = _lastspace(ar)'
                right_virt_spaces[i] = _firstspace(ar)
            end

            if !ismissing(ac)
                !ismissing(left_virt_spaces[i]) &&
                    (
                    left_virt_spaces[i] == _firstspace(ac) ||
                        throw(SpaceMismatch("Left virtual space of AC on site $(i) doesn't match"))
                )
                !ismissing(right_virt_spaces[i + 1]) &&
                    (
                    right_virt_spaces[i + 1] == _lastspace(ac)' ||
                        throw(SpaceMismatch("Right virtual space of AC on site $(i) doesn't match"))
                )

                right_virt_spaces[i + 1] = _lastspace(ac)'
                left_virt_spaces[i] = _firstspace(ac)
            end
        end

        for (i, c) in enumerate(Cs)
            ismissing(c) && continue
            !ismissing(left_virt_spaces[i]) && (
                left_virt_spaces[i] == _firstspace(c) ||
                    throw(SpaceMismatch("Left virtual space of C on site $(i - 1) doesn't match"))
            )
            !ismissing(right_virt_spaces[i]) && (
                right_virt_spaces[i] == _lastspace(c)' ||
                    throw(SpaceMismatch("Right virtual space of C on site $(i - 1) doesn't match"))
            )
        end
        return new{A, B}(ALs, ARs, ACs, Cs)
    end
end

_not_missing_type(::Type{Missing}) = throw(ArgumentError("Only missing type present"))
function _not_missing_type(::Type{T}) where {T}
    if T isa Union
        return (!(T.a === Missing) && !(T.b === Missing)) ? T :
            !(T.a === Missing) ? _not_missing_type(T.a) : _not_missing_type(T.b)
    else
        return T
    end
end

function Base.getproperty(ψ::FiniteMPS, prop::Symbol)
    if prop == :AL
        return ALView(ψ)
    elseif prop == :AR
        return ARView(ψ)
    elseif prop == :AC
        return ACView(ψ)
    elseif prop == :C
        return CView(ψ)
    elseif prop == :center
        return _gaugecenter(ψ)
    else
        return getfield(ψ, prop)
    end
end

function Base.propertynames(::FiniteMPS)
    return (:AL, :AR, :AC, :C, :center)
end

"""
    _gaugecenter(ψ::FiniteMPS)::HalfInt

Return the location of the MPS center.

`center::HalfInt`:
- `isinteger(center)` → `center` is a whole number and indicates the location of the first `AC` tensor present in `ψ.ACs`
- `ishalfodd(center)` → `center` is a half-odd-integer, meaning that there are no `AC` tensors, and indicating between which sites the bond tensor lives.

## Example
```julia
ψ = FiniteMPS(3, ℂ^2, ℂ^16)
ψ.center # returns 7/2, bond tensor is to the right of the 3rd site
ψ.AC[1]   # moves center to first site
ψ.center # returns 1
```
"""
function _gaugecenter(ψ::FiniteMPS)::HalfInt
    L = length(ψ)

    center = findfirst(!ismissing, ψ.ACs) # give priority to integer values of center
    if isnothing(center)
        center = findfirst(!ismissing, ψ.Cs)
        isnothing(center) && throw(ArgumentError("No center found, invalid state"))
        return (center - 1 / 2)
    end
    isnothing(center) && throw(ArgumentError("No center found, invalid state"))
    return center
end
#===========================================================================================
Constructors
===========================================================================================#

function FiniteMPS(As::Vector{<:GenericMPSTensor}; normalize = false, overwrite = false)
    # TODO: copying the input vector is probably not necessary, as we are constructing new
    # vectors anyways, maybe deprecate `overwrite`.
    As = overwrite ? As : copy(As)
    N = length(As)
    As[1] = MatrixAlgebraKit.copy_input(qr_compact, As[1])
    local C
    for i in eachindex(As)
        As[i], C = qr_compact!(As[i]; positive = true)
        normalize && normalize!(C)
        i == N || (As[i + 1] = _transpose_front(C * _transpose_tail(As[i + 1])))
    end

    A = eltype(As)
    B = typeof(C)

    Cs = Vector{Union{Missing, B}}(missing, N + 1)
    ALs = Vector{Union{Missing, A}}(missing, N)
    ARs = Vector{Union{Missing, A}}(missing, N)
    ACs = Vector{Union{Missing, A}}(missing, N)

    ALs .= As
    Cs[end] = C

    return FiniteMPS(ALs, ARs, ACs, Cs)
end

function FiniteMPS(
        f, elt, Pspaces::Vector{<:Union{S, CompositeSpace{S}}}, maxVspaces::Vector{S};
        normalize = true, left::S = unitspace(S), right::S = unitspace(S)
    ) where {S <: ElementarySpace}
    N = length(Pspaces)
    length(maxVspaces) == N - 1 ||
        throw(DimensionMismatch("length of physical spaces ($N) and virtual spaces $(length(maxVspaces)) should differ by 1"))

    # limit the maximum virtual dimension such that result is full rank
    fusedPspaces = fuse.(Pspaces) # for working with multiple physical spaces
    Vspaces = similar(maxVspaces, N + 1)

    Vspaces[1] = left
    for k in 2:N
        Vspaces[k] = infimum(fuse(Vspaces[k - 1], fusedPspaces[k - 1]), maxVspaces[k - 1])
        dim(Vspaces[k]) > 0 || @warn "no fusion channels available at site $k"
    end

    Vspaces[end] = right
    for k in reverse(2:N)
        Vspaces[k] = infimum(Vspaces[k], fuse(Vspaces[k + 1], dual(fusedPspaces[k])))
        dim(Vspaces[k]) > 0 || @warn "no fusion channels available at site $k"
    end

    # construct MPS
    tensors = @. f(elt, Vspaces[1:(end - 1)] ⊗ Pspaces ← Vspaces[2:end])
    return FiniteMPS(tensors; normalize, overwrite = true)
end
function FiniteMPS(
        f, elt, Pspaces::Vector{<:Union{S, CompositeSpace{S}}}, maxVspace::S;
        kwargs...
    ) where {S <: ElementarySpace}
    maxVspaces = fill(maxVspace, length(Pspaces) - 1)
    return FiniteMPS(f, elt, Pspaces, maxVspaces; kwargs...)
end
function FiniteMPS(
        Pspaces::Vector{<:Union{S, CompositeSpace{S}}}, maxVspaces::Union{S, Vector{S}};
        kwargs...
    ) where {S <: ElementarySpace}
    return FiniteMPS(rand, Defaults.eltype, Pspaces, maxVspaces; kwargs...)
end


function FiniteMPS(
        elt::Type, Pspaces::Vector{<:Union{S, CompositeSpace{S}}}, maxVspaces::Union{S, Vector{S}};
        kwargs...
    ) where {S <: ElementarySpace}
    return FiniteMPS(rand, elt, Pspaces, maxVspaces; kwargs...)
end


# Also accept single physical space and length
function FiniteMPS(N::Int, V::VectorSpace, args...; kwargs...)
    return FiniteMPS(fill(V, N), args...; kwargs...)
end
function FiniteMPS(f, elt, N::Int, V::VectorSpace, args...; kwargs...)
    return FiniteMPS(f, elt, fill(V, N), args...; kwargs...)
end

# Also accept ProductSpace of physical spaces
FiniteMPS(P::ProductSpace, args...; kwargs...) = FiniteMPS(collect(P), args...; kwargs...)
function FiniteMPS(f, elt, P::ProductSpace, args...; kwargs...)
    return FiniteMPS(f, elt, collect(P), args...; kwargs...)
end

# construct from dense state
# TODO: make planar?
function FiniteMPS(ψ::AbstractTensor)
    A = _transpose_front(
        insertrightunit(transpose(insertrightunit(ψ, numind(ψ); dual = true)), numind(ψ) + 1; dual = true)
    )
    return FiniteMPS(decompose_localmps(A); normalize = false, overwrite = true)
end

#===========================================================================================
Utility
===========================================================================================#

Base.size(ψ::FiniteMPS, args...) = size(ψ.ALs, args...)
Base.length(ψ::FiniteMPS) = length(ψ.ALs)
Base.eltype(ψtype::Type{<:FiniteMPS}) = site_type(ψtype) # this might not be true
function Base.similar(ψ::FiniteMPS{A, B}) where {A, B}
    return FiniteMPS{A, B}(similar(ψ.ALs), similar(ψ.ARs), similar(ψ.ACs), similar(ψ.Cs))
end
# an empty state with promoted scalar type: no tensor is materialised yet
function Base.similar(ψ::FiniteMPS, ::Type{S}) where {S <: Number}
    A = similar_scalartype(site_type(ψ), S)
    B = similar_scalartype(bond_type(ψ), S)
    N = length(ψ)
    return FiniteMPS{A, B}(
        Vector{Union{Missing, A}}(missing, N), Vector{Union{Missing, A}}(missing, N),
        Vector{Union{Missing, A}}(missing, N), Vector{Union{Missing, B}}(missing, N + 1)
    )
end

Base.isfinite(::Type{<:FiniteMPS}) = true
GeometryStyle(::Type{<:FiniteMPS}) = FiniteChainStyle()

Base.eachindex(ψ::FiniteMPS) = eachindex(ψ.AL)
Base.eachindex(l::IndexStyle, ψ::FiniteMPS) = eachindex(l, ψ.AL)
Base.checkbounds(::Type{Bool}, ψ::FiniteMPS, i::Integer) = 1 <= i <= length(ψ)

Base.@propagate_inbounds function Base.getindex(ψ::FiniteMPS, i::Int)
    c = ψ.center

    @boundscheck checkbounds(ψ, i)

    if ishalfodd(c)
        c -= 1 / 2
    end

    return if i > Int(c)
        ψ.AR[i]
    elseif i == Int(c)
        ψ.AC[i]
    else
        ψ.AL[i]
    end
end

function AC2(psi::FiniteMPS, site::Int; kind = :ACAR)
    if kind == :ACAR
        return psi.AC[site] * _transpose_tail(psi.AR[site + 1])
    elseif kind == :ALAC
        return psi.AL[site] * _transpose_tail(psi.AC[site + 1])
    else
        throw(ArgumentError("Invalid kind: $kind"))
    end
end

f_if_not_missing(f, x) = ismissing(x) ? x : f(x)
_copy_if_not_missing(x) = f_if_not_missing(copy, x)
_complex_if_not_missing(x) = f_if_not_missing(complex, x)

function Base.copy(mps::FiniteMPS)
    mps2 = similar(mps)
    mps2.ALs .= _copy_if_not_missing.(mps.ALs)
    mps2.ARs .= _copy_if_not_missing.(mps.ARs)
    mps2.ACs .= _copy_if_not_missing.(mps.ACs)
    mps2.Cs .= _copy_if_not_missing.(mps.Cs)
    return mps2
end

function Base.complex(mps::FiniteMPS)
    scalartype(mps) <: Complex && return mps
    ALs = _complex_if_not_missing.(mps.ALs)
    ARs = _complex_if_not_missing.(mps.ARs)
    Cs = _complex_if_not_missing.(mps.Cs)
    ACs = _complex_if_not_missing.(mps.ACs)
    TA = Base.promote_op(complex, site_type(mps))
    TB = Base.promote_op(complex, bond_type(mps))
    return FiniteMPS(
        collect(Union{Missing, TA}, ALs),
        collect(Union{Missing, TA}, ARs),
        collect(Union{Missing, TA}, ACs),
        collect(Union{Missing, TB}, Cs)
    )
end

@inline function Base.getindex(ψ::FiniteMPS, I::AbstractUnitRange)
    return Base.getindex.(Ref(ψ), I)
end

function Base.convert(::Type{TensorMap}, ψ::FiniteMPS)
    T = foldl(ψ.AR[2:end]; init = first(ψ.AC)) do x, y
        return _transpose_front(x * _transpose_tail(y))
    end

    # remove utility legs
    isunitspace(space(T, 1)) || throw(ArgumentError("utility leg not trivial"))
    isunitspace(space(T, numind(T))') || throw(ArgumentError("utility leg not trivial"))
    UTU = transpose(
        removeunit(_transpose_tail(removeunit(T, numind(T))), 1), (reverse(ntuple(identity, numind(T) - 2)), ())
    )

    return UTU
end

site_type(::Type{<:FiniteMPS{A}}) where {A} = A
bond_type(::Type{<:FiniteMPS{<:Any, B}}) where {B} = B

function left_virtualspace(ψ::FiniteMPS, n::Integer)
    checkbounds(ψ, n)
    return !ismissing(ψ.ALs[n]) ? left_virtualspace(ψ.ALs[n]) :
        !ismissing(ψ.ARs[n]) ? left_virtualspace(ψ.ARs[n]) :
        dual(_lastspace(ψ.C[n - 1]))
end
function right_virtualspace(ψ::FiniteMPS, n::Integer)
    checkbounds(ψ, n)
    return !ismissing(ψ.ARs[n]) ? right_virtualspace(ψ.ARs[n]) :
        !ismissing(ψ.ALs[n]) ? right_virtualspace(ψ.ALs[n]) :
        _firstspace(ψ.C[n])
end

function physicalspace(ψ::FiniteMPS{<:GenericMPSTensor{<:Any, N}}, n::Integer) where {N}
    N == 1 && return ProductSpace{spacetype(ψ)}()
    return physicalspace(coalesce(ψ.ALs[n], ψ.ARs[n], ψ.ACs[n]))
end

TensorKit.space(ψ::FiniteMPS{<:MPSTensor}, n::Integer) = space(ψ.AC[n], 2)
function TensorKit.space(ψ::FiniteMPS{<:GenericMPSTensor}, n::Integer)
    t = ψ.AC[n]
    S = spacetype(t)
    return ProductSpace{S}(space.(Ref(t), Base.front(Base.tail(TensorKit.allind(t)))))
end

"""
    max_virtualspaces(ψ::FiniteMPS)
    max_virtualspaces(Ps::Vector{<:Union{S,CompositeSpace{S}}}; left=unitspace(S), right=unitspace(S))

Compute the maximal virtual spaces of a given finite MPS or its physical spaces.
"""
function max_virtualspaces(
        Ps::Vector{<:Union{S, CompositeSpace{S}}}; left = unitspace(S), right = unitspace(S)
    ) where {S <: ElementarySpace}
    Vs = similar(Ps, length(Ps) + 1)
    Vs[1] = left
    Vs[end] = right
    for k in 2:length(Ps)
        Vs[k] = fuse(Vs[k - 1], fuse(Ps[k - 1]))
    end
    for k in reverse(2:length(Ps))
        Vs[k] = infimum(Vs[k], fuse(Vs[k + 1], dual(fuse(Ps[k]))))
    end
    return Vs
end
function max_virtualspaces(ψ::FiniteMPS)
    return max_virtualspaces(
        physicalspace(ψ);
        left = left_virtualspace(ψ, 1), right = right_virtualspace(ψ, length(ψ))
    )
end

"""
    max_Ds(ψ::FiniteMPS) -> Vector{Float64}

Compute the dimension of the maximal virtual space at a given site.
"""
max_Ds(ψ::FiniteMPS) = dim.(max_virtualspaces(ψ))


#===========================================================================================
Linear Algebra
===========================================================================================#

#=
Scaling is in-place on a copy, so it cannot convert the scalar type: use `complex(ψ)` first if
`a` is complex and `ψ` is not. Addition does promote, through `similar(ψ, ::Type{S})`.
=#
Base.:*(ψ::FiniteMPS, a::Number) = rmul!(copy(ψ), a)
Base.:*(a::Number, ψ::FiniteMPS) = lmul!(a, copy(ψ))

function Base.:+(ψ₁::FiniteMPS, ψ₂::FiniteMPS)
    N = length(ψ₁)
    N == length(ψ₂) ||
        throw(DimensionMismatch("Cannot add states of length $N and $(length(ψ₂))"))
    left_virtualspace(ψ₁, 1) == left_virtualspace(ψ₂, 1) &&
        right_virtualspace(ψ₁, N) == right_virtualspace(ψ₂, N) ||
        throw(SpaceMismatch("Cannot add states with different boundary virtual spaces"))

    # A single site has no internal bond to fuse -- the boundary virtual spaces are fixed by the
    # check above -- so the sum is simply the sum of the two center tensors. The generic branch
    # below cannot express this: it splits the chain into a left and a right block and fuses them
    # at the seam, and for `N == 1` there is no seam.
    N == 1 && return FiniteMPS([ψ₁.AC[1] + ψ₂.AC[1]])

    halfN = div(N, 2)

    # Take a snapshot of the tensors that make up the two states, in a single canonical form:
    # `AL[1] ⋯ AL[halfN] C[halfN] AR[halfN + 1] ⋯ AR[N]`. Gauging is lazy, so every read may
    # move the gauge center; only tensors that belong to the same gauge may be combined.
    ψ₁.C[halfN], ψ₂.C[halfN] # settle both gauge centers at the seam first
    AL₁, AL₂ = ψ₁.AL[1:halfN], ψ₂.AL[1:halfN]
    AR₁, AR₂ = ψ₁.AR[(halfN + 1):N], ψ₂.AR[(halfN + 1):N] # indexed as `AR[i - halfN]`
    Cmid₁, Cmid₂ = ψ₁.C[halfN], ψ₂.C[halfN]

    ψ = similar(ψ₁, promote_type(scalartype(ψ₁), scalartype(ψ₂)))

    # left half
    F₁ = isometry(
        storagetype(ψ), (_lastspace(AL₁[1]) ⊕ _lastspace(AL₂[1]))', _lastspace(AL₁[1])'
    )
    F₂ = left_null(F₁)
    @assert _lastspace(F₂) == _lastspace(AL₂[1])

    AL = AL₁[1] * F₁' + AL₂[1] * F₂'
    ψ.ALs[1], R = left_orth!(AL)

    for i in 2:halfN
        A₁ = _transpose_front(F₁ * _transpose_tail(AL₁[i]))
        A₂ = _transpose_front(F₂ * _transpose_tail(AL₂[i]))

        F₁ = isometry(
            storagetype(ψ), (_lastspace(A₁) ⊕ _lastspace(AL₂[i]))', _lastspace(A₁)'
        )
        F₂ = left_null(F₁)
        @assert _lastspace(F₂) == _lastspace(AL₂[i])

        AL = _transpose_front(R * _transpose_tail(A₁ * F₁' + A₂ * F₂'))
        ψ.ALs[i], R = left_orth!(AL)
    end

    C₁ = F₁ * Cmid₁
    C₂ = F₂ * Cmid₂

    # right half
    F₁ = isometry(
        storagetype(ψ), _firstspace(AR₁[end]) ⊕ _firstspace(AR₂[end]), _firstspace(AR₁[end])
    )
    F₂ = left_null(F₁)
    @assert _lastspace(F₂) == _firstspace(AR₂[end])'

    AR = F₁ * _transpose_tail(AR₁[end]) + F₂ * _transpose_tail(AR₂[end])
    L, AR′ = right_orth!(AR)
    ψ.ARs[end] = _transpose_front(AR′)

    for i in Iterators.reverse((halfN + 1):(N - 1))
        A₁ = _transpose_tail(AR₁[i - halfN] * F₁')
        A₂ = _transpose_tail(AR₂[i - halfN] * F₂')

        F₁ = isometry(
            storagetype(ψ), _firstspace(A₁) ⊕ _firstspace(A₂), _firstspace(A₁)
        )
        F₂ = left_null(F₁)
        @assert _lastspace(F₂) == _firstspace(A₂)'

        AR = _transpose_tail(_transpose_front(F₁ * A₁ + F₂ * A₂) * L)
        L, AR′ = right_orth!(AR)
        ψ.ARs[i] = _transpose_front(AR′)
    end

    # center
    C₁ = C₁ * F₁'
    C₂ = C₂ * F₂'
    ψ.Cs[halfN + 1] = R * (C₁ + C₂) * L

    return ψ
end

Base.:-(ψ₁::FiniteMPS, ψ₂::FiniteMPS) = ψ₁ + (-1 * ψ₂)

function TensorKit.lmul!(a::Number, ψ::FiniteMPS)
    ψ.ACs .*= a
    ψ.Cs .*= a
    return ψ
end

function TensorKit.rmul!(ψ::FiniteMPS, a::Number)
    ψ.ACs .*= a
    ψ.Cs .*= a
    return ψ
end

function TensorKit.dot(ψ₁::FiniteMPS, ψ₂::FiniteMPS)
    #todo : rewrite this without having to gauge
    length(ψ₁) == length(ψ₂) || throw(ArgumentError("MPS with different length"))
    if ψ₁ === ψ₂
        return convert(Base.promote_op(inner, scalartype(ψ₁), scalartype(ψ₂)), norm(ψ₁)^2)
    end
    ρr = TransferMatrix(ψ₂.AR[2:end], ψ₁.AR[2:end]) * r_RR(ψ₂)
    return tr(_transpose_front(ψ₁.AC[1])' * _transpose_front(ψ₂.AC[1]) * ρr)
end

function TensorKit.norm(ψ::FiniteMPS)
    c = ψ.center
    if isinteger(c) # center is an AC
        return norm(ψ.AC[Int(c)])
    else # center is a bond-tensor
        return norm(ψ.C[Int(c - 1 / 2)])
    end
end
TensorKit.normalize!(ψ::FiniteMPS) = rmul!(ψ, 1 / norm(ψ))
TensorKit.normalize(ψ::FiniteMPS) = normalize!(copy(ψ))

#===========================================================================================
Fixedpoints
===========================================================================================#

function r_RR(ψ::FiniteMPS, site::Int = length(ψ))
    Vr = right_virtualspace(ψ.AR[site])
    return isomorphism(storagetype(site_type(ψ)), Vr ← Vr)
end
function l_LL(ψ::FiniteMPS, site::Int = 1)
    Vl = left_virtualspace(ψ.AL[site])
    return isomorphism(storagetype(site_type(ψ)), Vl ← Vl)
end
