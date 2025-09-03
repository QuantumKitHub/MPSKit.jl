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
              normalize=true, left=oneunit(S), right=oneunit(S)) where {S<:ElementarySpace}
    FiniteMPS([f, eltype], N::Int, physicalspace::Union{S,CompositeSpace{S}},
              maxvirtualspaces::Union{S,Vector{S}};
              normalize=true, left=oneunit(S), right=oneunit(S)) where {S<:ElementarySpace}
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
- `left=oneunit(S)`: left-most virtual space
- `right=oneunit(S)`: right-most virtual space
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
    for i in 1:(N - 1)
        As[i], C = leftorth(As[i]; alg = QRpos())
        normalize && normalize!(C)
        As[i + 1] = _transpose_front(C * _transpose_tail(As[i + 1]))
    end

    As[end], C = leftorth(As[end]; alg = QRpos())
    normalize && normalize!(C)

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
        normalize = true, left::S = oneunit(S), right::S = oneunit(S)
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
    tensors = MPSTensor.(f, elt, Pspaces, Vspaces[1:(end - 1)], Vspaces[2:end])
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
    U = ones(scalartype(ψ), oneunit(spacetype(ψ)))
    A = _transpose_front(
        U * transpose(ψ * U', ((), reverse(ntuple(identity, numind(ψ) + 1))))
    )
    return FiniteMPS(decompose_localmps(A); normalize = false, overwrite = true)
end

#===========================================================================================
Utility
===========================================================================================#

Base.size(ψ::FiniteMPS, args...) = size(ψ.ALs, args...)
Base.length(ψ::FiniteMPS) = length(ψ.ALs)
Base.eltype(ψtype::Type{<:FiniteMPS}) = site_type(ψtype) # this might not be true
Base.copy(ψ::FiniteMPS) = FiniteMPS(copy(ψ.ALs), copy(ψ.ARs), copy(ψ.ACs), copy(ψ.Cs))
function Base.similar(ψ::FiniteMPS{A, B}) where {A, B}
    return FiniteMPS{A, B}(similar(ψ.ALs), similar(ψ.ARs), similar(ψ.ACs), similar(ψ.Cs))
end

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

# TODO: check where gauge center is to determine efficient kind
AC2(psi::FiniteMPS, site::Int) = psi.AC[site] * _transpose_tail(psi.AR[site + 1])

_complex_if_not_missing(x) = ismissing(x) ? x : complex(x)
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
    space(T, 1) == oneunit(spacetype(T)) || throw(ArgumentError("utility leg not trivial"))
    space(T, numind(T)) == oneunit(spacetype(T))' ||
        throw(ArgumentError("utility leg not trivial"))
    U = ones(scalartype(ψ), oneunit(spacetype(ψ)))
    UTU = transpose(
        U' * _transpose_tail(T * U), (reverse(ntuple(identity, numind(T) - 2)), ())
    )

    return UTU
end

site_type(::Type{<:FiniteMPS{A}}) where {A} = A
bond_type(::Type{<:FiniteMPS{<:Any, B}}) where {B} = B
function TensorKit.storagetype(::Union{MPS, Type{MPS}}) where {A, MPS <: FiniteMPS{A}}
    return storagetype(A)
end

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
    max_virtualspaces(Ps::Vector{<:Union{S,CompositeSpace{S}}}; left=oneunit(S), right=oneunit(S))

Compute the maximal virtual spaces of a given finite MPS or its physical spaces.
"""
function max_virtualspaces(
        Ps::Vector{<:Union{S, CompositeSpace{S}}}; left = oneunit(S), right = oneunit(S)
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

function Base.summary(io::IO, ψ::FiniteMPS)
    return print(io, "$(length(ψ))-site FiniteMPS ($(scalartype(ψ)), $(spacetype(ψ)))")
end
function Base.show(io::IO, ::MIME"text/plain", ψ::FiniteMPS)
    println(io, summary(ψ), ":")
    context = IOContext(io, :typeinfo => eltype(ψ), :compact => true)
    return show(context, ψ)
end
Base.show(io::IO, ψ::FiniteMPS) = show(convert(IOContext, io), ψ)
function Base.show(io::IOContext, ψ::FiniteMPS)
    charset = (; start = "┌", mid = "├", stop = "└", ver = "│", dash = "──")
    limit = get(io, :limit, false)::Bool
    half_screen_rows = limit ? div(displaysize(io)[1] - 8, 2) : typemax(Int)
    if !haskey(io, :compact)
        io = IOContext(io, :compact => true)
    end

    L = length(ψ)
    c = ψ.center

    for site in HalfInt.(reverse((1 / 2):(1 / 2):(L + 1 / 2)))
        if site < half_screen_rows || site > L - half_screen_rows
            if site > c # ARs
                if isinteger(site)
                    println(
                        io, Int(site) == L ? charset.start : charset.mid, charset.dash,
                        " AR[$(Int(site))]: ", ψ.ARs[Int(site)]
                    )
                end
            elseif site == c # AC or C
                if isinteger(c) # center is an AC
                    println(
                        io, if site == L
                            charset.start
                        elseif site == 1
                            charset.stop
                        else
                            charset.mid
                        end, charset.dash, " AC[$(Int(site))]: ", ψ.ACs[Int(site)]
                    )
                else # center is a bond-tensor
                    println(
                        io, if site == HalfInt(L + 1 / 2)
                            charset.start
                        elseif site == HalfInt(1 / 2)
                            charset.stop
                        else
                            charset.ver
                        end, " C[$(Int(site - 1 / 2))]: ", ψ.Cs[Int(site + 1 / 2)]
                    )
                end
            else
                if isinteger(site)
                    println(
                        io, site == 1 ? charset.stop : charset.mid, charset.dash,
                        " AL[$(Int(site))]: ", ψ.ALs[Int(site)]
                    )
                end
            end
        elseif site == half_screen_rows
            println(io, charset.ver, "⋮")
        end
    end
    return nothing
end

#===========================================================================================
Linear Algebra
===========================================================================================#

#=
No support yet for converting the scalar type, also no in-place operations
=#
Base.:*(ψ::FiniteMPS, a::Number) = rmul!(copy(ψ), a)
Base.:*(a::Number, ψ::FiniteMPS) = lmul!(a, copy(ψ))

function Base.:+(ψ₁::MPS, ψ₂::MPS) where {MPS <: FiniteMPS}
    length(ψ₁) == length(ψ₂) ||
        throw(DimensionMismatch("Cannot add states of length $(length(ψ₁)) and $(length(ψ₂))"))
    @assert length(ψ₁) > 1 "not implemented for length < 2"

    ψ = similar(ψ₁)
    fill!(ψ.ALs, missing)
    fill!(ψ.ARs, missing)
    fill!(ψ.ACs, missing)
    fill!(ψ.Cs, missing)

    halfN = div(length(ψ), 2)

    # left half
    F₁ = isometry(
        storagetype(ψ), (_lastspace(ψ₁.AL[1]) ⊕ _lastspace(ψ₂.AL[1]))', _lastspace(ψ₁.AL[1])'
    )
    F₂ = leftnull(F₁)
    @assert _lastspace(F₂) == _lastspace(ψ₂.AL[1])

    AL = ψ₁.AL[1] * F₁' + ψ₂.AL[1] * F₂'
    ψ.ALs[1], R = leftorth!(AL)

    for i in 2:halfN
        AL₁ = _transpose_front(F₁ * _transpose_tail(ψ₁.AL[i]))
        AL₂ = _transpose_front(F₂ * _transpose_tail(ψ₂.AL[i]))

        F₁ = isometry(
            storagetype(ψ), (_lastspace(AL₁) ⊕ _lastspace(ψ₂.AL[i]))', _lastspace(AL₁)'
        )
        F₂ = leftnull(F₁)
        @assert _lastspace(F₂) == _lastspace(ψ₂.AL[i])

        AL = _transpose_front(R * _transpose_tail(AL₁ * F₁' + AL₂ * F₂'))
        ψ.ALs[i], R = leftorth!(AL)
    end

    C₁ = F₁ * ψ₁.C[halfN]
    C₂ = F₂ * ψ₂.C[halfN]

    # right half
    F₁ = isometry(
        storagetype(ψ), _firstspace(ψ₁.AR[end]) ⊕ _firstspace(ψ₂.AR[end]), _firstspace(ψ₁.AR[end])
    )
    F₂ = leftnull(F₁)
    @assert _lastspace(F₂) == _firstspace(ψ₂.AR[end])'

    AR = F₁ * _transpose_tail(ψ₁.AR[end]) + F₂ * _transpose_tail(ψ₂.AR[end])
    L, AR′ = rightorth!(AR)
    ψ.ARs[end] = _transpose_front(AR′)

    for i in Iterators.reverse((halfN + 1):(length(ψ) - 1))
        AR₁ = _transpose_tail(ψ₁.AR[i] * F₁')
        AR₂ = _transpose_tail(ψ₂.AR[i] * F₂')

        F₁ = isometry(
            storagetype(ψ), _firstspace(ψ₁.AR[i]) ⊕ _firstspace(AR₂), _firstspace(ψ₁.AR[i])
        )
        F₂ = leftnull(F₁)
        @assert _lastspace(F₂) == _firstspace(AR₂)'

        AR = _transpose_tail(_transpose_front(F₁ * AR₁ + F₂ * AR₂) * L)
        L, AR′ = rightorth!(AR)
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

function r_RR(ψ::FiniteMPS{T}) where {T}
    return isomorphism(storagetype(T), domain(ψ.AR[end]), domain(ψ.AR[end]))
end
function l_LL(ψ::FiniteMPS{T}) where {T}
    return isomorphism(storagetype(T), space(ψ.AL[1], 1), space(ψ.AL[1], 1))
end
