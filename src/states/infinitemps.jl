"""
    InfiniteMPS{A <: GenericMPSTensor, B <: MPSBondTensor} <: AbtractMPS

Type that represents an infinite Matrix Product State.

## Fields
- `AL` -- left-gauged MPS tensors
- `AR` -- right-gauged MPS tensors
- `AC` -- center-gauged MPS tensors
- `C` -- gauge tensors

## Notes
By convention, we have that:
- `AL[i] * C[i]` = `AC[i]` = `C[i-1] * AR[i]`
- `AL[i]' * AL[i] = 1`
- `AR[i] * AR[i]' = 1`

---

## Constructors

Recommended ways to construct an infinite (periodic unit-cell) MPS are:

- Using an MPS structure of spaces

  ```julia
  rand([rng], [T], structure::InfiniteMPSStructure; tol, maxiter)
  randn([rng], [T], structure::InfiniteMPSStructure; tol, maxiter)
  ```

  Build an [`InfiniteMPSStructure`](@ref) with physical spaces and (maximal) virtual spaces for the unit cell.

- From unit-cell site tensors

  ```julia
  InfiniteMPS(As::AbstractVector{<:GenericMPSTensor}; tol, maxiter)
  ```

  Takes a vector `As` of (full-rank preferred) site tensors defining one unit cell.
  The tensors are gauge-fixed (left/right) and internal bond tensors `C` are produced.
  If any tensor isn't full rank, a warning is emitted and `makefullrank!` is applied.

- From left- or right- gauged tensors and an initial gauge tensors

  ```julia
  InfiniteMPS(ALs::AbstractVector{<:GenericMPSTensor}, C₀::MPSBondTensor; tol, maxiter)
  InfiniteMPS(C₀::MPSBondTensor, ARs::AbstractVector{<:GenericMPSTensor}; tol, maxiter)
  ```

  Starts from gauged tensors `ALs` or `ARs` and an initial center bond `C₀` and completes the other gauge.

### Keywords (passed to [`gaugefix!`](@ref))
- `tol`: convergence tolerance for gauge fixing.
- `maxiter`: maximum iterations in gauge fixing.
- Additional keyword arguments accepted by `gaugefix!` (e.g. `order = :L | :R`).

### Examples
```julia
using MPSKit, TensorKit
ps  = PeriodicVector([ℂ^2, ℂ^2, ℂ^2])              # physical spaces, 3-site unit cell
vs  = PeriodicVector([ℂ^4, ℂ^8, ℂ^4])               # maximal virtual spaces
m   = InfiniteMPSStructure(ps, vs)
ψ   = rand(ComplexF64, m; tol=1e-10)

# Construct from pre-built site tensors
As = map(i -> rand(ComplexF64, m[i]), 1:length(m))
ψ2 = InfiniteMPS(As; tol=1e-10)
```

!!! warning "Deprecated constructors"
    Older constructors like `InfiniteMPS(pspaces, Dspaces)` or with `[f, T]` are deprecated.
    Use `rand`/`randn` with an [`InfiniteMPSStructure`](@ref) instead.
"""
struct InfiniteMPS{A <: GenericMPSTensor, B <: MPSBondTensor} <: AbstractMPS
    AL::PeriodicVector{A}
    AR::PeriodicVector{A}
    C::PeriodicVector{B}
    AC::PeriodicVector{A}

    # constructor from data
    function InfiniteMPS{A, B}(
            AL::PeriodicVector{A}, AR::PeriodicVector{A},
            C::PeriodicVector{B}, AC::PeriodicVector{A} = AL .* C
        ) where {A <: GenericMPSTensor, B <: MPSBondTensor}
        # verify lengths are compatible
        L = length(AL)
        L == length(AR) == length(C) == length(AC) ||
            throw(ArgumentError("incompatible lengths of AL, AR, C, and AC"))
        # verify tensors are compatible
        spacetype(A) == spacetype(B) ||
            throw(SpaceMismatch("incompatible space types of AL and C"))

        return new{A, B}(AL, AR, C, AC)
    end

    # undef constructors
    function InfiniteMPS{A, B}(
            ::UndefInitializer, L::Integer
        ) where {A <: GenericMPSTensor, B <: MPSBondTensor}
        AL = PeriodicVector(Vector{A}(undef, L))
        AR = PeriodicVector(Vector{A}(undef, L))
        C = PeriodicVector(Vector{B}(undef, L))
        AC = PeriodicVector(Vector{A}(undef, L))
        return new{A, B}(AL, AR, C, AC)
    end

end

function InfiniteMPS{A, B}(
        ::UndefInitializer, structure::InfiniteMPSStructure
    ) where {A <: GenericMPSTensor, B <: MPSBondTensor}
    psi = InfiniteMPS{A, B}(undef, length(structure))
    for i in 1:length(structure)
        V = structure[i]
        psi.AL[i] = A(undef, V)
        psi.AR[i] = A(undef, V)
        psi.AC[i] = A(undef, V)
        Vr = right_virtualspace(structure, i)
        psi.C[i] = B(undef, Vr ← Vr)
    end
    return psi
end
function InfiniteMPS{A}(
        ::UndefInitializer, structure::InfiniteMPSStructure
    ) where {A <: GenericMPSTensor}
    B = tensormaptype(spacetype(A), 1, 1, storagetype(A))
    return InfiniteMPS{A, B}(undef, structure)

end
function InfiniteMPS(
        AL::PeriodicVector{A}, AR::PeriodicVector{A},
        C::PeriodicVector{B}, AC::PeriodicVector{A} = AL .* C
    ) where {A <: GenericMPSTensor, B <: MPSBondTensor}
    (L = length(AL)) == length(AR) == length(C) == length(AC) ||
        throw(ArgumentError("incompatible lengths of AL, AR, C, and AC"))
    spacetype(A) == spacetype(B) ||
        throw(SpaceMismatch("incompatible space types of AL and C"))

    for i in 1:L
        numind(AL[i]) == numind(AR[i]) == numind(AC[i]) ||
            throw(SpaceMismatch("incompatible spaces at site $i"))
        space(AL[i]) == space(AR[i]) == space(AC[i]) ||
            throw(SpaceMismatch("incompatible spaces at site $i"))
        domain(C[i]) == codomain(C[i]) ||
            throw(SpaceMismatch("Non-square C at site $i"))
        right_virtualspace(AL[i]) == left_virtualspace(AR[i + 1]) ||
            throw(SpaceMismatch("incompatible spaces between site $i and site $(i + 1)"))

        # verify that the spaces are non-zero
        (dim(space(AL[i])) > 0 && dim(space(C[i])) > 0) ||
            @warn "no fusion channels available at site $i"
    end

    return InfiniteMPS{A, B}(AL, AR, C, AC)
end

function InfiniteMPS(
        AL::AbstractVector{A}, AR::AbstractVector{A}, C::AbstractVector{B},
        AC::AbstractVector{A} = AL .* C
    ) where {A <: GenericMPSTensor, B <: MPSBondTensor}
    return InfiniteMPS(
        convert(PeriodicVector{A}, AL), convert(PeriodicVector{A}, AR),
        convert(PeriodicVector{B}, C), convert(PeriodicVector{A}, AC)
    )
end

#===========================================================================================
Constructors
===========================================================================================#

function InfiniteMPS(A::AbstractVector{<:GenericMPSTensor}; kwargs...)
    # check spaces
    leftvspaces = circshift(left_virtualspace.(A), -1)
    rightvspaces = right_virtualspace.(A)
    isnothing(findfirst(leftvspaces .!= rightvspaces)) ||
        throw(SpaceMismatch("incompatible virtual spaces $leftvspaces and $rightvspaces"))

    # check rank
    A_copy = PeriodicArray(copy.(A)) # copy to avoid side effects
    all(isfullrank, A_copy) ||
        @warn "Constructing an MPS from tensors that are not full rank"
    makefullrank!(A_copy)

    structure = InfiniteMPSStructure(A_copy)
    ψ = InfiniteMPS{eltype(A)}(undef, structure)

    # gaugefix the MPS
    V = left_virtualspace(ψ, 1)
    C₀ = isomorphism(storagetype(eltype(A_copy)), V, V)
    gaugefix!(ψ, A_copy, C₀; kwargs...)
    mul!.(ψ.AC, ψ.AL, ψ.C)

    return ψ
end

function InfiniteMPS(AL::AbstractVector{<:GenericMPSTensor}, C₀::MPSBondTensor; kwargs...)
    AL = PeriodicArray(AL)

    all(isfullrank, AL) || @error "Constructing an MPS from tensors that are not full rank"
    ψ = InfiniteMPS{eltype(AL)}(undef, InfiniteMPSStructure(AL))
    ψ.AL .= AL

    # gaugefix the MPS
    gaugefix!(ψ, AL, C₀; order = :R, kwargs...)
    mul!.(ψ.AC, ψ.AL, ψ.C)

    return ψ
end

function InfiniteMPS(C₀::MPSBondTensor, AR::AbstractVector{<:GenericMPSTensor}; kwargs...)
    AR = PeriodicArray(AR)

    all(isfullrank, AR) || @error "Constructing an MPS from tensors that are not full rank"
    ψ = InfiniteMPS{eltype(AR)}(undef, InfiniteMPSStructure(AR))
    ψ.AR .= AR

    # gaugefix the MPS
    gaugefix!(ψ, AR, C₀; order = :L, kwargs...)
    mul!.(ψ.AC, ψ.AL, ψ.C)

    return ψ
end

for randfun in (:rand, :randn)
    randfun! = Symbol(randfun, :!)
    @eval function Random.$randfun(rng::Random.AbstractRNG, T::Type, structure::InfiniteMPSStructure; kwargs...)
        As = map(i -> $randfun(rng, T, structure[i]), 1:length(structure))
        return InfiniteMPS(As; kwargs...)
    end
    @eval function Random.$randfun!(rng::Random.AbstractRNG, mps::InfiniteMPS)
        foreach(Base.Fix1(rng, $randfun!), mps.AC)
        C₀ = $randfun!(rng, mps.C[0])
        gaugefix!(mps, mps.AC, C₀)
        mul!.(mps.AC, mps.AL, mps.C)
        return mps
    end
end

# Deprecate old constructor syntaxes
# ----------------------------------
Base.@deprecate(
    InfiniteMPS(pspace::S, Dspace::S; kwargs...) where {S <: IndexSpace},
    rand(InfiniteMPSStructure(pspace, Dspace); kwargs...)
)
Base.@deprecate(
    InfiniteMPS(pspaces::AbstractVector{S}, Dspaces::AbstractVector{S}; kwargs...) where {S <: IndexSpace},
    rand(InfiniteMPSStructure(pspaces, Dspaces); kwargs...)
)
Base.@deprecate(
    InfiniteMPS(f, T::Type, pspace::S, Dspace::S; kwargs...) where {S <: IndexSpace},
    f(T, InfiniteMPSStructure(pspace, Dspace); kwargs...)
)
Base.@deprecate(
    InfiniteMPS(f, T::Type, pspaces::AbstractVector{S}, Dspaces::AbstractVector{S}; kwargs...) where {S <: IndexSpace},
    f(T, InfiniteMPSStructure(pspaces, Dspaces); kwargs...)
)

Base.@deprecate(
    InfiniteMPS(ds::AbstractVector{Int}, Ds::AbstractVector{Int}),
    rand(InfiniteMPSStructure(ComplexSpace.(ds), ComplexSpace.(Ds)))
)
Base.@deprecate(
    InfiniteMPS(f, T::Type, ds::AbstractVector{Int}, Ds::AbstractVector{Int}),
    f(T, InfiniteMPSStructure(ComplexSpace.(ds), ComplexSpace.(Ds)))
)

#===========================================================================================
Utility
===========================================================================================#

function AC2(ψ::InfiniteMPS, i::Integer; kind = :ACAR)
    if kind == :ACAR
        return ψ.AC[i] * _transpose_tail(ψ.AR[i + 1])
    elseif kind == :ALAC
        return ψ.AL[i] * _transpose_tail(ψ.AC[i + 1])
    else
        throw(ArgumentError("Invalid kind: $kind"))
    end
end

Base.size(ψ::InfiniteMPS, args...) = size(ψ.AL, args...)
Base.length(ψ::InfiniteMPS) = length(ψ.AL)
Base.eltype(ψ::InfiniteMPS) = eltype(typeof(ψ))
Base.eltype(::Type{<:InfiniteMPS{A}}) where {A} = A
Base.isfinite(::Type{<:InfiniteMPS}) = false
GeometryStyle(::Type{<:InfiniteMPS}) = InfiniteChainStyle()

Base.copy(ψ::InfiniteMPS) = InfiniteMPS(copy(ψ.AL), copy(ψ.AR), copy(ψ.C), copy(ψ.AC))
function Base.copy!(ψ::InfiniteMPS, ϕ::InfiniteMPS)
    ψ.AL .= _copy!!.(ψ.AL, ϕ.AL)
    ψ.AR .= _copy!!.(ψ.AR, ϕ.AR)
    ψ.AC .= _copy!!.(ψ.AC, ϕ.AC)
    ψ.C .= _copy!!.(ψ.C, ϕ.C)
    return ψ
end
# possible in-place copy
function _copy!!(dst::AbstractTensorMap, src::AbstractTensorMap)
    return space(dst) == space(src) ? copy!(dst, src) : copy(src)
end

function Base.complex(ψ::InfiniteMPS)
    scalartype(ψ) <: Complex && return ψ
    return InfiniteMPS(complex.(ψ.AL), complex.(ψ.AR), complex.(ψ.C), complex.(ψ.AC))
end

function Base.repeat(ψ::InfiniteMPS, i::Int)
    return InfiniteMPS(repeat(ψ.AL, i), repeat(ψ.AR, i), repeat(ψ.C, i), repeat(ψ.AC, i))
end
function Base.similar(ψ::InfiniteMPS{A, B}) where {A, B}
    return InfiniteMPS{A, B}(similar(ψ.AL), similar(ψ.AR), similar(ψ.C), similar(ψ.AC))
end
function Base.circshift(ψ::InfiniteMPS, n)
    return InfiniteMPS(
        circshift(ψ.AL, n), circshift(ψ.AR, n), circshift(ψ.C, n), circshift(ψ.AC, n)
    )
end

Base.eachindex(ψ::InfiniteMPS) = eachindex(ψ.AL)
Base.eachindex(l::IndexStyle, ψ::InfiniteMPS) = eachindex(l, ψ.AL)
eachsite(ψ::InfiniteMPS) = PeriodicArray(eachindex(ψ))

Base.checkbounds(::Type{Bool}, ψ::InfiniteMPS, i::Integer) = true

site_type(::Type{<:InfiniteMPS{A}}) where {A} = A
bond_type(::Type{<:InfiniteMPS{<:Any, B}}) where {B} = B

left_virtualspace(ψ::InfiniteMPS, n::Integer) = left_virtualspace(ψ.AL[n])
right_virtualspace(ψ::InfiniteMPS, n::Integer) = right_virtualspace(ψ.AL[n])
physicalspace(ψ::InfiniteMPS, n::Integer) = physicalspace(ψ.AL[n])

# TensorKit.space(ψ::InfiniteMPS{<:MPSTensor}, n::Integer) = space(ψ.AC[n], 2)
# function TensorKit.space(ψ::InfiniteMPS{<:GenericMPSTensor}, n::Integer)
#     t = ψ.AC[n]
#     S = spacetype(t)
#     return ProductSpace{S}(space.(Ref(t), Base.front(Base.tail(TensorKit.allind(t)))))
# end

TensorKit.norm(ψ::InfiniteMPS) = norm(ψ.AC[1])
function TensorKit.normalize!(ψ::InfiniteMPS)
    normalize!.(ψ.C)
    normalize!.(ψ.AC)
    return ψ
end

function TensorKit.dot(ψ₁::InfiniteMPS, ψ₂::InfiniteMPS; krylovdim = 30)
    init = similar(ψ₁.AL[1], _firstspace(ψ₂.AL[1]) ← _firstspace(ψ₁.AL[1]))
    randomize!(init)
    val, = fixedpoint(
        TransferMatrix(ψ₂.AL, ψ₁.AL), init, :LM, Arnoldi(; krylovdim = krylovdim)
    )
    return val
end
function Base.isapprox(ψ₁::InfiniteMPS, ψ₂::InfiniteMPS; kwargs...)
    return isapprox(dot(ψ₁, ψ₂), 1; kwargs...)
end

#===========================================================================================
Fixedpoints
===========================================================================================#

"""
    l_RR(ψ, location)

Left dominant eigenvector of the `AR`-`AR` transfermatrix.
"""
l_RR(ψ::InfiniteMPS, loc::Int = 1) = adjoint(ψ.C[loc - 1]) * ψ.C[loc - 1]

"""
    l_RL(ψ, location)

Left dominant eigenvector of the `AR`-`AL` transfermatrix.
"""
l_RL(ψ::InfiniteMPS, loc::Int = 1) = ψ.C[loc - 1]

"""
    l_LR(ψ, location)

Left dominant eigenvector of the `AL`-`AR` transfermatrix.
"""
l_LR(ψ::InfiniteMPS, loc::Int = 1) = ψ.C[loc - 1]'

"""
    l_LL(ψ, location)

Left dominant eigenvector of the `AL`-`AL` transfermatrix.
"""
function l_LL(ψ::InfiniteMPS{A}, loc::Int = 1) where {A}
    return isomorphism(storagetype(A), left_virtualspace(ψ, loc), left_virtualspace(ψ, loc))
end

"""
    r_RR(ψ, location)

Right dominant eigenvector of the `AR`-`AR` transfermatrix.
"""
function r_RR(ψ::InfiniteMPS{A}, loc::Int = length(ψ)) where {A}
    return isomorphism(
        storagetype(A), right_virtualspace(ψ, loc), right_virtualspace(ψ, loc)
    )
end

"""
    r_RL(ψ, location)

Right dominant eigenvector of the `AR`-`AL` transfermatrix.
"""
r_RL(ψ::InfiniteMPS, loc::Int = length(ψ)) = ψ.C[loc]'

"""
    r_LR(ψ, location)

Right dominant eigenvector of the `AL`-`AR` transfermatrix.
"""
r_LR(ψ::InfiniteMPS, loc::Int = length(ψ)) = ψ.C[loc]

"""
    r_LL(ψ, location)

Right dominant eigenvector of the `AL`-`AL` transfermatrix.
"""
r_LL(ψ::InfiniteMPS, loc::Int = length(ψ)) = ψ.C[loc] * adjoint(ψ.C[loc])
