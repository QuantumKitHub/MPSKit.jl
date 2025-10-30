#===========================================================================================
Tensor types
===========================================================================================#

"""
    MPOTensor{S}

Tensor type for representing local MPO tensors, with the index convention `W ⊗ S ← N ⊗ E`,
where `N`, `E`, `S` and `W` denote the north, east, south and west virtual spaces respectively.
"""
const MPOTensor{S} = AbstractTensorMap{T, S, 2, 2} where {T}
const MPSBondTensor{S} = AbstractTensorMap{T, S, 1, 1} where {T}
const GenericMPSTensor{S, N} = AbstractTensorMap{T, S, N, 1} where {T} # some functions are also defined for "general mps tensors" (used in peps code)
const MPSTensor{S} = GenericMPSTensor{S, 2} # the usual mps tensors on which we work

"""
    MPSTensor([f, eltype], d::Int, left_D::Int, [right_D]::Int])
    MPSTensor([f, eltype], physicalspace::Union{S,CompositeSpace{S}}, 
              left_virtualspace::S, [right_virtualspace]::S) where {S<:ElementarySpace}

Construct an `MPSTensor` with given physical and virtual spaces.

### Arguments
- `f::Function=rand`: initializer function for tensor data
- `eltype::Type{<:Number}=ComplexF64`: scalar type of tensors

- `physicalspace::Union{S,CompositeSpace{S}}`: physical space
- `left_virtualspace::S`: left virtual space
- `right_virtualspace::S`: right virtual space, defaults to equal left

- `d::Int`: physical dimension
- `left_D::Int`: left virtual dimension
- `right_D::Int`: right virtual dimension
"""
function MPSTensor(
        ::UndefInitializer, eltype, P::Union{S, CompositeSpace{S}}, Vₗ::S, Vᵣ::S = Vₗ
    ) where {S <: ElementarySpace}
    return TensorMap{eltype}(undef, Vₗ ⊗ P ← Vᵣ)
end
function MPSTensor(
        f, eltype, P::Union{S, CompositeSpace{S}}, Vₗ::S, Vᵣ::S = Vₗ
    ) where {S <: ElementarySpace}
    A = MPSTensor(undef, eltype, P, Vₗ, Vᵣ)
    if f === rand
        return rand!(A)
    elseif f === randn
        return randn!(A)
    elseif f === zeros
        return zeros!(A)
    else
        throw(ArgumentError("Unsupported initializer function: $f"))
    end
end
# TODO: reinstate function initializers?
function MPSTensor(
        P::Union{S, CompositeSpace{S}}, Vₗ::S, Vᵣ::S = Vₗ
    ) where {S <: ElementarySpace}
    return MPSTensor(rand, Defaults.eltype, P, Vₗ, Vᵣ)
end

"""
    MPSTensor([f, eltype], d::Int, Dₗ::Int, [Dᵣ]::Int])

Construct an `MPSTensor` with given physical and virtual dimensions.

### Arguments
- `f::Function=rand`: initializer function for tensor data
- `eltype::Type{<:Number}=ComplexF64`: scalar type of tensors
- `d::Int`: physical dimension
- `Dₗ::Int`: left virtual dimension
- `Dᵣ::Int`: right virtual dimension
"""
MPSTensor(f, eltype, d::Int, Dₗ::Int, Dᵣ::Int = Dₗ) = MPSTensor(f, eltype, ℂ^d, ℂ^Dₗ, ℂ^Dᵣ)
MPSTensor(d::Int, Dₗ::Int; Dᵣ::Int = Dₗ) = MPSTensor(ℂ^d, ℂ^Dₗ, ℂ^Dᵣ)

"""
    MPSTensor(A::AbstractArray)

Convert an array to an `MPSTensor`.
"""
function MPSTensor(A::AbstractArray{T}) where {T <: Number}
    @assert ndims(A) > 2 "MPSTensor should have at least 3 dims, but has $ndims(A)"
    sz = size(A)
    t = TensorMap(undef, T, foldl(⊗, ComplexSpace.(sz[1:(end - 1)])) ← ℂ^sz[end])
    t[] .= A
    return t
end
# Tensor accessors
# ----------------
@doc """
    AC2(ψ::AbstractMPS, i; kind=:ACAR)

Obtain the two-site (center) gauge tensor at site `i` of the MPS `ψ`.
If this hasn't been computed before, this can be computed as:
- `kind=:ACAR` : AC[i] * AR[i+1]
- `kind=:ALAC` : AL[i] * AC[i+1]
""" AC2

# ===========================================================================================
# MPS types
# ===========================================================================================

abstract type AbstractMPS end
abstract type AbstractFiniteMPS <: AbstractMPS end

Base.eltype(ψ::AbstractMPS) = eltype(typeof(ψ))
VectorInterface.scalartype(T::Type{<:AbstractMPS}) = scalartype(site_type(T))

function Base.checkbounds(ψ::AbstractMPS, i)
    return Base.checkbounds(Bool, ψ, i) || throw(BoundsError(ψ, i))
end

"""
    site_type(ψ::AbstractMPS)
    site_type(ψtype::Type{<:AbstractMPS})

Return the type of the site tensors of an `AbstractMPS`.
"""
site_type(ψ::AbstractMPS) = site_type(typeof(ψ))

"""
    bond_type(ψ::AbstractMPS)
    bond_type(ψtype::Type{<:AbstractMPS})

Return the type of the bond tensors of an `AbstractMPS`.
"""
bond_type(ψ::AbstractMPS) = bond_type(typeof(ψ))

TensorKit.spacetype(ψ::AbstractMPS) = spacetype(typeof(ψ))
TensorKit.spacetype(ψtype::Type{<:AbstractMPS}) = spacetype(site_type(ψtype))
TensorKit.sectortype(ψ::AbstractMPS) = sectortype(typeof(ψ))
TensorKit.sectortype(ψtype::Type{<:AbstractMPS}) = sectortype(site_type(ψtype))

"""
    left_virtualspace(ψ::AbstractMPS, [pos=1:length(ψ)])

Return the virtual space of the bond to the left of sites `pos`.

!!! warning
    In rare cases, the gauge tensor on the virtual space might not be square, and as a result it
    cannot always be guaranteed that `right_virtualspace(ψ, i - 1) == left_virtualspace(ψ, i)`
"""
function left_virtualspace end
left_virtualspace(A::GenericMPSTensor) = space(A, 1)
left_virtualspace(O::MPOTensor) = space(O, 1)
left_virtualspace(ψ::AbstractMPS) = map(Base.Fix1(left_virtualspace, ψ), eachsite(ψ))

"""
    right_virtualspace(ψ::AbstractMPS, [pos=1:length(ψ)])

Return the virtual space of the bond to the right of site(s) `pos`.

!!! warning
    In rare cases, the gauge tensor on the virtual space might not be square, and as a result it
    cannot always be guaranteed that `right_virtualspace(ψ, i - 1) == left_virtualspace(ψ, i)`
"""
function right_virtualspace end
right_virtualspace(A::GenericMPSTensor) = space(A, numind(A))'
right_virtualspace(O::MPOTensor) = space(O, 4)'
right_virtualspace(ψ::AbstractMPS) = map(Base.Fix1(right_virtualspace, ψ), eachsite(ψ))

"""
    physicalspace(ψ::AbstractMPS, [pos=1:length(ψ)])

Return the physical space of the site tensor at site `i`.
"""
function physicalspace end
physicalspace(A::MPSTensor) = space(A, 2)
physicalspace(A::GenericMPSTensor) = prod(x -> space(A, x), 2:(numind(A) - 1))
physicalspace(O::MPOTensor) = space(O, 2)
physicalspace(O::AbstractBlockTensorMap{<:Any, <:Any, 2, 2}) = only(space(O, 2))
physicalspace(ψ::AbstractMPS) = map(Base.Fix1(physicalspace, ψ), eachsite(ψ))

"""
    eachsite(state::AbstractMPS)

Return an iterator over the sites of the MPS `state`.
"""
eachsite(ψ::AbstractMPS) = eachindex(ψ)

# ===========================================================================================
# MPS manifolds
# ===========================================================================================
"""
    abstract type AbstractMPSManifold{S <: ElementarySpace}

Abstract supertype for the characterization of an MPS manifold, i.e. the sizes of all tensors that are involved.
These types are used mostly as convenient dispatch 
"""
abstract type AbstractMPSManifold{S <: ElementarySpace} end

TensorKit.spacetype(::Type{<:AbstractMPSManifold{S}}) where {S} = S

physicalspace(manifold::AbstractMPSManifold, i::Integer) = physicalspace(manifold)[i]
left_virtualspace(manifold::AbstractMPSManifold, i::Integer) = left_virtualspace(manifold)[i]
right_virtualspace(manifold::AbstractMPSManifold, i::Integer) = right_virtualspace(manifold)[i]

Base.getindex(manifold::AbstractMPSManifold, site::Integer) =
    left_virtualspace(manifold, site) ⊗ physicalspace(manifold, site) ← right_virtualspace(manifold, site)

"""
    FiniteMPSManifold(pspaces, vspaces) <: AbstractMPSManifold{S}

Full characterization of all [`FiniteMPS`](@ref) spaces. Both `pspaces` and `vspaces` are `Vector`s that hold the
local physical and virtual spaces, such that we have `length(pspaces) + 1 == length(vspaces)`.
These objects can be used to construct MPS for a given space, for example through [`Base.randn`](@ref) or similar functions.

## Constructors

    FiniteMPSManifold(physicalspaces::AbstractVector{<:TensorSpace}, max_virtualspaces; [left_virtualspace], [right_virtualspace])

To construct a `FiniteMPSManifold`, you should provide the `physicalspaces`, along with the maximal desired virtual space.
This latter can be provided as a single `<:ElementarySpace`, or, if a site-dependent maximum is desired through a vector thereof.
In that case, `length(physicalspaces) == length(max_virtualspaces) + 1`.

It is also possible to add non-trivial virtual spaces on both the left and the right edge of the MPS.
This might be useful to construct "charged" MPS, or to work with [`WindowMPS`](@ref) that may be embedded in a larger system.

!!! note
    As the supplied virtual spaces are interpreted as a maximum, this function will automatically construct "full-rank" spaces.
    These are chosen such that every MPS tensor that would be generated could have full rank. See also [`makefullrank!](@ref).
"""
struct FiniteMPSManifold{S <: ElementarySpace, S′ <: TensorSpace{S}} <: AbstractMPSManifold{S}
    pspaces::Vector{S′}
    vspaces::Vector{S}
end

function FiniteMPSManifold(
        physicalspaces::AbstractVector{S′}, max_virtualspaces::AbstractVector{S};
        left_virtualspace::S = oneunit(S), right_virtualspace::S = oneunit(S)
    ) where {S <: ElementarySpace, S′ <: TensorSpace{S}}
    L₁ = length(physicalspaces)
    L₂ = length(max_virtualspaces)
    L₁ == L₂ + 1 ||
        throw(DimensionMismatch(lazy"`|physicalspaces|` ($L₁) should be 1 more than `|max_virtualspaces|` ($L₂)"))

    # copy to avoid side-effects and get correct array type
    pspaces = collect(physicalspaces)
    vspaces = vcat(left_virtualspace, max_virtualspaces, right_virtualspace)
    manifold = FiniteMPSManifold{S, S′}(pspaces, vspaces)

    # ensure all spaces are full rank -- use vspaces as maximum
    return makefullrank!(manifold)
end
function FiniteMPSManifold(
        physicalspaces::AbstractVector{S′}, max_virtualspace::S; kwargs...
    ) where {S <: ElementarySpace, S′ <: TensorSpace{S}}
    return FiniteMPSManifold(physicalspaces, fill(max_virtualspace, length(physicalspaces) - 1))
end
function FiniteMPSManifold(mps_tensors::AbstractVector{A}) where {A <: GenericMPSTensor}
    numin(V) == 1 || throw(ArgumentError("Not a valid MPS tensor space"))
    pspaces = map(physicalspace, mps_tensors)
    vspaces = Vector{spacetype(A)}(undef, length(mps_tensors) - 1)
    for (i, mps_tensor) in enumerate(mps_tensors)
        i == length(mps_tensors) && continue
        vspaces[i] = right_virtualspace(mps_tensor)
    end
    return FiniteMPSManifold(
        pspaces, vspaces;
        left_virtualspace = left_virtualspace(first(mps_tensors)),
        right_virtualspace = right_virtualspace(last(mps_tensors))
    )
end

"""
    InfiniteMPSManifold(pspaces, vspaces) <: AbstractMPSManifold{S}

Full characterization of all [`InfiniteMPS`](@ref) spaces. Both `pspaces` and `vspaces` are `PeriodicVector`s that hold the
local physical and virtual spaces, such that we must have `length(pspaces) == length(vspaces)`.
These objects can be used to construct MPS for a given space, for example through [`Base.randn`](@ref) or similar functions.

## Constructors

    FiniteMPSManifold(physicalspaces::AbstractVector{<:TensorSpace}, max_virtualspaces; [left_virtualspace], [right_virtualspace])

To construct a `FiniteMPSManifold`, you should provide the `physicalspaces`, along with the maximal desired virtual space.
This latter can be provided as a single `<:ElementarySpace`, or, if a site-dependent maximum is desired through a vector thereof.
In that case, `length(physicalspaces) == length(max_virtualspaces)`.

!!! note
    As the supplied virtual spaces are interpreted as a maximum, this function will automatically construct "full-rank" spaces.
    These are chosen such that every MPS tensor that would be generated could have full rank. See also [`makefullrank!](@ref).
"""
struct InfiniteMPSManifold{S <: ElementarySpace, S′ <: Union{S, CompositeSpace{S}}} <: AbstractMPSManifold{S}
    pspaces::PeriodicVector{S′}
    vspaces::PeriodicVector{S}
end

function InfiniteMPSManifold(
        physicalspaces::AbstractVector{S′}, virtualspaces::AbstractVector{S}
    ) where {S <: ElementarySpace, S′ <: Union{S, CompositeSpace{S}}}
    L₁ = length(physicalspaces)
    L₂ = length(virtualspaces)
    L₁ == L₂ ||
        throw(DimensionMismatch(lazy"`|physicalspaces|` ($L₁) should be equal to `|virtualspaces|` ($L₂)"))

    # copy to avoid side-effects and get correct array type
    pspaces = collect(physicalspaces)
    vspaces = collect(max_virtualspaces)
    manifold = InfiniteMPSManifold{S, S′}(pspaces, vspaces)

    # ensure all spaces are full rank -- use vspaces as maximum
    return makefullrank!(manifold)
end

Base.length(manifold::AbstractMPSManifold) = length(physicalspace(manifold))

physicalspace(manifold::Union{FiniteMPSManifold, InfiniteMPSManifold}) = manifold.pspaces
left_virtualspace(manifold::FiniteMPSManifold) = manifold.vspaces[1:(end - 1)]
left_virtualspace(manifold::InfiniteMPSManifold) = manifold.vspaces
right_virtualspace(manifold::FiniteMPSManifold) = manifold.vspaces[2:end]
right_virtualspace(manifold::InfiniteMPSManifold) = PeriodicVector(circshift(manifold.vspaces, 1))

# MPS constructors
# ----------------
for randf in (:rand, :randn, :randexp, :randisometry)
    _docstr = """
        $randf([rng=default_rng()], [T=Float64], manifold::AbstractMPSManifold) -> mps
        
    Generate an `mps` with tensors generated by `$randf`.

    See also [`Random.$(randf)!`](@ref).
    """
    _docstr! = """
        $(randf)!([rng=default_rng()], mps::AbstractMPS) -> mps
        
    Fill the tensors of `mps` with entries generated by `$(randf)!`.

    See also [`Random.$(randf)`](@ref).
    """

    if randf != :randisometry
        randfun = GlobalRef(Random, randf)
        randfun! = GlobalRef(Random, Symbol(randf, :!))
    else
        randfun = randf
        randfun! = Symbol(randf, :!)
    end

    @eval begin
        @doc $_docstr $randfun(::Type, ::AbstractMPSManifold)
        @doc $_docstr! $randfun!(::AbstractMPSManifold)

        # filling in default eltype
        $randfun(manifold::AbstractMPSManifold) = $randfun(Defaults.eltype, manifold)
        function $randfun(rng::Random.AbstractRNG, manifold::AbstractMPSManifold)
            return $randfun(rng, Defaults.eltype, manifold)
        end

        # filling in default rng
        function $randfun(::Type{T}, manifold::AbstractMPSManifold) where {T}
            return $randfun(Random.default_rng(), T, manifold)
        end
        $randfun!(mps::AbstractMPS) = $randfun!(Random.default_rng(), mps)
    end
end

"""
    isfullrank(A::GenericMPSTensor; side=:both)

Determine whether the given tensor is full rank, i.e. whether both the map from the left
virtual space and the physical space to the right virtual space, and the map from the right
virtual space and the physical space to the left virtual space are injective.
"""
isfullrank(A::GenericMPSTensor; kwargs...) = isfullrank(space(A); kwargs...)
function isfullrank(V::TensorKit.TensorMapSpace; side = :both)
    Vₗ = V[1]
    Vᵣ = V[numind(V)]
    P = ⊗(getindex.(Ref(V), 2:(numind(V) - 1))...)
    return if side === :both
        Vₗ ⊗ P ≿ Vᵣ' && Vₗ' ≾ P ⊗ Vᵣ
    elseif side === :right
        Vₗ ⊗ P ≿ Vᵣ'
    elseif side === :left
        Vₗ' ≾ P ⊗ Vᵣ
    else
        throw(ArgumentError("Invalid side: $side"))
    end
end
isfullrank(manifold::AbstractMPSManifold; kwargs...) = all(i -> isfullrank(manifold[i]; kwargs...), 1:length(manifold))

function makefullrank!(manifold::FiniteMPSManifold)
    # left-to-right sweep
    for site in 1:length(manifold)
        if !isfullrank(manifold[site]; side = :right)
            maxspace = fuse(left_virtualspace(manifold, i), fuse(physicalspace(manifold, site)))
            manifold.vspaces[site + 1] = infimum(right_virtualspace(manifold, site), maxspace)
        end
    end
    # right-to-left sweep
    for site in reverse(1:length(manifold))
        if !isfullrank(manifold[site]; side = :left)
            maxspace = fuse(right_virtualspace(manifold, i), dual(fuse(physicalspace(manifold, site))))
            manifold.vspaces[site] = infimum(left_virtualspace(manifold, site), maxspace)
        end
    end
    return manifold
end
function makefullrank!(manifold::InfiniteMPSManifold)
    haschanged = true
    while haschanged
        haschanged = false
        # left-to-right sweep
        for site in 1:length(manifold)
            if !isfullrank(manifold[site]; side = :right)
                maxspace = fuse(left_virtualspace(manifold, i), fuse(physicalspace(manifold, site)))
                manifold.vspaces[site + 1] = infimum(right_virtualspace(manifold, site), maxspace)
                haschanged = true
            end
        end
        # right-to-left sweep
        for site in reverse(1:length(manifold))
            if !isfullrank(manifold[site]; side = :left)
                maxspace = fuse(right_virtualspace(manifold, i), dual(fuse(physicalspace(manifold, site))))
                manifold.vspaces[site] = infimum(left_virtualspace(manifold, site), maxspace)
                haschanged = true
            end
        end
    end
    return manifold
end

"""
    makefullrank!(A::PeriodicVector{<:GenericMPSTensor}; alg_leftorth = Defaults.alg_qr(), alg_rightorth = Defaults.alg_lq())

Make the set of MPS tensors full rank by performing a series of orthogonalizations.
"""
function makefullrank!(A::PeriodicVector{<:GenericMPSTensor}; alg_leftorth = Defaults.alg_qr(), alg_rightorth = Defaults.alg_lq())
    while true
        i = findfirst(!isfullrank, A)
        isnothing(i) && break
        if !isfullrank(A[i]; side = :left)
            L, Q = right_orth!(_transpose_tail(A[i]); alg = alg_rightorth)
            A[i] = _transpose_front(Q)
            A[i - 1] = A[i - 1] * L
        else
            A[i], R = left_orth!(A[i]; alg = alg_leftorth)
            A[i + 1] = _transpose_front(R * _transpose_tail(A[i + 1]))
        end
    end
    return A
end
