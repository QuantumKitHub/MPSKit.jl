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
# MPS structure
# ===========================================================================================
"""
    abstract type AbstractMPSStructure{S <: ElementarySpace}

Abstract supertype for the structure of an MPS, i.e. the sizes of all tensors that are involved.
These types are used mostly as convenient dispatch hooks.

See also [`FiniteMPSStructure`](@ref) and [`InfiniteMPSStructure`](@ref).
"""
abstract type AbstractMPSStructure{S <: ElementarySpace} end

TensorKit.spacetype(::Type{<:AbstractMPSStructure{S}}) where {S} = S

physicalspace(structure::AbstractMPSStructure, i::Integer) = physicalspace(structure)[i]
left_virtualspace(structure::AbstractMPSStructure, i::Integer) = left_virtualspace(structure)[i]
right_virtualspace(structure::AbstractMPSStructure, i::Integer) = right_virtualspace(structure)[i]

Base.getindex(structure::AbstractMPSStructure, site::Integer) =
    left_virtualspace(structure, site) ⊗ physicalspace(structure, site) ← right_virtualspace(structure, site)

"""
    FiniteMPSStructure(pspaces, vspaces) <: AbstractMPSStructure{S}

Full characterization of all [`FiniteMPS`](@ref) spaces. Both `pspaces` and `vspaces` are `Vector`s that hold the
local physical and virtual spaces, such that we have `length(pspaces) + 1 == length(vspaces)`.
These objects can be used to construct MPS for a given space, for example through [`Base.randn`](@ref) or similar functions.

## Constructors

    FiniteMPSStructure(physicalspaces::AbstractVector{<:TensorSpace}, max_virtualspaces; [left_virtualspace], [right_virtualspace])

To construct a `FiniteMPSStructure`, you should provide the `physicalspaces`, along with the maximal desired virtual space.
This latter can be provided as a single `<:ElementarySpace`, or, if a site-dependent maximum is desired through a vector thereof.
In that case, `length(physicalspaces) == length(max_virtualspaces) + 1`.

It is also possible to add non-trivial virtual spaces on both the left and the right edge of the MPS.
This might be useful to construct "charged" MPS, or to work with [`WindowMPS`](@ref) that may be embedded in a larger system.

!!! note
    As the supplied virtual spaces are interpreted as a maximum, this function will automatically construct "full-rank" spaces.
    These are chosen such that every MPS tensor that would be generated could have full rank. See also [`makefullrank!](@ref).
"""
struct FiniteMPSStructure{S <: ElementarySpace, S′ <: TensorSpace{S}} <: AbstractMPSStructure{S}
    pspaces::Vector{S′}
    vspaces::Vector{S}

    # disable default constructor
    function FiniteMPSStructure{S, S′}(
            pspaces::Vector{S′}, vspaces::Vector{S}
        ) where {S <: ElementarySpace, S′ <: TensorSpace{S}}
        return new{S, S′}(pspaces, vspaces)
    end
end

function FiniteMPSStructure(
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
    structure = FiniteMPSStructure{S, S′}(pspaces, vspaces)

    # ensure all spaces are full rank -- use vspaces as maximum
    return makefullrank!(structure)
end
function FiniteMPSStructure(
        physicalspaces::AbstractVector{S′}, max_virtualspace::S; kwargs...
    ) where {S <: ElementarySpace, S′ <: TensorSpace{S}}
    return FiniteMPSStructure(physicalspaces, fill(max_virtualspace, length(physicalspaces) - 1); kwargs...)
end
function FiniteMPSStructure(mps_tensors::AbstractVector{A}) where {A <: GenericMPSTensor}
    numin(A) == 1 || throw(ArgumentError("Not a valid MPS tensor space"))
    pspaces = map(physicalspace, mps_tensors)
    vspaces = Vector{spacetype(A)}(undef, length(mps_tensors) - 1)
    for (i, mps_tensor) in enumerate(mps_tensors)
        i == length(mps_tensors) && continue
        vspaces[i] = right_virtualspace(mps_tensor)
    end
    return FiniteMPSStructure(
        pspaces, vspaces;
        left_virtualspace = left_virtualspace(first(mps_tensors)),
        right_virtualspace = right_virtualspace(last(mps_tensors))
    )
end

"""
    InfiniteMPSStructure(pspaces, vspaces) <: AbstractMPSStructure{S}

Full characterization of all [`InfiniteMPS`](@ref) spaces. Both `pspaces` and `vspaces` are `PeriodicVector`s that hold the
local physical and virtual spaces, such that we must have `length(pspaces) == length(vspaces)`.
These objects can be used to construct MPS for a given space, for example through [`Base.randn`](@ref) or similar functions.

## Constructors

    FiniteMPSStructure(physicalspaces::AbstractVector{<:TensorSpace}, max_virtualspaces; [left_virtualspace], [right_virtualspace])

To construct a `FiniteMPSStructure`, you should provide the `physicalspaces`, along with the maximal desired virtual space.
This latter can be provided as a single `<:ElementarySpace`, or, if a site-dependent maximum is desired through a vector thereof.
In that case, `length(physicalspaces) == length(max_virtualspaces)`.

!!! note
    As the supplied virtual spaces are interpreted as a maximum, this function will automatically construct "full-rank" spaces.
    These are chosen such that every MPS tensor that would be generated could have full rank. See also [`makefullrank!](@ref).
"""
struct InfiniteMPSStructure{S <: ElementarySpace, S′ <: Union{S, CompositeSpace{S}}} <: AbstractMPSStructure{S}
    pspaces::PeriodicVector{S′}
    vspaces::PeriodicVector{S}
end

function InfiniteMPSStructure(
        physicalspaces::AbstractVector{S′}, virtualspaces::AbstractVector{S}
    ) where {S <: ElementarySpace, S′ <: Union{S, CompositeSpace{S}}}
    L₁ = length(physicalspaces)
    L₂ = length(virtualspaces)
    L₁ == L₂ ||
        throw(DimensionMismatch(lazy"`|physicalspaces|` ($L₁) should be equal to `|virtualspaces|` ($L₂)"))

    # copy to avoid side-effects and get correct array type
    pspaces = collect(physicalspaces)
    vspaces = collect(virtualspaces)
    structure = InfiniteMPSStructure{S, S′}(pspaces, vspaces)

    # ensure all spaces are full rank -- use vspaces as maximum
    return makefullrank!(structure)
end
function InfiniteMPSStructure(mps_tensors::AbstractVector{A}) where {A <: GenericMPSTensor}
    pspaces = PeriodicVector(map(physicalspace, mps_tensors))
    vspaces = PeriodicVector(map(left_virtualspace, mps_tensors))
    for i in eachindex(vspaces)
        vspaces[i] == right_virtualspace(mps_tensors[i - 1]) ||
            throw(SpaceMismatch("incompatible spaces between site $(i - 1) and $i"))
    end
    return InfiniteMPSStructure(pspaces, vspaces)
end

Base.length(structure::AbstractMPSStructure) = length(physicalspace(structure))

physicalspace(structure::Union{FiniteMPSStructure, InfiniteMPSStructure}) = structure.pspaces
left_virtualspace(structure::FiniteMPSStructure) = structure.vspaces[1:(end - 1)]
left_virtualspace(structure::InfiniteMPSStructure) = structure.vspaces
right_virtualspace(structure::FiniteMPSStructure) = structure.vspaces[2:end]
right_virtualspace(structure::InfiniteMPSStructure) = PeriodicVector(circshift(structure.vspaces, 1))

# Utility
function Base.repeat(structure::FiniteMPSStructure, i::Integer)
    last(structure.vspaces) == first(structure.vspaces) || throw(SpaceMismatch())
    pspaces = repeat(structure.pspaces, i)
    vspaces = push!(repeat(left_virtualspace(structure), i), last(structure.vspaces))
    return FiniteMPSStructure(pspaces, vspaces)
end
function Base.repeat(structure::InfiniteMPSStructure, i::Integer)
    pspaces = repeat(structure.pspaces, i)
    vspaces = repeat(structure.vspaces, i)
    return InfiniteMPSStructure(pspaces, vspaces)
end

function Base.vcat(structure1::FiniteMPSStructure, structure2::FiniteMPSStructure)
    last(right_virtualspace(structure1)) == first(left_virtualspace(structure2)) || throw(SpaceMismatch())
    pspaces = vcat(structure1.pspaces, structure2.pspaces)
    vspaces = push!(vcat(left_virtualspace(structure1), left_virtualspace(structure2)), last(structure2.vspaces))
    return FiniteMPSStructure(pspaces, vspaces)
end
Base.vcat(structure::FiniteMPSStructure, structures::FiniteMPSStructure...) = foldl(vcat, (structure, structures...))
function Base.vcat(structure::InfiniteMPSStructure, structures::InfiniteMPSStructure...)
    pspaces = vcat(structure.pspaces, Base.Fix2(getproperty, :pspaces).(structures)...)
    vspaces = vcat(structure.vspaces, Base.Fix2(getproperty, :vspaces).(structures)...)
    return InfiniteMPSStructure(pspaces, vspaces)
end

# MPS constructors
# ----------------
for randf in (:rand, :randn)
    _docstr = """
        $randf([rng=default_rng()], [T=Float64], structure::AbstractMPSStructure) -> mps
        
    Generate an `mps` with tensors generated by `$randf`.

    See also [`Random.$(randf)!`](@ref).
    """
    _docstr! = """
        $(randf)!([rng=default_rng()], mps::AbstractMPS) -> mps
        
    Fill the tensors of `mps` with entries generated by `$(randf)!`.

    See also [`Random.$(randf)`](@ref).
    """

    randfun = GlobalRef(Random, randf)
    randfun! = GlobalRef(Random, Symbol(randf, :!))

    @eval begin
        @doc $_docstr $randfun(::Type, ::AbstractMPSStructure)
        @doc $_docstr! $randfun!(::AbstractMPSStructure)

        # filling in default eltype
        $randfun(structure::AbstractMPSStructure) = $randfun(Defaults.eltype, structure)
        function $randfun(rng::Random.AbstractRNG, structure::AbstractMPSStructure)
            return $randfun(rng, Defaults.eltype, structure)
        end

        # filling in default rng
        function $randfun(::Type{T}, structure::AbstractMPSStructure) where {T}
            return $randfun(Random.default_rng(), T, structure)
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
isfullrank(structure::AbstractMPSStructure; kwargs...) =
    all(i -> isfullrank(structure[i]; kwargs...), 1:length(structure))

function makefullrank!(structure::FiniteMPSStructure)
    # left-to-right sweep
    for site in 1:length(structure)
        if !isfullrank(structure[site]; side = :right)
            maxspace = fuse(left_virtualspace(structure, site), fuse(physicalspace(structure, site)))
            structure.vspaces[site + 1] = infimum(right_virtualspace(structure, site), maxspace)
        end
    end
    # right-to-left sweep
    for site in reverse(1:length(structure))
        if !isfullrank(structure[site]; side = :left)
            maxspace = fuse(right_virtualspace(structure, site), dual(fuse(physicalspace(structure, site))))
            structure.vspaces[site] = infimum(left_virtualspace(structure, site), maxspace)
        end
    end
    return structure
end
function makefullrank!(structure::InfiniteMPSStructure)
    haschanged = true
    while haschanged
        haschanged = false
        # left-to-right sweep
        for site in 1:length(structure)
            if !isfullrank(structure[site]; side = :right)
                maxspace = fuse(left_virtualspace(structure, site), fuse(physicalspace(structure, site)))
                structure.vspaces[site + 1] = infimum(right_virtualspace(structure, site), maxspace)
                haschanged = true
            end
        end
        # right-to-left sweep
        for site in reverse(1:length(structure))
            if !isfullrank(structure[site]; side = :left)
                maxspace = fuse(right_virtualspace(structure, site), dual(fuse(physicalspace(structure, site))))
                structure.vspaces[site] = infimum(left_virtualspace(structure, site), maxspace)
                haschanged = true
            end
        end
    end
    return structure
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
