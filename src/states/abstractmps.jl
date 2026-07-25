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
    $(TYPEDEF)

Helper type that stores the spaces of an `AbstractTensorMap` following the `GenericMPSTensor`
leg convention `Vₗ ⊗ P ← Vᵣ`, i.e. left virtual space and physical space(s) in the codomain,
right virtual space in the domain. Centralizing this convention here avoids having to define
constructors directly on `MPSTensor`/`GenericMPSTensor` (which are aliases for
`AbstractTensorMap` and thus not owned by this package).

### Fields
$(TYPEDFIELDS)
"""
struct MPSMapSpace{S <: ElementarySpace, Sₚ <: Union{S, CompositeSpace{S}}}
    "left virtual space"
    Vₗ::S
    "physical space; either a single `ElementarySpace` (one physical leg, as in `MPSTensor`) or a `CompositeSpace{S}` (several physical legs, as in `GenericMPSTensor`)"
    P::Sₚ
    "right virtual space"
    Vᵣ::S
end

"""
    MPSMapSpace(Vₗ::S, P::Sₚ)

Construct an `MPSMapSpace` with `Vᵣ` defaulting to `Vₗ`.
"""
MPSMapSpace(Vₗ::S, P::Sₚ) where {S <: ElementarySpace, Sₚ <: Union{S, CompositeSpace{S}}} =
    MPSMapSpace(Vₗ, P, Vₗ)

"""
    MPSMapSpace(d::Int, Dₗ::Int, [Dᵣ]::Int)

Construct an `MPSMapSpace` with given physical and virtual dimensions, using `ComplexSpace`
(`ℂ`) for all three spaces. `Dᵣ` defaults to `Dₗ`.

### Arguments
- `d::Int`: physical dimension
- `Dₗ::Int`: left virtual dimension
- `Dᵣ::Int`: right virtual dimension, defaults to `Dₗ`
"""
MPSMapSpace(d::Int, Dₗ::Int, Dᵣ::Int = Dₗ) = MPSMapSpace(ℂ^d, ℂ^Dₗ, ℂ^Dᵣ)

const _MPSMAPSPACE_FILL_DESCRIPTIONS = Dict(
    :rand => "filled with uniformly distributed random entries",
    :randn => "filled with normally distributed random entries",
    :zeros => "filled with zeros",
)

for f in (:rand, :randn, :zeros)
    fill_description = _MPSMAPSPACE_FILL_DESCRIPTIONS[f]
    @eval begin
        @doc """
            $($f)([T::Type=Defaults.eltype], A::MPSMapSpace)

        Construct a tensor with `eltype` `T` and spaces `A.Vₗ ⊗ A.P ← A.Vᵣ`, $($fill_description).
        """
        function Base.$f(::Type{T}, A::MPSMapSpace) where {T}
            return $f(T, A.Vₗ ⊗ A.P ← A.Vᵣ)
        end
        Base.$f(A::MPSMapSpace) = $f(Defaults.eltype, A)
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

"""
    makefullrank!(A::PeriodicVector{<:GenericMPSTensor}; alg=Defaults.alg_orth())

Make the set of MPS tensors full rank by performing a series of orthogonalizations.
"""
function makefullrank!(A::PeriodicVector{<:GenericMPSTensor}; alg_orth = Defaults.alg_orth())
    while true
        i = findfirst(!isfullrank, A)
        isnothing(i) && break
        if !isfullrank(A[i]; side = :left)
            L, Q = right_orth!(_transpose_tail(A[i]); alg = alg_orth)
            A[i] = _transpose_front(Q)
            A[i - 1] = A[i - 1] * L
        else
            A[i], R = left_orth!(A[i]; alg = alg_orth)
            A[i + 1] = _transpose_front(R * _transpose_tail(A[i + 1]))
        end
    end
    return A
end

function makefullrank!(virtualspaces::PeriodicVector{S}, physicalspaces::PeriodicVector{S}) where {S <: ElementarySpace}
    haschanged = true
    while haschanged
        haschanged = false
        for i in 1:length(virtualspaces)
            Vmax = fuse(virtualspaces[i - 1], physicalspaces[i - 1])
            if !(virtualspaces[i] ≾ Vmax)
                virtualspaces[i] = infimum(virtualspaces[i], Vmax)
                haschanged = true
            end
        end
        for i in reverse(1:length(virtualspaces))
            Vmax = fuse(dual(physicalspaces[i - 1]), virtualspaces[i])
            if !(virtualspaces[i - 1] ≾ Vmax)
                virtualspaces[i - 1] = infimum(virtualspaces[i - 1], Vmax)
                haschanged = true
            end
        end
    end

    return virtualspaces
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

#===========================================================================================
MPS types
===========================================================================================#

abstract type AbstractMPS end
abstract type AbstractFiniteMPS <: AbstractMPS end

Base.eltype(ψ::AbstractMPS) = eltype(typeof(ψ))
VectorInterface.scalartype(T::Type{<:AbstractMPS}) = scalartype(site_type(T))
Base.isfinite(ψ::AbstractMPS) = isfinite(typeof(ψ))

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

TensorKit.storagetype(ψtype::Type{<:AbstractMPS}) = storagetype(site_type(ψtype))

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

TensorKit.leftunit(ψ::AbstractMPS) = leftunit(first(sectors(left_virtualspace(ψ, 1))))
TensorKit.rightunit(ψ::AbstractMPS) = rightunit(first(sectors(right_virtualspace(ψ, 1))))
