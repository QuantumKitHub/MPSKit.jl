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
    makefullrank!(A::PeriodicVector{<:GenericMPSTensor}; alg=Defalts.alg_qr())

Make the set of MPS tensors full rank by performing a series of orthogonalizations.
"""
function makefullrank!(A::PeriodicVector{<:GenericMPSTensor}; alg = Defaults.alg_qr())
    while true
        i = findfirst(!isfullrank, A)
        isnothing(i) && break
        if !isfullrank(A[i]; side = :left)
            L, Q = _right_orth!(_transpose_tail(A[i]); alg)
            A[i] = _transpose_front(Q)
            A[i - 1] = A[i - 1] * L
        else
            A[i], R = _left_orth!(A[i]; alg)
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

AC2(psi::AbstractMPS, site::Int; kwargs...) = AC2(GeometryStyle(psi), psi, site; kwargs...)

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
eachsite(ψ::AbstractMPS) = eachsite(GeometryStyle(ψ), ψ)

eachsite(::GeometryStyle, ψ::AbstractMPS) = eachindex(ψ)

#===========================================================================================
TensorKit utility
===========================================================================================#

function TensorKit.dot(ψ₁::AbstractMPS, ψ₂::AbstractMPS; kwargs...)
    geometry_style = GeometryStyle(ψ₁) & GeometryStyle(ψ₂)
    return TensorKit.dot(geometry_style, ψ₁, ψ₂; kwargs...)
end
function Base.isapprox(ψ₁::AbstractMPS, ψ₂::AbstractMPS; kwargs...)
    return isapprox(dot(ψ₁, ψ₂), 1; kwargs...)
end
TensorKit.norm(ψ::AbstractMPS) = TensorKit.norm(GeometryStyle(ψ), ψ)
TensorKit.normalize!(ψ::AbstractMPS) = TensorKit.normalize!(GeometryStyle(ψ), ψ)
TensorKit.normalize(ψ::AbstractMPS) = normalize!(copy(ψ))

#===========================================================================================
Fixedpoints
===========================================================================================#

"""
    l_RR(ψ, location)

Left dominant eigenvector of the `AR`-`AR` transfermatrix.
"""
l_RR(ψ::AbstractMPS, loc::Int = 1) = l_RR(GeometryStyle(ψ), ψ, loc)

"""
    l_RL(ψ, location)

Left dominant eigenvector of the `AR`-`AL` transfermatrix.
"""
l_RL(ψ::AbstractMPS, loc::Int = 1) = l_RL(GeometryStyle(ψ), ψ, loc)

"""
    l_LR(ψ, location)

Left dominant eigenvector of the `AL`-`AR` transfermatrix.
"""
l_LR(ψ::AbstractMPS, loc::Int = 1) = l_LR(GeometryStyle(ψ), ψ, loc)

"""
    l_LL(ψ, location)

Left dominant eigenvector of the `AL`-`AL` transfermatrix.
"""
l_LL(ψ::AbstractMPS, loc::Int = 1) = l_LL(GeometryStyle(ψ), ψ, loc)

"""
    r_RR(ψ, location)

Right dominant eigenvector of the `AR`-`AR` transfermatrix.
"""
r_RR(ψ::AbstractMPS, loc::Int = length(ψ)) = r_RR(GeometryStyle(ψ), ψ, loc)

"""
    r_RL(ψ, location)

Right dominant eigenvector of the `AR`-`AL` transfermatrix.
"""
r_RL(ψ::AbstractMPS, loc::Int = length(ψ)) = r_RL(GeometryStyle(ψ), ψ, loc)

"""
    r_LR(ψ, location)

Right dominant eigenvector of the `AL`-`AR` transfermatrix.
"""
r_LR(ψ::AbstractMPS, loc::Int = length(ψ)) = r_LR(GeometryStyle(ψ), ψ, loc)

"""
    r_LL(ψ, location)

Right dominant eigenvector of the `AL`-`AL` transfermatrix.
"""
r_LL(ψ::AbstractMPS, loc::Int = length(ψ)) = r_LL(GeometryStyle(ψ), ψ, loc)
