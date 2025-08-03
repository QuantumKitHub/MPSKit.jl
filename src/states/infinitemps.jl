"""
    InfiniteMPS{A<:GenericMPSTensor,B<:MPSBondTensor} <: AbtractMPS

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
    InfiniteMPS([f, eltype], physicalspaces::Vector{<:Union{S, CompositeSpace{S}},
                virtualspaces::Vector{<:Union{S, CompositeSpace{S}};
                kwargs...) where {S<:ElementarySpace}
    InfiniteMPS(As::AbstractVector{<:GenericMPSTensor}; kwargs...)
    InfiniteMPS(ALs::AbstractVector{<:GenericMPSTensor}, C₀::MPSBondTensor;
                kwargs...)

Construct an MPS via a specification of physical and virtual spaces, or from a list of
tensors `As`, or a list of left-gauged tensors `ALs`.

### Arguments
- `As::AbstractVector{<:GenericMPSTensor}`: vector of site tensors
- `ALs::AbstractVector{<:GenericMPSTensor}`: vector of left-gauged site tensors
- `C₀::MPSBondTensor`: initial gauge tensor

- `f::Function=rand`: initializer function for tensor data
- `eltype::Type{<:Number}=ComplexF64`: scalar type of tensors

- `physicalspaces::AbstractVector{<:Union{S, CompositeSpace{S}}`: list of physical spaces
- `virtualspaces::AbstractVector{<:Union{S, CompositeSpace{S}}`: list of virtual spaces

### Keywords
- `tol`: gauge fixing tolerance
- `maxiter`: gauge fixing maximum iterations
"""
struct InfiniteMPS{A <: GenericMPSTensor, B <: MPSBondTensor} <: AbstractMPS
    AL::PeriodicVector{A}
    AR::PeriodicVector{A}
    C::PeriodicVector{B}
    AC::PeriodicVector{A}
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
    function InfiniteMPS(
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

        for i in 1:L
            N = numind(AL[i])
            N == numind(AR[i]) == numind(AC[i]) ||
                throw(SpaceMismatch("incompatible spaces at site $i"))

            # verify that the physical spaces are compatible
            phys_ind = 2:(N - 1)
            all(
                space.(Ref(AL[i]), phys_ind) .== space.(Ref(AR[i]), phys_ind) .==
                    space.(Ref(AC[i]), phys_ind)
            ) ||
                throw(SpaceMismatch("incompatible physical spaces at site $i"))

            # verify that the virtual spaces are compatible
            space(AL[i], 1) == dual(space(AL[i - 1], N)) &&
                space(AR[i], 1) == dual(space(AR[i - 1], N)) &&
                space(AC[i], 1) == space(AL[i], 1) &&
                space(AC[i], N) == space(AR[i], N) &&
                space(C[i], 1) == dual(space(AL[i], N)) &&
                space(AR[i], 1) == dual(space(C[i - 1], 2)) ||
                throw(SpaceMismatch("incompatible virtual spaces at site $i"))
            # verify that the spaces are non-zero
            dim(space(AL[i])) > 0 && dim(space(C[i])) > 0 ||
                @warn "no fusion channels available at site $i"
        end
        return new{A, B}(AL, AR, C, AC)
    end
end

#===========================================================================================
Constructors
===========================================================================================#

function InfiniteMPS(
        AL::AbstractVector{A}, AR::AbstractVector{A}, C::AbstractVector{B},
        AC::AbstractVector{A} = AL .* C
    ) where {A <: GenericMPSTensor, B <: MPSBondTensor}
    return InfiniteMPS(
        convert(PeriodicVector{A}, AL), convert(PeriodicVector{A}, AR),
        convert(PeriodicVector{B}, C), convert(PeriodicVector{A}, AC)
    )
end

function InfiniteMPS(
        pspaces::AbstractVector{S}, Dspaces::AbstractVector{S};
        kwargs...
    ) where {S <: IndexSpace}
    return InfiniteMPS(MPSTensor.(pspaces, circshift(Dspaces, 1), Dspaces); kwargs...)
end
function InfiniteMPS(
        f, elt::Type{<:Number}, pspaces::AbstractVector{S}, Dspaces::AbstractVector{S};
        kwargs...
    ) where {S <: IndexSpace}
    return InfiniteMPS(
        MPSTensor.(f, elt, pspaces, circshift(Dspaces, 1), Dspaces);
        kwargs...
    )
end
InfiniteMPS(d::S, D::S) where {S <: Union{Int, <:IndexSpace}} = InfiniteMPS([d], [D])
function InfiniteMPS(
        f, elt::Type{<:Number}, d::S, D::S
    ) where {S <: Union{Int, <:IndexSpace}}
    return InfiniteMPS(f, elt, [d], [D])
end
function InfiniteMPS(ds::AbstractVector{Int}, Ds::AbstractVector{Int})
    return InfiniteMPS(ComplexSpace.(ds), ComplexSpace.(Ds))
end
function InfiniteMPS(
        f, elt::Type{<:Number}, ds::AbstractVector{Int}, Ds::AbstractVector{Int}, kwargs...
    )
    return InfiniteMPS(f, elt, ComplexSpace.(ds), ComplexSpace.(Ds); kwargs...)
end

function InfiniteMPS(A::AbstractVector{<:GenericMPSTensor}; kwargs...)
    # check spaces
    leftvspaces = circshift(_firstspace.(A), -1)
    rightvspaces = conj.(_lastspace.(A))
    isnothing(findfirst(leftvspaces .!= rightvspaces)) ||
        throw(SpaceMismatch("incompatible virtual spaces $leftvspaces and $rightvspaces"))

    # check rank
    A_copy = PeriodicArray(copy.(A)) # copy to avoid side effects
    all(isfullrank, A_copy) ||
        @warn "Constructing an MPS from tensors that are not full rank"
    makefullrank!(A_copy)

    AR = A_copy

    leftvspaces = circshift(_firstspace.(AR), -1)
    rightvspaces = conj.(_lastspace.(AR))
    isnothing(findfirst(leftvspaces .!= rightvspaces)) ||
        throw(SpaceMismatch("incompatible virtual spaces $leftvspaces and $rightvspaces"))

    # initial guess for the gauge tensor
    V = _firstspace(A_copy[1])
    C₀ = isomorphism(storagetype(eltype(A_copy)), V, V)

    # initialize tensor storage
    AL = similar.(AR)
    AC = similar.(AR)
    C = similar(AR, typeof(C₀))
    ψ = InfiniteMPS{eltype(AL), eltype(C)}(AL, AR, C, AC)

    # gaugefix the MPS
    gaugefix!(ψ, A_copy, C₀; kwargs...)
    mul!.(ψ.AC, ψ.AL, ψ.C)

    return ψ
end

function InfiniteMPS(AL::AbstractVector{<:GenericMPSTensor}, C₀::MPSBondTensor; kwargs...)
    AL = PeriodicArray(copy.(AL))

    all(isfullrank, AL) ||
        @warn "Constructing an MPS from tensors that are not full rank"

    # initialize tensor storage
    AC = similar.(AL)
    AR = similar.(AL)
    C = similar(AR, typeof(C₀))
    ψ = InfiniteMPS{eltype(AL), eltype(C)}(AL, AR, C, AC)

    # gaugefix the MPS
    gaugefix!(ψ, AL, C₀; order = :R, kwargs...)
    mul!.(ψ.AC, ψ.AL, ψ.C)

    return ψ
end

function InfiniteMPS(C₀::MPSBondTensor, AR::AbstractVector{<:GenericMPSTensor}; kwargs...)
    AR = PeriodicArray(copy.(AR))

    # initialize tensor storage
    AC = similar.(AR)
    AL = similar.(AR)
    C = similar(AR, typeof(C₀))
    ψ = InfiniteMPS{eltype(AL), eltype(C)}(AL, AR, C, AC)

    # gaugefix the MPS
    gaugefix!(ψ, AR, C₀; order = :L, kwargs...)
    mul!.(ψ.AC, ψ.AL, ψ.C)

    return ψ
end

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
Base.copy(ψ::InfiniteMPS) = InfiniteMPS(copy(ψ.AL), copy(ψ.AR), copy(ψ.C), copy(ψ.AC))
function Base.copy!(ψ::InfiniteMPS, ϕ::InfiniteMPS)
    copy!.(ψ.AL, ϕ.AL)
    copy!.(ψ.AR, ϕ.AR)
    copy!.(ψ.AC, ϕ.AC)
    copy!.(ψ.C, ϕ.C)
    return ψ
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

function Base.show(io::IO, ::MIME"text/plain", ψ::InfiniteMPS)
    L = length(ψ)
    println(io, L == 1 ? "single site" : "$L-site", " InfiniteMPS:")
    context = IOContext(io, :typeinfo => eltype(ψ), :compact => true)
    return show(context, ψ)
end
Base.show(io::IO, ψ::InfiniteMPS) = show(convert(IOContext, io), ψ)
function Base.show(io::IOContext, ψ::InfiniteMPS)
    charset = (; mid = "├", ver = "│", dash = "──")
    limit = get(io, :limit, false)::Bool
    half_screen_rows = limit ? div(displaysize(io)[1] - 8, 2) : typemax(Int)
    if !haskey(io, :compact)
        io = IOContext(io, :compact => true)
    end
    L = length(ψ)
    println(io, charset.ver, "   ⋮")
    for site in reverse(1:L)
        if site < half_screen_rows || site > L - half_screen_rows
            if site == L
                println(io, charset.ver, " C[$site]: ", ψ.C[site])
            end
            println(io, charset.mid, charset.dash, " AL[$site]: ", ψ.AL[site])
        elseif site == half_screen_rows
            println(io, charset.ver, "⋮")
        end
    end
    println(io, charset.ver, "   ⋮")
    return nothing
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
