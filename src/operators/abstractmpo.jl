# Matrix Product Operators
# ========================
"""
    abstract type AbstractMPO{O<:MPOTensor} <: AbstractVector{O} end

Abstract supertype for Matrix Product Operators (MPOs).
"""
abstract type AbstractMPO{O<:MPOTensor} <: AbstractVector{O} end

# Hamiltonian Matrix Product Operators
# ====================================
"""
    abstract type AbstractHMPO{O<:MPOTensor} <: AbstractMPO{O}

Abstract supertype for Hamiltonian MPOs.
"""
abstract type AbstractHMPO{O<:MPOTensor} <: AbstractMPO{O} end

function HMPO(lattice::AbstractVector{S}, terms...) where {S<:VectorSpace}
    if lattice isa PeriodicArray
        return InfiniteHamiltonianMPO(lattice, terms...)
    else
        return FiniteHamiltonianMPO(lattice, terms...)
    end
end
function HMPO(operator::AbstractTensorMap{E,S,N,N}; L=Inf) where {E,S,N}
    @assert domain(operator) == codomain(operator) "Not a valid Hamiltonian operator."
    @assert allequal(collect(domain(operator))) "The operator must have the same local spaces."

    if isfinite(L)
        lattice = repeat([space(operator, 1)], L)
        return HMPO(lattice, ntuple(x -> x + i - 1, N) => operator for i in 1:(L - (N - 1)))
    else
        lattice = PeriodicArray([space(operator, 1)])
        return HMPO(lattice, ntuple(identity, N) => operator)
    end
end

# useful union types
const SparseMPO{O<:SparseBlockTensorMap} = AbstractMPO{O}

# By default, define things in terms of parent
Base.size(mpo::AbstractMPO, args...) = size(parent(mpo), args...)
Base.length(mpo::AbstractMPO) = length(parent(mpo))

@inline Base.getindex(mpo::AbstractMPO, args...) = getindex(parent(mpo), args...)
@inline function Base.setindex!(mpo::AbstractMPO, value::MPOTensor, i::Int)
    setindex!(parent(mpo), value, i)
    return mpo
end

# Properties
# ----------
left_virtualspace(mpo::AbstractMPO, site::Int) = left_virtualspace(mpo[site])
right_virtualspace(mpo::AbstractMPO, site::Int) = right_virtualspace(mpo[site])
physicalspace(mpo::AbstractMPO, site::Int) = physicalspace(mpo[site])
physicalspace(mpo::AbstractMPO) = map(physicalspace, mpo)

for ftype in (:spacetype, :sectortype, :storagetype)
    @eval TensorKit.$ftype(mpo::AbstractMPO) = $ftype(typeof(mpo))
    @eval TensorKit.$ftype(::Type{MPO}) where {MPO<:AbstractMPO} = $ftype(eltype(MPO))
end

# Utility functions
# -----------------
function jordanmpotensortype(::Type{S}, ::Type{T}) where {S<:VectorSpace,T<:Number}
    TT = Base.promote_typejoin(tensormaptype(S, 2, 2, T), BraidingTensor{T,S})
    return SparseBlockTensorMap{TT}
end

# Show
# ----
function Base.show(io::IO, ::MIME"text/plain", W::AbstractMPO)
    L = length(W)
    println(io, L == 1 ? "single site " : "$L-site ", typeof(W), ":")
    context = IOContext(io, :typeinfo => eltype(W), :compact => true)
    return show(context, W)
end

Base.show(io::IO, mpo::AbstractMPO) = show(convert(IOContext, io), mpo)
function Base.show(io::IOContext, mpo::AbstractMPO)
    charset = (; top="┬", bot="┴", mid="┼", ver="│", dash="──")
    limit = get(io, :limit, false)::Bool
    half_screen_rows = limit ? div(displaysize(io)[1] - 8, 2) : typemax(Int)
    L = length(mpo)

    # used to align all mposite infos regardless of the length of the mpo (100 takes up more space than 5)
    npad = floor(Int, log10(L))
    mpoletter = mpo isa MPOHamiltonian ? "W" : "O"
    isfinite = (mpo isa FiniteMPO) || (mpo isa FiniteMPOHamiltonian)

    !isfinite && println(io, "╷  ⋮")
    for site in reverse(1:L)
        if site < half_screen_rows || site > L - half_screen_rows
            if site == L && isfinite
                println(io, charset.top, " $mpoletter[$site]: ",
                        repeat(" ", npad - floor(Int, log10(site))), mpo[site])
            elseif (site == 1) && isfinite
                println(io, charset.bot, " $mpoletter[$site]: ",
                        repeat(" ", npad - floor(Int, log10(site))), mpo[site])
            else
                println(io, charset.mid, " $mpoletter[$site]: ",
                        repeat(" ", npad - floor(Int, log10(site))), mpo[site])
            end
        elseif site == half_screen_rows
            println(io, "   ", "⋮")
        end
    end
    !isfinite && println(io, "╵  ⋮")
    return nothing
end

function braille(H::SparseMPO)
    isfinite = (H isa FiniteMPO) || (H isa FiniteMPOHamiltonian)
    dash = "🭻"
    stride = 2 #amount of dashes between braille
    L = length(H)

    brailles = Vector{Vector{String}}(undef, L)
    buffer = IOBuffer()
    for (i, W) in enumerate(H)
        BlockTensorKit.show_braille(buffer, W)
        brailles[i] = split(String(take!(buffer)))
    end

    maxheight = maximum(length.(brailles))

    for i in 1:maxheight
        line = ""
        line *= ((i == 1 && !isfinite) ? ("... " * dash) : " ")
        line *= (i > 1 && !isfinite) ? "    " : ""
        for (j, braille) in enumerate(brailles)
            line *= (checkbounds(Bool, braille, i) ? braille[i] :
                     repeat(" ", length(braille[1])))
            if j < L
                line *= repeat(((i == 1) ? dash : " "), stride)
            end
        end
        line *= ((i == 1 && !isfinite) ? (dash * " ...") : " ")
        println(line)
    end
    return nothing
end

# Linear Algebra
# --------------
Base.:+(mpo::AbstractMPO) = scale(mpo, One())
Base.:-(mpo::AbstractMPO) = scale(mpo, -1)
Base.:-(mpo1::AbstractMPO, mpo2::AbstractMPO) = mpo1 + (-mpo2)

Base.:*(α::Number, mpo::AbstractMPO) = scale(mpo, α)
Base.:*(mpo::AbstractMPO, α::Number) = scale(mpo, α)
Base.:/(mpo::AbstractMPO, α::Number) = scale(mpo, inv(α))
Base.:\(α::Number, mpo::AbstractMPO) = scale(mpo, inv(α))

VectorInterface.scale(mpo::AbstractMPO, α::Number) = scale!(copy(mpo), α)

LinearAlgebra.norm(mpo::AbstractMPO) = sqrt(abs(dot(mpo, mpo)))

function Base.:(^)(a::AbstractMPO, n::Int)
    n >= 1 || throw(DomainError(n, "n should be a positive integer"))
    return Base.power_by_squaring(a, n)
end

Base.conj(mpo::AbstractMPO) = conj!(copy(mpo))
function Base.conj!(mpo::AbstractMPO)
    for i in 1:length(mpo)
        mpo[i] = _conj_mpo(mpo[i])
    end
    return mpo
end

function _conj_mpo(O::MPOTensor)
    return @plansor O′[-1 -2; -3 -4] := conj(O[-1 -3; -2 -4])
end
