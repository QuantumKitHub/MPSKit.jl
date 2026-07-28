struct ALView{S, E, N} <: AbstractArray{E, N}
    parent::S
    ALView(parent::S) where {S} = new{S, site_type(S), length(size(parent))}(parent)
end

function Base.getindex(v::ALView{<:FiniteMPS, E}, i::Int)::E where {E}
    ismissing(v.parent.ALs[i]) && v.parent.C[i] # by getting C[i], we are garantueeing that AL[i] exists
    return v.parent.ALs[i]
end

function Base.getindex(v::ALView{<:WindowMPS, E}, i::Int)::E where {E}
    i <= length(v.parent) || throw(ArgumentError("out of bounds"))
    i < 1 && return v.parent.left_gs.AL[i]
    return ALView(v.parent.window)[i]
end

Base.getindex(v::ALView{<:Multiline}, i::Int, j::Int) = v.parent[i].AL[j]
function Base.setindex!(v::ALView{<:Multiline}, vec, i::Int, j::Int)
    return setindex!(v.parent[i].AL, vec, j)
end

struct ARView{S, E, N} <: AbstractArray{E, N}
    parent::S
    ARView(parent::S) where {S} = new{S, site_type(S), length(size(parent))}(parent)
end

function Base.getindex(v::ARView{<:FiniteMPS, E}, i::Int)::E where {E}
    # by getting C[i-1], we are garantueeing that AR[i] exists
    ismissing(v.parent.ARs[i]) && v.parent.C[i - 1]
    return v.parent.ARs[i]
end

function Base.getindex(v::ARView{<:WindowMPS, E}, i::Int)::E where {E}
    i >= 1 || throw(ArgumentError("out of bounds"))
    i > length(v.parent) && return v.parent.right_gs.AR[i]
    return ARView(v.parent.window)[i]
end

Base.getindex(v::ARView{<:Multiline}, i::Int, j::Int) = v.parent[i].AR[j]
function Base.setindex!(v::ARView{<:Multiline}, vec, i::Int, j::Int)
    return setindex!(v.parent[i].AR, vec, j)
end

struct CView{S, E, N} <: AbstractArray{E, N}
    parent::S
    CView(parent::S) where {S} = new{S, bond_type(S), length(size(parent))}(parent)
end

function Base.getindex(v::CView{<:FiniteMPS, E}, i::Int)::E where {E}
    ismissing(v.parent.Cs[i + 1]) || return v.parent.Cs[i + 1]

    if i == 0 || !ismissing(v.parent.ALs[i]) # center is too far right
        center = findfirst(!ismissing, v.parent.ACs)
        if isnothing(center)
            center = findfirst(!ismissing, v.parent.Cs)
            @assert !isnothing(center) "Invalid state"
            center -= 1 # offset in Cs vs C
            @assert !ismissing(v.parent.ALs[center]) "Invalid state"
            v.parent.ACs[center] = _mul_tail(v.parent.ALs[center], v.parent.Cs[center + 1])
        end

        for j in Iterators.reverse((i + 1):center)
            # only factorize what is not cached yet
            if ismissing(v.parent.Cs[j]) || ismissing(v.parent.ARs[j])
                v.parent.Cs[j], v.parent.ARs[j], _ = right_gauge(v.parent.ACs[j])
            end
            if j != i + 1 # last AC not needed
                v.parent.ACs[j - 1] = _mul_tail(v.parent.ALs[j - 1], v.parent.Cs[j])
            end
        end
    else # center is too far left
        center = findlast(!ismissing, v.parent.ACs)
        if isnothing(center)
            center = findlast(!ismissing, v.parent.Cs)
            @assert !isnothing(center) "Invalid state"
            @assert !ismissing(v.parent.ARs[center]) "Invalid state"
            v.parent.ACs[center] = _mul_front(v.parent.Cs[center], v.parent.ARs[center])
        end

        for j in center:i
            # only factorize what is not cached yet
            if ismissing(v.parent.ALs[j]) || ismissing(v.parent.Cs[j + 1])
                v.parent.ALs[j], v.parent.Cs[j + 1], _ = left_gauge(v.parent.ACs[j])
            end
            if j != i # last AC not needed
                v.parent.ACs[j + 1] = _mul_front(v.parent.Cs[j + 1], v.parent.ARs[j + 1])
            end
        end
    end

    return v.parent.Cs[i + 1]
end

function Base.setindex!(v::CView{<:FiniteMPS}, vec, i::Int)
    if ismissing(v.parent.Cs[i + 1])
        if !ismissing(v.parent.ALs[i])
            v.parent.Cs[i + 1], v.parent.ARs[i + 1], _ = right_gauge(v.parent.AC[i + 1])
        else
            v.parent.ALs[i], v.parent.Cs[i + 1], _ = left_gauge(v.parent.AC[i])
        end
    end

    v.parent.Cs .= missing
    v.parent.ACs .= missing
    v.parent.ALs[(i + 1):end] .= missing
    v.parent.ARs[1:i] .= missing

    return setindex!(v.parent.Cs, vec, i + 1)
end

Base.getindex(v::CView{<:WindowMPS}, i::Int) = CView(v.parent.window)[i]
function Base.setindex!(v::CView{<:WindowMPS}, vec, i::Int)
    return setindex!(CView(v.parent.window), vec, i)
end
Base.getindex(v::CView{<:Multiline}, i::Int, j::Int) = v.parent[i].C[j]
function Base.setindex!(v::CView{<:Multiline}, vec, i::Int, j::Int)
    return setindex!(v.parent[i].C, vec, j)
end;

struct ACView{S, E, N} <: AbstractArray{E, N}
    parent::S
    ACView(parent::S) where {S} = new{S, site_type(S), length(size(parent))}(parent)
end

function Base.getindex(v::ACView{<:FiniteMPS, E}, i::Int)::E where {E}
    ismissing(v.parent.ACs[i]) || return v.parent.ACs[i]

    if !ismissing(v.parent.ARs[i]) # center is too far left
        v.parent.ACs[i] = _mul_front(v.parent.C[i - 1], v.parent.ARs[i])
    elseif !ismissing(v.parent.ALs[i])
        v.parent.ACs[i] = _mul_tail(v.parent.ALs[i], v.parent.C[i])
    else
        error("Invalid state")
    end

    return v.parent.ACs[i]
end

function Base.setindex!(v::ACView{<:FiniteMPS}, vec::GenericMPSTensor, i::Int)
    if ismissing(v.parent.ACs[i])
        i < length(v) && v.parent.AR[i + 1]
        i > 1 && v.parent.AL[i - 1]
    end

    v.parent.ACs .= missing
    v.parent.Cs .= missing
    v.parent.ALs[i:end] .= missing
    v.parent.ARs[1:i] .= missing
    return setindex!(v.parent.ACs, vec, i)
end

function Base.setindex!(
        v::ACView{<:FiniteMPS}, vec::Tuple{<:GenericMPSTensor, <:GenericMPSTensor}, i::Int
    )
    if ismissing(v.parent.ACs[i])
        i < length(v) && v.parent.AR[i + 1]
        i > 1 && v.parent.AL[i - 1]
    end

    v.parent.ACs .= missing
    v.parent.Cs .= missing
    v.parent.ALs[i:end] .= missing
    v.parent.ARs[1:i] .= missing

    a, b = vec
    return if isa(a, MPSBondTensor) #c/ar
        if !(scalartype(parent(v)) <: Real) && (scalartype(a) <: Real)
            setindex!(v.parent.Cs, complex(a), i)
        else
            setindex!(v.parent.Cs, a, i)
        end
        setindex!(v.parent.ARs, b, i)
    elseif isa(b, MPSBondTensor) #al/c
        if !(scalartype(parent(v)) <: Real) && (scalartype(b) <: Real)
            setindex!(v.parent.Cs, complex(b), i + 1)
        else
            setindex!(v.parent.Cs, b, i + 1)
        end
        setindex!(v.parent.ALs, a, i)
    else
        throw(ArgumentError("invalid value types"))
    end
end

function Base.getindex(v::ACView{<:WindowMPS, E}, i::Int)::E where {E}
    (i >= 1 && i <= length(v.parent)) || throw(ArgumentError("out of bounds"))
    return ACView(v.parent.window)[i]
end
function Base.setindex!(v::ACView{<:WindowMPS}, vec, i::Int)
    return setindex!(ACView(v.parent.window), vec, i)
end

Base.getindex(v::ACView{<:Multiline}, i::Int, j::Int) = v.parent[i].AC[j]
function Base.setindex!(v::ACView{<:Multiline}, vec, i::Int, j::Int)
    return setindex!(v.parent[i].AC, vec, j)
end

#--- define the rest of the abstractarray interface
Base.size(psi::Union{ACView, ALView, ARView}) = size(psi.parent)

#=
CView is tricky. It starts at 0 for finitemps/WindowMPS, but for multiline Infinitemps objects, it should start at 1.
=#
Base.size(psi::CView{<:AbstractFiniteMPS}) = (length(psi.parent) + 1,)
Base.axes(psi::CView{<:AbstractFiniteMPS}) = map(n -> 0:(n - 1), size(psi))

Base.size(psi::CView{<:Multiline{<:InfiniteMPS}}) = size(psi.parent)
function Base.size(psi::CView{<:Multiline{<:AbstractFiniteMPS}})
    return (length(psi.parent.data), length(first(psi.parent.data)) + 1)
end
function Base.axes(psi::CView{<:Multiline{<:AbstractFiniteMPS}})
    return (Base.OneTo(length(psi.parent.data)), 0:length(first(psi.parent.data)))
end

#the checkbounds for multiline objects needs to be changed, as the first index is periodic
#however if it is a Multiline(Infinitemps), then the second index is also periodic!
function Base.checkbounds(
        ::Type{Bool},
        psi::Union{ACView{<:Multiline}, ALView{<:Multiline}, ARView{<:Multiline}, CView{<:Multiline}},
        a, b
    )
    return if first(psi.parent.data) isa InfiniteMPS
        true
    else
        checkbounds(Bool, CView(first(psi.parent.data)), b)
    end
end

# Gauging routines
# ----------------
@doc """
    left_gauge(AC, [alg]) -> AL, C, ϵ
    right_gauge(AC, [alg]) -> C, AR, ϵ

Factor an updated center MPS tensor `AC` into left- or right-canonical form, `AC ≈ AL * C`
(left, with `AL` left-isometric) or `AC ≈ C * AR` (right, with `AR` right-isometric), without
modifying `AC`. `right_gauge` handles the MPS leg permutation internally, so `AR` is returned in
standard MPS-tensor form.

`alg` selects the factorization and defaults to a (positive) QR/LQ center-move that preserves the virtual bond.
Passing a [`TruncatedAlgorithm`](@extref MatrixAlgebraKit.TruncatedAlgorithm) instead performs a truncated SVD may shrink the bond.

Also returns the truncation error `ϵ`: the 2-norm of the discarded singular values from the
truncated SVD, or `0` for a norm-preserving QR/LQ gauge.
"""
left_gauge
@doc (@doc left_gauge) right_gauge

left_gauge(AC) = left_gauge(AC, Defaults.alg_orth())
function left_gauge(AC, alg)
    AL, C = left_orth(AC; alg)
    return AL, C, zero(real(scalartype(AC)))
end
function left_gauge(AC, alg::MatrixAlgebraKit.TruncatedAlgorithm)
    U, S, Vᴴ, ϵ = svd_trunc(AC, alg)
    C = LinearAlgebra.lmul!(S, Vᴴ) # C = S * Vᴴ, matching `LeftOrthViaSVD`
    return U, C, ϵ
end

right_gauge(AC) = right_gauge(AC, Defaults.alg_orth())
function right_gauge(AC, alg)
    C, AR = right_orth(_transpose_tail(AC); alg)
    return C, _transpose_front(AR), zero(real(scalartype(AC)))
end
function right_gauge(AC, alg::MatrixAlgebraKit.TruncatedAlgorithm)
    U, S, Vᴴ, ϵ = svd_trunc(_transpose_tail(AC), alg)
    C = LinearAlgebra.rmul!(U, S) # C = U * S, matching `RightOrthViaSVD`
    return C, _transpose_front(Vᴴ), ϵ
end

@doc """
    left_gauge!(ψ, pos, AC, [alg]; normalize = false) -> ψ, ϵ
    right_gauge!(ψ, pos, AC, [alg]; normalize = false) -> ψ, ϵ

Gauge an updated center tensor `AC` at site `pos` and install it into `ψ` in one step: factor
`AC` with [`left_gauge`](@ref) / [`right_gauge`](@ref) and write the canonical tensors back,
shifting the gauge center past `pos` (to the right for `left_gauge!`, to the left for
`right_gauge!`). `alg` is forwarded to [`left_gauge`](@ref) / [`right_gauge`](@ref) and hence may
be a [`TruncatedAlgorithm`](@extref MatrixAlgebraKit.TruncatedAlgorithm) to truncate the bond.

By default the factors are installed as-is. Pass `normalize = true` to renormalize the bond
tensor, so `ψ` stays normalized after a local update that changed its norm.

Also returns the truncation error `ϵ` of the factorization: the 2-norm of the discarded singular
values from a truncated SVD gauge, or `0` for a norm-preserving QR gauge.
"""
left_gauge!
@doc (@doc left_gauge!) right_gauge!

function left_gauge!(ψ::AbstractFiniteMPS, pos::Int, AC, alg = Defaults.alg_orth(); normalize::Bool = false)
    AL, C, ϵ = left_gauge(AC, alg)
    normalize && normalize!(C)
    ψ.AC[pos] = (AL, C)
    return ψ, ϵ
end
function right_gauge!(ψ::AbstractFiniteMPS, pos::Int, AC, alg = Defaults.alg_orth(); normalize::Bool = false)
    C, AR, ϵ = right_gauge(AC, alg)
    normalize && normalize!(C)
    ψ.AC[pos] = (C, AR)
    return ψ, ϵ
end

@doc """
    gauge!(ψ, pos, direction, AC, [alg]; normalize = false) -> ψ, ϵ

Direction-dispatching wrapper around [`left_gauge!`](@ref) / [`right_gauge!`](@ref): gauge an
updated center tensor `AC` at site `pos` and install it into `ψ`, shifting the gauge center past
`pos` to the right for `direction = Val(:right)` and to the left for `Val(:left)`. `alg` and
`normalize` are forwarded unchanged, and the truncation error `ϵ` is returned as-is.
"""
function gauge!(ψ::AbstractFiniteMPS, pos::Int, ::Val{Dir}, AC, alg = Defaults.alg_orth(); kwargs...) where {Dir}
    Dir === :right && return left_gauge!(ψ, pos, AC, alg; kwargs...)
    Dir === :left && return right_gauge!(ψ, pos, AC, alg; kwargs...)
    return throw(ArgumentError(lazy"invalid direction `$Dir`"))
end

@doc """
    gauge2!(ψ, pos, direction, AC2, alg; normalize = false) -> ψ, ϵ

Two-site analogue of [`gauge!`](@ref): factor an updated two-site center tensor `AC2` spanning
sites `pos` and `pos+1` with the truncated SVD `alg` and install the resulting canonical tensors
into `ψ` in one step, shifting the gauge center past the bond.
(To the right for `direction = Val(:right)`, to the left for `Val(:left)`).

Returns the truncation error `ϵ`, the 2-norm of the discarded singular values. Pass
`normalize = true` to renormalize the bond tensor, so `ψ` stays normalized after a local update
that changed its norm.
"""
function gauge2!(
        ψ::AbstractFiniteMPS, pos::Int, ::Val{Dir}, AC2, alg;
        normalize::Bool = false
    ) where {Dir}
    al, c, ar, ϵ = svd_trunc!(AC2, alg)
    normalize && normalize!(c)
    # the SVD center `c` (singular values) is real-valued; promote it to the state's scalartype so
    # the installed tensors stay consistent, without needlessly forcing a real-valued state complex.
    C = scalartype(ψ) <: Complex ? complex(c) : c
    if Dir === :right
        ψ.AC[pos] = (al, C)
        ψ.AC[pos + 1] = (C, _transpose_front(ar))
    elseif Dir === :left
        ψ.AC[pos + 1] = (C, _transpose_front(ar))
        ψ.AC[pos] = (al, C)
    else
        throw(ArgumentError(lazy"invalid direction `$Dir`"))
    end
    return ψ, ϵ
end
