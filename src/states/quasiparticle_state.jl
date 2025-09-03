#=
Should not be constructed by the user - acts like a vector (used in eigsolve)
I think it makes sense to see these things as an actual state instead of return an array of B tensors (what we used to do)
This will allow us to plot energy density (finite qp) and measure observeables.
=#

struct LeftGaugedQP{S, T1, T2, E <: Number}
    # !(left_gs === right_gs) => domain wall excitation
    left_gs::S
    right_gs::S

    VLs::Vector{T1} # AL' VL = 0 (and VL*X = B)
    Xs::Vector{T2} # contains variational parameters

    momentum::E
end

struct RightGaugedQP{S, T1, T2, E <: Number}
    # !(left_gs === right_gs) => domain wall excitation
    left_gs::S
    right_gs::S

    Xs::Vector{T2}
    VRs::Vector{T1}

    momentum::E
end

function leftgaugedqptype(::Type{S}, ::Type{E}) where {S, E <: Number}
    T1 = eltype(S)
    T2 = tensormaptype(spacetype(T1), 1, 2, storagetype(T1))
    return LeftGaugedQP{S, T1, T2, E}
end

#constructors
function LeftGaugedQP(
        datfun, left_gs, right_gs = left_gs;
        sector = one(sectortype(left_gs)), momentum = 0.0
    )
    # find the left null spaces for the TNS
    excitation_space = Vect[typeof(sector)](sector => 1)
    VLs = convert(Vector, map(leftnull, left_gs.AL))
    Xs = map(enumerate(VLs)) do (loc, vl)
        x = similar(
            vl,
            right_virtualspace(vl) ← excitation_space ⊗ right_virtualspace(right_gs, loc)
        )
        fill_data!(x, datfun)
        return x
    end
    sum(dim, Xs) == 0 && @warn "LeftGaugedQP: No possible fusion channels"
    left_gs isa InfiniteMPS ||
        momentum == zero(momentum) ||
        @warn "momentum is ignored for finite quasiparticles"
    return LeftGaugedQP(left_gs, right_gs, VLs, Xs, momentum)
end
function LeftGaugedQP(
        datfun, left_gs::MultilineMPS, right_gs::MultilineMPS = left_gs;
        sector = one(sectortype(left_gs)), momentum = 0.0
    )
    # not sure why this is needed for type stability
    Tresult = leftgaugedqptype(eltype(parent(left_gs)), typeof(momentum))
    qp_rows = Vector{Tresult}(undef, size(left_gs, 1))
    for row in eachindex(qp_rows)
        qp_rows[row] = LeftGaugedQP(datfun, left_gs[row], right_gs[row]; sector, momentum)
    end
    return Multiline(qp_rows)
end

function RightGaugedQP(
        datfun, left_gs, right_gs = left_gs;
        sector = one(sectortype(left_gs)), momentum = 0.0
    )
    # find the left null spaces for the TNS
    excitation_space = Vect[typeof(sector)](sector => 1)
    VRs = convert(Vector, map(rightnull! ∘ _transpose_tail, right_gs.AR))
    Xs = map(enumerate(VRs)) do (i, vr)
        x = similar(
            vr,
            left_virtualspace(left_gs, i)' ← excitation_space ⊗ _firstspace(vr)
        )
        return fill_data!(x, datfun)
    end
    left_gs isa InfiniteMPS ||
        momentum == zero(momentum) ||
        @warn "momentum is ignored for finite quasiparticles"
    return RightGaugedQP(left_gs, right_gs, Xs, VRs, momentum)
end

#gauge dependent code
function Base.similar(v::LeftGaugedQP, ::Type{T} = scalartype(v)) where {T <: Number}
    return LeftGaugedQP(v.left_gs, v.right_gs, v.VLs, similar.(v.Xs, T), v.momentum)
end
function Base.similar(v::RightGaugedQP, ::Type{T} = scalartype(v)) where {T <: Number}
    return RightGaugedQP(v.left_gs, v.right_gs, similar.(v.Xs, T), v.VRs, v.momentum)
end

Base.getindex(v::LeftGaugedQP, i::Int) = v.VLs[mod1(i, end)] * v.Xs[mod1(i, end)]
function Base.getindex(v::RightGaugedQP, i::Int)
    return @plansor t[-1 -2; -3 -4] := v.Xs[mod1(i, end)][-1; -3 1] * v.VRs[mod1(i, end)][1; -4 -2]
end

function Base.setindex!(v::LeftGaugedQP, B, i::Int)
    v.Xs[mod1(i, end)] = v.VLs[mod1(i, end)]' * B
    return v
end
function Base.setindex!(v::RightGaugedQP, B, i::Int)
    @plansor v.Xs[mod1(i, end)][-1; -2 -3] := B[-1 1; -2 2] * conj(v.VRs[mod1(i, end)][-3; 2 1])
    return v
end

#conversion between gauges (partially implemented)
function Base.convert(
        ::Type{RightGaugedQP}, input::LeftGaugedQP{S}
    ) where {S <: InfiniteMPS}
    rg = RightGaugedQP(
        zero, input.left_gs, input.right_gs;
        sector = first(sectors(auxiliaryspace(input))), momentum = input.momentum
    )
    len = length(input)

    #construct environments
    rBs = [
        @plansor t[-1; -2 -3] := input[len][-1 2; -2 3] *
            conj(input.right_gs.AR[len][-3 2; 3]) * exp(1im * input.momentum)
    ]
    for i in (len - 1):-1:1
        t = TransferMatrix(input.left_gs.AL[i], input.right_gs.AR[i]) * rBs[end]
        @plansor t[-1; -2 -3] += input[i][-1 2; -2 3] * conj(input.right_gs.AR[i][-3 2; 3])
        push!(rBs, exp(1im * input.momentum) * t)
    end
    rBs = reverse(rBs)

    tm = TransferMatrix(input.left_gs.AL, input.right_gs.AR)

    if istrivial(input)
        tm = regularize(tm, l_LR(input.right_gs), r_LR(input.right_gs))
    end

    rBE, convhist = linsolve(
        tm, rBs[1], rBs[1], GMRES(), 1, -exp(1im * input.momentum * len)
    )
    convhist.converged == 0 && @warn "failed to converge: normres = $(convhist.normres)"

    rBs[1] = rBE
    for i in len:-1:2
        rBE = TransferMatrix(input.left_gs.AL[i], input.right_gs.AR[i]) * rBE *
            exp(1im * input.momentum)
        rBs[i] += rBE
    end

    #final contraction is now easy
    for i in 1:len
        @plansor T[-1 -2; -3 -4] := input.left_gs.AL[i][-1 -2; 1] *
            rBs[mod1(i + 1, end)][1; -3 -4]
        @plansor T[-1 -2; -3 -4] += input[i][-1 -2; -3 -4]
        rg[i] = T
    end

    return rg
end
function Base.convert(
        ::Type{LeftGaugedQP}, input::RightGaugedQP{S}
    ) where {S <: InfiniteMPS}
    lg = LeftGaugedQP(
        zero, input.left_gs, input.right_gs;
        sector = first(sectors(auxiliaryspace(input))), momentum = input.momentum
    )
    len = length(input)

    lBs = [
        @plansor t[-1; -2 -3] := input[1][1 2; -2 -3] *
            conj(input.left_gs.AL[1][1 2; -1])
    ] ./
        exp(1im * input.momentum)
    for i in 2:len
        t = lBs[end] * TransferMatrix(input.right_gs.AR[i], input.left_gs.AL[i])
        @plansor t[-1; -2 -3] += input[i][1 2; -2 -3] * conj(input.left_gs.AL[i][1 2; -1])
        push!(lBs, t / exp(1im * input.momentum))
    end

    tm = TransferMatrix(input.right_gs.AR, input.left_gs.AL)
    if istrivial(input)
        tm = regularize(tm, l_RL(input.right_gs), r_RL(input.right_gs))
    end

    lBE, convhist = linsolve(
        flip(tm), lBs[end], lBs[end], GMRES(), 1, -1 / exp(1im * input.momentum * len)
    )
    convhist.converged == 0 && @warn "failed to converge: normres = $(convhist.normres)"

    lBs[end] = lBE
    for i in 1:(len - 1)
        lBE = lBE * TransferMatrix(input.right_gs.AR[i], input.left_gs.AL[i]) /
            exp(1im * input.momentum)
        lBs[i] += lBE
    end

    for i in 1:len
        @plansor T[-1 -2; -3 -4] := lBs[mod1(i - 1, len)][-1; -3 1] *
            input.right_gs.AR[i][1 -2; -4]
        @plansor T[-1 -2; -3 -4] += input[i][-1 -2; -3 -4]
        lg[i] = T
    end

    return lg
end

# gauge independent code
const QP{S, T1, T2} = Union{LeftGaugedQP{S, T1, T2}, RightGaugedQP{S, T1, T2}}
const FiniteQP{S <: FiniteMPS, T1, T2} = QP{S, T1, T2}
const InfiniteQP{S <: InfiniteMPS, T1, T2} = QP{S, T1, T2}
const MultilineQP{Q <: QP} = Multiline{Q}

TensorKit.spacetype(::Union{QP{S}, Type{<:QP{S}}}) where {S} = spacetype(S)
TensorKit.sectortype(::Union{QP{S}, Type{<:QP{S}}}) where {S} = sectortype(S)

physicalspace(state::QP, i::Int) = physicalspace(state.left_gs, i)
physicalspace(state::QP) = physicalspace(state.left_gs)
left_virtualspace(state::QP, i::Int) = left_virtualspace(state.left_gs, i)
left_virtualspace(state::QP) = map(Base.Fix1(left_virtualspace, state), eachsite(state))
right_virtualspace(state::QP, i::Int) = right_virtualspace(state.right_gs, i)
right_virtualspace(state::QP) = map(Base.Fix1(right_virtualspace, state), eachsite(state))
auxiliaryspace(state::QP) = space(state.Xs[1], 2)
auxiliarysector(state::QP) = only(sectors(auxiliaryspace(state)))
eachsite(state::QP) = eachsite(state.left_gs)

istopological(qp::QP) = qp.left_gs !== qp.right_gs
istrivial(qp::QP) = !istopological(qp) && isone(auxiliarysector(qp))

Base.copy(a::QP) = copy!(similar(a), a)
Base.copyto!(a::QP, b::QP) = copy!(a, b)
function Base.copy!(a::T, b::T) where {T <: QP}
    for (i, j) in zip(a.Xs, b.Xs)
        copy!(i, j)
    end
    return a
end
Base.@constprop :aggressive function Base.getproperty(qp::QP, s::Symbol)
    if s == :trivial
        Base.depwarn("`qp.trivial` is deprecated in favor of `istrivial` and `istopological`", :trivial)
        return !istopological(qp)
    else
        return getfield(qp, s)
    end
end

function Base.:-(v::T, w::T) where {T <: QP}
    t = similar(v)
    t.Xs[:] = (v.Xs - w.Xs)[:]
    return t
end
function Base.:+(v::T, w::T) where {T <: QP}
    t = similar(v)
    t.Xs[:] = (v.Xs + w.Xs)[:]
    return t
end

LinearAlgebra.dot(v::T, w::T) where {T <: QP} = sum(dot.(v.Xs, w.Xs))
LinearAlgebra.norm(v::QP) = norm(norm.(v.Xs))
LinearAlgebra.normalize!(w::QP) = rmul!(w, 1 / norm(w))
Base.length(v::QP) = length(v.Xs)
Base.eltype(::Type{<:QP{<:Any, <:Any, T}}) where {T} = T

function LinearAlgebra.mul!(w::T, a::Number, v::T) where {T <: QP}
    @inbounds for (i, j) in zip(w.Xs, v.Xs)
        LinearAlgebra.mul!(i, a, j)
    end
    return w
end

function LinearAlgebra.mul!(w::T, v::T, a::Number) where {T <: QP}
    @inbounds for (i, j) in zip(w.Xs, v.Xs)
        LinearAlgebra.mul!(i, j, a)
    end
    return w
end
function LinearAlgebra.rmul!(v::QP, a::Number)
    for x in v.Xs
        LinearAlgebra.rmul!(x, a)
    end
    return v
end

function LinearAlgebra.axpy!(a::Number, v::T, w::T) where {T <: QP}
    @inbounds for (i, j) in zip(w.Xs, v.Xs)
        LinearAlgebra.axpy!(a, j, i)
    end
    return w
end
function LinearAlgebra.axpby!(a::Number, v::T, b::Number, w::T) where {T <: QP}
    @inbounds for (i, j) in zip(w.Xs, v.Xs)
        LinearAlgebra.axpby!(a, j, b, i)
    end
    return w
end

Base.:*(v::QP, a::Number) = mul!(similar(v), a, v)
Base.:*(a::Number, v::QP) = mul!(similar(v), a, v)

Base.zero(v::QP) = v * 0;

function Base.convert(::Type{<:FiniteMPS}, v::QP{S}) where {S <: FiniteMPS}
    #very slow and clunky, but shouldn't be performance critical anyway

    elt = scalartype(v)

    utl = auxiliaryspace(v)
    ou = oneunit(utl)
    utsp = ou ⊕ ou
    upper = isometry(storagetype(site_type(v.left_gs)), utsp, ou)
    lower = leftnull(upper)
    upper_I = upper * upper'
    lower_I = lower * lower'
    uplow_I = upper * lower'

    Ls = v.left_gs.AL[1:end]
    Rs = v.right_gs.AR[1:end]

    #step 0 : fuse the utility leg of B with the first leg of B
    orig_Bs = map(i -> v[i], 1:length(v))
    Bs = map(orig_Bs) do t
        frontmap = isomorphism(
            storagetype(t), fuse(utl * _firstspace(t)), utl * _firstspace(t)
        )
        @plansor tt[-1 -2; -3] := t[1 -2; 2 -3] * frontmap[-1; 2 1]
    end

    function simplefuse(temp)
        frontmap = isomorphism(
            storagetype(temp), fuse(space(temp, 1) * space(temp, 2)),
            space(temp, 1) * space(temp, 2)
        )
        backmap = isomorphism(
            storagetype(temp), space(temp, 5)' * space(temp, 4)',
            fuse(space(temp, 5)' * space(temp, 4)')
        )

        return @plansor tempp[-1 -2; -3] := frontmap[-1; 1 2] * temp[1 2 -2 3; 4] *
            backmap[4 3; -3]
    end

    #step 1 : pass utl through Ls
    passer = isomorphism(storagetype(eltype(Ls)), utl, utl)
    for (i, L) in enumerate(Ls)
        @plansor temp[-1 -2 -3 -4; -5] := L[-2 -3; -4] * passer[-1; -5]
        Ls[i] = simplefuse(temp)
    end

    #step 2 : embed all Ls/Bs/Rs in the same space
    superspaces = map(zip(Ls, Rs)) do (L, R)
        return supremum(space(L, 1), space(R, 1))
    end
    push!(superspaces, supremum(_lastspace(Ls[end])', _lastspace(Rs[end])'))

    for i in 1:(length(v) + 1)
        Lf = isometry(
            storagetype(Ls[i <= length(v) ? i : i - 1]), superspaces[i],
            i <= length(v) ? _firstspace(Ls[i]) : _lastspace(Ls[i - 1])'
        )
        Rf = isometry(
            storagetype(Rs[i <= length(v) ? i : i - 1]), superspaces[i],
            i <= length(v) ? _firstspace(Rs[i]) : _lastspace(Rs[i - 1])'
        )

        if i <= length(v)
            @plansor Ls[i][-1 -2; -3] := Lf[-1; 1] * Ls[i][1 -2; -3]
            @plansor Rs[i][-1 -2; -3] := Rf[-1; 1] * Rs[i][1 -2; -3]
            @plansor Bs[i][-1 -2; -3] := Lf[-1; 1] * Bs[i][1 -2; -3]
        end

        if i > 1
            @plansor Ls[i - 1][-1 -2; -3] := Ls[i - 1][-1 -2; 1] * conj(Lf[-3; 1])
            @plansor Rs[i - 1][-1 -2; -3] := Rs[i - 1][-1 -2; 1] * conj(Rf[-3; 1])
            @plansor Bs[i - 1][-1 -2; -3] := Bs[i - 1][-1 -2; 1] * conj(Rf[-3; 1])
        end
    end

    #step 3 : fuse the correct *_I with the correct tensor (and enforce boundary conditions)
    function doboundary(temp1, pos)
        if pos == 1
            @plansor temp2[-1 -2 -3 -4; -5] := temp1[1 -2 -3 -4; -5] * conj(upper[1; -1])
        elseif pos == length(v)
            @plansor temp2[-1 -2 -3 -4; -5] := temp1[-1 -2 -3 -4; 1] * lower[1; -5]
        else
            temp2 = temp1
        end

        return temp2
    end

    for i in 1:length(v)
        @plansor temp[-1 -2 -3 -4; -5] := Ls[i][-2 -3; -4] * upper_I[-1; -5]
        temp = doboundary(temp, i)
        Ls[i] = simplefuse(temp) * (i < length(v))

        @plansor temp[-1 -2 -3 -4; -5] := Rs[i][-2 -3; -4] * lower_I[-1; -5]
        temp = doboundary(temp, i)
        Rs[i] = simplefuse(temp) * (i > 1)

        @plansor temp[-1 -2 -3 -4; -5] := Bs[i][-2 -3; -4] * uplow_I[-1; -5]
        temp = doboundary(temp, i)
        Bs[i] = simplefuse(temp)
    end

    return FiniteMPS(Ls + Rs + Bs; normalize = false)
end

Base.@constprop :aggressive function Base.getproperty(exci::MultilineQP, s::Symbol)
    if s == :momentum
        return first(exci.data).momentum
    elseif s == :left_gs
        Multiline(map(x -> x.left_gs, exci.data))
    elseif s == :right_gs
        Multiline(map(x -> x.right_gs, exci.data))
    elseif s == :trivial
        return reduce(&, map(x -> x.trivial, exci.data); init = true)
    else
        return getfield(exci, s)
    end
end

# These should really all be the same, so it might make sense to take the first instead
istrivial(exci::MultilineQP) = all(istrivial, exci.data)
istopological(exci::MultilineQP) = all(istopological, exci.data)

# VectorInterface
# ---------------

VectorInterface.scalartype(T::Type{<:QP}) = scalartype(eltype(T))

function VectorInterface.zerovector(ϕ::QP, ::Type{S}) where {S <: Number}
    ϕ = similar(ϕ, S)
    return zerovector!(ϕ)
end

function VectorInterface.zerovector!(ϕ::QP)
    zerovector!.(ϕ.Xs)
    return ϕ
end

# TODO: zerovector!!?

function VectorInterface.scale(ϕ::QP, α::Number)
    return scale!(zerovector(ϕ, VectorInterface.promote_scale(ϕ, α)), ϕ, α)
end

function VectorInterface.scale!(ϕ::QP, α::Number)
    scale!.(ϕ.Xs, α)
    return ϕ
end

VectorInterface.scale!!(ϕ::QP, α::Number) = scale!(ϕ, α)

function VectorInterface.scale!(ϕ₁::QP, ϕ₂::QP, α::Number)
    scale!.(ϕ₁.Xs, ϕ₂.Xs, α)
    return ϕ₁
end

VectorInterface.scale!!(ϕ₁::QP, ϕ₂::QP, α::Number) = scale!(ϕ₁, ϕ₂, α)

# add

function VectorInterface.add(ϕ₁::QP, ϕ₂::QP, α::Number, β::Number)
    # TODO: this might be more efficient by calling `add` on Xs directly
    ϕ = zerovector(ϕ₁, VectorInterface.promote_add(ϕ₁, ϕ₂, α, β))
    return add!(scale!(ϕ, ϕ₁, β), ϕ₂, α)
end

function VectorInterface.add!(ϕ₁::QP, ϕ₂::QP, α::Number, β::Number)
    add!.(ϕ₁.Xs, ϕ₂.Xs, α, β)
    return ϕ₁
end

VectorInterface.add!!(ϕ₁::QP, ϕ₂::QP, α::Number, β::Number) = add!(ϕ₁, ϕ₂, α, β)

# inner

function VectorInterface.inner(ϕ₁::QP, ϕ₂::QP)
    return inner(ϕ₁.Xs, ϕ₂.Xs)
end
