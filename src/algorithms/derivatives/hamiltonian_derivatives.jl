const _HAM_MPS_TYPES = Union{
    FiniteMPS{<:MPSTensor},
    WindowMPS{<:MPSTensor},
    InfiniteMPS{<:MPSTensor},
}

# Single site derivative
# ----------------------
"""
    JordanMPO_AC_Hamiltonian{O1, O2, O3}

Efficient operator for representing the single-site derivative of a `MPOHamiltonian` sandwiched between two MPSs.
In particular, this operator aims to make maximal use of the structure of the `MPOHamiltonian` to reduce the number of operations required to apply the operator to a tensor.

`GL_I` and `GR_E` are the left environment in the not-started channel and the right
environment in the finished channel. They are identities, stored as `missing`, if both MPSs
are the same.
"""
struct JordanMPO_AC_Hamiltonian{O1, O2, O3, Bk <: AbstractBackend, Al} <: DerivativeOperator
    D::Union{O1, Missing} # onsite
    I::Union{O1, Missing} # not started
    E::Union{O1, Missing} # finished
    C::Union{O2, Missing} # starting
    B::Union{O2, Missing} # ending
    A::Union{O3, Missing} # continuing
    GL_I::Union{O1, Missing} # not started, left environment
    GR_E::Union{O1, Missing} # finished, right environment
    backend::Bk           # contraction backend used by the matvec
    allocator::Al         # scratch-buffer allocator used by the matvec

    function JordanMPO_AC_Hamiltonian{O1, O2, O3, Bk, Al}(
            D::Union{O1, Missing}, I::Union{O1, Missing}, E::Union{O1, Missing},
            C::Union{O2, Missing}, B::Union{O2, Missing}, A::Union{O3, Missing},
            GL_I::Union{O1, Missing}, GR_E::Union{O1, Missing},
            backend::Bk, allocator::Al
        ) where {O1, O2, O3, Bk <: AbstractBackend, Al}
        return new{O1, O2, O3, Bk, Al}(D, I, E, C, B, A, GL_I, GR_E, backend, allocator)
    end
end
function JordanMPO_AC_Hamiltonian{O1, O2, O3}(
        D, I, E, C, B, A, GL_I, GR_E, backend = DefaultBackend(), allocator = DefaultAllocator()
    ) where {O1, O2, O3}
    return JordanMPO_AC_Hamiltonian{O1, O2, O3, typeof(backend), typeof(allocator)}(
        ismissing(D) ? D : convert(O1, D), ismissing(I) ? I : convert(O1, I),
        ismissing(E) ? E : convert(O1, E), ismissing(C) ? C : convert(O2, C),
        ismissing(B) ? B : convert(O2, B), ismissing(A) ? A : convert(O3, A),
        ismissing(GL_I) ? GL_I : convert(O1, GL_I), ismissing(GR_E) ? GR_E : convert(O1, GR_E),
        backend, allocator
    )
end

function AC_hamiltonian(
        site::Int, below::_HAM_MPS_TYPES, operator::MPOHamiltonian, above::_HAM_MPS_TYPES, envs;
        prepare::Bool = true,
        backend::AbstractBackend = DefaultBackend(), allocator = DefaultAllocator()
    )
    prepare && has_halves(envs) &&
        return hamiltonian_from_halves(Val(1), site, below, envs, backend, allocator)
    GL = leftenv(envs, site, below; backend, allocator)
    GR = rightenv(envs, site, below; backend, allocator)
    W = operator[site]
    H_AC = JordanMPO_AC_Hamiltonian(GL, W, GR; samestate = below === above, backend, allocator)
    return prepare ? prepare_operator!!(H_AC) : H_AC
end

"""
    JordanMPO_AC_Hamiltonian(GL, W, GR; samestate = true, backend, allocator)

Construct the single-site derivative from its environments. If `samestate`, the bra and ket
are the same state, whose not-started left and finished right environments are identities.
"""
function JordanMPO_AC_Hamiltonian(
        GL::MPSTensor, W::JordanMPOTensor, GR::MPSTensor; samestate::Bool = true,
        backend::AbstractBackend = DefaultBackend(), allocator = DefaultAllocator()
    )
    WA = W.A
    A = MPO_AC_Hamiltonian(GL[2:(end - 1)], WA, GR[2:(end - 1)], backend, allocator)
    L = jordan_hamiltonian_left(Val(1), GL, W, samestate, backend, allocator)
    R = jordan_hamiltonian_right(Val(1), GR, W, samestate, backend, allocator)
    # specialization for nearest neighbours
    return assemble_operator_halves(L, R, nonzero_length(WA) == 0 ? missing : A, typeof(A), backend, allocator)
end

function prepare_operator!!(H::JordanMPO_AC_Hamiltonian{O1, O2, O3}) where {O1, O2, O3}
    A = ismissing(H.A) ? missing : prepare_operator!!(H.A)
    return JordanMPO_AC_Hamiltonian{O1, O2, prepared_operator_type(O3)}(
        H.D, H.I, H.E, H.C, H.B, A, H.GL_I, H.GR_E, H.backend, H.allocator
    )
end


# Two site derivative
# -------------------
"""
    JordanMPO_AC2_Hamiltonian{O1, O2, O3, O4}

Efficient operator for representing the two-site derivative of a `MPOHamiltonian` sandwiched between two MPSs.
In particular, this operator aims to make maximal use of the structure of the `MPOHamiltonian` to reduce the number of operations required to apply the operator to a tensor.

`GL_I` and `GR_E` are as for [`JordanMPO_AC_Hamiltonian`](@ref).
"""
struct JordanMPO_AC2_Hamiltonian{O1, O2, O3, O4, Bk <: AbstractBackend, Al} <: DerivativeOperator
    II::Union{O1, Missing} # not_started
    IC::Union{O2, Missing} # starting right
    ID::Union{O1, Missing} # onsite right
    CB::Union{O2, Missing} # starting left - ending right
    CA::Union{O3, Missing} # starting left - continuing right
    AB::Union{O3, Missing} # continuing left - ending right
    AA::Union{O4, Missing} # continuing left - continuing right
    BE::Union{O2, Missing} # ending left
    DE::Union{O1, Missing} # onsite left
    EE::Union{O1, Missing} # finished
    GL_I::Union{O1, Missing} # not started, left environment
    GR_E::Union{O1, Missing} # finished, right environment
    backend::Bk            # contraction backend used by the matvec
    allocator::Al          # scratch-buffer allocator used by the matvec

    function JordanMPO_AC2_Hamiltonian{O1, O2, O3, O4, Bk, Al}(
            II::Union{O1, Missing}, IC::Union{O2, Missing}, ID::Union{O1, Missing},
            CB::Union{O2, Missing}, CA::Union{O3, Missing},
            AB::Union{O3, Missing}, AA::Union{O4, Missing},
            BE::Union{O2, Missing}, DE::Union{O1, Missing}, EE::Union{O1, Missing},
            GL_I::Union{O1, Missing}, GR_E::Union{O1, Missing},
            backend::Bk, allocator::Al
        ) where {O1, O2, O3, O4, Bk <: AbstractBackend, Al}
        return new{O1, O2, O3, O4, Bk, Al}(II, IC, ID, CB, CA, AB, AA, BE, DE, EE, GL_I, GR_E, backend, allocator)
    end
end
function JordanMPO_AC2_Hamiltonian{O1, O2, O3, O4}(
        II, IC, ID, CB, CA, AB, AA, BE, DE, EE, GL_I, GR_E,
        backend = DefaultBackend(), allocator = DefaultAllocator()
    ) where {O1, O2, O3, O4}
    return JordanMPO_AC2_Hamiltonian{O1, O2, O3, O4, typeof(backend), typeof(allocator)}(
        ismissing(II) ? II : convert(O1, II), ismissing(IC) ? IC : convert(O2, IC),
        ismissing(ID) ? ID : convert(O1, ID), ismissing(CB) ? CB : convert(O2, CB),
        ismissing(CA) ? CA : convert(O3, CA), ismissing(AB) ? AB : convert(O3, AB),
        ismissing(AA) ? AA : convert(O4, AA), ismissing(BE) ? BE : convert(O2, BE),
        ismissing(DE) ? DE : convert(O1, DE), ismissing(EE) ? EE : convert(O1, EE),
        ismissing(GL_I) ? GL_I : convert(O1, GL_I), ismissing(GR_E) ? GR_E : convert(O1, GR_E),
        backend, allocator
    )
end

function AC2_hamiltonian(
        site::Int, below::_HAM_MPS_TYPES, operator::MPOHamiltonian, above::_HAM_MPS_TYPES, envs;
        prepare::Bool = true,
        backend::AbstractBackend = DefaultBackend(), allocator = DefaultAllocator()
    )
    prepare && has_halves(envs) &&
        return hamiltonian_from_halves(Val(2), site, below, envs, backend, allocator)
    GL = leftenv(envs, site, below; backend, allocator)
    GR = rightenv(envs, site + 1, below; backend, allocator)
    W1, W2 = operator[site], operator[site + 1]
    H_AC2 = JordanMPO_AC2_Hamiltonian(GL, W1, W2, GR; samestate = below === above, backend, allocator)
    return prepare ? prepare_operator!!(H_AC2) : H_AC2
end

for f in (:AC_hamiltonian, :AC2_hamiltonian)
    @eval function $f(
            site::Int, below::WindowMPS, operator::WindowMPOHamiltonian, above::WindowMPS,
            envs; kwargs...
        )
        return $f(site, below, operator.finite_ham, above, envs; kwargs...)
    end
end

"""
    _connected_channels(A1, A2) -> NTuple{3, Vector{Int}}

The continuing-channel indices that can carry a contribution across both bonds, as `(rows, mids, cols)`.
A middle channel counts only if it is reachable from some row of `A1` *and* reaches some column of `A2`,
and a row or column counts only if it meets such a middle channel.
Everything else contributes exactly zero.

Returns three empty index vectors when nothing connects, as in the nearest-neighbour case.

The channel indices are small dense integers, so reachability is tracked in flat bit-flag arrays and read
out with `findall`, which is already ascending.
That ordering is load-bearing: the three index vectors are used to slice `A1`/`A2` and the environments,
which only line up if all of them keep the original channel order.
"""
function _connected_channels(A1, A2)
    (nonzero_length(A1) == 0 || nonzero_length(A2) == 0) && return (Int[], Int[], Int[])
    keys1, keys2 = nonzero_keys(A1), nonzero_keys(A2)
    @assert size(A1, 4) == size(A2, 1) "A-blocks do not share a bond"

    # a middle channel is retained iff it is both fed from the left and feeding to the right
    FED, FEEDS, BOTH = 0x01, 0x02, 0x03
    flags = zeros(UInt8, size(A1, 4))
    for I in keys1
        @inbounds flags[I[4]] |= FED
    end
    for I in keys2
        @inbounds flags[I[1]] |= FEEDS
    end
    mids = findall(==(BOTH), flags)
    isempty(mids) && return (Int[], Int[], Int[])

    rows = falses(size(A1, 1))
    for I in keys1
        @inbounds flags[I[4]] == BOTH && (rows[I[1]] = true)
    end
    cols = falses(size(A2, 4))
    for I in keys2
        @inbounds flags[I[1]] == BOTH && (cols[I[4]] = true)
    end

    return findall(rows), mids, findall(cols)
end

function JordanMPO_AC2_Hamiltonian(
        GL::MPSTensor, W1::JordanMPOTensor, W2::JordanMPOTensor, GR::MPSTensor; samestate::Bool = true,
        backend::AbstractBackend = DefaultBackend(), allocator = DefaultAllocator()
    )
    pair = JordanPairData(W1, W2, storagetype(GL))
    # continuing - continuing, restricted to the channels that can actually contribute
    rows, mids, cols = pair.channels
    GL2, GR2 = GL[2:(end - 1)], GR[2:(end - 1)]
    O4 = typeof(MPO_AC2_Hamiltonian(GL2, pair.A1, pair.A2, GR2, backend, allocator))
    AA = isempty(mids) ? missing : MPO_AC2_Hamiltonian(GL2[rows], pair.A1, pair.A2, GR2[cols], backend, allocator)
    L = jordan_hamiltonian_left(Val(2), GL, pair, samestate, backend, allocator)
    R = jordan_hamiltonian_right(Val(2), GR, pair, samestate, backend, allocator)
    return assemble_operator_halves(L, R, AA, O4, backend, allocator)
end

function prepare_operator!!(H::JordanMPO_AC2_Hamiltonian{O1, O2, O3, O4}) where {O1, O2, O3, O4}
    AA = ismissing(H.AA) ? missing : prepare_operator!!(H.AA)
    return JordanMPO_AC2_Hamiltonian{O1, O2, O3, prepared_operator_type(O4)}(
        H.II, H.IC, H.ID, H.CB, H.CA, H.AB, AA, H.BE, H.DE, H.EE, H.GL_I, H.GR_E, H.backend, H.allocator
    )
end

# Actions
# -------
# Terms that start before the window act on `xL`, the ket with the not-started left
# environment applied, and terms that finish within it on `xR`, with the finished right one.
function _apply_boundaries(GL_I, x::MPSTensor, GR_E, backend, allocator)
    ismissing(GL_I) && ismissing(GR_E) && return x, x
    xL = if ismissing(GL_I)
        x
    else
        @plansor backend = backend allocator = allocator xL[-1 -2; -3] := GL_I[-1; 1] * x[1 -2; -3]
    end
    xR = if ismissing(GR_E)
        x
    else
        @plansor backend = backend allocator = allocator xR[-1 -2; -3] := x[-1 -2; 1] * GR_E[-3; 1]
    end
    return xL, xR
end
function _apply_boundaries(GL_I, x::MPOTensor, GR_E, backend, allocator)
    ismissing(GL_I) && ismissing(GR_E) && return x, x
    xL = if ismissing(GL_I)
        x
    else
        @plansor backend = backend allocator = allocator xL[-1 -2; -3 -4] := GL_I[-1; 1] * x[1 -2; -3 -4]
    end
    xR = if ismissing(GR_E)
        x
    else
        @plansor backend = backend allocator = allocator xR[-1 -2; -3 -4] := x[-1 -2; 1 -4] * GR_E[-3; 1]
    end
    return xL, xR
end
# The output lives on the bra, whose virtual spaces are those of `xL` on the left and `xR` on the right.
_zero_output(x, xL, xR) = xL === xR ? zerovector(x) : zerovector!(similar(x, codomain(xL) ← domain(xR)))
# The identity in the onsite folds of `prepare_operator!!` is the boundary environment.
_boundary(G, t, i) = ismissing(G) ? TensorKit.id(storagetype(t), space(t, i)) : G

function (H::JordanMPO_AC_Hamiltonian)(x::MPSTensor)
    backend, allocator = H.backend, H.allocator
    xL, xR = _apply_boundaries(H.GL_I, x, H.GR_E, backend, allocator)
    y = ismissing(H.A) ? _zero_output(x, xL, xR) : H.A(x)

    if !ismissing(H.D)
        xLR = last(_apply_boundaries(missing, xL, H.GR_E, backend, allocator))
        @plansor backend = backend allocator = allocator y[-1 -2; -3] += xLR[-1 1; -3] * H.D[-2; 1]
    end
    ismissing(H.E) || @plansor backend = backend allocator = allocator y[-1 -2; -3] += H.E[-1; 1] * xR[1 -2; -3]
    ismissing(H.I) || @plansor backend = backend allocator = allocator y[-1 -2; -3] += xL[-1 -2; 1] * H.I[1; -3]
    ismissing(H.C) || @plansor backend = backend allocator = allocator y[-1 -2; -3] += xL[-1 2; 1] * H.C[-2 -3; 2 1]
    ismissing(H.B) || @plansor backend = backend allocator = allocator y[-1 -2; -3] += H.B[-1 -2; 1 2] * xR[1 2; -3]

    return y
end

function (H::JordanMPO_AC2_Hamiltonian)(x::MPOTensor)
    backend, allocator = H.backend, H.allocator
    xL, xR = _apply_boundaries(H.GL_I, x, H.GR_E, backend, allocator)
    y = ismissing(H.AA) ? _zero_output(x, xL, xR) : H.AA(x)
    xLR = if ismissing(H.ID) && ismissing(H.CB) && ismissing(H.DE)
        x
    else
        last(_apply_boundaries(missing, xL, H.GR_E, backend, allocator))
    end

    ismissing(H.II) || @plansor backend = backend allocator = allocator y[-1 -2; -3 -4] += xL[-1 -2; 1 -4] * H.II[-3; 1]
    ismissing(H.IC) || @plansor backend = backend allocator = allocator y[-1 -2; -3 -4] += xL[-1 -2; 1 2] * H.IC[-4 -3; 2 1]
    ismissing(H.ID) || @plansor backend = backend allocator = allocator y[-1 -2; -3 -4] += xLR[-1 -2; -3 1] * H.ID[-4; 1]
    ismissing(H.CB) || @plansor backend = backend allocator = allocator y[-1 -2; -3 -4] += xLR[-1 1; -3 2] * H.CB[-2 -4; 1 2]
    ismissing(H.CA) || @plansor backend = backend allocator = allocator y[-1 -2; -3 -4] += xL[-1 1; 3 2] * H.CA[-2 -4 -3; 1 2 3]
    ismissing(H.AB) || @plansor backend = backend allocator = allocator y[-1 -2; -3 -4] += xR[1 2; -3 3] * H.AB[-1 -2 -4; 1 2 3]
    ismissing(H.BE) || @plansor backend = backend allocator = allocator y[-1 -2; -3 -4] += xR[1 2; -3 -4] * H.BE[-1 -2; 1 2]
    ismissing(H.DE) || @plansor backend = backend allocator = allocator y[-1 -2; -3 -4] += xLR[-1 1; -3 -4] * H.DE[-2; 1]
    ismissing(H.EE) || @plansor backend = backend allocator = allocator y[-1 -2; -3 -4] += xR[1 -2; -3 -4] * H.EE[-1; 1]

    return y
end


# Left and right halves
# ---------------------
"""
    JordanPairData

Operator-only data for the adjacent sites `i` and `i + 1` with operators `W1` and `W2`: the
products `C_i * B_{i+1}`, `A_i * B_{i+1}` and `C_i * A_{i+1}`, each `missing` if absent, and
the continuing blocks restricted to the connected `channels = (rows, mids, cols)` of
`_connected_channels`.
"""
struct JordanPairData{W <: JordanMPOTensor, T <: MPOTensor, U, V, A <: MPOTensor}
    W1::W
    W2::W
    CB::Union{Missing, T}
    AB::Union{Missing, U}
    CA::Union{Missing, V}
    channels::NTuple{3, Vector{Int}}
    A1::A
    A2::A
    function JordanPairData{W, T, U, V, A}(
            W1, W2, CB, AB, CA, channels, A1, A2
        ) where {W <: JordanMPOTensor, T <: MPOTensor, U, V, A <: MPOTensor}
        return new{W, T, U, V, A}(W1, W2, CB, AB, CA, channels, A1, A2)
    end
end

Base.@assume_effects :foldable function jordan_pair_type(::Type{W}, ::Type{M}) where {W <: JordanMPOTensor, M}
    S = spacetype(W)
    T = tensormaptype(S, 2, 2, M)
    O32, O23 = tensormaptype(S, 3, 2, M), tensormaptype(S, 2, 3, M)
    U = SparseBlockTensorMap{O32, scalartype(O32), S, 3, 2, 5}
    V = SparseBlockTensorMap{O23, scalartype(O23), S, 2, 3, 5}
    A = SparseBlockTensorMap{tensormaptype(S, 2, 2, storagetype(W)), scalartype(W), S, 2, 2, 4}
    return JordanPairData{W, T, U, V, A}
end

function JordanPairData(W1::JordanMPOTensor, W2::JordanMPOTensor, ::Type{M}) where {M}
    # block accessors recompute a fresh `SparseBlockTensorMap` on every access
    A1, B2, C1, A2 = W1.A, W2.B, W1.C, W2.A
    CB = if nonzero_length(C1) > 0 && nonzero_length(B2) > 0
        @plansor cb[-1 -2; -3 -4] := C1[-1; -3 1] * B2[1 -2; -4]
        only(cb)
    else
        missing
    end
    AB = if nonzero_length(A1) > 0 && nonzero_length(B2) > 0
        @plansor ab[-1 -2 -3; -4 -5] := A1[-1 -2; -4 1] * B2[1 -3; -5]
    else
        missing
    end
    CA = if nonzero_length(C1) > 0 && nonzero_length(A2) > 0
        @plansor ca[-1 -2; -3 -4 -5] := C1[-1; -3 1] * A2[1 -2; -4 -5]
    else
        missing
    end
    channels = _connected_channels(A1, A2)
    rows, mids, cols = channels
    A1r, A2r = A1[rows, 1:1, 1:1, mids], A2[mids, 1:1, 1:1, cols]
    return jordan_pair_type(typeof(W1), M)(W1, W2, CB, AB, CA, channels, A1r, A2r)
end

_jordan_types(G) = _jordan_types(typeof(G))
function _jordan_types(::Type{G}) where {G <: AbstractTensorMap}
    S, M = spacetype(G), storagetype(G)
    return tensormaptype(S, 1, 1, M), tensormaptype(S, 2, 2, M), tensormaptype(S, 3, 3, M)
end

_onsite(W) = nonzero_length(W.D) > 0 ? only(W.D) : missing
function _ending(GL, W, backend, allocator)
    WB = W.B
    nonzero_length(WB) > 0 || return missing
    GL2 = GL[2:(end - 1)]
    @plansor backend = backend allocator = allocator b[-1 -2; -3 -4] := GL2[-1 1; -3] * WB[1 -2; -4]
    return only(b)
end
function _starting(GR, W, backend, allocator)
    WC = W.C
    nonzero_length(WC) > 0 || return missing
    GR2 = GR[2:(end - 1)]
    @plansor backend = backend allocator = allocator c[-1 -2; -3 -4] := WC[-1; -3 1] * GR2[-4 1; -2]
    return only(c)
end

# The terms of `JordanMPO_AC_Hamiltonian` and `JordanMPO_AC2_Hamiltonian` that depend on only
# one environment, excluding the continuing `A`/`AA` term, with the default backend and
# allocator. Each onsite, not-started, finished or operator-only term is assigned to one half,
# and folded into the largest term of that half that applies to the same boundaries.
function jordan_hamiltonian_left(::Val{1}, GL, W::JordanMPOTensor, samestate::Bool, backend, allocator)
    O1, O2, _ = _jordan_types(GL)
    GL_I = samestate ? missing : removeunit(GL[1], 2)
    E = size(W, 1) > 1 ? removeunit(GL[end], 2) : missing
    D = nonzero_length(W.C) == 0 ? _onsite(W) : missing
    B = _ending(GL, W, backend, allocator)
    if !ismissing(B)
        if !ismissing(D)
            GL_I′ = _boundary(GL_I, B, 1)
            @plansor backend = backend allocator = allocator B[-1 -2; -3 -4] += GL_I′[-1; -3] * D[-2; -4]
        end
        if !ismissing(E)
            Id = id(storagetype(B), space(B, 2))
            @plansor backend = backend allocator = allocator B[-1 -2; -3 -4] += E[-1; -3] * Id[-2; -4]
        end
        D = E = missing
    end
    return JordanMPO_AC_Hamiltonian{O1, O2, Missing}(D, missing, E, missing, B, missing, GL_I, missing)
end
function jordan_hamiltonian_right(::Val{1}, GR, W::JordanMPOTensor, samestate::Bool, backend, allocator)
    O1, O2, _ = _jordan_types(GR)
    GR_E = samestate ? missing : transpose(removeunit(GR[end], 2))
    I = size(W, 4) > 1 ? removeunit(GR[1], 2) : missing
    C = _starting(GR, W, backend, allocator)
    if !ismissing(C)
        D = _onsite(W)
        if !ismissing(D)
            GR_E′ = _boundary(GR_E, C, 2)
            @plansor backend = backend allocator = allocator C[-1 -2; -3 -4] += D[-1; -3] * GR_E′[-2; -4]
        end
        if !ismissing(I)
            Id = id(storagetype(C), space(C, 1))
            @plansor backend = backend allocator = allocator C[-1 -2; -3 -4] += Id[-1; -3] * I[-4; -2]
        end
        I = missing
    end
    return JordanMPO_AC_Hamiltonian{O1, O2, Missing}(missing, I, missing, C, missing, missing, missing, GR_E)
end

function jordan_hamiltonian_left(::Val{2}, GL, pair::JordanPairData, samestate::Bool, backend, allocator)
    O1, O2, O3 = _jordan_types(GL)
    W = pair.W1
    GL_I = samestate ? missing : removeunit(GL[1], 2)
    EE = size(W, 1) > 1 ? removeunit(GL[end], 2) : missing
    DE = _onsite(W)
    BE = _ending(GL, W, backend, allocator)
    # `CB` belongs to the right half if it has `CA` to fold it into.
    CB = ismissing(pair.CA) ? pair.CB : missing
    AB = if ismissing(pair.AB)
        missing
    else
        GL2 = GL[2:(end - 1)]
        @plansor backend = backend allocator = allocator ab[-1 -2 -3; -4 -5 -6] := GL2[-1 1; -4] * pair.AB[1 -2 -3; -5 -6]
        only(ab)
    end
    if !ismissing(AB)
        GL_I′ = _boundary(GL_I, AB, 1)
        Id = id(storagetype(AB), space(AB, 3))
        ismissing(CB) || @plansor backend = backend allocator = allocator AB[-1 -2 -3; -4 -5 -6] += GL_I′[-1; -4] * CB[-2 -3; -5 -6]
        ismissing(BE) || @plansor backend = backend allocator = allocator AB[-1 -2 -3; -4 -5 -6] += BE[-1 -2; -4 -5] * Id[-3; -6]
        ismissing(DE) || @plansor backend = backend allocator = allocator AB[-1 -2 -3; -4 -5 -6] += GL_I′[-1; -4] * DE[-2; -5] * Id[-3; -6]
        if !ismissing(EE)
            Id2 = id(storagetype(AB), space(AB, 2) ⊗ space(AB, 3))
            @plansor backend = backend allocator = allocator AB[-1 -2 -3; -4 -5 -6] += EE[-1; -4] * Id2[-2 -3; -5 -6]
        end
        CB = BE = DE = EE = missing
    elseif !ismissing(BE)
        if !ismissing(DE)
            GL_I′ = _boundary(GL_I, BE, 1)
            @plansor backend = backend allocator = allocator BE[-1 -2; -3 -4] += GL_I′[-1; -3] * DE[-2; -4]
        end
        if !ismissing(EE)
            Id = id(storagetype(BE), space(BE, 2))
            @plansor backend = backend allocator = allocator BE[-1 -2; -3 -4] += EE[-1; -3] * Id[-2; -4]
        end
        DE = EE = missing
    end
    return JordanMPO_AC2_Hamiltonian{O1, O2, O3, Missing}(
        missing, missing, missing, CB, missing, AB, missing, BE, DE, EE, GL_I, missing
    )
end
function jordan_hamiltonian_right(::Val{2}, GR, pair::JordanPairData, samestate::Bool, backend, allocator)
    O1, O2, O3 = _jordan_types(GR)
    W = pair.W2
    GR_E = samestate ? missing : transpose(removeunit(GR[end], 2))
    II = size(W, 4) > 1 ? transpose(removeunit(GR[1], 2)) : missing
    ID = _onsite(W)
    IC = _starting(GR, W, backend, allocator)
    CA = if ismissing(pair.CA)
        missing
    else
        GR2 = GR[2:(end - 1)]
        @plansor backend = backend allocator = allocator ca[-1 -2 -3; -4 -5 -6] := pair.CA[-1 -2; -4 -5 1] * GR2[-6 1; -3]
        only(ca)
    end
    if !ismissing(CA)
        GR_E′ = _boundary(GR_E, CA, 3)
        Id = id(storagetype(CA), space(CA, 1))
        CB = pair.CB
        ismissing(CB) || @plansor backend = backend allocator = allocator CA[-1 -2 -3; -4 -5 -6] += CB[-1 -2; -4 -5] * GR_E′[-3; -6]
        ismissing(IC) || @plansor backend = backend allocator = allocator CA[-1 -2 -3; -4 -5 -6] += Id[-1; -4] * IC[-2 -3; -5 -6]
        ismissing(ID) || @plansor backend = backend allocator = allocator CA[-1 -2 -3; -4 -5 -6] += Id[-1; -4] * ID[-2; -5] * GR_E′[-3; -6]
        if !ismissing(II)
            Id2 = id(storagetype(CA), space(CA, 1) ⊗ space(CA, 2))
            @plansor backend = backend allocator = allocator CA[-1 -2 -3; -4 -5 -6] += Id2[-1 -2; -4 -5] * II[-3; -6]
        end
        IC = ID = II = missing
    elseif !ismissing(IC)
        if !ismissing(ID)
            GR_E′ = _boundary(GR_E, IC, 2)
            @plansor backend = backend allocator = allocator IC[-1 -2; -3 -4] += ID[-1; -3] * GR_E′[-2; -4]
        end
        if !ismissing(II)
            Id = id(storagetype(IC), space(IC, 1))
            @plansor backend = backend allocator = allocator IC[-1 -2; -3 -4] += Id[-1; -3] * II[-2; -4]
        end
        ID = II = missing
    end
    return JordanMPO_AC2_Hamiltonian{O1, O2, O3, Missing}(
        II, IC, ID, missing, CA, missing, missing, missing, missing, missing, missing, GR_E
    )
end

function assemble_operator_halves(L::JordanMPO_AC_Hamiltonian{O1, O2}, R, A, ::Type{O3}, backend, allocator) where {O1, O2, O3}
    return JordanMPO_AC_Hamiltonian{O1, O2, O3}(L.D, R.I, L.E, R.C, L.B, A, L.GL_I, R.GR_E, backend, allocator)
end
function assemble_operator_halves(L::JordanMPO_AC2_Hamiltonian{O1, O2, O3}, R, AA, ::Type{O4}, backend, allocator) where {O1, O2, O3, O4}
    return JordanMPO_AC2_Hamiltonian{O1, O2, O3, O4}(
        R.II, R.IC, R.ID, L.CB, R.CA, L.AB, AA, L.BE, L.DE, L.EE, L.GL_I, R.GR_E, backend, allocator,
    )
end

# Prepared halves for `FiniteEnvironments`
# -----------------------------------------
"""
    JordanHalf

One environment's prepared half of an effective `AC` or `AC2` Hamiltonian: the terms of
`jordan_hamiltonian_left` or `jordan_hamiltonian_right`, and `continuing`, that side's
prepared environment of the continuing term, or `missing` if absent.
"""
struct JordanHalf{H, C <: AbstractTensorMap}
    terms::H
    continuing::Union{Missing, C}
    function JordanHalf{H, C}(terms, continuing) where {H, C <: AbstractTensorMap}
        return new{H, C}(terms, continuing)
    end
end

function jordan_halftypes(::Val{1}, ::Type{G}, ::Type{W}) where {G, W}
    O1, O2, _ = _jordan_types(G)
    H = JordanMPO_AC_Hamiltonian{O1, O2, Missing, DefaultBackend, DefaultAllocator}
    L, R = halftypes(Val(1), G, W)
    return JordanHalf{H, L}, JordanHalf{H, R}
end
function jordan_halftypes(::Val{2}, ::Type{G}, ::Type{W}) where {G, W}
    O1, O2, O3 = _jordan_types(G)
    H = JordanMPO_AC2_Hamiltonian{O1, O2, O3, Missing, DefaultBackend, DefaultAllocator}
    L, R = halftypes(Val(2), G, W)
    return JordanHalf{H, L}, JordanHalf{H, R}
end

function environment_halves(O::MPOHamiltonian, ::Type{G}, N) where {G}
    W = eltype(O)
    (G <: MPSTensor && isconcretetype(W)) || return nothing
    AC = PreparedHalves{jordan_halftypes(Val(1), G, W)..., G}(N)
    AC2 = PreparedHalves{jordan_halftypes(Val(2), G, W)..., G}(N)
    pairs = fill!(Vector{Union{Nothing, jordan_pair_type(W, storagetype(G))}}(undef, max(N - 1, 0)), nothing)
    return (; AC, AC2, pairs)
end

function pair_data!(envs::FiniteEnvironments, i::Int)
    pairs = envs.halves.pairs
    pair = pairs[i]
    isnothing(pair) || return pair
    O = envs.operator
    return pairs[i] = JordanPairData(O[i], O[i + 1], storagetype(eltype(envs.GLs)))
end

function prepare_operator_left(n::Val{1}, O::MPOHamiltonian, envs, GL, i, backend, allocator)
    W = O[i]
    WA = W.A
    terms = jordan_hamiltonian_left(n, GL, W, isnothing(envs.above), backend, allocator)
    continuing = nonzero_length(WA) > 0 ? prepare_operator_left(n, GL[2:(end - 1)], WA, backend, allocator) : missing
    L, _ = jordan_halftypes(n, eltype(envs.GLs), eltype(O))
    return L(terms, continuing)
end
function prepare_operator_right(n::Val{1}, O::MPOHamiltonian, envs, GR, i, backend, allocator)
    W = O[i]
    WA = W.A
    terms = jordan_hamiltonian_right(n, GR, W, isnothing(envs.above), backend, allocator)
    continuing = nonzero_length(WA) > 0 ? prepare_operator_right(n, GR[2:(end - 1)], WA, backend, allocator) : missing
    _, R = jordan_halftypes(n, eltype(envs.GRs), eltype(O))
    return R(terms, continuing)
end
function prepare_operator_left(n::Val{2}, O::MPOHamiltonian, envs, GL, i, backend, allocator)
    pair = pair_data!(envs, i)
    terms = jordan_hamiltonian_left(n, GL, pair, isnothing(envs.above), backend, allocator)
    rows, mids, _ = pair.channels
    continuing = isempty(mids) ? missing : prepare_operator_left(n, GL[2:(end - 1)][rows], pair.A1, backend, allocator)
    L, _ = jordan_halftypes(n, eltype(envs.GLs), eltype(O))
    return L(terms, continuing)
end
function prepare_operator_right(n::Val{2}, O::MPOHamiltonian, envs, GR, i, backend, allocator)
    pair = pair_data!(envs, i - 1)
    terms = jordan_hamiltonian_right(n, GR, pair, isnothing(envs.above), backend, allocator)
    _, mids, cols = pair.channels
    continuing = isempty(mids) ? missing : prepare_operator_right(n, GR[2:(end - 1)][cols], pair.A2, backend, allocator)
    _, R = jordan_halftypes(n, eltype(envs.GRs), eltype(O))
    return R(terms, continuing)
end

function assemble_operator_halves(L::JordanHalf{<:Any, CL}, R::JordanHalf{<:Any, CR}, backend, allocator) where {CL, CR}
    P = precomputed_type(CL, CR, typeof(backend), typeof(allocator))
    A = ismissing(L.continuing) ? missing : P(L.continuing, R.continuing, backend, allocator)
    return assemble_operator_halves(L.terms, R.terms, A, P, backend, allocator)
end
