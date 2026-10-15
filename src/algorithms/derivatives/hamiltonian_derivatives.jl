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
"""
struct JordanMPO_AC_Hamiltonian{O1, O2, O3, Bk <: AbstractBackend, Al} <: DerivativeOperator
    D::Union{O1, Missing} # onsite
    I::Union{O1, Missing} # not started
    E::Union{O1, Missing} # finished
    C::Union{O2, Missing} # starting
    B::Union{O2, Missing} # ending
    A::Union{O3, Missing} # continuing
    backend::Bk           # contraction backend used by the matvec
    allocator::Al         # scratch-buffer allocator used by the matvec

    function JordanMPO_AC_Hamiltonian{O1, O2, O3, Bk, Al}(
            D::Union{O1, Missing}, I::Union{O1, Missing}, E::Union{O1, Missing},
            C::Union{O2, Missing}, B::Union{O2, Missing}, A::Union{O3, Missing},
            backend::Bk, allocator::Al
        ) where {O1, O2, O3, Bk <: AbstractBackend, Al}
        return new{O1, O2, O3, Bk, Al}(D, I, E, C, B, A, backend, allocator)
    end
end
function JordanMPO_AC_Hamiltonian{O1, O2, O3}(
        D, I, E, C, B, A, backend = DefaultBackend(), allocator = DefaultAllocator()
    ) where {O1, O2, O3}
    return JordanMPO_AC_Hamiltonian{O1, O2, O3, typeof(backend), typeof(allocator)}(
        ismissing(D) ? D : convert(O1, D), ismissing(I) ? I : convert(O1, I),
        ismissing(E) ? E : convert(O1, E), ismissing(C) ? C : convert(O2, C),
        ismissing(B) ? B : convert(O2, B), ismissing(A) ? A : convert(O3, A),
        backend, allocator
    )
end

function AC_hamiltonian(
        site::Int, below::_HAM_MPS_TYPES, operator::MPOHamiltonian, above::_HAM_MPS_TYPES, envs;
        prepare::Bool = true,
        backend::AbstractBackend = DefaultBackend(), allocator = DefaultAllocator()
    )
    @assert below === above "JordanMPO assumptions break"
    GL = leftenv(envs, site, below; backend, allocator)
    GR = rightenv(envs, site, below; backend, allocator)
    W = operator[site]
    H_AC = JordanMPO_AC_Hamiltonian(GL, W, GR; backend, allocator)
    return prepare ? prepare_operator!!(H_AC) : H_AC
end

function JordanMPO_AC_Hamiltonian(
        GL::MPSTensor, W::JordanMPOTensor, GR::MPSTensor;
        backend::AbstractBackend = DefaultBackend(), allocator = DefaultAllocator()
    )
    WA = W.A
    A = MPO_AC_Hamiltonian(GL[2:(end - 1)], WA, GR[2:(end - 1)], backend, allocator)
    L = left_half(JordanMPO_AC_Hamiltonian, GL, W, backend, allocator)
    R = right_half(JordanMPO_AC_Hamiltonian, GR, W, backend, allocator)
    # specialization for nearest neighbours
    return assemble_operator_halves(L, R, nonzero_length(WA) == 0 ? missing : A, typeof(A), backend, allocator)
end

function prepare_operator!!(
        H::JordanMPO_AC_Hamiltonian{O1, O2, O3}
    ) where {O1, O2, O3}
    backend, allocator = H.backend, H.allocator
    C::Union{Missing, O2} = H.C
    B::Union{Missing, O2} = H.B

    # onsite
    D::Union{Missing, O1} = if ismissing(H.D)
        missing
    elseif !ismissing(C)
        Id = TensorKit.id(storagetype(C), space(C, 2))
        @plansor backend = backend allocator = allocator C[-1 -2; -3 -4] += H.D[-1; -3] * Id[-2; -4]
        missing
    elseif !ismissing(B)
        Id = TensorKit.id(storagetype(B), space(B, 1))
        @plansor backend = backend allocator = allocator B[-1 -2; -3 -4] += Id[-1; -3] * H.D[-2; -4]
        missing
    else
        H.D
    end

    # not_started
    I::Union{Missing, O1} = if ismissing(H.I)
        missing
    elseif !ismissing(C)
        Id = id(storagetype(C), space(C, 1))
        @plansor backend = backend allocator = allocator C[-1 -2; -3 -4] += Id[-1; -3] * H.I[-4; -2]
        missing
    else
        H.I
    end

    # finished
    E::Union{Missing, O1} = if ismissing(H.E)
        missing
    elseif !ismissing(B)
        Id = id(storagetype(B), space(B, 2))
        @plansor backend = backend allocator = allocator B[-1 -2; -3 -4] += H.E[-1; -3] * Id[-2; -4]
        missing
    else
        H.E
    end

    O3′ = prepared_operator_type(O3)
    A = ismissing(H.A) ? H.A : prepare_operator!!(H.A)

    return JordanMPO_AC_Hamiltonian{O1, O2, O3′}(D, I, E, C, B, A, backend, allocator)::JordanMPO_AC_Hamiltonian{O1, O2, O3′}
end


# Two site derivative
# -------------------
"""
    JordanMPO_AC2_Hamiltonian{O1, O2, O3, O4}

Efficient operator for representing the single-site derivative of a `MPOHamiltonian` sandwiched between two MPSs.
In particular, this operator aims to make maximal use of the structure of the `MPOHamiltonian` to reduce the number of operations required to apply the operator to a tensor.
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
    backend::Bk            # contraction backend used by the matvec
    allocator::Al          # scratch-buffer allocator used by the matvec

    function JordanMPO_AC2_Hamiltonian{O1, O2, O3, O4, Bk, Al}(
            II::Union{O1, Missing}, IC::Union{O2, Missing}, ID::Union{O1, Missing},
            CB::Union{O2, Missing}, CA::Union{O3, Missing},
            AB::Union{O3, Missing}, AA::Union{O4, Missing},
            BE::Union{O2, Missing}, DE::Union{O1, Missing}, EE::Union{O1, Missing},
            backend::Bk, allocator::Al
        ) where {O1, O2, O3, O4, Bk <: AbstractBackend, Al}
        return new{O1, O2, O3, O4, Bk, Al}(II, IC, ID, CB, CA, AB, AA, BE, DE, EE, backend, allocator)
    end
end
function JordanMPO_AC2_Hamiltonian{O1, O2, O3, O4}(
        II, IC, ID, CB, CA, AB, AA, BE, DE, EE,
        backend = DefaultBackend(), allocator = DefaultAllocator()
    ) where {O1, O2, O3, O4}
    return JordanMPO_AC2_Hamiltonian{O1, O2, O3, O4, typeof(backend), typeof(allocator)}(
        ismissing(II) ? II : convert(O1, II), ismissing(IC) ? IC : convert(O2, IC),
        ismissing(ID) ? ID : convert(O1, ID), ismissing(CB) ? CB : convert(O2, CB),
        ismissing(CA) ? CA : convert(O3, CA), ismissing(AB) ? AB : convert(O3, AB),
        ismissing(AA) ? AA : convert(O4, AA), ismissing(BE) ? BE : convert(O2, BE),
        ismissing(DE) ? DE : convert(O1, DE), ismissing(EE) ? EE : convert(O1, EE),
        backend, allocator
    )
end

function AC2_hamiltonian(
        site::Int, below::_HAM_MPS_TYPES, operator::MPOHamiltonian, above::_HAM_MPS_TYPES, envs;
        prepare::Bool = true,
        backend::AbstractBackend = DefaultBackend(), allocator = DefaultAllocator()
    )
    @assert below === above "JordanMPO assumptions break"
    GL = leftenv(envs, site, below; backend, allocator)
    GR = rightenv(envs, site + 1, below; backend, allocator)
    W1, W2 = operator[site], operator[site + 1]
    H_AC2 = JordanMPO_AC2_Hamiltonian(GL, W1, W2, GR; backend, allocator)
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
        GL::MPSTensor, W1::JordanMPOTensor, W2::JordanMPOTensor, GR::MPSTensor;
        backend::AbstractBackend = DefaultBackend(), allocator = DefaultAllocator()
    )
    pair = JordanPairData(W1, W2, storagetype(GL))
    GL2, GR2 = GL[2:(end - 1)], GR[2:(end - 1)]
    O4 = _AA_type(GL2, pair, GR2, backend, allocator)
    # continuing - continuing, restricted to the channels that can actually contribute
    rows, mids, cols = pair.channels
    AA = isempty(mids) ? missing :
        MPO_AC2_Hamiltonian(GL2[rows], pair.A1, pair.A2, GR2[cols], backend, allocator)
    L = left_half(JordanMPO_AC2_Hamiltonian, GL, W1, pair, backend, allocator)
    R = right_half(JordanMPO_AC2_Hamiltonian, GR, W2, pair, backend, allocator)
    return assemble_operator_halves(L, R, AA, O4, backend, allocator)
end

function prepare_operator!!(
        H::JordanMPO_AC2_Hamiltonian{O1, O2, O3, O4}
    ) where {O1, O2, O3, O4}
    backend, allocator = H.backend, H.allocator

    CA::Union{Missing, O3} = H.CA
    AB::Union{Missing, O3} = H.AB

    CB::Union{Missing, O2} = if !ismissing(CA) && !ismissing(H.CB)
        Id = TensorKit.id(storagetype(H.CB), space(CA, 3))
        @plansor backend = backend allocator = allocator CA[-1 -2 -3; -4 -5 -6] += H.CB[-1 -2; -4 -5] * Id[-3; -6]
        missing
    elseif !ismissing(AB) && !ismissing(H.CB)
        Id = TensorKit.id(storagetype(H.CB), space(AB, 1))
        @plansor backend = backend allocator = allocator AB[-1 -2 -3; -4 -5 -6] += H.CB[-2 -3; -5 -6] * Id[-1; -4]
        missing
    else
        H.CB
    end

    # starting right
    IC::Union{Missing, O2} = if !ismissing(CA) && !ismissing(H.IC)
        Id = TensorKit.id(storagetype(H.IC), space(CA, 1))
        @plansor backend = backend allocator = allocator CA[-1 -2 -3; -4 -5 -6] += Id[-1; -4] * H.IC[ -2 -3; -5 -6]
        missing
    else
        H.IC
    end

    # ending left
    BE::Union{Missing, O2} = if !ismissing(AB) && !ismissing(H.BE)
        Id = TensorKit.id(storagetype(H.BE), space(AB, 3))
        @plansor backend = backend allocator = allocator AB[-1 -2 -3; -4 -5 -6] += H.BE[-1 -2; -4 -5] * Id[-3; -6]
        missing
    else
        H.BE
    end

    # onsite left
    DE::Union{Missing, O1} = if !ismissing(BE) && !ismissing(H.DE)
        Id = TensorKit.id(storagetype(H.DE), space(BE, 1))
        @plansor backend = backend allocator = allocator BE[-1 -2; -3 -4] += Id[-1; -3] * H.DE[-2; -4]
        missing
    elseif !ismissing(AB) && !ismissing(H.DE)
        Id1 = id(storagetype(H.DE), space(AB, 1))
        Id2 = id(storagetype(H.DE), space(AB, 3))
        @plansor backend = backend allocator = allocator AB[-1 -2 -3; -4 -5 -6] += Id1[-1; -4] * H.DE[-2; -5] * Id2[-3; -6]
        missing
        # TODO: could also try in CA?
    else
        H.DE
    end

    # onsite right
    ID::Union{Missing, O1} = if !ismissing(IC) && !ismissing(H.ID)
        Id = TensorKit.id(storagetype(H.ID), space(IC, 2))
        @plansor backend = backend allocator = allocator IC[-1 -2; -3 -4] += H.ID[-1; -3] * Id[-2; -4]
        missing
    elseif !ismissing(CA) && !ismissing(H.ID)
        Id1 = TensorKit.id(storagetype(H.ID), space(CA, 1))
        Id2 = TensorKit.id(storagetype(H.ID), space(CA, 3))
        @plansor backend = backend allocator = allocator CA[-1 -2 -3; -4 -5 -6] += Id1[-1; -4] * H.ID[-2; -5] * Id2[-3; -6]
        missing
    else
        H.ID
    end

    # finished
    II::Union{Missing, O1} = if !ismissing(IC) && !ismissing(H.II)
        I = id(storagetype(H.II), space(IC, 1))
        @plansor backend = backend allocator = allocator IC[-1 -2; -3 -4] += I[-1; -3] * H.II[-2; -4]
        II = missing
    elseif !ismissing(CA) && !ismissing(H.II)
        I = id(storagetype(H.II), space(CA, 1) ⊗ space(CA, 2))
        @plansor backend = backend allocator = allocator CA[-1 -2 -3; -4 -5 -6] += I[-1 -2; -4 -5] * H.II[-3; -6]
        II = missing
    else
        H.II
    end

    # unstarted
    EE::Union{Missing, O1} = if !ismissing(BE) && !ismissing(H.EE)
        I = id(storagetype(H.EE), space(BE, 2))
        @plansor backend = backend allocator = allocator BE[-1 -2; -3 -4] += H.EE[-1; -3] * I[-2; -4]
        EE = missing
    elseif !ismissing(AB) && !ismissing(H.EE)
        I = id(storagetype(H.EE), space(AB, 2) ⊗ space(AB, 3))
        @plansor backend = backend allocator = allocator AB[-1 -2 -3; -4 -5 -6] += H.EE[-1; -4] * I[-2 -3; -5 -6]
        EE = missing
    else
        H.EE
    end

    O4′ = prepared_operator_type(O4)
    AA = prepare_operator!!(H.AA)

    return JordanMPO_AC2_Hamiltonian{O1, O2, O3, O4′}(II, IC, ID, CB, CA, AB, AA, BE, DE, EE, backend, allocator)
end

# Actions
# -------
function (H::JordanMPO_AC_Hamiltonian)(x::MPSTensor)
    backend, allocator = H.backend, H.allocator
    y = ismissing(H.A) ? zerovector(x) : H.A(x)

    ismissing(H.D) || @plansor backend = backend allocator = allocator y[-1 -2; -3] += x[-1 1; -3] * H.D[-2; 1]
    ismissing(H.E) || @plansor backend = backend allocator = allocator y[-1 -2; -3] += H.E[-1; 1] * x[1 -2; -3]
    ismissing(H.I) || @plansor backend = backend allocator = allocator y[-1 -2; -3] += x[-1 -2; 1] * H.I[1; -3]
    ismissing(H.C) || @plansor backend = backend allocator = allocator y[-1 -2; -3] += x[-1 2; 1] * H.C[-2 -3; 2 1]
    ismissing(H.B) || @plansor backend = backend allocator = allocator y[-1 -2; -3] += H.B[-1 -2; 1 2] * x[1 2; -3]

    return y
end

function (H::JordanMPO_AC2_Hamiltonian)(x::MPOTensor)
    backend, allocator = H.backend, H.allocator
    y = ismissing(H.AA) ? zerovector(x) : H.AA(x)

    ismissing(H.II) || @plansor backend = backend allocator = allocator y[-1 -2; -3 -4] += x[-1 -2; 1 -4] * H.II[-3; 1]
    ismissing(H.IC) || @plansor backend = backend allocator = allocator y[-1 -2; -3 -4] += x[-1 -2; 1 2] * H.IC[-4 -3; 2 1]
    ismissing(H.ID) || @plansor backend = backend allocator = allocator y[-1 -2; -3 -4] += x[-1 -2; -3 1] * H.ID[-4; 1]
    ismissing(H.CB) || @plansor backend = backend allocator = allocator y[-1 -2; -3 -4] += x[-1 1; -3 2] * H.CB[-2 -4; 1 2]
    ismissing(H.CA) || @plansor backend = backend allocator = allocator y[-1 -2; -3 -4] += x[-1 1; 3 2] * H.CA[-2 -4 -3; 1 2 3]
    ismissing(H.AB) || @plansor backend = backend allocator = allocator y[-1 -2; -3 -4] += x[1 2; -3 3] * H.AB[-1 -2 -4; 1 2 3]
    ismissing(H.BE) || @plansor backend = backend allocator = allocator y[-1 -2; -3 -4] += x[1 2; -3 -4] * H.BE[-1 -2; 1 2]
    ismissing(H.DE) || @plansor backend = backend allocator = allocator y[-1 -2; -3 -4] += x[-1 1; -3 -4] * H.DE[-2; 1]
    ismissing(H.EE) || @plansor backend = backend allocator = allocator y[-1 -2; -3 -4] += x[1 -2; -3 -4] * H.EE[-1; 1]

    return y
end


# Left and right halves
# ---------------------
"""
    JordanPairData

Operator-only data for the adjacent sites `i` and `i + 1`: the products `C_i * B_{i+1}`,
`A_i * B_{i+1}` and `C_i * A_{i+1}`, each `missing` if absent, and the continuing blocks
restricted to the connected `channels = (rows, mids, cols)` of `_connected_channels`.
"""
struct JordanPairData{T <: MPOTensor, U, V, A <: MPOTensor}
    CB::Union{Missing, T}
    AB::Union{Missing, U}
    CA::Union{Missing, V}
    channels::NTuple{3, Vector{Int}}
    A1::A
    A2::A
end

function JordanPairData(W1::JordanMPOTensor, W2::JordanMPOTensor, ::Type{M}) where {M}
    S = spacetype(W1)
    T = tensormaptype(S, 2, 2, M)
    O32, O23 = tensormaptype(S, 3, 2, M), tensormaptype(S, 2, 3, M)
    U = SparseBlockTensorMap{O32, scalartype(O32), S, 3, 2, 5}
    V = SparseBlockTensorMap{O23, scalartype(O23), S, 2, 3, 5}
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
    return JordanPairData{T, U, V, typeof(A1r)}(CB, AB, CA, channels, A1r, A2r)
end

function _jordan_types(G)
    S, M = spacetype(G), storagetype(G)
    return tensormaptype(S, 1, 1, M), tensormaptype(S, 2, 2, M), tensormaptype(S, 3, 3, M)
end
_A_type(GL2, A, GR2, backend, allocator) =
    MPO_AC_Hamiltonian{typeof(GL2), typeof(A), typeof(GR2), typeof(backend), typeof(allocator)}
_AA_type(GL2, pair, GR2, backend, allocator) = MPO_AC2_Hamiltonian{
    typeof(GL2), typeof(pair.A1), typeof(pair.A2), typeof(GR2), typeof(backend), typeof(allocator),
}

_onsite(D) = nonzero_length(D) > 0 ? only(D) : missing
function _ending(GL2, B, backend, allocator)
    nonzero_length(B) > 0 || return missing
    @plansor backend = backend allocator = allocator b[-1 -2; -3 -4] := GL2[-1 1; -3] * B[1 -2; -4]
    return only(b)
end
function _starting(GR2, C, backend, allocator)
    nonzero_length(C) > 0 || return missing
    @plansor backend = backend allocator = allocator c[-1 -2; -3 -4] := C[-1; -3 1] * GR2[-4 1; -2]
    return only(c)
end

# The terms of `JordanMPO_AC_Hamiltonian` and `JordanMPO_AC2_Hamiltonian` that depend on only
# one environment, excluding the continuing `A`/`AA` term. Each onsite or operator-only term
# is assigned to the half whose `prepare_operator!!` can fold it.
function left_half(::Type{<:JordanMPO_AC_Hamiltonian}, GL, W::JordanMPOTensor, backend, allocator)
    GL2 = GL[2:(end - 1)]
    O1, O2, _ = _jordan_types(GL)
    O3 = _A_type(GL2, W.A, GL2, backend, allocator)
    E = size(W, 1) > 1 ? removeunit(GL[end], 2) : missing
    D = nonzero_length(W.C) == 0 ? _onsite(W.D) : missing
    B = _ending(GL2, W.B, backend, allocator)
    return JordanMPO_AC_Hamiltonian{O1, O2, O3}(D, missing, E, missing, B, missing, backend, allocator)
end
function right_half(::Type{<:JordanMPO_AC_Hamiltonian}, GR, W::JordanMPOTensor, backend, allocator)
    GR2 = GR[2:(end - 1)]
    O1, O2, _ = _jordan_types(GR)
    O3 = _A_type(GR2, W.A, GR2, backend, allocator)
    I = size(W, 4) > 1 ? removeunit(GR[1], 2) : missing
    WC = W.C
    D = nonzero_length(WC) > 0 ? _onsite(W.D) : missing
    C = _starting(GR2, WC, backend, allocator)
    return JordanMPO_AC_Hamiltonian{O1, O2, O3}(D, I, missing, C, missing, missing, backend, allocator)
end

function left_half(::Type{<:JordanMPO_AC2_Hamiltonian}, GL, W::JordanMPOTensor, pair::JordanPairData, backend, allocator)
    GL2 = GL[2:(end - 1)]
    O1, O2, O3 = _jordan_types(GL)
    O4 = _AA_type(GL2, pair, GL2, backend, allocator)
    EE = size(W, 1) > 1 ? removeunit(GL[end], 2) : missing
    AB = if ismissing(pair.AB)
        missing
    else
        @plansor backend = backend allocator = allocator ab[-1 -2 -3; -4 -5 -6] := GL2[-1 1; -4] * pair.AB[1 -2 -3; -5 -6]
        only(ab)
    end
    # CB is folded into CA if available, otherwise into AB.
    CB = ismissing(pair.CA) ? pair.CB : missing
    return JordanMPO_AC2_Hamiltonian{O1, O2, O3, O4}(
        missing, missing, missing, CB, missing, AB, missing,
        _ending(GL2, W.B, backend, allocator), _onsite(W.D), EE, backend, allocator,
    )
end
function right_half(::Type{<:JordanMPO_AC2_Hamiltonian}, GR, W::JordanMPOTensor, pair::JordanPairData, backend, allocator)
    GR2 = GR[2:(end - 1)]
    O1, O2, O3 = _jordan_types(GR)
    O4 = _AA_type(GR2, pair, GR2, backend, allocator)
    II = size(W, 4) > 1 ? transpose(removeunit(GR[1], 2)) : missing
    CA = if ismissing(pair.CA)
        missing
    else
        @plansor backend = backend allocator = allocator ca[-1 -2 -3; -4 -5 -6] := pair.CA[-1 -2; -4 -5 1] * GR2[-6 1; -3]
        only(ca)
    end
    CB = ismissing(CA) ? missing : pair.CB
    return JordanMPO_AC2_Hamiltonian{O1, O2, O3, O4}(
        II, _starting(GR2, W.C, backend, allocator), _onsite(W.D), CB, CA,
        missing, missing, missing, missing, missing, backend, allocator,
    )
end

function assemble_operator_halves(L::JordanMPO_AC_Hamiltonian{O1, O2}, R, A, ::Type{O3}, backend, allocator) where {O1, O2, O3}
    D = ismissing(L.D) ? R.D : L.D
    return JordanMPO_AC_Hamiltonian{O1, O2, O3}(D, R.I, L.E, R.C, L.B, A, backend, allocator)
end
function assemble_operator_halves(L::JordanMPO_AC2_Hamiltonian{O1, O2, O3}, R, AA, ::Type{O4}, backend, allocator) where {O1, O2, O3, O4}
    CB = ismissing(L.CB) ? R.CB : L.CB
    return JordanMPO_AC2_Hamiltonian{O1, O2, O3, O4}(
        R.II, R.IC, R.ID, CB, R.CA, L.AB, AA, L.BE, L.DE, L.EE, backend, allocator,
    )
end

# Prepared halves for `DMRGEnvironments`
# ---------------------------------------
"""
    JordanHalf

One environment's prepared half of an effective `AC` or `AC2` Hamiltonian. `continuing` is
that side's fused environment for assembling the continuing term, or `missing` if absent.
"""
struct JordanHalf{H, C <: AbstractTensorMap}
    prepared::H
    continuing::Union{Missing, C}
end
JordanHalf(prepared, continuing, ::Type{C}) where {C} = JordanHalf{typeof(prepared), C}(prepared, continuing)

pair_data(O::MPOHamiltonian, GL) = [JordanPairData(O[i], O[i + 1], storagetype(GL)) for i in 1:(length(O) - 1)]

function prepare_lefthalf(GL, O::MPOHamiltonian, pairs, i, nsites, backend, allocator)
    GL2 = GL[2:(end - 1)]
    _, O2, _ = _jordan_types(GL)
    if nsites == 1
        W = O[i]
        A = W.A
        half = left_half(JordanMPO_AC_Hamiltonian, GL, W, backend, allocator)
        continuing = nonzero_length(A) > 0 ? _prepare_GL_O(GL2, A, backend, allocator) : missing
    else
        i < length(O) || return missing
        pair = pairs[i]
        half = left_half(JordanMPO_AC2_Hamiltonian, GL, O[i], pair, backend, allocator)
        rows, mids, _ = pair.channels
        continuing = isempty(mids) ? missing : _prepare_GL_O(GL2[rows], pair.A1, backend, allocator)
    end
    return JordanHalf(prepare_operator!!(half), continuing, O2)
end
function prepare_righthalf(GR, O::MPOHamiltonian, pairs, i, nsites, backend, allocator)
    GR2 = GR[2:(end - 1)]
    S, M = spacetype(GR), storagetype(GR)
    if nsites == 1
        W = O[i]
        half = right_half(JordanMPO_AC_Hamiltonian, GR, W, backend, allocator)
        continuing = nonzero_length(W.A) > 0 ? TensorMap(GR2) : missing
        return JordanHalf(prepare_operator!!(half), continuing, tensormaptype(S, 2, 1, M))
    end
    i > 1 || return missing
    pair = pairs[i - 1]
    half = right_half(JordanMPO_AC2_Hamiltonian, GR, O[i], pair, backend, allocator)
    _, mids, cols = pair.channels
    continuing = isempty(mids) ? missing : _prepare_O_GR(pair.A2, GR2[cols], backend, allocator)
    return JordanHalf(prepare_operator!!(half), continuing, tensormaptype(S, 2, 2, M))
end

function AC_hamiltonian(
        i::Int, below::_HAM_MPS_TYPES, O::MPOHamiltonian, above::_HAM_MPS_TYPES,
        envs::DMRGEnvironments;
        prepare::Bool = true, backend::AbstractBackend = envs.backend, allocator = envs.allocator,
    )
    @assert below === above "JordanMPO assumptions break"
    GL, GR = leftenv(envs, i, below), rightenv(envs, i, below)
    if !(prepare && envs.nsites == 1)
        H = JordanMPO_AC_Hamiltonian(GL, O[i], GR; backend, allocator)
        return prepare ? prepare_operator!!(H) : H
    end
    l, r = lefthalf(envs, i), righthalf(envs, i)
    T = prepared_operator_type(_A_type(GL, O[i].A, GR, backend, allocator))
    A = ismissing(l.continuing) ? missing : T(l.continuing, r.continuing, backend, allocator)
    return assemble_operator_halves(l.prepared, r.prepared, A, T, backend, allocator)
end
function AC2_hamiltonian(
        i::Int, below::_HAM_MPS_TYPES, O::MPOHamiltonian, above::_HAM_MPS_TYPES,
        envs::DMRGEnvironments;
        prepare::Bool = true, backend::AbstractBackend = envs.backend, allocator = envs.allocator,
    )
    @assert below === above "JordanMPO assumptions break"
    GL, GR = leftenv(envs, i, below), rightenv(envs, i + 1, below)
    if !(prepare && envs.nsites == 2)
        H = JordanMPO_AC2_Hamiltonian(GL, O[i], O[i + 1], GR; backend, allocator)
        return prepare ? prepare_operator!!(H) : H
    end
    l, r = lefthalf(envs, i), righthalf(envs, i + 1)
    T = prepared_operator_type(_AA_type(GL, envs.pairs[i], GR, backend, allocator))
    AA = ismissing(l.continuing) ? missing : T(l.continuing, r.continuing, backend, allocator)
    return assemble_operator_halves(l.prepared, r.prepared, AA, T, backend, allocator)
end
