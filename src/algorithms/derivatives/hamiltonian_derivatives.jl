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
        return _AC_hamiltonian_from_halves(site, below, operator, envs, backend, allocator)
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
    L = left_half(JordanMPO_AC_Hamiltonian, GL, W, samestate, backend, allocator)
    R = right_half(JordanMPO_AC_Hamiltonian, GR, W, samestate, backend, allocator)
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
        GR_E = _boundary(H.GR_E, C, 2)
        @plansor backend = backend allocator = allocator C[-1 -2; -3 -4] += H.D[-1; -3] * GR_E[-2; -4]
        missing
    elseif !ismissing(B)
        GL_I = _boundary(H.GL_I, B, 1)
        @plansor backend = backend allocator = allocator B[-1 -2; -3 -4] += GL_I[-1; -3] * H.D[-2; -4]
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

    return JordanMPO_AC_Hamiltonian{O1, O2, O3′}(D, I, E, C, B, A, H.GL_I, H.GR_E, backend, allocator)::JordanMPO_AC_Hamiltonian{O1, O2, O3′}
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
        return _AC2_hamiltonian_from_halves(site, below, operator, envs, backend, allocator)
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
    GL2, GR2 = GL[2:(end - 1)], GR[2:(end - 1)]
    O4 = _AA_type(GL2, pair, GR2, backend, allocator)
    # continuing - continuing, restricted to the channels that can actually contribute
    rows, mids, cols = pair.channels
    AA = isempty(mids) ? missing :
        MPO_AC2_Hamiltonian(GL2[rows], pair.A1, pair.A2, GR2[cols], backend, allocator)
    L = left_half(JordanMPO_AC2_Hamiltonian, GL, W1, pair, samestate, backend, allocator)
    R = right_half(JordanMPO_AC2_Hamiltonian, GR, W2, pair, samestate, backend, allocator)
    return assemble_operator_halves(L, R, AA, O4, backend, allocator)
end

function prepare_operator!!(
        H::JordanMPO_AC2_Hamiltonian{O1, O2, O3, O4}
    ) where {O1, O2, O3, O4}
    backend, allocator = H.backend, H.allocator

    CA::Union{Missing, O3} = H.CA
    AB::Union{Missing, O3} = H.AB

    CB::Union{Missing, O2} = if !ismissing(CA) && !ismissing(H.CB)
        GR_E = _boundary(H.GR_E, CA, 3)
        @plansor backend = backend allocator = allocator CA[-1 -2 -3; -4 -5 -6] += H.CB[-1 -2; -4 -5] * GR_E[-3; -6]
        missing
    elseif !ismissing(AB) && !ismissing(H.CB)
        GL_I = _boundary(H.GL_I, AB, 1)
        @plansor backend = backend allocator = allocator AB[-1 -2 -3; -4 -5 -6] += H.CB[-2 -3; -5 -6] * GL_I[-1; -4]
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
        GL_I = _boundary(H.GL_I, BE, 1)
        @plansor backend = backend allocator = allocator BE[-1 -2; -3 -4] += GL_I[-1; -3] * H.DE[-2; -4]
        missing
    elseif !ismissing(AB) && !ismissing(H.DE)
        GL_I = _boundary(H.GL_I, AB, 1)
        Id2 = id(storagetype(H.DE), space(AB, 3))
        @plansor backend = backend allocator = allocator AB[-1 -2 -3; -4 -5 -6] += GL_I[-1; -4] * H.DE[-2; -5] * Id2[-3; -6]
        missing
        # TODO: could also try in CA?
    else
        H.DE
    end

    # onsite right
    ID::Union{Missing, O1} = if !ismissing(IC) && !ismissing(H.ID)
        GR_E = _boundary(H.GR_E, IC, 2)
        @plansor backend = backend allocator = allocator IC[-1 -2; -3 -4] += H.ID[-1; -3] * GR_E[-2; -4]
        missing
    elseif !ismissing(CA) && !ismissing(H.ID)
        Id1 = TensorKit.id(storagetype(H.ID), space(CA, 1))
        GR_E = _boundary(H.GR_E, CA, 3)
        @plansor backend = backend allocator = allocator CA[-1 -2 -3; -4 -5 -6] += Id1[-1; -4] * H.ID[-2; -5] * GR_E[-3; -6]
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

    return JordanMPO_AC2_Hamiltonian{O1, O2, O3, O4′}(II, IC, ID, CB, CA, AB, AA, BE, DE, EE, H.GL_I, H.GR_E, backend, allocator)
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

Base.@assume_effects :foldable function jordan_pair_type(::Type{W}, ::Type{M}) where {W <: JordanMPOTensor, M}
    S = spacetype(W)
    T = tensormaptype(S, 2, 2, M)
    O32, O23 = tensormaptype(S, 3, 2, M), tensormaptype(S, 2, 3, M)
    U = SparseBlockTensorMap{O32, scalartype(O32), S, 3, 2, 5}
    V = SparseBlockTensorMap{O23, scalartype(O23), S, 2, 3, 5}
    A = SparseBlockTensorMap{tensormaptype(S, 2, 2, storagetype(W)), scalartype(W), S, 2, 2, 4}
    return JordanPairData{T, U, V, A}
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
    return jordan_pair_type(typeof(W1), M)(CB, AB, CA, channels, A1r, A2r)
end

_jordan_types(G) = _jordan_types(typeof(G))
function _jordan_types(::Type{G}) where {G <: AbstractTensorMap}
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
_not_started(GL, samestate) = samestate ? missing : removeunit(GL[1], 2)
_finished(GR, samestate) = samestate ? missing : transpose(removeunit(GR[end], 2))

# The terms of `JordanMPO_AC_Hamiltonian` and `JordanMPO_AC2_Hamiltonian` that depend on only
# one environment, excluding the continuing `A`/`AA` term. Each onsite or operator-only term
# is assigned to the half whose `prepare_operator!!` can fold it.
function left_half(::Type{<:JordanMPO_AC_Hamiltonian}, GL, W::JordanMPOTensor, samestate::Bool, backend, allocator)
    GL2 = GL[2:(end - 1)]
    O1, O2, _ = _jordan_types(GL)
    E = size(W, 1) > 1 ? removeunit(GL[end], 2) : missing
    D = nonzero_length(W.C) == 0 ? _onsite(W.D) : missing
    B = _ending(GL2, W.B, backend, allocator)
    return JordanMPO_AC_Hamiltonian{O1, O2, Missing}(
        D, missing, E, missing, B, missing, _not_started(GL, samestate), missing, backend, allocator
    )
end
function right_half(::Type{<:JordanMPO_AC_Hamiltonian}, GR, W::JordanMPOTensor, samestate::Bool, backend, allocator)
    GR2 = GR[2:(end - 1)]
    O1, O2, _ = _jordan_types(GR)
    I = size(W, 4) > 1 ? removeunit(GR[1], 2) : missing
    WC = W.C
    D = nonzero_length(WC) > 0 ? _onsite(W.D) : missing
    C = _starting(GR2, WC, backend, allocator)
    return JordanMPO_AC_Hamiltonian{O1, O2, Missing}(
        D, I, missing, C, missing, missing, missing, _finished(GR, samestate), backend, allocator
    )
end

function left_half(
        ::Type{<:JordanMPO_AC2_Hamiltonian}, GL, W::JordanMPOTensor, pair::JordanPairData, samestate::Bool,
        backend, allocator
    )
    GL2 = GL[2:(end - 1)]
    O1, O2, O3 = _jordan_types(GL)
    EE = size(W, 1) > 1 ? removeunit(GL[end], 2) : missing
    AB = if ismissing(pair.AB)
        missing
    else
        @plansor backend = backend allocator = allocator ab[-1 -2 -3; -4 -5 -6] := GL2[-1 1; -4] * pair.AB[1 -2 -3; -5 -6]
        only(ab)
    end
    # CB is folded into CA if available, otherwise into AB.
    CB = ismissing(pair.CA) ? pair.CB : missing
    return JordanMPO_AC2_Hamiltonian{O1, O2, O3, Missing}(
        missing, missing, missing, CB, missing, AB, missing,
        _ending(GL2, W.B, backend, allocator), _onsite(W.D), EE,
        _not_started(GL, samestate), missing, backend, allocator,
    )
end
function right_half(
        ::Type{<:JordanMPO_AC2_Hamiltonian}, GR, W::JordanMPOTensor, pair::JordanPairData, samestate::Bool,
        backend, allocator
    )
    GR2 = GR[2:(end - 1)]
    O1, O2, O3 = _jordan_types(GR)
    II = size(W, 4) > 1 ? transpose(removeunit(GR[1], 2)) : missing
    CA = if ismissing(pair.CA)
        missing
    else
        @plansor backend = backend allocator = allocator ca[-1 -2 -3; -4 -5 -6] := pair.CA[-1 -2; -4 -5 1] * GR2[-6 1; -3]
        only(ca)
    end
    CB = ismissing(CA) ? missing : pair.CB
    return JordanMPO_AC2_Hamiltonian{O1, O2, O3, Missing}(
        II, _starting(GR2, W.C, backend, allocator), _onsite(W.D), CB, CA,
        missing, missing, missing, missing, missing,
        missing, _finished(GR, samestate), backend, allocator,
    )
end

function assemble_operator_halves(L::JordanMPO_AC_Hamiltonian{O1, O2}, R, A, ::Type{O3}, backend, allocator) where {O1, O2, O3}
    D = ismissing(L.D) ? R.D : L.D
    return JordanMPO_AC_Hamiltonian{O1, O2, O3}(D, R.I, L.E, R.C, L.B, A, L.GL_I, R.GR_E, backend, allocator)
end
function assemble_operator_halves(L::JordanMPO_AC2_Hamiltonian{O1, O2, O3}, R, AA, ::Type{O4}, backend, allocator) where {O1, O2, O3, O4}
    CB = ismissing(L.CB) ? R.CB : L.CB
    return JordanMPO_AC2_Hamiltonian{O1, O2, O3, O4}(
        R.II, R.IC, R.ID, CB, R.CA, L.AB, AA, L.BE, L.DE, L.EE, L.GL_I, R.GR_E, backend, allocator,
    )
end

# Prepared halves for `FiniteEnvironments`
# -----------------------------------------
"""
    JordanHalf

One environment's prepared half of an effective `AC` or `AC2` Hamiltonian, which uses the
default backend and allocator. `continuing` is that side's fused environment for assembling
the continuing term, or `missing` if absent.
"""
struct JordanHalf{H, C <: AbstractTensorMap}
    prepared::H
    continuing::Union{Missing, C}
end

# Halves have no continuing term.
prepared_operator_type(::Type{Missing}) = Missing

function _default_backend(H::JordanMPO_AC_Hamiltonian{O1, O2, O3}) where {O1, O2, O3}
    return JordanMPO_AC_Hamiltonian{O1, O2, O3}(H.D, H.I, H.E, H.C, H.B, H.A, H.GL_I, H.GR_E)
end
function _default_backend(H::JordanMPO_AC2_Hamiltonian{O1, O2, O3, O4}) where {O1, O2, O3, O4}
    return JordanMPO_AC2_Hamiltonian{O1, O2, O3, O4}(
        H.II, H.IC, H.ID, H.CB, H.CA, H.AB, H.AA, H.BE, H.DE, H.EE, H.GL_I, H.GR_E
    )
end

function jordan_halftypes(::typeof(AC_hamiltonian), ::Type{G}, ::Type{W}) where {G, W}
    O1, O2, _ = _jordan_types(G)
    H = JordanMPO_AC_Hamiltonian{O1, O2, Missing, DefaultBackend, DefaultAllocator}
    L, R = halftypes(AC_hamiltonian, G, W)
    return JordanHalf{H, L}, JordanHalf{H, R}
end
function jordan_halftypes(::typeof(AC2_hamiltonian), ::Type{G}, ::Type{W}) where {G, W}
    O1, O2, O3 = _jordan_types(G)
    H = JordanMPO_AC2_Hamiltonian{O1, O2, O3, Missing, DefaultBackend, DefaultAllocator}
    L, R = halftypes(AC2_hamiltonian, G, W)
    return JordanHalf{H, L}, JordanHalf{H, R}
end

function environment_halves(O::MPOHamiltonian, ::Type{G}, N) where {G}
    W = eltype(O)
    (G <: MPSTensor && isconcretetype(W)) || return nothing
    AC = PreparedHalves{jordan_halftypes(AC_hamiltonian, G, W)..., G}(N)
    AC2 = PreparedHalves{jordan_halftypes(AC2_hamiltonian, G, W)..., G}(N)
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

function prepare_lefthalf(kind::typeof(AC_hamiltonian), O::MPOHamiltonian, envs, GL, i, backend, allocator)
    W = O[i]
    A = W.A
    half = left_half(JordanMPO_AC_Hamiltonian, GL, W, isnothing(envs.above), backend, allocator)
    continuing = nonzero_length(A) > 0 ? _prepare_GL_O(GL[2:(end - 1)], A, backend, allocator) : missing
    L, _ = jordan_halftypes(kind, eltype(envs.GLs), eltype(O))
    return L(_default_backend(prepare_operator!!(half)), continuing)
end
function prepare_righthalf(kind::typeof(AC_hamiltonian), O::MPOHamiltonian, envs, GR, i, backend, allocator)
    W = O[i]
    half = right_half(JordanMPO_AC_Hamiltonian, GR, W, isnothing(envs.above), backend, allocator)
    continuing = nonzero_length(W.A) > 0 ? TensorMap(GR[2:(end - 1)]) : missing
    _, R = jordan_halftypes(kind, eltype(envs.GRs), eltype(O))
    return R(_default_backend(prepare_operator!!(half)), continuing)
end
function prepare_lefthalf(kind::typeof(AC2_hamiltonian), O::MPOHamiltonian, envs, GL, i, backend, allocator)
    pair = pair_data!(envs, i)
    half = left_half(JordanMPO_AC2_Hamiltonian, GL, O[i], pair, isnothing(envs.above), backend, allocator)
    rows, mids, _ = pair.channels
    continuing = isempty(mids) ? missing : _prepare_GL_O(GL[2:(end - 1)][rows], pair.A1, backend, allocator)
    L, _ = jordan_halftypes(kind, eltype(envs.GLs), eltype(O))
    return L(_default_backend(prepare_operator!!(half)), continuing)
end
function prepare_righthalf(kind::typeof(AC2_hamiltonian), O::MPOHamiltonian, envs, GR, i, backend, allocator)
    pair = pair_data!(envs, i - 1)
    half = right_half(JordanMPO_AC2_Hamiltonian, GR, O[i], pair, isnothing(envs.above), backend, allocator)
    _, mids, cols = pair.channels
    continuing = isempty(mids) ? missing : _prepare_O_GR(pair.A2, GR[2:(end - 1)][cols], backend, allocator)
    _, R = jordan_halftypes(kind, eltype(envs.GRs), eltype(O))
    return R(_default_backend(prepare_operator!!(half)), continuing)
end

function _AC_hamiltonian_from_halves(site, below, O::MPOHamiltonian, envs, backend, allocator)
    G, W = eltype(envs.GLs), eltype(O)
    P = prepared_operator_type(MPO_AC_Hamiltonian{G, W, G, typeof(backend), typeof(allocator)})
    l = lefthalf(AC_hamiltonian, envs, site, below; backend, allocator)
    r = righthalf(AC_hamiltonian, envs, site, below; backend, allocator)
    A = ismissing(l.continuing) ? missing : P(l.continuing, r.continuing, backend, allocator)
    return assemble_operator_halves(l.prepared, r.prepared, A, P, backend, allocator)
end
function _AC2_hamiltonian_from_halves(site, below, O::MPOHamiltonian, envs, backend, allocator)
    G, W = eltype(envs.GLs), eltype(O)
    P = prepared_operator_type(MPO_AC2_Hamiltonian{G, W, W, G, typeof(backend), typeof(allocator)})
    l = lefthalf(AC2_hamiltonian, envs, site, below; backend, allocator)
    r = righthalf(AC2_hamiltonian, envs, site + 1, below; backend, allocator)
    AA = ismissing(l.continuing) ? missing : P(l.continuing, r.continuing, backend, allocator)
    return assemble_operator_halves(l.prepared, r.prepared, AA, P, backend, allocator)
end
