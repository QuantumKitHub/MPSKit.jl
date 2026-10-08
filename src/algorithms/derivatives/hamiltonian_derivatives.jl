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
    # block accessors recompute a fresh `SparseBlockTensorMap` on every access, so bind
    # them once and reuse the locals throughout
    WA, WB, WC, WD = W.A, W.B, W.C, W.D
    GL2 = GL[2:(end - 1)]
    GR2 = GR[2:(end - 1)]

    # onsite
    D = nonzero_length(WD) > 0 ? only(WD) : missing

    # not started
    I = size(W, 4) == 1 ? missing : removeunit(GR[1], 2)

    # finished
    E = size(W, 1) == 1 ? missing : removeunit(GL[end], 2)

    # starting
    C = if nonzero_length(WC) > 0
        @plansor backend = backend allocator = allocator starting[-1 -2; -3 -4] ≔ WC[-1; -3 1] * GR2[-4 1; -2]
        only(starting)
    else
        missing
    end

    # ending
    B = if nonzero_length(WB) > 0
        @plansor backend = backend allocator = allocator ending[-1 -2; -3 -4] ≔ GL2[-1 1; -3] * WB[1 -2; -4]
        only(ending)
    else
        missing
    end

    # continuing
    A = MPO_AC_Hamiltonian(GL2, WA, GR2, backend, allocator)

    # obtaining storagetype of environments since these should have already mixed
    # the types of the operator and state
    S = spacetype(GL)
    M = storagetype(GL)
    O1 = tensormaptype(S, 1, 1, M)
    O2 = tensormaptype(S, 2, 2, M)
    O3 = typeof(A)

    # specialization for nearest neighbours
    nonzero_length(WA) == 0 && (A = missing)

    return JordanMPO_AC_Hamiltonian{O1, O2, O3}(D, I, E, C, B, A, backend, allocator)
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
    # block accessors recompute a fresh `SparseBlockTensorMap` on every access, so bind
    # them once and reuse the locals throughout
    A1, B1, C1, D1 = W1.A, W1.B, W1.C, W1.D
    A2, B2, C2, D2 = W2.A, W2.B, W2.C, W2.D
    GL2 = GL[2:(end - 1)]
    GR2 = GR[2:(end - 1)]

    # not started
    II = size(W2, 4) == 1 ? missing : transpose(removeunit(GR[1], 2))

    # finished
    EE = size(W1, 1) == 1 ? missing : removeunit(GL[end], 2)

    # starting right
    IC = if nonzero_length(C2) > 0
        @plansor backend = backend allocator = allocator IC_[-1 -2; -3 -4] ≔ C2[-1; -3 1] * GR2[-4 1; -2]
        only(IC_)
    else
        missing
    end

    # onsite left
    DE = nonzero_length(D1) > 0 ? only(D1) : missing

    # onsite right
    ID = nonzero_length(D2) > 0 ? only(D2) : missing

    # starting left - ending right
    CB = if nonzero_length(C1) > 0 && nonzero_length(B2) > 0
        @plansor backend = backend allocator = allocator CB_[-1 -2; -3 -4] ≔ C1[-1; -3 1] * B2[1 -2; -4]
        # have to convert to complex if hamiltonian is real but states are complex
        scalartype(GL) <: Complex ? complex(only(CB_)) : only(CB_)
    else
        missing
    end

    # starting left - continuing right
    CA = if nonzero_length(C1) > 0 && nonzero_length(A2) > 0
        @plansor backend = backend allocator = allocator CA_[-1 -2 -3; -4 -5 -6] ≔ C1[-1; -4 1] * A2[1 -2; -5 2] *
            GR2[-6 2; -3]
        only(CA_)
    else
        missing
    end

    # continuing left - ending right
    AB = if nonzero_length(A1) > 0 && nonzero_length(B2) > 0
        @plansor backend = backend allocator = allocator AB_[-1 -2 -3; -4 -5 -6] ≔ GL2[-1 2; -4] * A1[2 -2; -5 1] *
            B2[1 -3; -6]
        only(AB_)
    else
        missing
    end

    # ending left
    BE = if nonzero_length(B1) > 0
        @plansor backend = backend allocator = allocator BE_[-1 -2; -3 -4] ≔ GL2[-1 2; -3] * B1[2 -2; -4]
        only(BE_)
    else
        missing
    end

    S = spacetype(GL)
    M = storagetype(GL)
    O1 = tensormaptype(S, 1, 1, M)
    O2 = tensormaptype(S, 2, 2, M)
    O3 = tensormaptype(S, 3, 3, M)
    # slicing preserves the block-tensor types, so `AA`'s type does not depend on whether the
    # channels end up restricted - no need to build a throwaway operator just to read it off
    O4 = MPO_AC2_Hamiltonian{
        typeof(GL2), typeof(A1), typeof(A2), typeof(GR2), typeof(backend), typeof(allocator),
    }

    # continuing - continuing, restricted to the channels that can actually contribute
    channels = _connected_channels(A1, A2)
    AA = if isempty(channels[2])
        missing
    else
        rows, mids, cols = channels
        MPO_AC2_Hamiltonian(
            GL2[rows], A1[rows, 1:1, 1:1, mids], A2[mids, 1:1, 1:1, cols], GR2[cols],
            backend, allocator
        )
    end

    return JordanMPO_AC2_Hamiltonian{O1, O2, O3, O4}(
        II, IC, ID,
        CB, CA,
        AB, AA,
        BE, DE, EE,
        backend, allocator
    )

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

# Cached Jordan operator data and directional contributions
# -------------------------------------------------------
"""
    JordanSiteData

Operator-only data for one site of a Jordan MPO, shared by every environment snapshot
in a solve. In the block form `[I C D; 0 A B; 0 0 I]`, `C` starts a term, `A` continues
it, and `B` ends it. `D` is the onsite term converted to the environment's storage type.

The boundary flags indicate which identity channels exist. The cached operator data
must remain read-only throughout the solve.
Absent onsite terms use `missing` while retaining a concrete field type.
"""
struct JordanSiteData{
        A <: MPOTensor,
        B <: AbstractTensorMap{<:Any, <:Any, 2, 1},
        C <: AbstractTensorMap{<:Any, <:Any, 1, 2},
        D <: MPSBondTensor,
    }
    "continuing block between internal MPO channels"
    A::A
    "ending block from internal channels to the finished channel"
    B::B
    "starting block from the unstarted channel to internal channels"
    C::C
    "onsite operator, or `missing` if absent"
    D::Union{Missing, D}
    "whether the outgoing unstarted channel exists separately from the finished channel"
    unstarted::Bool
    "whether the incoming finished channel exists separately from the unstarted channel"
    finished::Bool
    function JordanSiteData{A, B, C, D}(a, b, c, d, unstarted, finished) where {A, B, C, D}
        return new{A, B, C, D}(a, b, c, d, unstarted, finished)
    end
end
"""
    JordanPairData

Operator-only data for the adjacent sites `i` and `i + 1`. The `CB` contraction joins a
term starting at `i` to one ending at `i + 1`; it is independent of the MPS and computed
once. `channels = (rows, mids, cols)` identifies paths through both continuing blocks.
The restricted `A1` and `A2` retain only those paths when constructing the `AA` part of
an effective two-site Hamiltonian. Empty channel vectors mean there is no such path.
"""
struct JordanPairData{S <: JordanSiteData, T <: MPOTensor, A <: MPOTensor}
    "shared site data at `i`"
    left::S
    "shared site data at `i + 1`"
    right::S
    "operator-only `C_i * B_{i+1}` contraction, or `missing`"
    CB::Union{Missing, T}
    "connected incoming, intermediate, and outgoing internal channel indices"
    channels::NTuple{3, Vector{Int}}
    "left continuing block restricted to the connected rows and intermediate channels"
    A1::A
    "right continuing block restricted to the intermediate channels and connected columns"
    A2::A

    function JordanPairData{S, T, A}(left, right, CB, channels, A1, A2) where {S, T, A}
        return new{S, T, A}(left, right, CB, channels, A1, A2)
    end
end
"""
    JordanOperatorData

Fixed Jordan MPO metadata for a solve. Site and pair entries contain no MPS-dependent
contractions and remain shared when directional environment records are replaced.
Finite caches store `N - 1` pair entries. A periodic IDMRG2 cache will additionally need
the seam pair `(N, 1)` and boundary record updates after its explicit seam solves.
"""
struct JordanOperatorData{S <: JordanSiteData, P <: JordanPairData{S}}
    "one-site block metadata, indexed by site"
    sites::Vector{S}
    "two-site metadata, indexed by the left site of each bond"
    pairs::Vector{P}
end

"""
    JordanEnvironmentSide

One direction's contribution to an effective `AC` or `AC2` Hamiltonian at a particular
sweep position. Unlike the fixed `JordanSiteData` and `JordanPairData`, this combines their
operator blocks with the current `GL` or `GR`, so its contractions depend on the MPS.
`raw` contains the unfused partial operator without its `A`/`AA` term;
`prepared` is an independently prepared copy, including any onsite/`CB` folding.
Both are read-only snapshots: preparation must not fold terms into the cached raw data.

`continuing` stores the matrix needed to assemble the remaining `A`/`AA` term. For `AC`
it is fused `GL * A` on the left and dense `GR` on the right; for `AC2` it is fused,
channel-restricted `GL * A` or `A * GR`. Unfused one-site `GL * A` and `A * GR`
contractions are temporary inputs to pair preparation and are not retained in the record.
Environment transfers use the full MPO independently of these contributions. `continuing`
is `missing` when that contribution is absent; a local expansion refresh omits pair
preparation that will be replaced before use.
"""
struct JordanEnvironmentSide{
        H <: Union{JordanMPO_AC_Hamiltonian, JordanMPO_AC2_Hamiltonian},
        P <: Union{JordanMPO_AC_Hamiltonian, JordanMPO_AC2_Hamiltonian},
        C <: AbstractTensorMap,
    }
    "unprepared partial Hamiltonian, excluding the continuing `A`/`AA` term"
    raw::H
    "prepared partial Hamiltonian with independent mutable folding targets"
    prepared::P
    "fused matrix for continuing-term assembly, or `missing`"
    continuing::Union{Missing, C}
    function JordanEnvironmentSide{H, P, C}(raw, prepared, continuing) where {H, P, C}
        return new{H, P, C}(raw, prepared, continuing)
    end
end

function cache_operator_data(O::MPOHamiltonian, GL, N)
    S, M = spacetype(GL), storagetype(GL)
    O1, O2 = tensormaptype(S, 1, 1, M), tensormaptype(S, 2, 2, M)
    sites = map(1:N) do i
        W = O[i]
        A, B, C, D = W.A, W.B, W.C, W.D
        onsite = nonzero_length(D) > 0 ? convert(O1, only(D)) : missing
        JordanSiteData{typeof(A), typeof(B), typeof(C), O1}(
            A, B, C, onsite, size(W, 4) > 1, size(W, 1) > 1,
        )
    end
    pairs = map(1:(N - 1)) do i
        l, r = sites[i], sites[i + 1]
        CB = if nonzero_length(l.C) > 0 && nonzero_length(r.B) > 0
            @plansor cb[-1 -2; -3 -4] := l.C[-1; -3 1] * r.B[1 -2; -4]
            convert(O2, only(cb))
        else
            missing
        end
        channels = _connected_channels(l.A, r.A)
        A1, A2 = if isempty(channels[2])
            l.A, r.A
        else
            rows, mids, cols = channels
            l.A[rows, 1:1, 1:1, mids], r.A[mids, 1:1, 1:1, cols]
        end
        JordanPairData{eltype(sites), O2, typeof(A1)}(l, r, CB, channels, A1, A2)
    end
    return JordanOperatorData(sites, pairs)
end

# Preparing a partial operator may mutate its B/C or AB/CA fields. Copy only those
# fields once when publishing the record, retaining an independent unprepared view.
function copy_operator_side(H::JordanMPO_AC_Hamiltonian{O1, O2, O3}) where {O1, O2, O3}
    Hcopy = JordanMPO_AC_Hamiltonian{O1, O2, O3}(
        H.D, H.I, H.E, ismissing(H.C) ? missing : copy(H.C),
        ismissing(H.B) ? missing : copy(H.B), H.A, H.backend, H.allocator,
    )
    return Hcopy
end
function copy_operator_side(H::JordanMPO_AC2_Hamiltonian{O1, O2, O3, O4}) where {O1, O2, O3, O4}
    Hcopy = JordanMPO_AC2_Hamiltonian{O1, O2, O3, O4}(
        H.II, ismissing(H.IC) ? missing : copy(H.IC), H.ID, H.CB,
        ismissing(H.CA) ? missing : copy(H.CA), ismissing(H.AB) ? missing : copy(H.AB),
        H.AA, ismissing(H.BE) ? missing : copy(H.BE), H.DE, H.EE, H.backend, H.allocator,
    )
    return Hcopy
end

prepare_operator_side(H) = prepare_operator!!(copy_operator_side(H))

function _contract_GL_O(GL, O, backend, allocator)
    @plansor backend = backend allocator = allocator L[-1 -2 -3; -4 -5] := GL[-1 1; -4] * O[1 -2; -5 -3]
    return L
end
function _contract_O_GR(O, GR, backend, allocator)
    @plansor backend = backend allocator = allocator R[-1 -2; -4 -5 -3] := O[-3 -5; -2 1] * GR[-1 1; -4]
    return R
end
function _fuse_cached_environment(t, nout, nin, backend, allocator)
    cp = allocator_checkpoint!(allocator)
    result = _fuse_env(t, nout, nin, backend, allocator)
    allocator_reset!(allocator, cp)
    return result
end
# Build raw directional ingredients and the unfused continuing contraction once.
# AC and AC2 preparation consume these independently; pair-only solves skip AC fusion.
function left_AC_ingredients(GL, site, backend, allocator)
    GL2 = GL[2:(end - 1)]
    E = site.finished ? removeunit(GL[end], 2) : missing
    B = if nonzero_length(site.B) > 0
        @plansor backend = backend allocator = allocator b[-1 -2; -3 -4] := GL2[-1 1; -3] * site.B[1 -2; -4]
        only(b)
    else
        missing
    end
    # Match the existing folding preference: onsite into C, then B, otherwise standalone.
    D = nonzero_length(site.C) == 0 ? site.D : missing
    O1, O2 = tensormaptype(spacetype(GL), 1, 1, storagetype(GL)), tensormaptype(spacetype(GL), 2, 2, storagetype(GL))
    O3 = typeof(MPO_AC_Hamiltonian(GL2, site.A, GL2, backend, allocator))
    raw = JordanMPO_AC_Hamiltonian{O1, O2, O3}(D, missing, E, missing, B, missing, backend, allocator)
    contraction = nonzero_length(site.A) > 0 ? SparseBlockTensorMap(_contract_GL_O(GL2, site.A, backend, allocator)) : missing
    return raw, contraction
end
function prepare_left_AC(raw::JordanMPO_AC_Hamiltonian{O1, O2}, contraction, backend, allocator) where {O1, O2}
    continuing = ismissing(contraction) ? missing : _fuse_cached_environment(contraction, 1, 2, backend, allocator)
    prepared = prepare_operator_side(raw)
    return JordanEnvironmentSide{typeof(raw), typeof(prepared), O2}(raw, prepared, continuing)
end
function right_AC_ingredients(GR, site, backend, allocator; prepare_pair::Bool = false)
    GR2 = GR[2:(end - 1)]
    I = site.unstarted ? removeunit(GR[1], 2) : missing
    C = if nonzero_length(site.C) > 0
        @plansor backend = backend allocator = allocator c[-1 -2; -3 -4] := site.C[-1; -3 1] * GR2[-4 1; -2]
        only(c)
    else
        missing
    end
    D = nonzero_length(site.C) > 0 ? site.D : missing
    S, M = spacetype(GR), storagetype(GR)
    O1, O2 = tensormaptype(S, 1, 1, M), tensormaptype(S, 2, 2, M)
    O3 = typeof(MPO_AC_Hamiltonian(GR2, site.A, GR2, backend, allocator))
    raw = JordanMPO_AC_Hamiltonian{O1, O2, O3}(D, I, missing, C, missing, missing, backend, allocator)
    contraction = prepare_pair && nonzero_length(site.A) > 0 ? SparseBlockTensorMap(_contract_O_GR(site.A, GR2, backend, allocator)) : missing
    return raw, contraction, GR2
end
function prepare_right_AC(raw, GR2, site, backend, allocator)
    continuing = nonzero_length(site.A) > 0 ? TensorMap(GR2) : missing
    prepared = prepare_operator_side(raw)
    C = tensormaptype(spacetype(GR2), 2, 1, storagetype(GR2))
    return JordanEnvironmentSide{typeof(raw), typeof(prepared), C}(raw, prepared, continuing)
end

function prepare_left_AC2(GL, pair, ingredients, contraction, backend, allocator)
    GL2 = GL[2:(end - 1)]
    l, r = pair.left, pair.right
    EE, BE = ingredients.E, ingredients.B
    AB = if nonzero_length(l.A) > 0 && nonzero_length(r.B) > 0
        LA = contraction
        @plansor backend = backend allocator = allocator ab[-1 -2 -3; -4 -5 -6] := LA[-1 -2 1; -4 -5] * r.B[1 -3; -6]
        only(ab)
    else
        missing
    end
    # CB is folded into CA if available, otherwise into AB.
    has_CA = nonzero_length(l.C) > 0 && nonzero_length(r.A) > 0
    CB = has_CA ? missing : pair.CB
    S, M = spacetype(GL), storagetype(GL)
    O1, O2, O3 = tensormaptype(S, 1, 1, M), tensormaptype(S, 2, 2, M), tensormaptype(S, 3, 3, M)
    O4 = typeof(MPO_AC2_Hamiltonian(GL2, pair.A1, pair.A2, GL2, backend, allocator))
    raw = JordanMPO_AC2_Hamiltonian{O1, O2, O3, O4}(
        missing, missing, missing, CB, missing, AB, missing, BE, l.D, EE, backend, allocator,
    )
    continuing = if isempty(pair.channels[2])
        missing
    else
        _, mids, _ = pair.channels
        # Restrict after contracting: rows with no edge to mids contributed zero already.
        LA = contraction[1:1, 1:1, mids, 1:1, 1:1]
        _fuse_cached_environment(LA, 1, 2, backend, allocator)
    end
    prepared = prepare_operator_side(raw)
    return JordanEnvironmentSide{typeof(raw), typeof(prepared), O2}(raw, prepared, continuing)
end
function prepare_right_AC2(GR, pair, ingredients, contraction, backend, allocator)
    GR2 = GR[2:(end - 1)]
    l, r = pair.left, pair.right
    II = ismissing(ingredients.I) ? missing : transpose(ingredients.I)
    IC = ingredients.C
    CA = if nonzero_length(l.C) > 0 && nonzero_length(r.A) > 0
        AR = contraction
        @plansor backend = backend allocator = allocator ca[-1 -2 -3; -4 -5 -6] := l.C[-1; -4 1] * AR[-6 -5; -3 -2 1]
        only(ca)
    else
        missing
    end
    CB = ismissing(CA) ? missing : pair.CB
    S, M = spacetype(GR), storagetype(GR)
    O1, O2, O3 = tensormaptype(S, 1, 1, M), tensormaptype(S, 2, 2, M), tensormaptype(S, 3, 3, M)
    O4 = typeof(MPO_AC2_Hamiltonian(GR2, pair.A1, pair.A2, GR2, backend, allocator))
    raw = JordanMPO_AC2_Hamiltonian{O1, O2, O3, O4}(
        II, IC, r.D, CB, CA, missing, missing, missing, missing, missing, backend, allocator,
    )
    continuing = if isempty(pair.channels[2])
        missing
    else
        _, mids, _ = pair.channels
        AR = contraction[1:1, 1:1, 1:1, 1:1, mids]
        _fuse_cached_environment(AR, 2, 1, backend, allocator)
    end
    prepared = prepare_operator_side(raw)
    return JordanEnvironmentSide{typeof(raw), typeof(prepared), O2}(raw, prepared, continuing)
end

function prepare_left_environment(GL, data::JordanOperatorData, i, backend, allocator; one_site::Bool = true, two_site::Bool = true, local_only::Bool = false)
    prepare_pair = two_site && !local_only && i < length(data.sites)
    one_site || prepare_pair || return GL, missing, missing
    ingredients, contraction = left_AC_ingredients(GL, data.sites[i], backend, allocator)
    ac = one_site ? prepare_left_AC(ingredients, contraction, backend, allocator) : missing
    ac2 = prepare_pair ? prepare_left_AC2(GL, data.pairs[i], ingredients, contraction, backend, allocator) : missing
    return GL, ac, ac2
end
function prepare_right_environment(GR, data::JordanOperatorData, i, backend, allocator; one_site::Bool = true, two_site::Bool = true, local_only::Bool = false)
    prepare_pair = two_site && !local_only && i > 1
    one_site || prepare_pair || return GR, missing, missing
    ingredients, contraction, GR2 = right_AC_ingredients(GR, data.sites[i], backend, allocator; prepare_pair)
    ac = one_site ? prepare_right_AC(ingredients, GR2, data.sites[i], backend, allocator) : missing
    ac2 = prepare_pair ? prepare_right_AC2(GR, data.pairs[i - 1], ingredients, contraction, backend, allocator) : missing
    return GR, ac, ac2
end

function AC_hamiltonian(
        i::Int, below::_HAM_MPS_TYPES, O::MPOHamiltonian, above::_HAM_MPS_TYPES,
        cache::DMRGSweepCache;
        prepare::Bool = true, backend::AbstractBackend = cache.backend, allocator = cache.allocator,
    )
    @assert below === above "JordanMPO assumptions break"
    GL, GR = leftenv(cache, i, below), rightenv(cache, i, below)
    if !cache.one_site
        # An explicit AC query remains available in a pair-only solve, without
        # publishing an unused one-site record on every environment update.
        H = JordanMPO_AC_Hamiltonian(GL, O[i], GR; backend, allocator)
        return prepare ? prepare_operator!!(H) : H
    end
    l, r = cache.left[i].one_site, cache.right[i].one_site
    L, R = prepare ? (l.prepared, r.prepared) : (copy_operator_side(l.raw), copy_operator_side(r.raw))
    rawtype = MPO_AC_Hamiltonian{typeof(GL), typeof(cache.operator_data.sites[i].A), typeof(GR), typeof(backend), typeof(allocator)}
    continuing_type = prepare ? prepared_operator_type(rawtype) : rawtype
    A = if ismissing(l.continuing)
        missing
    elseif prepare
        continuing_type(l.continuing, r.continuing, backend, allocator)
    else
        MPO_AC_Hamiltonian(GL[2:(end - 1)], cache.operator_data.sites[i].A, GR[2:(end - 1)], backend, allocator)
    end
    return assemble_operator_sides(L, R, A, continuing_type, backend, allocator)
end
function AC2_hamiltonian(
        i::Int, below::_HAM_MPS_TYPES, O::MPOHamiltonian, above::_HAM_MPS_TYPES,
        cache::DMRGSweepCache;
        prepare::Bool = true, backend::AbstractBackend = cache.backend, allocator = cache.allocator,
    )
    @assert below === above "JordanMPO assumptions break"
    GL, GR = leftenv(cache, i, below), rightenv(cache, i + 1, below)
    l, r = cache.left[i].two_site, cache.right[i + 1].two_site
    L, R = prepare ? (l.prepared, r.prepared) : (copy_operator_side(l.raw), copy_operator_side(r.raw))
    pair = cache.operator_data.pairs[i]
    rawtype = MPO_AC2_Hamiltonian{typeof(GL), typeof(pair.A1), typeof(pair.A2), typeof(GR), typeof(backend), typeof(allocator)}
    continuing_type = prepare ? prepared_operator_type(rawtype) : rawtype
    AA = if ismissing(l.continuing)
        missing
    elseif prepare
        continuing_type(l.continuing, r.continuing, backend, allocator)
    else
        rows, _, cols = pair.channels
        MPO_AC2_Hamiltonian(GL[2:(end - 1)][rows], pair.A1, pair.A2, GR[2:(end - 1)][cols], backend, allocator)
    end
    return assemble_operator_sides(L, R, AA, continuing_type, backend, allocator)
end

function assemble_operator_sides(L::JordanMPO_AC_Hamiltonian{O1, O2}, R, A, ::Type{O3}, backend, allocator) where {O1, O2, O3}
    D = ismissing(L.D) ? R.D : L.D
    return JordanMPO_AC_Hamiltonian{O1, O2, O3}(D, R.I, L.E, R.C, L.B, A, backend, allocator)
end
function assemble_operator_sides(L::JordanMPO_AC2_Hamiltonian{O1, O2, O3}, R, AA, ::Type{O4}, backend, allocator) where {O1, O2, O3, O4}
    CB = ismissing(L.CB) ? R.CB : L.CB
    return JordanMPO_AC2_Hamiltonian{O1, O2, O3, O4}(
        R.II, R.IC, R.ID, CB, R.CA, L.AB, AA, L.BE, L.DE, L.EE, backend, allocator,
    )
end
