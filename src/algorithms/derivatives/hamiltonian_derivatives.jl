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
    GL = leftenv(envs, site, below)
    GR = rightenv(envs, site, below)
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
    GL = leftenv(envs, site, below)
    GR = rightenv(envs, site + 1, below)
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
        @plansor backend = backend allocator = allocator CA_[-1 -2 -3; -4 -5 -6] ≔ C1[-1; -4 2] * A2[2 -2; -5 1] *
            GR2[-6 1; -3]
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

    # continuing - continuing
    AA = MPO_AC2_Hamiltonian(GL2, A1, A2, GR2, backend, allocator)

    S = spacetype(GL)
    M = storagetype(GL)
    O1 = tensormaptype(S, 1, 1, M)
    O2 = tensormaptype(S, 2, 2, M)
    O3 = tensormaptype(S, 3, 3, M)
    O4 = typeof(AA)

    if nonzero_length(A1) == 0 && nonzero_length(A2) == 0
        AA = missing
    else
        mask1 = falses(size(A1, 1), size(A1, 4))
        for I in nonzero_keys(A1)
            mask1[I[1], I[4]] = true
        end

        mask2 = falses(size(A2, 1), size(A2, 4))
        for I in nonzero_keys(A2)
            mask2[I[1], I[4]] = true
        end

        mask_left = transpose(mask1) * trues(size(mask1, 1))
        mask_right = mask2 * trues(size(mask2, 2))
        all(iszero, mask_left .* mask_right) && (AA = missing)
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
