const _HAM_MPS_TYPES = Union{
    FiniteMPS{<:MPSTensor},
    WindowMPS{<:MPSTensor},
    InfiniteMPS{<:MPSTensor},
}

# Single site derivative
# ----------------------
"""
    JordanMPO_AC_Hamiltonian{O1,O2,O3}

Efficient operator for representing the single-site derivative of a `MPOHamiltonian` sandwiched between two MPSs.
In particular, this operator aims to make maximal use of the structure of the `MPOHamiltonian` to reduce the number of operations required to apply the operator to a tensor.
"""
struct JordanMPO_AC_Hamiltonian{O1, O2, O3} <: DerivativeOperator
    D::Union{O1, Missing} # onsite
    I::Union{O1, Missing} # not started
    E::Union{O1, Missing} # finished
    C::Union{O2, Missing} # starting
    B::Union{O2, Missing} # ending
    A::Union{O3, Missing} # continuing

    function JordanMPO_AC_Hamiltonian{O1, O2, O3}(
            D::Union{O1, Missing}, I::Union{O1, Missing}, E::Union{O1, Missing},
            C::Union{O2, Missing}, B::Union{O2, Missing}, A::Union{O3, Missing}
        ) where {O1, O2, O3}
        return new{O1, O2, O3}(D, I, E, C, B, A)
    end
end

function AC_hamiltonian(
        site::Int, below::_HAM_MPS_TYPES, operator::MPOHamiltonian, above::_HAM_MPS_TYPES, envs
    )
    @assert below === above "JordanMPO assumptions break"
    GL = leftenv(envs, site, below)
    GR = rightenv(envs, site, below)
    W = operator[site]
    return JordanMPO_AC_Hamiltonian(GL, W, GR)
end

function JordanMPO_AC_Hamiltonian(GL::MPSTensor, W::JordanMPOTensor, GR::MPSTensor)
    # onsite
    D = nonzero_length(W.D) > 0 ? only(W.D) : missing

    # not started
    I = size(W, 4) == 1 ? missing : removeunit(GR[1], 2)

    # finished
    E = size(W, 1) == 1 ? missing : removeunit(GL[end], 2)

    # starting
    C = if nonzero_length(W.C) > 0
        GR_2 = GR[2:(end - 1)]
        @plansor starting[-1 -2; -3 -4] ≔ W.C[-1; -3 1] * GR_2[-4 1; -2]
        only(starting)
    else
        missing
    end

    # ending
    B = if nonzero_length(W.B) > 0
        GL_2 = GL[2:(end - 1)]
        @plansor ending[-1 -2; -3 -4] ≔ GL_2[-1 1; -3] * W.B[1 -2; -4]
        only(ending)
    else
        missing
    end

    # continuing
    A = MPO_AC_Hamiltonian(GL[2:(end - 1)], W.A, GR[2:(end - 1)])

    # obtaining storagetype of environments since these should have already mixed
    # the types of the operator and state
    S = spacetype(GL)
    M = storagetype(GL)
    O1 = tensormaptype(S, 1, 1, M)
    O2 = tensormaptype(S, 2, 2, M)
    O3 = typeof(A)

    # specialization for nearest neighbours
    nonzero_length(W.A) == 0 && (A = missing)

    return JordanMPO_AC_Hamiltonian{O1, O2, O3}(D, I, E, C, B, A)
end

function prepare_operator!!(
        H::JordanMPO_AC_Hamiltonian{O1, O2, O3}, backend::AbstractBackend, allocator
    ) where {O1, O2, O3}
    C = H.C
    B = H.B

    # onsite
    D = if ismissing(H.D)
        missing
    elseif !ismissing(C)
        Id = TensorKit.id(storagetype(C), space(C, 2))
        @plansor C[-1 -2; -3 -4] += H.D[-1; -3] * Id[-2; -4]
        missing
    elseif !ismissing(B)
        Id = TensorKit.id(storagetype(B), space(B, 1))
        @plansor B[-1 -2; -3 -4] += Id[-1; -3] * H.D[-2; -4]
        missing
    else
        W.D
    end

    # not_started
    I = if ismissing(H.I)
        missing
    elseif !ismissing(C)
        Id = id(storagetype(C), space(C, 1))
        @plansor C[-1 -2; -3 -4] += Id[-1; -3] * H.I[-4; -2]
        missing
    else
        H.I
    end

    # finished
    E = if ismissing(H.E)
        missing
    elseif !ismissing(B)
        Id = id(storagetype(B), space(B, 2))
        @plansor B[-1 -2; -3 -4] += H.E[-1; -3] * Id[-2; -4]
        missing
    else
        H.E
    end

    O3′ = Core.Compiler.return_type(prepare_operator!!, Tuple{O3, typeof(backend), typeof(allocator)})
    A = ismissing(H.A) ? H.A : prepare_operator!!(H.A, backend, allocator)

    return JordanMPO_AC_Hamiltonian{O1, O2, O3′}(D, I, E, C, B, A)
end


# Two site derivative
# -------------------
"""
    JordanMPO_AC2_Hamiltonian{O1,O2,O3,O4}

Efficient operator for representing the single-site derivative of a `MPOHamiltonian` sandwiched between two MPSs.
In particular, this operator aims to make maximal use of the structure of the `MPOHamiltonian` to reduce the number of operations required to apply the operator to a tensor.
"""
struct JordanMPO_AC2_Hamiltonian{O1, O2, O3, O4} <: DerivativeOperator
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

    function JordanMPO_AC2_Hamiltonian{O1, O2, O3, O4}(
            II::Union{O1, Missing}, IC::Union{O2, Missing}, ID::Union{O1, Missing},
            CB::Union{O2, Missing}, CA::Union{O3, Missing},
            AB::Union{O3, Missing}, AA::Union{O4, Missing},
            BE::Union{O2, Missing}, DE::Union{O1, Missing}, EE::Union{O1, Missing}
        ) where {O1, O2, O3, O4}
        return new{O1, O2, O3, O4}(II, IC, ID, CB, CA, AB, AA, BE, DE, EE)
    end
end

function AC2_hamiltonian(
        site::Int, below::_HAM_MPS_TYPES, operator::MPOHamiltonian, above::_HAM_MPS_TYPES, envs
    )
    @assert below === above "JordanMPO assumptions break"
    GL = leftenv(envs, site, below)
    GR = rightenv(envs, site + 1, below)
    W1, W2 = operator[site], operator[site + 1]
    return JordanMPO_AC2_Hamiltonian(GL, W1, W2, GR)
end

function JordanMPO_AC2_Hamiltonian(GL::MPSTensor, W1::JordanMPOTensor, W2::JordanMPOTensor, GR::MPSTensor)
    # not started
    II = size(W2, 4) == 1 ? missing : transpose(removeunit(GR[1], 2))

    # finished
    EE = size(W1, 1) == 1 ? missing : removeunit(GL[end], 2)

    # starting right
    IC = if nonzero_length(W2.C) > 0
        @plansor IC_[-1 -2; -3 -4] ≔ W2.C[-1; -3 1] * GR[2:(end - 1)][-4 1; -2]
        only(IC_)
    else
        missing
    end

    # onsite left
    DE = nonzero_length(W1.D) > 0 ? only(W1.D) : missing

    # onsite right
    ID = nonzero_length(W2.D) > 0 ? only(W2.D) : missing

    # starting left - ending right
    CB = if nonzero_length(W1.C) > 0 && nonzero_length(W2.B) > 0
        @plansor CB_[-1 -2; -3 -4] ≔ W1.C[-1; -3 1] * W2.B[1 -2; -4]
        # have to convert to complex if hamiltonian is real but states are complex
        scalartype(GL) <: Complex ? complex(only(CB_)) : only(CB_)
    else
        missing
    end

    # starting left - continuing right
    CA = if nonzero_length(W1.C) > 0 && nonzero_length(W2.A) > 0
        @plansor CA_[-1 -2 -3; -4 -5 -6] ≔ W1.C[-1; -4 2] * W2.A[2 -2; -5 1] *
            GR[2:(end - 1)][-6 1; -3]
        only(CA_)
    else
        missing
    end

    # continuing left - ending right
    AB = if nonzero_length(W1.A) > 0 && nonzero_length(W2.B) > 0
        @plansor AB_[-1 -2 -3; -4 -5 -6] ≔ GL[2:(end - 1)][-1 2; -4] * W1.A[2 -2; -5 1] *
            W2.B[1 -3; -6]
        only(AB_)
    else
        missing
    end

    # ending left
    BE = if nonzero_length(W1.B) > 0
        @plansor BE_[-1 -2; -3 -4] ≔ GL[2:(end - 1)][-1 2; -3] * W1.B[2 -2; -4]
        only(BE_)
    else
        missing
    end

    # continuing - continuing
    AA = MPO_AC2_Hamiltonian(GL[2:(end - 1)], W1.A, W2.A, GR[2:(end - 1)])

    S = spacetype(GL)
    M = storagetype(GL)
    O1 = tensormaptype(S, 1, 1, M)
    O2 = tensormaptype(S, 2, 2, M)
    O3 = tensormaptype(S, 3, 3, M)
    O4 = typeof(AA)

    if nonzero_length(W1.A) == 0 && nonzero_length(W2.A) == 0
        AA = missing
    else
        mask1 = falses(size(W1.A, 1), size(W1.A, 4))
        for I in nonzero_keys(W1.A)
            mask1[I[1], I[4]] = true
        end

        mask2 = falses(size(W2.A, 1), size(W2.A, 4))
        for I in nonzero_keys(W2.A)
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
        BE, DE, EE
    )

end

function prepare_operator!!(
        H::JordanMPO_AC2_Hamiltonian{O1, O2, O3, O4}, backend::AbstractBackend, allocator
    ) where {O1, O2, O3, O4}

    CA = H.CA
    AB = H.AB

    CB = if !ismissing(CA) && !ismissing(H.CB)
        Id = TensorKit.id(storagetype(H.CB), space(CA, 3))
        @plansor CA[-1 -2 -3; -4 -5 -6] += H.CB[-1 -2; -4 -5] * Id[-3; -6]
        missing
    elseif !ismissing(AB) && !ismissing(H.CB)

    else
        H.CB
    end

    # starting right
    IC = if !ismissing(CA) && !ismissing(H.IC)
        Id = TensorKit.id(storagetype(H.IC), space(CA, 1))
        @plansor CA[-1 -2 -3; -4 -5 -6] += Id[-1; -4] * H.IC[ -2 -3; -5 -6]
        missing
    else
        H.IC
    end

    # ending left
    BE = if !ismissing(AB) && !ismissing(H.BE)
        Id = TensorKit.id(storagetype(H.BE), space(AB, 3))
        @plansor AB[-1 -2 -3; -4 -5 -6] += H.BE[-1 -2; -4 -5] * Id[-3; -6]
        missing
    else
        H.BE
    end

    # onsite left
    DE = if !ismissing(BE) && !ismissing(H.DE)
        Id = TensorKit.id(storagetype(H.DE), space(BE, 1))
        @plansor BE[-1 -2; -3 -4] += Id[-1; -3] * H.DE[-2; -4]
        missing
    elseif !ismissing(AB) && !ismissing(H.DE)
        Id1 = id(storagetype(H.DE), space(AB, 1))
        Id2 = id(storagetype(H.DE), space(AB, 3))
        @plansor AB[-1 -2 -3; -4 -5 -6] += Id1[-1; -4] * H.DE[-2; -5] * Id2[-3; -6]
        missing
        # TODO: could also try in CA?
    else
        H.DE
    end

    # onsite right
    ID = if !ismissing(IC) && !ismissing(H.ID)
        Id = TensorKit.id(storagetype(H.ID), space(IC, 2))
        @plansor IC[-1 -2; -3 -4] += H.ID[-1; -3] * Id[-2; -4]
        missing
    elseif !ismissing(CA) && !ismissing(H.ID)
        Id1 = TensorKit.id(storagetype(H.ID), space(CA, 1))
        Id2 = TensorKit.id(storagetype(H.ID), space(CA, 3))
        @plansor CA[-1 -2 -3; -4 -5 -6] += Id1[-1; -4] * H.ID[-2; -5] * Id2[-3; -6]
        missing
    else
        H.ID
    end

    # finished
    II = if !ismissing(IC) && !ismissing(H.II)
        I = id(storagetype(H.II), space(IC, 1))
        @plansor IC[-1 -2; -3 -4] += I[-1; -3] * H.II[-2; -4]
        II = missing
    elseif !ismissing(CA) && !ismissing(H.II)
        I = id(storagetype(H.II), space(CA, 1) ⊗ space(CA, 2))
        @plansor CA[-1 -2 -3; -4 -5 -6] += I[-1 -2; -4 -5] * H.II[-3; -6]
        II = missing
    else
        H.II
    end

    # unstarted
    EE = if !ismissing(BE) && !ismissing(H.EE)
        I = id(storagetype(H.EE), space(BE, 2))
        @plansor BE[-1 -2; -3 -4] += H.EE[-1; -3] * I[-2; -4]
        EE = missing
    elseif !ismissing(AB) && !ismissing(H.EE)
        I = id(storagetype(H.EE), space(AB, 2) ⊗ space(AB, 3))
        @plansor AB[-1 -2 -3; -4 -5 -6] += H.EE[-1; -4] * I[-2 -3; -5 -6]
        EE = missing
    else
        H.EE
    end

    O4′ = Core.Compiler.return_type(prepare_operator!!, Tuple{O4, typeof(backend), typeof(allocator)})
    AA = prepare_operator!!(H.AA, backend, allocator)

    return JordanMPO_AC2_Hamiltonian{O1, O2, O3, O4′}(II, IC, ID, CB, CA, AB, AA, BE, DE, EE)
end

# Actions
# -------
function (H::JordanMPO_AC_Hamiltonian)(x::MPSTensor)
    y = ismissing(H.A) ? zerovector(x) : H.A(x)

    ismissing(H.D) || @plansor y[-1 -2; -3] += x[-1 1; -3] * H.D[-2; 1]
    ismissing(H.E) || @plansor y[-1 -2; -3] += H.E[-1; 1] * x[1 -2; -3]
    ismissing(H.I) || @plansor y[-1 -2; -3] += x[-1 -2; 1] * H.I[1; -3]
    ismissing(H.C) || @plansor y[-1 -2; -3] += x[-1 2; 1] * H.C[-2 -3; 2 1]
    ismissing(H.B) || @plansor y[-1 -2; -3] += H.B[-1 -2; 1 2] * x[1 2; -3]

    return y
end

function (H::JordanMPO_AC2_Hamiltonian)(x::MPOTensor)
    y = ismissing(H.AA) ? zerovector(x) : H.AA(x)

    ismissing(H.II) || @plansor y[-1 -2; -3 -4] += x[-1 -2; 1 -4] * H.II[-3; 1]
    ismissing(H.IC) || @plansor y[-1 -2; -3 -4] += x[-1 -2; 1 2] * H.IC[-4 -3; 2 1]
    ismissing(H.ID) || @plansor y[-1 -2; -3 -4] += x[-1 -2; -3 1] * H.ID[-4; 1]
    ismissing(H.CB) || @plansor y[-1 -2; -3 -4] += x[-1 1; -3 2] * H.CB[-2 -4; 1 2]
    ismissing(H.CA) || @plansor y[-1 -2; -3 -4] += x[-1 1; 3 2] * H.CA[-2 -4 -3; 1 2 3]
    ismissing(H.AB) || @plansor y[-1 -2; -3 -4] += x[1 2; -3 3] * H.AB[-1 -2 -4; 1 2 3]
    ismissing(H.BE) || @plansor y[-1 -2; -3 -4] += x[1 2; -3 -4] * H.BE[-1 -2; 1 2]
    ismissing(H.DE) || @plansor y[-1 -2; -3 -4] += x[-1 1; -3 -4] * H.DE[-2; 1]
    ismissing(H.EE) || @plansor y[-1 -2; -3 -4] += x[1 -2; -3 -4] * H.EE[-1; 1]

    return y
end
