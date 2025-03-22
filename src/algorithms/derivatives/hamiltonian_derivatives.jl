"""
    JordanMPO_∂∂AC{O1,O2,O3}

Efficient operator for representing the single-site derivative of a `MPOHamiltonian` sandwiched between two MPSs.
In particular, this operator aims to make maximal use of the structure of the `MPOHamiltonian` to reduce the number of operations required to apply the operator to a tensor.
"""
struct JordanMPO_∂∂AC{O1,O2,O3}
    onsite::Union{O1,Missing}
    not_started::Union{O1,Missing}
    finished::Union{O1,Missing}
    starting::Union{O2,Missing}
    ending::Union{O2,Missing}
    continuing::O3
end
function JordanMPO_∂∂AC(onsite, not_started, finished, starting, ending, continuing)
    tensor = coalesce(onsite, not_started, finished, starting, ending)
    ismissing(tensor) && throw(ArgumentError("unable to determine type"))
    S = spacetype(tensor)
    M = storagetype(tensor)
    O1 = tensormaptype(S, 1, 1, M)
    O2 = tensormaptype(S, 2, 2, M)
    return JordanMPO_∂∂AC{O1,O2,typeof(continuing)}(onsite, not_started, finished, starting,
                                                    ending, continuing)
end

Base.:*(H::JordanMPO_∂∂AC, x) = H(x)

function ∂∂AC(pos::Int, mps, operator::MPOHamiltonian, envs)
    GL = leftenv(envs, pos, mps)
    GR = rightenv(envs, pos, mps)
    W = operator[pos]

    # starting
    if !isfinite(operator) || pos < length(operator)
        C = W[1, 1, 1, 2:(end - 1)]
        GR_2 = GR[2:(end - 1)]
        if nonzero_length(C) > 0 && nonzero_length(GR_2) > 0
            @plansor starting_[-1 -2; -3 -4] ≔ removeunit(C, 1)[-1; -3 1] *
                                               GR_2[-4 1; -2]
            starting = only(starting_)
        else
            starting = missing
        end
    else
        starting = missing
    end

    # ending
    if !isfinite(operator) || pos > 1
        B = W[2:(end - 1), 1, 1, end]
        GL_2 = GL[2:(end - 1)]
        if nonzero_length(B) > 0 && nonzero_length(GL_2) > 0
            @plansor ending_[-1 -2; -3 -4] ≔ GL_2[-1 1; -3] * removeunit(B, 4)[1 -2; -4]
            ending = nonzero_length(ending_) > 0 ? only(ending_) : missing
        else
            ending = missing
        end
    else
        ending = missing
    end

    # onsite
    if haskey(W, CartesianIndex(1, 1, 1, lastindex(W, 4)))
        if !ismissing(starting)
            D = removeunit(W[1, 1, 1, end], 1)
            @plansor starting[-1 -2; -3 -4] += D[-1; -3 1] * GR[end][-4 1; -2]
            onsite = missing
        elseif !ismissing(ending)
            D = removeunit(W[1, 1, 1, end], 4)
            @plansor ending[-1 -2; -3 -4] += GL[1][-1 1; -3] * D[1 -2; -4]
            onsite = missing
        else
            onsite = removeunit(removeunit(W[1, 1, 1, end], 4), 1)
        end
    else
        onsite = missing
    end

    # not_started
    if (!isfinite(operator) || pos > 1) && !ismissing(starting)
        I = id(storagetype(GR[1]), physicalspace(W))
        @plansor starting[-1 -2; -3 -4] += I[-1; -3] * removeunit(GR[1], 2)[-4; -2]
        not_started = missing
    else
        not_started = removeunit(GR[1], 2)
    end

    # finished
    if (!isfinite(operator) || pos < length(operator)) && !ismissing(ending)
        I = id(storagetype(GL[end]), physicalspace(W))
        @plansor ending[-1 -2; -3 -4] += removeunit(GL[end], 2)[-1; -3] * I[-2; -4]
        finished = missing
    else
        finished = removeunit(GL[end], 2)
    end

    # continuing
    A = W[2:(end - 1), 1, 1, 2:(end - 1)]
    continuing = (GL[2:(end - 1)], A, GR[2:(end - 1)])

    return JordanMPO_∂∂AC(onsite, not_started, finished, starting, ending, continuing)
end

function ∂∂AC(site::Int, mps, operator::MPOHamiltonian{<:JordanMPOTensor}, envs)
    GL = leftenv(envs, site, mps)
    GR = rightenv(envs, site, mps)
    W = operator[site]

    # starting
    if nonzero_length(W.C) > 0
        GR_2 = GR[2:(end - 1)]
        @plansor starting_[-1 -2; -3 -4] ≔ W.C[-1; -3 1] * GR_2[-4 1; -2]
        starting = only(starting_)
    else
        starting = missing
    end

    # ending
    if nonzero_length(W.B) > 0
        GL_2 = GL[2:(end - 1)]
        @plansor ending_[-1 -2; -3 -4] ≔ GL_2[-1 1; -3] * W.B[1 -2; -4]
        ending = only(ending_)
    else
        ending = missing
    end

    # onsite
    if nonzero_length(W.D) > 0
        if !ismissing(starting)
            @plansor starting[-1 -2; -3 -4] += W.D[-1; -3] * removeunit(GR[end], 2)[-4; -2]
            onsite = missing
        elseif !ismissing(ending)
            @plansor ending[-1 -2; -3 -4] += removeunit(GL[1], 2)[-1; -3] * W.D[-2; -4]
            onsite = missing
        else
            onsite = W.D
        end
    else
        onsite = missing
    end

    # not_started
    if (!isfinite(operator) || site < length(operator)) && !ismissing(starting)
        I = id(storagetype(GR[1]), physicalspace(W))
        @plansor starting[-1 -2; -3 -4] += I[-1; -3] * removeunit(GR[1], 2)[-4; -2]
        not_started = missing
    else
        not_started = removeunit(GR[1], 2)
    end

    # finished
    if (!isfinite(operator) || site > 1) && !ismissing(ending)
        I = id(storagetype(GL[end]), physicalspace(W))
        @plansor ending[-1 -2; -3 -4] += removeunit(GL[end], 2)[-1; -3] * I[-2; -4]
        finished = missing
    else
        finished = removeunit(GL[end], 2)
    end

    # continuing
    A = W.A
    continuing = (GL[2:(end - 1)], A, GR[2:(end - 1)])

    tensor = coalesce(onsite, not_started, finished, starting, ending)
    ismissing(tensor) && throw(ArgumentError("unable to determine type"))
    S = spacetype(tensor)
    M = storagetype(tensor)
    O1 = tensormaptype(S, 1, 1, M)
    O2 = tensormaptype(S, 2, 2, M)

    return JordanMPO_∂∂AC{O1,O2,typeof(continuing)}(onsite, not_started, finished, starting,
                                                    ending, continuing)
end

function (H::JordanMPO_∂∂AC)(x::MPSTensor)
    y = zerovector!(similar(x))

    if !ismissing(H.onsite)
        @plansor y[-1 -2; -3] += x[-1 1; -3] * H.onsite[-2; 1]
    end

    if !ismissing(H.finished)
        @plansor y[-1 -2; -3] += H.finished[-1; 1] * x[1 -2; -3]
    end

    if !ismissing(H.not_started)
        @plansor y[-1 -2; -3] += x[-1 -2; 1] * H.not_started[1; -3]
    end

    if !ismissing(H.starting)
        @plansor y[-1 -2; -3] += x[-1 2; 1] * H.starting[-2 -3; 2 1]
    end

    if !ismissing(H.ending)
        @plansor y[-1 -2; -3] += H.ending[-1 -2; 1 2] * x[1 2; -3]
    end

    GL, A, GR = H.continuing
    if nonzero_length(A) > 0
        @plansor y[-1 -2; -3] += GL[-1 5; 4] * x[4 2; 1] * A[5 -2; 2 3] * GR[1 3; -3]
    end

    return y
end

"""
    JordanMPO_∂∂AC2{O1,O2,O3,O4}

Efficient operator for representing the single-site derivative of a `MPOHamiltonian` sandwiched between two MPSs.
In particular, this operator aims to make maximal use of the structure of the `MPOHamiltonian` to reduce the number of operations required to apply the operator to a tensor.
"""
struct JordanMPO_∂∂AC2{O1,O2,O3,O4}
    II::Union{O1,Missing} # not_started
    IC::Union{O2,Missing} # starting right
    ID::Union{O1,Missing} # onsite right
    CB::Union{O2,Missing} # starting left - ending right
    CA::Union{O3,Missing} # starting left - continuing right
    AB::Union{O3,Missing} # continuing left - ending right
    AA::O4 # continuing left - continuing right
    BE::Union{O2,Missing} # ending left
    DE::Union{O1,Missing} # onsite left
    EE::Union{O1,Missing} # finished
end
function JordanMPO_∂∂AC2(II, IC, ID, CB, CA, AB, AA, BE, DE, EE)
    tensor = coalesce(II, IC, ID, CB, CA, AB, AA, BE, DE, EE)
    ismissing(tensor) && throw(ArgumentError("unable to determine type"))
    S = spacetype(tensor)
    M = storagetype(tensor)
    O1 = tensormaptype(S, 1, 1, M)
    O2 = tensormaptype(S, 2, 2, M)
    O3 = tensormaptype(S, 3, 3, M)
    return JordanMPO_∂∂AC2{O1,O2,O3,typeof(AA)}(II, IC, ID, CB, CA, AB, AA, BE, DE, EE)
end

Base.:*(H::JordanMPO_∂∂AC2, x) = H(x)

function ∂∂AC2(site::Int, mps, operator::MPOHamiltonian{<:JordanMPOTensor}, envs)
    GL = leftenv(envs, site, mps)
    GR = rightenv(envs, site + 1, mps)
    W1 = operator[site]
    W2 = operator[site + 1]

    # starting left - continuing right
    if nonzero_length(W1.C) > 0 && nonzero_length(W2.A) > 0
        @plansor CA_[-1 -2 -3; -4 -5 -6] ≔ W1.C[-1; -4 2] * W2.A[2 -2; -5 1] *
                                           GR[2:(end - 1)][-6 1; -3]
        CA = only(CA_)
    else
        CA = missing
    end

    # continuing left - ending right
    if nonzero_length(W1.A) > 0 && nonzero_length(W2.B) > 0
        @plansor AB_[-1 -2 -3; -4 -5 -6] ≔ GL[2:(end - 1)][-1 2; -4] * W1.A[2 -2; -5 1] *
                                           W2.B[1 -3; -6]
        AB = only(AB_)
    else
        AB = missing
    end

    # middle
    if nonzero_length(W1.C) > 0 && nonzero_length(W2.B) > 0
        if !ismissing(CA)
            @plansor CA[-1 -2 -3; -4 -5 -6] += W1.C[-1; -4 1] * W2.B[1 -2; -5] *
                                               removeunit(GR[end], 2)[-6; -3]
            CB = missing
        elseif !ismissing(AB)
            @plansor AB[-1 -2 -3; -4 -5 -6] += removeunit(GL[1], 2)[-1; -4] *
                                               W1.C[-2; -5 1] *
                                               W2.B[1 -3; -6]
            CB = missing
        else
            @plansor CB_[-1 -2; -3 -4] ≔ W1.C[-1; -3 1] * W2.B[1 -2; -4]
            CB = only(CB_)
        end
    else
        CB = missing
    end

    # starting right
    if nonzero_length(W2.C) > 0
        if !ismissing(CA)
            I = id(storagetype(GR[1]), physicalspace(W1))
            @plansor CA[-1 -2 -3; -4 -5 -6] += (I[-1; -4] * W2.C[-2; -5 1]) *
                                               GR[2:(end - 1)][-6 1; -3]
            IC = missing
        else
            @plansor IC[-1 -2; -3 -4] ≔ W2.C[-1; -3 1] * GR[2:(end - 1)][-4 1; -2]
        end
    else
        IC = missing
    end

    # ending left
    if nonzero_length(W1.B) > 0
        if !ismissing(AB)
            I = id(storagetype(GL[end]), physicalspace(W2))
            @plansor AB[-1 -2 -3; -4 -5 -6] += GL[2:(end - 1)][-1 1; -4] *
                                               (W1.B[1 -2; -5] * I[-3; -6])
            BE = missing
        else
            @plansor BE[-1 -2; -3 -4] ≔ GL[2:(end - 1)][-1 2; -3] * W1.B[2 -2; -4]
        end
    else
        BE = missing
    end

    # onsite left
    if nonzero_length(W1.D) > 0
        if !ismissing(BE)
            @plansor BE[-1 -2; -3 -4] += removeunit(GL[1], 2)[-1; -3] * W1.D[-2; -4]
            DE = missing
        elseif !ismissing(AB)
            I = id(storagetype(GL[end]), physicalspace(W2))
            @plansor AB[-1 -2 -3; -4 -5 -6] += removeunit(GL[1], 2)[-1; -4] *
                                               (W1.D[-2; -5] * I[-3; -6])
            DE = missing
            # TODO: could also try in CA?
        else
            DE = only(W1.D)
        end
    else
        DE = missing
    end

    # onsite right
    if nonzero_length(W2.D) > 0
        if !ismissing(IC)
            @plansor IC[-1 -2; -3 -4] += W2.D[-1; -3] * removeunit(GR[end], 2)[-4; -2]
            ID = missing
        elseif !ismissing(CA)
            I = id(storagetype(GR[1]), physicalspace(W1))
            @plansor CA[-1 -2 -3; -4 -5 -6] += (I[-1; -4] * W2.D[-2; -5]) *
                                               removeunit(GR[end], 2)[-6; -3]
            ID = missing
        else
            ID = only(W2.D)
        end
    else
        ID = missing
    end

    # finished
    if !ismissing(IC)
        I = id(storagetype(GR[1]), physicalspace(W2))
        @plansor IC[-1 -2; -3 -4] += I[-1; -3] * removeunit(GR[1], 2)[-4; -2]
        II = missing
    elseif !ismissing(CA)
        I = id(storagetype(GR[1]), physicalspace(W1) ⊗ physicalspace(W2))
        @plansor CA[-1 -2 -3; -4 -5 -6] += I[-1 -2; -4 -5] * removeunit(GR[1], 2)[-6; -3]
        II = missing
    else
        II = transpose(removeunit(GR[1], 2))
    end

    # unstarted
    if !ismissing(BE)
        I = id(storagetype(GL[end]), physicalspace(W1))
        @plansor BE[-1 -2; -3 -4] += removeunit(GL[end], 2)[-1; -3] * I[-2; -4]
        EE = missing
    elseif !ismissing(AB)
        I = id(storagetype(GL[end]), physicalspace(W1) ⊗ physicalspace(W2))
        @plansor AB[-1 -2 -3; -4 -5 -6] += removeunit(GL[end], 2)[-1; -4] * I[-2 -3; -5 -6]
        EE = missing
    else
        EE = removeunit(GL[end], 2)
    end

    # continuing - continuing
    # TODO: MPO_∂∂AC2 code reuse + optimization
    AA = (GL[2:(end - 1)], W1.A, W2.A, GR[2:(end - 1)])

    return JordanMPO_∂∂AC2(II, IC, ID, CB, CA, AB, AA, BE, DE, EE)
end

function (H::JordanMPO_∂∂AC2)(x::MPOTensor)
    y = zerovector!(similar(x))

    if !ismissing(H.II)
        @plansor y[-1 -2; -3 -4] += x[-1 -2; 1 -4] * H.II[-3; 1]
    end

    if !ismissing(H.IC)
        @plansor y[-1 -2; -3 -4] += x[-1 -2; 1 2] * H.IC[-4 -3; 2 1]
    end

    if !ismissing(H.ID)
        @plansor y[-1 -2; -3 -4] += x[-1 -2; -3 1] * H.ID[-4; 1]
    end

    if !ismissing(H.CB)
        @plansor y[-1 -2; -3 -4] += x[-1 1; -3 2] * H.CB[-2 -4; 1 2]
    end

    if !ismissing(H.CA)
        @plansor y[-1 -2; -3 -4] += x[-1 1; 3 2] * H.CA[-2 -4 -3; 1 2 3]
    end

    if !ismissing(H.AB)
        @plansor y[-1 -2; -3 -4] += x[1 2; -3 3] * H.AB[-1 -2 -4; 1 2 3]
    end

    if !ismissing(H.BE)
        @plansor y[-1 -2; -3 -4] += x[1 2; -3 -4] * H.BE[-1 -2; 1 2]
    end

    if !ismissing(H.DE)
        @plansor y[-1 -2; -3 -4] += x[-1 1; -3 -4] * H.DE[-2; 1]
    end

    if !ismissing(H.EE)
        @plansor y[-1 -2; -3 -4] += x[1 -2; -3 -4] * H.EE[-1; 1]
    end

    GL, A1, A2, GR = H.AA
    if nonzero_length(A1) > 0 && nonzero_length(A2) > 0
        # TODO: there are too many entries here!
        @plansor y[-1 -2; -3 -4] += GL[-1 7; 6] * x[6 5; 1 3] * A1[7 -2; 5 4] *
                                    A2[4 -4; 3 2] * GR[1 2; -3]
    end

    return y
end
