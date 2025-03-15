"""
    struct JordanMPO_
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
            error()
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

struct JordanMPO_∂∂AC2{O1,O2,O3,O4}
    onsite_left::Union{O1,Missing}
    onsite_right::Union{O1,Missing}
    not_started::Union{O1,Missing}
    finished::Union{O1,Missing}
    start_end::Union{O2,Missing}
    starting_left::Union{O3,Missing}
    starting_right::Union{O2,Missing}
    ending_left::Union{O2,Missing}
    ending_right::Union{O3,Missing}
    continuing::O4
end
