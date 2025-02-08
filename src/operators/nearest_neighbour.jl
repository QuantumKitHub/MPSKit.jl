struct NN_Hamiltonian{T<:AbstractTensorMap{<:Any,<:Any,2,2}}
    H::T
end

VectorInterface.scalartype(::Type{NN_Hamiltonian{T}}) where {T} = scalartype(T)

struct NN_H_∂∂AC{L,R}
    GL::L
    GR::R
end
struct NN_H_∂∂AC2{L,M,R}
    GL::L
    H::M
    GR::R
end

function ∂∂AC(pos::Int, mps, operator::NN_Hamiltonian, envs)
    I = id(storagetype(operator.H), space(operator.H, 1))
    @tensor GL[-1 -2; -3 -4] := leftenv(envs, pos, mps)[-1; -3] * I[-2; -4]
    if pos > 1
        @tensor GL[-1 -2; -3 -4] += mps.AL[pos - 1][1 2; -3] *
                                    conj(mps.AL[pos - 1][1 3; -1]) *
                                    operator.H[3 -2; 2 -4]
    end
    @tensor GR[-1 -2; -3 -4] := conj(I[-2; -4]) * rightenv(envs, pos, mps)[-1; -3]
    if pos < length(mps)
        @tensor GR[-1 -2; -3 -4] += operator.H[-4 2; -2 3] * mps.AR[pos + 1][-1 3; 1] *
                                    conj(mps.AR[pos + 1][-3 2; 1])
    end
    return NN_H_∂∂AC(GL, GR)
end

function ∂∂AC2(pos::Int, mps, operator::NN_Hamiltonian, envs)
    I = id(storagetype(operator.H), space(operator.H, 1))
    @tensor GL[-1 -2; -3 -4] := leftenv(envs, pos, mps)[-1; -3] * I[-2; -4]
    if pos > 1
        @tensor GL[-1 -2; -3 -4] += mps.AL[pos - 1][1 2; -3] *
                                    conj(mps.AL[pos - 1][1 3; -1]) *
                                    operator.H[3 -2; 2 -4]
    end
    @tensor GR[-1 -2; -3 -4] := conj(I[-2; -4]) * rightenv(envs, pos + 1, mps)[-1; -3]
    if pos < (length(mps) - 1)
        @tensor GR[-1 -2; -3 -4] += operator.H[-4 2; -2 3] * mps.AR[pos + 2][-1 3; 1] *
                                    conj(mps.AR[pos + 2][-3 2; 1])
    end
    return NN_H_∂∂AC2(GL, operator.H, GR)
end

(H_eff::NN_H_∂∂AC)(x::MPSTensor) = H_eff * x
(H_eff::NN_H_∂∂AC2)(x::MPOTensor) = H_eff * x

function Base.:*(H_eff::NN_H_∂∂AC, x::MPSTensor)
    @tensor y[-1 -2; -3] := H_eff.GL[-1 -2; 1 2] * x[1 2; -3] +
                            x[-1 1; 2] * H_eff.GR[2 1; -3 -2]
    # @tensor y[-1 -2; -3] += x[-1 1; 2] * H_eff.GR[2 1; -3 -2]
    return y
end

function Base.:*(H_eff::NN_H_∂∂AC2, x::MPOTensor)
    @tensor contractcheck = true y[-1 -2; -3 -4] := H_eff.GL[-1 -2; 1 2] * x[1 2; -3 -4]
    @tensor contractcheck = true y[-1 -2; -3 -4] += H_eff.H[-2 -4; 1 2] * x[-1 1; -3 2]
    @tensor contractcheck = true y[-1 -2; -3 -4] += H_eff.GR[1 2; -3 -4] * x[-1 -2; 1 2]
    return y
end

const FiniteNNEnvironments = FiniteEnvironments{<:Any,<:NN_Hamiltonian}

function environments(mps::FiniteMPS, operator::NN_Hamiltonian, above=nothing)
    leftstart = l_LL(mps)
    rightstart = r_RR(mps)
    return environments(mps, operator, above, leftstart, rightstart)
end

function rightenv(ca::FiniteNNEnvironments, ind, state)
    a = findfirst(i -> !(state.AR[i] === ca.rdependencies[i]), length(state):-1:(ind + 1))
    a = isnothing(a) ? nothing : length(state) - a + 1

    if !isnothing(a)
        # we need to recalculate
        for j in reverse((ind + 1):a)
            above = isnothing(ca.above) ? state.AR[j] : ca.above.AR[j]
            T = TransferMatrix(above, state.AR[j])
            ca.GRs[j] = T * ca.GRs[j + 1]
            if j < length(state)
                @tensor ca.GRs[j][-1; -2] += state.AR[j][-1 5; 4] *
                                             state.AR[j + 1][4 2; 1] *
                                             conj(state.AR[j + 1][7 3; 1]) *
                                             conj(state.AR[j][-2 6; 7]) *
                                             ca.operator.H[6 3; 5 2]
            end
            ca.rdependencies[j] = state.AR[j]
        end
    end

    return ca.GRs[ind + 1]
end

function leftenv(ca::FiniteNNEnvironments, ind, state)
    a = findfirst(i -> !(state.AL[i] === ca.ldependencies[i]), 1:(ind - 1))

    if !isnothing(a)
        #we need to recalculate
        for j in a:(ind - 1)
            above = isnothing(ca.above) ? state.AL[j] : ca.above.AL[j]
            T = TransferMatrix(above, state.AL[j])
            ca.GLs[j + 1] = ca.GLs[j] * T
            if j > 1
                @tensor ca.GLs[j + 1][-1; -2] += state.AL[j - 1][1 2; 4] *
                                                 state.AL[j][4 5; -2] *
                                                 conj(state.AL[j][7 6; -1]) *
                                                 conj(state.AL[j - 1][1 3; 7]) *
                                                 ca.operator.H[3 6; 2 5]
            end
            ca.ldependencies[j] = state.AL[j]
        end
    end

    return ca.GLs[ind]
end

expectation_value(mps::FiniteMPS, H::NN_Hamiltonian, envs...) = zero(scalartype(mps))
