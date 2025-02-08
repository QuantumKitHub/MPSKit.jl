struct JordanMPOTensor{T,S,
                       TA<:AbstractTensorMap{T,S,2,2},
                       TB<:AbstractTensorMap{T,S,1,2},
                       TC<:AbstractTensorMap{T,S,1,2},
                       TD<:AbstractTensorMap{T,S,1,1}}
    A::TA
    B::TB
    C::TC
    D::TD
end

function JordanMPOTensor(h::SparseBlockTensorMap)
    A = h[2:(end - 1), 1, 1, 2:(end - 1)]
    B = transpose(removeunit(h[2:(end - 1), 1, 1, end], 4), ((2,), (1, 3)))
    C = removeunit(h[1, 1, 1, 2:(end - 1)], 1)
    D = removeunit(removeunit(h[1:1, 1, 1, end:end], 4), 1)
    return JordanMPOTensor(A, B, C, D)
end

# wrapper to indicate GL[1] is isomorphism
struct MPOHamiltonian_GL{T,S,T1,T2} <: AbstractBlockTensorMap{T,S,2,1}
    GL::T1
    GLend::T2
end
function MPOHamiltonian_GL(GL::T1, GLend::T2) where {T1,T2}
    T = scalartype(T1)
    S = spacetype(T1)
    return MPOHamiltonian_GL{T,S,T1,T2}(GL, GLend)
end
function MPOHamiltonian_GL(GL)
    # @assert GL[1] ≈ isomorphism(storagetype(GL[1]), space(GL[1]))
    return MPOHamiltonian_GL(GL[2:(end - 1)], removeunit(GL[end], 2))
end

# wrapper to indicate GR[end] is isomorphism
struct MPOHamiltonian_GR{T,S,T1,T2} <: AbstractBlockTensorMap{T,S,2,1}
    GR::T1
    GRbegin::T2
end
function MPOHamiltonian_GR(GR::T1, GRbegin::T2) where {T1,T2}
    T = scalartype(T1)
    S = spacetype(T1)
    return MPOHamiltonian_GR{T,S,T1,T2}(GR, GRbegin)
end

function MPOHamiltonian_GR(GR)
    # @assert GR[end] ≈ isomorphism(storagetype(GR[end]), space(GR[end]))
    return MPOHamiltonian_GR(GR[2:(end - 1)], removeunit(GR[1], 2))
end

function ∂∂AC2(pos::Int, mps, operator::MPO{<:JordanMPOTensor}, envs)
    O1 = operator[pos]
    O2 = operator[pos + 1]
    GL = MPOHamiltonian_GL(leftenv(envs, pos, mps))
    GR = MPOHamiltonian_GR(rightenv(envs, pos + 1, mps))
    return MPO_∂∂AC2(O1, O2, GL, GR)
end

function ∂AC(x::MPSTensor, O::JordanMPOTensor, leftenv::MPOHamiltonian_GL,
             rightenv::MPOHamiltonian_GR)::typeof(x)
    @tensor begin
        y[-1 -2; -3] := leftenv.GLend[-1; 1] * x[1 -2; -3] +                # everything done
                        x[-1 -2; 1] * rightenv.GRbegin[1; -3] +             # nothing done
                        x[-1 1; -3] * O.D[-2; 1] +                          # onsite
                        leftenv.GL[-1 2; 1] * O.B[-2; 2 3] * x[1 3; -3] +   # end
                        x[-1 2; 1] * O.C[-2; 2 3] * rightenv.GR[1 3; -3] +  # start
                        leftenv.GL[-1 5; 4] * x[4 2; 1] * O.A[5 -2; 2 3] *  # continue
                        rightenv.GR[1 3; -3]
    end
    return y isa BlockTensorMap ? only(y) : y
end

hasterm(::AbstractTensorMap) = true
hasterm(::BlockTensorMap) = true
hasterm(t::SparseBlockTensorMap) = nonzero_length(t) > 0

function ∂AC2(x::MPOTensor, O1::JordanMPOTensor, O2::JordanMPOTensor,
              leftenv::MPOHamiltonian_GL, rightenv::MPOHamiltonian_GR)
    # everything done and nothing done always non-empty
    @tensor y[-1 -2; -3 -4] := leftenv.GLend[-1; 1] * x[1 -2; -3 -4] +
                               x[-1 -2; 1 -4] * rightenv.GRbegin[1; -3]
    if hasterm(O1.D) # onsite left
        @tensor y[-1 -2; -3 -4] += x[-1 1; -3 -4] * O1.D[-2; 1]
    end
    if hasterm(O2.D) # onsite right
        @tensor y[-1 -2; -3 -4] += x[-1 -2; -3 1] * O2.D[-4; 1]
    end
    if hasterm(O1.B) && hasterm(O2.C) # start and stop
        @tensor y[-1 -2; -3 -4] += x[-1 1; -3 2] * O1.C[-2; 1 3] *
                                   O2.B[-4; 3 2]
    end
    if hasterm(O2.C) # start right
        @tensor y[-1 -2; -3 -4] += x[-1 -2; 1 2] * O2.C[-4; 2 3] * rightenv.GR[1 3; -3]
    end
    if hasterm(O1.C) && hasterm(O2.A) # start left
        @tensor y[-1 -2; -3 -4] += x[-1 4; 1 2] * O1.C[-2; 4 5] * O2.A[5 -4; 2 3] *
                                   rightenv.GR[1 3; -3]
    end
    if hasterm(O1.B) # end left
        @tensor y[-1 -2; -3 -4] += x[1 3; -3 -4] * leftenv.GL[-1 2; 1] * O1.B[-2; 2 3]
    end
    if hasterm(O1.A) && hasterm(O2.B) # end right
        @tensor y[-1 -2; -3 -4] += x[1 3; -3 5] * leftenv.GL[-1 2; 1] * O1.A[2 -2; 3 4] *
                                   O2.B[-4; 4 5]
    end
    if hasterm(O1.A) && hasterm(O2.A) # continue
        @tensor y[-1 -2; -3 -4] += leftenv.GL[-1 7; 6] * x[6 5; 1 3] * O1.A[7 -2; 5 4] *
                                   O2.A[4 -4; 3 2] * rightenv.GR[1 2; -3]
    end
    return y isa BlockTensorMap ? only(y) : y
end
