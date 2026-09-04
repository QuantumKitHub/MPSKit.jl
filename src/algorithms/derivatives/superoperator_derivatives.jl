# Derivative operators for superoperators acting on MPDOs
# =======================================================
# An MPDO state does not match `_HAM_MPS_TYPES`, which is restricted to `MPSTensor`s, so
# `AC_hamiltonian` falls through to the generic `MPO_AC_Hamiltonian` constructor and applies
# the whole `JordanMPOTensor` -- structural zeros included -- on every matvec. The
# environments are unaffected: those dispatch on the operator alone and still exploit the
# Jordan structure level by level. Only the local application is dense.

# dense bra-side ∂AC: the ket-side method of `mpo_derivatives.jl` with `τ` and the operator
# swapped along the MPO line, and the operator transposed onto the dual bra leg
function (h::MPO_AC_Hamiltonian{<:MPSTensor, <:BraSide, <:MPSTensor})(
        x::GenericMPSTensor{<:Any, 3}
    )
    backend, allocator = h.backend, h.allocator
    Wt = _bra_transpose(only(h.operators))
    @plansor backend = backend allocator = allocator begin
        y[-1 -2 -3; -4] ≔ h.leftenv[-1 7; 6] * x[6 4 2; 1] *
            τ[7 -2; 4 5] * Wt[5 -3; 2 3] * h.rightenv[1 3; -4]
    end
    return y isa AbstractBlockTensorMap ? only(y) : y
end

# dense bra-side ∂AC2, the two-site analogue of the above: `τ` and each operator swap
# places along the MPO line, and both operators are transposed onto the dual bra legs
function (h::MPO_AC2_Hamiltonian{<:MPSTensor, <:BraSide, <:BraSide, <:MPSTensor})(
        x::AbstractTensorMap{<:Any, <:Any, 3, 3}
    )
    backend, allocator = h.backend, h.allocator
    W1 = _bra_transpose(h.operators[1])
    W2 = _bra_transpose(h.operators[2])
    @plansor backend = backend allocator = allocator begin
        y[-1 -2 -3; -4 -5 -6] ≔ h.leftenv[-1 11; 10] * x[10 8 6; 1 2 4] *
            h.rightenv[1 3; -4] * τ[11 -2; 8 9] * W1[9 -3; 6 7] *
            τ[7 -6; 4 5] * W2[5 -5; 2 3]
    end
    return y isa AbstractBlockTensorMap ? only(y) : y
end

# SuperOperator interface
# -----------------------
# Everything is expressed through the lazy-sum view, so the two terms get an independent
# environment each (`MultipleEnvironments`) and combine into a `LazySum` of derivative
# operators, which already knows how to be applied and exponentiated.

function environments(below, O::SuperOperator, above = below; kwargs...)
    return environments(below, _lazysum(O), above; kwargs...)
end
function environments(below, O::SuperOperator, above, alg; kwargs...)
    return environments(below, _lazysum(O), above, alg; kwargs...)
end

function recalculate!(envs, below, O::SuperOperator, above = below; kwargs...)
    return recalculate!(envs, below, _lazysum(O), above; kwargs...)
end
function recalculate!(envs, below, O::SuperOperator, above, alg; kwargs...)
    return recalculate!(envs, below, _lazysum(O), above, alg; kwargs...)
end

for hamiltonian in (:C_hamiltonian, :AC_hamiltonian, :AC2_hamiltonian)
    @eval function $hamiltonian(site::Int, below, operator::SuperOperator, above, envs; kwargs...)
        return $hamiltonian(site, below, _lazysum(operator), above, envs; kwargs...)
    end
end

function expectation_value(ψ, O::SuperOperator, envs...)
    return expectation_value(ψ, _lazysum(O), envs...)
end
