"""
    correlator(ψ, O1, O2, i, j)
    correlator(ψ, O12, i, j)

Compute the 2-point correlator <ψ|O1[i]O2[j]|ψ> for inserting `O1` at `i` and `O2` at `j`.
Also accepts ranges for `j`.
"""
function correlator end

function correlator(state::AbstractMPS, O₁::MPOTensor, O₂::MPOTensor, i::Int, j::Int)
    return first(correlator(state, O₁, O₂, i, j:j))
end

function correlator(state::AbstractMPS, O₁::MPOTensor, O₂::MPOTensor, i::Int,
                    js::AbstractRange{Int})
    first(js) > i || @error "i should be smaller than j ($i, $(first(js)))"
    S₁ = _firstspace(O₁)
    S₁ == oneunit(S₁) || throw(ArgumentError("O₁ should start with a trivial leg."))
    S₂ = _lastspace(O₂)
    S₂ == S₁' || throw(ArgumentError("O₂ should end with a trivial leg."))

    G = similar(js, scalartype(state))
    U = Tensor(ones, S₁)

    @plansor Vₗ[-1 -2; -3] := state.AC[i][3 4; -3] * conj(U[1]) * O₁[1 2; 4 -2] *
                              conj(state.AC[i][3 2; -1])
    ctr = i + 1

    for (k, j) in enumerate(js)
        if j > ctr
            Vₗ = Vₗ * TransferMatrix(state.AR[ctr:(j - 1)])
        end
        G[k] = @plansor Vₗ[2 3; 5] * state.AR[j][5 6; 7] * O₂[3 4; 6 1] * U[1] *
                        conj(state.AR[j][2 4; 7])
        ctr = j
    end
    return G
end

function correlator(state::AbstractMPS, O₁₂::AbstractTensorMap{S,2,2}, i::Int, j) where {S}
    O₁, O₂ = decompose_localmpo(add_util_leg(O₁₂))
    return correlator(state, O₁, O₂, i, j)
end
