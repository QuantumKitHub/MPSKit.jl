function TensorKit.leftorth(W::JordanMPOTensor; alg = QRpos())
    # orthogonalize second column against first
    WI = removeunit(W[1, 1, 1, 1], 1)
    @tensor t[l; r] := conj(WI[p; p' l]) * W.C[p; p' r]
    I = sectortype(W)
    S = spacetype(W)

    @plansor t[l; r] := conj(removeunit(W[1, 1, 1, 1], 1)[p; p' l]) * W.C[p; p' r]
    # @plansor C′[p; p' r] := W.C[p; p' r] - WI[p; p' l] * t[l; r]
    @plansor C′[p; p' r] := -WI[p; p' l] * t[l; r]
    add!(C′, W.C)

    # QR of second column
    CA = transpose(cat(insertleftunit(C′, 1), W.A; dims = 1), ((3, 1, 2), (4,)))
    Q, r = leftorth!(CA; alg)
    Q′ = transpose(Q, ((2, 3), (1, 4)))
    V = codomain(W) ← physicalspace(W) ⊗ BlockTensorKit.oplus(oneunit(S), right_virtualspace(Q′), oneunit(S))
    Q1 = SparseBlockTensorMap(Q′[2:end, 1, 1, 1])
    Q2 = removeunit(SparseBlockTensorMap(Q′[1:1, 1, 1, 1]), 1)

    # Assemble output
    W′ = JordanMPOTensor(V, Q1, W.B, Q2, W.D)

    R = similar(W, right_virtualspace(W′) ← right_virtualspace(W′))
    R[1, 1] = id!(R[1, 1])
    R[1, 2] = t
    R[2, 2] = r
    R[3, 3] = id!(R[3, 3])

    return W′, R
end

function left_canonicalize!(H::MPOHamilonian, i::Int; alg = QRPos())
    @assert i != 1 "TBA"

    W = H[i]

    # orthogonalize second column against first
    WI = removeunit(W[1, 1, 1, 1], 1)
    @tensor t[l; r] := conj(WI[p; p' l]) * W.C[p; p' r]
    # @plansor C′[p; p' r] := W.C[p; p' r] - WI[p; p' l] * t[l; r]
    @plansor C′[p; p' r] := -WI[p; p' l] * t[l; r]
    add!(C′, W.C)

    # QR of second column
    CA = transpose(cat(insertleftunit(C′, 1), W.A; dims = 1), ((3, 1, 2), (4,)))
    Q, R = leftorth!(CA; alg)
    Q′ = transpose(Q, ((2, 3), (1, 4)))
    Q1 = SparseBlockTensorMap(Q′[2:end, 1, 1, 1])
    Q2 = removeunit(SparseBlockTensorMap(Q′[1:1, 1, 1, 1]), 1)
    V = BlockTensorKit.oplus(oneunit(spacetype(W)), right_virtualspace(Q′), oneunit(spacetype(W)))
    H[i] = JordanMPOTensor(codomain(W) ← physicalspace(W) ⊗ V, Q1, W.B, Q2, W.D)

    # absorb into next site
    W′ = H[i + 1]
    @plansor A′[l p; p' r] := R[l; r'] * W′.A[r' p; p' r]
    @plansor B′[l p; p'] := R[l; r] * W′.B[r p; p']
    @plansor C′[l p; p' r] := t[l; r'] * W′.A[r' p; p' r]
    C′ = add!(removeunit(C, 1), W′.C)
    @plansor D′[l p; p'] := t[l; r] * W′.B[r p; p']
    D′ = add!(removeunit(D, 1), W′.D)

    H[i + 1] = JordanMPOTensor(V ⊗ physicalspace(W′) ← domain(W′), A′, B′, C′, D′)
    return H
end
