function TensorKit.leftorth!(W::JordanMPOTensor; alg = QRpos())
    # orthogonalize second column against first
    WI = removeunit(W[1, 1, 1, 1], 1)
    @tensor t[l; r] := conj(WI[p; p' l]) * W.C[p; p' r]
    I = sectortype(W)
    S = spacetype(W)

    @tensor t[l; r] := conj(removeunit(W[1, 1, 1, 1], 1)[p; p' l]) * W.C[p; p' r]
    @tensor C′[p; p' r] := W.C[p; p' r] - WI[p; p' l] * t[l; r]

    # QR of second column
    CA = cat(insertleftunit(C′, 1), W.A; dims = 1)
    Q, r = leftorth(CA, ((3, 1, 2), (4,)); alg)
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
