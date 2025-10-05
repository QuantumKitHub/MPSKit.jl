function left_canonicalize!(
        H::FiniteMPOHamiltonian, i::Int;
        alg = Defaults.alg_qr(), trscheme::TruncationStrategy = notrunc()
    )
    1 ≤ i < length(H) || throw(ArgumentError("Bounds error in canonicalize"))

    W = H[i]
    S = spacetype(W)
    d = sqrt(dim(physicalspace(W)))

    # orthogonalize second column against first
    WI = removeunit(W[1, 1, 1, 1], 1)
    @tensor t[l; r] := conj(WI[p; p' l]) * W.C[p; p' r]
    # TODO: the following is currently broken due to a TensorKit bug
    # @plansor C′[p; p' r] := W.C[p; p' r] - WI[p; p' l] * t[l; r]
    @plansor C′[p; p' r] := -WI[p; p' l] * t[l; r]
    add!(C′, W.C)

    # QR of second column
    if size(W, 1) == 1
        tmp = transpose(C′, ((2, 1), (3,)))
        Q, R = _left_orth!(tmp; alg, trunc = trscheme)

        if dim(R) == 0 # fully truncated
            V = BlockTensorKit.oplus(oneunit(S), oneunit(S))
            Q1 = typeof(W.A)(undef, SumSpace{S}() ⊗ physicalspace(W) ← physicalspace(W) ⊗ SumSpace{S}())
            Q2 = typeof(W.C)(undef, physicalspace(W) ← physicalspace(W) ⊗ SumSpace{S}())
        else
            V = BlockTensorKit.oplus(oneunit(S), space(R, 1), oneunit(S))
            scale!(Q, d)
            scale!(R, inv(d))
            Q1 = typeof(W.A)(undef, SumSpace{S}() ⊗ physicalspace(W) ← physicalspace(W) ⊗ space(R, 1))
            Q2 = transpose(Q, ((2,), (1, 3)))
        end
        H[i] = JordanMPOTensor(codomain(W) ← physicalspace(W) ⊗ V, Q1, W.B, Q2, W.D)
    else
        tmp = transpose(cat(insertleftunit(C′, 1), W.A; dims = 1), ((3, 1, 2), (4,)))
        Q, R = _left_orth!(tmp; alg, trunc = trscheme)

        if dim(R) == 0 # fully truncated
            V = BlockTensorKit.oplus(oneunit(S), oneunit(S))
            Q1 = typeof(W.A)(undef, SumSpace{S}() ⊗ physicalspace(W) ← physicalspace(W) ⊗ SumSpace{S}())
            Q2 = typeof(W.C)(undef, physicalspace(W) ← physicalspace(W) ⊗ SumSpace{S}())
        else
            scale!(Q, d)
            scale!(R, inv(d))
            Q′ = transpose(Q, ((2, 3), (1, 4)))
            Q1 = Q′[2:end, 1, 1, 1]
            Q2 = removeunit(SparseBlockTensorMap(Q′[1:1, 1, 1, 1]), 1)
            V = BlockTensorKit.oplus(oneunit(S), right_virtualspace(Q′), oneunit(S))
        end
        H[i] = JordanMPOTensor(codomain(W) ← physicalspace(W) ⊗ V, Q1, W.B, Q2, W.D)
    end

    # absorb into next site
    W′ = H[i + 1]

    if size(W′, 4) > 1 && dim(R) != 0
        @plansor A′[l p; p' r] := R[l; r'] * W′.A[r' p; p' r]
    else
        A′ = similar(W′.A, right_virtualspace(H[i].A) ⊗ physicalspace(W′) ← domain(W′.A))
    end

    if size(W′, 4) > 1
        @plansor C′[l p; p' r] := t[l; r'] * W′.A[r' p; p' r]
        C′ = add!(removeunit(C′, 1), W′.C)
    else
        C′ = W′.C # empty
    end

    if dim(R) != 0
        @plansor B′[l p; p'] := R[l; r] * W′.B[r p; p']
    else
        B′ = similar(W′.B, right_virtualspace(H[i].A) ⊗ physicalspace(W′) ← domain(W′.B))
    end

    @plansor D′[l p; p'] := t[l; r] * W′.B[r p; p']
    D′ = add!(removeunit(D′, 1), W′.D)

    H[i + 1] = JordanMPOTensor(
        right_virtualspace(H[i]) ⊗ physicalspace(W′) ← domain(W′),
        A′, B′, C′, D′
    )
    return H
end

function right_canonicalize!(
        H::FiniteMPOHamiltonian, i::Int;
        alg = Defaults.alg_lq(), trscheme::TruncationStrategy = notrunc()
    )
    1 < i ≤ length(H) || throw(ArgumentError("Bounds error in canonicalize"))

    W = H[i]
    S = spacetype(W)
    d = sqrt(dim(physicalspace(W)))

    # orthogonalize second row against last
    WI = removeunit(W[end, 1, 1, end], 4)
    @plansor t[l; r] := conj(WI[r p; p']) * W.B[l p; p']
    # TODO: the following is currently broken due to a TensorKit bug
    # @plansor B′[l p; p'] := W.B[l p; p'] - WI[r p; p'] * t[l; r]
    @plansor B′[l p; p'] := -WI[r p; p'] * t[l; r]
    add!(B′, W.B)

    # LQ of second row
    if size(W, 4) == 1
        tmp = transpose(B′, ((1,), (3, 2)))
        R, Q = _right_orth!(tmp; alg, trunc = trscheme)

        if dim(R) == 0
            V = BlockTensorKit.oplus(oneunit(S), oneunit(S))
            Q1 = typeof(W.A)(undef, SumSpace{S}() ⊗ physicalspace(W) ← physicalspace(W) ⊗ SumSpace{S}())
            Q2 = typeof(W.B)(undef, SumSpace{S}() ⊗ physicalspace(W) ← physicalspace(W))
        else
            V = BlockTensorKit.oplus(oneunit(S), space(Q, 1), oneunit(S))
            scale!(Q, d)
            scale!(R, inv(d))
            Q1 = typeof(W.A)(undef, space(Q, 1) ⊗ physicalspace(W) ← physicalspace(W) ⊗ SumSpace{S}())
            Q2 = transpose(Q, ((1, 3), (2,)))
        end
        H[i] = JordanMPOTensor(V ⊗ physicalspace(W) ← domain(W), Q1, Q2, W.C, W.D)
    else
        tmp = transpose(cat(insertleftunit(B′, 4), W.A; dims = 4), ((1,), (3, 4, 2)))
        R, Q = _right_orth!(tmp; alg, trunc = trscheme)
        if dim(R) == 0
            V = BlockTensorKit.oplus(oneunit(S), oneunit(S))
            Q1 = typeof(W.A)(undef, SumSpace{S}() ⊗ physicalspace(W) ← physicalspace(W) ⊗ SumSpace{S}())
            Q2 = typeof(W.B)(undef, SumSpace{S}() ⊗ physicalspace(W) ← physicalspace(W))
        else
            scale!(Q, d)
            scale!(R, inv(d))
            Q′ = transpose(Q, ((1, 4), (2, 3)))
            Q1 = SparseBlockTensorMap(Q′[1, 1, 1, 2:end])
            Q2 = removeunit(SparseBlockTensorMap(Q′[1, 1, 1, 1:1]), 4)
            V = BlockTensorKit.oplus(oneunit(S), left_virtualspace(Q′), oneunit(S))
        end
        H[i] = JordanMPOTensor(V ⊗ physicalspace(W) ← domain(W), Q1, Q2, W.C, W.D)
    end

    # absorb into previous site
    W′ = H[i - 1]

    if size(W′, 1) > 1 && dim(R) != 0
        @plansor A′[l p; p' r] := W′.A[l p; p' r'] * R[r'; r]
    else
        A′ = similar(W′.A, codomain(W′.A) ← physicalspace(W′.A) ⊗ left_virtualspace(H[i].A))
    end

    if size(W′, 1) > 1
        @plansor B′[l p; p' r] := W′.A[l p; p' r'] * t[r'; r]
        B′ = add!(removeunit(B′, 4), W′.B)
    else
        B′ = W′.B
    end

    if dim(R) != 0
        @plansor C′[p; p' r] := W′.C[p; p' r'] * R[r'; r]
    else
        C′ = similar(W′.C, codomain(W′.C) ← physicalspace(W′) ⊗ left_virtualspace(H[i].A))
    end

    @plansor D′[p; p' r] := W′.C[p; p' r'] * t[r'; r]
    D′ = add!(removeunit(D′, 3), W′.D)
    H[i - 1] = JordanMPOTensor(codomain(W′) ← physicalspace(W′) ⊗ V, A′, B′, C′, D′)
    return H
end
