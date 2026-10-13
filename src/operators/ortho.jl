# TODO: no `add!` exists for this mixed pair yet, and the fallback breaks on zero-length blocks
_dense(t::SparseBlockTensorMap) = BlockTensorMap(t)
_dense(t) = t

# Finite MPOHamiltonians
# ----------------------

"""
    left_canonicalize!(H::FiniteMPOHamiltonian; alg = Defaults.alg_orth()) -> H, L
    left_canonicalize!(H::InfiniteMPOHamiltonian; tol, maxiter, verbosity) -> H, L

Bring `H` into left-canonical form in place, where the upper-left block `[1 C; 0 A]` of every site
has orthonormal columns with respect to the trace inner product normalized by the physical
dimension. The identity components of `C` are gauged away, and the decompositions respect the block
structure.

Also returns the interaction-channel blocks `L` of the gauge transformations on every bond, such
that `diag(1, L[i - 1], 1) * H₀[i] = H[i] * diag(1, L[i], 1)` up to the identity components that
are gauged away, with `L[0]` the identity for a `FiniteMPOHamiltonian`.

For a `FiniteMPOHamiltonian`, this is a single sweep of QR decompositions from left to right. Any
other (possibly truncating) algorithm `alg` is applied to the interaction channels only, as used by
[`SvdCut`](@ref).

For an `InfiniteMPOHamiltonian`, the sweeps over the unit cell are repeated until the gauge
transformation is unitary, removing redundant interaction channels along the way.

# Keyword arguments

- `alg = Defaults.alg_orth()`: (finite) algorithm for the decomposition of the interaction channels
- `tol = Defaults.tolgauge`: (infinite) convergence tolerance on the deviation of the gauge
  transformation from a unitary one, which is also the relative tolerance for discarding redundant
  channels
- `maxiter = Defaults.maxiter`: (infinite) maximal number of sweeps over the unit cell
- `verbosity = VERBOSE_WARN`: (infinite) setting for how much information is displayed

# References

* [Parker et al. Phys. Rev. B 102 (2020)](@cite parker2020)
"""
function left_canonicalize!(
        H::FiniteMPOHamiltonian; alg::MatrixAlgebraKit.AbstractAlgorithm = Defaults.alg_orth()
    )
    Ls = map(1:(length(H) - 1)) do i
        H[i], t, R = _left_orth_jordan(H[i]; alg)
        H[i + 1] = _absorb_left_gauge(H[i], t, R, H[i + 1])
        return R
    end
    return H, Ls
end

"""
    right_canonicalize!(H::FiniteMPOHamiltonian; alg = Defaults.alg_orth()) -> H, L
    right_canonicalize!(H::InfiniteMPOHamiltonian; tol, maxiter, verbosity) -> H, L

Bring `H` into right-canonical form in place, where the lower-right block `[A B; 0 1]` of every site
has orthonormal rows with respect to the trace inner product normalized by the physical dimension.
The identity components of `B` are gauged away, and the decompositions respect the block structure.

Also returns the interaction-channel blocks `L` of the gauge transformations on every bond, such
that `H₀[i] * diag(1, L[i], 1) = diag(1, L[i - 1], 1) * H[i]` up to the identity components that
are gauged away, with `L[length(H)]` the identity for a `FiniteMPOHamiltonian`.

For a `FiniteMPOHamiltonian`, this is a single sweep of LQ decompositions from right to left. For an
`InfiniteMPOHamiltonian`, the sweeps over the unit cell are repeated until the gauge transformation
is unitary, removing redundant interaction channels along the way.

See [`left_canonicalize!`](@ref) for the keyword arguments.
"""
function right_canonicalize!(
        H::FiniteMPOHamiltonian; alg::MatrixAlgebraKit.AbstractAlgorithm = Defaults.alg_orth()
    )
    Ls = map(length(H):-1:2) do i
        R, t, H[i] = _right_orth_jordan(H[i]; alg)
        H[i - 1] = _absorb_right_gauge(H[i - 1], R, t, H[i])
        return R
    end
    return H, reverse!(Ls)
end

# Infinite MPOHamiltonians
# ------------------------

function left_canonicalize!(
        H::InfiniteMPOHamiltonian;
        tol::Real = Defaults.tolgauge, maxiter::Int = Defaults.maxiter, verbosity::Int = VERBOSE_WARN
    )
    alg_trunc = MatrixAlgebraKit.TruncatedAlgorithm(Defaults.alg_svd(), trunctol(; rtol = tol))
    alg = MatrixAlgebraKit.LeftOrthViaSVD(alg_trunc)
    Ls = PeriodicVector(Vector{Any}(nothing, length(H)))
    return with_verbosity(; verbosity) do
        log = IterLog("MPO LC")
        @log_initialization loginit!(log, Inf)
        for iter in 1:maxiter
            local t, R
            for i in 1:length(H)
                H[i], t, R = _left_orth_jordan(H[i]; alg)
                H[i + 1] = _absorb_left_gauge(H[i], t, R, H[i + 1])
                Ls[i] = isnothing(Ls[i]) || dim(R) == 0 ? R : R * Ls[i]
            end
            ϵ = _gauge_distance(t, R)
            if ϵ <= tol
                @log_convergence logfinish!(log, iter, ϵ)
                break
            elseif iter == maxiter
                @log_nonconvergence logcancel!(log, iter, ϵ)
            else
                @log_iteration logiter!(log, iter, ϵ)
            end
        end
        return H, map(identity, Ls)
    end
end

function right_canonicalize!(
        H::InfiniteMPOHamiltonian;
        tol::Real = Defaults.tolgauge, maxiter::Int = Defaults.maxiter, verbosity::Int = VERBOSE_WARN
    )
    alg_trunc = MatrixAlgebraKit.TruncatedAlgorithm(Defaults.alg_svd(), trunctol(; rtol = tol))
    alg = MatrixAlgebraKit.RightOrthViaSVD(alg_trunc)
    Ls = PeriodicVector(Vector{Any}(nothing, length(H)))
    return with_verbosity(; verbosity) do
        log = IterLog("MPO RC")
        @log_initialization loginit!(log, Inf)
        for iter in 1:maxiter
            local t, R
            for i in length(H):-1:1
                R, t, H[i] = _right_orth_jordan(H[i]; alg)
                H[i - 1] = _absorb_right_gauge(H[i - 1], R, t, H[i])
                Ls[i - 1] = isnothing(Ls[i - 1]) || dim(R) == 0 ? R : Ls[i - 1] * R
            end
            ϵ = _gauge_distance(t, R)
            if ϵ <= tol
                @log_convergence logfinish!(log, iter, ϵ)
                break
            elseif iter == maxiter
                @log_nonconvergence logcancel!(log, iter, ϵ)
            else
                @log_iteration logiter!(log, iter, ϵ)
            end
        end
        return H, map(identity, Ls)
    end
end

# distance of the gauge transformation `[1 t 0; 0 R 0; 0 0 1]` (or its mirror) to a unitary one
function _gauge_distance(t, R)
    codomain(R) == domain(R) || return Inf
    return norm(t) + norm(R' * R - id(storagetype(R), domain(R)))
end

# Implementation
# --------------

# Decompose `W = Q * G` with `G = [1 t 0; 0 R 0; 0 0 1]` and `Q` left-canonical.
function _left_orth_jordan(W::JordanMPOTensor; alg = Defaults.alg_orth())
    P = physicalspace(W)
    _rightunit = rightunitspace(P)
    S = spacetype(W)
    d = sqrt(dim(P))
    WA, WB, WD = W.A, W.B, W.D

    # QR of the upper-left block `[1 C; 0 A]` with the identity as first column, which keeps all other
    # columns of `Q` orthogonal to it, even if the block is rank-deficient
    rows = 1:max(size(W, 1) - 1, 1)
    V = BlockTensorMap(SparseBlockTensorMap(W)[rows, 1, 1, 1:(end - 1)])
    Q, R = left_orth!(transpose(V, ((3, 1, 2), (4,))); alg = Defaults.alg_orth())
    Π = _split_unit(domain(Q)[1], _rightunit, storagetype(W))
    Q = (Q * Π')[:, :, :, 2:2]
    R = Π * R
    t = scale!(R[1:1, 2:end], inv(d))
    R = R[2:2, 2:end]

    # other (possibly truncating) algorithms only act on the interaction channels
    if alg != Defaults.alg_orth()
        U, R = left_orth!(R; alg)
        Q = Q * U
    end
    scale!(R, inv(d))

    if dim(R) == 0 # fully truncated
        V = _rightunit ⊞ _rightunit
        Q1 = typeof(WA)(undef, codomain(WA) ← P ⊗ SumSpace{S}())
        Q2 = typeof(W.C)(undef, P ← P ⊗ SumSpace{S}())
    else
        Q = scale!(transpose(Q, ((2, 3), (1, 4))), d)
        Q1 = size(W, 1) == 1 ? typeof(WA)(undef, SumSpace{S}() ⊗ P ← P ⊗ right_virtualspace(Q)) :
            SparseBlockTensorMap(Q[2:end, 1, 1, 1])
        Q2 = removeunit(SparseBlockTensorMap(Q[1:1, 1, 1, 1]), 1)
        V = ⊞(_rightunit, right_virtualspace(Q), _rightunit)
    end
    return JordanMPOTensor(codomain(W) ← P ⊗ V, Q1, WB, Q2, WD), t, R
end

# Compute `G * W′` for the gauge `G = [1 t 0; 0 R 0; 0 0 1]` obtained from `_left_orth_jordan`,
# where `Q` is the tensor to the left of `W′`.
function _absorb_left_gauge(Q::JordanMPOTensor, t, R, W′::JordanMPOTensor)
    W′A, W′B, W′C, W′D = W′.A, W′.B, W′.C, W′.D

    if size(W′, 4) > 1 && dim(R) != 0
        @plansor A′[l p; p' r] := R[l; r'] * W′A[r' p; p' r]
    else
        A′ = similar(W′A, right_virtualspace(Q.A) ⊗ physicalspace(W′) ← domain(W′A))
    end

    if size(W′, 4) > 1
        @plansor C′[l p; p' r] := t[l; r'] * W′A[r' p; p' r]
        C′ = add!(removeunit(C′, 1), _dense(W′C))
    else
        C′ = W′C # empty
    end

    if dim(R) != 0
        @plansor B′[l p; p'] := R[l; r] * W′B[r p; p']
    else
        B′ = similar(W′B, right_virtualspace(Q.A) ⊗ physicalspace(W′) ← domain(W′B))
    end

    @plansor D′[l p; p'] := t[l; r] * W′B[r p; p']
    D′ = add!(removeunit(D′, 1), _dense(W′D))

    return JordanMPOTensor(
        right_virtualspace(Q) ⊗ physicalspace(W′) ← domain(W′), A′, B′, C′, D′
    )
end

# Decompose `W = G * Q` with `G = [1 0 0; 0 R t; 0 0 1]` and `Q` right-canonical.
function _right_orth_jordan(W::JordanMPOTensor; alg = Defaults.alg_orth())
    P = physicalspace(W)
    _rightunit = rightunitspace(P)
    S = spacetype(W)
    d = sqrt(dim(P))
    WA, WC, WD = W.A, W.C, W.D

    # LQ of the lower-right block `[A B; 0 1]` with the identity as first row, which keeps all other
    # rows of `Q` orthogonal to it, even if the block is rank-deficient
    cols = min(2, size(W, 4)):size(W, 4)
    V = SparseBlockTensorMap(W)[2:end, 1, 1, cols]
    V = BlockTensorMap(cat(V[end:end, 1, 1, :], V[1:(end - 1), 1, 1, :]; dims = 1))
    R, Q = right_orth!(transpose(V, ((1,), (3, 4, 2))); alg = Defaults.alg_orth())
    Π = _split_unit(codomain(Q)[1], _rightunit, storagetype(W))
    Q = (Π * Q)[2:2, :, :, :]
    R = R * Π'
    t = scale!(R[2:end, 1:1], inv(d))
    R = R[2:end, 2:2]

    # other (possibly truncating) algorithms only act on the interaction channels
    if alg != Defaults.alg_orth()
        R, U = right_orth!(R; alg)
        Q = U * Q
    end
    scale!(R, inv(d))

    if dim(R) == 0 # fully truncated
        V = _rightunit ⊞ _rightunit
        Q1 = typeof(WA)(undef, SumSpace{S}() ⊗ P ← domain(WA))
        Q2 = typeof(W.B)(undef, SumSpace{S}() ⊗ P ← P)
    else
        Q = scale!(transpose(Q, ((1, 4), (2, 3))), d)
        Q1 = size(W, 4) == 1 ? typeof(WA)(undef, left_virtualspace(Q) ⊗ P ← P ⊗ SumSpace{S}()) :
            SparseBlockTensorMap(Q[1, 1, 1, 1:(end - 1)])
        Q2 = removeunit(SparseBlockTensorMap(Q[1, 1, 1, end:end]), 4)
        V = ⊞(_rightunit, left_virtualspace(Q), _rightunit)
    end
    return R, t, JordanMPOTensor(V ⊗ P ← domain(W), Q1, Q2, WC, WD)
end

# Compute `W′ * G` for the gauge `G = [1 0 0; 0 R t; 0 0 1]` obtained from `_right_orth_jordan`,
# where `Q` is the tensor to the right of `W′`.
function _absorb_right_gauge(W′::JordanMPOTensor, R, t, Q::JordanMPOTensor)
    W′A, W′B, W′C, W′D = W′.A, W′.B, W′.C, W′.D

    if size(W′, 1) > 1 && dim(R) != 0
        @plansor A′[l p; p' r] := W′A[l p; p' r'] * R[r'; r]
    else
        A′ = similar(W′A, codomain(W′A) ← physicalspace(W′A) ⊗ left_virtualspace(Q.A))
    end

    if size(W′, 1) > 1
        @plansor B′[l p; p' r] := W′A[l p; p' r'] * t[r'; r]
        B′ = add!(removeunit(B′, 4), _dense(W′B))
    else
        B′ = W′B
    end

    if dim(R) != 0
        @plansor C′[p; p' r] := W′C[p; p' r'] * R[r'; r]
    else
        C′ = similar(W′C, codomain(W′C) ← physicalspace(W′) ⊗ left_virtualspace(Q.A))
    end

    @plansor D′[p; p' r] := W′C[p; p' r'] * t[r'; r]
    D′ = add!(removeunit(D′, 3), _dense(W′D))
    return JordanMPOTensor(
        codomain(W′) ← physicalspace(W′) ⊗ left_virtualspace(Q), A′, B′, C′, D′
    )
end

# Isometry that splits the first unit-sector basis vector of `V = ⊞(X)` off into its own block.
function _split_unit(V::SumSpace, unit, ::Type{A}) where {A}
    X = only(V)
    c₀ = only(sectors(unit))
    rest = X ⊖ unit
    T = scalartype(A)
    TT = tensormaptype(spacetype(X), 1, 1, A)
    Π₁ = TT(undef, unit ← X)
    copyto!(block(Π₁, c₀), [one(T) zeros(T, 1, blockdim(X, c₀) - 1)])
    Π₂ = TT(undef, rest ← X)
    for (c, b) in blocks(Π₂)
        Id = Matrix{T}(LinearAlgebra.I, size(b, 1), size(b, 1))
        copyto!(b, c == c₀ ? [zeros(T, size(b, 1)) Id] : Id)
    end
    Π = BlockTensorMap{TT}(undef, (unit ⊞ rest) ← V)
    Π[1, 1] = Π₁
    Π[2, 1] = Π₂
    return Π
end
