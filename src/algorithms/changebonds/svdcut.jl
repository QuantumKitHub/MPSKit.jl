"""
$(TYPEDEF)

An algorithm that uses truncated SVD to change the bond dimension of a state or operator.
This is achieved by a sweeping algorithm that locally performs (optimal) truncations in a gauged basis.

# Fields

$(TYPEDFIELDS)

# Truncation scale

For an `MPO` the norm is gauged out and spread evenly over the sites, so every truncated bond sees the same reference scale `‖O‖^(2/length(mpo))`.
An `MPOHamiltonian` is instead truncated in the Jordan basis with the identity level projected out, where the singular values carry the scale of the interactions crossing that bond.

# See also

Used as the `algorithm` argument of [`changebonds`](@ref) and [`changebonds!`](@ref).

# References

* [Parker et al. Phys. Rev. B 102 (2020)](@cite parker2020)
"""
@kwdef struct SvdCut{S} <: Algorithm
    "algorithm used for the singular value decomposition"
    alg_svd::S = Defaults.alg_svd()

    "algorithm used for [truncation](@extref MatrixAlgebraKit.TruncationStrategy) of the gauge tensors"
    trunc::TruncationStrategy
end

function changebonds(ψ::AbstractFiniteMPS, alg::SvdCut; kwargs...)
    return changebonds!(copy(ψ), alg; kwargs...)
end
function changebonds!(ψ::AbstractFiniteMPS, alg::SvdCut; normalize::Bool = true)
    for i in (length(ψ) - 1):-1:1
        U, S, Vᴴ = svd_trunc(ψ.C[i]; trunc = alg.trunc, alg = alg.alg_svd)
        AL′ = ψ.AL[i] * U
        ψ.AC[i] = (AL′, S)
        AR′ = _transpose_front(Vᴴ * _transpose_tail(ψ.AR[i + 1]))
        ψ.AC[i + 1] = (S, AR′)
    end
    return normalize ? normalize!(ψ) : ψ
end

function changebonds!(ψ::AbstractFiniteMPS, H, alg::SvdCut, envs)
    ψ = changebonds!(ψ, alg)
    recalculate!(envs, ψ, H)
    return ψ, envs
end

changebonds(mpo::FiniteMPO, alg::SvdCut) = changebonds!(copy(mpo), alg)
function changebonds!(mpo::FiniteMPO, alg::SvdCut)
    N = length(mpo)
    N == 1 && return mpo

    # gauge left to right, keeping the total norm out of band
    logS = zero(float(real(scalartype(mpo))))
    local carry
    for i in 1:(N - 1)
        if i == 1
            A = transpose(mpo[1], ((3, 1, 2), (4,)))
        else
            @plansor A[-3 -1 -2; -4] := carry[-1; 1] * mpo[i][1 -2; -3 -4]
        end
        Q, carry = left_orth!(A)
        logS += _extract_norm!(carry)
        @inbounds mpo[i] = transpose(Q, ((2, 3), (1, 4)))
    end
    @plansor A[-1 -2; -3 -4] := carry[-1; 1] * mpo[N][1 -2; -3 -4]
    logS += _extract_norm!(A)
    @inbounds mpo[N] = A

    # spread it evenly, so that every bond of the truncation sweep sees the same scale
    f = exp(logS / N)
    for i in 1:N
        @inbounds scale!(mpo[i], f)
    end

    # truncate right to left, splitting the carry norm evenly across each bond
    O = transpose(mpo[N], ((1,), (3, 4, 2)))
    for i in (N - 1):-1:1
        U, S, Vᴴ = svd_trunc!(O; trunc = alg.trunc, alg = alg.alg_svd)
        _warn_empty_bond(i, space(S, 1))
        n = sqrt(norm(S))
        iszero(n) && (n = one(n))
        @inbounds mpo[i + 1] = transpose(scale!(Vᴴ, n), ((1, 4), (2, 3)))
        if i > 1
            @plansor O[-1; -3 -4 -2] := mpo[i][-1 -2; -3 2] * U[2; 1] * S[1; -4] / n
        else
            @plansor mpo[1][-1 -2; -3 -4] := mpo[1][-1 -2; -3 2] * U[2; 1] * S[1; -4] / n
        end
    end

    return mpo
end

# scale `t` to unit norm, returning the log of the factor that was divided out
function _extract_norm!(t)
    n = norm(t)
    (iszero(n) || !isfinite(n)) && return zero(float(real(typeof(n))))
    scale!(t, inv(n))
    return log(n)
end

function _warn_empty_bond(bond::Int, V)
    dim(V) == 0 && @warn "`SvdCut` truncated the bond between sites $bond and $(bond + 1) down to zero dimensions; the resulting operator is identically zero. Loosen `trunc` or check the scale of the input operator."
    return nothing
end

# TODO: this assumes the MPO is infinite, and does weird things for finite MPOs.
function changebonds(mpo::InfiniteMPO, alg::SvdCut)
    return convert(InfiniteMPO, changebonds(convert(InfiniteMPS, mpo), alg))
end
function changebonds(mpo::MultilineMPO, alg::SvdCut)
    return Multiline(map(Base.Fix2(changebonds, alg), parent(mpo)))
end
function changebonds(ψ::MultilineMPS, alg::SvdCut)
    return Multiline(map(Base.Fix2(changebonds, alg), parent(ψ)))
end
function changebonds(ψ::InfiniteMPS, alg::SvdCut)
    copied = copy.(ψ.AL)
    ncr = ψ.C[1]

    for i in 1:length(ψ)
        U, ncr, = svd_trunc(ψ.C[i]; trunc = alg.trunc, alg = alg.alg_svd)
        copied[i] = copied[i] * U
        copied[i + 1] = _transpose_front(U' * _transpose_tail(copied[i + 1]))
    end

    # make sure everything is full rank:
    makefullrank!(copied)

    # if the bond dimension is not changed, we can keep the same center, otherwise recompute
    ψ = if space(ncr, 1) != space(copied[1], 1)
        InfiniteMPS(copied)
    else
        C₀ = ncr isa TensorMap ? ncr : TensorMap(ncr)
        InfiniteMPS(copied, C₀)
    end
    return normalize!(ψ)
end

function changebonds(ψ, H, alg::SvdCut, envs = nothing)
    newψ = changebonds(ψ, alg)
    return newψ, environments(newψ, H, newψ)
end

changebonds(mpo::FiniteMPOHamiltonian, alg::SvdCut) = changebonds!(copy(mpo), alg)
function changebonds!(H::FiniteMPOHamiltonian, alg::SvdCut)
    # orthogonality center to the left
    for i in length(H):-1:2
        H = right_canonicalize!(H, i)
    end

    # a Jordan bond has a start and a finish level on top of its interaction channels
    channels_before = [dim(left_virtualspace(H, i)) - 2 for i in 2:length(H)]

    # swipe right
    alg_trunc = MatrixAlgebraKit.TruncatedAlgorithm(alg.alg_svd, alg.trunc)
    for i in 1:(length(H) - 1)
        H = left_canonicalize!(H, i; alg = MatrixAlgebraKit.LeftOrthViaSVD(alg_trunc))
    end
    # swipe left -- TODO: do we really need this double sweep?
    for i in length(H):-1:2
        H = right_canonicalize!(H, i; alg = MatrixAlgebraKit.RightOrthViaSVD(alg_trunc))
    end

    emptied = findall(
        i -> channels_before[i] > 0 && dim(left_virtualspace(H, i + 1)) - 2 == 0,
        eachindex(channels_before)
    )
    isempty(emptied) || @warn "`SvdCut` discarded every interaction crossing bond(s) $(emptied .+ 1); those terms are gone from the compressed operator. Loosen `trunc` or check the scale of the input Hamiltonian."

    return H
end
