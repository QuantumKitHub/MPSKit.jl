"""
$(TYPEDEF)

An algorithm that expands the bond dimension by adding random unitary vectors that are
orthogonal to the existing state.

## Fields

$(TYPEDFIELDS)
"""
@kwdef struct RandExpand{S} <: Algorithm
    "algorithm used for the singular value decomposition"
    alg_svd::S = Defaults.alg_svd()

    "algorithm used for [truncation](@extref MatrixAlgebraKit.TruncationStrategy] the expanded space"
    trscheme::TruncationStrategy
end

function changebonds!(ψ::InfiniteMPS, alg::RandExpand)
    AL′ = map(ψ.AL) do A
        # find random orthogonal vectors
        A_perp = randn!(similar(A, codomain(A) ← fuse(codomain(A)) ⊖ right_virtualspace(A)))
        normalize!(add!(A_perp, A * (A' * A_perp), -1))
        A′, _, _ = svd_trunc!(A_perp; alg = alg.alg_svd, trunc = alg.trscheme)
        return A′
    end

    AR′ = PeriodicVector(
        map(enumerate(ψ.AR)) do (i, A)
            At = _transpose_tail(A)
            A_perp = randn!(similar(At, fuse(domain(At)) ⊖ left_virtualspace(A) ← domain(At)))
            normalize!(add!(A_perp, (A_perp * At') * At, -1))
            trunc = truncspace(right_virtualspace(AL′[i - 1]))
            _, _, A′ = svd_trunc!(A_perp; alg = alg.alg_svd, trunc)
            return A′
        end
    )

    return _expand!(ψ, AL′, AR′)
end

function changebonds!(ψ::MultilineMPS, alg::RandExpand)
    return Multiline(map(x -> changebonds!(x, alg), ψ.data))
end

changebonds(ψ::AbstractMPS, alg::RandExpand) = changebonds!(copy(ψ), alg)
changebonds(ψ::MultilineMPS, alg::RandExpand) = changebonds!(copy(ψ), alg)

function changebonds!(ψ::AbstractFiniteMPS, alg::RandExpand)
    for i in 1:(length(ψ) - 1)
        AC2 = randomize!(_transpose_front(ψ.AC[i]) * _transpose_tail(ψ.AR[i + 1]))

        #Calculate nullspaces for left and right
        NL = left_null(ψ.AC[i])
        NR = right_null!(_transpose_tail(ψ.AR[i + 1]; copy = true))

        #Use this nullspaces and SVD decomposition to determine the optimal expansion space
        intermediate = normalize!(adjoint(NL) * AC2 * adjoint(NR))
        _, _, Vᴴ = svd_trunc!(intermediate; trunc = alg.trscheme, alg = alg.alg_svd)

        ar_re = Vᴴ * NR
        ar_le = zerovector!(similar(ar_re, codomain(ψ.AC[i]) ← space(Vᴴ, 1)))

        nal, nc = qr_compact!(catdomain(ψ.AC[i], ar_le))
        nar = _transpose_front(catcodomain(_transpose_tail(ψ.AR[i + 1]), ar_re))

        ψ.AC[i] = (nal, nc)
        ψ.AC[i + 1] = (nc, nar)
    end

    return normalize!(ψ)
end
