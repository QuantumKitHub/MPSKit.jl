"""
$(TYPEDEF)

An algorithm that expands the bond dimension by adding random unitary vectors that are
orthogonal to the existing state. This means that additional directions are added to
`AL` and `AR` that are contained in the nullspace of both. Note that this is happens in
parallel, and therefore the expansion will never go beyond the local two-site subspace.

The truncation strategy dictates the number of expanded states, by generating uniformly
distributed weights for each state in the two-site space and truncating that.

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
    T = eltype(ψ.AL)
    AL′ = similar(ψ.AL)
    AR′ = similar(ψ.AR, tensormaptype(spacetype(T), 1, numind(T) - 1, storagetype(T)))

    for i in 1:length(ψ)
        # obtain spaces by sampling the support of both the left and right nullspace
        VL = left_null(ψ.AL[i])
        VR = right_null!(_transpose_tail(ψ.AR[i + 1]; copy = true))
        V = sample_space(infimum(right_virtualspace(VL), space(VR, 1)), alg.trscheme)

        # obtain (orthogonal) directions as isometries in that direction
        XL = randisometry(scalartype(VL), right_virtualspace(VL) ← V)
        AL′[i] = VL * XL
        XR = randisometry(scalartype(VR), space(VR, 1) ← V)
        AR′[i + 1] = XR * VR
    end

    return _expand!(ψ, AL′, AR′)
end

function changebonds!(ψ::MultilineMPS, alg::RandExpand)
    foreach(Base.Fix2(changebonds!, alg), ψ.data)
    return ψ
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

"""
    sample_space(V, strategy)

Sample basis states within a given `V::VectorSpace` by creating weights for each state that
are distributed uniformly, and then truncating according to the given `strategy`.
"""
function sample_space(V, strategy)
    S = TensorKit.SectorDict(c => Random.rand(dim(V, c)) for c in sectors(V))
    ind = MatrixAlgebraKit.findtruncated(S, strategy)
    return TensorKit.Factorizations.truncate_space(V, ind)
end
