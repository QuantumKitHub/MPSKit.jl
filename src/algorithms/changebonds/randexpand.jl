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

    "expansion range that is considered for selecting the orthogonal subspace"
    range::Int = 1
end

function changebonds!(ψ::InfiniteMPS, alg::RandExpand)
    # obtain the spaces for the expanded directions by sampling
    virtualspaces = PeriodicVector(
        map(1:length(ψ)) do i
            Vmax = mapreduce(fuse, reverse(0:alg.range)) do j
                j == alg.range ? left_virtualspace(ψ, i - j) : physicalspace(ψ, i - j - 1)
            end ⊖ left_virtualspace(ψ, i)
            return sample_space(Vmax, alg.trscheme)
        end
    )

    # ensure the resulting tensors are full rank (space-wise)
    makefullrank!(virtualspaces, physicalspace(ψ))

    # add vectors orthogonal to the current state
    AL′ = similar(ψ.AL)
    T = eltype(AL′)
    AR′ = similar(ψ.AR, tensormaptype(spacetype(T), 1, numind(T) - 1, storagetype(T)))

    for i in 1:length(ψ)
        VL = left_null(ψ.AL[i])
        XL = similar(VL, right_virtualspace(VL) ← virtualspaces[i + 1])
        foreach(((c, b),) -> TensorKit.one!(b), blocks(XL))
        AL′[i] = VL * XL

        VR = right_null!(_transpose_tail(ψ.AR[i]))
        XR = similar(VR, virtualspaces[i] ← space(VR, 1))
        foreach(((c, b),) -> TensorKit.one!(b), blocks(XR))
        AR′[i] = XR * VR
    end

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

function sample_space(V, strategy)
    S = TensorKit.SectorDict(c => Random.randexp(dim(V, c)) for c in sectors(V))
    ind = MatrixAlgebraKit.findtruncated(S, strategy)
    return TensorKit.Factorizations.truncate_space(V, ind)
end
