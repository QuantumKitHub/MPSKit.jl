"""
$(TYPEDEF)

An algorithm that expands the bond dimension by adding random unitary vectors that are
orthogonal to the existing state. This means that additional directions are added to
`AL` and `AR` that are contained in the nullspace of both. Note that this is happens in
parallel, and therefore the expansion will never go beyond the local two-site subspace.

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
        XR = randisometry(storagetype(VL), space(VR, 1) ← V)
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

function sample_space(V, strategy)
    S = TensorKit.SectorDict(c => sort!(Random.rand(dim(V, c)); rev = true) for c in sectors(V))
    ind = MatrixAlgebraKit.findtruncated_svd(S, strategy)
    return TensorKit.Factorizations.truncate_space(V, ind)
end


"""
$(TYPEDEF)

An algorithm that expands the bond dimension by adding random unitary vectors that are
orthogonal to the existing state, in a sweeping fashion. Additionally, some random noise
is added to the state in order for it to remain gauge-able.

## Fields

$(TYPEDFIELDS)
"""
@kwdef struct RandPerturbedExpand{S} <: Algorithm
    "algorithm used for the singular value decomposition"
    alg_svd::S = Defaults.alg_svd()

    "algorithm used for [truncation](@extref MatrixAlgebraKit.TruncationStrategy] the expanded space"
    trscheme::TruncationStrategy

    "amount of noise that is added to the current state"
    noisefactor::Float64 = eps()^(3 / 4)

    "algorithm used for gauging the state"
    alg_gauge = Defaults.alg_gauge(; dynamic_tols = false)
end

function changebonds!(ψ::InfiniteMPS, alg::RandPerturbedExpand)
    for i in 1:length(ψ)
        # obtain space by sampling the support of left nullspace
        # add (orthogonal) directions as isometries in that direction
        VL = left_null(ψ.AL[i])
        V = sample_space(right_virtualspace(VL), alg.trscheme)
        XL = randisometry(scalartype(VL), right_virtualspace(VL) ← V)
        ψ.AL[i] = catdomain(ψ.AL[i], VL * XL)

        # make sure the next site fits, by "absorbing" into a larger tensor
        # with some random noise to ensure state is still gauge-able
        AL = ψ.AL[i + 1]
        AL′ = similar(AL, right_virtualspace(ψ.AL[i]) ⊗ physicalspace(AL) ← right_virtualspace(AL))
        scale!(randn!(AL′), alg.noisefactor)
        ψ.AL[i + 1] = TensorKit.absorb!(AL′, AL)
    end

    # properly regauge the state:
    makefullrank!(ψ.AL)
    ψ.AR .= similar.(ψ.AL)
    # ψ.AC .= similar.(ψ.AL)

    # initial guess for gauge is embedded original C
    C₀ = similar(ψ.C[0], right_virtualspace(ψ.AL[end]) ← left_virtualspace(ψ.AL[1]))
    absorb!(id!(C₀), ψ.C[0])

    gaugefix!(ψ, ψ.AL, C₀; order = :R, alg.alg_gauge.maxiter, alg.alg_gauge.tol)

    for i in reverse(1:length(ψ))
        # obtain space by sampling the support of left nullspace
        # add (orthogonal) directions as isometries in that direction
        AR_tail = _transpose_tail(ψ.AR[i])
        VR = right_null(AR_tail)
        V = sample_space(space(VR, 1), alg.trscheme)
        XR = randisometry(scalartype(VR), space(VR, 1) ← V)
        ψ.AR[i] = _transpose_front(catcodomain(AR_tail, XR' * VR))

        # make sure the next site fits, by "absorbing" into a larger tensor
        # with some random noise to ensure state is still gauge-able
        AR = ψ.AR[i - 1]
        AR′ = similar(AR, left_virtualspace(AR) ⊗ physicalspace(AR) ← left_virtualspace(ψ.AR[i]))
        scale!(randn!(AR′), alg.noisefactor)
        ψ.AR[i - 1] = TensorKit.absorb!(AR′, AR)
    end

    # properly regauge the state:
    makefullrank!(ψ.AR)
    ψ.AL .= similar.(ψ.AR)
    ψ.AC .= similar.(ψ.AR)

    # initial guess for gauge is embedded original C
    C₀ = similar(ψ.C[0], right_virtualspace(ψ.AR[end]) ← left_virtualspace(ψ.AR[1]))
    absorb!(id!(C₀), ψ.C[0])

    gaugefix!(ψ, ψ.AR, C₀; order = :LR, alg.alg_gauge.maxiter, alg.alg_gauge.tol)
    mul!.(ψ.AC, ψ.AL, ψ.C)

    return ψ
end

changebonds(ψ::InfiniteMPS, alg::RandPerturbedExpand) = changebonds!(copy(ψ), alg)