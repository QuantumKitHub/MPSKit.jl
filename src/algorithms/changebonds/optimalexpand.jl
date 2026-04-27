"""
$(TYPEDEF)

An algorithm that expands the given mps as described in
[Zauner-Stauber et al. Phys. Rev. B 97 (2018)](@cite zauner-stauber2018), by selecting the
dominant contributions of a two-site updated MPS tensor, orthogonal to the original ψ.

## Fields

$(TYPEDFIELDS)
"""
@kwdef struct OptimalExpand{S} <: Algorithm
    "algorithm used for the singular value decomposition"
    alg_svd::S = Defaults.alg_svd()

    "algorithm used for truncating the expanded space"
    trscheme::TruncationStrategy
end

function changebonds(
        ψ::InfiniteMPS, H::InfiniteMPOHamiltonian, alg::OptimalExpand,
        envs = environments(ψ, H)
    )
    T = eltype(ψ.AL)
    AL′ = similar(ψ.AL)
    AR′ = similar(ψ.AR, tensormaptype(spacetype(T), 1, numind(T) - 1, storagetype(T)))
    for i in 1:length(ψ)
        # determine optimal expansion spaces around bond i
        AC2 = AC2_projection(i, ψ, H, ψ, envs; kind = :ACAR)

        # Use the nullspaces and SVD decomposition to determine the optimal expansion space
        VL = left_null(ψ.AL[i])
        VR = right_null!(_transpose_tail(ψ.AR[i + 1]; copy = true))
        intermediate = normalize!(adjoint(VL) * AC2 * adjoint(VR))
        U, _, Vᴴ = svd_trunc!(intermediate; trunc = alg.trscheme, alg = alg.alg_svd)

        AL′[i] = VL * U
        AR′[i + 1] = Vᴴ * VR
    end

    newψ = _expand(ψ, AL′, AR′)
    recalculate!(envs, newψ, H)
    return newψ, envs
end

function changebonds(ψ::MultilineMPS, H, alg::OptimalExpand, envs = environments(ψ, H))
    TL = eltype(ψ.AL)
    AL′ = PeriodicMatrix{TL}(undef, size(ψ.AL))
    TR = tensormaptype(spacetype(TL), 1, numind(TL) - 1, storagetype(TL))
    AR′ = PeriodicMatrix{TR}(undef, size(ψ.AR))

    # determine optimal expansion spaces around bond i
    for i in 1:size(ψ, 1), j in 1:size(ψ, 2)
        AC2 = AC2_projection(CartesianIndex(i - 1, j), ψ, H, ψ, envs; kind = :ACAR)

        # Use the nullspaces and SVD decomposition to determine the optimal expansion space
        VL = left_null(ψ.AL[i, j])
        VR = right_null!(_transpose_tail(ψ.AR[i, j + 1]; copy = true))
        intermediate = normalize!(adjoint(VL) * AC2 * adjoint(VR))
        U, _, Vᴴ = svd_trunc!(intermediate; trunc = alg.trscheme, alg = alg.alg_svd)

        AL′[i, j] = VL * U
        AR′[i, j + 1] = Vᴴ * VR
    end

    newψ = _expand(ψ, AL′, AR′)
    recalculate!(envs, newψ, H)
    return newψ, envs
end

function changebonds(ψ::AbstractFiniteMPS, H, alg::OptimalExpand, envs = environments(ψ, H))
    return changebonds!(copy(ψ), H, alg, envs)
end
function changebonds!(ψ::AbstractFiniteMPS, H, alg::OptimalExpand, envs = environments(ψ, H))
    #inspired by the infinite mps algorithm, alternative is to use https://arxiv.org/pdf/1501.05504.pdf

    #the idea is that we always want to expand the state in such a way that there are zeros at site i
    #but "optimal vectors" at site i+1
    #so during optimization of site i, you have access to these optimal vectors :)

    for i in 1:(length(ψ) - 1)
        AC2 = AC2_projection(i, ψ, H, ψ, envs)

        #Calculate nullspaces for left and right
        NL = left_null(ψ.AC[i])
        NR = right_null!(_transpose_tail(ψ.AR[i + 1]; copy = true))

        #Use this nullspaces and SVD decomposition to determine the optimal expansion space
        intermediate = normalize!(adjoint(NL) * AC2 * adjoint(NR))
        _, _, V, = svd_trunc!(intermediate; trunc = alg.trscheme, alg = alg.alg_svd)

        ar_re = V * NR
        ar_le = zerovector!(similar(ar_re, codomain(ψ.AC[i]) ← space(V, 1)))

        nal, nc = qr_compact!(catdomain(ψ.AC[i], ar_le))
        nar = _transpose_front(catcodomain(_transpose_tail(ψ.AR[i + 1]), ar_re))

        ψ.AC[i] = (nal, nc)
        ψ.AC[i + 1] = (nc, nar)
    end

    return (ψ, envs)
end
