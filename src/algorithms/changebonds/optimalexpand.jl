"""
    struct OptimalExpand <: Algorithm end

An algorithm that expands the given mps using the algorithm given in the
[VUMPS paper](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.045145), by
selecting the dominant contributions of a two-site updated MPS tensor, orthogonal to the
original ψ.

# Fields
- `trscheme::TruncationScheme = truncdim(1)` : The truncation scheme to use.
"""
@kwdef struct OptimalExpand <: Algorithm
    trscheme::TruncationScheme = truncdim(1)
end

function changebonds(ψ::InfiniteMPS, H, alg::OptimalExpand, envs=environments(ψ, H))
    T = eltype(ψ.AL)
    AL′ = similar(ψ.AL)
    AR′ = similar(ψ.AR, tensormaptype(spacetype(T), 1, numind(T) - 1, storagetype(T)))
    for i in 1:length(ψ)
        # determine optimal expansion spaces around bond i
        AC2 = _transpose_front(ψ.AC[i]) * _transpose_tail(ψ.AR[i + 1])
        AC2 = ∂∂AC2(i, ψ, H, envs) * AC2

        # Use the nullspaces and SVD decomposition to determine the optimal expansion space
        VL = leftnull(ψ.AL[i])
        VR = rightnull!(_transpose_tail(ψ.AR[i + 1]))
        intermediate = adjoint(VL) * AC2 * adjoint(VR)
        U, _, V, = tsvd!(intermediate; trunc=alg.trscheme, alg=SVD())

        AL′[i] = VL * U
        AR′[i + 1] = V * VR
    end

    newψ = _expand(ψ, AL′, AR′)
    return newψ, envs
end

function changebonds(ψ::InfiniteMPS, H::DenseMPO, alg::OptimalExpand,
                     envs=environments(ψ, H))
    (nmψ, envs) = changebonds(convert(MPSMultiline, ψ), convert(MPOMultiline, H), alg, envs)
    return (convert(InfiniteMPS, nmψ), envs)
end

function changebonds(ψ::MPSMultiline, H, alg::OptimalExpand, envs=environments(ψ, H))
    TL = eltype(ψ.AL)
    AL′ = PeriodicMatrix{TL}(undef, size(ψ.AL))
    TR = tensormaptype(spacetype(TL), 1, numind(TL) - 1, storagetype(TL))
    AR′ = PeriodicMatrix{TR}(undef, size(ψ.AR))

    # determine optimal expansion spaces around bond i
    for i in 1:size(ψ, 1), j in 1:size(ψ, 2)
        AC2 = _transpose_front(ψ.AC[i - 1, j]) * _transpose_tail(ψ.AR[i - 1, j + 1])
        AC2 = ∂∂AC2(i - 1, j, ψ, H, envs) * AC2

        # Use the nullspaces and SVD decomposition to determine the optimal expansion space
        VL = leftnull(ψ.AL[i, j])
        VR = rightnull!(_transpose_tail(ψ.AR[i, j + 1]))
        intermediate = adjoint(VL) * AC2 * adjoint(VR)
        U, _, V, = tsvd!(intermediate; trunc=alg.trscheme, alg=SVD())

        AL′[i, j] = VL * U
        AR′[i, j + 1] = V * VR
    end

    return _expand(ψ, AL′, AR′), envs
end

function changebonds(ψ::AbstractFiniteMPS, H, alg::OptimalExpand, envs=environments(ψ, H))
    return changebonds!(copy(ψ), H, alg, envs)
end
function changebonds!(ψ::AbstractFiniteMPS, H, alg::OptimalExpand, envs=environments(ψ, H))
    #inspired by the infinite mps algorithm, alternative is to use https://arxiv.org/pdf/1501.05504.pdf

    #the idea is that we always want to expand the state in such a way that there are zeros at site i
    #but "optimal vectors" at site i+1
    #so during optimization of site i, you have access to these optimal vectors :)

    for i in 1:(length(ψ) - 1)
        AC2 = _transpose_front(ψ.AC[i]) * _transpose_tail(ψ.AR[i + 1])
        AC2 = ∂∂AC2(i, ψ, H, envs) * AC2

        #Calculate nullspaces for left and right
        NL = leftnull(ψ.AC[i])
        NR = rightnull!(_transpose_tail(ψ.AR[i + 1]))

        #Use this nullspaces and SVD decomposition to determine the optimal expansion space
        intermediate = adjoint(NL) * AC2 * adjoint(NR)
        _, _, V, = tsvd!(intermediate; trunc=alg.trscheme, alg=SVD())

        ar_re = V * NR
        ar_le = zerovector!(similar(ar_re, codomain(ψ.AC[i]) ← space(V, 1)))

        (nal, nc) = leftorth!(catdomain(ψ.AC[i], ar_le); alg=QRpos())
        nar = _transpose_front(catcodomain(_transpose_tail(ψ.AR[i + 1]), ar_re))

        ψ.AC[i] = (nal, nc)
        ψ.AC[i + 1] = (nc, nar)
    end

    return (ψ, envs)
end
