"""
    struct RandExpand <: Algorithm end

An algorithm that expands the bond dimension by adding random unitary vectors that are
orthogonal to the existing state. This is achieved by performing a truncated SVD on a random
two-site MPS tensor, which is made orthogonal to the existing state.

# Fields
- `trscheme::TruncationScheme = truncdim(1)` : The truncation scheme to use.
"""
@kwdef struct RandExpand <: Algorithm
    trscheme::TruncationScheme = truncdim(1)
end

function changebonds(ψ::InfiniteMPS, alg::RandExpand)
    T = eltype(ψ.AL)
    AL′ = similar(ψ.AL)
    AR′ = similar(ψ.AR, tensormaptype(spacetype(T), 1, numind(T) - 1, storagetype(T)))
    for i in 1:length(ψ)
        # determine optimal expansion spaces around bond i
        AC2 = randomize!(_transpose_front(ψ.AC[i]) * _transpose_tail(ψ.AR[i + 1]))

        # Use the nullspaces and SVD decomposition to determine the optimal expansion space
        VL = leftnull(ψ.AL[i])
        VR = rightnull!(_transpose_tail(ψ.AR[i + 1]))
        intermediate = adjoint(VL) * AC2 * adjoint(VR)
        U, _, V, = tsvd!(intermediate; trunc=alg.trscheme, alg=SVD())

        AL′[i] = VL * U
        AR′[i + 1] = V * VR
    end

    return _expand(ψ, AL′, AR′)
end

function changebonds(ψ::MPSMultiline, alg::RandExpand)
    return Multiline(map(x -> changebonds(x, alg), ψ.data))
end

changebonds(ψ::AbstractFiniteMPS, alg::RandExpand) = changebonds!(copy(ψ), alg)
function changebonds!(ψ::AbstractFiniteMPS, alg::RandExpand)
    for i in 1:(length(ψ) - 1)
        AC2 = randomize!(_transpose_front(ψ.AC[i]) * _transpose_tail(ψ.AR[i + 1]))

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

    return normalize!(ψ)
end
