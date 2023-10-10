"""
    struct SvdCut <: Algorithm end

An algorithm that uses truncated SVD to change the bond dimension of a Ψ.

# Fields
- `trscheme::TruncationScheme = notrunc()` : The truncation scheme to use.
"""
@kwdef struct SvdCut <: Algorithm
    trscheme::TruncationScheme = notrunc()
end

changebonds(Ψ::AbstractFiniteMPS, alg::SvdCut) = changebonds!(copy(Ψ), alg)
function changebonds!(Ψ::AbstractFiniteMPS, alg::SvdCut)
    for i in (length(Ψ) - 1):-1:1
        U, S, V, = tsvd(Ψ.CR[i]; trunc=alg.trscheme, alg=TensorKit.SVD())
        AL′ = Ψ.AL[i] * U
        Ψ.AC[i] = (AL′, complex(S))
        AR′ = _transpose_front(V' * _transpose_tail(Ψ.AR[i + 1]))
        Ψ.AC[i + 1] = (complex(S), AR′)
    end
    return normalize!(Ψ)
end

function changebonds(Ψ::DenseMPO, alg::SvdCut)
    return convert(DenseMPO, changebonds(convert(InfiniteMPS, Ψ), alg))
end
function changebonds(Ψ::MPOMultiline, alg::SvdCut)
    return convert(MPOMultiline, changebonds(convert(MPSMultiline, Ψ), alg))
end
function changebonds(Ψ::MPSMultiline, alg::SvdCut)
    return Multiline(map(x -> changebonds(x, alg), Ψ.data))
end
function changebonds(Ψ::InfiniteMPS, alg::SvdCut)
    copied = complex.(Ψ.AL)
    ncr = Ψ.CR[1]

    for i in 1:length(Ψ)
        U, ncr, = tsvd(Ψ.CR[i]; trunc=alg.trscheme, alg=TensorKit.SVD())
        copied[i] = copied[i] * U
        copied[i + 1] = _transpose_front(U' * _transpose_tail(copied[i + 1]))
    end

    return InfiniteMPS(copied, complex(ncr))
end

function changebonds(Ψ, H, alg::SvdCut, envs=environments(Ψ, H))
    return (changebonds(Ψ, alg), envs)
end
