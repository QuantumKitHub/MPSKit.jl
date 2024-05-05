"""
    struct SvdCut <: Algorithm end

An algorithm that uses truncated SVD to change the bond dimension of a ψ.

# Fields
- `trscheme::TruncationScheme = notrunc()` : The truncation scheme to use.
"""
@kwdef struct SvdCut <: Algorithm
    trscheme::TruncationScheme = notrunc()
end

changebonds(ψ::AbstractFiniteMPS, alg::SvdCut) = changebonds!(copy(ψ), alg)
function changebonds!(ψ::AbstractFiniteMPS, alg::SvdCut)
    for i in (length(ψ) - 1):-1:1
        U, S, V, = tsvd(ψ.CR[i]; trunc=alg.trscheme, alg=TensorKit.SVD())
        AL′ = ψ.AL[i] * U
        ψ.AC[i] = (AL′, complex(S))
        AR′ = _transpose_front(V * _transpose_tail(ψ.AR[i + 1]))
        ψ.AC[i + 1] = (complex(S), AR′)
    end
    return normalize!(ψ)
end

changebonds(mpo::FiniteMPS, alg::SvdCut) = changebonds!(copy(mpo), alg)
function changebonds!(mpo::FiniteMPO, alg::SvdCut)
    # left to right
    O_left = transpose(mpo.opp[1], (3, 1, 2), (4,))
    for i in 2:length(mpo)
        U, S, V, = tsvd!(O_left; trunc=alg.trscheme, alg=TensorKit.SVD())
        mpo.opp[i - 1] = transpose(U, (2, 3), (1, 4))
        if i < length(mpo)
            @plansor O_left[-3 -1 -2; -4] := S[-1; 1] * V[1; 2] * mpo.opp[i][2 -2; -3 -4]
        end
    end

    # right to left
    @plansor O_right[-1; -3 -4 -2] := S[-1; 1] * V[1; 2] * mpo.opp[end][2 -2; -3 -4]
    for i in (length(mpo) - 1):-1:2
        U, S, V, = tsvd!(O_right; trunc=alg.trscheme, alg=TensorKit.SVD())
        opp′[i + 1] = transpose(V, (1, 4), (2, 3))
        if i > 2
            @plansor O_right[-1; -3 -4 -2] := mpo.opp[i][-1 -2; -3 2] * U[2; 1] * S[1; -4]
        end
    end

    @plansor mpo.opp[1][-1 -2; -3 -4] := mpo.opp[1][-1 -2; -3 2] * U[2; 1] * S[1; -4]
    return mpo
end

# TODO: this assumes the MPO is infinite, and does weird things for finite MPOs.
function changebonds(ψ::DenseMPO, alg::SvdCut)
    return convert(DenseMPO, changebonds(convert(InfiniteMPS, ψ), alg))
end
function changebonds(ψ::MPOMultiline, alg::SvdCut)
    return convert(MPOMultiline, changebonds(convert(MPSMultiline, ψ), alg))
end
function changebonds(ψ::MPSMultiline, alg::SvdCut)
    return Multiline(map(x -> changebonds(x, alg), ψ.data))
end
function changebonds(ψ::InfiniteMPS, alg::SvdCut)
    copied = complex.(ψ.AL)
    ncr = ψ.CR[1]

    for i in 1:length(ψ)
        U, ncr, = tsvd(ψ.CR[i]; trunc=alg.trscheme, alg=TensorKit.SVD())
        copied[i] = copied[i] * U
        copied[i + 1] = _transpose_front(U' * _transpose_tail(copied[i + 1]))
    end

    return normalize!(InfiniteMPS(copied, complex(ncr)))
end

function changebonds(ψ, H, alg::SvdCut, envs=environments(ψ, H))
    return (changebonds(ψ, alg), envs)
end
