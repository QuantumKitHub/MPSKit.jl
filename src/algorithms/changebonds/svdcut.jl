"""
    struct SvdCut <: Algorithm end

An algorithm that uses truncated SVD to change the bond dimension of a ψ.

# Fields
- `trscheme::TruncationScheme = notrunc()` : The truncation scheme to use.
"""
@kwdef struct SvdCut <: Algorithm
    trscheme::TruncationScheme = notrunc()
end

function changebonds(ψ::AbstractFiniteMPS, alg::SvdCut; kwargs...)
    return changebonds!(copy(ψ), alg; kwargs...)
end
function changebonds!(ψ::AbstractFiniteMPS, alg::SvdCut; normalize::Bool=true)
    for i in (length(ψ) - 1):-1:1
        U, S, V, = tsvd(ψ.CR[i]; trunc=alg.trscheme, alg=TensorKit.SVD())
        AL′ = ψ.AL[i] * U
        ψ.AC[i] = (AL′, complex(S))
        AR′ = _transpose_front(V * _transpose_tail(ψ.AR[i + 1]))
        ψ.AC[i + 1] = (complex(S), AR′)
    end
    return normalize ? normalize!(ψ) : ψ
end

# Note: it might be better to go to an MPS representation first
# such that the SVD cut happens in a canonical form;
# this would still not be the correct norm, so we will ignore this for now.
# this implementation cuts the bond dimension in a non-canonical basis from left to right,
# but then the basis would be somewhat canonical automatically, so in the reverse direction
# we can cut the bond dimension in a canonical basis?
changebonds(mpo::FiniteMPO, alg::SvdCut) = changebonds!(copy(mpo), alg)
function changebonds!(mpo::FiniteMPO, alg::SvdCut)
    # cannot cut a MPO with only one site
    length(mpo) == 1 && return mpo

    # left to right
    O_left = transpose(mpo.opp[1], ((3, 1, 2), (4,)))
    local O_right
    for i in 2:length(mpo)
        U, S, V, = tsvd!(O_left; trunc=alg.trscheme, alg=TensorKit.SVD())
        mpo.opp[i - 1] = transpose(U, ((2, 3), (1, 4)))
        if i < length(mpo)
            @plansor O_left[-3 -1 -2; -4] := S[-1; 1] * V[1; 2] * mpo.opp[i][2 -2; -3 -4]
        else
            @plansor O_right[-1; -3 -4 -2] := S[-1; 1] * V[1; 2] * mpo.opp[end][2 -2; -3 -4]
        end
    end

    # right to left
    for i in (length(mpo) - 1):-1:1
        U, S, V, = tsvd!(O_right; trunc=alg.trscheme, alg=TensorKit.SVD())
        mpo.opp[i + 1] = transpose(V, ((1, 4), (2, 3)))
        if i > 1
            @plansor O_right[-1; -3 -4 -2] := mpo.opp[i][-1 -2; -3 2] * U[2; 1] * S[1; -4]
        else
            @plansor mpo.opp[1][-1 -2; -3 -4] := mpo.opp[1][-1 -2; -3 2] * U[2; 1] *
                                                 S[1; -4]
        end
    end

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

    # make sure everything is full rank:
    makefullrank!(copied)

    # if the bond dimension is not changed, we can keep the same center, otherwise recompute
    ψ = if space(ncr, 1) != space(copied[1], 1)
        InfiniteMPS(copied)
    else
        InfiniteMPS(copied, complex(ncr))
    end
    return normalize!(ψ)
end

function changebonds(ψ, H, alg::SvdCut, envs=environments(ψ, H))
    return (changebonds(ψ, alg), envs)
end
