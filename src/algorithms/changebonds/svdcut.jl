"
    Truncate a given state using svd
"
@kwdef struct SvdCut <: Algorithm
    trscheme = notrunc()
end

changebonds(state::AbstractFiniteMPS, alg::SvdCut) = changebonds!(copy(state), alg);
function changebonds!(state::AbstractFiniteMPS, alg::SvdCut)
    for i in (length(state) - 1):-1:1
        (U, S, V) = tsvd(state.CR[i]; trunc=alg.trscheme, alg=TensorKit.SVD())

        state.AC[i] = (state.AL[i] * U, complex(S))
        state.AC[i + 1] = (complex(S),
                           _transpose_front(V * _transpose_tail(state.AR[i + 1])))
    end

    return state
end

function changebonds(state::DenseMPO, alg::SvdCut)
    return convert(DenseMPO, changebonds(convert(InfiniteMPS, state), alg))
end;
function changebonds(state::MPOMultiline, alg::SvdCut)
    return convert(MPOMultiline, changebonds(convert(MPSMultiline, state), alg))
end
function changebonds(state::MPSMultiline, alg::SvdCut)
    return Multiline(map(x -> changebonds(x, alg), state.data))
end
function changebonds(state::InfiniteMPS, alg::SvdCut)
    copied = complex.(state.AL)
    ncr = state.CR[1]

    for i in 1:length(state)
        (U, ncr, V) = tsvd(state.CR[i]; trunc=alg.trscheme, alg=TensorKit.SVD())
        copied[i] = copied[i] * U
        copied[i + 1] = _transpose_front(U' * _transpose_tail(copied[i + 1]))
    end

    return InfiniteMPS(copied, complex(ncr))
end

function changebonds(state, H, alg::SvdCut, envs=environments(state, H))
    return (changebonds(state, alg), envs)
end
