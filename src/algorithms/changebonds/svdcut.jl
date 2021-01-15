"
    Truncate a given state using svd
"
@with_kw struct SvdCut <: Algorithm
    trscheme = notrunc()
end

changebonds(state::Union{FiniteMPS{T},MPSComoving{T}},alg::SvdCut) where T = changebonds!(copy(state),alg);
function changebonds!(state::Union{FiniteMPS{T},MPSComoving{T}},alg::SvdCut) where T
    newstate = copy(state);

    for i in length(state)-1:-1:1
        (U,S,V) = tsvd(state.CR[i],trunc=alg.trscheme,alg=TensorKit.SVD());

        state.AC[i] = (state.AL[i]*U,complex(S));
        state.AC[i+1] = (complex(S),_permute_front(V*_permute_tail(state.AR[i+1])));
    end

    return state
end

changebonds(state::PeriodicMPO,alg::SvdCut) = convert(PeriodicMPO,changebonds(convert(MPSMultiline,state),alg))
changebonds(state::InfiniteMPS,alg::SvdCut) = convert(InfiniteMPS,changebonds(convert(MPSMultiline,state),alg))
function changebonds(state::MPSMultiline,alg::SvdCut)
    copied = copy(state.AL);
    for i in 1:size(state,1),
        j in 1:size(state,2)

        (U,state.CR[i,j],V) = tsvd(state.CR[i,j],trunc=alg.trscheme,alg=TensorKit.SVD());
        copied[i,j] = copied[i,j]*U
        copied[i,j+1] = _permute_front(U'*_permute_tail(copied[i,j+1]))
    end
    MPSMultiline(copied)
end

changebonds(state,H,alg::SvdCut,envs=environments(state,H)) = (changebonds(state,alg),envs)
