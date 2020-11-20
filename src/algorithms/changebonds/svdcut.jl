"
    Truncate a given state using svd
"
@with_kw struct SvdCut <: Algorithm
    trscheme = notrunc()
end

function changebonds!(state::Union{FiniteMPS{T},MPSComoving{T}},alg::SvdCut) where T<: GenericMPSTensor{Sp,N} where {Sp,N} # made it work for GenericMPSTensor
    for i in length(state)-1:-1:1
        a = state.AC[i]
        b = _permute_tail(state.AR[i+1]);

        (U,S,V) = tsvd(a*b,trunc=alg.trscheme,alg=TensorKit.SVD());

        state.AC[i] = (U,complex(S));
        state.AC[i+1] = (complex(S),_permute_front(V));
    end

    return state
end

changebonds!(state::PeriodicMPO,alg::SvdCut) = copyto!(state,changebonds!(convert(MPSMultiline,state),alg))
changebonds!(state::InfiniteMPS,alg::SvdCut) = copyto!(state,changebonds!(convert(MPSMultiline,state),alg))
function changebonds!(state::MPSMultiline,alg::SvdCut)
    for i in 1:size(state,1),
        j in 1:size(state,2)

        (U,state.CR[i,j],V) = tsvd(state.CR[i,j],trunc=alg.trscheme,alg=TensorKit.SVD());
        state.AL[i,j] = state.AL[i,j]*U
        state.AL[i,j+1] = _permute_front(U'*_permute_tail(state.AL[i,j+1]))
    end

    reorth!(state)
end

function changebonds!(state,H,alg::SvdCut,envs=environments(state,H))
    changebonds!(state,alg);
    envs isa AbstractInfEnv && recalculate!(envs,state);

    return state,envs;
end
