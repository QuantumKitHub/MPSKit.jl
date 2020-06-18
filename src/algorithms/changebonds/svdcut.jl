"
    Truncate a given state using svd
"
@with_kw struct SvdCut <: Algorithm
    trschemes = [notrunc()]
end

function changebonds(state::Union{FiniteMPS{T},MPSComoving{T}},alg::SvdCut) where T<: GenericMPSTensor{Sp,N} where {Sp,N} # made it work for GenericMPSTensor
    for i in length(state)-1:-1:1
        a = state.AC[i]
        b = _permute_tail(state.AR[i+1]);

        (U,S,V) = tsvd(a*b,trunc=alg.trschemes[mod1(i,end)],alg=TensorKit.SVD());

        state.AC[i] = (U,complex(S));
        state.AC[i+1] = (complex(S),_permute_front(V));
    end

    return state
end

function changebonds(state::InfiniteMPS,alg::SvdCut)
    nAL = copy(state.AL); #not actually left orthonormalized

    for i in 1:length(state)
        (U,S,V) = tsvd(state.CR[i],trunc=alg.trschemes[mod1(i,end)],alg=TensorKit.SVD());
        @tensor nAL[i][-1 -2;-3]:=nAL[i][-1,-2,1]*U[1,-3]
        @tensor nAL[i+1][-1 -2;-3]:=conj(U[1,-1])*nAL[i+1][1,-2,-3]
    end

    return InfiniteMPS(nAL)
end

function changebonds(state,H,alg::SvdCut,pars=params(state,H))
    state = changebonds(state,alg);
    return state,pars;
end
