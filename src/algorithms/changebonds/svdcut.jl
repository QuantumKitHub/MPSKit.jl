"
    Truncate a given state using svd
"
@with_kw struct SvdCut <: Algorithm
    trschemes = [notrunc()]
end

function changebonds(state::Union{FiniteMPS{T},MPSComoving{T}},alg::SvdCut) where T<: GenericMPSTensor{Sp,N} where {Sp,N} # made it work for GenericMPSTensor
    state = leftorth(state,renorm=false)

    for i in length(state)-1:-1:1
        a = state[i]#permute(state[i],ntuple(x->x,Val{N}()),(N+1,));
        b = permute(state[i+1],(1,),ntuple(x->x+1,Val{N}()));

        (U,S,V) = tsvd(a*b,trunc=alg.trschemes[mod1(i,end)],alg=TensorKit.SVD());

        state[i] = U*S;
        state[i+1] = permute(V,ntuple(x->x,Val{N}()),(N+1,))
    end

    return state
end

function changebonds(state::InfiniteMPS,H,alg::SvdCut,pars=params(state,H))
    state = changebonds(state,alg);
    return state,pars;
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
