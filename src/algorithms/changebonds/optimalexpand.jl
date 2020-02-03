"expands the given mps using the algorithm given in the vumps paper"
@with_kw struct OptimalExpand<:Algorithm
    trscheme::TruncationScheme = truncdim(1)
end

function changebonds(state::MpsCenterGauged, H::Hamiltonian,alg::OptimalExpand,pars=params(state,H))
    for i in 1:length(state)
        @tensor ACAR[-1 -2;-3 -4]:=state.AC[i][-1,-2,1]*state.AR[i+1][1,-3,-4]
        AC2 = ac2_prime(ACAR,i,state,pars)

        #Calculate nullspaces for AL and AR
        NL = leftnull(state.AL[i],(1,2),(3,))
        NR = rightnull(state.AR[i+1],(1,),(2,3))

        #Use this nullspaces and SVD decomposition to determine the optimal expansion space
        @tensor intermediate[-1;-2]:=conj(NL[1,2,-1])*AC2[1,2,3,4]*conj(NR[-2,3,4])
        (U,S,V) = svd(intermediate,trunc=alg.trscheme,alg=TensorKit.SVD())

        #expand AL
        @tensor al_re[-1 -2;-3]:=NL[-1,-2,1]*U[1,-3]
        state.AL[i]=TensorKit.catdomain(state.AL[i],al_re)

        al_le=TensorMap(zeros,space(U,2)',space(state.AL[i+1],2)'*space(state.AL[i+1],3)')
        state.AL[i+1]=permuteind(TensorKit.catcodomain(permuteind(state.AL[i+1],(1,),(2,3)),al_le),(1,2,),(3,))

        #expand AR
        @tensor ar_re[-1;-2 -3]:=V[-1,1]*NR[1,-2,-3]
        state.AR[i+1]=permuteind(TensorKit.catcodomain(permuteind(state.AR[i+1],(1,),(2,3)),ar_re),(1,2),(3,))
        ar_le=TensorMap(zeros,space(state.AR[i],1)*space(state.AR[i],2),space(U,2)')
        state.AR[i]=TensorKit.catdomain(state.AR[i],ar_le)

        #fix C
        le=TensorMap(zeros,space(U,2)',space(state.CR[i],1))
        state.CR[i]=TensorKit.catcodomain(state.CR[i],le)
        re=TensorMap(zeros,space(state.CR[i],1),space(U,2)')
        state.CR[i]=TensorKit.catdomain(state.CR[i],re)

        state.AC[i]=state.AL[i]*state.CR[i]
        state.AC[i+1]=state.AL[i+1]*state.CR[i+1]

        #we should update the params "properly", and not in this wasteful way (params don't change after all)
        pars=params(state,H)
    end
    return state,pars
end

function changebonds(state::Union{FiniteMps,MpsComoving}, H::Hamiltonian,alg::OptimalExpand,pars=params(state,H))
    #inspired by the infinite mps algorithm, alternative is to use https://arxiv.org/pdf/1501.05504.pdf
    #didn't use the paper because generically it'd be annoying to implement (again having to fuse and stuff)

    #the idea is that we always want to expand the state in such a way that there are zeros at site i
    #but "optimal vectors" at site i+1
    #so during optimization of site i, you have access to these optimal vectors :)

    for i in 1:(length(state)-1)
        @tensor ACAR[-1 -2;-3 -4]:=state[i][-1,-2,1]*state[i+1][1,-3,-4]
        AC2=ac2_prime(ACAR,i,state,pars)

        #Calculate nullspaces for left and right
        NL = leftnull(state[i],(1,2),(3,))
        NR = rightnull(state[i+1],(1,),(2,3))

        #Use this nullspaces and SVD decomposition to determine the optimal expansion space
        @tensor intermediate[-1;-2]:=conj(NL[1,2,-1])*AC2[1,2,3,4]*conj(NR[-2,3,4])
        (U,S,V) = svd(intermediate,trunc=alg.trscheme,alg=TensorKit.SVD())

        @tensor ar_re[-1;-2 -3]:=V[-1,1]*NR[1,-2,-3]
        state[i+1]=permuteind(TensorKit.catcodomain(permuteind(state[i+1],(1,),(2,3)),ar_re),(1,2),(3,))
        ar_le=TensorMap(zeros,space(state[i],1)*space(state[i],2),space(V,1))
        state[i]=TensorKit.catdomain(state[i],ar_le)

        (state[i],C)=TensorKit.leftorth(state[i],(1,2),(3,))
        @tensor state[i+1][-1 -2;-3] := C[-1,1]*state[i+1][1,-2,-3]
    end

    state = rightorth(state)
    return state,pars
end
