"
    expands the given mps using the algorithm given in the vumps paper
"
@with_kw struct OptimalExpand<:Algorithm
    trscheme::TruncationScheme = truncdim(1)
end

function changebonds(state::InfiniteMPS, H::Hamiltonian,alg::OptimalExpand,pars=params(state,H))
    for i in 1:length(state)
        ACAR = _permute_front(state.AC[i])*_permute_tail(state.AR[i+1])
        AC2 = ac2_prime(ACAR,i,state,pars)

        if norm(AC2) == 0 #we cannot really optimally expand
            @info "AC2 == 0; using rand"
            AC2 = TensorMap(rand,eltype(AC2),codomain(AC2),domain(AC2))
        end

        #Calculate nullspaces for AL and AR
        NL = leftnull(state.AL[i])
        NR = rightnull(_permute_tail(state.AR[i+1]))

        #Use this nullspaces and SVD decomposition to determine the optimal expansion space
        intermediate = adjoint(NL)*AC2*adjoint(NR)

        (U,S,V) = tsvd(intermediate,trunc=alg.trscheme,alg=TensorKit.SVD())

        #expand AL
        al_re = NL*U
        state.AL[i] = TensorKit.catdomain(state.AL[i],al_re)

        al_le = TensorMap(zeros,space(U,2)',domain(_permute_tail(state.AL[i+1])))
        state.AL[i+1] = _permute_front(TensorKit.catcodomain(_permute_tail(state.AL[i+1]),al_le))

        #expand AR
        ar_re = V*NR
        state.AR[i+1] = _permute_front(TensorKit.catcodomain(_permute_tail(state.AR[i+1]),ar_re))

        ar_le = TensorMap(zeros,codomain(state.AR[i]),space(U,2)')
        state.AR[i] = TensorKit.catdomain(state.AR[i],ar_le)

        #fix C
        le = TensorMap(zeros,space(U,2)',space(state.CR[i],1))
        state.CR[i] = TensorKit.catcodomain(state.CR[i],le)
        re = TensorMap(zeros,space(state.CR[i],1),space(U,2)')
        state.CR[i] = TensorKit.catdomain(state.CR[i],re)

        state.AC[i] = state.AL[i]*state.CR[i]
        state.AC[i+1] = state.AL[i+1]*state.CR[i+1]

        #we should update the params "properly", and not in this wasteful way (params don't change after all)
        poison!(pars);
    end
    return state,pars
end

function changebonds(state::Union{FiniteMPS,MPSComoving}, H::Hamiltonian,alg::OptimalExpand,pars=params(state,H))
    #inspired by the infinite mps algorithm, alternative is to use https://arxiv.org/pdf/1501.05504.pdf
    #didn't use the paper because generically it'd be annoying to implement (again having to fuse and stuff)

    #the idea is that we always want to expand the state in such a way that there are zeros at site i
    #but "optimal vectors" at site i+1
    #so during optimization of site i, you have access to these optimal vectors :)

    for i in 1:(length(state)-1)
        ACAR = _permute_front(state.AC[i])*_permute_tail(state.AR[i+1])
        AC2 = ac2_prime(ACAR,i,state,pars)

        #Calculate nullspaces for left and right
        NL = leftnull(state.AC[i])
        NR = rightnull(_permute_tail(state.AR[i+1]))

        #Use this nullspaces and SVD decomposition to determine the optimal expansion space
        intermediate = adjoint(NL) * AC2 * adjoint(NR);
        (U,S,V) = tsvd(intermediate,trunc=alg.trscheme,alg=TensorKit.SVD())

        ar_re = V*NR;
        ar_le = TensorMap(zeros,codomain(state.AC[i]),space(V,1))

        state.AR[i+1] = _permute_front(TensorKit.catcodomain(_permute_tail(state.AR[i+1]),ar_re))
        state.AC[i] = TensorKit.catdomain(state.AC[i],ar_le)
    end

    return state,pars
end
