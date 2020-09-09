"
    expands the given mps using the algorithm given in the vumps paper
"
@with_kw struct OptimalExpand<:Algorithm
    trscheme::TruncationScheme = truncdim(1)
end


function changebonds(state::InfiniteMPS, H::Hamiltonian,alg::OptimalExpand,pars=params(state,H))
    #determine optimal expansion spaces around bond i
    exps = map(1:length(state)) do i
        ACAR = MPSKit._permute_front(state.AC[i])*MPSKit._permute_tail(state.AR[i+1])
        AC2 = ac2_prime(ACAR,i,state,pars)

        #Calculate nullspaces for AL and AR
        NL = leftnull(state.AL[i])
        NR = rightnull(MPSKit._permute_tail(state.AR[i+1]))

        #Use this nullspaces and SVD decomposition to determine the optimal expansion space
        intermediate = adjoint(NL)*AC2*adjoint(NR)
        (U,S,V) = tsvd(intermediate,trunc=alg.trscheme,alg=TensorKit.SVD())

        (NL*U,V*NR)
    end

    pexp = MPSKit.PeriodicArray(collect(exps));

    #do the actual expansion
    for i in 1:length(state)
        al = MPSKit._permute_tail(TensorKit.catdomain(state.AL[i],pexp[i][1]))
        lz = TensorMap(zeros,_lastspace(pexp[i-1][1])',domain(al))
        state.AL[i] = MPSKit._permute_front(TensorKit.catcodomain(al,lz))

        ar = MPSKit._permute_front(TensorKit.catcodomain(MPSKit._permute_tail(state.AR[i+1]),pexp[i][2]))
        rz = TensorMap(zeros,codomain(ar),space(pexp[i+1][2],1))
        state.AR[i+1] = TensorKit.catdomain(ar,rz)

        l = TensorMap(zeros,codomain(state.CR[i]),space(pexp[i][2],1))
        state.CR[i] = TensorKit.catdomain(state.CR[i],l)
        r = TensorMap(zeros,_lastspace(pexp[i][1])',domain(state.CR[i]))
        state.CR[i] = TensorKit.catcodomain(state.CR[i],r)

        state.AC[i] = state.AL[i]*state.CR[i]
    end

    poison!(pars)

    return state,pars
end

function MPSKit.changebonds(state::InfiniteMPS,H::PeriodicMPO,alg,pars=params(state,H))
    (nmstate,pars) = changebonds(convert(MPSMultiline,state),H,alg,pars);
    return convert(InfiniteMPS,nmstate),pars
end

function MPSKit.changebonds(state::MPSMultiline, H,alg::OptimalExpand,pars=params(state,H))
    #=
        todo : merge this with the MPSCentergauged implementation
    =#
    #determine optimal expansion spaces around bond i
    exps = map(Iterators.product(1:size(state,1),1:size(state,2))) do (i,j)
        ACAR = _permute_front(state.AC[i-1,j])*_permute_tail(state.AR[i-1,j+1])
        AC2 = ac2_prime(ACAR,i-1,j,state,pars)

        #Calculate nullspaces for AL and AR
        NL = leftnull(state.AL[i,j])
        NR = rightnull(_permute_tail(state.AR[i,j+1]))

        #Use this nullspaces and SVD decomposition to determine the optimal expansion space
        intermediate = adjoint(NL)*AC2*adjoint(NR)
        (U,S,V) = tsvd(intermediate,trunc=alg.trscheme,alg=TensorKit.SVD())

        (NL*U,V*NR)
    end

    pexp = PeriodicArray(collect(exps));

    #do the actual expansion
    for (i,j) in Iterators.product(1:size(state,1),1:size(state,2))
        al = _permute_tail(TensorKit.catdomain(state.AL[i,j],pexp[i,j][1]))
        lz = TensorMap(zeros,_lastspace(pexp[i,j-1][1])',domain(al))

        state.AL[i,j] = _permute_front(TensorKit.catcodomain(al,lz))

        ar = _permute_front(TensorKit.catcodomain(_permute_tail(state.AR[i,j+1]),pexp[i,j][2]))
        rz = TensorMap(zeros,codomain(ar),space(pexp[i,j+1][2],1))
        state.AR[i,j+1] = TensorKit.catdomain(ar,rz)

        l = TensorMap(zeros,codomain(state.CR[i,j]),space(pexp[i,j][2],1))
        state.CR[i,j] = TensorKit.catdomain(state.CR[i,j],l)
        r = TensorMap(zeros,_lastspace(pexp[i,j][1])',domain(state.CR[i,j]))
        state.CR[i,j] = TensorKit.catcodomain(state.CR[i,j],r)

        state.AC[i,j] = state.AL[i,j]*state.CR[i,j]
    end

    poison!(pars)

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

        (nal,nc) = leftorth(TensorKit.catdomain(state.AC[i],ar_le),alg=QRpos())
        nar = _permute_front(TensorKit.catcodomain(_permute_tail(state.AR[i+1]),ar_re));

        state.AC[i] = (nal,nc)
        state.AC[i+1] = (nc,nar)
    end

    return state,pars
end
