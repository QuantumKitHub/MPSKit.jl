"
expands the bond dimension by adding random unitary vectors
"
@with_kw struct RandExpand<:Algorithm
    trscheme::TruncationScheme = truncdim(1)
end


function changebonds(state::InfiniteMPS,alg::RandExpand)
    #determine optimal expansion spaces around bond i
    pexp = PeriodicArray(map(1:length(state)) do i
        ACAR = _transpose_front(state.AC[i])*_transpose_tail(state.AR[i+1])
        AC2 = TensorMap(rand,eltype(ACAR),codomain(ACAR),domain(ACAR));


        #Calculate nullspaces for AL and AR
        NL = leftnull(state.AL[i])
        NR = rightnull(_transpose_tail(state.AR[i+1]))

        #Use this nullspaces and SVD decomposition to determine the optimal expansion space
        intermediate = adjoint(NL)*AC2*adjoint(NR)
        (U,S,V) = tsvd(intermediate,trunc=alg.trscheme,alg=SVD())

        (NL*U,V*NR)
    end)

    newstate = copy(state);

    #do the actual expansion
    for i in 1:length(state)
        al = _transpose_tail(catdomain(newstate.AL[i],pexp[i][1]))
        lz = TensorMap(zeros,_lastspace(pexp[i-1][1])',domain(al))
        newstate.AL[i] = _transpose_front(catcodomain(al,lz))

        ar = _transpose_front(catcodomain(_transpose_tail(newstate.AR[i+1]),pexp[i][2]))
        rz = TensorMap(zeros,codomain(ar),space(pexp[i+1][2],1))
        newstate.AR[i+1] = catdomain(ar,rz)

        l = TensorMap(zeros,codomain(newstate.CR[i]),space(pexp[i][2],1))
        newstate.CR[i] = catdomain(newstate.CR[i],l)
        r = TensorMap(zeros,_lastspace(pexp[i][1])',domain(newstate.CR[i]))
        newstate.CR[i] = catcodomain(newstate.CR[i],r)

        newstate.AC[i] = newstate.AL[i]*newstate.CR[i]
    end

    return newstate
end

function changebonds(state::MPSMultiline,alg::RandExpand)
    #=
        todo : merge this with the MPSCentergauged implementation
    =#
    #determine optimal expansion spaces around bond i
    pexp = PeriodicArray(map(product(1:size(state,1),1:size(state,2))) do (i,j)

        ACAR = _transpose_front(state.AC[i-1,j])*_transpose_tail(state.AR[i-1,j+1])
        AC2 = TensorMap(rand,eltype(ACAR),codomain(ACAR),domain(ACAR));


        #Calculate nullspaces for AL and AR
        NL = leftnull(state.AL[i,j])
        NR = rightnull(_transpose_tail(state.AR[i,j+1]))

        #Use this nullspaces and SVD decomposition to determine the optimal expansion space
        intermediate = adjoint(NL)*AC2*adjoint(NR)
        (U,S,V) = tsvd(intermediate,trunc=alg.trscheme,alg=SVD())

        (NL*U,V*NR)
    end)

    newstate = copy(state);

    #do the actual expansion
    for i in 1:size(state,1),
        j in 1:size(state,2)
        al = _transpose_tail(catdomain(newstate.AL[i,j],pexp[i,j][1]))
        lz = TensorMap(zeros,_lastspace(pexp[i,j-1][1])',domain(al))

        newstate.AL[i,j] = _transpose_front(catcodomain(al,lz))

        ar = _transpose_front(catcodomain(_transpose_tail(newstate.AR[i,j+1]),pexp[i,j][2]))
        rz = TensorMap(zeros,codomain(ar),space(pexp[i,j+1][2],1))
        newstate.AR[i,j+1] = catdomain(ar,rz)

        l = TensorMap(zeros,codomain(newstate.CR[i,j]),space(pexp[i,j][2],1))
        newstate.CR[i,j] = catdomain(newstate.CR[i,j],l)
        r = TensorMap(zeros,_lastspace(pexp[i,j][1])',domain(newstate.CR[i,j]))
        newstate.CR[i,j] = catcodomain(newstate.CR[i,j],r)

        newstate.AC[i,j] = newstate.AL[i,j]*newstate.CR[i,j]
    end

    return newstate
end

changebonds(state::Union{FiniteMPS,MPSComoving}, alg::RandExpand) = changebonds!(copy(state),alg)
function changebonds!(state::Union{FiniteMPS,MPSComoving},alg::RandExpand)
    for i in 1:(length(state)-1)
        ACAR = _transpose_front(state.AC[i])*_transpose_tail(state.AR[i+1])
        AC2 = TensorMap(rand,eltype(ACAR),codomain(ACAR),domain(ACAR));

        #Calculate nullspaces for left and right
        NL = leftnull(state.AC[i])
        NR = rightnull(_transpose_tail(state.AR[i+1]))

        #Use this nullspaces and SVD decomposition to determine the optimal expansion space
        intermediate = adjoint(NL) * AC2 * adjoint(NR);
        (U,S,V) = tsvd(intermediate,trunc=alg.trscheme,alg=SVD())

        ar_re = V*NR;
        ar_le = TensorMap(zeros,codomain(state.AC[i]),space(V,1))

        (nal,nc) = leftorth(catdomain(state.AC[i],ar_le),alg=QRpos())
        nar = _transpose_front(catcodomain(_transpose_tail(state.AR[i+1]),ar_re));

        state.AC[i] = (nal,nc)
        state.AC[i+1] = (nc,nar)
    end

    state
end
