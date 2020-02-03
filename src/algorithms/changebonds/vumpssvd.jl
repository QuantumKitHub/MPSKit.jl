"
    use an idmrg2 step to truncate/expand the bond dimension
"
@with_kw struct VumpsSvdCut <: Algorithm
    tol_gauge = Defaults.tolgauge
    tol_galerkin = Defaults.tol
    tol_eigenval = Defaults.tol
    trscheme = notrunc()
end

function changebonds_1(state::MpsCenterGauged, H::Hamiltonian,alg::VumpsSvdCut,pars=nothing) #would be more efficient if we also repeated pars
    #the unitcell==1 case is unique, because there you have a sef-consistency condition

    #expand the one site to two sites
    (nstate,_)=MpsCenterGauged(repeat(state.AL,2))
    nH=repeat(H,2)

    nstate,npars=changebonds(nstate,nH,alg)

    A1 = nstate.AL[1]
    A2 = nstate.AL[2]
    D1 = dim(space(A1,1))
    p  = dim(space(A1,2))
    D2 = dim(space(A1,3))

    AL = A1
    #collapse back to 1 site
    if D2 > D1
        @tensor AL[-1 -2;-3] := leftorth(A1, (1, 2),(3,), truncdim(D1))[-1,-2,-3]
    elseif D1 > D2
        @tensor AL[-1 -2;-3] := leftorth(A2, (1, 2),(3,), truncdim(D2))[-1,-2,-3]
    end

    nstate=MpsCenterGauged([AL];tol=alg.tol_gauge)
    pars = params(H,nstate)
    return nstate, pars
end

function changebonds_n(state::MpsCenterGauged, H::Hamiltonian,alg::VumpsSvdCut,pars=params(state,H))
    meps=0.0
    for loc in 1:length(state)
        @tensor AC2[-1 -2 -3;-4] := state.AC[loc][-1,-2,1]*state.AR[loc+1][1,-3,-4]

        (vals,vecs,_) = eigsolve(x->ac2_prime(x,loc,state,pars),AC2, 1, :SR, tol = alg.tol_eigenval; ishermitian=true )
        nAC2=vecs[1]

        (vals,vecs,_)  = eigsolve(x->c_prime(x,loc+1,state,pars),state.CR[loc+1], 1, :SR, tol = alg.tol_eigenval; ishermitian=true )
        nC2=vecs[1]

        #find the updated two site AL
        QAC2,_ = TensorKit.leftorth(nAC2,(1,2,3,),(4,), alg=TensorKit.Polar())
        QC2 ,_ = TensorKit.leftorth(nC2 ,(1,),(2,)    , alg=TensorKit.Polar())

        #new AL2, reusing the memory alocated for nac2
        @tensor nAC2[-1,-2,-3,-4] = QAC2[-1,-2,-3,1]*conj(QC2[-4,1])

        (AL1,S,V,eps) = svd(nAC2, (1,2), (3,4), trunc=alg.trscheme)
        AL2=S*V
        meps=max(eps,meps)

        #make a new state using the updated A's
        allAls = copy(state.AL)
        allAls[loc]   = AL1
        allAls[loc+1] = AL2

        state,_ = MpsCenterGauged(allAls; tol = alg.tol_gauge)
        pars = params(state,H)
    end

    return state, pars
end

changebonds(state::MpsCenterGauged,H,alg::VumpsSvdCut,pars=params(state,H)) = (length(state) == 1) ? changebonds_1(state,H,alg,pars) : changebonds_n(state,H,alg,pars);
