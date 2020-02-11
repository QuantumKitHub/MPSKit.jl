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
    nstate = MpsCenterGauged(repeat(state.AL,2))
    nH = repeat(H,2)

    nstate,npars = changebonds(nstate,nH,alg)

    A1 = nstate.AL[1]; A2 = nstate.AL[2]
    D1 = space(A1,1); D2 = space(A2,1)

    #collapse back to 1 site
    if dim(D2) != dim(D1)
        (nstate,nH) = changebonds(nstate,nH,SvdCut(trschemes = [truncdim(min(dim(D1),dim(D2)))]))
    end

    nstate = MpsCenterGauged([nstate.AL[1]];tol=alg.tol_gauge)
    pars = params(nstate,H)
    return nstate, pars
end

function changebonds_n(state::MpsCenterGauged, H::Hamiltonian,alg::VumpsSvdCut,pars=params(state,H))
    meps=0.0
    for loc in 1:length(state)
        @tensor AC2[-1 -2;-3 -4] := state.AC[loc][-1,-2,1]*state.AR[loc+1][1,-3,-4]

        (vals,vecs,_) = eigsolve(x->ac2_prime(x,loc,state,pars),AC2, 1, :SR, tol = alg.tol_eigenval; ishermitian=true )
        nAC2=vecs[1]

        (vals,vecs,_)  = eigsolve(x->c_prime(x,loc+1,state,pars),state.CR[loc+1], 1, :SR, tol = alg.tol_eigenval; ishermitian=true )
        nC2=vecs[1]

        #find the updated two site AL
        QAC2,_ = leftorth(nAC2,(1,2,3,),(4,), alg=TensorKit.Polar())
        QC2 ,_ = leftorth(nC2 ,(1,),(2,)    , alg=TensorKit.Polar())

        #new AL2, reusing the memory alocated for nac2
        @tensor nAC2[-1,-2,-3,-4] = QAC2[-1,-2,-3,1]*conj(QC2[-4,1])

        (AL1,S,V,eps) = tsvd(nAC2, (1,2), (3,4), trunc=alg.trscheme)
        AL2=S*V
        meps=max(eps,meps)

        #make a new state using the updated A's
        allAls = copy(state.AL)
        allAls[loc]   = permute(AL1,(1,2),(3,))
        allAls[loc+1] = permute(AL2,(1,2),(3,))

        state = MpsCenterGauged(allAls; tol = alg.tol_gauge)
        pars = params(state,H)
    end

    return state, pars
end

changebonds(state::MpsCenterGauged,H,alg::VumpsSvdCut,pars=params(state,H)) = (length(state) == 1) ? changebonds_1(state,H,alg,pars) : changebonds_n(state,H,alg,pars);
