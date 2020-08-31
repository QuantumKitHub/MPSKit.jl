"
    use an idmrg2 step to truncate/expand the bond dimension
"
@with_kw struct VumpsSvdCut <: Algorithm
    tol_gauge = Defaults.tolgauge
    tol_galerkin = Defaults.tol
    tol_eigenval = Defaults.tol
    trscheme = notrunc()
end

function changebonds_1(state::InfiniteMPS, H::Hamiltonian,alg::VumpsSvdCut,pars=params(state,H)) #would be more efficient if we also repeated pars
    #the unitcell==1 case is unique, because there you have a sef-consistency condition

    #expand the one site to two sites
    nstate = InfiniteMPS(repeat(state.AL,2))
    nH = repeat(H,2)

    nstate,npars = changebonds(nstate,nH,alg)

    A1 = nstate.AL[1]; A2 = nstate.AL[2]
    D1 = space(A1,1); D2 = space(A2,1)

    #collapse back to 1 site
    if dim(D2) != dim(D1)
        (nstate,npars) = changebonds(nstate,nH,SvdCut(trschemes = [truncdim(min(dim(D1),dim(D2)))]),npars)
    end

    collapsed = InfiniteMPS([nstate.AL[1]];tol=alg.tol_gauge)
    return collapsed, pars
end

function changebonds_n(state::InfiniteMPS, H::Hamiltonian,alg::VumpsSvdCut,pars=params(state,H))
    meps=0.0
    for loc in 1:length(state)
        @tensor AC2[-1 -2;-3 -4] := state.AC[loc][-1,-2,1]*state.AR[loc+1][1,-3,-4]

        (vals,vecs,_) = let state=state,pars=pars
            eigsolve(x->ac2_prime(x,loc,state,pars),AC2, 1, :SR, tol = alg.tol_eigenval; ishermitian=false )
        end
        nAC2=vecs[1]

        (vals,vecs,_)  = let state=state,pars=pars
            eigsolve(x->c_prime(x,loc+1,state,pars),state.CR[loc+1], 1, :SR, tol = alg.tol_eigenval; ishermitian=false )
        end
        nC2=vecs[1]

        #svd ac2, get new AL1 and S,V ---> AC
        (AL1,S,V,eps) = tsvd(nAC2, (1,2), (3,4), trunc=alg.trscheme)
        @tensor AC[-1,-2,-3]:=S[-1,1]*V[1,-2,-3]
        meps=max(eps,meps)

        #find AL2 from AC and C as in vumps paper
        QAC,temp = leftorth( AC,(1,2,),(3,), alg=TensorKit.QRpos())
        QC ,_ = leftorth(nC2 ,(1,),(2,) , alg=TensorKit.QRpos())
        dom_map = isometry(domain(QC),domain(QAC))

        @tensor AL2[-1,-2,-3] := QAC[-1,-2,1]*conj(dom_map[2,1])*conj(QC[-3,2])

        #make a new state using the updated A's
        allAls = copy(state.AL)
        allAls[loc]   = permute(AL1,(1,2),(3,))
        allAls[loc+1] = permute(AL2,(1,2),(3,))

        state = InfiniteMPS(allAls; tol = alg.tol_gauge)
    end

    return state, pars
end

changebonds(state::InfiniteMPS,H,alg::VumpsSvdCut,pars=params(state,H)) = (length(state) == 1) ? changebonds_1(state,H,alg,pars) : changebonds_n(state,H,alg,pars);
