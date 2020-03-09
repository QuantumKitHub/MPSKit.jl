function changebonds(state::T, H::ComAct,alg,pars=params(state,H)) where T <: FiniteMPO
    @info "$(typeof(alg)) not implemented for finite mpo; using slow fallback"

    mstate = mpo2mps(state);
    (a,_) = splitham(H.below);
    (_,b) = splitham(H.above);
    nH = a+b
    (nmstate,_) = changebonds(mstate,nH,alg);

    nstate = mps2mpo(nmstate,[space(s,2) for s in state.A])::T;
    return nstate,params(nstate,H)
end
