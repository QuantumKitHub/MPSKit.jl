"
    SimpleEnv does nothing fancy to ensure the correctness of the environments it returns.
    Supports setleftenv! and setrightenv!
    Only used internally (in idmrg); no public constructor is provided
"
struct SimpleEnv{H<:Operator,V} <:Cache
    opp :: H
    lw :: PeriodicArray{V,2}
    rw :: PeriodicArray{V,2}
end

function SimpleEnv(state,pars::AbstractInfEnv)
    lw = similar(pars.lw)
    rw = similar(pars.rw)
    for i = 1:length(state)
        lw[i,:] = leftenv(pars,i,state)
        rw[i,:] = rightenv(pars,i,state)
    end

    return SimpleEnv(pars.opp,lw,rw)
end

leftenv(pars::SimpleEnv,pos::Int,state) = pars.lw[pos,:];
function setleftenv!(pars::SimpleEnv,pos,mps,lw)
    for i in 1:length(lw)
        pars.lw[pos,i] = lw[i]
    end
end

rightenv(pars::SimpleEnv,pos::Int,state) = pars.rw[pos,:];
function setrightenv!(pars::SimpleEnv,pos,mps,rw)
    for i in 1:length(rw)
        pars.rw[pos,i] = rw[i]
    end
end
