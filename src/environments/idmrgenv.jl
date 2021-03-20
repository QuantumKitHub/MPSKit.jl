#=
Idmrg environments are only to be used internally.
They have to be updated manually, without any kind of checks
=#

struct IDMRGEnv{H<:MPOHamiltonian,V<:MPSTensor}
    opp :: H
    lw :: PeriodicArray{V,2}
    rw :: PeriodicArray{V,2}
end

function IDMRGEnv(state::InfiniteMPS,env::MPOHamInfEnv)
    state === env.dependency || recalculate!(env,state);
    IDMRGEnv(env.opp,deepcopy(env.lw),deepcopy(env.rw));
end

leftenv(envs::IDMRGEnv,pos::Int,state=nothing) = envs.lw[pos,:];
function setleftenv!(envs::IDMRGEnv,pos,lw)
    envs.lw[pos,:] = lw[:]
end

rightenv(envs::IDMRGEnv,pos::Int,state=nothing) = envs.rw[pos,:];
function setrightenv!(envs::IDMRGEnv,pos,rw)
    envs.rw[pos,:] = rw[:]
end
