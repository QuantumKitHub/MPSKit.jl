#=
Idmrg environments are only to be used internally.
They have to be updated manually, without any kind of checks
=#

struct IDMRGEnv{H,V}
    opp :: H
    lw :: PeriodicArray{V,2}
    rw :: PeriodicArray{V,2}
end

function IDMRGEnv(state::Union{MPSMultiline,InfiniteMPS},env)
    state === env.dependency || recalculate!(env,state);
    IDMRGEnv(env.opp,deepcopy(env.lw),deepcopy(env.rw));
end

leftenv(envs::IDMRGEnv,pos::Int) = envs.lw[:,pos];
leftenv(envs::IDMRGEnv,row::Int,col::Int) = envs.lw[row,col];
leftenv(envs::IDMRGEnv,pos::Int,state::InfiniteMPS) = envs.lw[:,pos];
function setleftenv!(envs::IDMRGEnv,pos,lw)
    envs.lw[:,pos] = lw[:]
end
function setleftenv!(envs::IDMRGEnv,row,col,val)
    envs.lw[row,col] = val;
end

rightenv(envs::IDMRGEnv,row::Int,col::Int) = envs.rw[row,col];
rightenv(envs::IDMRGEnv,pos::Int) = envs.rw[:,pos];
rightenv(envs::IDMRGEnv,pos::Int,state::InfiniteMPS) = envs.rw[:,pos];
function setrightenv!(envs::IDMRGEnv,pos,rw)
    envs.rw[:,pos] = rw[:]
end
function setrightenv!(envs::IDMRGEnv,row,col,val)
    envs.rw[row,col] = val;
end
