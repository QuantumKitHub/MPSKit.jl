#=
Idmrg environments are only to be used internally.
They're environments of a finite state, but follow periodic boundary conditions
and we need to be able to update the edges
=#

struct IdmrgEnv{E<:FinEnv}
    finenv::E
end

function Base.getproperty(ca::IdmrgEnv,s::Symbol)
    if s == :opp
        return ca.finenv.opp;
    else
        return getfield(ca,s);
    end
end


#    envs = environments(state,ham,leftenv(oenvs,1,st),rightenv(oenvs,length(st),st));
function idmrgenv(state::FiniteMPS,ham,leftstart,rightstart)
    finenv = environments(state,ham,leftstart,rightstart)
    IdmrgEnv(finenv)
end

function growleft!(env::IdmrgEnv,newleft)
    env.finenv.leftenvs[1][:] = newleft
    env.finenv.ldependencies[:] = similar.(env.finenv.ldependencies)
end

function growright!(env::IdmrgEnv,newright)
    env.finenv.rightenvs[end][:] = newright
    env.finenv.rdependencies[:] = similar.(env.finenv.rdependencies)
end


leftenv(ca::IdmrgEnv,ind,state) = leftenv(ca.finenv,mod1(ind,length(state)),state)
rightenv(ca::IdmrgEnv,ind,state) = rightenv(ca.finenv,mod1(ind,length(state)),state)
