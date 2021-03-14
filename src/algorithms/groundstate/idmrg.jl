"""
onesite infinite dmrg
"""
@with_kw struct Idmrg1{} <: Algorithm
    tol_galerkin::Float64 = Defaults.tol
    tol_gauge::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
    verbose::Bool = Defaults.verbose
end


function find_groundstate(st::InfiniteMPS, ham::Hamiltonian,alg::Idmrg1,oenvs=environments(st,ham))
    state = FiniteMPS([st.AC[1];st.AR[2:end]]);
    envs = environments(state,ham,leftenv(oenvs,1,st),rightenv(oenvs,length(st),st));

    delta = 0.0;

    for topit in 1:alg.maxiter
        prevc = state.CR[0];

        for pos = 1:length(state)
            (eigvals,vecs) = eigsolve(state.AC[pos],1,:SR,Lanczos()) do x
                ac_prime(x,pos,state,envs)
            end
            state.AC[pos] = vecs[1]
        end

        envs.leftenvs[1][:] = leftenv(envs,length(state)+1,state);

        for pos = length(state):-1:1

            (eigvals,vecs) = eigsolve(state.AC[pos],1,:SR,Lanczos()) do x
                ac_prime(x,pos,state,envs)
            end
            state.AC[pos] = vecs[1]
        end

        #update the environment:
        envs.rightenvs[end][:] = rightenv(envs,0,state);

        delta = norm(prevc-state.CR[0]);
        delta<alg.tol_galerkin && break;
        alg.verbose && @info "idmrg iter $(topit) err $(delta)"
    end

    st = InfiniteMPS(state.AL[1:end],state.CR[end],tol=alg.tol_gauge);
    oenvs = environments(st,ham,tol=oenvs.tol,maxiter=oenvs.maxiter)
    return st,oenvs,delta;
end
