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
    envs = idmrgenv(state,ham,leftenv(oenvs,1,st),rightenv(oenvs,length(st),st));
    delta::Float64 = 2*alg.tol_galerkin;

    for topit in 1:alg.maxiter
        delta = 0.0;

        for pos = 1:length(state)
            (eigvals,vecs) = eigsolve(state.AC[pos],1,:SR,Lanczos()) do x
                ac_prime(x,pos,state,envs)
            end
            state.AC[pos] = vecs[1]

        end

        curc = state.CR[end];

        newleft = transfer_left(leftenv(envs,length(state),state),ham,length(state),state.AL[end],state.AL[end]);
        growleft!(envs,newleft);

        for pos = length(state):-1:1

            (eigvals,vecs) = eigsolve(state.AC[pos],1,:SR,Lanczos()) do x
                ac_prime(x,pos,state,envs)
            end
            state.AC[pos] = vecs[1]
        end

        #update the environment:
        newright = transfer_right(rightenv(envs,1,state),ham,1,state.AR[1],state.AR[1]);
        growright!(envs,newright);

        delta = norm(curc-state.CR[0]);
        delta<alg.tol_galerkin && break;
        alg.verbose && @info "idmrg iter $(topit) err $(delta)"
    end

    st = InfiniteMPS(state.AL[1:end],state.CR[end],tol=alg.tol_gauge);
    oenvs = environments(st,ham,tol=oenvs.tol,maxiter=oenvs.maxiter)
    return st,oenvs,delta;
end

"""
twosite infinite dmrg
"""
@with_kw struct Idmrg2{} <: Algorithm
    tol_galerkin::Float64 = Defaults.tol
    tol_gauge::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
    verbose::Bool = Defaults.verbose
    trscheme = truncerr(1e-6);
end

function find_groundstate(st::InfiniteMPS, ham::Hamiltonian,alg::Idmrg2,oenvs=environments(st,ham))
    length(st) < 2 && throw(ArgumentError("unit cell should be >= 2"))

    state = FiniteMPS([st.AC[1];st.AR[2:end]]);
    envs = idmrgenv(state,ham,leftenv(oenvs,1,st),rightenv(oenvs,length(st),st));

    delta = 0.0;

    for topit in 1:alg.maxiter
        delta = 0.0;

        #sweep from left to right
        for pos = 1:length(state)-1
            ac2 = state.AC[pos]*_permute_tail(state.AR[pos+1]);
            (eigvals,vecs) = eigsolve(ac2,1,:SR,Lanczos()) do x
                ac2_prime(x,pos,state,envs)
            end

            (al,c,ar,ϵ) = tsvd(vecs[1],trunc=alg.trscheme,alg=TensorKit.SVD())
            normalize!(c);

            state.AC[pos] = (al,complex(c))
            state.AC[pos+1] = (complex(c),_permute_front(ar))
        end

        #update the edge
        @tensor ac2[-1 -2;-3 -4] := state.AL[end][-1,-2,1]*state.AL[1][1,-3,2]*state.CR[1][2,-4];
        (eigvals,vecs) = eigsolve(ac2,1,:SR,Lanczos()) do x
            ac2_prime(x,length(state),state,envs)
        end

        (al,c,ar,ϵ) = tsvd(vecs[1],trunc=alg.trscheme,alg=TensorKit.SVD())
        normalize!(c);

        #grow the environments
        newleft = transfer_left(leftenv(envs,length(state),state),ham,length(state),al,al);
        newright = transfer_right(rightenv(envs,1,state),ham,1,_permute_front(ar),_permute_front(ar));
        growleft!(envs,newleft);
        growright!(envs,newright);
        state.AC[end] = (al,complex(c));
        state.ALs[1] = _permute_front(c*ar)*inv(state.CR[1]);

        curc = complex(c);

        #sweep from right to left
        for pos = length(state)-1:-1:1

            ac2 = state.AC[pos]*_permute_tail(state.AR[pos+1]);
            (eigvals,vecs) = eigsolve(ac2,1,:SR,Lanczos()) do x
                ac2_prime(x,pos,state,envs)
            end

            (al,c,ar,ϵ) = tsvd(vecs[1],trunc=alg.trscheme,alg=TensorKit.SVD())
            normalize!(c);

            state.AC[pos] = (al,complex(c))
            state.AC[pos+1] = (complex(c),_permute_front(ar))
        end

        #update the edge
        @tensor ac2[-1 -2;-3 -4] := state.CR[end-1][-1,1]*state.AR[end][1,-2,2]*state.AR[1][2,-3,-4]
        (eigvals,vecs) = eigsolve(ac2,1,:SR,Lanczos()) do x
            ac2_prime(x,length(state),state,envs)
        end

        (al,c,ar,ϵ) = tsvd(vecs[1],trunc=alg.trscheme,alg=TensorKit.SVD())
        normalize!(c);

        #grow the environments
        newleft = transfer_left(leftenv(envs,length(state),state),ham,length(state),al,al);
        newright = transfer_right(rightenv(envs,1,state),ham,1,_permute_front(ar),_permute_front(ar));
        growleft!(envs,newleft);
        growright!(envs,newright);
        state.AC[1] = (complex(c),_permute_front(ar))
        state.ARs[end] = _permute_front(inv(state.CR[end-1])*_permute_tail(al*c))

        #update error
        d1 = Diagonal(convert(Array,curc));
        d2 = Diagonal(convert(Array,complex(c)));
        minl = min(length(d1),length(d2));
        delta = norm(d1[1:minl]-d2[1:minl])

        delta<alg.tol_galerkin && break;
        alg.verbose && @info "idmrg iter $(topit) err $(delta)"
    end

    st = InfiniteMPS(state.AL[1:end],state.CR[end],tol=alg.tol_gauge);
    oenvs = environments(st,ham,tol=oenvs.tol,maxiter=oenvs.maxiter)
    return st,oenvs,delta;
end
