"
    onesite dmrg
"
@with_kw struct Dmrg <: Algorithm
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
    verbose::Bool = Defaults.verbose
    manager::Algorithm = SimpleManager();
end

@bm function find_groundstate(state::Union{FiniteMPS,MPSComoving}, H::Hamiltonian,alg::Dmrg,parameters = params(state,H))
    tol=alg.tol;maxiter=alg.maxiter
    iter = 0; delta = 2*tol

    while iter < maxiter && delta > tol
        delta=0.0

        #dynamical bond expansion
        state, parameters = managebonds(state,H,alg.manager,parameters)

        for pos=1:(length(state)-1)
            (eigvals,vecs) =  let state = state,parameters = parameters
                eigsolve(state.AC[pos],1,:SR,Lanczos()) do x
                    ac_prime(x,pos,state,parameters)
                end
            end
            delta = max(delta,1-abs(dot(state.AC[pos],vecs[1])))

            state.AC[pos] = vecs[1]
        end

        for pos=length(state):-1:2
            (eigvals,vecs) =  let state = state,parameters = parameters
                eigsolve(state.AC[pos],1,:SR,Lanczos()) do x
                    ac_prime(x,pos,state,parameters)
                end
            end

            delta = max(delta,1-abs(dot(state.AC[pos],vecs[1])))

            state.AC[pos] = vecs[1]
        end

        alg.verbose && @show (iter,delta)
        flush(stdout)
        #finalize
        iter += 1
    end

    return state,parameters,delta
end

"twosite dmrg"
@with_kw struct Dmrg2 <: Algorithm
    tol = Defaults.tol;
    maxiter = Defaults.maxiter;
    trscheme = truncerr(1e-6);
    verbose = Defaults.verbose
end

@bm function find_groundstate(state::Union{FiniteMPS,MPSComoving}, H::Hamiltonian,alg::Dmrg2,parameters = params(state,H))

    tol=alg.tol;maxiter=alg.maxiter
    iter = 0; delta = 2*tol

    while iter < maxiter && delta > tol
        delta=0.0

        ealg = Lanczos()

        for pos=1:(length(state)-1)
            @tensor ac2[-1 -2; -3 -4]:=state[pos][-1,-2,1]*state[pos+1][1,-3,-4]
            (eigvals,vecs) =  let state=state,parameters=parameters
                eigsolve(x->ac2_prime(x,pos,state,parameters),ac2,1,:SR,ealg)
            end
            newA2center = vecs[1]

            (al,c,ar) = tsvd(newA2center,trunc=alg.trscheme)

            #yeah, we need a different convergence criterium
            @tensor ov[-1,-2,-3,-4]:=al[-1,-2,1]*c[1,2]*ar[2,-3,-4]-al[1,2,3]*c[3,4]*ar[4,5,6]*conj(state[pos][1,2,7])*conj(state[pos+1][7,5,6])*state[pos][-1,-2,9]*state[pos+1][9,-3,-4]

            newdelta = norm(ov)
            delta = max(delta,abs(newdelta))
            state[pos] = al
            @tensor state[pos+1][-1 -2 ; -3]:=c[-1,1]*ar[1,-2,-3]

        end

        for pos=length(state):-1:2
            @tensor ac2[-1 -2; -3 -4]:=state[pos-1][-1,-2,1]*state[pos][1,-3,-4]
            (eigvals,vecs) =  let state=state,parameters=parameters
                eigsolve(x->ac2_prime(x,pos-1,state,parameters),ac2,1,:SR,ealg)
            end
            newA2center = vecs[1]

            (al,c,ar) = tsvd(newA2center,trunc=alg.trscheme)

            @tensor ov[-1,-2,-3,-4]:=al[-1,-2,1]*c[1,2]*ar[2,-3,-4]-al[1,2,3]*c[3,4]*ar[4,5,6]*conj(state[pos-1][1,2,7])*conj(state[pos][7,5,6])*state[pos-1][-1,-2,9]*state[pos][9,-3,-4]
            newdelta = norm(ov)
            delta = max(delta,abs(newdelta))

            state[pos] = permute(ar,(1,2),(3,))
            @tensor state[pos-1][-1 -2 ; -3]:=al[-1,-2,1]*c[1,-3]
        end

        alg.verbose && @show (iter,delta)
        flush(stdout)
        #finalize
        iter += 1
    end

    return rightorth!(state),parameters,delta
end
