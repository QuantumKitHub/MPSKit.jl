"
    onesite dmrg
"
@with_kw struct Dmrg{F} <: Algorithm
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
    verbose::Bool = Defaults.verbose
    finalize::F = Defaults._finalize
end

function find_groundstate(state::Union{FiniteMPS,MPSComoving}, H::Hamiltonian,alg::Dmrg,parameters = params(state,H))
    tol=alg.tol;maxiter=alg.maxiter
    iter = 0; delta = 2*tol

    while iter < maxiter && delta > tol
        delta=0.0

        for pos = [1:(length(state)-1);length(state):-1:2]
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

        iter += 1

        #finalize
        (state,parameters,sc) = alg.finalize(iter,state,H,parameters);
        delta = sc ? delta : 2*tol; # if finalize decides we shouldn't converge, then don't
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

function find_groundstate(state::Union{FiniteMPS,MPSComoving}, H::Hamiltonian,alg::Dmrg2,parameters = params(state,H))
    tol=alg.tol;maxiter=alg.maxiter
    iter = 0; delta = 2*tol

    while iter < maxiter && delta > tol
        delta=0.0

        ealg = Lanczos()

        for pos=[1:(length(state)-1);length(state)-2:-1:1]
            @tensor ac2[-1 -2; -3 -4]:=state.AC[pos][-1,-2,1]*state.AR[pos+1][1,-3,-4]
            (eigvals,vecs) =  let state=state,parameters=parameters
                eigsolve(x->ac2_prime(x,pos,state,parameters),ac2,1,:SR,ealg)
            end
            newA2center = vecs[1]

            (al,c,ar) = tsvd(newA2center,trunc=alg.trscheme)

            #yeah, we need a different convergence criterium
            @tensor ov[-1,-2,-3,-4]:=al[-1,-2,1]*c[1,2]*ar[2,-3,-4]-al[1,2,3]*c[3,4]*ar[4,5,6]*conj(state.AC[pos][1,2,7])*conj(state.AR[pos+1][7,5,6])*state.AC[pos][-1,-2,9]*state.AR[pos+1][9,-3,-4]
            delta = max(delta,norm(ov))

            state.AC[pos] = (al,complex(c))
            state.AC[pos+1] = (complex(c),_permute_front(ar))
        end

        alg.verbose && @show (iter,delta)
        flush(stdout)
        #finalize
        iter += 1
    end

    return state,parameters,delta
end
