"
    onesite dmrg
"
@with_kw struct Dmrg{F} <: Algorithm
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
    verbose::Bool = Defaults.verbose
    finalize::F = Defaults._finalize
end

find_groundstate(state,H,alg::Dmrg,envs...) = find_groundstate!(copy(state),H,alg,envs...)
function find_groundstate!(state::Union{FiniteMPS,MPSComoving}, H::Hamiltonian,alg::Dmrg,parameters = environments(state,H))
    tol=alg.tol;maxiter=alg.maxiter
    iter = 0; delta::Float64 = 2*tol

    while iter < maxiter && delta > tol
        delta=0.0

        for pos = [1:(length(state)-1);length(state):-1:2]
            (eigvals,vecs) =  let state = state,parameters = parameters
                eigsolve(state.AC[pos],1,:SR,Lanczos()) do x
                    ac_prime(x,pos,state,parameters)
                end
            end
            delta = max(delta,calc_galerkin(state,pos,parameters))

            state.AC[pos] = vecs[1]
        end

        alg.verbose && @info "Iteraton $(iter) error $(delta)"
        flush(stdout)

        iter += 1

        #finalize
        (state,parameters,sc) = alg.finalize(iter,state,H,parameters);
        delta = sc ? delta : 2*tol; # if finalize decides we shouldn't converge, then don't
    end

    return state,parameters,delta
end

"twosite dmrg"
@with_kw struct Dmrg2{F} <: Algorithm
    tol = Defaults.tol;
    maxiter = Defaults.maxiter;
    trscheme = truncerr(1e-6);
    verbose = Defaults.verbose
    finalize::F = Defaults._finalize
end

find_groundstate(state,H,alg::Dmrg2,envs...) = find_groundstate!(copy(state),H,alg,envs...)
function find_groundstate!(state::Union{FiniteMPS,MPSComoving}, H::Hamiltonian,alg::Dmrg2,parameters = environments(state,H))
    tol=alg.tol;maxiter=alg.maxiter
    iter = 0; delta::Float64 = 2*tol

    while iter < maxiter && delta > tol
        delta=0.0

        ealg = Lanczos()

        #left to right sweep
        for pos= 1:(length(state)-1)
            @tensor ac2[-1 -2; -3 -4]:=state.AC[pos][-1,-2,1]*state.AR[pos+1][1,-3,-4]
            (eigvals,vecs) =  let state=state,parameters=parameters
                eigsolve(x->ac2_prime(x,pos,state,parameters),ac2,1,:SR,ealg)
            end
            newA2center = vecs[1]

            (al,c,ar,ϵ) = tsvd(newA2center,trunc=alg.trscheme,alg=TensorKit.SVD())
            delta += ϵ;
            state.AC[pos] = (al,complex(c))
            state.AC[pos+1] = (complex(c),_permute_front(ar))
        end


        for pos = length(state)-2:-1:1
            @tensor ac2[-1 -2; -3 -4]:=state.AL[pos][-1,-2,1]*state.AC[pos+1][1,-3,-4]
            (eigvals,vecs) =  let state=state,parameters=parameters
                eigsolve(x->ac2_prime(x,pos,state,parameters),ac2,1,:SR,ealg)
            end
            newA2center = vecs[1]

            (al,c,ar,ϵ) = tsvd(newA2center,trunc=alg.trscheme,alg=TensorKit.SVD())
            delta += ϵ;
            state.AC[pos+1] = (complex(c),_permute_front(ar))
            state.AC[pos] = (al,complex(c))
        end

        alg.verbose && @info "Iteraton $(iter) truncation error $(delta)"
        flush(stdout)
        #finalize
        (state,parameters,sc) = alg.finalize(iter,state,H,parameters);
        delta = sc ? delta : 2*tol
        iter += 1
    end

    return state,parameters,delta
end
