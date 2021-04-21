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
function find_groundstate!(state::Union{FiniteMPS,MPSComoving}, H::Hamiltonian,alg::Dmrg,envs = environments(state,H))
    tol=alg.tol;maxiter=alg.maxiter
    iter = 0; delta::Float64 = 2*tol

    while iter < maxiter && delta > tol
        delta=0.0

        for pos = [1:(length(state)-1);length(state):-1:2]
            (eigvals,vecs) = @closure eigsolve(state.AC[pos],1,:SR,Lanczos()) do x
                ac_prime(x,pos,state,envs)
            end
            delta = max(delta,calc_galerkin(state,pos,envs))

            state.AC[pos] = vecs[1]
        end

        alg.verbose && @info "Iteraton $(iter) error $(delta)"
        flush(stdout)

        iter += 1

        #finalize
        (state,envs) = alg.finalize(iter,state,H,envs)::Tuple{typeof(state),typeof(envs)};
    end

    return state,envs,delta
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
function find_groundstate!(state::Union{FiniteMPS,MPSComoving}, H::Hamiltonian,alg::Dmrg2,envs = environments(state,H))
    tol=alg.tol;maxiter=alg.maxiter
    iter = 0; delta::Float64 = 2*tol

    while iter < maxiter && delta > tol
        delta=0.0

        ealg = Lanczos()

        #left to right sweep
        for pos= 1:(length(state)-1)
            @tensor ac2[-1 -2; -3 -4]:=state.AC[pos][-1,-2,1]*state.AR[pos+1][1,-3,-4]
            (eigvals,vecs) = @closure eigsolve(ac2,1,:SR,ealg) do x
                ac2_prime(x,pos,state,envs)
            end
            newA2center = vecs[1]

            (al,c,ar,ϵ) = tsvd(newA2center,trunc=alg.trscheme,alg=TensorKit.SVD())
            normalize!(c);
            v = @tensor ac2[1,2,3,4]*conj(al[1,2,5])*conj(c[5,6])*conj(ar[6,3,4])
            delta = max(delta,abs(1-abs(v)));

            state.AC[pos] = (al,complex(c))
            state.AC[pos+1] = (complex(c),_permute_front(ar))
        end


        for pos = length(state)-2:-1:1
            @tensor ac2[-1 -2; -3 -4]:=state.AL[pos][-1,-2,1]*state.AC[pos+1][1,-3,-4]
            (eigvals,vecs) = @closure eigsolve(ac2,1,:SR,ealg) do x
                ac2_prime(x,pos,state,envs)
            end
            newA2center = vecs[1]

            (al,c,ar,ϵ) = tsvd(newA2center,trunc=alg.trscheme,alg=TensorKit.SVD())
            normalize!(c);
            v = @tensor ac2[1,2,3,4]*conj(al[1,2,5])*conj(c[5,6])*conj(ar[6,3,4])
            delta = max(delta,abs(1-abs(v)));

            state.AC[pos+1] = (complex(c),_permute_front(ar))
            state.AC[pos] = (al,complex(c))
        end

        alg.verbose && @info "Iteraton $(iter) error $(delta)"
        flush(stdout)
        #finalize
        (state,envs) = alg.finalize(iter,state,H,envs)::Tuple{typeof(state),typeof(envs)};
        iter += 1
    end

    return state,envs,delta
end
