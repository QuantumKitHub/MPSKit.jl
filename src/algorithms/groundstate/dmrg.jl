"
    onesite dmrg
"
@with_kw struct DMRG{F} <: Algorithm
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
    verbose::Bool = Defaults.verbose
    finalize::F = Defaults._finalize
end

find_groundstate(state, H, alg::DMRG, envs...) = find_groundstate!(copy(state), H, alg, envs...)
function find_groundstate!(state::AbstractFiniteMPS, H, alg::DMRG, envs=environments(state, H))
    tol = alg.tol
    maxiter = alg.maxiter
    iter = 0
    delta::Float64 = 2 * tol

    while iter < maxiter && delta > tol
        delta = 0.0

        for pos = [1:(length(state)-1); length(state):-1:2]
            h = ∂∂AC(pos, state, H, envs)
            (eigvals, vecs) = eigsolve(h, state.AC[pos], 1, :SR, Lanczos())
            delta = max(delta, calc_galerkin(state, pos, envs))

            state.AC[pos] = vecs[1]
        end

        alg.verbose && @info "Iteraton $(iter) error $(delta)"
        flush(stdout)

        iter += 1

        #finalize
        (state, envs) = alg.finalize(iter, state, H, envs)::Tuple{typeof(state),typeof(envs)}
    end

    return state, envs, delta
end

"twosite dmrg"
@with_kw struct DMRG2{F} <: Algorithm
    tol = Defaults.tol
    maxiter = Defaults.maxiter
    trscheme = truncerr(1e-6)
    verbose = Defaults.verbose
    finalize::F = Defaults._finalize
end

find_groundstate(state, H, alg::DMRG2, envs...) = find_groundstate!(copy(state), H, alg, envs...)
function find_groundstate!(state::AbstractFiniteMPS, H, alg::DMRG2, envs=environments(state, H))
    tol = alg.tol
    maxiter = alg.maxiter
    iter = 0
    delta::Float64 = 2 * tol

    while iter < maxiter && delta > tol
        delta = 0.0

        ealg = Lanczos()

        #left to right sweep
        @time begin
            for pos = 1:(length(state)-1)
                @plansor ac2[-1 -2; -3 -4] := state.AC[pos][-1 -2; 1] * state.AR[pos+1][1 -4; -3]

                h = ∂∂AC2(pos, state, H, envs)

                (eigvals, vecs) = eigsolve(h, ac2, 1, :SR, ealg)
                newA2center = first(vecs)

                (al, c, ar, ϵ) = tsvd(newA2center, trunc=alg.trscheme, alg=TensorKit.SVD())
                normalize!(c)
                v = @plansor ac2[1 2; 3 4] * conj(al[1 2; 5]) * conj(c[5; 6]) * conj(ar[6; 3 4])
                delta = max(delta, abs(1 - abs(v)))

                state.AC[pos] = (al, complex(c))
                state.AC[pos+1] = (complex(c), _transpose_front(ar))
            end


            for pos = length(state)-2:-1:1
                @plansor ac2[-1 -2; -3 -4] := state.AL[pos][-1 -2; 1] * state.AC[pos+1][1 -4; -3]

                h = ∂∂AC2(pos, state, H, envs)

                (eigvals, vecs) = eigsolve(h, ac2, 1, :SR, ealg)
                newA2center = first(vecs)

                (al, c, ar, ϵ) = tsvd(newA2center, trunc=alg.trscheme, alg=TensorKit.SVD())
                normalize!(c)
                v = @plansor ac2[1 2; 3 4] * conj(al[1 2; 5]) * conj(c[5; 6]) * conj(ar[6; 3 4])
                delta = max(delta, abs(1 - abs(v)))

                state.AC[pos+1] = (complex(c), _transpose_front(ar))
                state.AC[pos] = (al, complex(c))
            end
        end

        alg.verbose && @info "Iteraton $(iter) error $(delta)"
        flush(stdout)
        #finalize
        (state, envs) = alg.finalize(iter, state, H, envs)::Tuple{typeof(state),typeof(envs)}
        iter += 1
    end

    return state, envs, delta
end
