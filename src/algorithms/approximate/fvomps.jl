# dispatch to in-place method
function approximate(ψ, toapprox, alg::Union{DMRG,DMRG2}, envs...)
    return approximate!(copy(ψ), toapprox, alg, envs...)
end

function approximate!(init::AbstractFiniteMPS, sq, alg, envs=environments(init, sq))
    tor = approximate!(init, [sq], alg, [envs])
    return (tor[1], tor[2][1], tor[3])
end

function approximate!(init::AbstractFiniteMPS, squash::Vector, alg::DMRG2,
                      envs=[environments(init, sq) for sq in squash])
    tol = alg.tol
    maxiter = alg.maxiter
    iter = 0
    delta = 2 * tol

    while iter < maxiter && delta > tol
        delta = 0.0

        init, envs = alg.finalize(iter, init, squash,
                                  envs)::Tuple{typeof(init),typeof(envs)}

        for pos in [1:(length(init) - 1); (length(init) - 2):-1:1]
            ac2 = init.AC[pos] * _transpose_tail(init.AR[pos + 1])

            nac2 = sum(map(zip(squash, envs)) do (sq, pr)
                           return ac2_proj(pos, init, pr)
                       end)

            (al, c, ar) = tsvd(nac2; trunc=alg.trscheme)

            delta = max(delta, norm(al * c * ar - ac2) / norm(ac2))

            init.AC[pos] = (al, complex(c))
            init.AC[pos + 1] = (complex(c), _transpose_front(ar))
        end

        alg.verbose && @info "2site dmrg iter $(iter) error $(delta)"

        #finalize
        iter += 1
    end

    delta > tol && @warn "2site dmrg failed to converge $(delta)>$(tol)"
    return init, envs, delta
end

function approximate!(init::AbstractFiniteMPS, squash::Vector, alg::DMRG,
                      envs=[environments(init, sq) for sq in squash])
    tol = alg.tol
    maxiter = alg.maxiter
    iter = 0
    delta = 2 * tol

    while iter < maxiter && delta > tol
        delta = 0.0

        #finalize
        (init, envs) = alg.finalize(iter, init, squash,
                                    envs)::Tuple{typeof(init),typeof(envs)}

        for pos in [1:(length(init) - 1); length(init):-1:2]
            newac = sum(map(zip(squash, envs)) do (sq, pr)
                            return ac_proj(pos, init, pr)
                        end)

            delta = max(delta, norm(newac - init.AC[pos]) / norm(newac))

            init.AC[pos] = newac
        end

        alg.verbose && @info "dmrg iter $(iter) error $(delta)"

        iter += 1
    end

    delta > tol && @warn "dmrg failed to converge $(delta)>$(tol)"
    return init, envs, delta
end
