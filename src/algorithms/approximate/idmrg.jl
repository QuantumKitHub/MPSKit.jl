function approximate(ost::MPSMultiline, toapprox::Tuple{<:MPOMultiline,<:MPSMultiline},
                     alg::IDMRG1, oenvs=environments(ost, toapprox))
    ψ = copy(ost)
    mpo, above = toapprox
    envs = IDMRGEnv(ost, oenvs)
    log = IterLog("IDMRG")
    ϵ::Float64 = 2 * alg.tol

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ)
        for iter in 1:(alg.maxiter)
            C_current = ψ.CR[:, 0]

            # left to right sweep
            for col in 1:size(ψ, 2), row in 1:size(ψ, 1)
                h = MPO_∂∂AC(mpo[row, col], leftenv(envs, row, col),
                             rightenv(envs, row, col))
                ψ.AC[row + 1, col] = h * above.AC[row, col]
                normalize!(ψ.AC[row + 1, col])

                ψ.AL[row + 1, col], ψ.CR[row + 1, col] = leftorth(ψ.AC[row + 1, col])

                tm = TransferMatrix(above.AL[row, col], mpo[row, col], ψ.AL[row + 1, col])
                setleftenv!(envs, row, col + 1, normalize(leftenv(envs, row, col) * tm))
            end

            # right to left sweep
            for col in size(ψ, 2):-1:1, row in 1:size(ψ, 1)
                h = MPO_∂∂AC(mpo[row, col], leftenv(envs, row, col),
                             rightenv(envs, row, col))
                ψ.AC[row + 1, col] = h * above.AC[row, col]
                normalize!(ψ.AC[row + 1, col])

                ψ.CR[row + 1, col - 1], temp = rightorth(_transpose_tail(ψ.AC[row + 1, col]))
                ψ.AR[row + 1, col] = _transpose_front(temp)

                tm = TransferMatrix(above.AR[row, col], mpo[row, col], ψ.AR[row + 1, col])
                setrightenv!(envs, row, col - 1, normalize(tm * rightenv(envs, row, col)))
            end

            ϵ = norm(C_current - ψ.CR[:, 0])

            if ϵ < alg.tol
                @infov 2 logfinish!(log, iter, ϵ)
                break
            end
            if iter == alg.maxiter
                @warnv 1 logcancel!(log, iter, ϵ)
            else
                @infov 3 logiter!(log, iter, ϵ)
            end
        end
    end

    nst = MPSMultiline(map(x -> x, ψ.AR); tol=alg.tol_gauge)
    nenvs = environments(nst, toapprox)
    return nst, nenvs, ϵ
end

function approximate(ost::MPSMultiline, toapprox::Tuple{<:MPOMultiline,<:MPSMultiline},
                     alg::IDMRG2, oenvs=environments(ost, toapprox))
    length(ost) < 2 && throw(ArgumentError("unit cell should be >= 2"))
    mpo, above = toapprox
    ψ = copy(ost)
    envs = IDMRGEnv(ost, oenvs)
    ϵ::Float64 = 2 * alg.tol
    log = IterLog("IDMRG2")

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ)
        for iter in 1:(alg.maxiter)
            C_current = ψ.CR[:, 0]

            # sweep from left to right
            for col in 1:size(ψ, 2), row in 1:size(ψ, 1)
                ac2 = above.AC[row, col] * _transpose_tail(above.AR[row, col + 1])
                h = MPO_∂∂AC2(mpo[row, col], mpo[row, col + 1], leftenv(envs, row, col),
                              rightenv(envs, row, col + 1))

                al, c, ar, = tsvd!(h * ac2; trunc=alg.trscheme, alg=TensorKit.SVD())
                normalize!(c)

                ψ.AL[row + 1, col] = al
                ψ.CR[row + 1, col] = complex(c)
                ψ.AR[row + 1, col + 1] = _transpose_front(ar)

                setleftenv!(envs, row, col + 1,
                            normalize(leftenv(envs, row, col) *
                                      TransferMatrix(above.AL[row, col], mpo[row, col],
                                                     ψ.AL[row + 1, col])))
                setrightenv!(envs, row, col,
                             normalize(TransferMatrix(above.AR[row, col + 1],
                                                      mpo[row, col + 1],
                                                      ψ.AR[row + 1, col + 1]) *
                                       rightenv(envs, row, col + 1)))
            end

            # sweep from right to left
            for col in (size(ψ, 2) - 1):-1:0, row in 1:size(ψ, 1)
                ac2 = above.AL[row, col] * _transpose_tail(above.AC[row, col + 1])
                h = MPO_∂∂AC2(mpo[row, col], mpo[row, col + 1], leftenv(envs, row, col),
                              rightenv(envs, row, col + 1))

                al, c, ar, = tsvd!(h * ac2; trunc=alg.trscheme, alg=TensorKit.SVD())
                normalize!(c)

                ψ.AL[row + 1, col] = al
                ψ.CR[row + 1, col] = complex(c)
                ψ.AR[row + 1, col + 1] = _transpose_front(ar)

                setleftenv!(envs, row, col + 1,
                            normalize(leftenv(envs, row, col) *
                                      TransferMatrix(above.AL[row, col], mpo[row, col],
                                                     ψ.AL[row + 1, col])))
                setrightenv!(envs, row, col,
                             normalize(TransferMatrix(above.AR[row, col + 1],
                                                      mpo[row, col + 1],
                                                      ψ.AR[row + 1, col + 1]) *
                                       rightenv(envs, row, col + 1)))
            end

            # update error
            ϵ = sum(zip(C_current, ψ.CR[:, 0])) do (c1, c2)
                smallest = infimum(_firstspace(c1), _firstspace(c2))
                e1 = isometry(_firstspace(c1), smallest)
                e2 = isometry(_firstspace(c2), smallest)
                return norm(e2' * c2 * e2 - e1' * c1 * e1)
            end

            if ϵ < alg.tol
                @infov 2 logfinish!(log, iter, ϵ)
                break
            end
            if iter == alg.maxiter
                @warnv 1 logcancel!(log, iter, ϵ)
            else
                @infov 3 logiter!(log, iter, ϵ)
            end
        end
    end

    nst = MPSMultiline(map(x -> x, ψ.AR); tol=alg.tol_gauge)
    nenvs = environments(nst, toapprox)
    return nst, nenvs, ϵ
end
