function leading_boundary(ψ::MultilineMPS, operator, alg::IDMRG,
                          envs=environments(ψ, operator))
    log = IterLog("IDMRG")
    ϵ::Float64 = 2 * alg.tol
    local iter

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ, expectation_value(ψ, operator, envs))
        for outer iter in 1:(alg.maxiter)
            alg_eigsolve = updatetol(alg.alg_eigsolve, iter, ϵ)
            C_current = ψ.C[:, 0]

            # left to right sweep
            for col in 1:size(ψ, 2)
                Hac = AC_hamiltonian(col, ψ, operator, ψ, envs)
                _, ψ.AC[:, col] = fixedpoint(Hac, ψ.AC[:, col], :LM, alg_eigsolve)

                for row in 1:size(ψ, 1)
                    ac = ψ.AC[row, col]
                    (col == size(ψ, 2)) && (ac = copy(ac)) # needed in next sweep
                    ψ.AL[row, col], ψ.C[row, col] = leftorth!(ac)
                end

                transfer_leftenv!(envs, ψ, operator, ψ, col + 1)
            end

            # right to left sweep
            for col in size(ψ, 2):-1:1
                Hac = AC_hamiltonian(col, ψ, operator, ψ, envs)
                _, ψ.AC[:, col] = fixedpoint(Hac, ψ.AC[:, col], :LM, alg_eigsolve)

                for row in 1:size(ψ, 1)
                    ψ.C[row, col - 1], temp = rightorth!(_transpose_tail(ψ.AC[row, col]))
                    ψ.AR[row, col] = _transpose_front(temp)
                end

                transfer_rightenv!(envs, ψ, operator, ψ, col - 1)
            end

            normalize!(envs, ψ, operator, ψ)

            ϵ = norm(C_current - ψ.C[:, 0])

            if ϵ < alg.tol
                @infov 2 logfinish!(log, iter, ϵ, expectation_value(ψ, operator, envs))
                break
            end
            if iter == alg.maxiter
                @warnv 1 logcancel!(log, iter, ϵ, expectation_value(ψ, operator, envs))
            else
                @infov 3 logiter!(log, iter, ϵ, expectation_value(ψ, operator, envs))
            end
        end
    end

    alg_gauge = updatetol(alg.alg_gauge, iter, ϵ)
    ψ = MultilineMPS(map(x -> x, ψ.AR); alg_gauge.tol, alg_gauge.maxiter)

    recalculate!(envs, ψ, operator, ψ)
    return ψ, envs, ϵ
end

function leading_boundary(ψ::MultilineMPS, operator, alg::IDMRG2,
                          envs=environments(ψ, operator))
    size(ψ, 2) < 2 && throw(ArgumentError("unit cell should be >= 2"))
    ϵ::Float64 = 2 * alg.tol
    log = IterLog("IDMRG2")
    local iter

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ)
        for outer iter in 1:(alg.maxiter)
            alg_eigsolve = updatetol(alg.alg_eigsolve, iter, ϵ)
            C_current = ψ.C[:, 0]

            # sweep from left to right
            for site in 1:(size(ψ, 2) - 1)
                ac2 = AC2(ψ, site; kind=:ACAR)
                h = AC2_hamiltonian(site, ψ, operator, ψ, envs)
                _, ac2′ = fixedpoint(h, ac2, :LM, alg_eigsolve)

                for row in 1:size(ψ, 1)
                    al, c, ar = tsvd!(ac2′[row]; trunc=alg.trscheme, alg=alg.alg_svd)
                    normalize!(c)

                    ψ.AL[row + 1, site] = al
                    ψ.C[row + 1, site] = complex(c)
                    ψ.AR[row + 1, site + 1] = _transpose_front(ar)
                    ψ.AC[row + 1, site + 1] = _transpose_front(c * ar)
                end

                transfer_leftenv!(envs, ψ, operator, ψ, site + 1)
                transfer_rightenv!(envs, ψ, operator, ψ, site)
            end

            normalize!(envs, ψ, operator, ψ)

            # update the edge
            site = size(ψ, 2)
            ψ.AL[:, end] .= ψ.AC[:, end] ./ ψ.C[:, end]
            ψ.AC[:, 1] .= _mul_tail.(ψ.AL[:, 1], ψ.C[:, 1])
            ac2 = AC2(ψ, site; kind=:ALAC)
            h = AC2_hamiltonian(site, ψ, operator, ψ, envs)
            _, ac2′ = fixedpoint(h, ac2, :LM, alg_eigsolve)

            for row in 1:size(ψ, 1)
                al, c, ar = tsvd!(ac2′[row]; trunc=alg.trscheme, alg=alg.alg_svd)
                normalize!(c)

                ψ.AL[row + 1, site] = al
                ψ.C[row + 1, site] = complex(c)
                ψ.AR[row + 1, site + 1] = _transpose_front(ar)

                ψ.AC[row + 1, site] = _mul_tail(al, c)
                ψ.AC[row + 1, 1] = _transpose_front(c * ar)
                ψ.AL[row + 1, 1] = ψ.AC[row + 1, 1] / ψ.C[row + 1, 1]
            end

            # TODO: decide if we should compare at the half-sweep level?
            # C_current = ψ.C[:, site]

            transfer_leftenv!(envs, ψ, operator, ψ, 1)
            transfer_rightenv!(envs, ψ, operator, ψ, 0)

            # sweep from right to left
            for site in reverse(1:(size(ψ, 2) - 1))
                ac2 = AC2(ψ, site; kind=:ALAC)
                h = AC2_hamiltonian(site, ψ, operator, ψ, envs)
                _, ac2′ = fixedpoint(h, ac2, :LM, alg_eigsolve)

                for row in 1:size(ψ, 1)
                    al, c, ar = tsvd!(ac2′[row]; trunc=alg.trscheme, alg=alg.alg_svd)
                    normalize!(c)

                    ψ.AL[row + 1, site] = al
                    ψ.C[row + 1, site] = complex(c)
                    ψ.AR[row + 1, site + 1] = _transpose_front(ar)
                end

                transfer_leftenv!(envs, ψ, operator, ψ, site + 1)
                transfer_rightenv!(envs, ψ, operator, ψ, site)
            end

            normalize!(envs, ψ, operator, ψ)

            # update the edge
            ψ.AC[:, end] .= _mul_front.(ψ.C[:, end - 1], ψ.AR[:, end])
            ψ.AR[:, 1] .= _transpose_front.(ψ.C[:, end] .\ _transpose_tail.(ψ.AC[:, 1]))
            ac2 = AC2(ψ, 0; kind=:ACAR)
            h = AC2_hamiltonian(0, ψ, operator, ψ, envs)
            _, ac2′ = fixedpoint(h, ac2, :LM, alg_eigsolve)

            for row in 1:size(ψ, 1)
                al, c, ar = tsvd!(ac2′[row]; trunc=alg.trscheme, alg=alg.alg_svd)
                normalize!(c)

                ψ.AL[row + 1, end] = al
                ψ.C[row + 1, end] = complex(c)
                ψ.AR[row + 1, 1] = _transpose_front(ar)

                ψ.AR[row + 1, end] = _transpose_front(ψ.C[row + 1, end - 1] \
                                                      _transpose_tail(al * c))
                ψ.AC[row + 1, 1] = _transpose_front(c * ar)
            end

            transfer_leftenv!(envs, ψ, operator, ψ, 1)
            transfer_rightenv!(envs, ψ, operator, ψ, 0)

            # update error
            ϵ = sum(zip(C_current, ψ.C[:, 0])) do (c1, c2)
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

    alg_gauge = updatetol(alg.alg_gauge, iter, ϵ)
    ψ = MultilineMPS(map(identity, ψ.AR); alg_gauge.tol, alg_gauge.maxiter)

    recalculate!(envs, ψ, operator, ψ)
    return ψ, envs, ϵ
end
