function approximate!(
        ψ::MultilineMPS, toapprox::Tuple{<:MultilineMPO, <:MultilineMPS}, alg::IDMRG,
        envs = environments(ψ, toapprox)
    )
    log = IterLog("IDMRG")
    ϵ::Float64 = 2 * alg.tol
    local iter

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ)
        for outer iter in 1:(alg.maxiter)
            C_current = ψ.C[:, 0]

            # left to right sweep
            for col in 1:size(ψ, 2)
                for row in 1:size(ψ, 1)
                    ψ.AC[row + 1, col] = AC_projection(
                        CartesianIndex(row, col), ψ, toapprox, envs
                    )
                    normalize!(ψ.AC[row + 1, col])
                    ψ.AL[row + 1, col], ψ.C[row + 1, col] = leftorth!(ψ.AC[row + 1, col])
                end
                transfer_leftenv!(envs, ψ, toapprox, col + 1)
            end

            # right to left sweep
            for col in reverse(1:size(ψ, 2))
                for row in 1:size(ψ, 1)
                    ψ.AC[row + 1, col] = AC_projection(
                        CartesianIndex(row, col), ψ, toapprox, envs
                    )
                    normalize!(ψ.AC[row + 1, col])
                    ψ.C[row + 1, col - 1], temp = rightorth!(_transpose_tail(ψ.AC[row + 1, col]))
                    ψ.AR[row + 1, col] = _transpose_front(temp)
                end
                transfer_rightenv!(envs, ψ, toapprox, col - 1)
            end
            normalize!(envs, ψ, toapprox)

            ϵ = norm(C_current - ψ.C[:, 0])

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

    # TODO: immediately compute in-place
    alg_gauge = updatetol(alg.alg_gauge, iter, ϵ)
    ψ′ = MultilineMPS(map(x -> x, ψ.AR); alg_gauge.tol, alg_gauge.maxiter)
    copy!(ψ, ψ′) # ensure output destination is unchanged

    recalculate!(envs, ψ, toapprox)
    return ψ, envs, ϵ
end

function approximate!(
        ψ::MultilineMPS, toapprox::Tuple{<:MultilineMPO, <:MultilineMPS},
        alg::IDMRG2, envs = environments(ψ, toapprox)
    )
    size(ψ, 2) < 2 && throw(ArgumentError("unit cell should be >= 2"))
    ϵ::Float64 = 2 * alg.tol
    log = IterLog("IDMRG2")
    O, ϕ = toapprox
    local iter

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ)
        for outer iter in 1:(alg.maxiter)
            C_current = ψ.C[:, 0]

            # sweep from left to right
            for site in 1:size(ψ, 2)
                for row in 1:size(ψ, 1)
                    AC2′ = AC2_projection(
                        CartesianIndex(row, site), ψ, toapprox, envs;
                        kind = :ACAR
                    )
                    al, c, ar = svd_trunc!(AC2′; trunc = alg.trscheme, alg = alg.alg_svd)
                    normalize!(c)

                    ψ.AL[row + 1, site] = al
                    ψ.C[row + 1, site] = complex(c)
                    ψ.AR[row + 1, site + 1] = _transpose_front(ar)
                    ψ.AC[row + 1, site + 1] = _transpose_front(c * ar)
                end

                transfer_leftenv!(envs, ψ, toapprox, site + 1)
                transfer_rightenv!(envs, ψ, toapprox, site)
            end

            normalize!(envs, ψ, toapprox)

            # sweep from right to left
            for site in reverse(0:(size(ψ, 2) - 1))
                for row in 1:size(ψ, 1)
                    AC2′ = AC2_projection(
                        CartesianIndex(row, site), ψ, toapprox, envs;
                        kind = :ALAC
                    )
                    al, c, ar = svd_trunc!(AC2′; trunc = alg.trscheme, alg = alg.alg_svd)
                    normalize!(c)

                    ψ.AL[row + 1, site] = al
                    ψ.C[row + 1, site] = complex(c)
                    ψ.AR[row + 1, site + 1] = _transpose_front(ar)
                end

                transfer_leftenv!(envs, ψ, toapprox, site + 1)
                transfer_rightenv!(envs, ψ, toapprox, site)
            end

            normalize!(envs, ψ, toapprox)

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

    # TODO: immediately compute in-place
    alg_gauge = updatetol(alg.alg_gauge, iter, ϵ)
    ψ′ = MultilineMPS(map(identity, ψ.AR); alg_gauge.tol, alg_gauge.maxiter)
    copy!(ψ, ψ′) # ensure output destination is unchanged

    recalculate!(envs, ψ, toapprox)
    return ψ, envs, ϵ
end
