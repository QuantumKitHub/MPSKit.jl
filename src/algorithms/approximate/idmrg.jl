function approximate!(
        ψ::MultilineMPS, toapprox::Tuple{<:MultilineMPO, <:MultilineMPS}, alg::IDMRG,
        envs = environments(ψ, toapprox...)
    )
    allocator = default_allocator(ψ, SerialScheduler())
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
                        CartesianIndex(row, col), ψ, toapprox, envs;
                        alg.backend, allocator
                    )
                    normalize!(ψ.AC[row + 1, col])
                    ψ.AL[row + 1, col], ψ.C[row + 1, col] = left_orth!(ψ.AC[row + 1, col])
                end
                transfer_leftenv!(envs, ψ, toapprox, col + 1)
            end

            # right to left sweep
            for col in reverse(1:size(ψ, 2))
                for row in 1:size(ψ, 1)
                    ψ.AC[row + 1, col] = AC_projection(
                        CartesianIndex(row, col), ψ, toapprox, envs;
                        alg.backend, allocator
                    )
                    normalize!(ψ.AC[row + 1, col])
                    ψ.C[row + 1, col - 1], temp = right_orth!(_transpose_tail(ψ.AC[row + 1, col]))
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
    alg_gauge = adapt_solver(alg.alg_gauge; iter, g_global = ϵ)
    ψ′ = MultilineMPS(map(x -> x, ψ.AR); alg_gauge.tol, alg_gauge.maxiter)
    copy!(ψ, ψ′) # ensure output destination is unchanged

    recalculate!(envs, ψ, toapprox)
    return ψ, envs, AlgorithmInfo(; converged = ϵ <= alg.tol, normres = ϵ, numiter = iter)
end

function approximate!(
        ψ::MultilineMPS, toapprox::Tuple{<:MultilineMPO, <:MultilineMPS},
        alg::IDMRG2, envs = environments(ψ, toapprox...)
    )
    allocator = default_allocator(ψ, SerialScheduler())
    size(ψ, 2) < 2 && throw(ArgumentError("unit cell should be >= 2"))
    ϵ::Float64 = 2 * alg.tol
    log = IterLog("IDMRG2")
    O, ϕ = toapprox
    local iter, acc

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ)
        for outer iter in 1:(alg.maxiter)
            acc = TruncationAccumulator(ψ) # fresh each sweep, reported truncation is the final sweep's
            C_current = ψ.C[:, 0]

            # sweep from left to right
            for site in 1:(size(ψ, 2) - 1)
                for row in 1:size(ψ, 1)
                    AC2′ = AC2_projection(
                        CartesianIndex(row, site), ψ, toapprox, envs;
                        kind = :ACAR, alg.backend, allocator
                    )
                    al, c, ar, ϵ_trunc = svd_trunc!(AC2′; trunc = alg.trunc, alg = alg.alg_svd)
                    push_error!(acc, ϵ_trunc)
                    normalize!(c)

                    ψ.AL[row + 1, site] = al
                    ψ.C[row + 1, site] = complex(c)
                    ψ.AR[row + 1, site + 1] = _transpose_front(ar)
                    ψ.AC[row + 1, site + 1] = _transpose_front(c * ar)
                end

                transfer_leftenv!(envs, ψ, toapprox, site + 1)
                transfer_rightenv!(envs, ψ, toapprox, site)
            end

            # update the edge
            ψ.AL[1, end] = ψ.AC[1, end] / ψ.C[1, end]
            ψ.AC[1, 1] = _mul_tail(ψ.AL[1, 1], ψ.C[1, 1])
            for row in 1:size(ψ, 1)
                AC2′ = AC2_projection(
                    CartesianIndex(row, size(ψ, 2)), ψ, toapprox, envs;
                    kind = :ALAC, alg.backend, allocator
                )
                al, c, ar, ϵ_trunc = svd_trunc!(AC2′; trunc = alg.trunc, alg = alg.alg_svd)
                push_error!(acc, ϵ_trunc)
                normalize!(c)

                ψ.AL[row + 1, end] = al
                ψ.C[row + 1, end] = complex(c)
                ψ.AR[row + 1, 1] = _transpose_front(ar)

                ψ.AC[row + 1, end] = _mul_tail(al, c)
                ψ.AC[row + 1, 1] = _transpose_front(c * ar)
                ψ.AL[row + 1, 1] = ψ.AC[row + 1, 1] / ψ.C[row + 1, 1]

            end
            # update environments
            transfer_leftenv!(envs, ψ, toapprox, 1)
            transfer_rightenv!(envs, ψ, toapprox, 0)

            normalize!(envs, ψ, toapprox)

            # sweep from right to left
            for site in reverse(1:(size(ψ, 2) - 1))
                for row in 1:size(ψ, 1)
                    AC2′ = AC2_projection(
                        CartesianIndex(row, site), ψ, toapprox, envs;
                        kind = :ALAC, alg.backend, allocator
                    )
                    al, c, ar, ϵ_trunc = svd_trunc!(AC2′; trunc = alg.trunc, alg = alg.alg_svd)
                    push_error!(acc, ϵ_trunc)
                    normalize!(c)

                    ψ.AL[row + 1, site] = al
                    ψ.C[row + 1, site] = complex(c)
                    ψ.AR[row + 1, site + 1] = _transpose_front(ar)
                end

                transfer_leftenv!(envs, ψ, toapprox, site + 1)
                transfer_rightenv!(envs, ψ, toapprox, site)
            end

            # update the edge
            ψ.AC[1, end] = _mul_front(ψ.C[1, end - 1], ψ.AR[1, end])
            ψ.AR[1, 1] = _transpose_front(ψ.C[1, end] \ _transpose_tail(ψ.AC[1, 1]))
            for row in 1:size(ψ, 1)
                AC2′ = AC2_projection(
                    CartesianIndex(row, 0), ψ, toapprox, envs;
                    kind = :ACAR, alg.backend, allocator
                )
                al, c, ar, ϵ_trunc = svd_trunc!(AC2′; trunc = alg.trunc, alg = alg.alg_svd)
                push_error!(acc, ϵ_trunc)
                normalize!(c)

                ψ.AL[row, end] = al
                ψ.C[row, end] = complex(c)
                ψ.AR[row, 1] = _transpose_front(ar)

                ψ.AR[row, end] = _transpose_front(ψ.C[row, end - 1] \ _transpose_tail(al * c))
                ψ.AC[row, 1] = _transpose_front(c * ar)
            end
            transfer_leftenv!(envs, ψ, toapprox, 1)
            transfer_rightenv!(envs, ψ, toapprox, 0)

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
    alg_gauge = adapt_solver(alg.alg_gauge; iter, g_global = ϵ)
    ψ′ = MultilineMPS(map(identity, ψ.AR); alg_gauge.tol, alg_gauge.maxiter)
    copy!(ψ, ψ′) # ensure output destination is unchanged

    recalculate!(envs, ψ, toapprox)
    info = AlgorithmInfo(; converged = ϵ <= alg.tol, normres = ϵ, truncation = acc, numiter = iter)
    return ψ, envs, info
end
