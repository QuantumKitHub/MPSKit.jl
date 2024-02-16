struct UniformOrthogonalization{A,B,C}
    tol::Float64
    maxiter::Int
    verbosity::Int
    leftorthalg::A
    rightorthalg::B
    eigalg::C
    eig_miniter::Int
end
function UniformOrthogonalization(; tol=Defaults.tolgauge, maxiter=Defaults.maxiter,
                                  verbosity=VERBOSE_WARN, leftorthalg=TensorKit.QRpos(),
                                  rightorthalg=TensorKit.LQpos(),
                                  eigalg=default_orth_eigalg(tol, maxiter), eig_miniter=10)
    return UniformOrthogonalization(tol, maxiter, verbosity, leftorthalg, rightorthalg,
                                    eigalg, eig_miniter)
end

function default_orth_eigalg(tol, maxiter)
    eigalg = Arnoldi(; krylovdim=30, maxiter)
    tol_min = tol / 10
    tol_max = Inf
    tol_factor = 1
    return ThrottledTol(eigalg, tol_min, tol_max, tol_factor)
end

Base.@deprecate(uniform_leftorth!(AL, CR, A; kwargs...),
                uniform_leftorth!(AL, CR, A, UniformOrthogonalization(; kwargs...)))

"""
    uniform_leftorth!(AL, CR, A, alg::UniformOrthogonalization)

Solves `AL[i] * CR[i] = CR[i] * A[i+1]`.
"""
function uniform_leftorth!(AL, CR, A, alg::UniformOrthogonalization)
    normalize!(CR[end])
    local δ
    for iter in 1:(alg.maxiter)
        if iter > alg.eig_miniter
            # attempt to somewhat replicate previous code: tol = max(δ², tol / 10)
            eigalg = updatetol(alg.eigalg, 1, δ^2)
            _, vecs = eigsolve(flip(TransferMatrix(A, AL)), CR[end], 1, :LM, eigalg)
            _, CR[end] = leftorth!(vecs[1]; alg=alg.leftorthalg)
        end

        C_old = CR[end]
        for loc in 1:length(AL)
            # TODO: there are definitely unnecessary allocations here,
            # and the transposition of A should be moved outside of the loop
            AL[loc] = _transpose_front(CR[mod1(loc - 1, end)] * _transpose_tail(A[loc]))
            AL[loc], CR[loc] = leftorth!(AL[loc]; alg=alg.leftorthalg)
            normalize!(CR[loc])
        end

        if space(C_old) == space(CR[end])
            δ = norm(C_old - CR[end])
            δ <= alg.tol && break
        end

        alg.verbosity >= VERBOSE_ITER && @info "LEFTORTH iteration $iter: δ = $δ"
    end
    alg.verbosity >= VERBOSE_WARN && δ > alg.tol &&
        @warn "LEFTORTH maximum iterations: δ = $δ"
    return AL, CR
end

Base.@deprecate(uniform_rightorth!(AR, CR, A; kwargs...),
                uniform_rightorth!(AR, CR, A, UniformOrthogonalization(; kwargs...)))

"""
    uniform_rightorth!(AR, CR, A, alg::UniformOrthogonalization)

Solves C[i-1] * AR[i] = A[i] * C[i].
"""
function uniform_rightorth!(AR, CR, A, alg::UniformOrthogonalization)
    normalize!(CR[end])
    local δ
    for iter in 1:(alg.maxiter)
        if iter > alg.eig_miniter
            eigalg = updatetol(alg.eigalg, 1, δ^2)
            _, vecs = eigsolve(TransferMatrix(A, AR), CR[end], 1, :LM, eigalg)
            CR[end], _ = rightorth!(vecs[1]; alg=alg.rightorthalg)
        end

        C_old = CR[end]

        for loc in length(AR):-1:1
            # TODO: there are definitely unnecessary allocations here
            AR[loc] = A[loc] * CR[loc]
            CR[mod1(loc - 1, end)], tmp = rightorth!(_transpose_tail(AR[loc]);
                                                         alg=alg.rightorthalg)
            AR[loc] = _transpose_front(tmp)
            normalize!(CR[mod1(loc - 1, end)])
        end

        if space(C_old) == space(CR[end])
            δ = norm(C_old - CR[end])
            δ <= alg.tol && break
        end

        alg.verbosity >= VERBOSE_ITER && @info "RIGHTORTH iteration $iter: δ = $δ"
    end
    alg.verbosity >= VERBOSE_WARN && δ > alg.tol &&
        @warn "RIGHTORTH maximum iterations: δ = $δ"
    return AR, CR
end
