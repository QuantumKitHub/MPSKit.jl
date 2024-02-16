"""
    UniformGauging{A,B,C} <: Algorithm

Algorithm for bringing an infinite MPS into a uniform gauge.

# Fields
- `tol::Float64`: tolerance for convergence
- `maxiter::Int`: maximum number of iterations
- `verbosity::Int`: verbosity level
- `leftorthalg::A`: left-orthogonalization algorithm
- `rightorthalg::B`: right-orthogonalization algorithm
- `eigalg::C`: eigensolver algorithm
- `eig_miniter::Int`: minimum number of iterations before eigensolver is used
"""
struct UniformGauging{A,B,C} <: Algorithm
    tol::Float64
    maxiter::Int
    verbosity::Int
    leftorthalg::A
    rightorthalg::B
    order::Symbol
    eigalg::C
    eig_miniter::Int
end
function UniformGauging(; tol=Defaults.tolgauge, maxiter=Defaults.maxiter,
                        verbosity=VERBOSE_WARN, leftorthalg=TensorKit.QRpos(),
                        rightorthalg=TensorKit.LQpos(), order::Symbol=:LR,
                        eigalg=default_orth_eigalg(tol, maxiter), eig_miniter=10)
    return UniformGauging(tol, maxiter, verbosity, leftorthalg, rightorthalg, order,
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
                uniform_leftorth!(AL, CR, A, UniformGauging(; kwargs...)))

@doc """
    uniform_gauge(A, [C₀]; kwargs...) -> AL, AR, CR
    uniform_gauge!(AL, AR, CR, A, [C₀]; kwargs...) -> AL, AR, CR

Brings an infinite MPS, characterized by the tensors `A`, into the center gauge.
"""
uniform_gauge, uniform_gauge!

function uniform_gauge(A::AbstractVector{<:GenericMPSTensor}, C₀=gauge_init(A); kwargs...)
    AL = PeriodicArray(similar(A))
    AR = PeriodicArray(similar(A))
    CR = PeriodicArray(similar(A, typeof(C₀)))
    return uniform_gauge!(AL, AR, CR, A, C₀; kwargs...)
end

function uniform_gauge!(AL, AR, CR, A, C₀; kwargs...)
    CR[end] = C₀
    return uniform_gauge!(AL, AR, CR, A; kwargs...)
end
function uniform_gauge!(AL, AR, CR, A; kwargs...)
    alg = gauge_algselector(; kwargs...)
    if alg.order === :LR
        uniform_leftgauge!(AL, CR, A, alg)
        uniform_rightgauge!(AR, CR, AL, alg)
    elseif alg.order === :RL
        uniform_rightgauge!(AR, CR, A, alg)
        uniform_leftgauge!(AL, CR, AR, alg)
    elseif alg.order === :L
        AR .= A
        uniform_leftgauge!(AL, CR, AR, alg)
    elseif alg.order === :R
        AL .= A
        uniform_rightgauge!(AR, CR, AL, alg)
    else
        throw(ArgumentError("Invalid order: $(alg.order)"))
    end
    return AL, AR, CR
end

function gauge_algselector(; alg=nothing, kwargs...)
    return isnothing(alg) ? UniformGauging(; kwargs...) : alg
end

function gauge_init(A::AbstractVector{<:GenericMPSTensor})
    C = isomorphism(storagetype(A[1]), _firstspace(A[1]),
                       _firstspace(A[1]))
    return C
end

# Subroutines
# -----------

"""
    uniform_leftgauge!(AL, CR, A, alg::UniformGauging)

Solves `AL[i] * CR[i] = CR[i] * A[i+1]`.
"""
function uniform_leftgauge!(AL, CR, A, alg::UniformGauging)
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

Base.@deprecate(uniform_rightgauge!(AR, CR, A; kwargs...),
                uniform_rightgauge!(AR, CR, A, UniformGauging(; kwargs...)))

"""
    uniform_rightgauge!(AR, CR, A, alg::UniformGauging)

Solves C[i-1] * AR[i] = A[i] * C[i].
"""
function uniform_rightgauge!(AR, CR, A, alg::UniformGauging)
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