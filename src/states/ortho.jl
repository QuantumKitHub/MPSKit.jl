# Algorithms
# ----------

const _GAUGE_ALG_EIGSOLVE = DynamicTol(Arnoldi(; krylovdim=30, eager=true),
                                       Defaults.tolgauge / 10, Inf, 1)

"""
    struct LeftCanonical <: Algorithm

Algorithm for bringing an `InfiniteMPS` into the left-canonical form.
"""
@kwdef struct LeftCanonical <: Algorithm
    tol::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
    verbosity::Int = VERBOSE_WARN

    alg_orth = QRpos()
    alg_eigsolve = _GAUGE_ALG_EIGSOLVE
    eig_miniter::Int = 10
end

"""
    struct RightCanonical <: Algorithm

Algorithm for bringing an `InfiniteMPS` into the right-canonical form.
"""
@kwdef struct RightCanonical <: Algorithm
    tol::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
    verbosity::Int = VERBOSE_WARN

    alg_orth = LQpos()
    alg_eigsolve = _GAUGE_ALG_EIGSOLVE
    eig_miniter::Int = 10
end

"""
    struct MixedCanonical <: Algorithm

"""
struct MixedCanonical <: Algorithm
    alg_leftcanonical::LeftCanonical
    alg_rightcanonical::RightCanonical
    order::Symbol
end

function MixedCanonical(; tol::Real=Defaults.tolgauge, maxiter::Int=Defaults.maxiter,
                        verbosity::Int=VERBOSE_WARN, alg_orth=QRpos(),
                        alg_eigsolve=_GAUGE_ALG_EIGSOLVE,
                        eig_miniter::Int=10, order::Symbol=:LR)
    if alg_orth isa QR || alg_orth isa QRpos
        alg_leftorth = alg_orth
        alg_rightorth = alg_orth'
    elseif alg_orth isa LQ || alg_orth isa LQpos
        alg_leftorth = alg_orth'
        alg_rightorth = alg_orth
    else
        throw(ArgumentError("Invalid orthogonalization algorithm: $(typeof(alg_orth))"))
    end

    left = LeftCanonical(; tol, maxiter, verbosity, alg_orth=alg_leftorth,
                         alg_eigsolve, eig_miniter)
    right = RightCanonical(; tol, maxiter=maxiter, verbosity, alg_orth=alg_rightorth,
                           alg_eigsolve, eig_miniter)

    return MixedCanonical(left, right, order)
end

# Interface
# ---------

@doc """
    gaugefix!(ψ::InfiniteMPS, A, C₀; kwargs...) -> ψ
    gaugefix!(ψ::InfiniteMPS, A, C₀, alg::Algorithm) -> ψ

Bring an `InfiniteMPS` into a uniform gauge, using the specified algorithm.
"""
gaugefix!

function gaugefix!(ψ::InfiniteMPS, A, C₀=ψ.CR[end]; order=:LR, kwargs...)
    alg = if order === :LR || order === :RL
        MixedCanonical(; order, kwargs...)
    elseif order === :L
        LeftCanonical(; kwargs...)
    elseif order === :R
        RightCanonical(; kwargs...)
    else
        throw(ArgumentError("Invalid order: $order"))
    end

    return gaugefix!(ψ, A, C₀, alg)
end

# expert mode: actual implementation
function gaugefix!(ψ::InfiniteMPS, A, C₀, alg::MixedCanonical)
    if alg.order === :LR
        gaugefix!(ψ, A, C₀, alg.alg_leftcanonical)
        gaugefix!(ψ, ψ.AL, ψ.CR[end], alg.alg_rightcanonical)
    elseif alg.order === :RL
        gaugefix!(ψ, A, C₀, alg.alg_rightcanonical)
        gaugefix!(ψ, ψ.AR, ψ.CR[end], alg.alg_leftcanonical)
    else
        throw(ArgumentError("Invalid order: $(alg.order)"))
    end
    return ψ
end
function gaugefix!(ψ::InfiniteMPS, A, C₀, alg::LeftCanonical)
    uniform_leftorth!((ψ.AL, ψ.CR), A, C₀, alg)
    return ψ
end
function gaugefix!(ψ::InfiniteMPS, A, C₀, alg::RightCanonical)
    uniform_rightorth!((ψ.AR, ψ.CR), A, C₀, alg)
    return ψ
end

@doc """
    regauge!(AC::GenericMPSTensor, CR::MPSBondTensor; alg=QRpos()) -> AL
    regauge!(CL::MPSBondTensor, AC::GenericMPSTensor; alg=LQpos()) -> AR

Bring updated `AC` and `C` tensors back into a consistent set of left or right canonical
tensors. This minimizes `∥AC - AL * CR∥` or `∥AC - CL * AR∥`. The optimal algorithm uses
`Polar()` decompositions, but `QR`-based algorithms are typically more performant. Note that
the first signature is slightly faster, as it avoids an intermediate transposition.
"""
regauge!

function regauge!(AC::GenericMPSTensor, CR::MPSBondTensor; alg=QRpos())
    Q_AC, _ = leftorth!(AC; alg)
    Q_C, _ = leftorth!(CR; alg)
    return mul!(AC, Q_AC, Q_C')
end
function regauge!(CL::MPSBondTensor, AC::GenericMPSTensor; alg=LQpos())
    AC_tail = _transpose_tail(AC)
    _, Q_AC = rightorth!(AC_tail; alg)
    _, Q_C = rightorth!(CL; alg)
    AR_tail = mul!(AC_tail, Q_C', Q_AC)
    return _transpose_front(AR_tail)
end

# Implementation
# --------------

function uniform_leftorth!((AL, CR), A, C₀, alg::LeftCanonical)
    CR[end] = normalize!(C₀)
    return LoggingExtras.withlevel(; alg.verbosity) do
        # initialize algorithm and temporary variables
        log = IterLog("LC")
        A_tail = _transpose_tail.(A) # pre-transpose A
        CA_tail = similar.(A_tail)  # pre-allocate workspace
        state = (; AL, CR, A, A_tail, CA_tail, iter=0, ϵ=Inf)
        it = IterativeSolver(alg, state)
        loginit!(log, it.ϵ)

        # iteratively solve
        for (AL, CR) in it
            iter, ϵ = it.iter, it.ϵ
            if ϵ < it.tol
                @infov 2 logfinish!(log, iter, ϵ)
                return AL, CR
            elseif iter > it.maxiter
                @warnv 2 logcancel!(log, iter, ϵ)
                return AL, CR
            end
            @infov 3 logiter!(log, iter, ϵ)
        end
    end
end

function Base.iterate(it::IterativeSolver{LeftCanonical}, state=it.state)
    C₀ = gauge_eigsolve_step!(it, state)
    C₁ = gauge_orth_step!(it, state)
    ϵ = oftype(state.ϵ, norm(C₀ - C₁))

    iter = state.iter + 1
    it.state = (; state.AL, state.CR, state.A, state.A_tail, state.CA_tail, iter, ϵ)

    return (it.state.AL, it.state.CR), it.state
end

function gauge_eigsolve_step!(it::IterativeSolver{LeftCanonical}, state)
    (; AL, CR, A, iter, ϵ) = state
    if iter ≥ it.eig_miniter
        alg_eigsolve = updatetol(it.alg_eigsolve, 1, ϵ^2)
        _, vec = fixedpoint(flip(TransferMatrix(A, AL)), CR[end], :LM, alg_eigsolve)
        _, CR[end] = leftorth!(vec; alg=it.alg_orth)
    end
    return CR[end]
end

function gauge_orth_step!(it::IterativeSolver{LeftCanonical}, state)
    (; AL, CR, A_tail, CA_tail) = state
    for i in 1:length(AL)
        mul!(CA_tail[i], CR[i - 1], A_tail[i])
        _repartition!(AL[i], CA_tail[i])
        AL[i], CR[i] = leftorth!(AL[i]; alg=it.alg_orth)
    end
    normalize!(CR[end])
    return CR[end]
end

function uniform_rightorth!((AR, CR), A, C₀, alg::RightCanonical)
    CR[end] = normalize!(C₀)
    return LoggingExtras.withlevel(; alg.verbosity) do
        # initialize algorithm and temporary variables
        log = IterLog("RC")
        AC_tail = _similar_tail.(A) # pre-allocate workspace
        state = (; AR, CR, A, AC_tail, iter=0, ϵ=Inf)
        it = IterativeSolver(alg, state)
        loginit!(log, it.ϵ)

        # iteratively solve
        for (AR, CR) in it
            iter, ϵ = it.iter, it.ϵ
            if ϵ < it.tol
                @infov 2 logfinish!(log, iter, ϵ)
                return AR, CR
            elseif iter > it.maxiter
                @warnv 2 logcancel!(log, iter, ϵ)
                return AR, CR
            end
            @infov 3 logiter!(log, iter, ϵ)
        end
    end
end

function Base.iterate(it::IterativeSolver{RightCanonical}, state=it.state)
    C₀ = gauge_eigsolve_step!(it, state)
    C₁ = gauge_orth_step!(it, state)
    ϵ = oftype(state.ϵ, norm(C₀ - C₁))

    iter = state.iter + 1
    it.state = (; state.AR, state.CR, state.A, state.AC_tail, iter, ϵ)

    return (it.state.AR, it.state.CR), it.state
end

function gauge_eigsolve_step!(it::IterativeSolver{RightCanonical}, state)
    (; AR, CR, A, iter, ϵ) = state
    if iter ≥ it.eig_miniter
        alg_eigsolve = updatetol(it.alg_eigsolve, 1, ϵ^2)
        _, vec = fixedpoint(TransferMatrix(A, AR), CR[end], :LM, alg_eigsolve)
        CR[end], _ = rightorth!(vec; alg=it.alg_orth)
    end
    return CR[end]
end

function gauge_orth_step!(it::IterativeSolver{RightCanonical}, state)
    (; A, AR, CR, AC_tail) = state
    for i in length(AR):-1:1
        AC = mul!(AR[i], A[i], CR[i])   # use AR as temporary storage for A * C
        tmp = _repartition!(AC_tail[i], AC)
        CR[i - 1], tmp = rightorth!(tmp; alg=it.alg_orth)
        _repartition!(AR[i], tmp)       # TODO: avoid doing this every iteration
    end
    normalize!(CR[end])
    return CR[end]
end
