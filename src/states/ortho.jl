# Algorithms
# ----------

const _GAUGE_ALG_EIGSOLVE = Defaults.alg_eigsolve(; ishermitian = false, tol_factor = 1)

"""
$(TYPEDEF)

Algorithm for bringing an `InfiniteMPS` into the left-canonical form.

## Fields

$(TYPEDFIELDS)
"""
@kwdef struct LeftCanonical <: Algorithm
    "tolerance for convergence criterium"
    tol::Float64 = Defaults.tolgauge
    "maximal amount of iterations"
    maxiter::Int = Defaults.maxiter
    "setting for how much information is displayed"
    verbosity::Int = VERBOSE_WARN

    "algorithm used for orthogonalization of the tensors"
    alg_orth = QRpos()
    "algorithm used for the eigensolver"
    alg_eigsolve = _GAUGE_ALG_EIGSOLVE
    "minimal amount of iterations before using the eigensolver steps"
    eig_miniter::Int = 10
end

"""
$(TYPEDEF)

Algorithm for bringing an `InfiniteMPS` into the right-canonical form.

## Fields

$(TYPEDFIELDS)
"""
@kwdef struct RightCanonical <: Algorithm
    "tolerance for convergence criterium"
    tol::Float64 = Defaults.tolgauge
    "maximal amount of iterations"
    maxiter::Int = Defaults.maxiter
    "setting for how much information is displayed"
    verbosity::Int = VERBOSE_WARN

    "algorithm used for orthogonalization of the tensors"
    alg_orth = LQpos()
    "algorithm used for the eigensolver"
    alg_eigsolve = _GAUGE_ALG_EIGSOLVE
    "minimal amount of iterations before using the eigensolver steps"
    eig_miniter::Int = 10
end

"""
$(TYPEDEF)

Algorithm for bringing an `InfiniteMPS` into the mixed-canonical form.

## Fields

$(TYPEDFIELDS)
"""
struct MixedCanonical <: Algorithm
    "algorithm for bringing an `InfiniteMPS` into left-canonical form."
    alg_leftcanonical::LeftCanonical
    "algorithm for bringing an `InfiniteMPS` into right-canonical form."
    alg_rightcanonical::RightCanonical
    "order in which to apply the canonicalizations, should be `:L`, `:R`, `:LR` or `:RL`"
    order::Symbol
end

function MixedCanonical(;
        tol::Real = Defaults.tolgauge, maxiter::Int = Defaults.maxiter,
        verbosity::Int = VERBOSE_WARN, alg_orth = QRpos(),
        alg_eigsolve = _GAUGE_ALG_EIGSOLVE,
        eig_miniter::Int = 10, order::Symbol = :LR
    )
    if alg_orth isa QR || alg_orth isa QRpos
        alg_leftorth = alg_orth
        alg_rightorth = alg_orth'
    elseif alg_orth isa LQ || alg_orth isa LQpos
        alg_leftorth = alg_orth'
        alg_rightorth = alg_orth
    else
        throw(ArgumentError("Invalid orthogonalization algorithm: $(typeof(alg_orth))"))
    end

    left = LeftCanonical(;
        tol, maxiter, verbosity, alg_orth = alg_leftorth, alg_eigsolve, eig_miniter
    )
    right = RightCanonical(;
        tol, maxiter = maxiter, verbosity, alg_orth = alg_rightorth, alg_eigsolve, eig_miniter
    )

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

function gaugefix!(ψ::InfiniteMPS, A, C₀ = ψ.C[end]; order = :LR, kwargs...)
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
        gaugefix!(ψ, ψ.AL, ψ.C[end], alg.alg_rightcanonical)
    elseif alg.order === :RL
        gaugefix!(ψ, A, C₀, alg.alg_rightcanonical)
        gaugefix!(ψ, ψ.AR, ψ.C[end], alg.alg_leftcanonical)
    else
        throw(ArgumentError("Invalid order: $(alg.order)"))
    end
    return ψ
end
function gaugefix!(ψ::InfiniteMPS, A, C₀, alg::LeftCanonical)
    uniform_leftorth!((ψ.AL, ψ.C), A, C₀, alg)
    return ψ
end
function gaugefix!(ψ::InfiniteMPS, A, C₀, alg::RightCanonical)
    uniform_rightorth!((ψ.AR, ψ.C), A, C₀, alg)
    return ψ
end

@doc """
    regauge!(AC::GenericMPSTensor, C::MPSBondTensor; alg=QRpos()) -> AL
    regauge!(CL::MPSBondTensor, AC::GenericMPSTensor; alg=LQpos()) -> AR

Bring updated `AC` and `C` tensors back into a consistent set of left or right canonical
tensors. This minimizes `∥AC_i - AL_i * C_i∥` or `∥AC_i - C_{i-1} * AR_i∥`. The optimal algorithm uses
`Polar()` decompositions, but `QR`-based algorithms are typically more performant.

!!! note
    Computing `AL` is slightly faster than `AR`, as it avoids an intermediate transposition.
"""
regauge!

function regauge!(AC::GenericMPSTensor, C::MPSBondTensor; alg = QRpos())
    Q_AC, _ = leftorth!(AC; alg)
    Q_C, _ = leftorth!(C; alg)
    return mul!(AC, Q_AC, Q_C')
end
function regauge!(AC::Vector{<:GenericMPSTensor}, C::Vector{<:MPSBondTensor}; alg = QRpos())
    for i in 1:length(AC)
        regauge!(AC[i], C[i]; alg)
    end
    return AC
end
function regauge!(CL::MPSBondTensor, AC::GenericMPSTensor; alg = LQpos())
    AC_tail = _transpose_tail(AC)
    _, Q_AC = rightorth!(AC_tail; alg)
    _, Q_C = rightorth!(CL; alg)
    AR_tail = mul!(AC_tail, Q_C', Q_AC)
    return repartition!(AC, AR_tail)
end
function regauge!(CL::Vector{<:MPSBondTensor}, AC::Vector{<:GenericMPSTensor}; alg = LQpos())
    for i in length(CL):-1:1
        regauge!(CL[i], AC[i]; alg)
    end
    return CL
end
# fix ambiguity + error
regauge!(::MPSBondTensor, ::MPSBondTensor; alg = QRpos()) = error("method ambiguity")
function regauge!(::Vector{<:MPSBondTensor}, ::Vector{<:MPSBondTensor}; alg = QRpos())
    return error("method ambiguity")
end

# Implementation
# --------------

function uniform_leftorth!((AL, C), A, C₀, alg::LeftCanonical)
    C[end] = normalize!(C₀)
    return LoggingExtras.withlevel(; alg.verbosity) do
        # initialize algorithm and temporary variables
        log = IterLog("LC")
        A_tail = _transpose_tail.(A) # pre-transpose A
        CA_tail = similar.(A_tail)  # pre-allocate workspace
        state = (; AL, C, A, A_tail, CA_tail, iter = 0, ϵ = Inf)
        it = IterativeSolver(alg, state)
        loginit!(log, it.ϵ)

        # iteratively solve
        for (AL, C) in it
            iter, ϵ = it.iter, it.ϵ
            if ϵ < it.tol
                @infov 2 logfinish!(log, iter, ϵ)
                return AL, C
            elseif iter > it.maxiter
                @warnv 2 logcancel!(log, iter, ϵ)
                return AL, C
            end
            @infov 3 logiter!(log, iter, ϵ)
        end
    end
end

function Base.iterate(it::IterativeSolver{LeftCanonical}, state = it.state)
    C₀ = gauge_eigsolve_step!(it, state)
    C₁ = gauge_orth_step!(it, state)
    ϵ = oftype(state.ϵ, norm(C₀ - C₁))

    iter = state.iter + 1
    it.state = (; state.AL, state.C, state.A, state.A_tail, state.CA_tail, iter, ϵ)

    return (it.state.AL, it.state.C), it.state
end

function gauge_eigsolve_step!(it::IterativeSolver{LeftCanonical}, state)
    (; AL, C, A, iter, ϵ) = state
    if iter ≥ it.eig_miniter
        alg_eigsolve = updatetol(it.alg_eigsolve, 1, ϵ^2)
        _, vec = fixedpoint(flip(TransferMatrix(A, AL)), C[end], :LM, alg_eigsolve)
        _, C[end] = leftorth!(vec; alg = it.alg_orth)
    end
    return C[end]
end

function gauge_orth_step!(it::IterativeSolver{LeftCanonical}, state)
    (; AL, C, A_tail, CA_tail) = state
    for i in 1:length(AL)
        mul!(CA_tail[i], C[i - 1], A_tail[i])
        repartition!(AL[i], CA_tail[i])
        AL[i], C[i] = leftorth!(AL[i]; alg = it.alg_orth)
    end
    normalize!(C[end])
    return C[end]
end

function uniform_rightorth!((AR, C), A, C₀, alg::RightCanonical)
    C[end] = normalize!(C₀)
    return LoggingExtras.withlevel(; alg.verbosity) do
        # initialize algorithm and temporary variables
        log = IterLog("RC")
        AC_tail = _similar_tail.(A) # pre-allocate workspace
        state = (; AR, C, A, AC_tail, iter = 0, ϵ = Inf)
        it = IterativeSolver(alg, state)
        loginit!(log, it.ϵ)

        # iteratively solve
        for (AR, C) in it
            iter, ϵ = it.iter, it.ϵ
            if ϵ < it.tol
                @infov 2 logfinish!(log, iter, ϵ)
                return AR, C
            elseif iter > it.maxiter
                @warnv 2 logcancel!(log, iter, ϵ)
                return AR, C
            end
            @infov 3 logiter!(log, iter, ϵ)
        end
    end
end

function Base.iterate(it::IterativeSolver{RightCanonical}, state = it.state)
    C₀ = gauge_eigsolve_step!(it, state)
    C₁ = gauge_orth_step!(it, state)
    ϵ = oftype(state.ϵ, norm(C₀ - C₁))

    iter = state.iter + 1
    it.state = (; state.AR, state.C, state.A, state.AC_tail, iter, ϵ)

    return (it.state.AR, it.state.C), it.state
end

function gauge_eigsolve_step!(it::IterativeSolver{RightCanonical}, state)
    (; AR, C, A, iter, ϵ) = state
    if iter ≥ it.eig_miniter
        alg_eigsolve = updatetol(it.alg_eigsolve, 1, ϵ^2)
        _, vec = fixedpoint(TransferMatrix(A, AR), C[end], :LM, alg_eigsolve)
        C[end], _ = rightorth!(vec; alg = it.alg_orth)
    end
    return C[end]
end

function gauge_orth_step!(it::IterativeSolver{RightCanonical}, state)
    (; A, AR, C, AC_tail) = state
    for i in length(AR):-1:1
        AC = mul!(AR[i], A[i], C[i])   # use AR as temporary storage for A * C
        tmp = repartition!(AC_tail[i], AC)
        C[i - 1], tmp = rightorth!(tmp; alg = it.alg_orth)
        repartition!(AR[i], tmp)       # TODO: avoid doing this every iteration
    end
    normalize!(C[end])
    return C[end]
end
