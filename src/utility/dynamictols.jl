module DynamicTols

import ..MPSKit: Algorithm
using Accessors
using DocStringExtensions
using MatrixAlgebraKit: DefaultAlgorithm
using KrylovKit: KrylovKit, Lanczos, Arnoldi

export adapt_solver, DynamicTol, AdaptiveKrylov
# deprecated, kept for backwards compatibility (see the bottom of this file)
export updatetol

# ============================================================
# Unified solver-adaptation interface
# ============================================================

@doc """
    adapt_solver(alg; iter, decay_rate, g_local, g_global, eps_trunc)

Resolve a (possibly adaptive) solver specification `alg` into a concrete solver, using
whichever adaptation signals are supplied as keyword arguments. Numeric signals default to
`0.0` (and `iter` to `1`) — i.e. "no contribution" — so each caller passes only what it has in
scope:

  - `iter`       — outer iteration count (drives the `1/√iter` damping)
  - `decay_rate` — measured per-matvec contraction factor of the previous solve
  - `g_local`    — local (per-bond) gradient / Galerkin norm
  - `g_global`   — global (sweep-wide) gradient / convergence-error scalar
  - `eps_trunc`  — local truncation error

The generic fallback returns `alg` unchanged (for plain `Lanczos`/`Arnoldi`/… solvers, which
are not adapted). `DynamicTol` and `AdaptiveKrylov` provide the adaptive implementations.
""" adapt_solver

adapt_solver(alg; kwargs...) = alg

# Wrapper for dynamic tolerance adjustment
# ----------------------------------------

"""
$(TYPEDEF)

Algorithm wrapper with dynamically adjusted tolerances. Only the wrapped solver's tolerance is
retuned; its Krylov budget (if any) is left fixed — this is the simpler counterpart to
[`AdaptiveKrylov`](@ref).

## Fields

$(TYPEDFIELDS)

See also [`adapt_solver`](@ref).
"""
struct DynamicTol{A} <: Algorithm
    "parent algorithm"
    alg::A

    "minimal value of the dynamic tolerance"
    tol_min::Float64

    "maximal value of the dynamic tolerance"
    tol_max::Float64

    "tolerance factor for updating relative to the current (global) gradient norm"
    tol_factor::Float64

    "factor on the local truncation error, sets an inner-solve tolerance floor (`0` ⇒ truncation-agnostic)"
    truncation_factor::Float64

    function DynamicTol(
            alg::A, tol_min::Real, tol_max::Real, tol_factor::Real, truncation_factor::Real = 0.0
        ) where {A}
        0 <= tol_min <= tol_max ||
            throw(ArgumentError("tol_min must be between 0 and tol_max"))
        truncation_factor >= 0 ||
            throw(ArgumentError("truncation_factor must be non-negative, got $truncation_factor"))
        return new{A}(alg, tol_min, tol_max, tol_factor, truncation_factor)
    end
end
function DynamicTol(alg; tol_min = 1.0e-6, tol_max = 1.0e-2, tol_factor = 0.1, truncation_factor = 0.0)
    return DynamicTol(alg, tol_min, tol_max, tol_factor, truncation_factor)
end

"""
    adapt_solver(alg::DynamicTol; iter, g_global, eps_trunc, ...)

Tighten only the wrapped solver's tolerance (its Krylov budget, if any, is left fixed). The
target combines the truncation-error floor with the global-gradient-driven convergence target,
optionally damped by the iteration count:

    tol = clamp(max(truncation_factor·eps_trunc, tol_factor·g_global) / √iter, tol_min, tol_max)

Per-bond callers (finite DMRG) supply `g_global`/`eps_trunc` and leave `iter = 1` (no damping);
per-sweep/global callers (VUMPS/iDMRG/…) supply the global error as `g_global` together with
`iter`, and `eps_trunc` defaults to `0`.
"""
function adapt_solver(
        alg::DynamicTol;
        iter::Integer = 1, g_global::Real = 0.0, eps_trunc::Real = 0.0, kwargs...
    )
    trunc_tol = alg.truncation_factor * eps_trunc
    conv_tol = alg.tol_factor * g_global
    tol = clamp(max(trunc_tol, conv_tol) / sqrt(max(iter, 1)), alg.tol_min, alg.tol_max)

    return _updatetol(alg.alg, tol)
end

# default implementation with Accessors.jl, but can be hooked into
function _updatetol(alg, tol::Real)
    return Accessors.@set alg.tol = tol
end
function _updatetol(alg::DefaultAlgorithm, tol::Real)
    kwargs = merge(alg.kwargs, (; tol = tol))
    return DefaultAlgorithm(; kwargs...)
end

# Set several solver parameters at once (tol, krylovdim, maxiter). Used by the adaptive
# controller, which retunes more than just the tolerance. `Lanczos`/`Arnoldi` expose these
# as plain fields, so `Accessors.setproperties` rebuilds the immutable struct in one shot.
function _set_eigsolve_params(alg::Union{Lanczos, Arnoldi}; kwargs...)
    return Accessors.setproperties(alg, NamedTuple(kwargs))
end

# ============================================================
# Adaptive Krylov controller (per-bond, stateful)
# ============================================================

"""
$(TYPEDEF)

Adaptive controller for local eigensolvers in DMRG-like algorithms.

This controller attempts to balance minimizing the number of local function applications with
minimizing the total number of iterations in order to obtain the globally fastest convergence.
This is driven by the local and global gradient norm, the truncation error and the measured
decay rate of previous iterations in an attempt to obtain fast convergence for gapped systems
while avoiding stagnation for gapless ones.

## Fields

$(TYPEDFIELDS)

See also [`adapt_solver`](@ref).
"""
struct AdaptiveKrylov{T, O <: KrylovKit.Orthogonalizer} <: Algorithm
    "orthogonalizer passed to the instantiated `Lanczos`/`Arnoldi`"
    orth::O

    "minimal Krylov subspace dimension (conservative/cold-start default)"
    krylovdim_min::Int
    "maximal Krylov subspace dimension"
    krylovdim_max::Int

    "minimal number of restart iterations"
    iter_min::Int
    "maximal number of restart iterations"
    iter_max::Int

    "lower bound on the dynamically chosen tolerance"
    tol_min::Float64
    "upper bound on the dynamically chosen tolerance"
    tol_max::Float64
    "factor on the local truncation error, sets the inner-solve tolerance floor"
    truncation_factor::Float64
    "factor on the global gradient norm, sets the convergence-driven tolerance"
    tol_factor::Float64

    "whether to use the eager (early-stopping) solver mode"
    eager::Bool
    "verbosity of the instantiated eigensolver"
    verbosity::Int

    "hermitian flag (`Val(true)` → `Lanczos`, `Val(false)` → `Arnoldi`)"
    ishermitian::Val{T}

    function AdaptiveKrylov{T, O}(
            orth::O, krylovdim_min, krylovdim_max, iter_min, iter_max,
            tol_min, tol_max, truncation_factor, tol_factor, eager, verbosity, ishermitian::Val{T}
        ) where {T, O <: KrylovKit.Orthogonalizer}
        0 < krylovdim_min <= krylovdim_max ||
            throw(ArgumentError("need 0 < krylovdim_min ≤ krylovdim_max, got ($krylovdim_min, $krylovdim_max)"))
        1 <= iter_min <= iter_max ||
            throw(ArgumentError("need 1 ≤ iter_min ≤ iter_max, got ($iter_min, $iter_max)"))
        0 <= tol_min <= tol_max ||
            throw(ArgumentError("need 0 ≤ tol_min ≤ tol_max, got ($tol_min, $tol_max)"))
        truncation_factor >= 0 ||
            throw(ArgumentError("truncation_factor must be non-negative, got $truncation_factor"))
        tol_factor >= 0 ||
            throw(ArgumentError("tol_factor must be non-negative, got $tol_factor"))
        return new{T, O}(
            orth, krylovdim_min, krylovdim_max, iter_min, iter_max,
            tol_min, tol_max, truncation_factor, tol_factor, eager, verbosity, ishermitian
        )
    end
end

function AdaptiveKrylov(;
        ishermitian::Bool = true, orth::KrylovKit.Orthogonalizer = KrylovKit.KrylovDefaults.orth,
        krylovdim_min::Int = 3, krylovdim_max::Int = 16,
        iter_min::Int = 1, iter_max::Int = 1,
        tol_min::Real = 1.0e-12, tol_max::Real = 1.0e-2,
        truncation_factor::Real = 1.0e-1, tol_factor::Real = 1.0e-1,
        eager::Bool = true, verbosity::Int = 0
    )
    return AdaptiveKrylov{ishermitian, typeof(orth)}(
        orth, krylovdim_min, krylovdim_max, iter_min, iter_max,
        tol_min, tol_max, truncation_factor, tol_factor, eager, verbosity, Val(ishermitian)
    )
end

"""
    adapt_solver(alg::AdaptiveKrylov; decay_rate, g_local, g_global, eps_trunc, ...)

Build a concrete `Lanczos`/`Arnoldi` for the current site. The target tolerance is the same as
[`DynamicTol`](@ref)'s per-bond tolerance, but the Krylov `krylovdim`/`maxiter` are additionally
predicted from the measured `decay_rate` and the local gradient norm `g_local`. A `decay_rate`
of `0` (the default) signals a cold start and falls back to the minimal budget.
"""
function adapt_solver(
        alg::AdaptiveKrylov{T}; decay_rate::Real = 0.0, g_local::Real = 0.0,
        g_global::Real = 0.0, eps_trunc::Real = 0.0, kwargs...
    ) where {T}
    # 1. target tolerance: inexact inner solve depending on outer convergence,
    #    never below the truncation error we already incur.
    tol = clamp(
        max(alg.truncation_factor * eps_trunc, alg.tol_factor * g_global), alg.tol_min, alg.tol_max
    )

    # the measured decay rate must be a genuine contraction in (0, 1); a non-positive or ≥ 1
    # rate signals an uninitialized or hard/stalling bond → fall back to the largest budget.
    ρ = clamp(decay_rate, 0.0, 1.0)
    if !(0 < ρ < 1)
        krylovdim = iszero(ρ) ? alg.krylovdim_min : alg.krylovdim_max
        maxiter = iszero(ρ) ? alg.iter_min : alg.iter_max
        return (T ? Lanczos : Arnoldi)(; alg.orth, krylovdim, maxiter, tol, alg.eager, alg.verbosity)
    end

    # 2. predicted matvecs from the *measured* decay and local convergence
    #    estimate through Lanczos bounds: going from g_local to tol in factors of ρ
    R = max(g_local, alg.tol_min) / tol
    nmatvecs = round(Int, log(R) / log(inv(ρ)))

    # 3. krylovdim and maxiter from matvec budget:
    #    prioritize krylovdim and compensate with maxiter
    #    thick restarts keep 3/5 of krylovdim, i.e. ~2/5 new vectors per cycle
    krylovdim = clamp(nmatvecs, alg.krylovdim_min, alg.krylovdim_max)
    maxiter = clamp(cld(5nmatvecs, 2krylovdim) - 3, alg.iter_min, alg.iter_max)

    return (T ? Lanczos : Arnoldi)(; alg.orth, krylovdim, maxiter, tol, alg.eager, alg.verbosity)
end

# ============================================================
# Deprecated entry points (subsumed by `adapt_solver`)
# ============================================================

Base.@deprecate updatetol(alg, iter::Integer, ϵ::Real) adapt_solver(alg; iter = iter, g_global = ϵ)

end
