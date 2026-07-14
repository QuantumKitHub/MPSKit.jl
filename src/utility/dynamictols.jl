module DynamicTols

import ..MPSKit: Algorithm
using Accessors
using DocStringExtensions
using MatrixAlgebraKit: DefaultAlgorithm
using KrylovKit: KrylovKit, Lanczos, Arnoldi

export updatetol, DynamicTol
export AdaptiveKrylov, instantiate_algorithm

@doc """
    updatetol(alg, iter, ϵ)

Update the tolerance of the algorithm `alg` based on the current iteration `iter` and the current error `ϵ`.
""" updatetol

updatetol(alg, iter::Integer, ϵ::Real) = alg

# Wrapper for dynamic tolerance adjustment
# ----------------------------------------

"""
$(TYPEDEF)

Algorithm wrapper with dynamically adjusted tolerances.

## Fields

$(TYPEDFIELDS)

See also [`updatetol`](@ref).
"""
struct DynamicTol{A} <: Algorithm
    "parent algorithm"
    alg::A

    "minimal value of the dynamic tolerance"
    tol_min::Float64

    "maximal value of the dynamic tolerance"
    tol_max::Float64

    "tolerance factor for updating relative to current algorithm error"
    tol_factor::Float64

    function DynamicTol(
            alg::A, tol_min::Real, tol_max::Real, tol_factor::Real
        ) where {A}
        0 <= tol_min <= tol_max ||
            throw(ArgumentError("tol_min must be between 0 and tol_max"))
        return new{A}(alg, tol_min, tol_max, tol_factor)
    end
end
function DynamicTol(alg; tol_min = 1.0e-6, tol_max = 1.0e-2, tol_factor = 0.1)
    return DynamicTol(alg, tol_min, tol_max, tol_factor)
end

"""
    updatetol(alg::DynamicTol, iter, ϵ)

Update the tolerance of the algorithm `alg` based on the current iteration `iter` and the current error `ϵ`,
where the new tolerance is given by
    
    new_tol = clamp(ϵ * alg.tol_factor / sqrt(iter), alg.tol_min, alg.tol_max)
"""
function updatetol(alg::DynamicTol, iter::Integer, ϵ::Real)
    iter = max(iter, one(iter))
    new_tol = clamp(ϵ * alg.tol_factor / sqrt(iter), alg.tol_min, alg.tol_max)
    return _updatetol(alg.alg, new_tol)
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

See also [`instantiate_algorithm`](@ref).
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
    instantiate_algorithm(alg, decay_rate, g_local, g_global, eps_trunc)

Turn a (possibly adaptive) local-eigensolver specification `alg` into a concrete KrylovKit
algorithm for the current site, using the measured `decay_rate` of the previous solve, the
local gradient norm `g_local`, the global gradient norm `g_global` and the local truncation
error `eps_trunc`. The generic fallback returns `alg` unchanged (for plain `Lanczos`/`Arnoldi`).
"""
instantiate_algorithm(alg, args...) = alg

# a `DynamicTol` wrapper is resolved to a concrete solver by tightening its tolerance with the
# global gradient norm (the per-sweep, tol-only "legacy" behaviour, now driven per site).
function instantiate_algorithm(
        alg::DynamicTol, decay_rate::Real, g_local::Real, g_global::Real, eps_trunc::Real
    )
    tol = clamp(g_global * alg.tol_factor, alg.tol_min, alg.tol_max)
    return _updatetol(alg.alg, tol)
end

function instantiate_algorithm(
        alg::AdaptiveKrylov{T}, decay_rate::Real, g_local::Real, g_global::Real, eps_trunc::Real
    ) where {T}
    # 1. target tolerance: inexact inner solve depending on outer convergence,
    #    never below the truncation error we already incur.
    trunc_tol = alg.truncation_factor * eps_trunc
    conv_tol = alg.tol_factor * g_global
    tol = clamp(max(trunc_tol, conv_tol), alg.tol_min, alg.tol_max)

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

end
