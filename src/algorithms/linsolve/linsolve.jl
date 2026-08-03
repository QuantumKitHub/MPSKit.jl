# Flavours
# --------
"""
$(TYPEDEF)

Abstract supertype selecting how the local linear subproblem is posed in an MPS [`linsolve`](@ref)
sweep. See [`Galerkin`](@ref) and [`LeastSquares`](@ref).
"""
abstract type LinsolveFormulation end

"""
$(TYPEDEF)

Local formulation imposing the Galerkin condition: the residual `(a₀ + a₁·A)·x − b` is projected
orthogonal to the local tangent space, yielding the effective system `(a₀ + a₁·A_eff)·x = b_eff`
which is solved with the configured Krylov `solver`. Use a `CG` solver for hermitian
positive-definite `A`, and `GMRES`/`BiCGStab` for general (non-hermitian or indefinite) `A`.
"""
struct Galerkin <: LinsolveFormulation end

"""
$(TYPEDEF)

Local formulation that minimizes `‖(a₀ + a₁·A)·x − b‖²`, i.e. solves the normal equations
`M†M·x = M†b` with `M = a₀ + a₁·A`, built from squared operator environments. The normal operator
is unconditionally positive-definite (solve it with `CG`) at the cost of squaring the condition
number.

Currently assumes `A` is hermitian (the resolvent / dynamical-DMRG case), for which
`M†M = |a₀|² + 2·Re(ā₀·a₁)·A + |a₁|²·A²` and `M†b = ā₀·b + ā₁·A·b`. This is a generalization of the
[`Jeckelmann`](@ref) dynamical-DMRG functional.
"""
struct LeastSquares <: LinsolveFormulation end

# Algorithms
# ----------
"""
$(TYPEDEF)

Single-site DMRG-style sweeping algorithm for the MPS linear solver [`linsolve`](@ref). It keeps
the bond dimension of the initial guess fixed and, at each site, solves the local linear problem
selected by `formulation` with the local Krylov `solver`.

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct DMRGSolve{F <: LinsolveFormulation, S, FIN} <: Algorithm
    "formulation of the local subproblem, either [`Galerkin`](@ref) or [`LeastSquares`](@ref)"
    formulation::F = Galerkin()
    "local linear solver; a plain KrylovKit solver, or one wrapped in `DynamicTol` for per-bond adaptive tolerances (the default)"
    solver::S = Defaults.alg_linsolve()
    "tolerance for convergence criterium"
    tol::Float64 = Defaults.tol
    "maximal amount of iterations"
    maxiter::Int = Defaults.maxiter
    "setting for how much information is displayed"
    verbosity::Int = Defaults.verbosity
    "callback function applied after each iteration, of signature `finalize(iter, x, A, envs) -> x, envs`"
    finalize::FIN = Defaults._finalize
end

"""
$(TYPEDEF)

Two-site DMRG-style sweeping algorithm for the MPS linear solver [`linsolve`](@ref). Each bond
update solves the local two-site linear problem selected by `formulation` and truncates the enlarged
bond back down with `alg_gauge` (a truncated SVD built from `trunc`), making the bond dimension
adaptive.

# Fields

$(TYPEDFIELDS)
"""
struct DMRGSolve2{F <: LinsolveFormulation, S, G, FIN} <: Algorithm
    "formulation of the local subproblem, either [`Galerkin`](@ref) or [`LeastSquares`](@ref)"
    formulation::F
    "local linear solver; a plain KrylovKit solver, or one wrapped in `DynamicTol` for per-bond adaptive tolerances"
    solver::S
    "tolerance for convergence criterium"
    tol::Float64
    "maximal amount of iterations"
    maxiter::Int
    "setting for how much information is displayed"
    verbosity::Int
    "truncated SVD used for the post-update gauge"
    alg_gauge::G
    "callback function applied after each iteration, of signature `finalize(iter, x, A, envs) -> x, envs`"
    finalize::FIN
end
function DMRGSolve2(;
        formulation = Galerkin(), solver = Defaults.alg_linsolve(), tol = Defaults.tol,
        maxiter = Defaults.maxiter, verbosity = Defaults.verbosity,
        alg_svd = Defaults.alg_svd(), trunc, finalize = Defaults._finalize
    )
    alg_gauge = MatrixAlgebraKit.TruncatedAlgorithm(alg_svd, trunc)
    return DMRGSolve2(formulation, solver, tol, maxiter, verbosity, alg_gauge, finalize)
end

# Interface
# ---------
@doc """
    linsolve(x₀, A, b, [algorithm], [environments]; kwargs...) -> (x, environments, ϵ)
    linsolve!(x₀, A, b, [algorithm], [environments]; kwargs...) -> (x, environments, ϵ)

Solve the linear system `(a₀ + a₁·A)·x = b` for the MPS `x`, where `A` is an MPO / MPOHamiltonian
and `b` is an MPS. The `a₀ + a₁·A` shift follows `KrylovKit.linsolve` and is applied implicitly;
the defaults `a₀ = 0`, `a₁ = 1` give the plain system `A·x = b`. `linsolve!` solves in-place,
overwriting `x₀`.

The initial guess `x₀` comes first, mirroring [`find_groundstate`](@ref) and [`approximate`](@ref)
(the operator/right-hand side follow), so it sits in the same argument slot in both the copying and
in-place forms.

Currently only finite MPS are supported.

# Arguments
- `x₀::AbstractMPS`: initial guess (and, for `linsolve!`, the state overwritten with the solution)
- `A`: operator of the linear system (MPO / MPOHamiltonian)
- `b::AbstractMPS`: right-hand side
- `algorithm`: linear-solve algorithm, see [`DMRGSolve`](@ref) and [`DMRGSolve2`](@ref)
- `[environments]`: MPS environment manager for the operator sandwich `⟨x|A|x⟩`

# Keyword Arguments
The keyword-based call (no explicit `algorithm`) selects an algorithm from the structure flags:
- `tol::Float64`: tolerance for convergence criterium
- `maxiter::Int`: maximum amount of iterations
- `verbosity::Int`: display progress information
- `a₀`, `a₁`: shift/scale scalars of the system `(a₀ + a₁·A)·x = b`
- `ishermitian::Bool`, `isposdef::Bool`: declare structure of `(a₀ + a₁·A)` so the default local
  solver is chosen accordingly (`CG` for positive-definite, `GMRES` otherwise)
- `trunc`: if supplied, a truncated two-site sweep ([`DMRGSolve2`](@ref)) is prepended to
  adapt the bond dimension before the single-site algorithm polishes the result

# Returns
- `x::AbstractMPS`: the (bond-dimension-limited) solution
- `environments`: operator environments corresponding to `x`
- `ϵ::Float64`: final convergence error — the largest local residual `‖(a₀ + a₁·A)·x − b‖`
  over the sweep, relative to `‖b‖` (the linear-solve analogue of the Galerkin error used by
  [`find_groundstate`](@ref))
""" linsolve

# NOTE: `@doc str a, b` attaches only to the last binding, so share it explicitly
@doc (@doc linsolve) linsolve!

# scalar-promote an initial state so it can hold a complex solution when the shift is complex
function _promote_state(x::AbstractMPS, a₀, a₁)
    T = promote_type(scalartype(x), typeof(a₀), typeof(a₁))
    return T <: Complex ? complex(x) : x
end

# default (adaptive) local solver from the declared operator structure, mirroring KrylovKit's rule
function _default_linsolve_algorithm(
        x::AbstractMPS; tol, maxiter, verbosity, ishermitian, isposdef, trunc
    )
    x isa AbstractFiniteMPS ||
        throw(ArgumentError("`linsolve` currently only supports finite MPS"))
    solver = Defaults.alg_linsolve(; ishermitian, isposdef, tol, maxiter)
    alg = DMRGSolve(; solver, tol, maxiter, verbosity)
    if !isnothing(trunc)
        alg = DMRGSolve2(;
            solver, tol = min(1.0e-2, 100tol), maxiter, verbosity, trunc
        ) & alg
    end
    return alg
end

# keyword form: build a default algorithm from the initial guess and dispatch
function linsolve(
        x₀::AbstractMPS, A, b::AbstractMPS;
        tol = Defaults.tol, maxiter = Defaults.maxiter, verbosity = Defaults.verbosity,
        a₀ = 0, a₁ = 1, ishermitian = false, isposdef = false, trunc = nothing
    )
    alg = _default_linsolve_algorithm(
        x₀; tol, maxiter, verbosity, ishermitian, isposdef, trunc
    )
    return linsolve(x₀, A, b, alg; a₀, a₁)
end

# explicit-algorithm form: copy (and scalar-promote) the guess and solve in-place. The `envs`
# splat lets `linsolve!` build the environments from the promoted copy when none are supplied.
function linsolve(x₀, A, b, alg::Union{DMRGSolve, DMRGSolve2}, envs...; a₀ = 0, a₁ = 1)
    return linsolve!(_promote_state(copy(x₀), a₀, a₁), A, b, alg, envs...; a₀, a₁)
end

# sequential chaining of algorithms (e.g. a two-site pass then a single-site polish)
function linsolve(x₀, A, b, alg::UnionAlg, envs...; a₀ = 0, a₁ = 1)
    x, newenvs, = linsolve(x₀, A, b, alg.alg1, envs...; a₀, a₁)
    return linsolve(x, A, b, alg.alg2, newenvs; a₀, a₁)
end
function linsolve!(x₀, A, b, alg::UnionAlg, envs...; a₀ = 0, a₁ = 1)
    x, newenvs, = linsolve!(x₀, A, b, alg.alg1, envs...; a₀, a₁)
    return linsolve!(x, A, b, alg.alg2, newenvs; a₀, a₁)
end
