# Flavours
# --------
"""
$(TYPEDEF)

Abstract supertype selecting how the local linear subproblem is posed in an MPS
[`linear_solve`](@ref) sweep. See [`Galerkin`](@ref) and [`LeastSquares`](@ref).
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

Single-site DMRG-style sweeping algorithm for the MPS linear solver [`linear_solve`](@ref). It keeps
the bond dimension of the initial guess fixed and, at each site, solves the local linear problem
selected by `formulation` with the local Krylov `solver`.

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct DMRGSolve{L <: LinsolveFormulation, S, F, B} <: Algorithm
    "formulation of the local subproblem, either [`Galerkin`](@ref) or [`LeastSquares`](@ref)"
    formulation::L = Galerkin()
    "local linear solver; a plain KrylovKit solver, or one wrapped in `DynamicTol` for per-bond adaptive tolerances (the default)"
    solver::S = Defaults.alg_linsolve()
    "tolerance for convergence criterium"
    tol::Float64 = Defaults.tol
    "maximal amount of iterations"
    maxiter::Int = Defaults.maxiter
    "setting for how much information is displayed"
    verbosity::Int = Defaults.verbosity
    "callback function applied after each iteration, of signature `finalize(iter, x, A, envs) -> x, envs`"
    finalize::F = Defaults._finalize
    "backend for tensor contractions and index manipulations"
    backend::B = Defaults.backend()
end

"""
$(TYPEDEF)

Two-site DMRG-style sweeping algorithm for the MPS linear solver [`linear_solve`](@ref). Each bond
update solves the local two-site linear problem selected by `formulation` and truncates the enlarged
bond back down with `alg_gauge` (a truncated SVD built from `trunc`), making the bond dimension
adaptive.

# Fields

$(TYPEDFIELDS)
"""
struct DMRGSolve2{L <: LinsolveFormulation, S, G, F, B} <: Algorithm
    "formulation of the local subproblem, either [`Galerkin`](@ref) or [`LeastSquares`](@ref)"
    formulation::L
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
    finalize::F
    "backend for tensor contractions and index manipulations"
    backend::B
end
function DMRGSolve2(;
        formulation = Galerkin(), solver = Defaults.alg_linsolve(), tol = Defaults.tol,
        maxiter = Defaults.maxiter, verbosity = Defaults.verbosity,
        alg_svd = Defaults.alg_svd(), trunc, finalize = Defaults._finalize,
        backend = Defaults.backend()
    )
    alg_gauge = MatrixAlgebraKit.TruncatedAlgorithm(alg_svd, trunc)
    return DMRGSolve2(formulation, solver, tol, maxiter, verbosity, alg_gauge, finalize, backend)
end

# Interface
# ---------
"""
    linear_solve(x₀, A, b, [algorithm], [environments]; kwargs...) -> (x, environments, info)
    linear_solve!(x₀, A, b, [algorithm], [environments]; kwargs...) -> (x, environments, info)

Solve the linear system `(a₀ + a₁·A)·x = b` for the MPS `x`, where `A` is an MPO / MPOHamiltonian
and `b` is an MPS. The `a₀ + a₁·A` shift follows `KrylovKit.linsolve` and is applied implicitly;
the defaults `a₀ = 0`, `a₁ = 1` give the plain system `A·x = b`. `linear_solve!` solves in-place,
overwriting `x₀`.

Currently only finite MPS are supported.

# Arguments
- `x₀::AbstractMPS`: initial guess (and, for `linear_solve!`, the state overwritten with the
  solution)
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
- `isposdef::Bool`: declare `(a₀ + a₁·A)` positive-definite, so the local solver is `CG` instead of `GMRES`
- `trunc`: if supplied, a truncated two-site sweep ([`DMRGSolve2`](@ref)) is prepended to
  adapt the bond dimension before the single-site algorithm polishes the result

# Returns
- `x::AbstractMPS`: the (bond-dimension-limited) solution
- `environments`: operator environments corresponding to `x`
- `info::AlgorithmInfo`: how the algorithm terminated. `info.galerkin` is the Galerkin error,
  the largest local residual `‖(a₀ + a₁·A)·x − b‖` of the last sweep relative to `‖b‖`, and `info.converged` whether
  it met the stopping test. [`DMRGSolve2`](@ref) also reports `info.truncation_errors`. See
  [`AlgorithmInfo`](@ref).
"""
function linear_solve end
@doc (@doc linear_solve) function linear_solve! end

# scalar-promote an initial state so it can hold a complex solution when the shift is complex
function promote_to_shift(x::AbstractMPS, a₀, a₁)
    T = promote_type(scalartype(x), typeof(a₀), typeof(a₁))
    return T <: Complex ? complex(x) : x
end

# a loose two-site pass to grow the bond dimension (if `trunc` is given), then a single-site polish
function linear_solve_algorithm(; solver, tol, maxiter, verbosity, trunc, backend = Defaults.backend())
    alg = DMRGSolve(; solver, tol, maxiter, verbosity, backend)
    isnothing(trunc) && return alg
    return DMRGSolve2(;
        solver, tol = min(1.0e-2, 100tol), maxiter, verbosity, trunc, backend
    ) & alg
end

function linear_solve(
        x₀::AbstractMPS, A, b::AbstractMPS;
        tol = Defaults.tol, maxiter = Defaults.maxiter, verbosity = Defaults.verbosity,
        a₀ = 0, a₁ = 1, isposdef = false, trunc = nothing
    )
    x₀ isa AbstractFiniteMPS ||
        throw(ArgumentError("`linear_solve` currently only supports finite MPS"))
    solver = Defaults.alg_linsolve(; isposdef, tol, maxiter)
    alg = linear_solve_algorithm(; solver, tol, maxiter, verbosity, trunc)
    return linear_solve(x₀, A, b, alg; a₀, a₁)
end

# `envs...` lets `linear_solve!` build the environments from the promoted copy when none are supplied
function linear_solve(x₀, A, b, alg::Union{DMRGSolve, DMRGSolve2}, envs...; a₀ = 0, a₁ = 1)
    return linear_solve!(promote_to_shift(copy(x₀), a₀, a₁), A, b, alg, envs...; a₀, a₁)
end

# sequential chaining of algorithms (e.g. a two-site pass then a single-site polish)
function linear_solve(x₀, A, b, alg::UnionAlg, envs...; a₀ = 0, a₁ = 1)
    x, newenvs, = linear_solve(x₀, A, b, alg.alg1, envs...; a₀, a₁)
    return linear_solve!(x, A, b, alg.alg2, newenvs; a₀, a₁)
end
function linear_solve!(x₀, A, b, alg::UnionAlg, envs...; a₀ = 0, a₁ = 1)
    x, newenvs, = linear_solve!(x₀, A, b, alg.alg1, envs...; a₀, a₁)
    return linear_solve!(x, A, b, alg.alg2, newenvs; a₀, a₁)
end
