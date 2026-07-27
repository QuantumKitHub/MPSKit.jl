"""
$(TYPEDEF)

Abstract supertype for the different flavours of dynamical DMRG.
"""
abstract type DDMRG_Flavour end

"""
$(TYPEDEF)

An alternative approach to the dynamical DMRG algorithm, without quadratic terms but with a
less controlled approximation.
This algorithm minimizes the following cost function
```math
⟨ψ|(z - H)|ψ⟩ - ⟨ψ|ψ₀⟩ - ⟨ψ₀|ψ⟩
```

Returns the approximation of ``⟨ψ₀|\\frac{1}{z - H}|ψ₀⟩`` and ``\\frac{1}{z - H}|ψ₀⟩``.

# See also

[`Jeckelmann`](@ref) for the original approach.
"""
struct NaiveInvert <: DDMRG_Flavour end

"""
$(TYPEDEF)

The original flavour of dynamical DMRG, which minimizes functional (14) from Jeckelmann2002.
Writing ``ω = \\mathrm{Re}(z)`` and ``η = \\mathrm{Im}(z)``, this is
```math
W(ψ) = ⟨ψ|(ω - H)^2 + η^2|ψ⟩ + η(⟨ψ₀|ψ⟩ + ⟨ψ|ψ₀⟩)
```
which attains its minimum at
```math
((ω - H)^2 + η^2)|ψ⟩ = -η|ψ₀⟩
```

The solution of that equation is the imaginary part of the propagator; together with equation (11)
from that same paper it determines the full ``⟨ψ₀|\\frac{1}{z - H}|ψ₀⟩``. Because of that
reconstruction step this flavour requires ``η = \\mathrm{Im}(z) ≠ 0``.

Returns the approximation of ``⟨ψ₀|\\frac{1}{z - H}|ψ₀⟩`` and the minimizer ``|ψ⟩`` of the functional
above (*not* ``\\frac{1}{z - H}|ψ₀⟩`` itself).

# See also

[`NaiveInvert`](@ref) for a less costly but less accurate alternative.

# References

* [Jeckelmann. Phys. Rev. B 66 (2002)](@cite jeckelmann2002)
"""
struct Jeckelmann <: DDMRG_Flavour end

# default local linear solver per flavour, following the structure of the effective operator:
# `z - H` is non-hermitian for complex `z`, while the Jeckelmann operator is hermitian but (after the
# rescaling below) indefinite. Both currently resolve to GMRES, since KrylovKit has no `linsolve`
# method for `MINRES` yet; declaring the structure means Jeckelmann picks it up once it does.
_ddmrg_solver(::NaiveInvert) = Defaults.alg_linsolve()
_ddmrg_solver(::Jeckelmann) = Defaults.alg_linsolve(; ishermitian = true)

"""
$(TYPEDEF)

A dynamical DMRG method for calculating dynamical properties and excited states, based on a
variational principle for dynamical correlation functions.

This is a thin wrapper around [`linsolve`](@ref): the sweep, the convergence criterion and the
adaptive local tolerances are those of [`DMRGSolve`](@ref) / [`DMRGSolve2`](@ref).

# Fields

$(TYPEDFIELDS)

# See also

Used as the `algorithm` argument of [`propagator`](@ref).

# References

* [Jeckelmann. Phys. Rev. B 66 (2002)](@cite jeckelmann2002)
"""
struct DynamicalDMRG{F <: DDMRG_Flavour, S, T} <: Algorithm
    "flavour of the algorithm to use, either of type [`NaiveInvert`](@ref) or [`Jeckelmann`](@ref)"
    flavour::F
    "local linear solver; a plain KrylovKit solver, or one wrapped in `DynamicTol` for per-bond adaptive tolerances (the default)"
    solver::S
    "tolerance for convergence criterium, measured as the relative residual of the linear system"
    tol::Float64
    "maximal amount of iterations"
    maxiter::Int
    "setting for how much information is displayed"
    verbosity::Int
    "if supplied, a truncated two-site sweep ([`DMRGSolve2`](@ref)) is prepended to adapt the bond dimension"
    trunc::T
end
function DynamicalDMRG(;
        flavour = NaiveInvert(), solver = _ddmrg_solver(flavour), tol = Defaults.tol,
        maxiter = Defaults.maxiter, verbosity = Defaults.verbosity, trunc = nothing
    )
    return DynamicalDMRG(flavour, solver, tol, maxiter, verbosity, trunc)
end

# mirrors `_default_linsolve_algorithm`: a loose two-site pass to grow the bond dimension, then a
# single-site polish at the requested tolerance
function _ddmrg_algorithm(alg::DynamicalDMRG)
    (; solver, tol, maxiter, verbosity) = alg
    alg_1site = DMRGSolve(; solver, tol, maxiter, verbosity)
    isnothing(alg.trunc) && return alg_1site
    return DMRGSolve2(;
        solver, tol = min(1.0e-2, 100tol), maxiter, verbosity, alg.trunc
    ) & alg_1site
end

# The linear system `(a₀ + a₁·A)·x = |ψ₀⟩` solved by each flavour.
#
# `NaiveInvert` is the resolvent itself, `(z - H)·x = |ψ₀⟩`.
#
# `Jeckelmann` is functional (14)'s stationarity condition `((ω - H)² + η²)·ψ = -η|ψ₀⟩`. Scaling that
# equation by `-1/η` puts it in the `(a₀ + a₁·A)·x = b` form with `b = |ψ₀⟩` itself, so no scaled
# copy of the right-hand side is needed and `x` is the very same vector as before. The operator
# `A = H² - 2ω·H` is assembled as a `LinearCombination`, whose `AC_hamiltonian` method reproduces
# exactly the local operator this algorithm used to build by hand.
_ddmrg_shift(::NaiveInvert, z) = (z, -one(z))
function _ddmrg_shift(::Jeckelmann, z)
    η = imag(z)
    iszero(η) && throw(
        ArgumentError(
            "`Jeckelmann` requires `imag(z) != 0`; use `NaiveInvert` flavour for real `z`"
        )
    )
    return (-abs2(z) / η, -inv(η))
end

_ddmrg_operator(::NaiveInvert, H, x, envs, z) = (H, envs)
function _ddmrg_operator(::Jeckelmann, H, x, envs, z)
    H², envs² = squaredenvs(x, H, envs)
    A = LinearCombination((H², H), (one(real(z)), -2 * real(z)))
    return A, LazyLincoCache(A, (envs², envs))
end

# `G(z)` from the solution vector
_ddmrg_value(::NaiveInvert, x, ψ₀, z, H, envs) = dot(ψ₀, x)
function _ddmrg_value(::Jeckelmann, x, ψ₀, z, H, envs)
    ω, η = real(z), imag(z)
    # equation (11) of Jeckelmann2002: the solve only fixes the imaginary part of the propagator,
    # the real part follows from ⟨ψ₀|H|x⟩
    a = dot(ψ₀, x)
    cb = leftenv(envs, 1, ψ₀) * TransferMatrix(x.AL, H[1:length(ψ₀.AL)], ψ₀.AL)
    b = zero(a)
    for i in 1:length(cb)
        b += @plansor cb[i][1 2; 3] * x.C[end][3; 4] *
            rightenv(envs, length(ψ₀), ψ₀)[i][4 2; 5] * conj(ψ₀.C[end][1; 5])
    end
    return b / η - ω / η * a + 1im * a
end

"""
    propagator(ψ₀::AbstractFiniteMPS, z::Number, H, alg::DynamicalDMRG; init = ψ₀) -> (g, ψ)

Calculate the action of the propagator ``\\frac{1}{z - H}|ψ₀⟩`` using the dynamical DMRG
algorithm.

# Returns

- `g`: approximation of the propagator matrix element ``⟨ψ₀|\\frac{1}{z - H}|ψ₀⟩``
- `ψ`: for [`NaiveInvert`](@ref), the MPS approximation of ``\\frac{1}{z - H}|ψ₀⟩``; for
  [`Jeckelmann`](@ref), the vector its functional optimizes,
  ``-η[(ω - H)^2 + η^2]^{-1}|ψ₀⟩``, i.e. the imaginary part of the propagator, from which `g` is
  reconstructed.

`init` is used as the initial guess and is left untouched. The underlying variational problem is
solved with [`linsolve`](@ref); for full control over the sweep, call that directly, e.g.
`linsolve(ψ₀, H, ψ₀; a₀ = z, a₁ = -1)` for the [`NaiveInvert`](@ref) flavour.
"""
function propagator(
        ψ₀::AbstractFiniteMPS, z::Number, H, alg::DynamicalDMRG; init = ψ₀
    )
    a₀, a₁ = _ddmrg_shift(alg.flavour, z)
    x = _promote_state(copy(init), a₀, a₁)
    envs = environments(x, H, x)
    A, Aenvs = _ddmrg_operator(alg.flavour, H, x, envs, z)
    x, = linsolve!(x, A, ψ₀, _ddmrg_algorithm(alg), Aenvs; a₀, a₁)
    return _ddmrg_value(alg.flavour, x, ψ₀, z, H, envs), x
end

function squaredenvs(
        state::AbstractFiniteMPS, H, envs = environments(state, H, state)
    )
    H² = conj(H) * H
    L = length(state)

    # impose the correct boundary conditions (important for WindowMPS)
    leftstart = _contract_leftenv²(leftenv(envs, 1, state), leftenv(envs, 1, state))
    rightstart = _contract_rightenv²(rightenv(envs, L, state), rightenv(envs, L, state))

    # to construct the squared caches we will first initialize environments
    # then make all data invalid so it will be recalculated
    envs² = environments(state, H², state; leftstart, rightstart)
    for i in 1:L
        poison!(envs², i)
    end

    return H², envs²
end

function _contract_leftenv²(GL_top, GL_bot)
    V_mid = space(GL_bot, 2)' ⊗ space(GL_top, 2)
    F = isomorphism(storagetype(GL_top), fuse(V_mid)' ← V_mid)
    return @plansor GL[-1 -2; -3] := GL_top[1 3; -3] * conj(GL_bot[1 2; -1]) * F[-2; 2 3]
end

function _contract_rightenv²(GR_top, GR_bot)
    V_mid = space(GR_top, 2) ⊗ space(GR_bot, 2)'
    F = isomorphism(storagetype(GR_top), fuse(V_mid) ← V_mid)
    return @plansor GR[-1 -2; -3] := GR_top[-1 2; 1] * conj(GR_bot[-3 3; 1]) * F[-2; 2 3]
end
