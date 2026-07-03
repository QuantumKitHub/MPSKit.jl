@doc """
    approximate(ψ₀, (O, ψ), [environments]; kwargs...) -> (ψ, environments, ϵ)
    approximate(ψ₀, (O, ψ), algorithm, [environments]) -> (ψ, environments, ϵ)
    approximate!(ψ₀, (O, ψ), algorithm, [environments]) -> (ψ, environments, ϵ)
    approximate(ψ₀, ψ, algorithm, [environments]) -> (ψ, environments, ϵ)
    approximate!(ψ₀, ψ, algorithm, [environments]) -> (ψ, environments, ϵ)

Compute an approximation to the application of an operator `O` to the state `ψ` in the form
of an MPS `ψ₀`. If only a state `ψ` is supplied instead of the `(O, ψ)` pair, `ψ₀` is
approximated directly to `ψ` (i.e. `O` is taken to be the identity).

**Not every algorithm supports every combination of arguments below** — see the per-algorithm
notes at the end of this docstring before picking one.

# Arguments
- `ψ₀::AbstractMPS`: initial guess of the approximated state
- `(O::AbstractMPO, ψ::AbstractMPS)`: operator `O` and state `ψ` to be approximated
- `ψ::AbstractMPS`: state to be approximated directly (without an operator)
- `algorithm`: approximation algorithm. See below for a list of available algorithms.
- `[environments]`: MPS environment manager

# Keyword Arguments
The keyword-based call (no explicit `algorithm`) is a convenience method that picks an
algorithm for you based on the type of `ψ₀` (`DMRG`/`DMRG2` for a finite MPS, `VOMPS`/`IDMRG`/
`IDMRG2` for an infinite MPS) and only accepts the `(O, ψ)` tuple form of `toapprox`. Once you
pass an explicit `algorithm`, keywords are no longer accepted here — configure the algorithm
struct itself instead (e.g. `DMRG(; tol, maxiter, verbosity)`).
- `tol::Float64`: tolerance for convergence criterium
- `maxiter::Int`: maximum amount of iterations
- `verbosity::Int`: display progress information
- `trscheme`: if supplied, a truncated two-site sweep (`DMRG2`/`IDMRG2`) is prepended to
  refine the bond dimension before the single-site algorithm polishes the result.

# Algorithms
Each algorithm below only supports a subset of the general interface. Check this table before
picking one — in particular, note that **only `DMRG`/`DMRG2` accept a bare state `ψ`**; the
infinite algorithms always require an explicit `(O, ψ)` tuple, and **`VOMPS` has no in-place
`approximate!`** at all.

| Algorithm | Scheme                        | State `ψ₀`                        | bare `ψ` allowed? | `approximate!` |
|:--------- |:----------------------------- |:---------------------------------- |:------------------:|:--------------:|
| `DMRG`    | single-site, fixes bond dim    | `AbstractFiniteMPS`                | ✅                  | ✅              |
| `DMRG2`   | two-site, truncates via `trscheme` | `AbstractFiniteMPS`            | ✅                  | ✅              |
| `IDMRG`   | single-site, thermodynamic limit | `InfiniteMPS` / `MultilineMPS`  | ❌ (tuple only)     | ✅              |
| `IDMRG2`  | two-site, thermodynamic limit, needs unit cell ≥ 2 | `InfiniteMPS` / `MultilineMPS` | ❌ (tuple only) | ✅ |
| `VOMPS`   | tangent-space truncation       | `InfiniteMPS` / `MultilineMPS`     | ❌ (tuple only)     | ❌ (out-of-place only) |

`InfiniteMPS`/`InfiniteMPO` inputs are converted internally to `MultilineMPS`/`MultilineMPO`
for `IDMRG`, `IDMRG2`, and `VOMPS`; you can also pass those types directly.
"""
approximate, approximate!

# the trailing `environments` arguments for an operator/ket bundle:
# a tuple carries an explicit operator (3-argument form), a bare state means overlap (2-argument form).
_environment_args(Oϕ::Tuple) = Oϕ
_environment_args(ϕ) = (ϕ,)

function approximate(
        ψ::AbstractMPS, toapprox::Tuple{<:AbstractMPO, <:AbstractMPS},
        envs::AbstractMPSEnvironments = environments(ψ, toapprox...);
        tol = Defaults.tol, maxiter = Defaults.maxiter,
        verbosity = Defaults.verbosity, trscheme = nothing
    )
    if isa(ψ, InfiniteMPS)
        alg = VOMPS(; tol, verbosity, maxiter)
        if !isnothing(trscheme)
            alg = IDMRG2(; tol = min(1.0e-2, 100tol), verbosity, trscheme) & alg
        end
    elseif isa(ψ, AbstractFiniteMPS)
        alg = DMRG(; tol, maxiter, verbosity)
        if !isnothing(trscheme)
            alg = DMRG2(; tol = min(1.0e-2, 100tol), verbosity, trscheme) & alg
        end
    else
        throw(ArgumentError("Unknown input state type"))
    end

    return approximate(ψ, toapprox, alg, envs)
end


# implementation in terms of Multiline
function approximate(
        ψ::InfiniteMPS, toapprox::Tuple{<:InfiniteMPO, <:InfiniteMPS}, algorithm,
        envs = environments(ψ, toapprox...)
    )
    envs′ = Multiline([envs])
    multi, envs, δ = approximate(
        convert(MultilineMPS, ψ),
        (convert(MultilineMPO, toapprox[1]), convert(MultilineMPS, toapprox[2])),
        algorithm, envs′
    )
    ψ = convert(InfiniteMPS, multi)
    return ψ, envs, δ
end

# dispatch to in-place method
function approximate(
        ψ, toapprox, alg::Union{DMRG, DMRG2, IDMRG, IDMRG2}, envs...
    )
    return approximate!(copy(ψ), toapprox, alg, envs...)
end

# disambiguate
function approximate(
        ψ::InfiniteMPS, toapprox::Tuple{<:InfiniteMPO, <:InfiniteMPS},
        algorithm::Union{IDMRG, IDMRG2}, envs = environments(ψ, toapprox...)
    )
    envs′ = Multiline([envs])
    multi, envs, δ = approximate(
        convert(MultilineMPS, ψ),
        (convert(MultilineMPO, toapprox[1]), convert(MultilineMPS, toapprox[2])),
        algorithm, envs′
    )
    ψ = convert(InfiniteMPS, multi)
    return ψ, envs, δ
end
