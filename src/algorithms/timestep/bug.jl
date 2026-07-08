"""
$(TYPEDEF)

Single site MPS time-evolution algorithm based on the Basis-Update & Galerkin (BUG) integrator,
an unconventional robust integrator for dynamical low-rank approximation.

Unlike [`TDVP`](@ref), BUG advances both the basis-carrying (K-step) and the core (Galerkin C-step)
tensors *forward* in time. In particular there is no backward-in-time substep, which makes it a
natural choice for imaginary-time / dissipative evolution where the backward core step of TDVP
integrators can become unstable.

Passing a truncating `trscheme` (anything other than the default `notrunc()`) switches on
**rank-adaptivity**: each half-sweep augments every bond with the new directions discovered by the
evolved connecting tensor (keeping the old basis as the leading block, `[U₀ │ K₁]`) and then
truncates the enlarged bonds back down to the tolerance of `trscheme` by an truncation sweep.
The bond dimension therefore grows and shrinks automatically to track the entanglement of the
evolving state. The default `notrunc()` recovers the fixed-rank integrator exactly.

!!! note
    Real-time evolution does not normalize the resulting state: neither the augmentation nor
    the truncation normalizes, so the state norm reflects the accumulated truncation error.
    Imaginary-time evolution renormalizes after every half-sweep, similar to a ground-state search.

## Fields

$(TYPEDFIELDS)

## References

* [Ceruti et al. BIT Numer. Math. 62 (2022)](@cite ceruti2022)
"""
struct BUG{A, O, T, S, F} <: Algorithm
    "algorithm used in the exponential solvers"
    integrator::A

    "tolerance for gauging algorithm"
    tolgauge::Float64

    "maximal amount of iterations for gauging algorithm"
    gaugemaxiter::Int

    "algorithm used to re-orthonormalize the basis after each local update"
    alg_orth::O

    "truncation scheme used to cut the bond back down for rank-adaptive BUG"
    trscheme::T

    "algorithm used for the singular value decomposition"
    alg_svd::S

    "callback function applied after each iteration, of signature `finalize(iter, ψ, H, envs) -> ψ, envs`"
    finalize::F
end
function BUG(;
        integrator = Defaults.alg_expsolve(), tolgauge = Defaults.tolgauge,
        gaugemaxiter = Defaults.maxiter, alg_orth = Defaults.alg_orth(),
        trscheme = notrunc(), alg_svd = Defaults.alg_svd(),
        finalize = Defaults._finalize
    )
    return BUG(integrator, tolgauge, gaugemaxiter, alg_orth, trscheme, alg_svd, finalize)
end

function timestep!(
        ψ::AbstractFiniteMPS, H, t::Number, dt::Number, alg::BUG,
        envs::AbstractMPSEnvironments = environments(ψ, H, ψ);
        imaginary_evolution::Bool = false
    )
    # symmetric 2nd-order: a dt/2 left→right half-sweep composed with its dt/2 mirror. A truncating
    # `trscheme` truncates the augmented state back down after each half-sweep (rank-adaptive BUG).
    h = dt / 2
    _bug_sweep_right!(ψ, H, t, h, alg, envs; imaginary_evolution)       # left → right
    _bug_truncates(alg) && _bug_truncate!(ψ, alg; normalize = imaginary_evolution)
    _bug_sweep_left!(ψ, H, t + h, h, alg, envs; imaginary_evolution)    # right → left
    _bug_truncates(alg) && _bug_truncate!(ψ, alg; normalize = imaginary_evolution)
    return ψ, envs
end

# Rank-adaptivity is enabled by any truncating `trscheme` (the default `notrunc()` selects the
# fixed-rank path). Mirrors the `_truncates` gate used by `TDVP`/`DMRG`.
_bug_truncates(alg::BUG) = !(alg.trscheme isa MatrixAlgebraKit.NoTruncation)

# Truncate every internal bond back down to the `trscheme` tolerance with an optimal SVD sweep,
# discarding the singular-value tail of the augmented-Galerkin state. Environments are keyed on
# tensor identity, so they recompute lazily for the changed bonds — no explicit refresh needed.
function _bug_truncate!(ψ, alg::BUG; normalize::Bool = false)
    changebonds!(ψ, SvdCut(; trscheme = alg.trscheme, alg_svd = alg.alg_svd); normalize)
    return ψ
end

# Transport the old→new bond overlap across one site, contracting the front bond + all physical
# legs. `_bug_transport_left` propagates `T` (new ← old) rightward, `_bug_transport_right` the
# mirror (old ← new). Partition-based, so any number of physical legs is supported.
_bug_transport_left(AL_old, AL_new, T) = AL_new' * _mul_front(T, AL_old)
_bug_transport_right(AR_old, AR_new, T) =
    _transpose_tail(AR_old * T) * _transpose_tail(AR_new)'

# New basis + core for one internal node. Fixed-rank keeps the evolved isometry; rank-adaptive
# augments it old-first (`[U₀ │ K₁]`), deferring the cut to `_bug_truncate!`. The tuple order
# matches `ψ.AC[i]`: `(AL, C)` for a left→right sweep, `(C, AR)` for a right→left sweep.
function _bug_leftbasis(alg::BUG, U₀, AC)
    _bug_truncates(alg) || return left_gauge(AC, alg.alg_orth)
    Û, _ = _bug_augment_left(U₀, AC, alg.alg_orth)
    return Û, Û' * AC
end
function _bug_rightbasis(alg::BUG, U₀, AC)
    _bug_truncates(alg) || return right_gauge(AC, alg.alg_orth)
    Û, _ = _bug_augment_right(U₀, AC, alg.alg_orth)
    return _transpose_tail(AC) * _transpose_tail(Û)', Û
end

# Augment the RIGHT bond for the left→right sweep: given the old left-isometry `U₀` (`AL_old`) and
# the evolved candidate `K₁`, build `Û = [U₀ │ Ũ₁]` (`Vl⊗P ← Vr₀ ⊕ Vr_new`) whose column space
# contains both `range(U₀)` and `range(K₁)`, keeping the old basis as the leading per-sector block
# (no truncation). Returns `(Û, M)` with `M = Û' U₀ = [𝟙; 0]` the old bond's coordinates in `Û`.
function _bug_augment_left(U₀, K₁, alg_orth = Defaults.alg_orth())
    N = left_null(U₀)                       # Vl⊗P ← Vc, orthonormal complement of range(U₀)
    g = N' * K₁                             # Vc ← Vr_K, the part of K₁ orthogonal to U₀
    Q, _ = left_orth(g; alg = alg_orth)     # Vc ← Vr_new, orthonormal new directions
    Ũ₁ = N * Q                              # Vl⊗P ← Vr_new
    Û = catdomain(U₀, Ũ₁)                   # Vl⊗P ← (Vr₀ ⊕ Vr_new), old-first
    M = Û' * U₀                             # V̂ ← Vr₀, = [𝟙; 0] per sector
    return Û, M
end

# Mirror of `_bug_augment_left` for the right→left sweep: augment the LEFT bond on the
# `_transpose_tail` form (`Vl ← P⊗Vr`, in which a right-isometry has orthonormal rows). Given the
# old right-isometry `U₀` (`AR_old`) and candidate `K₁`, returns `(Û, M)` with `Û` right-canonical
# on `V̂ = Vl₀ ⊕ Vl_new` (old-first) and `M = û u₀' = [𝟙; 0]` per sector.
function _bug_augment_right(U₀, K₁, alg_orth = Defaults.alg_orth())
    u₀ = _transpose_tail(U₀)                          # Vl₀ ← P⊗Vr, right-isometric (u₀ u₀' = 𝟙)
    k₁ = _transpose_tail(K₁)                          # Vl_K ← P⊗Vr
    N = right_null!(_transpose_tail(U₀; copy = true)) # Vc ← P⊗Vr, complement of U₀'s row space
    g = k₁ * N'                                       # Vl_K ← Vc, the part of K₁ orthogonal to U₀
    _, Q = right_orth(g; alg = alg_orth)              # Q: Vl_new ← Vc, orthonormal new directions
    Ũ₁ = Q * N                                        # Vl_new ← P⊗Vr
    û = catcodomain(u₀, Ũ₁)                           # (Vl₀ ⊕ Vl_new) ← P⊗Vr, old-first
    Û = _transpose_front(û)                           # V̂ ⊗ P ← Vr
    M = û * u₀'                                        # V̂ ← Vl₀, = [𝟙; 0] per sector
    return Û, M
end

# Left→right half-sweep (root = last site, leaf = site 1): reproject the frozen old connecting
# tensor onto the already-updated left bases, evolve it forward, and install the new left basis.
function _bug_sweep_right!(ψ, H, t, τ, alg, envs; imaginary_evolution::Bool = false)
    L = length(ψ)
    ψ.AC[1]                       # gauge center to site 1
    ψ_old = copy(ψ)               # frozen bases / reprojection inputs
    T = isomorphism(scalartype(ψ), left_virtualspace(ψ_old, 1) ← left_virtualspace(ψ_old, 1))

    for i in 1:(L - 1)
        Ĉ = _mul_front(T, ψ_old.AC[i])                       # reproject old connecting tensor
        AC = integrate(AC_hamiltonian(i, ψ, H, ψ, envs), Ĉ, t, τ, alg.integrator; imaginary_evolution)
        U, C = _bug_leftbasis(alg, _mul_front(T, ψ_old.AL[i]), AC)
        T = _bug_transport_left(ψ_old.AL[i], U, T)
        ψ.AC[i] = (U, C)
    end

    # root (site L): evolve, no further basis update
    AC = integrate(
        AC_hamiltonian(L, ψ, H, ψ, envs), _mul_front(T, ψ_old.AC[L]),
        t, τ, alg.integrator; imaginary_evolution
    )
    imaginary_evolution && normalize!(AC)
    ψ.AC[L] = AC
    return ψ
end

# Right→left half-sweep (root = first site, leaf = last site): the mirror of `_bug_sweep_right!`.
function _bug_sweep_left!(ψ, H, t, τ, alg, envs; imaginary_evolution::Bool = false)
    L = length(ψ)
    ψ.AC[L]                       # gauge center to site L
    ψ_old = copy(ψ)
    T = isomorphism(scalartype(ψ), right_virtualspace(ψ_old, L) ← right_virtualspace(ψ_old, L))

    for i in L:-1:2
        Ĉ = ψ_old.AC[i] * T                                  # reproject old connecting tensor
        AC = integrate(AC_hamiltonian(i, ψ, H, ψ, envs), Ĉ, t, τ, alg.integrator; imaginary_evolution)
        C, U = _bug_rightbasis(alg, ψ_old.AR[i] * T, AC)
        T = _bug_transport_right(ψ_old.AR[i], U, T)
        ψ.AC[i] = (C, U)
    end

    # root (site 1): evolve, no further basis update
    AC = integrate(
        AC_hamiltonian(1, ψ, H, ψ, envs), ψ_old.AC[1] * T,
        t, τ, alg.integrator; imaginary_evolution
    )
    imaginary_evolution && normalize!(AC)
    ψ.AC[1] = AC
    return ψ
end

# copying version
function timestep(
        ψ::AbstractFiniteMPS, H, time::Number, timestep::Number,
        alg::BUG, envs::AbstractMPSEnvironments...;
        imaginary_evolution::Bool = false, kwargs...
    )
    isreal = (scalartype(ψ) <: Real && !imaginary_evolution)
    ψ′ = isreal ? complex(ψ) : copy(ψ)
    if length(envs) != 0 && isreal
        @warn "Currently cannot reuse real environments for complex evolution"
        envs′ = environments(ψ′, H, ψ′)
    elseif length(envs) == 1
        envs′ = only(envs)
    else
        @assert length(envs) == 0 "Invalid signature"
        envs′ = environments(ψ′, H, ψ′)
    end
    return timestep!(ψ′, H, time, timestep, alg, envs′; imaginary_evolution, kwargs...)
end
