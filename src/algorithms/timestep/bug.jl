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
    # symmetric 2nd-order: a dt/2 left→right half-sweep composed with its dt/2 mirror. Each
    # half-sweep reprojects the frozen old connecting tensor onto the already-updated bases, evolves
    # it forward, and installs the new basis; a truncating `trscheme` (rank-adaptive BUG) then cuts
    # the augmented state back down with an optimal SVD sweep. Fixed-rank keeps the evolved isometry,
    # rank-adaptive augments it old-first (`[U₀ │ K₁]`, deferring the cut to `changebonds!`).
    L = length(ψ)
    h = dt / 2
    truncates = !(alg.trscheme isa MatrixAlgebraKit.NoTruncation)
    svdcut = SvdCut(; trscheme = alg.trscheme, alg_svd = alg.alg_svd)

    # sweep left to right (root = last site)
    ψ.AC[1]                       # gauge center to site 1
    ψ_old = copy(ψ)               # frozen bases / reprojection inputs
    T = isomorphism(scalartype(ψ), left_virtualspace(ψ_old, 1) ← left_virtualspace(ψ_old, 1))
    for i in 1:(L - 1)
        Ĉ = _mul_front(T, ψ_old.AC[i])                       # reproject old connecting tensor
        AC = integrate(AC_hamiltonian(i, ψ, H, ψ, envs), Ĉ, t, h, alg.integrator; imaginary_evolution)
        U₀ = _mul_front(T, ψ_old.AL[i])                      # old left isometry in the new frame
        if truncates
            U = _bug_augment_left(U₀, AC, alg.alg_orth)
            C = U' * AC
        else
            U, C = left_gauge(AC, alg.alg_orth)
        end
        T = U' * U₀                                          # transport (new ← old)
        ψ.AC[i] = (U, C)
    end
    AC = integrate(
        AC_hamiltonian(L, ψ, H, ψ, envs), _mul_front(T, ψ_old.AC[L]),
        t, h, alg.integrator; imaginary_evolution
    )
    imaginary_evolution && normalize!(AC)
    ψ.AC[L] = AC
    truncates && changebonds!(ψ, svdcut; normalize = imaginary_evolution)

    # sweep right to left (root = first site), the mirror
    ψ.AC[L]                       # gauge center to site L
    ψ_old = copy(ψ)
    T = isomorphism(scalartype(ψ), right_virtualspace(ψ_old, L) ← right_virtualspace(ψ_old, L))
    for i in L:-1:2
        Ĉ = ψ_old.AC[i] * T                                  # reproject old connecting tensor
        AC = integrate(AC_hamiltonian(i, ψ, H, ψ, envs), Ĉ, t + h, h, alg.integrator; imaginary_evolution)
        U₀ = ψ_old.AR[i] * T                                 # old right isometry in the new frame
        if truncates
            U = _bug_augment_right(U₀, AC, alg.alg_orth)
            C = _transpose_tail(AC) * _transpose_tail(U)'
        else
            C, U = right_gauge(AC, alg.alg_orth)
        end
        T = _transpose_tail(U₀) * _transpose_tail(U)'        # transport (old ← new)
        ψ.AC[i] = (C, U)
    end
    AC = integrate(
        AC_hamiltonian(1, ψ, H, ψ, envs), ψ_old.AC[1] * T,
        t + h, h, alg.integrator; imaginary_evolution
    )
    imaginary_evolution && normalize!(AC)
    ψ.AC[1] = AC
    truncates && changebonds!(ψ, svdcut; normalize = imaginary_evolution)

    return ψ, envs
end

# Augment the RIGHT bond for the left→right sweep: orthonormalize the stacked `[U₀ │ K₁]` (the old
# left-isometry `U₀` first, then the evolved candidate `K₁`) with a single QR. This keeps `U₀` as
# the leading per-sector block and appends only the directions of `K₁` orthogonal to it, while the
# QR caps each charge sector at `dim(Vl⊗P, c)` so no direction outside `Vl⊗P` is ever added.
function _bug_augment_left(U₀, K₁, alg_orth = Defaults.alg_orth())
    Û, _ = left_orth(catdomain(U₀, K₁); alg = alg_orth)   # Vl⊗P ← (Vr₀ ⊕ Vr_new), old-first
    return Û
end

# Mirror of `_bug_augment_left` for the right→left sweep, on the `_transpose_tail` form
# (`Vl ← P⊗Vr`, in which a right-isometry has orthonormal rows): a single LQ orthonormalizes the
# stacked rows `[U₀; K₁]`, keeping `U₀` as the leading block.
function _bug_augment_right(U₀, K₁, alg_orth = Defaults.alg_orth())
    stacked = catcodomain(_transpose_tail(U₀), _transpose_tail(K₁))  # (Vl₀ ⊕ Vl_K) ← P⊗Vr
    _, Û = right_orth(stacked; alg = alg_orth)                       # V̂ ← P⊗Vr, old-first
    return _transpose_front(Û)
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
