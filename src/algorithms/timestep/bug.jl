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

# Parallel BUG
# ------------

"""
$(TYPEDEF)

Single site MPS time-evolution algorithm based on the *parallel* Basis-Update & Galerkin (BUG)
integrator for tree tensor networks, specialized to the linear (`FiniteMPS`) tree.

Unlike the sequential [`BUG`](@ref), every local problem is solved from the **same frozen `t₀`
snapshot** (`ψ₀ = copy(ψ)` and its environments): there is no sweep and no sequential dependency
between the local integrations, so they are mutually independent and parallelizable. Rooting the
caterpillar at the last site, every local center `AC[i]` is evolved forward from `t₀`. A cheap
leaves→root assembly pass (the augmentation `A_τ` of Ceruti et al. 2024, Alg. 4) then augments
every bond with the new directions discovered by the frozen evolutions — old basis first,
`[U₀ │ Ũ₁]` — where the new directions at each bond are orthonormalized from the evolved center
*stacked with a first-order coupling block* on the new rows of the previous bond, so that
directions discovered deep in the chain propagate up toward the root. The interior tensors of the
augmented state are isometries (the amplitude and phase of the evolved centers is discarded with
the orthonormalization); the root tensor is the evolved center `AC[L]` augmented with the
projection of the full time derivative onto the new directions, so the amplitude and all
first-order updates enter exactly once, at the root. Finally an SVD sweep truncates the (at most
doubled) bonds back down.
Like [`BUG`](@ref), it advances every tensor *forward* in time (no backward substep), which suits
imaginary-time / dissipative evolution.

The integrator is **first-order** in time (the discarded "new–new" coupling corners are `O(dt²)`),
in contrast to the second-order symmetrized sequential [`BUG`](@ref). The truncation tolerance of
`trscheme` maps to the BUG tolerance `ϑ`; because the global error accumulates as `c·n·ϑ` over `n`
steps, scale `ϑ` with `dt` for a fixed target accuracy. Any truncating `trscheme` (e.g.
`truncerror`) makes the bond dimension grow and shrink to track the entanglement of the evolving
state; the default `notrunc()` restores the pre-step virtual spaces after every step (fixed-rank
parallel BUG).

Setting `maxiter_rejection > 0` enables **step rejection**: when a truncating step keeps the *full*
augmented (doubled) space on some bond — a single doubling was not enough to resolve `trscheme` —
the step is recomputed as two half-steps, so a smaller per-step normal component `η` fits within one
doubling per sub-step (cf. `rejection_check` of Ceruti et al. 2024). The local error estimator
`η = ‖C̃‖/dt` (the frozen derivative projected onto the new directions, their eq. 6) is logged at
`@debug` level each step.

!!! warning "Experimental"
    This integrator is **work in progress**. The API and behaviour may change. The step-rejection
    trigger is the rank-saturation criterion; the `η`-threshold trigger `dt·η > c·ϑ` and the
    second-order augmented-Galerkin variant of Ceruti et al. (2024) are not yet implemented.

!!! note
    Real-time evolution does not normalize the resulting state: neither the augmentation nor the
    truncation normalizes, so the state norm reflects the accumulated truncation error.
    Imaginary-time evolution renormalizes after every step, similar to a ground-state search.

## Fields

$(TYPEDFIELDS)

## References

* Ceruti, Kusch & Lubich, *A parallel rank-adaptive integrator for dynamical low-rank
  approximation*, SIAM J. Sci. Comput. **46** (2024).
* Ceruti, Kusch, Lubich & Sulz, *A parallel Basis Update and Galerkin integrator for tree tensor
  networks*, arXiv:2412.00858 (2024).
"""
struct ParallelBUG{A, O, T, S, F} <: Algorithm
    "algorithm used in the exponential solvers"
    integrator::A

    "tolerance for gauging algorithm"
    tolgauge::Float64

    "maximal amount of iterations for gauging algorithm"
    gaugemaxiter::Int

    "algorithm used to re-orthonormalize the basis after each local update"
    alg_orth::O

    "truncation scheme used to cut the augmented bonds back down"
    trscheme::T

    "algorithm used for the singular value decomposition"
    alg_svd::S

    "safety constant `c` in the step-rejection threshold `h·η > c·ϑ` (paper value ≈ 10)"
    c::Float64

    "maximum number of rejection recomputes per step (0 disables step rejection)"
    maxiter_rejection::Int

    "callback function applied after each iteration, of signature `finalize(iter, ψ, H, envs) -> ψ, envs`"
    finalize::F
end
function ParallelBUG(;
        integrator = Defaults.alg_expsolve(), tolgauge = Defaults.tolgauge,
        gaugemaxiter = Defaults.maxiter, alg_orth = Defaults.alg_orth(),
        trscheme = notrunc(), alg_svd = Defaults.alg_svd(),
        c = 10.0, maxiter_rejection = 0,
        finalize = Defaults._finalize
    )
    return ParallelBUG(
        integrator, tolgauge, gaugemaxiter, alg_orth, trscheme, alg_svd,
        c, maxiter_rejection, finalize
    )
end

function timestep!(
        ψ::AbstractFiniteMPS, H, t::Number, dt::Number, alg::ParallelBUG,
        envs::AbstractMPSEnvironments = environments(ψ, H, ψ);
        imaginary_evolution::Bool = false
    )
    L = length(ψ)
    if L == 1                                   # single site: a plain forward center step
        AC = integrate(
            AC_hamiltonian(1, ψ, H, ψ, envs), ψ.AC[1], t, dt, alg.integrator; imaginary_evolution
        )
        imaginary_evolution && normalize!(AC)
        ψ.AC[1] = AC
        return ψ, envs
    end

    truncates = !(alg.trscheme isa MatrixAlgebraKit.NoTruncation)
    # remember the pre-step virtual spaces for the fixed-rank (`notrunc`) restore
    Vs = [right_virtualspace(ψ, b) for b in 1:(L - 1)]

    ϕ, ηh = _pbug_assemble(ψ, H, t, dt, alg; imaginary_evolution)
    augVs = [right_virtualspace(ϕ, b) for b in 1:(L - 1)]   # the (doubled) augmented spaces
    _pbug_truncate!(ϕ, alg, Vs; normalize = imaginary_evolution)

    # step rejection (opt-in via `maxiter_rejection > 0`): if the truncation kept the *full* augmented
    # space on some bond (a single doubling was not enough to resolve `trscheme`), the step is
    # under-resolved (Ceruti et al. 2024; cf. `rejection_check.m`). Recompute it as two half-steps —
    # sub-stepping lowers the per-step normal component `η = ηh/dt`, so one doubling per sub-step
    # suffices. Bounded by `maxiter_rejection`, after which the saturated step is accepted.
    if truncates && alg.maxiter_rejection > 0
        saturated = any(1:(L - 1)) do b
            return right_virtualspace(ϕ, b) == augVs[b] && augVs[b] != Vs[b]
        end
        @debug "ParallelBUG step" ηh saturated
        if saturated
            alg′ = _pbug_with_rejections(alg, alg.maxiter_rejection - 1)
            timestep!(ψ, H, t, dt / 2, alg′; imaginary_evolution)
            timestep!(ψ, H, t + dt / 2, dt / 2, alg′; imaginary_evolution)
            return ψ, environments(ψ, H, ψ)
        end
    end

    # mutate `ψ` in place to become the assembled state (the generic loop reuses the object)
    _pbug_overwrite!(ψ, ϕ)
    return ψ, environments(ψ, H, ψ)
end

# rebuild a `ParallelBUG` with a reduced rejection budget (the struct is immutable)
function _pbug_with_rejections(alg::ParallelBUG, n::Int)
    return ParallelBUG(
        alg.integrator, alg.tolgauge, alg.gaugemaxiter, alg.alg_orth,
        alg.trscheme, alg.alg_svd, alg.c, n, alg.finalize
    )
end

# Assemble the augmented (pre-truncation) parallel-BUG state from a single frozen `t₀` snapshot,
# following Ceruti et al. 2024 (arXiv:2412.00858) Alg. 1-4 specialized to the caterpillar tree
# rooted at site `L`. Two phases:
#
# 1. Galerkin-evolve every local center `AC[i]` from the frozen snapshot (the K-steps of
#    Alg. 2/3, on the amplitude-weighted subtree objects `Y_τ⁰ = U_τ⁰S_τ⁰`). These `L` local
#    solves are mutually independent, which is the parallel-in-time structure of the integrator.
# 2. A leaves→root augmentation pass (Alg. 4): at bond `i` the new directions `Ũᵢ` are
#    orthonormalized against the (zero-padded) old isometry from the evolved center *stacked with
#    the first-order coupling block* `C̃ᵢ = dt′·⟨Ũᵢ₋₁|H|ψ₀⟩-derivative of AC[i]` on the new rows of
#    the previous bond — this reconciliation is what lets directions discovered deep in the chain
#    propagate to the root. Only the range of these stacks is kept (their amplitude and phase are
#    discarded with the R-factor), so the interior site tensors of the augmented state are the
#    isometries `[old │ Ũᵢ]`, and all first-order (and amplitude) content is carried by the root
#    tensor `[C̄_L(t₁); C̃_L]`, whose coupling row projects the full derivative of `ψ₀` onto the
#    new directions. The "new-new" corners are implicitly zero, which is what makes the integrator
#    first order.
function _pbug_assemble(ψ, H, t, dt, alg::ParallelBUG; imaginary_evolution::Bool = false)
    L = length(ψ)
    ψ.AC[L]                                     # gauge to the root
    ψ₀ = copy(ψ)
    envs₀ = environments(ψ₀, H, ψ₀)
    dt′ = imaginary_evolution ? -dt : -im * dt

    # Pre-populate the frozen environments serially: `leftenv`/`rightenv` on `FiniteEnvironments`
    # lazily *mutate* their cache (finite_envs.jl), so the parallel phase-1 solves below must only
    # ever read them. After this pass every `AC_hamiltonian(i)` reads a warm, read-only cache.
    _pbug_warmup_envs!(envs₀, L, ψ₀)

    # phase 1: frozen-snapshot Galerkin evolutions (independent local solves). The interior
    # K-steps evolve the amplitude-weighted centers `AC[i]` (the paper's `Y_τ⁰ = U_τ⁰S_τ⁰`); only
    # their range is kept, so the amplitude they carry is discarded in the orthonormalization. All
    # `L` solves read the read-only snapshot and are mutually independent — the parallel-in-time
    # structure; `tmap!` threads them according to `Defaults.scheduler[]` (as in `tdvp.jl`).
    scheduler = Defaults.scheduler[]
    Cevo = Vector{typeof(ψ₀.AC[1])}(undef, L)
    tmap!(Cevo, 1:L; scheduler) do i
        return integrate(
            AC_hamiltonian(i, ψ₀, H, ψ₀, envs₀), ψ₀.AC[i], t, dt, alg.integrator;
            imaginary_evolution
        )
    end
    C̄L = Cevo[L]

    # phase 2: leaves→root augmentation sweep, threading the mixed ⟨augmented|H|old⟩ environments.
    # Alongside, accumulate `ηh = max‖C̃ᵢ‖ = h·η`: each coupling block `C̃ᵢ` is the frozen derivative
    # projected onto the new directions (the normal component the old basis misses), so its norm is
    # the local error estimator of Ceruti-Kusch-Lubich (2024, eq. 6), specialized to the caterpillar.
    As = Vector{typeof(ψ₀.AL[1])}(undef, L)
    ηh = zero(real(scalartype(ψ₀)))
    GLmix = _pbug_mixedenv_init(H, envs₀, ψ₀)   # full augmented rows (trivial at the left edge)
    local GLnew                                 # new-direction rows only
    for i in 1:(L - 1)
        if i == 1
            C⁰, Ĉ¹ = ψ₀.AL[1], Cevo[1]
        else
            # first-order coupling block on the new rows of bond i-1. Applied with the two-arg
            # (time-dependent) form at the midpoint `t + dt/2`, matching the anchor `integrate`
            # freezes the coefficients at (integrators.jl); a `TimedOperator` has no one-arg apply.
            C̃ = scale(_pbug_coupling_hamiltonian(GLnew, H, i, envs₀, ψ₀)(ψ₀.AC[i], t + dt / 2), dt′)
            ηh = max(ηh, norm(C̃))
            C⁰ = _pbug_stack_child(ψ₀.AL[i], zerovector!(similar(C̃)))
            Ĉ¹ = _pbug_stack_child(Cevo[i], C̃)
        end
        Ũ, = _pbug_newdirs(C⁰, Ĉ¹, alg.alg_orth)
        As[i] = catdomain(C⁰, Ũ)
        GLnew = _pbug_mixedenv_step(GLmix, H, i, ψ₀.AL[i], Ũ)
        i == L - 1 && break                     # the full mixed environment is no longer needed
        GLmix = _pbug_mixedenv_step(GLmix, H, i, ψ₀.AL[i], As[i])
    end
    # root: the amplitude is carried once, by the evolved center and its coupling row
    C̃L = scale(_pbug_coupling_hamiltonian(GLnew, H, L, envs₀, ψ₀)(ψ₀.AC[L], t + dt / 2), dt′)
    ηh = max(ηh, norm(C̃L))
    As[L] = _pbug_stack_child(C̄L, C̃L)

    return FiniteMPS(As; overwrite = true, normalize = imaginary_evolution), ηh
end

# Warm the lazily-cached frozen environments serially so the parallel phase-1 solves only read
# them: `leftenv`/`rightenv` on `FiniteEnvironments` mutate their cache. A `LazySum` yields a
# `MultipleEnvironments`, so recurse into its per-summand `FiniteEnvironments`.
function _pbug_warmup_envs!(envs::FiniteEnvironments, L, ψ₀)
    for i in 1:L
        leftenv(envs, i, ψ₀)
        rightenv(envs, i, ψ₀)
    end
    return envs
end
function _pbug_warmup_envs!(envs::MultipleEnvironments, L, ψ₀)
    foreach(e -> _pbug_warmup_envs!(e, L, ψ₀), envs.envs)
    return envs
end

# Mixed ⟨augmented|H|old⟩ left environments: initial (trivial-bond) environment and one-site
# transfer step, with the same operator dispatch as the effective-Hamiltonian constructors.
_pbug_mixedenv_init(H, envs, ψ₀) = leftenv(envs, 1, ψ₀)
function _pbug_mixedenv_init(H::MultipliedOperator, envs, ψ₀)
    return _pbug_mixedenv_init(H.op, envs, ψ₀)
end
function _pbug_mixedenv_init(H::LazySum, envs::MultipleEnvironments, ψ₀)
    return map((o, e) -> _pbug_mixedenv_init(o, e, ψ₀), H.ops, envs.envs)
end

_pbug_mixedenv_step(GL, H, i, above, below) = GL * TransferMatrix(above, H[i], below)
function _pbug_mixedenv_step(GL, H::MultipliedOperator, i, above, below)
    return _pbug_mixedenv_step(GL, H.op, i, above, below)
end
function _pbug_mixedenv_step(GLs::Vector, H::LazySum, i, above, below)
    return map((gl, o) -> _pbug_mixedenv_step(gl, o, i, above, below), GLs, H.ops)
end

# One-site effective derivative with the mixed new-direction rows as left environment and the
# frozen old right environment: applied to the center `AC[i]` this yields the first-order coupling
# block `C̃ᵢ` (Alg. 4's `h·F(Y_τ⁰) ×ⱼ U⁰* ×ᵢ Ũ¹*`).
function _pbug_coupling_hamiltonian(GL, H, i, envs, ψ₀)
    return MPO_AC_Hamiltonian(GL, H[i], rightenv(envs, i, ψ₀))
end
function _pbug_coupling_hamiltonian(GL, H::MultipliedOperator, i, envs, ψ₀)
    return MultipliedOperator(_pbug_coupling_hamiltonian(GL, H.op, i, envs, ψ₀), H.f)
end
function _pbug_coupling_hamiltonian(GLs::Vector, H::LazySum, i, envs::MultipleEnvironments, ψ₀)
    Hs = map((gl, o, e) -> _pbug_coupling_hamiltonian(gl, o, i, e, ψ₀), GLs, H.ops, envs.envs)
    elT = Union{D, MultipliedOperator{D}} where {D <: DerivativeOperator}
    return LazySum{elT}(Hs)
end

# new bond directions: the component of the evolved (stacked) candidate orthogonal to the old
# basis, re-orthonormalized. Returns the isometry `Ũ` and the new bond space.
function _pbug_newdirs(AL, Cevo, alg_orth = Defaults.alg_orth())
    N = left_null(AL)
    g = N' * Cevo
    Q, _ = left_orth(g; alg = alg_orth)
    Ũ = N * Q
    return Ũ, domain(Q)
end

# stack two MPS tensors along the child (left-virtual) bond: `[top; bot]`, doubling that bond
_pbug_stack_child(top, bot) =
    _transpose_front(catcodomain(_transpose_tail(top), _transpose_tail(bot)))

# Cut the augmented bonds back down. A truncating `trscheme` selects rank-adaptivity; the default
# `notrunc()` restores the pre-step virtual space of every bond (fixed-rank parallel BUG).
# Environments self-heal lazily for the changed bonds.
function _pbug_truncate!(ϕ, alg::ParallelBUG, Vs; normalize::Bool = false)
    if !(alg.trscheme isa MatrixAlgebraKit.NoTruncation)
        changebonds!(ϕ, SvdCut(; trscheme = alg.trscheme, alg_svd = alg.alg_svd); normalize)
    else
        # per-bond fixed-rank restore (mirrors the `SvdCut` sweep with a per-bond `trscheme`)
        for i in (length(ϕ) - 1):-1:1
            U, S, Vᴴ = svd_trunc(ϕ.C[i]; trunc = truncspace(Vs[i]), alg = alg.alg_svd)
            ϕ.AC[i] = (ϕ.AL[i] * U, S)
            ϕ.AC[i + 1] = (S, _transpose_front(Vᴴ * _transpose_tail(ϕ.AR[i + 1])))
        end
        normalize && normalize!(ϕ)
    end
    return ϕ
end

# overwrite the internal representation of `ψ` with that of `ϕ` (same length, possibly new bonds)
function _pbug_overwrite!(ψ::FiniteMPS, ϕ::FiniteMPS)
    for f in (:ALs, :ARs, :ACs, :Cs)
        copyto!(getfield(ψ, f), getfield(ϕ, f))
    end
    return ψ
end

# copying version, shared by both BUG integrators
function timestep(
        ψ::AbstractFiniteMPS, H, time::Number, timestep::Number,
        alg::Union{BUG, ParallelBUG}, envs::AbstractMPSEnvironments...;
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
