"""
$(TYPEDEF)

Single site MPS time-evolution algorithm based on the Basis-Update & Galerkin (BUG) integrator,
an unconventional robust integrator for dynamical low-rank approximation.

Unlike [`TDVP`](@ref), BUG advances both the basis (K-step) and the core (Galerkin C-step) tensors
*forward* in time, with no backward-in-time substep. This makes it a natural choice for
imaginary-time / dissipative evolution, where the backward core step of TDVP can become unstable.

Passing a truncating `trscheme` (anything other than `notrunc()`) switches on **rank-adaptivity**:
each half-sweep augments every bond with the new directions discovered by the evolved connecting
tensor (old basis first, `[U₀ │ K₁]`) and then truncates back down to the tolerance of `trscheme`.
The bond dimension grows and shrinks automatically to track the entanglement; `notrunc()` recovers
the fixed-rank integrator.

!!! note
    Real-time evolution does not normalize the resulting state, so the state norm reflects the
    accumulated truncation error. Imaginary-time evolution renormalizes after every half-sweep.

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
    # symmetric 2nd-order: a dt/2 left→right half-sweep composed with its dt/2 mirror
    L = length(ψ)
    h = dt / 2
    truncates = !(alg.trscheme isa MatrixAlgebraKit.NoTruncation)
    svdcut = SvdCut(; trscheme = alg.trscheme, alg_svd = alg.alg_svd)

    # left→right half-sweep (root = last site)
    ψ.AC[1]                       # gauge center to site 1
    ψ_old = copy(ψ)               # frozen bases / reprojection inputs
    T = isomorphism(scalartype(ψ), left_virtualspace(ψ_old, 1) ← left_virtualspace(ψ_old, 1))
    for i in 1:(L - 1)
        Ĉ = _mul_front(T, ψ_old.AC[i])                       # reproject old connecting tensor
        AC = integrate(AC_hamiltonian(i, ψ, H, ψ, envs), Ĉ, t, h, alg.integrator; imaginary_evolution)
        U₀ = _mul_front(T, ψ_old.AL[i])                      # old left isometry in the new frame
        if truncates
            U = _bug_augment_left(U₀, AC, alg.alg_orth)      # old-first augment; cut deferred to changebonds!
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

    # right→left half-sweep (root = first site), the mirror
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

# augment the RIGHT bond (left→right sweep): orthonormalize the stacked `[U₀ │ K₁]` (old isometry
# first) so `U₀` stays the leading per-sector block and only the new directions of `K₁` are appended.
function _bug_augment_left(U₀, K₁, alg_orth = Defaults.alg_orth())
    Û, _ = left_orth(catdomain(U₀, K₁); alg = alg_orth)
    return Û
end

# mirror of `_bug_augment_left` for the right→left sweep, on the `_transpose_tail` form (right-isometry
# with orthonormal rows): an LQ orthonormalizes the stacked rows `[U₀; K₁]`, keeping `U₀` leading.
function _bug_augment_right(U₀, K₁, alg_orth = Defaults.alg_orth())
    stacked = catcodomain(_transpose_tail(U₀), _transpose_tail(K₁))
    _, Û = right_orth(stacked; alg = alg_orth)
    return _transpose_front(Û)
end

# Parallel BUG
# ------------

# Shared supertype for the parallel-BUG integrators (`ParallelBUG` first order, `ParallelBUG2` second
# order); the assembly / truncation / rejection helpers below dispatch on it.
abstract type AbstractParallelBUG <: Algorithm end

"""
$(TYPEDEF)

Single site MPS time-evolution algorithm based on the *parallel* Basis-Update & Galerkin (BUG)
integrator for tree tensor networks, specialized to the linear (`FiniteMPS`) tree.

Unlike the sequential [`BUG`](@ref), every local center `AC[i]` is evolved forward from the **same
frozen `t₀` snapshot**: there is no sweep and no sequential dependency, so the local integrations are
mutually independent and parallelizable. A cheap leaves→root pass then augments every bond with the
new directions the frozen evolutions discovered (old basis first, `[U₀ │ Ũ₁]`), and a final SVD sweep
truncates the (at most doubled) bonds back down. Like [`BUG`](@ref) it advances every tensor *forward*
in time (no backward substep), which suits imaginary-time / dissipative evolution.

The integrator is **first-order** in time. Any truncating `trscheme` makes the bond dimension grow
and shrink to track the entanglement; the default `notrunc()` restores the pre-step virtual spaces
(fixed-rank parallel BUG). Setting `maxiter_rejection > 0` enables **step rejection**: a step that
saturates the doubling on some bond is recomputed as two half-steps (cf. Ceruti et al. 2024).

!!! warning "Experimental"
    This integrator is **work in progress**; the API and behaviour may change.

!!! note
    Real-time evolution does not normalize the resulting state, so the state norm reflects the
    accumulated truncation error. Imaginary-time evolution renormalizes after every step.

## Fields

$(TYPEDFIELDS)

## References

* Ceruti, Kusch & Lubich, *A parallel rank-adaptive integrator for dynamical low-rank
  approximation*, SIAM J. Sci. Comput. **46** (2024).
* Ceruti, Kusch, Lubich & Sulz, *A parallel Basis Update and Galerkin integrator for tree tensor
  networks*, arXiv:2412.00858 (2024).
"""
struct ParallelBUG{A, O, T, S, F} <: AbstractParallelBUG
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

"""
$(TYPEDEF)

Single site MPS time-evolution algorithm based on the *second-order* variant of the *parallel*
Basis-Update & Galerkin (BUG) integrator for tree tensor networks (Kusch 2024, "Variant 2"),
specialized to the linear (`FiniteMPS`) tree. It is to [`ParallelBUG`](@ref) what [`TDVP2`](@ref) is
to [`TDVP`](@ref): a separate, more accurate integrator sharing the same interface.

The genuine second order comes from **pre-augmenting every bond basis with one `H·ψ₀` application
before evolving**: each bond isometry is enlarged to rank `2r` with the directions opened by a single
effective-Hamiltonian application to the frozen center, the Galerkin K-steps are evolved on those
enriched environments, and the `O(dt²)` content the first-order scheme discards is transported to the
root, keeping the "new–new" corner zero (so the local error is `O(dt³)` rather than `O(dt²)`).
Everything else matches [`ParallelBUG`](@ref): one frozen `t₀` snapshot, mutually independent local
solves, amplitude carried once at the root, `notrunc()` for the fixed-rank variant, and forward-only
evolution (no backward substep).

!!! warning "Experimental"
    This integrator is **work in progress**; the API and behaviour may change. Its single-step error
    scales as `O(dt³)` (log–log slope `≈ 3`) versus the first-order `O(dt²)`. Only plain Hamiltonians
    are supported (no `LazySum` / time-dependent operators).

!!! note
    Real-time evolution does not normalize the resulting state, so the state norm reflects the
    accumulated truncation error. Imaginary-time evolution renormalizes after every step.

## Fields

$(TYPEDFIELDS)

## References

* Kusch, *Second-order robust parallel integrators for dynamical low-rank approximation*,
  arXiv:2403.02834 (2024).
* Ceruti, Kusch, Lubich & Sulz, *A parallel Basis Update and Galerkin integrator for tree tensor
  networks*, arXiv:2412.00858 (2024).
"""
struct ParallelBUG2{A, O, T, S, F} <: AbstractParallelBUG
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
function ParallelBUG2(;
        integrator = Defaults.alg_expsolve(), tolgauge = Defaults.tolgauge,
        gaugemaxiter = Defaults.maxiter, alg_orth = Defaults.alg_orth(),
        trscheme = notrunc(), alg_svd = Defaults.alg_svd(),
        c = 10.0, maxiter_rejection = 0,
        finalize = Defaults._finalize
    )
    return ParallelBUG2(
        integrator, tolgauge, gaugemaxiter, alg_orth, trscheme, alg_svd,
        c, maxiter_rejection, finalize
    )
end

function timestep!(
        ψ::AbstractFiniteMPS, H, t::Number, dt::Number, alg::AbstractParallelBUG,
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
    Vs = [right_virtualspace(ψ, b) for b in 1:(L - 1)]   # pre-step spaces, for the fixed-rank restore

    ϕ, ηh = _pbug_assemble(ψ, H, t, dt, alg; imaginary_evolution)
    augVs = [right_virtualspace(ϕ, b) for b in 1:(L - 1)]   # the (doubled) augmented spaces
    _pbug_truncate!(ϕ, alg, Vs; normalize = imaginary_evolution)

    # step rejection (opt-in): a bond that kept its full augmented space was under-resolved by a
    # single doubling; recompute as two half-steps so one doubling per sub-step suffices.
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

    # overwrite `ψ` in place with the assembled state (adopt `ϕ`'s tensors; identity-keyed envs self-heal)
    for f in (:ALs, :ARs, :ACs, :Cs)
        copyto!(getfield(ψ, f), getfield(ϕ, f))
    end
    return ψ, environments(ψ, H, ψ)
end

# rebuild a parallel-BUG algorithm with a reduced rejection budget (the structs are immutable)
function _pbug_with_rejections(alg::AbstractParallelBUG, n::Int)
    return typeof(alg)(
        alg.integrator, alg.tolgauge, alg.gaugemaxiter, alg.alg_orth,
        alg.trscheme, alg.alg_svd, alg.c, n, alg.finalize
    )
end

# ---- second-order (Variant 2) assembly ---------------------------------------------------------

# old-first LEFT enrichment `Û0[1..L-1]` (rank `2r`) plus the enriched left-environment chain `GLhat`
# (`⟨Û0|H|Û0⟩`): enlarge each `AL⁰[i]` with the range of the frozen derivative image `W[i]=(H·ψ₀)ᵢ`,
# stacked leaves→root with the mixed `⟨new|H|old⟩` coupling so directions opened deep in the chain
# reach the root. Envs are folded by explicit transfer (no `FiniteMPS` round-trip, so the zero-weight
# enriched directions do not collapse under canonicalization).
function _pbug2_left_enrich(ψ₀, H, envs₀, W, alg_orth)
    L = length(ψ₀)
    Û0 = Vector{typeof(ψ₀.AL[1])}(undef, L - 1)
    GLhat = Vector{Any}(undef, L)
    GLhat[1] = leftenv(envs₀, 1, ψ₀)
    GLmix = leftenv(envs₀, 1, ψ₀)               # mixed ⟨new|H|old⟩ chain
    local GLnew
    for i in 1:(L - 1)
        if i == 1
            C⁰, Ĉ = ψ₀.AL[1], W[1]
        else
            C̃ = MPO_AC_Hamiltonian(GLnew, H[i], rightenv(envs₀, i, ψ₀))(ψ₀.AC[i])
            C⁰ = _pbug_stack_child(ψ₀.AL[i], zerovector!(similar(C̃)))
            Ĉ = _pbug_stack_child(W[i], C̃)
        end
        Ũ, = _pbug_newdirs(C⁰, Ĉ, alg_orth)
        Û0[i] = catdomain(C⁰, Ũ)
        GLhat[i + 1] = GLhat[i] * TransferMatrix(Û0[i], H[i], Û0[i])
        GLnew = GLmix * TransferMatrix(ψ₀.AL[i], H[i], Ũ)      # ket=old, bra=new
        i == L - 1 && break
        GLmix = GLmix * TransferMatrix(ψ₀.AL[i], H[i], Û0[i])  # ket=old, bra=enriched
    end
    return Û0, GLhat
end

# old-first RIGHT enrichment `V̂0[2..L]` (rank `2r`) plus the enriched right-environment chain `GRhat`,
# the mirror of `_pbug2_left_enrich`. The interior K-step freezes this enriched right basis (freezing
# the old right basis instead only yields local slope 2).
function _pbug2_right_enrich(ψ₀, H, envs₀, W, alg_orth)
    L = length(ψ₀)
    V̂0 = Vector{typeof(ψ₀.AR[1])}(undef, L)
    GRhat = Vector{Any}(undef, L)
    GRhat[L] = rightenv(envs₀, L, ψ₀)
    GRmix = rightenv(envs₀, L, ψ₀)
    local GRnew
    for i in L:-1:2
        if i == L
            C⁰, Ĉ = ψ₀.AR[L], W[L]
        else
            C̃ = MPO_AC_Hamiltonian(leftenv(envs₀, i, ψ₀), H[i], GRnew)(ψ₀.AC[i])
            C⁰ = catdomain(ψ₀.AR[i], zerovector!(similar(C̃)))
            Ĉ = catdomain(W[i], C̃)
        end
        # new left-bond directions `Ĉ` opens beyond the right-isometry `C⁰` (in `_transpose_tail` form)
        N = right_null!(_transpose_tail(C⁰; copy = true))
        _, Q = right_orth(_transpose_tail(Ĉ) * N'; alg = alg_orth)
        V̂ = Q * N
        V̂0[i] = _transpose_front(catcodomain(_transpose_tail(C⁰), V̂))
        GRhat[i - 1] = TransferMatrix(V̂0[i], H[i], V̂0[i]) * GRhat[i]
        GRnew = TransferMatrix(ψ₀.AR[i], H[i], _transpose_front(V̂)) * GRmix  # ket=old, bra=new
        i == 2 && break
        GRmix = TransferMatrix(ψ₀.AR[i], H[i], V̂0[i]) * GRmix                # ket=old, bra=enriched
    end
    return V̂0, GRhat
end

# Genuine second-order parallel-BUG assembly (Kusch 2024, Variant 2 / `4r`) rooted at site `L`, from a
# single frozen `t₀` snapshot: (1) pre-augment the bond bases to rank `2r` with one `H·ψ₀` application
# (`Û0`/`V̂0`) and fold the enriched envs by explicit transfer; (2) K-step every center on the enriched
# envs, freezing `V̂0`; (3) a leaves→root sweep builds the `4r` isometries `[Û0 | Ũ2]` and transports
# the evolved-amplitude coupling `R = Ũ2ᵀĈ` through the frozen `V̂0`; (4) the root stacks the `2r`
# Galerkin `Kevo[L]` with the transported coupling, keeping the "new–new" corner zero (local `O(dt³)`).
function _pbug_assemble(ψ, H, t, dt, alg::ParallelBUG2; imaginary_evolution::Bool = false)
    L = length(ψ)
    ψ.AC[L]                                     # gauge to the root
    ψ₀ = copy(ψ)
    envs₀ = environments(ψ₀, H, ψ₀)
    _pbug_warmup_envs!(envs₀, L, ψ₀)
    scheduler = Defaults.scheduler[]

    # frozen `H·ψ₀` images (independent, threaded)
    W = Vector{typeof(ψ₀.AC[1])}(undef, L)
    tmap!(W, 1:L; scheduler) do i
        return AC_hamiltonian(i, ψ₀, H, ψ₀, envs₀)(ψ₀.AC[i])
    end

    Û0, GLhat = _pbug2_left_enrich(ψ₀, H, envs₀, W, alg.alg_orth)
    V̂0, GRhat = _pbug2_right_enrich(ψ₀, H, envs₀, W, alg.alg_orth)

    GRof(i) = i < L ? GRhat[i] : rightenv(envs₀, L, ψ₀)
    enrL(i) = i == 1 ? left_virtualspace(ψ₀, 1) :
        (i <= L - 1 ? space(Û0[i], 1) : only(domain(Û0[L - 1])))
    enrR(i) = i == L ? right_virtualspace(ψ₀, L) : space(V̂0[i + 1], 1)
    # embed the old center into the enriched bond spaces with zero weight in the new directions
    embed(i) = absorb!(
        zerovector!(similar(ψ₀.AC[i], (enrL(i) ⊗ physicalspace(ψ₀, i)) ← enrR(i))), ψ₀.AC[i]
    )

    # K/S steps on the enriched environments (freeze `V̂0`); independent ⇒ threaded
    Kevo = Vector{typeof(ψ₀.AC[1])}(undef, L)
    tmap!(Kevo, 1:L; scheduler) do i
        return integrate(
            MPO_AC_Hamiltonian(GLhat[i], H[i], GRof(i)), embed(i), t, dt, alg.integrator;
            imaginary_evolution
        )
    end

    # leaves→root assembly with the transported evolved-amplitude coupling
    As = Vector{typeof(ψ₀.AL[1])}(undef, L)
    ηh = zero(real(scalartype(ψ₀)))
    local R
    for i in 1:(L - 1)
        if i == 1
            C⁰, Ĉ¹ = Û0[1], Kevo[1]
        else
            C̃ = _transpose_front(R * _transpose_tail(V̂0[i]))   # transport R through frozen V̂0[i]
            ηh = max(ηh, norm(C̃))
            zc0 = zerovector!(similar(C̃, codomain(C̃) ← domain(Û0[i])))
            C⁰ = _pbug_stack_child(Û0[i], zc0)
            Ĉ¹ = _pbug_stack_child(Kevo[i], C̃)
        end
        Ũ2, = _pbug_newdirs(C⁰, Ĉ¹, alg.alg_orth)
        As[i] = catdomain(C⁰, Ũ2)
        R = Ũ2' * Ĉ¹                                            # evolved amplitude in the new dirs
    end
    # root: `2r` Galerkin `Kevo[L]` (Û0 rows) stacked with the transported coupling (Ũ2 rows); the
    # "new–new" corner stays zero (right bond trivial at the root), so the local error is `O(dt³)`.
    C̃L = _transpose_front(R * _transpose_tail(V̂0[L]))
    ηh = max(ηh, norm(C̃L))
    As[L] = _pbug_stack_child(Kevo[L], C̃L)

    return FiniteMPS(As; overwrite = true, normalize = imaginary_evolution), ηh
end

# First-order parallel-BUG assembly (Ceruti et al. 2024, arXiv:2412.00858, Alg. 1-4) rooted at site
# `L`. Phase 1: Galerkin-evolve every center `AC[i]` from the frozen snapshot (independent ⇒ parallel).
# Phase 2: a leaves→root sweep orthonormalizes each evolved center *stacked with a first-order coupling
# block* `C̃ᵢ` on the previous bond's new rows (`[old │ Ũᵢ]`), propagating deep new directions to the
# root. Interior tensors are pure isometries (amplitude/phase discarded); all first-order + amplitude
# content enters once, at the root `[C̄_L; C̃_L]`. The zero "new–new" corners make it first order.
function _pbug_assemble(ψ, H, t, dt, alg::ParallelBUG; imaginary_evolution::Bool = false)
    L = length(ψ)
    ψ.AC[L]                                     # gauge to the root
    ψ₀ = copy(ψ)
    envs₀ = environments(ψ₀, H, ψ₀)
    dt′ = imaginary_evolution ? -dt : -im * dt

    # warm the lazily-cached envs serially so the threaded phase-1 solves below only read them
    _pbug_warmup_envs!(envs₀, L, ψ₀)

    # phase 1: frozen-snapshot Galerkin evolutions (independent local solves, threaded)
    scheduler = Defaults.scheduler[]
    Cevo = Vector{typeof(ψ₀.AC[1])}(undef, L)
    tmap!(Cevo, 1:L; scheduler) do i
        return integrate(
            AC_hamiltonian(i, ψ₀, H, ψ₀, envs₀), ψ₀.AC[i], t, dt, alg.integrator;
            imaginary_evolution
        )
    end
    C̄L = Cevo[L]

    # phase 2: leaves→root augmentation, threading the mixed ⟨augmented|H|old⟩ envs. `ηh = max‖C̃ᵢ‖` is
    # the local error estimator (Ceruti-Kusch-Lubich 2024, eq. 6): the frozen derivative projected onto
    # the new directions the old basis misses.
    As = Vector{typeof(ψ₀.AL[1])}(undef, L)
    ηh = zero(real(scalartype(ψ₀)))
    GLmix = _pbug_mixedenv_init(H, envs₀, ψ₀)
    local GLnew
    for i in 1:(L - 1)
        if i == 1
            C⁰, Ĉ¹ = ψ₀.AL[1], Cevo[1]
        else
            # first-order coupling block on the new rows of bond i-1 (two-arg apply at midpoint
            # `t + dt/2`, matching what `integrate` freezes; a `TimedOperator` has no one-arg apply)
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

# warm the lazily-cached frozen environments serially (leftenv/rightenv on `FiniteEnvironments` mutate
# their cache), so the parallel phase-1 solves only read them. A `LazySum` yields a
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

# mixed ⟨augmented|H|old⟩ left environments: initial (trivial-bond) env and one-site transfer step,
# dispatching through `MultipliedOperator`/`LazySum` like the effective-Hamiltonian constructors.
_pbug_mixedenv_init(H, envs, ψ₀) = leftenv(envs, 1, ψ₀)
_pbug_mixedenv_init(H::MultipliedOperator, envs, ψ₀) = _pbug_mixedenv_init(H.op, envs, ψ₀)
function _pbug_mixedenv_init(H::LazySum, envs::MultipleEnvironments, ψ₀)
    return map((o, e) -> _pbug_mixedenv_init(o, e, ψ₀), H.ops, envs.envs)
end

_pbug_mixedenv_step(GL, H, i, above, below) = GL * TransferMatrix(above, H[i], below)
_pbug_mixedenv_step(GL, H::MultipliedOperator, i, above, below) =
    _pbug_mixedenv_step(GL, H.op, i, above, below)
function _pbug_mixedenv_step(GLs::Vector, H::LazySum, i, above, below)
    return map((gl, o) -> _pbug_mixedenv_step(gl, o, i, above, below), GLs, H.ops)
end

# one-site effective derivative with the mixed new-direction rows as left env and the frozen old right
# env: applied to the center `AC[i]` this yields the first-order coupling block `C̃ᵢ`.
_pbug_coupling_hamiltonian(GL, H, i, envs, ψ₀) = MPO_AC_Hamiltonian(GL, H[i], rightenv(envs, i, ψ₀))
function _pbug_coupling_hamiltonian(GL, H::MultipliedOperator, i, envs, ψ₀)
    return MultipliedOperator(_pbug_coupling_hamiltonian(GL, H.op, i, envs, ψ₀), H.f)
end
function _pbug_coupling_hamiltonian(GLs::Vector, H::LazySum, i, envs::MultipleEnvironments, ψ₀)
    Hs = map((gl, o, e) -> _pbug_coupling_hamiltonian(gl, o, i, e, ψ₀), GLs, H.ops, envs.envs)
    elT = Union{D, MultipliedOperator{D}} where {D <: DerivativeOperator}
    return LazySum{elT}(Hs)
end

# new bond directions: the component of the evolved (stacked) candidate orthogonal to the old basis,
# re-orthonormalized. NOTE: keep `left_null` here — swapping it for `project_complement!`+QR adds
# completion columns outside the old-basis complement and breaks two-site exactness / first order.
function _pbug_newdirs(AL, Cevo, alg_orth = Defaults.alg_orth())
    N = left_null(AL)
    g = N' * Cevo
    Q, _ = left_orth(g; alg = alg_orth)
    return N * Q, domain(Q)
end

# stack two MPS tensors along the child (left-virtual) bond: `[top; bot]`, doubling that bond
_pbug_stack_child(top, bot) =
    _transpose_front(catcodomain(_transpose_tail(top), _transpose_tail(bot)))

# cut the augmented bonds back down: a truncating `trscheme` selects rank-adaptivity, `notrunc()`
# restores the pre-step virtual space of every bond (fixed-rank). Environments self-heal lazily.
function _pbug_truncate!(ϕ, alg::AbstractParallelBUG, Vs; normalize::Bool = false)
    if !(alg.trscheme isa MatrixAlgebraKit.NoTruncation)
        changebonds!(ϕ, SvdCut(; trscheme = alg.trscheme, alg_svd = alg.alg_svd); normalize)
    else
        for i in (length(ϕ) - 1):-1:1
            U, S, Vᴴ = svd_trunc(ϕ.C[i]; trunc = truncspace(Vs[i]), alg = alg.alg_svd)
            ϕ.AC[i] = (ϕ.AL[i] * U, S)
            ϕ.AC[i + 1] = (S, _transpose_front(Vᴴ * _transpose_tail(ϕ.AR[i + 1])))
        end
        normalize && normalize!(ϕ)
    end
    return ϕ
end

# copying version, shared by both BUG integrators
function timestep(
        ψ::AbstractFiniteMPS, H, time::Number, timestep::Number,
        alg::Union{BUG, ParallelBUG, ParallelBUG2}, envs::AbstractMPSEnvironments...;
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
