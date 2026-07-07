"""
$(TYPEDEF)

Single site MPS time-evolution algorithm based on the Basis-Update & Galerkin (BUG) integrator,
an unconventional robust integrator for dynamical low-rank approximation.

Unlike [`TDVP`](@ref), BUG advances both the basis-carrying (K-step) and the core (Galerkin C-step)
tensors *forward* in time and never inverts the bond tensor. In particular it has no backward-in-time
substep, which makes it a natural choice for imaginary-time / dissipative evolution where the
backward core step of projector-splitting integrators can become unstable.

Passing a truncating `trscheme` (anything other than the default `notrunc()`) switches on
**rank-adaptivity**: each half-sweep augments every bond with the new directions discovered by the
evolved connecting tensor (keeping the old basis as the leading block, `[U₀ │ K₁]`) and then
truncates the enlarged bonds back down to the tolerance of `trscheme` by an optimal SVD sweep. The
bond dimension therefore grows and shrinks automatically to track the entanglement of the evolving
state. The default `notrunc()` recovers the fixed-rank integrator exactly.

!!! note
    Real-time evolution does not renormalize: neither the augmentation nor the truncation
    renormalizes, so the state norm reflects the accumulated truncation error. Imaginary-time
    evolution renormalizes after every half-sweep, like a ground-state search.

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
    # Symmetric (2nd-order) fixed-rank BUG. Following the rooted TTN recursion (source of
    # truth for the MPS caterpillar), one *first-order* BUG sweep evolves every site tensor
    # exactly once: a K-step (basis update) at the leaf boundary and a forward Galerkin
    # AC-step at every internal node, where the *un-evolved* old connecting tensor is first
    # reprojected onto the already-updated child bases. A single symmetrized time step is a
    # left→right half-sweep with step `dt/2` composed with its mirror right→left half-sweep,
    # which lifts the sequential first-order sweep to second order.
    #
    # This deliberately does *not* mirror TDVP's per-site "evolve AC / split / evolve the
    # split-off core backward" substep, nor a naive per-site "K-step + separate forward
    # Galerkin C-step": the latter re-evolves the moving core once per bond and is only
    # first-order-consistent to the wrong ODE (the state converges to the wrong direction as
    # `dt → 0`). BUG has no backward substep, so imaginary-time evolution stays stable; the
    # state is renormalized after each half-sweep when `imaginary_evolution = true`.
    if length(ψ) == 1
        Hac = AC_hamiltonian(1, ψ, H, ψ, envs)
        ψ.AC[1] = integrate(Hac, ψ.AC[1], t, dt, alg.integrator; imaginary_evolution)
        imaginary_evolution && normalize!(ψ)
        return ψ, envs
    end

    h = dt / 2
    if _bug_truncates(alg)
        # Rank-adaptive BUG: each half-sweep *augments* every bond with the new directions
        # discovered by the (Galerkin-)evolved connecting tensor, evolving the core on the
        # enlarged (old-first) basis, and is followed by an optimal SVD truncation back down to
        # the `trscheme` tolerance (Ceruti–Lubich–Walach 2022). Real-time evolution does not
        # renormalize, so the truncation error is reflected in the norm; imaginary-time
        # evolution renormalizes each half-sweep.
        _bug_sweep_right_adaptive!(ψ, H, t, h, alg, envs; imaginary_evolution)       # left → right
        _bug_truncate!(ψ, alg; normalize = imaginary_evolution)
        _bug_sweep_left_adaptive!(ψ, H, t + h, h, alg, envs; imaginary_evolution)    # right → left
        _bug_truncate!(ψ, alg; normalize = imaginary_evolution)
    else
        _bug_sweep_right!(ψ, H, t, h, alg, envs; imaginary_evolution)       # left → right
        _bug_sweep_left!(ψ, H, t + h, h, alg, envs; imaginary_evolution)    # right → left
    end
    return ψ, envs
end

# Rank-adaptivity is enabled by any truncating `trscheme` (the default `notrunc()` selects the
# byte-for-byte fixed-rank path). Mirrors the `_truncates` gate used by `TDVP`/`DMRG`.
_bug_truncates(alg::BUG) = !(alg.trscheme isa MatrixAlgebraKit.NoTruncation)

# Truncate every internal bond back down to the `trscheme` tolerance with an optimal
# (canonical-form) SVD sweep. This is the "discard the singular-value tail whose Frobenius norm
# ≤ ϑ" step of the rank-adaptive integrator, applied to the exact augmented-Galerkin state
# produced by a half-sweep. The effective environments passed to the next half-sweep are keyed on
# tensor identity, so they recompute lazily for the changed bonds — no explicit refresh needed.
function _bug_truncate!(ψ, alg::BUG; normalize::Bool = false)
    changebonds!(ψ, SvdCut(; trscheme = alg.trscheme, alg_svd = alg.alg_svd); normalize)
    return ψ
end

# Transport of the old→new basis overlap across one bond.
# `transport_right`: given the overlap `T` on bond `i` (mapping the new child bond to the old
# child bond, `old ← new`) and the old/new right-isometries at site `i`, returns the overlap on
# bond `i-1`. `transport_left` is the mirror (`new ← old`) for the left-to-right sweep.
function _bug_transport_right(AR_old, AR_new, T)
    @plansor Tnew[-1; -2] := AR_old[-1 1; 2] * T[2; 3] * conj(AR_new[-2 1; 3])
    return Tnew
end
function _bug_transport_left(AL_old, AL_new, T)
    @plansor Tnew[-1; -2] := conj(AL_new[1 2; -1]) * T[1; 3] * AL_old[3 2; -2]
    return Tnew
end

# Basis augmentation (Stage 2 building block; NOT yet wired into `timestep!`).
#
# `_bug_augment_left` augments the RIGHT virtual bond of a site for the left→right sweep.
# Given the OLD left-isometry `U₀` (an MPS tensor `Vl ⊗ P ← Vr₀`, i.e. `AL_old`) and the evolved
# single-site candidate `K₁` (same leg structure `Vl ⊗ P ← Vr_K`, the K-step / Galerkin output),
# it builds an augmented left-isometry `Û` (`Vl ⊗ P ← V̂`) whose column space contains both
# `range(U₀)` and `range(K₁)`, keeping the OLD basis as the leading per-sector block
# (`V̂ = Vr₀ ⊕ Vr_new`). This is the "old-basis-first" augmentation `[U₀ │ Ũ₁]` of the
# rank-adaptive BUG papers, which makes the reprojection `Ŝ₀ = Û* Y₀ = Y₀` exact.
#
# The appended directions come from the *single-site* candidate `K₁` (not a two-site
# `AC2_projection` as in `OptimalExpand`): take the component of `K₁` in the orthogonal complement
# of `U₀` (`left_null`), orthonormalize its column space, and `catdomain` it after `U₀`. This does
# NOT truncate — the appended block has the full rank of that complement (so `dim(V̂) ≤ 2·dim(Vr₀)`).
#
# Returns `(Û, M)`:
#   * `Û` — the augmented left-isometry (`Û' Û = 𝟙`, `Vl ⊗ P ← V̂`),
#   * `M = Û' * U₀` — the old bond's coordinates in the augmented basis (`V̂ ← Vr₀`), equal to
#     `[𝟙; 0]` per sector, ready to embed the old transport/core into the enlarged bond.
function _bug_augment_left(U₀, K₁, alg_orth = Defaults.alg_orth())
    N = left_null(U₀)                       # Vl⊗P ← Vc, orthonormal complement of range(U₀)
    g = N' * K₁                             # Vc ← Vr_K, the part of K₁ orthogonal to U₀
    Q, _ = left_orth(g; alg = alg_orth)     # Vc ← Vr_new, orthonormal new directions
    Ũ₁ = N * Q                              # Vl⊗P ← Vr_new
    Û = catdomain(U₀, Ũ₁)                   # Vl⊗P ← (Vr₀ ⊕ Vr_new), old-first
    M = Û' * U₀                             # V̂ ← Vr₀, = [𝟙; 0] per sector
    return Û, M
end

# `_bug_augment_right` is the mirror for the right→left sweep: it augments the LEFT virtual bond,
# working on the `_transpose_tail` form (`Vl ← P ⊗ Vr`, in which a right-isometry has orthonormal
# rows) and using `right_null!`/`catcodomain`, exactly as `changebond!(:left)` and `right_gauge` do.
# Given the OLD right-isometry `U₀` (`Vl₀ ⊗ P ← Vr`, i.e. `AR_old`) and the evolved candidate `K₁`
# (`Vl_K ⊗ P ← Vr`), it returns an augmented right-isometry `Û` (`V̂ ⊗ P ← Vr`) whose left bond is
# `V̂ = Vl₀ ⊕ Vl_new` (old-first) with row space containing both `U₀` and `K₁`.
#
# Returns `(Û, M)`:
#   * `Û` — the augmented right-isometry (`Û` is right-canonical, its tail satisfies `û û' = 𝟙`),
#   * `M = û * u₀'` (with `û = _transpose_tail(Û)`, `u₀ = _transpose_tail(U₀)`) — the old bond's
#     coordinates in the augmented left bond (`V̂ ← Vl₀`), equal to `[𝟙; 0]` per sector.
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

# Left→right half-sweep (root = last site, leaf = site 1): center ends at the last site.
function _bug_sweep_right!(ψ, H, t, τ, alg, envs; imaginary_evolution::Bool = false)
    L = length(ψ)
    ψ.AC[1]                                      # gauge center to site 1 (materialize AR[2..L])
    ψ_old = copy(ψ)                              # frozen "t₀" state for K-step inputs / old bases
    envs_old = environments(ψ_old, H, ψ_old)
    # `ψ` is mutated in place: its left bases become new (as installed), its right bases stay
    # `ψ_old`'s, so `envs` yields new-left / old-right effective Hamiltonians for the Galerkin.

    # leaf (site 1): K-step, keep only the new left isometry
    AC1 = integrate(AC_hamiltonian(1, ψ_old, H, ψ_old, envs_old), ψ_old.AC[1], t, τ, alg.integrator; imaginary_evolution)
    AL_new, C_new = left_gauge(AC1, alg.alg_orth)
    T = isomorphism(scalartype(ψ_old), left_virtualspace(ψ_old, 1) ← left_virtualspace(ψ_old, 1))
    T = _bug_transport_left(ψ_old.AL[1], AL_new, T)          # overlap on bond 1 (new ← old)
    ψ.AC[1] = (AL_new, C_new)

    for i in 2:L
        Ĉ = _mul_front(T, ψ_old.AC[i])                       # reproject old connecting tensor
        ACi = integrate(AC_hamiltonian(i, ψ, H, ψ, envs), Ĉ, t, τ, alg.integrator; imaginary_evolution)
        if i == L
            imaginary_evolution && normalize!(ACi)
            ψ.AC[L] = ACi
        else
            AL_new, C_new = left_gauge(ACi, alg.alg_orth)
            T = _bug_transport_left(ψ_old.AL[i], AL_new, T)
            ψ.AC[i] = (AL_new, C_new)
        end
    end
    return ψ
end

# Right→left half-sweep (root = first site, leaf = last site): center ends at the first site.
function _bug_sweep_left!(ψ, H, t, τ, alg, envs; imaginary_evolution::Bool = false)
    L = length(ψ)
    ψ.AC[L]                                      # gauge center to last site (materialize AL[1..L-1])
    ψ_old = copy(ψ)
    envs_old = environments(ψ_old, H, ψ_old)

    # leaf (site L): K-step, keep only the new right isometry
    ACL = integrate(AC_hamiltonian(L, ψ_old, H, ψ_old, envs_old), ψ_old.AC[L], t, τ, alg.integrator; imaginary_evolution)
    C_new, AR_new = right_gauge(ACL, alg.alg_orth)
    T = isomorphism(scalartype(ψ_old), right_virtualspace(ψ_old, L) ← right_virtualspace(ψ_old, L))
    T = _bug_transport_right(ψ_old.AR[L], AR_new, T)         # overlap on bond L-1 (old ← new)
    ψ.AC[L] = (C_new, AR_new)

    for i in (L - 1):-1:1
        Ĉ = ψ_old.AC[i] * T                                  # reproject old connecting tensor
        ACi = integrate(AC_hamiltonian(i, ψ, H, ψ, envs), Ĉ, t, τ, alg.integrator; imaginary_evolution)
        if i == 1
            imaginary_evolution && normalize!(ACi)
            ψ.AC[1] = ACi
        else
            C_new, AR_new = right_gauge(ACi, alg.alg_orth)
            T = _bug_transport_right(ψ_old.AR[i], AR_new, T)
            ψ.AC[i] = (C_new, AR_new)
        end
    end
    return ψ
end

# Rank-adaptive half-sweeps
# -------------------------
# These mirror `_bug_sweep_right!`/`_bug_sweep_left!` exactly, except that at every internal node
# the fixed-rank `left_gauge`/`right_gauge` is replaced by an *augmentation*: the evolved
# connecting tensor `ACᵢ` is used as the candidate `K₁` and its directions orthogonal to the
# (transported) old isometry are appended, keeping the old basis as the leading per-sector block
# (`_bug_augment_left`/`_bug_augment_right`). The augmented isometry `Û` (rank up to 2r) is
# installed as the new left/right basis *without in-sweep truncation*: the enlarged bond is then
# seen by the next node's Galerkin evolution (which fills the new directions), exactly the
# augmented-basis Galerkin of the rank-adaptive matrix/TTN integrator. The bond factor
# `C_new = Û* ACᵢ` reconstructs `ACᵢ` exactly (`range(ACᵢ) ⊆ range(Û)`), so the embedding of the
# half-sweep state is exact; the subsequent `_bug_truncate!` performs the SVD-tail truncation.
#
# NOTE (deviation from the "core = Û* ACᵢ; svd_trunc!" recipe): truncating that core in place
# cannot grow the bond — it is a `(≤2r ← r)` matrix of rank ≤ r, so its SVD keeps ≤ r directions
# and the bond never exceeds the old rank. Growth requires evolving the *next* node on the
# enlarged bond before cutting, which is why truncation is deferred to `_bug_truncate!` (an
# optimal compression of the exact augmented-Galerkin state — equivalent to SVD-truncating the
# augmented core, but globally optimal).
function _bug_sweep_right_adaptive!(ψ, H, t, τ, alg, envs; imaginary_evolution::Bool = false)
    L = length(ψ)
    ψ.AC[1]
    ψ_old = copy(ψ)
    envs_old = environments(ψ_old, H, ψ_old)

    # leaf (site 1): K-step, then augment the outgoing (right) bond
    AC1 = integrate(AC_hamiltonian(1, ψ_old, H, ψ_old, envs_old), ψ_old.AC[1], t, τ, alg.integrator; imaginary_evolution)
    Û, _ = _bug_augment_left(ψ_old.AL[1], AC1, alg.alg_orth)   # Vl⊗P ← V̂₁ (old-first, up to 2r)
    C_new = Û' * AC1                                           # V̂₁ ← old_bond_1 (exact: Û·C_new = AC1)
    T = isomorphism(scalartype(ψ_old), left_virtualspace(ψ_old, 1) ← left_virtualspace(ψ_old, 1))
    T = _bug_transport_left(ψ_old.AL[1], Û, T)                # V̂₁ ← old_bond_1
    ψ.AC[1] = (Û, C_new)

    for i in 2:L
        Ĉ = _mul_front(T, ψ_old.AC[i])                        # V̂_{i-1} ⊗ P ← old_bond_i
        ACi = integrate(AC_hamiltonian(i, ψ, H, ψ, envs), Ĉ, t, τ, alg.integrator; imaginary_evolution)
        if i == L
            imaginary_evolution && normalize!(ACi)
            ψ.AC[L] = ACi
        else
            U₀ = _mul_front(T, ψ_old.AL[i])                   # old isometry in the new left frame
            Û, _ = _bug_augment_left(U₀, ACi, alg.alg_orth)   # V̂_{i-1} ⊗ P ← V̂_i (grows bond i)
            C_new = Û' * ACi                                  # V̂_i ← old_bond_i
            T = _bug_transport_left(ψ_old.AL[i], Û, T)        # V̂_i ← old_bond_i
            ψ.AC[i] = (Û, C_new)
        end
    end
    return ψ
end

function _bug_sweep_left_adaptive!(ψ, H, t, τ, alg, envs; imaginary_evolution::Bool = false)
    L = length(ψ)
    ψ.AC[L]
    ψ_old = copy(ψ)
    envs_old = environments(ψ_old, H, ψ_old)

    # leaf (site L): K-step, then augment the outgoing (left) bond
    ACL = integrate(AC_hamiltonian(L, ψ_old, H, ψ_old, envs_old), ψ_old.AC[L], t, τ, alg.integrator; imaginary_evolution)
    Û, _ = _bug_augment_right(ψ_old.AR[L], ACL, alg.alg_orth)  # V̂_{L-1} ⊗ P ← Vr (old-first, up to 2r)
    C_new = _transpose_tail(ACL) * _transpose_tail(Û)'         # old_bond_{L-1} ← V̂_{L-1} (exact)
    T = isomorphism(scalartype(ψ_old), right_virtualspace(ψ_old, L) ← right_virtualspace(ψ_old, L))
    T = _bug_transport_right(ψ_old.AR[L], Û, T)               # old_bond_{L-1} ← V̂_{L-1}
    ψ.AC[L] = (C_new, Û)

    for i in (L - 1):-1:1
        Ĉ = ψ_old.AC[i] * T                                   # old_bond_{i-1} ⊗ P ← V̂_i
        ACi = integrate(AC_hamiltonian(i, ψ, H, ψ, envs), Ĉ, t, τ, alg.integrator; imaginary_evolution)
        if i == 1
            imaginary_evolution && normalize!(ACi)
            ψ.AC[1] = ACi
        else
            U₀ = ψ_old.AR[i] * T                              # old isometry in the new right frame
            Û, _ = _bug_augment_right(U₀, ACi, alg.alg_orth)  # V̂_{i-1} ⊗ P ← V̂_i (grows bond i-1)
            C_new = _transpose_tail(ACi) * _transpose_tail(Û)' # old_bond_{i-1} ← V̂_{i-1}
            T = _bug_transport_right(ψ_old.AR[i], Û, T)       # old_bond_{i-1} ← V̂_{i-1}
            ψ.AC[i] = (C_new, Û)
        end
    end
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
