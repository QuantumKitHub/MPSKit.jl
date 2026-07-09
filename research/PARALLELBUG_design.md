# Design: `ParallelBUG` integrator for `FiniteMPS` in MPSKit

Stage 3 of the BUG family (see `BUG_finiteMPS_design.md` §7, which left this "not yet pinned
down"). This note **derives** the concrete MPS-level recipe from the two 2024 parallel papers and
records the design decisions for a `ParallelBUG <: Algorithm`.

Written on branch `ld-parallelbug` (worktree), branched off `ld-bug` (which carries the sequential
fixed-rank + rank-adaptive `BUG`).

## 1. Source algorithm (transcribed from the papers)

References in `research/`:
* Ceruti, Kusch, Lubich 2024, *A Parallel Rank-Adaptive Integrator for DLRA*, SISC 46(3) — the **matrix** case (the block formulas below).
* Ceruti, Kusch, Lubich, Sulz 2024, *A parallel BUG Integrator for Tree Tensor Networks*, arXiv:2412.00858 — the **tree** generalization (Alg. 1–4). **Has no MPS section**; the caterpillar specialization here is our derivation.

### 1.1 Matrix parallel BUG (the load-bearing block structure)

Start from `Y₀ = U₀ S₀ V₀*`, rank `r`. One step `t₀ → t₁ = t₀ + h`. **Three independent forward
ODEs, all from `t₀` data, all mutually parallel:**

* **K-step** (evolve the left factor + core, right factor frozen):
  `K̇ = F(t, K V₀*) V₀`, `K(t₀) = U₀ S₀`. Then `[U₀ | K(t₁)]` QR → `Û = [U₀ | Ũ₁]` (old-first).
* **L-step** (evolve the right factor + core, left factor frozen):
  `L̇ = F(t, U₀ L*)* U₀`, `L(t₀) = V₀ S₀*`. Then `[V₀ | L(t₁)]` QR → `V̂ = [V₀ | Ṽ₁]`.
* **S-step** (evolve the core, BOTH factors frozen — this is the *parallel* twist; sequential BUG
  would use the augmented bases here, forcing a `2r×2r` solve after K/L):
  `Ṡ̄ = U₀* F(t, U₀ S̄ V₀*) V₀`, `S̄(t₀) = S₀`. Size stays `r×r`.

**Augmented core (eq. 3.4)** — assembled *algebraically*, no extra ODE:
```
        old V₀      new Ṽ₁
      ┌───────────┬──────────┐
old U₀│  S̄(t₁)    │  S̃ᴸ      │      S̃ᴷ = Ũ₁* K(t₁)          (K coupling, bottom-left)
      ├───────────┼──────────┤      S̃ᴸ = L(t₁)* Ṽ₁         (L coupling, top-right)
new Ũ₁│  S̃ᴷ       │   0       │      new–new corner = 0  ⇒ FIRST ORDER
      └───────────┴──────────┘
```
Discarding the `0` corner costs `O(h² + hε)` local error — this is exactly why parallel BUG is
**first order** (sequential BUG keeps that block and is second order). Truncate `Ŝ₁` by SVD to
tolerance `ϑ` → new `U₁, V₁, S₁`.

**Step rejection (§3.3):** normal-component estimator `η = ‖Ũ₁* F(t₀, Y₀) Ṽ₁‖` (= the discarded
corner / `h`). Reject & recompute the step on the *augmented* bases `Û, V̂` if either
(a) `r₁ = 2r` (truncation saturated the doubling cap), or (b) `h·η > c·ϑ` (`c ≈ 10`).

### 1.2 Tree/MPS generalization

Per node `τ` with children `τᵢ`: Galerkin-evolve the connecting tensor with **all surrounding bases
frozen at `t₀`** (Alg. 3); build each bond's new directions `Ũ¹_{τᵢ}` old-first (Alg. 4A); assemble
the augmented connecting tensor by **stacking coupling blocks mode-by-mode** (Alg. 4B):
```
Ĉ ← C̄¹;   for each child mode i:   Ĉ ← Tenᵢ( [ Matᵢ(Ĉ) ; Matᵢ(C̃ᵢ) ] )
C̃ᵢ = h · F_τ(Y₀) ×_{j≠i} U₀*_{τⱼ} ×ᵢ Ũ¹*_{τᵢ}          (all multi-new corners left at 0)
```
Robust first-order global bound (Thm 4.5): `‖Yₖ − A(tₖ)‖ ≤ c₁h + c₂ε + c₃δ + c₄·k·ϑ`, constants
independent of the bond singular values. Truncation term `c₄·k·ϑ` ⇒ scale `ϑ ∝ h`.

## 2. MPS specialization (caterpillar rooted at site L)

`FiniteMPS` = linear tree, rooted at site `L` (left-canonical: `AL[1..L-1]`, center at `L`). Physical
legs are **uncompressed leaves** (`U = 𝟙`), so they are never augmented — **only the `L-1` virtual
bonds grow.** Treat each **bond `b`** (between sites `b`, `b+1`) as the matrix-DLRA object:

| matrix object | MPS object at bond `b` | MPSKit primitive |
|---|---|---|
| `U₀` (left factor) | left block `AL[1..b]`, locally `AL[b]` | — |
| `V₀` (right factor) | right block `AR[b+1..L]`, locally `AR[b+1]` | — |
| `S₀` (core) | bond tensor `C[b]` | — |
| K-step `K(t₁)` | evolve site `b`: `AC_hamiltonian(b)` on `AC[b]`, frozen `t₀` | `integrate` |
| L-step `L(t₁)` | evolve site `b+1`: `AC_hamiltonian(b+1)` on `AC[b+1]`, frozen `t₀` | `integrate` |
| S-step `S̄(t₁)` | evolve bond `b`: `C_hamiltonian(b)` on `C[b]`, frozen `t₀` | `integrate` |
| `Ũ₁` (new left dirs) | `_bug_augment_left(AL[b], K(t₁))` (already in `bug.jl`) | `left_null`/`catdomain` |
| `Ṽ₁` (new right dirs) | `_bug_augment_right(AR[b+1], L(t₁))` (already in `bug.jl`) | `right_null`/`catcodomain` |

This reuses **exactly** the effective operators (`AC_hamiltonian`, `C_hamiltonian`, `integrate`) and
augmentation helpers (`_bug_augment_left/right`) already present — the only genuinely new pieces are
the frozen-`t₀` scheduling, the `2×2` core assembly, and step rejection.

**All local ODEs read from one frozen snapshot** `(ψ₀ = copy(ψ), envs₀ = environments(ψ₀, H, ψ₀))`,
so every `AC_hamiltonian(i)`/`C_hamiltonian(b)` is well-defined simultaneously (identical to how the
`InfiniteMPS` `timestep` in `tdvp.jl` evolves all `AC`/`C` from one frozen `envs`). This is the
parallelizable structure — reuse that file's `@sync`/`Threads.@spawn` + `tmap!` gated on
`Defaults.scheduler[]`.

## 2b. CORRECTION (implementation finding, must supersede §2's per-bond picture)

The per-bond matrix-DLRA mapping in §2 (an S-step via `C_hamiltonian(b)` on **every** bond) is
**wrong for MPS** and was falsified in implementation:

* **Phase overcounting.** MPSKit's `C_hamiltonian(b)` is the *full-environment* effective
  Hamiltonian, so each bond's S-step carries the full energy phase `exp(-iE·dt)`. With a core on
  every one of the `L-1` bonds these phases **compose**, giving `exp(-i(L-1)E·dt)` — the eigenstate
  phase is overcounted by exactly `(L-1)×` (measured: ratio 2.0 at `L=3`, 3.0 at `L=4`). There is no
  backward substep to cancel it (unlike TDVP).
* **No bond growth.** Augmenting from the *frozen-evolved `AC[i]`* cannot grow a bond, because
  `range(AC[i]) = range(AL[i])` up to the evolution, and worse, `_bug_augment_left(AL[i], AC[i])`
  extracts `g = N'·AC` which is ~0.

**The correct MPS mapping is the literal tree recursion, not L−1 independent matrix problems:**

* The caterpillar rooted at `L` is **one** tree. Only the **root** connecting tensor (`AC[L]`)
  carries amplitude/phase; every interior connecting tensor is an **isometry** (`AL[i]`). So there is
  **no per-bond `C_hamiltonian` S-step at all** — the amplitude is evolved **once**, at the root.
* Interior nodes `i<L` Galerkin-evolve the **isometry**: `C̄¹_i = integrate(AC_hamiltonian(i, ψ₀, H,
  ψ₀, envs₀), ψ₀.AL[i], t, dt)` — note `AL[i]`, *not* `AC[i]`. The evolved isometry leaves its old
  range (`g = N'·C̄¹_i ≠ 0`), so `_bug_augment_left(AL[i], C̄¹_i)` yields genuine new bond directions
  → bonds grow. The root evolves the amplitude: `C̄¹_L = integrate(AC_hamiltonian(L,…), ψ₀.AC[L],…)`.
* First order comes from the **explicit-Euler coupling blocks with zeroed multi-new corners**
  (Alg. 4B), assembled by the leaves→root recursion — *not* from transporting fully-evolved tensors
  (that accidentally recovers 2nd order, as observed).

This is the single hardest part and is the crux still to land; §2's table is retained only for the
matrix/2-site base case (which is exact).

## 3. Key design decisions (recorded)

1. **New name, not a flag on `BUG`.** A separate `ParallelBUG <: Algorithm` struct rather than
   `BUG(; parallel=true)`. Rationale: different order (1st vs 2nd), different control flow (no sweep,
   frozen envs, step rejection), and different field set (needs `c`, max rejections). Mirrors how
   `TDVP`/`TDVP2` are distinct structs. Keeps the well-tested `BUG` byte-for-byte.
2. **Always augment-then-truncate.** Unlike sequential `BUG`, the parallel integrator is
   *intrinsically* rank-adaptive: it doubles every bond then truncates. `notrunc()` therefore means
   "truncate back to the pre-step per-bond dimensions" (fixed-rank parallel BUG); any
   `truncerror`/`truncrank` gives genuine rank adaptivity. Reuse `SvdCut` for the truncation sweep,
   exactly as `_bug_truncate!` does.
3. **Reuse `_bug_augment_left/right`.** The old-first `[U₀ | Ũ₁]` construction with the `[𝟙;0]`
   overlap is already implemented, tested (`bug_augment.jl`), and sector-correct. The coupling blocks
   `S̃ᴷ = Ũ₁* K(t₁)`, `S̃ᴸ = L(t₁)* Ṽ₁` are contractions against the returned `Ũ₁`/`Ṽ₁`.
4. **Frozen-`t₀` snapshot + lazy env self-heal.** One `copy(ψ)` + its `environments`. After
   installing the truncated result, `FiniteEnvironments` recomputes lazily (no `recalculate!` exists
   for finite envs — confirmed). No old/new env bookkeeping (that was sequential BUG's H8 subtlety);
   here every read is from `t₀`, which is *simpler* than sequential BUG.
5. **Correctness gate = behavioral, mirroring `BUG` tests.** Rather than white-box block matching, the
   MPS assembly is validated against the same oracles the sequential `BUG` uses: (a) overlap vs a
   dense `exp(-iH·T)` reference, (b) agreement with `TDVP` to `O(dt)`, (c) **first-order** log–log
   slope ≈ 1 (vs `BUG`'s 2), (d) imaginary-time monotone energy lowering + norm, (e) symmetric-tensor
   sector/charge preservation. This is what "verify correctness similar to how the regular BUG is
   tested" means, and it makes the assembly detail self-correcting.
6. **Step rejection: opt-in, capped.** `maxiter_rejection` (default e.g. 3) bounds recomputes; `c`
   (default 10, the paper's value) sets the `h·η > c·ϑ` threshold. With `notrunc()` (fixed rank) the
   rank-saturation trigger is disabled (we deliberately cut back to the old rank). Rejection is a
   genuine-adaptivity feature; keep it simple and cheap (`η` from one frozen `F₀` apply).

## 4. Build plan (small steps, each validated)

1. **Struct + registration.** `ParallelBUG` mirroring `BUG` fields (+`c`, `maxiter_rejection`); kw
   constructor; `include` + export. No logic yet. Sanity: `ParallelBUG()` constructs.
2. **Serial driver, no rejection.** `timestep!` (+ copying `timestep`): frozen snapshot → per-site
   `AC_hamiltonian` + per-bond `C_hamiltonian` local solves → per-bond `2×2` augmented assembly →
   `SvdCut` truncation. Serial scheduler first (correctness before parallelism).
   Gate: energy conservation on a ground state; first-order slope; TDVP agreement; dense overlap.
3. **Imaginary time + norm.** Renormalize after truncation; monotone energy-lowering test.
4. **Symmetric tensors.** U(1)/Z2/SU(2) sector + charge preservation, dynamic grading (reuse
   `bug.jl`'s symmetric test scaffold).
5. **Step rejection.** `η` estimator + recompute loop; a test that a rank-1 start grows past a single
   doubling within one step via rejection.
6. **Parallelism.** Swap the per-site/per-bond loops onto `@sync`/`Threads.@spawn` + `tmap!` gated on
   `Defaults.scheduler[]` (mirror `tdvp.jl`'s `InfiniteMPS` branch). Gate: identical results to the
   serial path (scheduler must not change the answer).

## 5. Risk register (inherits `BUG_finiteMPS_design.md` §9 H1–H10, plus)

| # | Risk | Mitigation |
|---|---|---|
| P1 | MPS assembly of the `2×2` block across *both* bonds of each site is subtle (each bond has one new space, from its left neighbor); easy to double-count the core (cf. §3 sequential warning) | Behavioral gate (§3.5): first-order slope + dense overlap will expose a wrong assembly; build the 2-site case first where it reduces to the exact matrix formulas |
| P2 | New-new corner must be *exactly* zero (first-order); a stray coupling makes it inconsistent | Assemble with explicit `zerovector!` block; assert bond dims double before truncation |
| P3 | Thread safety of shared frozen graded snapshot (H9) | Genuine `copy`; read-only `envs₀`; follow `tdvp.jl` infinite threading structure; no shared per-sector buffers |
| P4 | `ϑ` accumulates as `c₄·k·ϑ`; users expect TDVP-like accuracy | Document `ϑ ∝ h`; note first-order in docstring like `BUG`'s note |
