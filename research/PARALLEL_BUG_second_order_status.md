# Second-order parallel BUG (`ParallelBUG2`) — status and diagnosis

## RESOLVED (2026-07-11): genuine second order — faithful Kusch Variant 2

`ParallelBUG2` clears the decisive local-order gate with a **uniform** single-step slope **≈ 3** in `dt`
(vs the first-order `ParallelBUG`'s slope ≈ 2). Measured with the shared harness
(`test/algorithms/parallelbug.jl`, TFIsing, `dts = [0.04, 0.02, 0.01, 0.005]`, `trscheme = truncerror(atol=1e-12)`):

| L | bond | slopes | note |
|---|---|---|---|
| 2 | ℙ^4 | (exact) | machine-exact, no dropped corner |
| 3 | ℙ^8 | (exact) | full-rank enriched basis completes the block |
| 4 | ℙ^8 | ≈ 3.0 | genuine second order |
| 5 | ℙ^8 | ≈ 3.0 | genuine second order |
| 6 | ℙ^4 | ≈ 3.0 | unsaturated interior bond (a real dropped corner) |
| 7 | ℙ^4 | ≈ 3.0 | stays second order as the chain grows |

The construction implemented in `src/algorithms/timestep/bug.jl` (`_pbug2_assemble_core` and helpers):

1. **Enrich** both the left (`Û0`) and right (`V̂0`) bond bases to rank `2r` with one frozen `H·ψ₀`
   application, old-first, with the mixed-env coupling propagating deep directions to the root; fold the
   enriched environments `GLhat`/`GRhat` by **explicit transfer matrices** (no `FiniteMPS` round-trip, so
   the zero-weight enriched directions never collapse — the route-1 blocker below).
2. **K-step** every center on the enriched environments, **freezing the enriched right basis `V̂0`**
   (freezing the *old* right basis gives slope 2 — see route 2). Threaded like the first-order phase-1.
3. **Assemble** leaves→root: interior tensors are the `4r` isometries `[Û0 | Ũ2]` (new directions from the
   K-evolved centers), and the **evolved-amplitude coupling `R = Ũ2ᵀĈ`** is transported one site at a time
   through the frozen right isometry `V̂0`. This is the ingredient the earlier attempts missed: the
   first-order *frozen-derivative* coupling `h·F(Y0)` projected onto `Ũ2` is ≈ 0 (because `Ũ2 ⟂ Û0` and
   `Û0` already contains the `H·ψ₀` direction), so it transports no second-order content and the interior
   bonds stay first order (slope 2 at `L ≥ 6`). The evolved amplitude `Ũ2ᵀĈ` is the genuine `O(h²)` term,
   the caterpillar analogue of the matrix recipe's `Ũ2ᵀK(t1)` block.
4. **Root** core = the exact matrix `Ŝ1` assembly: the rank-`2r` Galerkin `S̄1 = Kevo[L]` in the `Û0` rows
   stacked with the transported coupling in the `Ũ2` rows, and the **new–new corner ZERO** (at the root the
   right bond is trivial, so the `LᵀṼ2` block vanishes too). This gives the published local error `O(h³)`.

This matches Kusch (2024) Variant 2 (`4r`, old-first bases, no predicted basis / no transfer matrix `Mi`),
specialized to the caterpillar rooted at the last site. The faithfulness signature is the **uniform** slope 3
(an earlier hybrid that replaced the root `Ŝ1` with a full `4r` Galerkin over-resolved to slope ≈ 4 at
small L — more accurate, but not the published structure; the exact zeroed-corner assembly removes it).
It clears the full shared verification (2-site exactness, energy/phase, TDVP agreement, imaginary-time
lowering, Z2 charge preservation, serial=threaded). Only plain Hamiltonians are supported so far (no
`LazySum`/time-dependent operators). It is NOT the reference-MATLAB path (that is Variant 1 with predicted
bases `U0_hat` + transfer matrices `Mi` + `3r`); the simpler Variant 2 was implemented as the handoff
recommended. `ParallelBUG2` is still marked experimental.

The diagnosis of the earlier dead ends is retained below for the record.

---

## Original status (superseded): the dead ends

One-sentence status (historical): the **first-order** `ParallelBUG` is audited and passes the decisive
local-order slope-≈2 gate at `L = 3, 4`; a genuine **second-order** `ParallelBUG2` is **not** achieved — the
tractable implementation routes measure local slope ≈ 2 (still first order) or break, and the correct
route is a substantial faithful port of the reference tree-tensor-network algorithm that remains to be
done. `ParallelBUG2` is committed as an experimental struct that is exact for `L ≤ 2` and `throw`s for
`L ≥ 3`, so it can never be silently mislabeled as second order.

This note records what was tried, the measured slopes, why each route fails, and the precise remaining
work, so the port can be resumed without re-deriving the dead ends.

## The decisive gate

Single-step local error `‖step(dt) − exp(−iH·dt)ψ₀‖` vs `dt`, log–log slope, full bonds (`ℙ^8`),
transverse-field Ising, `dts = [0.04, 0.02, 0.01, 0.005]`:

- **First order ⇒ slope ≈ 2**; **genuine second order ⇒ slope ≈ 3**.
- `L = 2` is exact (no dropped corner) and does **not** distinguish the orders — the gate needs `L ≥ 3`.
- Empirically `ParallelBUG` (first order) gives clean slope ≈ 2.0 at `L = 3, 4, 5, 6`. This is the
  committed gate in `test/algorithms/parallelbug.jl` (`@testset "local order slope L=$Lc"`).

## What was tried for second order, and the measured slope

The correct *mechanism* (Kusch 2024): pre-augment each bond basis with one `H·ψ₀` application **before**
evolving, run the Galerkin center at the enlarged rank on that basis, assemble with the new–new corner
kept zero (now `O(h³)`). Three MPS realizations were attempted:

1. **Reuse the first-order assembly on an `H·ψ₀`-enriched `FiniteMPS` snapshot `ψ̂0`** (the approved
   plan's central idea). **Structurally impossible.** The enriched directions carry *zero weight*
   (`ψ̂0 == ψ₀` as a state — verified overlap 1.0), so `FiniteMPS` canonicalization *collapses* them:
   the stored `AL[2]` keeps the enriched space (`ℙ^4`) but the mixed-gauge `AC[2]` collapses to `ℙ^2`,
   and the first-order `_pbug_assemble_core` then hits a `SpaceMismatch` at `L ≥ 3`. A zero-weight
   enriched basis cannot survive a canonical-form round-trip — full stop.

2. **Explicit enriched-left environment + rank-2r root Galerkin, no 4r augmentation** (materialize a
   `FiniteMPS` only from the final, nonzero-weight tensors — avoids the collapse). **Runs and is
   stable, but slope ≈ 2.0 at `L = 4`** (`L = 2, 3` come out machine-exact only because the enriched
   basis happens to complete the small left block). Enriching the left basis + a root Galerkin *alone*
   is a better-constant first-order method — the second-order correction also needs the interior K-step
   directions augmented to `4r`/`3r` and the enriched *right* basis frozen in the K-step.

3. **Predicted-basis Galerkin (`ψhat = compress(H·ψ₀)` to rank r) + transfer matrices + old-first
   augmentation.** **Broken: slope ≈ 0** (O(1) error ≈ 0.15, independent of `dt`). Evolving `ψ₀`'s
   center on `ψhat`'s environments and transferring back with an ad-hoc overlap matrix does *not*
   reproduce the reference's prolongation/restriction: the basis reconciliation between the
   predicted-basis core `C1_bar` and the old-first augmented basis is not captured, so the assembled
   state is O(1) wrong.

Route 2's slope ≈ 2 reproduces the prior (reverted, never-committed) attempt's result; route 1 is a new
structural finding; route 3 shows an ad-hoc predicted-basis handling is worse, not better.

## Why it is hard: the mixed-basis / basis-bookkeeping problem

The reference `TTN_integrator_parallel_2nd_order_nonglobal.m` (Variant 1, `3r`) does **not** enrich a
single shared bond space. Per node it builds, per child:

- a rank-`r` **predicted basis** `U0_hat` = orthonormalized range of the `H·ψ₀` image *through the old
  environment* `Q0_i` (line 38), **not** the old-first `[old | new]` 2r basis, and **not** a global
  SVD-compression of `H·ψ₀`;
- a **transfer matrix** `Mi = ⟨old | U0_hat⟩` (line 49) that maps the old core into the predicted basis;
- the core is evolved from `C0 = ttm(core, Mi)` on the `U0_hat` bases (`func_ODE`), giving `C1_bar` in
  the *predicted* basis;
- the augmented basis is `[old | K-evolved]` zero-padded to `3r`, and `C1_bar` is placed in rows
  `1:r` with the coupling `Ci` in rows `r+1:3r` (lines 135–164).

The subtlety that defeats the shortcuts: `C1_bar` lives in the `U0_hat` (predicted) basis while the
augmented columns are the `[old | K-new]` basis; the two are reconciled *implicitly* through `Mi` and
the specific `3r` block placement. Getting local slope 3 requires reproducing this reconciliation
exactly — approximating `U0_hat` (route 3) or skipping the predicted basis (route 2) loses it.

The left/right asymmetry compounds this: the matrix Variant-2 recipe uses *separate* enriched left
(`Û0`) and right (`V̂0`) bases whose new subspaces differ, so the interior K-step center is a genuinely
*mixed* (rectangular) object — exactly what prolongation/restriction manage and what a naive shared-bond
MPS representation cannot.

## Remaining work (to resume the port)

1. **Faithfully port the reference recursion** `TTN_integrator_parallel_2nd_order_nonglobal.m` to the
   `FiniteMPS` caterpillar (root at `L`), reusing MPSKit's effective-Hamiltonian environments as the
   prolongation/restriction. Build the rank-`r` predicted basis `U0_hat` per bond as the orthonormalized
   `H·ψ₀` image *through the old right environment* (not a global compression), the transfer `Mi`, evolve
   the core on `U0_hat` from `C0 = ttm(core, Mi)`, and assemble the `3r` core with `C1_bar` in the
   old-block rows and the coupling in rows `r+1:3r`, new–new corner zero. Re-gauge only the final
   (nonzero-weight) tensors into a `FiniteMPS`.
2. **Gate at `L ≥ 3` (use `L = 4, 5`; `L = 3` full-rank can be exact and masks the corner)** on local
   slope ≈ 3 before labeling it second order. Mirror the `ParallelBUG` slope harness already in
   `test/algorithms/parallelbug.jl`.
3. Cross-check against a MATLAB trace of the reference on a 3-site chain to pin down the `Mi` / `3r`
   basis reconciliation numerically — this is where the shortcuts went wrong and where a from-scratch
   derivation is most error-prone.

## Faithful-port attempt (Variant 2, `4r`) — how far it got and the real blocker

A second, deeper attempt targeted **Variant 2** (`4r`, [K24] §5.3) specifically because it uses
**old-first enriched bases** `Û0 = orth([U0, F0·V0])`, `V̂0 = orth([V0, F0ᵀ·U0])` and **no predicted
basis / no transfer matrix `Mi`** — so it sidesteps the predicted↔old reconciliation that broke route 3
above. Progress and findings:

- **Working building blocks** (verified to run and produce sensible spaces):
  - old-first **left** enrichment `Û0[i]` (2r) via the existing `_pbug_newdirs`/`_pbug_stack_child`
    machinery fed the `H·ψ₀` image (this is the retained `_pbug_preaugment`);
  - old-first **right** enrichment `V̂0[i]` (2r) via `_bug_augment_right` fed the `H·ψ₀` image;
  - enriched left/right **environments** by explicit `TransferMatrix` folding through `Û0` / `V̂0`
    (`GLhat[i]`, `GRhat[i]`) — no `FiniteMPS` round-trip, so no zero-weight collapse;
  - **old→enriched center embedding** via `isometry(enriched_bond ← old_bond)` (old-first ⇒ the
    canonical `[I;0]` injection) on the front (left) and tail (right) legs.
  - Confirmed the left- and right-enriched bond spaces genuinely **differ** (e.g. L=4, `ℙ^8`: bond 1
    is `ℙ^2` from the left but `ℙ^4` from the right; bond 3 the reverse), matching the matrix recipe's
    distinct `Û0`/`V̂0`. The final state uses the **left**-augmented bonds; `V̂0` is only the frozen
    right environment for the K-step.

- **The real blocker (precise):** the enrichment, K-step, and `4r` augmentation **cannot be
  precomputed as separate passes and then assembled** — they must be **interleaved in a single
  leaves→root sweep with growing bonds**. Precomputing `Û0` at rank `2r` and then augmenting to `4r`
  yields **inconsistent chained bonds**: site `i`'s right bond becomes `4r` after augmentation, but
  the precomputed `Û0[i+1]` was built with a `2r` left bond, so `As[i+1].left ≠ As[i].right`. The
  first-order `_pbug_assemble_core` avoids this by doing the augmentation *inside* the sweep (stacking
  each site's old block onto the previous site's already-grown new bond). The second-order sweep must
  do the same but additionally (a) enrich each site's old block with `H·ψ₀` on the *current* (already
  grown) bond, (b) run the K-step on the mixed `(Û0-left, V̂0-right)` enriched environments, (c)
  augment to `4r`, and (d) propagate the off-diagonal coupling blocks (`Ũ2ᵀK`, `Lᵀ Ṽ2`) — all in one
  pass. This interleaved, growing-bond, mixed-basis sweep with coupling propagation is the substantial
  remaining implementation; it is a rewrite of the assembly sweep, not an add-on.

## What is delivered now

- `ParallelBUG` (first order): audited against the §3.1 checklist (interior tensors are
  `[AL⁰ | Ũ]` isometries; solves act on the amplitude-weighted center `AC⁰`), and given a **committed
  local-order slope-≈2 gate at `L = 3, 4`** — the gate that would catch any regression or a mislabeled
  order. No code change was required (the audit confirmed the existing assembly).
- `ParallelBUG2`: struct + keyword constructor + `timestep!`/`timestep` wiring + export, sharing the
  `AbstractParallelBUG` supertype and the first-order truncation/rejection helpers. Exact for `L ≤ 2`;
  `throw`s an informative `ArgumentError` for `L ≥ 3` (pointing here) so it is never silently first
  order. The `_pbug_preaugment` pass (correct `H·ψ₀` left enrichment, verified to preserve the state
  to overlap 1.0) is retained as a building block for the port.
