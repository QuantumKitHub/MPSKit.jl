"""
$(TYPEDEF)

An algorithm that expands the bond dimension by selecting the dominant directions of the
projected two-site update orthogonal to the current state, just like [`OptimalExpand`](@ref),
but at single-site cost using a randomized range-finder (the "shrewd selection" of Controlled
Bond Expansion, Gleis et al. Phys. Rev. Lett. 130, 246402 (2023)).

Rather than forming the full projected two-site update (which requires applying the two-site
effective Hamiltonian), a random sketch of width `rank + oversampling` of the orthogonal
complement is folded into the effective *environment* first, collapsing the large bond before
the two-site contraction is ever materialized. The dominant directions are then recovered from
a small singular value decomposition of the sketched gradient.

!!! note
    This strategy is only defined for `FiniteMPS` (through [`changebond!`](@ref)), so that it
    can be used both standalone and as the `alg_expand` strategy of [`DMRG`](@ref).

Like [`OptimalExpand`](@ref), the expansion is by default state-preserving (the added
directions are connected through a zero block, as required for e.g. TDVP). When
`warmstart = true`, the new directions are instead seeded with the (physically scaled) sketched
two-site gradient, warm-starting the subsequent single-site optimization in ground-state search
(e.g. as the `alg_expand` strategy of [`DMRG`](@ref)); this alters the state, and the injected
amplitude scales with the gradient norm so that it vanishes automatically at convergence.

!!! note
    `ϵ_2site` is a randomized estimate of the two-site complement-gradient norm (a diagnostic
    only), not the exact ratio `‖g2‖ / ‖AC2‖` reported by [`OptimalExpand`](@ref). The folded
    application routes the `MPOHamiltonian` tensors through the generic transfer/one-site
    machinery and therefore does not exploit `JordanMPO` sparsity in the MPO bond dimension.

## Fields

$(TYPEDFIELDS)
"""
@kwdef struct SketchedExpand{S} <: Algorithm
    "algorithm used for the (small) singular value decomposition"
    alg_svd::S = Defaults.alg_svd()

    "algorithm used for truncating the expanded space"
    trscheme::TruncationStrategy

    "number of extra sketch columns drawn beyond the target rank (range-finder oversampling)"
    oversampling::Int = 5

    "whether to seed the new directions with the sketched two-site gradient (warm start, alters the state) instead of a zero block"
    warmstart::Bool = false

    "setting for how much information is displayed"
    verbosity::Int = Defaults.verbosity
end

"""
    sketch_space(V, alg::SketchedExpand)

Determine the space of the random sketch drawn from the complement space `V`: the target
subspace selected by `alg.trscheme`, enlarged by `alg.oversampling` extra directions (capped
by `V`). Reuses [`sample_space`](@ref) so the result is sector-correct for graded spaces.
"""
function sketch_space(V, alg::SketchedExpand)
    Vk = sample_space(V, alg.trscheme)
    alg.oversampling == 0 && return Vk
    Vp = typeof(V)(c => min(dim(V, c), alg.oversampling) for c in sectors(V))
    return infimum(V, Vk ⊕ Vp)
end

"""
    complement_space(Vfull::ElementarySpace, image::ElementarySpace)

The orthogonal complement `Vfull ⊖ image` of the range `image` inside the fused space `Vfull`,
computed sector-wise from the spaces alone (no factorization). For an isometry `A : Vfull ← image`
this is the space carried by `left_null(A)`; here `Vfull` is the fused codomain/domain of the
isometric MPS tensor and `image` its small virtual leg. Used to size and cap the random sketch.
"""
function complement_space(Vfull::ElementarySpace, image::ElementarySpace)
    @assert !isdual(Vfull)
    pairs = [c => (dim(Vfull, c) - dim(image, c)) for c in sectors(Vfull)]
    filter!(p -> last(p) > 0, pairs)
    return typeof(Vfull)(pairs)
end

# Finite system
# -------------
# The "shrewd selection" needs the orthogonal complement of the current state on either side of
# the bond. Rather than materializing an explicit complement basis with `left_null`/`right_null`,
# we use the complement projector of the *isometric* MPS tensor `A`: `I - A A'` (left-iso) and
# `I - A' A` (right-iso). The isometric tensor is obtained from the center tensor by a local
# `left_orth`/`right_orth!`, which leaves the MPS gauge (and any environments a caller maintains
# incrementally, e.g. CBE `DMRG`) untouched. On the sketch side this turns the range-finder into a
# narrow random draw projected into the complement; on the selection side the gradient is projected
# and its dominant directions read off directly from the SVD (its singular vectors land in the
# complement automatically). The kept rank is capped at the complement dimension: the projected
# object has only that many genuine singular values, and at edge bonds the complement may be empty,
# where an uncapped `svd_trunc` would otherwise pad the selection with non-orthogonal noise.
function changebond!(site::Int, ::Val{:right}, ψ::AbstractFiniteMPS, H, alg::SketchedExpand, envs; normalize::Bool = true)
    left = ψ.AC[site]
    right = ψ.AR[site + 1]
    AL, _ = left_orth(left)            # local left-isometric form (range(AL) = range(left))
    ARtt = _transpose_tail(right)      # right-isometric: ARtt * ARtt' = I

    # nothing to add when either complement is empty (e.g. edge bonds)
    compL = complement_space(fuse(codomain(AL)), only(domain(AL)))
    compR = complement_space(fuse(domain(ARtt)), only(codomain(ARtt)))
    (dim(compL) == 0 || dim(compR) == 0) && return ψ

    # sketch the left complement: Q = (I - AL AL') Ω lies in the complement of `left`, re-orthonormalized
    # by a narrow QR into a proper randomized range-finder basis (no full complement basis is formed)
    Vℓ = sketch_space(compL, alg)
    Ω = randisometry(scalartype(left), codomain(AL) ← Vℓ)
    Q, _ = qr_compact!(project_complement!(Ω, AL))

    # fold the sketch into the left environment (bra-leg becomes the sketch leg), then apply the
    # existing single-site effective Hamiltonian on site+1: this reconstructs `Q† AC2_projection`
    # without ever forming the full two-site update
    GL = leftenv(envs, site, ψ) * TransferMatrix(left, H[site], Q)
    Hac = MPO_AC_Hamiltonian(GL, H[site + 1], rightenv(envs, site + 1, ψ))
    Y = Hac * right

    # project onto the right complement with (I - AR' AR): the SVD right-vectors lie in the complement
    Ytt = _transpose_tail(Y)
    B = project_complement_right!(Ytt, ARtt)
    nrm = norm(B)
    trunc = alg.trscheme & MatrixAlgebraKit.truncrank(dim(compR))
    U, S, Vᴴ, ϵ_select = svd_trunc!(normalize!(B); trunc, alg = alg.alg_svd)
    @infov 4 "bond expansion (sketched)" site dir = :right ϵ_select ϵ_2site = nrm

    # optimal vectors at site+1 (identical bookkeeping to OptimalExpand)
    ar_re = Vᴴ
    if alg.warmstart
        # seed the new left directions with the (physically scaled) sketched gradient instead of
        # a zero block; `Q * U` maps the sketch-basis left vectors back to the full space
        nal, nc = qr_compact!(catdomain(left, nrm * (Q * U * S)))
    else
        nal_space = codomain(left) ← (only(domain(left)) ⊕ space(Vᴴ, 1))
        nal, nc = qr_compact!(absorb!(zerovector!(similar(left, nal_space)), left))
    end
    nar = _transpose_front(catcodomain(_transpose_tail(right), ar_re))

    normalize && normalize!(nc)
    ψ.AC[site] = (nal, nc)
    ψ.AC[site + 1] = (nc, nar)
    return ψ
end
function changebond!(site::Int, ::Val{:left}, ψ::AbstractFiniteMPS, H, alg::SketchedExpand, envs; normalize::Bool = true)
    left = ψ.AL[site - 1]              # left-isometric, left of center (valid)
    right = ψ.AC[site]
    _, ARtt = right_orth!(_transpose_tail(right; copy = true); trunc = notrunc())  # local right-iso

    # nothing to add when either complement is empty (e.g. edge bonds)
    compL = complement_space(fuse(codomain(left)), only(domain(left)))
    compR = complement_space(fuse(domain(ARtt)), only(codomain(ARtt)))
    (dim(compL) == 0 || dim(compR) == 0) && return ψ

    # sketch the right complement: Qr = (I - AR' AR) Ω lies in the complement of `right`, with
    # orthonormal rows recovered by a narrow LQ (the randomized range-finder basis)
    Vℓ = sketch_space(compR, alg)
    Ω = adjoint(randisometry(scalartype(right), domain(ARtt) ← Vℓ))
    _, Qr_o = lq_compact!(project_complement_right!(Ω, ARtt))
    Qr = _transpose_front(Qr_o)

    # fold the sketch into the right environment, then apply the single-site effective
    # Hamiltonian on site-1: reconstructs `AC2_projection NR†` (sketched) at single-site cost
    GR = TransferMatrix(right, H[site], Qr) * rightenv(envs, site, ψ)
    Hac = MPO_AC_Hamiltonian(leftenv(envs, site - 1, ψ), H[site - 1], GR)
    Y = Hac * left

    # project onto the left complement with (I - AL AL'): the SVD left-vectors lie in the complement
    B = project_complement!(Y, left)
    nrm = norm(B)
    trunc = alg.trscheme & MatrixAlgebraKit.truncrank(dim(compL))
    U, S, Vᴴ, ϵ_select = svd_trunc!(normalize!(B); trunc, alg = alg.alg_svd)
    @infov 4 "bond expansion (sketched)" site dir = :left ϵ_select ϵ_2site = nrm

    # optimal vectors at site-1 (identical bookkeeping to OptimalExpand)
    Q = U
    right_tail = _transpose_tail(right)
    if alg.warmstart
        # seed the new right directions with the (physically scaled) sketched gradient instead of
        # a zero block; `Vᴴ * Qr_o` maps the sketch-basis right vectors back to the full space
        nc, Qr2 = lq_compact!(catcodomain(right_tail, nrm * (S * Vᴴ * Qr_o)))
    else
        nc_space = (codomain(right_tail)[1] ⊕ space(Q, 3)') ← domain(right_tail)
        nc, Qr2 = lq_compact!(absorb!(zerovector!(similar(right_tail, nc_space)), right_tail))
    end
    AL_exp = catdomain(left, Q)

    normalize && normalize!(nc)
    ψ.AC[site] = (nc, _transpose_front(Qr2))
    ψ.AC[site - 1] = (AL_exp, nc)
    return ψ
end

changebonds(ψ::AbstractFiniteMPS, H, alg::SketchedExpand, envs = environments(ψ, H, ψ)) =
    changebonds!(copy(ψ), H, alg, envs)

function changebonds!(ψ::AbstractFiniteMPS, H, alg::SketchedExpand, envs = environments(ψ, H, ψ))
    LoggingExtras.withlevel(; alg.verbosity) do
        for i in 1:(length(ψ) - 1)
            changebond!(i, Val(:right), ψ, H, alg, envs)
        end
    end
    return ψ, envs
end
