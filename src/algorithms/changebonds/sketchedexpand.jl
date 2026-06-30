"""
$(TYPEDEF)

An algorithm that expands the bond dimension like [`OptimalExpand`](@ref) — selecting the
dominant directions of the projected two-site update orthogonal to the current state — but at
single-site cost using the randomized "shrewd selection" of Controlled Bond Expansion
(Gleis et al. Phys. Rev. Lett. 130, 246402 (2023)). A random sketch of the orthogonal complement
is folded into the effective environment, collapsing the large bond before the two-site update is
ever formed, and the dominant directions are read off a small singular value decomposition.

The state-preserving behaviour matches [`OptimalExpand`](@ref).

!!! note
    Only defined for `FiniteMPS` (through [`changebond!`](@ref)), so it can be used standalone or
    as the `alg_expand` strategy of [`DMRG`](@ref). The reported `ϵ_2site` is a randomized
    estimate, and the folded application does not exploit `JordanMPO` sparsity.

## Fields

$(TYPEDFIELDS)
"""
@kwdef struct SketchedExpand{S} <: Algorithm
    "algorithm used to orthonormalize the sketched complement (passed as the `alg` of `left_orth!`/`right_orth!`); `nothing` selects QR without oversampling and an SVD-based decomposition otherwise"
    alg_orth::S = nothing

    "algorithm used for truncating the expanded space"
    trscheme::TruncationStrategy

    "number of extra sketch columns drawn beyond the target rank (range-finder oversampling)"
    oversampling::Int = 0

    "setting for how much information is displayed"
    verbosity::Int = Defaults.verbosity
end

"""
    sketch_space(V, alg::SketchedExpand) -> Vℓ, Vk

The random-sketch space `Vℓ` drawn from the complement `V`, together with its oversampling-free
target `Vk` (selected by `alg.trscheme`). `Vℓ` enlarges `Vk` by `alg.oversampling` extra
directions (capped by `V`); the selection is truncated back to `Vk`.
"""
function sketch_space(V, alg::SketchedExpand)
    Vk = sample_space(V, alg.trscheme)
    alg.oversampling == 0 && return Vk, Vk
    Vp = typeof(V)(c => min(dim(V, c), alg.oversampling) for c in sectors(V))
    return infimum(V, Vk ⊕ Vp), Vk
end

"""
    complement_space(Vfull::ElementarySpace, image::ElementarySpace)

The orthogonal complement `Vfull ⊖ image`, computed sector-wise from the spaces alone (no
factorization). Used to size and cap the random sketch.
"""
function complement_space(Vfull::ElementarySpace, image::ElementarySpace)
    @assert !isdual(Vfull)
    pairs = [c => (dim(Vfull, c) - dim(image, c)) for c in sectors(Vfull)]
    filter!(p -> last(p) > 0, pairs)
    return typeof(Vfull)(pairs)
end

# Finite system
# -------------
# Rather than forming the two-site update, a random sketch of the orthogonal complement is folded
# into the effective environment, and the dominant directions are read off a small SVD of the
# sketched gradient. The complement projectors act with the isometric MPS tensors directly, which
# leaves the MPS gauge (and any incrementally-maintained environments) untouched.
function changebond!(site::Int, ::Val{:right}, ψ::AbstractFiniteMPS, H, alg::SketchedExpand, envs; normalize::Bool = true)
    left = ψ.AC[site]
    right = ψ.AR[site + 1]
    AL, _ = left_gauge(left)        # local left-isometric form
    ARtt = _transpose_tail(right)   # AR is already right-isometric

    # nothing to add when either complement is empty (e.g. edge bonds)
    compL = complement_space(fuse(codomain(AL)), only(domain(AL)))
    compR = complement_space(fuse(domain(ARtt)), only(codomain(ARtt)))
    (dim(compL) == 0 || dim(compR) == 0) && return ψ

    # random sketch of the left complement, folded into the left environment
    Vℓ, Vk = sketch_space(compL, alg)
    Ω = randisometry(scalartype(left), codomain(AL) ← Vℓ)
    Q, _ = qr_compact!(project_complement!(Ω, AL))
    GL = leftenv(envs, site, ψ) * TransferMatrix(left, H[site], Q)
    Hac = MPO_AC_Hamiltonian(GL, H[site + 1], rightenv(envs, site + 1, ψ))
    Y = Hac * right

    # orthonormalize the sketched right complement, truncating away the oversampling padding
    B = project_complement_right!(_transpose_tail(Y), ARtt)
    if dim(Vℓ) == dim(Vk)        # no oversampling: a plain QR/LQ suffices
        _, ar_re = right_orth!(B; alg = alg.alg_orth)
    else
        _, ar_re = right_orth!(B; trunc = truncspace(Vk), alg = alg.alg_orth)
    end

    # optimal vectors at site+1
    # embed `left` into the enlarged domain (zero weight in the new directions), leaving the state
    # unchanged
    nal_space = codomain(left) ← (only(domain(left)) ⊕ space(ar_re, 1))
    nal, nc = left_gauge(absorb!(zerovector!(similar(left, nal_space)), left))
    nar = _transpose_front(catcodomain(_transpose_tail(right), ar_re))

    normalize && normalize!(nc)
    ψ.AC[site] = (nal, nc)
    ψ.AC[site + 1] = (nc, nar)
    return ψ
end
function changebond!(site::Int, ::Val{:left}, ψ::AbstractFiniteMPS, H, alg::SketchedExpand, envs; normalize::Bool = true)
    left = ψ.AL[site - 1]
    right = ψ.AC[site]
    _, ARtt = right_orth!(_transpose_tail(right; copy = true); trunc = notrunc())  # local right-isometric form

    # nothing to add when either complement is empty (e.g. edge bonds)
    compL = complement_space(fuse(codomain(left)), only(domain(left)))
    compR = complement_space(fuse(domain(ARtt)), only(codomain(ARtt)))
    (dim(compL) == 0 || dim(compR) == 0) && return ψ

    # random sketch of the right complement, folded into the right environment
    Vℓ, Vk = sketch_space(compR, alg)
    Ω = adjoint(randisometry(scalartype(right), domain(ARtt) ← Vℓ))
    _, Qr_o = lq_compact!(project_complement_right!(Ω, ARtt))
    Qr = _transpose_front(Qr_o)
    GR = TransferMatrix(right, H[site], Qr) * rightenv(envs, site, ψ)
    Hac = MPO_AC_Hamiltonian(leftenv(envs, site - 1, ψ), H[site - 1], GR)
    Y = Hac * left

    # orthonormalize the sketched left complement, truncating away the oversampling padding
    B = project_complement!(Y, left)
    if dim(Vℓ) == dim(Vk)        # no oversampling: a plain QR/LQ suffices
        Q, _ = left_orth!(B; alg = alg.alg_orth)
    else
        Q, _ = left_orth!(B; trunc = truncspace(Vk), alg = alg.alg_orth)
    end

    # optimal vectors at site-1
    right_tail = _transpose_tail(right)
    # embed `_transpose_tail(right)` into the enlarged codomain (zero weight in the new
    # directions), leaving the state unchanged
    nc_space = (codomain(right_tail)[1] ⊕ space(Q, 3)') ← domain(right_tail)
    nc, Qr2 = lq_compact!(absorb!(zerovector!(similar(right_tail, nc_space)), right_tail))
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
