"""
$(TYPEDEF)

An algorithm that expands the given mps as described in
[Zauner-Stauber et al. Phys. Rev. B 97 (2018)](@cite zauner-stauber2018), by selecting the
dominant contributions of a two-site updated MPS tensor, orthogonal to the original ψ.

The expansion is state-preserving: the added directions are connected through a zero block,
so that the expanded state represents the same physical state as the original one (as required
for e.g. TDVP).

!!! note
    `trunc` bounds how much is *added* to each bond, not the total bond dimension that is kept.
    It is applied to the two-site update projected onto the orthogonal complement of the current
    state, so `truncrank(k)` grows every bond by at most `k` — capped by the dimension of the
    local two-site complement, which is why a bond can grow by less than `k`, or not at all.
    The `trunc` of [`SvdCut`](@ref), and of the drivers [`DMRG`](@ref) and [`TDVP`](@ref), has
    the other meaning: it bounds what is *kept*.

!!! note
    The projected block is normalized before the decomposition, so a value-based strategy
    (`trunctol`, `truncerror`) selects a *fraction of the complement weight* rather than an
    absolute error on the state, and the retained fraction does not shrink as the state
    converges. `truncrank` and `truncspace` are the strategies with a robust meaning here.

!!! note
    [`changebonds!`](@ref) is only defined for `FiniteMPS`, and modifies both the state and its environment.

# Fields

$(TYPEDFIELDS)

# See also

Used as the `algorithm` argument of [`changebonds`](@ref) and [`changebonds!`](@ref).
"""
@kwdef struct OptimalExpand{S, B} <: Algorithm
    "algorithm used for the singular value decomposition"
    alg_svd::S = Defaults.alg_svd()

    "[truncation strategy](@extref MatrixAlgebraKit.TruncationStrategy) selecting how many directions are *added* to each bond, rather than how much of the bond is kept"
    trunc::TruncationStrategy

    "backend for tensor contractions and index manipulations"
    backend::B = Defaults.backend()
end

# Simple wrapper to convert between diffrent type of InifniteMPS.
function changebonds(
        ψ::InfiniteMPS, operator::InfiniteMPO, alg::OptimalExpand, envs = environments(ψ, operator, ψ)
    )
    ψ′, envs′ = changebonds(
        convert(MultilineMPS, ψ), convert(MultilineMPO, operator), alg, Multiline([envs])
    )
    return convert(InfiniteMPS, ψ′), only(parent(envs′))
end

function changebonds(
        ψ::InfiniteMPS, H::InfiniteMPOHamiltonian, alg::OptimalExpand,
        envs = environments(ψ, H, ψ)
    )
    allocator = default_allocator(ψ, SerialScheduler())
    T = eltype(ψ.AL)
    AL′ = similar(ψ.AL)
    AR′ = similar(ψ.AR, tensormaptype(spacetype(T), 1, numind(T) - 1, storagetype(T)))
    for i in 1:length(ψ)
        # determine optimal expansion spaces around bond i
        AC2 = AC2_projection(i, ψ, H, ψ, envs; kind = :ACAR, alg.backend, allocator)

        # Use the nullspaces and SVD decomposition to determine the optimal expansion space
        VL = left_null(ψ.AL[i])
        VR = right_null!(_transpose_tail(ψ.AR[i + 1]; copy = true))
        intermediate = adjoint(VL) * AC2 * adjoint(VR)
        nrm = norm(intermediate)
        # skip empty-content bonds; normalizing zero would NaN the SVD
        nrm > eps(real(scalartype(intermediate)))^(3 / 4) && scale!(intermediate, inv(nrm))
        U, _, Vᴴ = svd_trunc!(intermediate; trunc = alg.trunc, alg = alg.alg_svd)

        AL′[i] = VL * U
        AR′[i + 1] = Vᴴ * VR
    end

    newψ = _expand(ψ, AL′, AR′)
    envs = environments(newψ, H, newψ)
    return newψ, envs
end

function changebonds(ψ::MultilineMPS, H, alg::OptimalExpand, envs = environments(ψ, H, ψ))
    allocator = default_allocator(ψ, SerialScheduler())
    TL = eltype(ψ.AL)
    AL′ = PeriodicMatrix{TL}(undef, size(ψ.AL))
    TR = tensormaptype(spacetype(TL), 1, numind(TL) - 1, storagetype(TL))
    AR′ = PeriodicMatrix{TR}(undef, size(ψ.AR))

    # determine optimal expansion spaces around bond i
    for i in 1:size(ψ, 1), j in 1:size(ψ, 2)
        AC2 = AC2_projection(
            CartesianIndex(i - 1, j), ψ, H, ψ, envs;
            kind = :ACAR, alg.backend, allocator
        )

        # Use the nullspaces and SVD decomposition to determine the optimal expansion space
        VL = left_null(ψ.AL[i, j])
        VR = right_null!(_transpose_tail(ψ.AR[i, j + 1]; copy = true))
        intermediate = adjoint(VL) * AC2 * adjoint(VR)
        nrm = norm(intermediate)
        nrm > eps(real(scalartype(intermediate)))^(3 / 4) && scale!(intermediate, inv(nrm))
        U, _, Vᴴ = svd_trunc!(intermediate; trunc = alg.trunc, alg = alg.alg_svd)

        AL′[i, j] = VL * U
        AR′[i, j + 1] = Vᴴ * VR
    end

    newψ = _expand(ψ, AL′, AR′)
    envs = environments(newψ, H, newψ) #  recalculate!(envs, newψ, H)
    return newψ, envs
end

# Finite system
# -------------
function changebond!(
        site::Int, ::Val{:right}, ψ::AbstractFiniteMPS, H, alg::OptimalExpand, envs;
        normalize::Bool = true,
        # a sweep that already holds one should pass it: this is called once per site
        allocator = default_allocator(ψ, SerialScheduler()),
    )
    bond = site
    left = ψ.AC[site]
    right = ψ.AR[site + 1]
    NL = left_null(left)
    NR = right_null!(_transpose_tail(right; copy = true))

    # two-site update from the projected effective Hamiltonian
    AC2 = AC2_projection(bond, ψ, H, ψ, envs; alg.backend, allocator)

    # select the dominant directions in the complement of the current state
    g2 = adjoint(NL) * AC2 * adjoint(NR)
    nrm = norm(g2)
    # nothing to expand here; normalizing a zero g2 would NaN the SVD
    nrm ≤ eps(real(scalartype(g2)))^(3 / 4) && return ψ
    _, _, Vᴴ = svd_trunc!(scale!(g2, inv(nrm)); trunc = alg.trunc, alg = alg.alg_svd)

    # optimal vectors at site+1
    ar_re = Vᴴ * NR
    # embed `left` into the enlarged domain (zero weight in the new directions), leaving the state
    # unchanged
    nal_space = codomain(left) ← (only(domain(left)) ⊕ space(Vᴴ, 1))
    nal, nc, _ = left_gauge(absorb!(zerovector!(similar(left, nal_space)), left))
    nar = _transpose_front(catcodomain(_transpose_tail(right), ar_re))

    normalize && normalize!(nc)
    ψ.AC[site] = (nal, nc)
    ψ.AC[site + 1] = (nc, nar)
    return ψ
end
function changebond!(
        site::Int, ::Val{:left}, ψ::AbstractFiniteMPS, H, alg::OptimalExpand, envs;
        normalize::Bool = true,
        # a sweep that already holds one should pass it: this is called once per site
        allocator = default_allocator(ψ, SerialScheduler()),
    )
    bond = site - 1
    left = ψ.AL[site - 1]
    right = ψ.AC[site]
    NL = left_null(left)
    NR = right_null!(_transpose_tail(right; copy = true))

    # two-site update from the projected effective Hamiltonian
    AC2 = AC2_projection(bond, ψ, H, ψ, envs; alg.backend, allocator)

    # select the dominant directions in the complement of the current state
    g2 = adjoint(NL) * AC2 * adjoint(NR)
    nrm = norm(g2)
    nrm ≤ eps(real(scalartype(g2)))^(3 / 4) && return ψ
    U, _, _ = svd_trunc!(scale!(g2, inv(nrm)); trunc = alg.trunc, alg = alg.alg_svd)

    # optimal vectors at site-1
    Q = NL * U
    right_tail = _transpose_tail(right)
    # embed `_transpose_tail(right)` into the enlarged codomain (zero weight in the new
    # directions), leaving the state unchanged
    nc_space = (codomain(right_tail)[1] ⊕ space(Q, 3)') ← domain(right_tail)
    nc, Qr = lq_compact!(absorb!(zerovector!(similar(right_tail, nc_space)), right_tail))
    AL_exp = catdomain(left, Q)

    normalize && normalize!(nc)
    ψ.AC[site] = (nc, _transpose_front(Qr))
    ψ.AC[site - 1] = (AL_exp, nc)
    return ψ
end

changebonds(ψ::AbstractFiniteMPS, H, alg::OptimalExpand, envs = environments(ψ, H, ψ)) =
    changebonds!(copy(ψ), H, alg, envs)

function changebonds!(ψ::AbstractFiniteMPS, H, alg::OptimalExpand, envs = environments(ψ, H, ψ))
    for i in 1:(length(ψ) - 1)
        changebond!(i, Val(:right), ψ, H, alg, envs)
    end
    return ψ, envs
end
