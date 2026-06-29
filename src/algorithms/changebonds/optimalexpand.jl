"""
$(TYPEDEF)

An algorithm that expands the given mps as described in
[Zauner-Stauber et al. Phys. Rev. B 97 (2018)](@cite zauner-stauber2018), by selecting the
dominant contributions of a two-site updated MPS tensor, orthogonal to the original ψ.

By default the expansion does not alter the state: the added directions are connected through a
zero block, so that the expanded state is identical to the original one (as required for e.g.
TDVP). When `warmstart = true`, the new directions are instead seeded with the (physically
scaled) two-site gradient itself, so the expanded state already moves toward the optimal
two-site update. This changes the state and is therefore only useful for ground-state search
(e.g. as the `alg_expand` strategy of [`DMRG`](@ref)), where it warm-starts the subsequent
single-site optimization; the injected amplitude scales with the gradient norm and so vanishes
automatically at convergence.

!!! note
    [`changebonds!`](@ref) is only defined for `FiniteMPS`, and modifies both the state and its environment.

## Fields

$(TYPEDFIELDS)
"""
@kwdef struct OptimalExpand{S} <: Algorithm
    "algorithm used for the singular value decomposition"
    alg_svd::S = Defaults.alg_svd()

    "algorithm used for truncating the expanded space"
    trscheme::TruncationStrategy

    "whether to seed the new directions with the two-site gradient (warm start, alters the state) instead of a zero block"
    warmstart::Bool = false

    "setting for how much information is displayed"
    verbosity::Int = Defaults.verbosity
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
    T = eltype(ψ.AL)
    AL′ = similar(ψ.AL)
    AR′ = similar(ψ.AR, tensormaptype(spacetype(T), 1, numind(T) - 1, storagetype(T)))
    for i in 1:length(ψ)
        # determine optimal expansion spaces around bond i
        AC2 = AC2_projection(i, ψ, H, ψ, envs; kind = :ACAR)

        # Use the nullspaces and SVD decomposition to determine the optimal expansion space
        VL = left_null(ψ.AL[i])
        VR = right_null!(_transpose_tail(ψ.AR[i + 1]; copy = true))
        intermediate = normalize!(adjoint(VL) * AC2 * adjoint(VR))
        U, _, Vᴴ = svd_trunc!(intermediate; trunc = alg.trscheme, alg = alg.alg_svd)

        AL′[i] = VL * U
        AR′[i + 1] = Vᴴ * VR
    end

    newψ = _expand(ψ, AL′, AR′)
    envs = environments(newψ, H, newψ)
    return newψ, envs
end

function changebonds(ψ::MultilineMPS, H, alg::OptimalExpand, envs = environments(ψ, H, ψ))
    TL = eltype(ψ.AL)
    AL′ = PeriodicMatrix{TL}(undef, size(ψ.AL))
    TR = tensormaptype(spacetype(TL), 1, numind(TL) - 1, storagetype(TL))
    AR′ = PeriodicMatrix{TR}(undef, size(ψ.AR))

    # determine optimal expansion spaces around bond i
    for i in 1:size(ψ, 1), j in 1:size(ψ, 2)
        AC2 = AC2_projection(CartesianIndex(i - 1, j), ψ, H, ψ, envs; kind = :ACAR)

        # Use the nullspaces and SVD decomposition to determine the optimal expansion space
        VL = left_null(ψ.AL[i, j])
        VR = right_null!(_transpose_tail(ψ.AR[i, j + 1]; copy = true))
        intermediate = normalize!(adjoint(VL) * AC2 * adjoint(VR))
        U, _, Vᴴ = svd_trunc!(intermediate; trunc = alg.trscheme, alg = alg.alg_svd)

        AL′[i, j] = VL * U
        AR′[i, j + 1] = Vᴴ * VR
    end

    newψ = _expand(ψ, AL′, AR′)
    envs = environments(newψ, H, newψ) #  recalculate!(envs, newψ, H)
    return newψ, envs
end


# Finite system
# -------------
function changebond!(site::Int, ::Val{:right}, ψ::AbstractFiniteMPS, H, alg::OptimalExpand, envs)
    bond = site
    left = ψ.AC[site]
    right = ψ.AR[site + 1]
    NL = left_null(left)
    NR = right_null!(_transpose_tail(right; copy = true))

    # two-site update from the projected effective Hamiltonian
    AC2 = AC2_projection(bond, ψ, H, ψ, envs)

    # select the dominant directions in the complement of the current state
    g2 = adjoint(NL) * AC2 * adjoint(NR)
    gradnorm = norm(g2)
    U, S, Vᴴ, ϵ_select = svd_trunc!(normalize!(g2); trunc = alg.trscheme, alg = alg.alg_svd)
    @infov 4 "bond expansion" site dir = :right ϵ_select ϵ_2site = gradnorm / norm(AC2)

    # optimal vectors at site+1
    ar_re = Vᴴ * NR
    if alg.warmstart
        # seed the new left directions with the (physically scaled) gradient instead of a zero
        # block, warm-starting the subsequent optimization (alters the state)
        nal, nc = qr_compact!(catdomain(left, gradnorm * (NL * U * S)))
    else
        # embed `left` into the enlarged domain (zero weight in the new directions)
        nal_space = codomain(left) ← (only(domain(left)) ⊕ space(Vᴴ, 1))
        nal, nc = qr_compact!(absorb!(zerovector!(similar(left, nal_space)), left))
    end
    nar = _transpose_front(catcodomain(_transpose_tail(right), ar_re))

    ψ.AC[site] = (nal, normalize!(nc))
    ψ.AC[site + 1] = (nc, nar)
    return ψ
end
function changebond!(site::Int, ::Val{:left}, ψ::AbstractFiniteMPS, H, alg::OptimalExpand, envs)
    bond = site - 1
    left = ψ.AL[site - 1]
    right = ψ.AC[site]
    NL = left_null(left)
    NR = right_null!(_transpose_tail(right; copy = true))

    # two-site update from the projected effective Hamiltonian
    AC2 = AC2_projection(bond, ψ, H, ψ, envs)

    # select the dominant directions in the complement of the current state
    g2 = adjoint(NL) * AC2 * adjoint(NR)
    gradnorm = norm(g2)
    U, S, Vᴴ, ϵ_select = svd_trunc!(normalize!(g2); trunc = alg.trscheme, alg = alg.alg_svd)
    @infov 4 "bond expansion" site dir = :left ϵ_select ϵ_2site = gradnorm / norm(AC2)

    # optimal vectors at site-1
    Q = NL * U
    right_tail = _transpose_tail(right)
    if alg.warmstart
        # seed the new right directions with the (physically scaled) gradient instead of a zero
        # block, warm-starting the subsequent optimization (alters the state)
        nc, Qr = lq_compact!(catcodomain(right_tail, gradnorm * (S * Vᴴ * NR)))
    else
        # embed `_transpose_tail(right)` into the enlarged codomain (zero weight in the new directions)
        nc_space = (codomain(right_tail)[1] ⊕ space(Q, 3)') ← domain(right_tail)
        nc, Qr = lq_compact!(absorb!(zerovector!(similar(right_tail, nc_space)), right_tail))
    end
    AL_exp = catdomain(left, Q)

    ψ.AC[site] = (normalize!(nc), _transpose_front(Qr))
    ψ.AC[site - 1] = (AL_exp, nc)
    return ψ
end

changebonds(ψ::AbstractFiniteMPS, H, alg::OptimalExpand, envs = environments(ψ, H, ψ)) =
    changebonds!(copy(ψ), H, alg, envs)

function changebonds!(ψ::AbstractFiniteMPS, H, alg::OptimalExpand, envs = environments(ψ, H, ψ))
    LoggingExtras.withlevel(; alg.verbosity) do
        for i in 1:(length(ψ) - 1)
            changebond!(i, Val(:right), ψ, H, alg, envs)
        end
    end
    return ψ, envs
end
