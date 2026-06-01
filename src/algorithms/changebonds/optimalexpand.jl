"""
$(TYPEDEF)

An algorithm that expands the given mps as described in
[Zauner-Stauber et al. Phys. Rev. B 97 (2018)](@cite zauner-stauber2018), by selecting the
dominant contributions of a two-site updated MPS tensor, orthogonal to the original ψ.

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
    ϵ_2site = norm(g2) / norm(AC2)
    _, _, Vᴴ, ϵ_select = svd_trunc!(normalize!(g2); trunc = alg.trscheme, alg = alg.alg_svd)

    # optimal vectors at site+1, zero weight at site
    ar_re = Vᴴ * NR
    # embed `left` into the enlarged domain (zero weight in the new directions)
    nal_space = codomain(left) ← (only(domain(left)) ⊕ space(Vᴴ, 1))
    nal, nc = qr_compact!(absorb!(zerovector!(similar(left, nal_space)), left))
    nar = _transpose_front(catcodomain(_transpose_tail(right), ar_re))

    ψ.AC[site] = (nal, normalize!(nc))
    ψ.AC[site + 1] = (nc, nar)
    return ψ, (; ϵ_select, ϵ_2site)
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
    ϵ_2site = norm(g2) / norm(AC2)
    U, _, _, ϵ_select = svd_trunc!(normalize!(g2); trunc = alg.trscheme, alg = alg.alg_svd)

    # optimal vectors at site-1, zero weight at site
    Q = NL * U
    # embed `_transpose_tail(right)` into the enlarged codomain (zero weight in the new directions)
    right_tail = _transpose_tail(right)
    nc_space = (codomain(right_tail)[1] ⊕ space(Q, 3)') ← domain(right_tail)
    nc, Qr = lq_compact!(absorb!(zerovector!(similar(right_tail, nc_space)), right_tail))
    AL_exp = catdomain(left, Q)

    ψ.AC[site] = (normalize!(nc), _transpose_front(Qr))
    ψ.AC[site - 1] = (AL_exp, nc)
    return ψ, (; ϵ_select, ϵ_2site)
end

changebonds(ψ::AbstractFiniteMPS, H, alg::OptimalExpand, envs = environments(ψ, H, ψ)) =
    changebonds!(copy(ψ), H, alg, envs)

function changebonds!(ψ::AbstractFiniteMPS, H, alg::OptimalExpand, envs = environments(ψ, H, ψ))
    for i in 1:(length(ψ) - 1)
        changebond!(i, Val(:right), ψ, H, alg, envs)
    end
    return ψ, envs
end
