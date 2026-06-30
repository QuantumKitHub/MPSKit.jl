"""
$(TYPEDEF)

An algorithm that expands the bond dimension by adding random unitary vectors that are
orthogonal to the existing state. This means that additional directions are added to
`AL` and `AR` that are contained in the nullspace of both. Note that this is happens in
parallel, and therefore the expansion will never go beyond the local two-site subspace.

The truncation strategy dictates the number of expanded states, by generating uniformly
distributed weights for each state in the two-site space and truncating that.

!!! note
    The environments are not used here, but [`changebonds!`](@ref) modifies both the state
    and environment so they remain consistent.

## Fields

$(TYPEDFIELDS)
"""
@kwdef struct RandExpand{S} <: Algorithm
    "algorithm used for the singular value decomposition"
    alg_svd::S = Defaults.alg_svd()

    "algorithm used for [truncation](@extref MatrixAlgebraKit.TruncationStrategy] the expanded space"
    trscheme::TruncationStrategy

    "setting for how much information is displayed"
    verbosity::Int = Defaults.verbosity
end

function changebonds!(ψ::InfiniteMPS, alg::RandExpand)
    T = eltype(ψ.AL)
    AL′ = similar(ψ.AL)
    AR′ = similar(ψ.AR, tensormaptype(spacetype(T), 1, numind(T) - 1, storagetype(T)))

    for i in 1:length(ψ)
        # obtain spaces by sampling the support of both the left and right nullspace
        VL = left_null(ψ.AL[i])
        VR = right_null!(_transpose_tail(ψ.AR[i + 1]; copy = true))
        V = sample_space(infimum(right_virtualspace(VL), space(VR, 1)), alg.trscheme)

        # obtain (orthogonal) directions as isometries in that direction
        XL = randisometry(scalartype(VL), right_virtualspace(VL) ← V)
        AL′[i] = VL * XL
        XR = randisometry(scalartype(VR), space(VR, 1) ← V)
        AR′[i + 1] = XR' * VR
    end

    return _expand!(ψ, AL′, AR′)
end

function changebonds!(ψ::MultilineMPS, alg::RandExpand)
    foreach(Base.Fix2(changebonds!, alg), ψ.data)
    return ψ
end

changebonds(ψ::AbstractMPS, alg::RandExpand) = changebonds!(copy(ψ), alg)
changebonds(ψ::MultilineMPS, alg::RandExpand) = changebonds!(copy(ψ), alg)

function changebonds(ψ, H, alg::RandExpand, envs)
    newψ = changebonds(ψ, alg)
    return newψ, environments(newψ, H, newψ)
end


function changebonds!(ψ, H, alg::RandExpand, envs)
    ψ = changebonds!(ψ, alg)
    recalculate!(envs, ψ, H)
    return ψ, envs
end


function changebonds!(ψ::AbstractFiniteMPS, alg::RandExpand)
    # the expansion directions are sampled from a randomized two-site update, so no Hamiltonian
    # or environments are required
    LoggingExtras.withlevel(; alg.verbosity) do
        for i in 1:(length(ψ) - 1)
            changebond!(i, Val(:right), ψ, nothing, alg, nothing)
        end
    end

    return normalize!(ψ)
end

function changebond!(site::Int, ::Val{:right}, ψ::AbstractFiniteMPS, H, alg::RandExpand, envs; normalize::Bool = true)
    bond = site
    left = ψ.AC[site]
    right = ψ.AR[site + 1]
    NL = left_null(left)
    NR = right_null!(_transpose_tail(right; copy = true))

    # randomized two-site update; H and envs are unused
    ac2 = randomize!(AC2(ψ, bond))

    # select the dominant directions in the complement of the current state
    g2 = adjoint(NL) * ac2 * adjoint(NR)
    gradnorm = norm(g2)
    _, _, Vᴴ, ϵ_select = svd_trunc!(normalize!(g2); trunc = alg.trscheme, alg = alg.alg_svd)
    @infov 4 "bond expansion" site dir = :right ϵ_select ϵ_2site = gradnorm / norm(ac2)

    # optimal vectors at site+1, zero weight at site
    ar_re = Vᴴ * NR
    # embed `left` into the enlarged domain (zero weight in the new directions)
    nal_space = codomain(left) ← (only(domain(left)) ⊕ space(Vᴴ, 1))
    nal, nc = left_gauge(absorb!(zerovector!(similar(left, nal_space)), left))
    nar = _transpose_front(catcodomain(_transpose_tail(right), ar_re))

    normalize && normalize!(nc)
    ψ.AC[site] = (nal, nc)
    ψ.AC[site + 1] = (nc, nar)
    return ψ
end
function changebond!(site::Int, ::Val{:left}, ψ::AbstractFiniteMPS, H, alg::RandExpand, envs; normalize::Bool = true)
    bond = site - 1
    left = ψ.AL[site - 1]
    right = ψ.AC[site]
    NL = left_null(left)
    NR = right_null!(_transpose_tail(right; copy = true))

    # randomized two-site update; H and envs are unused
    ac2 = randomize!(AC2(ψ, bond))

    # select the dominant directions in the complement of the current state
    g2 = adjoint(NL) * ac2 * adjoint(NR)
    gradnorm = norm(g2)
    U, _, _, ϵ_select = svd_trunc!(normalize!(g2); trunc = alg.trscheme, alg = alg.alg_svd)
    @infov 4 "bond expansion" site dir = :left ϵ_select ϵ_2site = gradnorm / norm(ac2)

    # optimal vectors at site-1, zero weight at site
    Q = NL * U
    # embed `_transpose_tail(right)` into the enlarged codomain (zero weight in the new directions)
    right_tail = _transpose_tail(right)
    nc_space = (codomain(right_tail)[1] ⊕ space(Q, 3)') ← domain(right_tail)
    nc, Qr = lq_compact!(absorb!(zerovector!(similar(right_tail, nc_space)), right_tail))
    AL_exp = catdomain(left, Q)

    normalize && normalize!(nc)
    ψ.AC[site] = (nc, _transpose_front(Qr))
    ψ.AC[site - 1] = (AL_exp, nc)
    return ψ
end

"""
    sample_space(V, strategy)

Sample basis states within a given `V::VectorSpace` by creating weights for each state that
are distributed uniformly, and then truncating according to the given `strategy`.
"""
function sample_space(V, strategy)
    S = TensorKit.SectorVector{Float64}(undef, V)
    Random.rand!(parent(S))
    ind = MatrixAlgebraKit.findtruncated(S, strategy)
    return TensorKit.Factorizations.truncate_space(V, ind)
end
