"""
    transfer_spectrum(above::InfiniteMPS, [below=above], [alg]; num_vals=20, kwargs...)
        -> TensorKit.SectorVector

Calculate the partial spectrum of the left transfer matrix corresponding to the overlap of a given `above` state and a `below` state.
The result is returned as a `TensorKit.SectorVector`, whose values can be inspected per sector by indexing, i.e. `transfer_spectrum(above)[sector]`.

# Arguments

- `above::InfiniteMPS`: the state for the "above" leg of the mixed transfer matrix.
- `below::InfiniteMPS=above`: the state for the "below" leg; defaults to the pure transfer matrix of `above`.
- `alg`: the eigensolver algorithm specification, resolved per sector via [`MatrixAlgebraKit.select_algorithm`](@extref MatrixAlgebraKit.select_algorithm).
  This can be a KrylovKit algorithm instance (used verbatim for every sector), a `MatrixAlgebraKit.DefaultAlgorithm` or `NamedTuple` bundling keyword arguments,
  or `nothing` (the default) to construct the eigensolver from the keyword arguments below.

# Keyword arguments

- `num_vals = 20`: the number of eigenvalues to compute. This can either be a single `Int`, which is used for every sector of the transfer space,
  or an `AbstractDict`/iterable of `sector => count` pairs to restrict the computation to specific sectors and request a different number of values per sector.
- `oversampling = 10`: additive margin on the Krylov dimension beyond the number of values requested in a sector (see `krylovdim` below).
- `oversampling_factor = 1`: proportionality factor between the Krylov dimension and the number of values requested in a sector (see `krylovdim` below).
- `krylovdim`: the Krylov dimension of the eigensolver. Unless given explicitly, this is chosen adaptively per sector as
  `max(Defaults.krylovdim, ceil(Int, oversampling_factor * num_vals) + oversampling)`, where `num_vals` is the number of values requested in that sector.
- `kwargs...`: further keyword arguments (e.g. `tol`, `maxiter`) are forwarded to `MatrixAlgebraKit.default_algorithm` to build the eigensolver.
  Passing eigensolver keyword arguments when `alg` is an algorithm instance is not allowed, and will result in an error.
"""
function transfer_spectrum(
        above::InfiniteMPS, below::InfiniteMPS = above, alg = nothing;
        num_vals = 20, kwargs...
    )
    S = TensorKit.check_spacetype(above, below)
    transferspace = fuse(left_virtualspace(above, 1) * left_virtualspace(below, 1)')
    howmany = _transfer_howmany(num_vals, transferspace)

    T = complex(scalartype(above))
    Vsp = S(c => n for (c, n) in howmany)
    spectrum = TensorKit.SectorVector{T}(undef, Vsp)

    A = flip(TransferMatrix(above.AL, below.AL))
    for (c, n) in howmany
        algorithm = MatrixAlgebraKit.select_algorithm(
            transfer_spectrum, above, alg; num_vals = n, kwargs...
        )
        Va = S(c => 1)
        Vt = left_virtualspace(above, 1)
        Vb = left_virtualspace(below, 1)
        init = randomize!(similar(above.AL[1], Vb ← Va' ⊗ Vt))
        eigenvals, _, convhist = eigsolve(A, init, n, :LM, algorithm)
        convhist.converged < n &&
            @warn "transfer spectrum in sector $c not converged: normres = $(convhist.normres)"
        spectrum[c] = view(eigenvals, 1:n)
    end
    return spectrum
end

# convenience: positional `alg` without repeating `below`
transfer_spectrum(above::InfiniteMPS, alg; kwargs...) =
    transfer_spectrum(above, above, alg; kwargs...)

# normalize the `num_vals` specification into a `SectorDict` of per-sector counts,
# capped by the transfer-space dimension in each sector
_transfer_howmany(num_vals::Int, V) =
    TensorKit.SectorDict(c => min(num_vals, dim(V, c)) for c in sectors(V))
function _transfer_howmany(num_vals, V)
    return TensorKit.SectorDict(
        c => min(n, dim(V, c)) for (c, n) in num_vals if dim(V, c) > 0
    )
end

# `num_vals` is intercepted here so it never reaches the algorithm constructor, and is
# exempt from the no-kwargs-with-`alg` rule (it is injected per sector by `transfer_spectrum`)
function MatrixAlgebraKit.select_algorithm(
        ::typeof(transfer_spectrum), above::Union{InfiniteMPS, Type{<:InfiniteMPS}},
        alg = nothing; num_vals = 20, kwargs...
    )
    if isnothing(alg)
        return MatrixAlgebraKit.default_algorithm(transfer_spectrum, above; num_vals, kwargs...)
    elseif alg isa NamedTuple || alg isa Base.Pairs
        isempty(kwargs) ||
            throw(ArgumentError("keyword arguments are not allowed when `alg` is a NamedTuple"))
        return MatrixAlgebraKit.default_algorithm(transfer_spectrum, above; alg..., num_vals)
    elseif alg isa DefaultAlgorithm
        isempty(kwargs) ||
            throw(ArgumentError("keyword arguments are not allowed when `alg` is given"))
        return MatrixAlgebraKit.default_algorithm(transfer_spectrum, above; alg.kwargs..., num_vals)
    elseif alg isa KrylovAlgorithm
        isempty(kwargs) ||
            throw(ArgumentError("keyword arguments are not allowed when `alg` is given"))
        return alg
    end
    throw(ArgumentError("Unknown alg $alg"))
end

function MatrixAlgebraKit.default_algorithm(
        ::typeof(transfer_spectrum), ::Type{<:InfiniteMPS}; num_vals = 20,
        tol = Defaults.tol, maxiter = Defaults.maxiter,
        oversampling = 10, oversampling_factor = 1,
        krylovdim = max(Defaults.krylovdim, ceil(Int, oversampling_factor * num_vals) + oversampling),
        verbosity = Defaults.VERBOSE_NONE, eager = true
    )
    return Arnoldi(; tol, maxiter, krylovdim, verbosity, eager)
end

"""
Find the closest fractions of π, differing at most ```tol_angle```
"""
function approx_angles(spectrum; tol_angle = 0.1)
    angles = angle.(spectrum) ./ π                          # ∈ ]-1, 1]
    angles_approx = rationalize.(angles, tol = tol_angle)     # ∈ [-1, 1]

    # Remove the effects of the branchcut.
    angles_approx[findall(angles_approx .== -1)] .= 1       # ∈ ]-1, 1]

    return angles_approx .* π                               # ∈ ]-π, π]
end

"""
    marek_gap(above::InfiniteMPS; sector=nothing, kwargs...)

Compute the gap `ϵ` for the asymptotics of the transfer matrix, as well as the Marek gap
`δ` as a scaling measure of the bond dimension, along with the associated angle `θ`.

By default this returns a `TensorKit.SectorDict` mapping each sector of the transfer spectrum
to its `(ϵ, δ, θ)` triplet. Passing a specific `sector` returns only that sector's triplet. The
remaining `kwargs` are passed on to [`transfer_spectrum`](@ref).
"""
function marek_gap(above::InfiniteMPS; sector = nothing, tol_angle = 0.1, num_vals = 20, kwargs...)
    if isnothing(sector)
        spectrum = transfer_spectrum(above; num_vals, kwargs...)
        return TensorKit.SectorDict(c => marek_gap(spectrum[c]; tol_angle) for c in keys(spectrum))
    end
    spectrum = transfer_spectrum(above; num_vals = Dict(sector => num_vals), kwargs...)
    return marek_gap(spectrum[sector]; tol_angle)
end

function marek_gap(spectrum::AbstractVector{T}; tol_angle = 0.1) where {T <: Number}
    # Remove 1s from the spectrum
    inds = findall(abs.(spectrum) .< 1 - 1.0e-12)
    length(spectrum) - length(inds) < 2 || @warn "Non-injective mps?"

    spectrum = spectrum[inds]

    angles = approx_angles(spectrum; tol_angle = tol_angle)
    θ = first(angles)

    spectrum_at_angle = spectrum[findall(angles .== θ)]

    lambdas = -log.(abs.(spectrum_at_angle))

    ϵ = first(lambdas)

    δ = Inf
    if length(lambdas) > 2
        δ = lambdas[2] - lambdas[1]
    end

    return ϵ, δ, θ
end

"""
    correlation_length(above::InfiniteMPS; sector=nothing, kwargs...)

Compute the correlation length of a given InfiniteMPS based on the next-to-leading
eigenvalue of the transfer matrix.

By default this returns a `TensorKit.SectorDict` mapping each sector of the transfer spectrum
to its correlation length. Passing a specific `sector` returns only that sector's correlation
length as a scalar. The remaining `kwargs` are passed on to [`transfer_spectrum`](@ref).
"""
function correlation_length(above::InfiniteMPS; sector = nothing, kwargs...)
    gaps = marek_gap(above; sector, kwargs...)
    isnothing(sector) || return 1 / first(gaps)                       # gaps :: (ϵ, δ, θ)
    return TensorKit.SectorDict(c => 1 / first(g) for (c, g) in gaps)  # gaps :: SectorDict
end

function correlation_length(spectrum::AbstractVector{T}; kwargs...) where {T <: Number}
    ϵ, = marek_gap(spectrum; kwargs...)
    return 1 / ϵ
end
