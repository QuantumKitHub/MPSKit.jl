"""
$(TYPEDEF)

Algorithm that approximates an open-boundary finite MPO-MPS product using a zip-up sweep, optionally
followed by a zip-down sweep in the opposite direction. The MPO and MPS are contracted one site at a
time, and the enlarged virtual bond is truncated immediately. The sweep direction is selected by
`left_to_right`.

    approximate((O, ϕ), alg::Zipup) -> ψ, info
    approximate!(ψ, (O, ϕ), alg::Zipup) -> ψ, info

Contrary to the variational algorithms, this algorithm requires no initial guess:
the in-place version simply uses `ψ` as the destination of the sweep, overwriting its contents, and may alias `ϕ`.
The out-of-place version allocates a destination with the promoted scalar type of `O` and `ϕ`.
Both return an [`AlgorithmInfo`](@ref) alongside the approximated state which contains the truncation information.

# Constructors

    Zipup(; trunc, alg_svd = Defaults.alg_svd(), left_to_right = true)
    Zipup(alg_zipup, [alg_zipdown]; left_to_right = true)

Create a `Zipup` algorithm with the given truncated gauge algorithm, or by passing a truncation scheme and singular value decomposition algorithm.
The keyword `trunc` can be either one truncation strategy for a single zip-up sweep, or a tuple `(zipup_trunc, zipdown_trunc)` for a zip-up sweep followed by a zip-down sweep.
Equivalently, one can pass the corresponding truncated gauge algorithms directly as `alg_zipup` and `alg_zipdown`.
The keyword `left_to_right` selects the direction of the zip-up sweep, the zip-down sweep always running in the opposite direction.

Following Paeckel et al., if the desired final bond dimension is `D`, one can use a more permissive zip-up truncation, e.g. rank `2D` with stricter tolerances, and use `alg_zipdown` to impose the final truncation.

# Fields

$(TYPEDFIELDS)

# References

* [Stoudenmire and White New J. Phys. 12 (2010)](@cite stoudenmire2010)
* [Paeckel et al. Ann. of Phys. 411 (2019)](@cite paeckel2019)
"""
struct Zipup{
        U <: MatrixAlgebraKit.TruncatedAlgorithm,
        D <: Union{Nothing, MatrixAlgebraKit.TruncatedAlgorithm},
    } <: Algorithm
    "algorithm used for gauging and truncating the local tensors during the zip-up sweep"
    alg_zipup::U
    "algorithm used for the final locally gauged truncation pass; `nothing` skips this pass"
    alg_zipdown::D
    "if `true`, zip up from left to right and truncate from right to left, and vice versa"
    left_to_right::Bool
end

function Zipup(alg_zipup, alg_zipdown = nothing; left_to_right::Bool = true)
    return Zipup(alg_zipup, alg_zipdown, left_to_right)
end

function Zipup(; trunc, alg_svd = Defaults.alg_svd(), left_to_right::Bool = true)
    if trunc isa TruncationStrategy
        return Zipup(MatrixAlgebraKit.TruncatedAlgorithm(alg_svd, trunc); left_to_right)
    elseif trunc isa Tuple{<:TruncationStrategy, <:TruncationStrategy}
        alg_zipup = MatrixAlgebraKit.TruncatedAlgorithm(alg_svd, trunc[1])
        alg_zipdown = MatrixAlgebraKit.TruncatedAlgorithm(alg_svd, trunc[2])
        return Zipup(alg_zipup, alg_zipdown; left_to_right)
    else
        throw(ArgumentError("`trunc` should be a truncation strategy or a tuple of two truncation strategies"))
    end
end

function approximate!(ψ::FiniteMPS, (O, ϕ)::Tuple{Any, <:FiniteMPS}, alg::Zipup)
    N = check_length(ψ, O, ϕ)
    T = TensorOperations.promote_contract(scalartype(O), scalartype(ϕ))
    promote_type(T, scalartype(ψ)) === scalartype(ψ) ||
        throw(ArgumentError("destination state with scalartype $(scalartype(ψ)) cannot hold the result with scalartype $T"))
    for i in 1:N
        physicalspace(ϕ, i) == _input_physicalspace(O[i]) ||
            throw(SpaceMismatch("MPO input physical space does not match MPS physical space at site $i"))
    end

    return if alg.left_to_right
        zip_left_right!(ψ, O, ϕ, alg.alg_zipup, alg.alg_zipdown)
    else
        zip_right_left!(ψ, O, ϕ, alg.alg_zipup, alg.alg_zipdown)
    end
end

function approximate(Oϕ::Tuple{Any, <:FiniteMPS}, alg::Zipup)
    O, ϕ = Oϕ
    T = TensorOperations.promote_contract(scalartype(O), scalartype(ϕ))
    return approximate!(similar(ϕ, T), Oϕ, alg)
end

@doc """
    zip_left_right!(ψ, O, ϕ, alg_zipup, [alg_zipdown]) -> ψ, info
    zip_right_left!(ψ, O, ϕ, alg_zipup, [alg_zipdown]) -> ψ, info

Contract the MPO `O` with the MPS `ϕ` in a single sweep, truncating the enlarged virtual bond at every
site with `alg_zipup`, and write the result into `ψ`. `zip_left_right!` zips up from left to right,
`zip_right_left!` from right to left. Unless `alg_zipdown` is `nothing`, a second sweep in the
opposite direction imposes a final truncation with `alg_zipdown` in a locally gauged basis, leaving
the gauge center of `ψ` at the far end. The destination may alias `ϕ`.

Also returns an [`AlgorithmInfo`](@ref) describing the truncation. Being a single sweep
rather than an iterative optimisation, there is no convergence measure, so it reports neither
`converged` nor any convergence entry; [`convergence_measure`](@ref) returns `nothing` for it.
"""
zip_left_right!
@doc (@doc zip_left_right!) zip_right_left!

function zip_left_right!(ψ::FiniteMPS, O, ϕ::FiniteMPS, alg_zipup, alg_zipdown = nothing)
    N = length(ψ)

    # obtain all input tensors before overwriting the destination, such that `ψ === ϕ` is allowed:
    # from here on, the input is only queried through `Aϕs`, never through `ϕ` itself
    Aϕs = map(i -> i == 1 ? ϕ.AC[1] : ϕ.AR[i], 1:N)

    # the sweep re-derives the entire state: discard all cached tensors, as their spaces are stale
    # TODO: "reallocate" tensors?"
    foreach(f -> fill!(f, missing), (ψ.ALs, ψ.ARs, ψ.ACs, ψ.Cs))

    A = storagetype(eltype(ψ))
    Fₗ = fuser(A, left_virtualspace(Aϕs[1]), left_virtualspace(O, 1))
    acc = TruncationAccumulator(ψ)

    # zip up from left to right, leaving the gauge center on the last site
    for i in 1:(N - 1)
        Aᶻ = _fuse_mpo_mps_left(O[i], Aϕs[i], Fₗ)
        AL, Fₗ, ϵᵢ = left_gauge(Aᶻ, alg_zipup) # right factor doubles as the next left fuser
        ψ.ALs[i] = AL
        push_error!(acc, ϵᵢ)
    end
    Fᵣ = fuser(A, right_virtualspace(Aϕs[N]), right_virtualspace(O, N))
    ψ.ACs[N] = _fuse_mpo_mps(O[N], Aϕs[N], Fₗ, Fᵣ)

    # zip down from right to left, truncating in a locally gauged basis
    if !isnothing(alg_zipdown)
        for i in N:-1:2
            ψ, ϵᵢ = right_gauge!(ψ, i, ψ.AC[i], alg_zipdown)
            push_error!(acc, ϵᵢ)
        end
    end

    return ψ, AlgorithmInfo(; truncation = acc)
end

function zip_right_left!(ψ::FiniteMPS, O, ϕ::FiniteMPS, alg_zipup, alg_zipdown = nothing)
    N = length(ψ)

    Aϕs = map(i -> i == N ? ϕ.AC[N] : ϕ.AL[i], 1:N)
    foreach(f -> fill!(f, missing), (ψ.ALs, ψ.ARs, ψ.ACs, ψ.Cs))

    A = storagetype(eltype(ψ))
    # the right-hand fusers are oriented as `(Vmps ⊗ Vmpo) ← Vfused`, matching the factor that
    # replaces them on the next site
    Vᵣ = right_virtualspace(Aϕs[N]) ⊗ right_virtualspace(O, N)
    Fᵣ = isomorphism(A, Vᵣ, fuse(Vᵣ))
    acc = TruncationAccumulator(ψ)

    # zip up from right to left, leaving the gauge center on the first site
    for i in N:-1:2
        Aᶻ = _fuse_mpo_mps_right(O[i], Aϕs[i], Fᵣ)
        Fᵣ, AR, ϵᵢ = _right_gauge_zip(Aᶻ, alg_zipup) # left factor doubles as the next right fuser
        ψ.ARs[i] = AR
        push_error!(acc, ϵᵢ)
    end
    # the carry is oriented such that it can simply be composed with the last local tensor
    Fₗ = fuser(A, left_virtualspace(Aϕs[1]), left_virtualspace(O, 1))
    ψ.ACs[1] = _fuse_mpo_mps_left(O[1], Aϕs[1], Fₗ) * Fᵣ

    # zip down from left to right, truncating in a locally gauged basis
    if !isnothing(alg_zipdown)
        for i in 1:(N - 1)
            ψ, ϵᵢ = left_gauge!(ψ, i, ψ.AC[i], alg_zipdown)
            push_error!(acc, ϵᵢ)
        end
    end

    return ψ, AlgorithmInfo(; truncation = acc)
end

# `right_gauge` for the tensors of a right-to-left zip-up sweep: these are already partitioned across
# the new bond, so the leg permutation that `right_gauge` applies to MPS tensors has to be skipped
function _right_gauge_zip(Aᶻ, alg::MatrixAlgebraKit.TruncatedAlgorithm)
    U, S, Vᴴ, ϵ = svd_trunc(Aᶻ, alg)
    C = LinearAlgebra.rmul!(U, S) # C = U * S, matching `RightOrthViaSVD`
    return C, _transpose_front(Vᴴ), ϵ
end
