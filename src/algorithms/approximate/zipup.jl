"""
$(TYPEDEF)

Algorithm that approximates an open-boundary finite MPO-MPS product using a left-to-right
zip-up sweep, optionally followed by a right-to-left zip-down sweep. The MPO and MPS are
contracted one site at a time, and the enlarged virtual bond is truncated immediately.

    approximate((O, ϕ), alg::Zipup) -> ψ, ϵ
    approximate!(ψ, (O, ϕ), alg::Zipup) -> ψ, ϵ

Contrary to the variational algorithms, this algorithm requires no initial guess: the in-place
version simply uses `ψ` as the destination of the sweep, overwriting its contents, and may alias
`ϕ`. The out-of-place version allocates a destination with the promoted scalar type of `O` and `ϕ`.
Both return the truncation error `ϵ` alongside the approximated state.

Only the input physical spaces of `O` have to match those of `ϕ`: for an `O` whose output physical
spaces differ, the result is a state with the output physical spaces of `O`.

## Fields

$(TYPEDFIELDS)

## Constructors

    Zipup(; trunc, alg_svd=Defaults.alg_svd())
    Zipup(alg_zipup, [alg_zipdown])

Create a `Zipup` algorithm with the given truncated gauge algorithm, or by passing a truncation
scheme and singular value decomposition algorithm. The keyword `trunc` can be either one
truncation strategy for a single zip-up sweep, or a tuple `(zipup_trunc, zipdown_trunc)` for a
zip-up sweep followed by a zip-down sweep. Equivalently, one can pass the corresponding truncated
gauge algorithms directly as `alg_zipup` and `alg_zipdown`.

Following Paeckel et al., if the desired final bond dimension is `D`, one can use a more
permissive zip-up truncation, e.g. rank `2D` with stricter tolerances, and use `alg_zipdown`
to impose the final truncation.

## References

- [Stoudenmire and White New J. Phys. 12 (2010)](@cite stoudenmire2010)
- [Paeckel et al. Ann. of Phys. 411 (2019)](@cite paeckel2019)
"""
struct Zipup{
        U <: MatrixAlgebraKit.TruncatedAlgorithm,
        D <: Union{Nothing, MatrixAlgebraKit.TruncatedAlgorithm},
    } <: Algorithm
    "algorithm used for gauging and truncating the local tensors during the zip-up sweep"
    alg_zipup::U
    "algorithm used for the final locally gauged truncation pass; `nothing` skips this pass"
    alg_zipdown::D
end

Zipup(alg_zipup) = Zipup(alg_zipup, nothing)

function Zipup(; trunc, alg_svd = Defaults.alg_svd())
    if trunc isa TruncationStrategy
        return Zipup(MatrixAlgebraKit.TruncatedAlgorithm(alg_svd, trunc))
    elseif trunc isa Tuple{<:TruncationStrategy, <:TruncationStrategy}
        alg_zipup = MatrixAlgebraKit.TruncatedAlgorithm(alg_svd, trunc[1])
        alg_zipdown = MatrixAlgebraKit.TruncatedAlgorithm(alg_svd, trunc[2])
        return Zipup(alg_zipup, alg_zipdown)
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

    ψ, ϵ = zipup!(ψ, O, ϕ, alg.alg_zipup)
    if !isnothing(alg.alg_zipdown)
        ψ, ϵ′ = zipdown!(ψ, alg.alg_zipdown)
        ϵ = max(ϵ, ϵ′)
    end
    return ψ, ϵ
end

function approximate(Oϕ::Tuple{Any, <:FiniteMPS}, alg::Zipup)
    O, ϕ = Oϕ
    T = TensorOperations.promote_contract(scalartype(O), scalartype(ϕ))
    return approximate!(similar(ϕ, T), Oϕ, alg)
end

"""
    zipup!(ψ, O, ϕ, alg) -> ψ, ϵ

Contract the MPO `O` with the MPS `ϕ` in a single left-to-right sweep, truncating the enlarged
virtual bond at every site with `alg`, and write the result into `ψ`. The destination is left with
its gauge center on the last site, and may alias `ϕ`.

Instead of fusing both virtual bonds of every site, only the left bond is fused: the right factor of
the truncated decomposition is simultaneously the truncation carry and the fuser of the next site, so
the enlarged object is never constructed.

Also returns the truncation error `ϵ`, the largest 2-norm of the discarded singular values over all
bonds.

Compatibility of the lengths, physical spaces and scalar types of `ψ`, `O` and `ϕ` is assumed, and
checked in [`approximate!`](@ref).
"""
function zipup!(ψ::FiniteMPS, O, ϕ::FiniteMPS, alg)
    N = length(ψ)

    # obtain all input tensors before overwriting the destination, such that `ψ === ϕ` is allowed:
    # from here on, the input is only queried through `Aϕs`, never through `ϕ` itself
    Aϕs = map(i -> i == 1 ? ϕ.AC[1] : ϕ.AR[i], 1:N)

    # the sweep re-derives the entire state: discard all cached tensors, as their spaces are stale
    # TODO: "reallocate" tensors?"
    foreach(f -> fill!(f, missing), (ψ.ALs, ψ.ARs, ψ.ACs, ψ.Cs))

    A = storagetype(eltype(ψ))
    Fₗ = fuser(A, left_virtualspace(Aϕs[1]), left_virtualspace(O, 1))
    ϵ = zero(real(scalartype(ψ)))
    for i in 1:(N - 1)
        Aᶻ = _fuse_mpo_mps_left(O[i], Aϕs[i], Fₗ)
        AL, Fₗ, ϵᵢ = left_gauge(Aᶻ, alg) # right factor doubles as the next left fuser
        ψ.ALs[i] = AL
        ϵ = max(ϵ, ϵᵢ)
    end
    Fᵣ = fuser(A, right_virtualspace(Aϕs[N]), right_virtualspace(O, N))
    ψ.ACs[N] = _fuse_mpo_mps(O[N], Aϕs[N], Fₗ, Fᵣ)

    return ψ, ϵ
end

"""
    zipdown!(ψ, alg) -> ψ, ϵ

Sweep `ψ` from right to left, truncating every bond with `alg` in a locally gauged basis, and moving
the gauge center to the leftmost bond in the process.

Also returns the truncation error `ϵ`, the largest 2-norm of the discarded singular values over all bonds.
"""
function zipdown!(ψ::AbstractFiniteMPS, alg)
    ϵ = zero(real(scalartype(ψ)))
    for i in length(ψ):-1:2
        ψ, ϵᵢ = right_gauge!(ψ, i, ψ.AC[i], alg)
        ϵ = max(ϵ, ϵᵢ)
    end
    return ψ, ϵ
end
