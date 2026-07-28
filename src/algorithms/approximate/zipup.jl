"""
$(TYPEDEF)

Algorithm that approximates an open-boundary finite MPO-MPS product using a left-to-right
zip-up sweep, optionally followed by a second `changebonds` sweep. The MPO and MPS are
contracted one site at a time, and the enlarged virtual bond is truncated immediately.

## Fields

$(TYPEDFIELDS)

## Constructors

    Zipup(; trscheme, alg_svd=Defaults.alg_svd())
    Zipup(alg_zipup, [alg_finalize])

Create a `Zipup` algorithm with the given [`SvdCut`](@ref), or by passing a truncation scheme
and singular value decomposition algorithm. If `alg_finalize` is provided, the state obtained
after the zip-up sweep is further compressed with `changebonds!`.

Following Paeckel et al., if the desired final bond dimension is `D`, one can use a more
permissive zip-up truncation, e.g. rank `2D` with stricter tolerances, and use `alg_finalize`
to impose the final truncation.

## References

- [Stoudenmire and White New J. Phys. 12 (2010)](@cite stoudenmire2010)
- [Paeckel et al. Ann. of Phys. 411 (2019)](@cite paeckel2019)
"""
struct Zipup{G <: SvdCut, F} <: Algorithm
    "algorithm used for gauging and truncating the local tensors during the zip-up sweep"
    alg_zipup::G
    "algorithm used for the final locally gauged truncation pass; `nothing` skips this pass"
    alg_finalize::F
end

Zipup(alg_zipup) = Zipup(alg_zipup, nothing)

function Zipup(;
        trscheme::TruncationStrategy, alg_svd = Defaults.alg_svd()
    )
    return Zipup(SvdCut(; alg_svd, trscheme))
end

function approximate((O, ψ)::Tuple{Any, <:FiniteMPS}, alg::Zipup)
    N = check_length(O, ψ)
    if !isunitspace(left_virtualspace(O, 1)) || !isunitspace(right_virtualspace(O, N))
        throw(ArgumentError("Zipup is only implemented for open-boundary MPOs"))
    end

    T = TensorOperations.promote_contract(scalartype(O), scalartype(ψ))
    A = TensorKit.similarstoragetype(eltype(ψ), T)

    Fₗ = fuser(A, left_virtualspace(ψ, 1), left_virtualspace(O, 1))
    local carry
    alg_zipup = MatrixAlgebraKit.TruncatedAlgorithm(
        alg.alg_zipup.alg_svd, alg.alg_zipup.trscheme
    )

    As = map(1:N) do i
        Aψ = i == 1 ? ψ.AC[1] : ψ.AR[i]
        physicalspace(Aψ) == physicalspace(O[i]) ||
            throw(SpaceMismatch("MPO input physical space does not match MPS physical space at site $i"))
        Fᵣ = fuser(A, right_virtualspace(ψ, i), right_virtualspace(O, i))
        Aᶻ = _fuse_mpo_mps(O[i], Aψ, Fₗ, Fᵣ)
        i > 1 && (Aᶻ = _mul_front(carry, Aᶻ))

        if i == N
            return Aᶻ
        else
            AL, C, _ = left_gauge(Aᶻ, alg_zipup)
            carry = C
            Fₗ = Fᵣ
            return AL
        end
    end

    ψ′ = FiniteMPS(As; normalize = false, overwrite = true)
    return isnothing(alg.alg_finalize) ?
        ψ′ : changebonds!(ψ′, alg.alg_finalize; normalize = false)
end
