"""
$(TYPEDEF)

Algorithm that approximates an open-boundary finite MPO-MPS product using a left-to-right
zip-up sweep. The MPO and MPS are contracted one site at a time, and the enlarged virtual
bond is truncated immediately using `trscheme`.

## Fields

$(TYPEDFIELDS)

## Constructors

    Zipup(; trscheme, alg_svd=Defaults.alg_svd())
    Zipup(alg_gauge)

Create a `Zipup` algorithm with the given truncated gauge algorithm, or by passing a
truncation scheme and singular value decomposition algorithm.

## References

- [Stoudenmire and White New J. Phys. 12 (2010)](@cite stoudenmire2010)
"""
struct Zipup{G} <: Algorithm
    "algorithm used for gauging and truncating the local tensors"
    alg_gauge::G
end

function Zipup(; trscheme::TruncationStrategy, alg_svd = Defaults.alg_svd())
    return Zipup(MatrixAlgebraKit.TruncatedAlgorithm(alg_svd, trscheme))
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
            AL, C, _ = left_gauge(Aᶻ, alg.alg_gauge)
            carry = C
            Fₗ = Fᵣ
            return AL
        end
    end

    return FiniteMPS(As; normalize = false, overwrite = true)
end
