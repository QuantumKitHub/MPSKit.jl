"""
$(TYPEDEF)

Algorithm that approximates an open-boundary finite MPO-MPS product using a left-to-right
zipper sweep. The MPO and MPS are contracted one site at a time, and the enlarged virtual
bond is truncated immediately using `trscheme`.

## Fields

$(TYPEDFIELDS)

## References

* [Sinha et al. Phys. Rev. B 109 (2024)](@cite sinha2024)
"""
@kwdef struct Zipper{S} <: Algorithm
    "algorithm used for the singular value decomposition"
    alg_svd::S = Defaults.alg_svd()

    "algorithm used for truncation of the local gauge tensors"
    trscheme::TruncationStrategy
end

function approximate((O, ψ)::Tuple{Any, <:FiniteMPS}, alg::Zipper)
    N = check_length(O, ψ)
    if !isunitspace(left_virtualspace(O, 1)) || !isunitspace(right_virtualspace(O, N))
        throw(ArgumentError("Zipper is only implemented for open-boundary MPOs"))
    end

    T = TensorOperations.promote_contract(scalartype(O), scalartype(ψ))
    A = TensorKit.similarstoragetype(eltype(ψ), T)

    Fₗ = fuser(A, left_virtualspace(ψ, 1), left_virtualspace(O, 1))
    local carry
    alg_gauge = MatrixAlgebraKit.TruncatedAlgorithm(alg.alg_svd, alg.trscheme)

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
            AL, C, _ = left_gauge(Aᶻ, alg_gauge)
            carry = C
            Fₗ = Fᵣ
            return AL
        end
    end

    return FiniteMPS(As; normalize = false, overwrite = true)
end
