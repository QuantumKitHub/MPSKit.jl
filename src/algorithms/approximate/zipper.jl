"""
    Zipper(; alg_svd=Defaults.alg_svd(), trscheme)

Algorithm that approximates an open-boundary finite MPO-MPS product using a right-to-left
zipper sweep. The MPO and MPS are contracted one site at a time, and the enlarged virtual
bond is truncated immediately using `trscheme`.

Use as:

    approximate((O, ψ), Zipper(; trscheme))

This returns an unnormalized compression of `O * ψ`, comparable to
`changebonds(O * ψ, SvdCut(; trscheme); normalize=false)`, but without storing
the fully enlarged product on every site.
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

    ψ′ = copy(ψ)
    T = TensorOperations.promote_contract(scalartype(O), scalartype(ψ))
    A = TensorKit.similarstoragetype(eltype(ψ), T)

    Fᵣ = fuser(A, right_virtualspace(ψ′, N), right_virtualspace(O, N))
    local carry
    alg_gauge = MatrixAlgebraKit.TruncatedAlgorithm(alg.alg_svd, alg.trscheme)

    As = map(N:-1:1) do i
        Aψ = i == 1 ? ψ′.AC[1] : ψ′.AR[i]
        physicalspace(Aψ) == physicalspace(O[i]) ||
            throw(SpaceMismatch("MPO input physical space does not match MPS physical space at site $i"))
        Fₗ = fuser(A, left_virtualspace(ψ′, i), left_virtualspace(O, i))
        Aᶻ = _fuse_mpo_mps(O[i], Aψ, Fₗ, Fᵣ)
        i < N && (Aᶻ = Aᶻ * carry)

        if i == 1
            return Aᶻ
        else
            C, AR = right_gauge(Aᶻ, alg_gauge)
            carry = C
            Fᵣ = Fₗ
            return AR
        end
    end

    return FiniteMPS(reverse(As); normalize = false, overwrite = true)
end
