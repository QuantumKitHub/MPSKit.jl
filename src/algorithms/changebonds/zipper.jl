@doc """
    zipper(O::FiniteMPO{<:GenericMPOTensor}, ψ::FiniteMPS, trscheme; alg_svd=Defaults.alg_svd()) -> ψ′
    zipper(O::FiniteMPO{<:GenericMPOTensor}, ψ::FiniteMPS, alg::SvdCut) -> ψ′

Apply a finite open-boundary MPO `O` to a finite MPS `ψ` using a right-to-left
zipper sweep. The MPO and MPS are contracted one site at a time, and the enlarged
virtual bond is truncated immediately using `trscheme`.

This is an unnormalized compression of `O * ψ`, comparable to
`changebonds(O * ψ, SvdCut(; trscheme); normalize=false)`, but without storing
the fully enlarged product on every site.
"""
function zipper end

function _mpo_input_physicalspace(O)
    return prod(x -> dual(space(O, x)), (numout(O) + 1):(numind(O) - 1))
end

function zipper(
        O::FiniteMPO{<:GenericMPOTensor}, ψ::FiniteMPS, trscheme::TruncationStrategy;
        alg_svd = Defaults.alg_svd()
    )
    return zipper(O, ψ, SvdCut(; alg_svd, trscheme))
end

function zipper(O::FiniteMPO{<:GenericMPOTensor}, ψ::FiniteMPS, alg::SvdCut)
    N = check_length(O, ψ)
    if !isunitspace(left_virtualspace(O, 1)) || !isunitspace(right_virtualspace(O, N))
        throw(ArgumentError("zipper is only implemented for open-boundary MPOs"))
    end

    ψ′ = copy(ψ)
    T = TensorOperations.promote_contract(scalartype(O), scalartype(ψ))
    A = TensorKit.similarstoragetype(eltype(ψ), T)

    Fᵣ = fuser(A, right_virtualspace(ψ′, N), right_virtualspace(O, N))
    local carry
    alg_gauge = MatrixAlgebraKit.TruncatedAlgorithm(alg.alg_svd, alg.trscheme)

    As = map(N:-1:1) do i
        Aψ = i == 1 ? ψ′.AC[1] : ψ′.AR[i]
        physicalspace(Aψ) == _mpo_input_physicalspace(O[i]) ||
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
