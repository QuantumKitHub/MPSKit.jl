"""
$(TYPEDEF)

Algorithm that approximates an open-boundary finite MPO-MPS product using a left-to-right
zip-up sweep, optionally followed by a right-to-left zip-down sweep. The MPO and MPS are
contracted one site at a time, and the enlarged virtual bond is truncated immediately.

## Fields

$(TYPEDFIELDS)

## Constructors

    Zipup(; trscheme, alg_svd=Defaults.alg_svd())
    Zipup(alg_zipup, [alg_zipdown])

Create a `Zipup` algorithm with the given truncated gauge algorithm, or by passing a truncation
scheme and singular value decomposition algorithm. If `alg_zipdown` is provided, the state
obtained after the zip-up sweep is further compressed by sweeping back from right to left.

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

function Zipup(;
        trscheme::TruncationStrategy, alg_svd = Defaults.alg_svd()
    )
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
            AL, C, _ = left_gauge(Aᶻ, alg.alg_zipup)
            carry = C
            Fₗ = Fᵣ
            return AL
        end
    end

    return isnothing(alg.alg_zipdown) ?
        FiniteMPS(As; normalize = false, overwrite = true) :
        _zipdown(As, alg.alg_zipdown)
end

function _zipdown(As::Vector{A}, alg::MatrixAlgebraKit.TruncatedAlgorithm) where {A}
    N = length(As)
    N == 1 && return FiniteMPS(As; normalize = false, overwrite = true)

    ARs = Vector{Union{Missing, A}}(missing, N)
    ALs = Vector{Union{Missing, A}}(missing, N)
    ACs = Vector{Union{Missing, A}}(missing, N)

    local C
    AC = As[N]
    for i in N:-1:2
        C, AR, _ = right_gauge(AC, alg)
        ARs[i] = AR
        AC = _mul_tail(As[i - 1], C)
    end

    B = typeof(C)
    ACs[1] = AC
    Cs = Vector{Union{Missing, B}}(missing, N + 1)
    return FiniteMPS(ALs, ARs, ACs, Cs)
end
