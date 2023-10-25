"""
    changebonds(ψ::AbstractMPS, H, alg, envs) -> ψ′, envs′
    changebonds(ψ::AbstractMPS, alg) -> ψ′

Change the bond dimension of `ψ` using the algorithm `alg`, and return the new `ψ` and the new `envs`.

See also: [`SvdCut`](@ref), [`RandExpand`](@ref), [`VUMPSSvdCut`](@ref), [`OptimalExpand`](@ref)
"""
function changebonds end
function changebonds! end

_expand(ψ, AL′, AR′) = _expand!(copy(ψ), AL′, AR′)
function _expand!(ψ::InfiniteMPS, AL′::PeriodicVector, AR′::PeriodicVector)
    for i in 1:length(ψ)
        # update AL: add vectors, make room for new vectors:
        # AL -> [AL expansion; 0 0]
        al′ = _transpose_tail(catdomain(ψ.AL[i], AL′[i]))
        lz = zerovector!(similar(al′, _lastspace(AL′[i - 1])' ← domain(al′)))
        ψ.AL[i] = _transpose_front(catcodomain(al′, lz))

        # update AR: add vectors, make room for new vectors:
        # AR -> [AR 0; expansion 0]
        ar′ = _transpose_front(catcodomain(_transpose_tail(ψ.AR[i + 1]), AR′[i + 1]))
        rz = zerovector!(similar(ar′, codomain(ar′) ← _firstspace(AR′[i + 2])))
        ψ.AR[i + 1] = catdomain(ar′, rz)

        # update C: add vectors, make room for new vectors:
        # C -> [C 0; 0 expansion]
        l = zerovector!(similar(ψ.CR[i], codomain(ψ.CR[i]) ← _firstspace(AR′[i + 1])))
        ψ.CR[i] = catdomain(ψ.CR[i], l)
        r = zerovector!(similar(ψ.CR[i], _lastspace(AL′[i])' ← domain(ψ.CR[i])))
        ψ.CR[i] = catcodomain(ψ.CR[i], r)

        # update AC: recalculate
        ψ.AC[i] = ψ.AL[i] * ψ.CR[i]
    end
    return normalize!(ψ)
end
function _expand!(ψ::MPSMultiline, AL′::PeriodicMatrix, AR′::PeriodicMatrix)
    for i in 1:size(ψ, 1)
        _expand!(ψ[i], AL′[i, :], AR′[i, :])
    end
    return ψ
end
