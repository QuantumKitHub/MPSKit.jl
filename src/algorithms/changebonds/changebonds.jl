@doc """
    changebonds(ψ::AbstractMPS, H, alg, envs) -> ψ′, envs′
    changebonds(ψ::AbstractMPS, alg) -> ψ′

Change the bond dimension of `ψ` using the algorithm `alg`, and return the new `ψ` and the new `envs`.
For AbstractInfiniteMPS, changebonds returns new environments without modifying the one provided.
changedbonds! can modifiy both the provided state and environments, depending on the algorithm.
For FiniteMPS, changebonds also modifies the environments.

See also: [`SvdCut`](@ref), [`RandExpand`](@ref), [`VUMPSSvdCut`](@ref), [`OptimalExpand`](@ref)
""" changebonds, changebonds!
function changebonds end
function changebonds! end

@doc """
    changebond(site, dir, ψ, [H], alg, [envs]) -> ψ
    changebond!(site, dir, ψ, [H], alg, [envs]) -> ψ

Expand a single bond of `ψ` in place by adding directions orthogonal to the current state, keeping the state in mixed-canonical form around the enriched bond.
The sweep direction `dir` is a `Val(:right)` or `Val(:left)` used for dispatch.
For `Val(:right)` the bond `(site, site + 1)` is enriched on the right tensor (`ψ.AR[site + 1]`) with zero weight added at `ψ.AC[site]`, so that a subsequent single-site optimization of `site` sees the new directions;
for `Val(:left)` the mirror is applied to bond `(site - 1, site)`.

See also [`changebonds`](@ref), [`changebonds!`](@ref).
""" changebond, changebond!
function changebond end
function changebond! end

_expand(ψ, AL′, AR′) = _expand!(copy(ψ), AL′, AR′)
function _expand!(ψ::InfiniteMPS, AL′::PeriodicVector, AR′::PeriodicVector)
    for i in 1:length(ψ)
        # update AL: add vectors, make room for new vectors:
        # AL -> [AL expansion; 0 0]
        al′ = _transpose_tail(catdomain(ψ.AL[i], AL′[i]))
        al_space = (codomain(al′)[1] ⊕ _lastspace(AL′[i - 1])') ← domain(al′)
        ψ.AL[i] = _transpose_front(absorb!(zerovector!(similar(al′, al_space)), al′))

        # update AR: add vectors, make room for new vectors:
        # AR -> [AR 0; expansion 0]
        ar′ = _transpose_front(catcodomain(_transpose_tail(ψ.AR[i + 1]), AR′[i + 1]))
        ar_space = codomain(ar′) ← (domain(ar′)[1] ⊕ _firstspace(AR′[i + 2]))
        ψ.AR[i + 1] = absorb!(zerovector!(similar(ar′, ar_space)), ar′)

        # update C: add vectors, make room for new vectors:
        # C -> [C 0; 0 expansion]
        c_dom = codomain(ψ.C[i]) ← (domain(ψ.C[i])[1] ⊕ _firstspace(AR′[i + 1]))
        ψ.C[i] = absorb!(zerovector!(similar(ψ.C[i], c_dom)), ψ.C[i])
        c_cod = (codomain(ψ.C[i])[1] ⊕ _lastspace(AL′[i])') ← domain(ψ.C[i])
        ψ.C[i] = absorb!(zerovector!(similar(ψ.C[i], c_cod)), ψ.C[i])

        # update AC: recalculate
        ψ.AC[i] = ψ.AL[i] * ψ.C[i]
    end
    return normalize!(ψ)
end
function _expand!(ψ::MultilineMPS, AL′::PeriodicMatrix, AR′::PeriodicMatrix)
    for i in 1:size(ψ, 1)
        _expand!(ψ[i], AL′[i, :], AR′[i, :])
    end
    return ψ
end

function changebond(site::Int, dir::Val, ψ::AbstractFiniteMPS, H, alg::Algorithm, envs)
    return changebond!(site, dir, copy(ψ), H, alg, envs)
end
