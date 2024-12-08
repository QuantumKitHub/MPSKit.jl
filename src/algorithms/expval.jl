"""
    expectation_value(ψ, O, [environments])
    expectation_value(ψ, inds => O)

Compute the expectation value of an operator `O` on a state `ψ`. 
Optionally, it is possible to make the computations more efficient by also passing in
previously calculated `environments`.

In general, the operator `O` may consist of an arbitrary MPO `O <: AbstractMPO` that acts
on all sites, or a local operator `O = inds => operator` acting on a subset of sites. 
In the latter case, `inds` is a tuple of indices that specify the sites on which the operator
acts, while the operator is either a `AbstractTensorMap` or a `FiniteMPO`.

# Arguments
* `ψ::AbstractMPS` : the state on which to compute the expectation value
* `O::Union{AbstractMPO,Pair}` : the operator to compute the expectation value of. 
    This can either be an `AbstractMPO`, or a pair of indices and local operator..
* `environments::Cache` : the environments to use for the calculation. If not given, they will be calculated.

# Examples
```jldoctest
julia> ψ = FiniteMPS(ones, Float64, 4, ℂ^2, ℂ^3);

julia> S_x = TensorMap(Float64[0 1; 1 0], ℂ^2, ℂ^2);

julia> round(expectation_value(ψ, 2 => S_x))
1.0

julia> round(expectation_value(ψ, (2, 3) => S_x ⊗ S_x))
1.0

julia> round(expectation_value(ψ, MPOHamiltonian(S_x)))
4.0
```
"""
function expectation_value end

# Local operators
# ---------------
function expectation_value(ψ::AbstractMPS, (inds, O)::Pair)
    @boundscheck foreach(Base.Fix1(Base.checkbounds, ψ), inds)

    sites, local_mpo = instantiate_operator(physicalspace(ψ), inds => O)
    @assert _firstspace(first(local_mpo)) == oneunit(_firstspace(first(local_mpo))) ==
            dual(_lastspace(last(local_mpo)))
    for (site, o) in zip(sites, local_mpo)
        if o isa MPOTensor
            physicalspace(ψ)[site] == physicalspace(o) ||
                throw(SpaceMismatch("physical space does not match at site $site"))
        end
    end

    Ut = fill_data!(similar(local_mpo[1], _firstspace(first(local_mpo))), one)

    # some special cases that avoid using transfer matrices
    if length(sites) == 1
        AC = ψ.AC[sites[1]]
        E = @plansor conj(AC[4 5; 6]) *
                     conj(Ut[1]) * local_mpo[1][1 5; 3 2] * Ut[2] *
                     AC[4 3; 6]
    elseif length(sites) == 2 && (sites[1] + 1 == sites[2])
        AC = ψ.AC[sites[1]]
        AR = ψ.AR[sites[2]]
        O1, O2 = local_mpo
        E = @plansor conj(AC[4 5; 10]) * conj(Ut[1]) * O1[1 5; 3 8] * AC[4 3; 6] *
                     conj(AR[10 9; 11]) * Ut[2] * O2[8 9; 7 2] * AR[6 7; 11]
    else
        # generic case: write as Vl * T^N * Vr
        # left side
        T = storagetype(site_type(ψ))
        @plansor Vl[-1 -2; -3] := isomorphism(T,
                                              left_virtualspace(ψ, sites[1] - 1),
                                              left_virtualspace(ψ, sites[1] - 1))[-1; -3] *
                                  conj(Ut[-2])

        # middle
        M = foldl(zip(sites, local_mpo); init=Vl) do v, (site, o)
            if o isa Number
                return scale!(v * TransferMatrix(ψ.AL[site], ψ.AL[site]), o)
            else
                return v * TransferMatrix(ψ.AL[site], o, ψ.AL[site])
            end
        end

        # right side
        E = @plansor M[1 2; 3] * Ut[2] * ψ.CR[sites[end]][3; 4] *
                     conj(ψ.CR[sites[end]][1; 4])
    end

    return E / norm(ψ)^2
end

# MPOHamiltonian
# --------------
function expectation_value(ψ::FiniteMPS, H::MPOHamiltonian,
                           envs::Cache=environments(ψ, H))
    L = length(ψ) ÷ 2
    GL = leftenv(envs, L, ψ)
    GR = rightenv(envs, L, ψ)
    AC = ψ.AC[L]
    E = sum(keys(H[L])) do (j, k)
        return @plansor GL[j][1 2; 3] * AC[3 7; 5] * GR[k][5 8; 6] * conj(AC[1 4; 6]) *
                        H[L][j, k][2 4; 7 8]
    end
    return E / norm(ψ)^2
end
function expectation_value(ψ::FiniteQP, H::MPOHamiltonian)
    return expectation_value(convert(FiniteMPS, ψ), H)
end
function expectation_value(ψ::InfiniteMPS, H::MPOHamiltonian,
                           envs::Cache=environments(ψ, H))
    # TODO: this presumably could be done more efficiently
    return sum(1:length(ψ)) do i
        return sum((H.odim):-1:1) do j
            ρ_LL = r_LL(ψ, i)
            util = fill_data!(similar(ψ.AL[1], space(envs.lw[H.odim, i + 1], 2)), one)
            GL = leftenv(envs, i, ψ)
            return @plansor (GL[j] * TransferMatrix(ψ.AL[i], H[i][j, H.odim], ψ.AL[i]))[1 2;
                                                                                        3] *
                            ρ_LL[3; 1] * conj(util[2])
        end
    end
end

# no definition for WindowMPS -> not well defined

# DenseMPO
# --------
function expectation_value(ψ::FiniteMPS, mpo::FiniteMPO)
    return dot(ψ, mpo, ψ) / dot(ψ, ψ)
end
function expectation_value(ψ::FiniteQP, mpo::FiniteMPO)
    return expectation_value(convert(FiniteMPS, ψ), mpo)
end
function expectation_value(ψ::InfiniteMPS, mpo::DenseMPO, envs...)
    return expectation_value(convert(MPSMultiline, ψ), convert(MPOMultiline, mpo), envs...)
end
function expectation_value(ψ::MPSMultiline, O::MPOMultiline,
                           envs::PerMPOInfEnv=environments(ψ, O))
    return prod(product(1:size(ψ, 1), 1:size(ψ, 2))) do (i, j)
        GL = leftenv(envs, i, j, ψ)
        GR = rightenv(envs, i, j, ψ)
        @plansor GL[1 2; 3] * O[i, j][2 4; 6 5] *
                 ψ.AC[i, j][3 6; 7] * GR[7 5; 8] *
                 conj(ψ.AC[i + 1, j][1 4; 8])
    end
end

# Lazy operators
# --------------
function expectation_value(ψ, op::UntimedOperator, args...)
    return op.f * expectation_value(ψ, op.op, args...)
end

function expectation_value(ψ, ops::LazySum, envs::MultipleEnvironments=environments(ψ, ops))
    return sum(((op, env),) -> expectation_value(ψ, op, env), zip(ops.ops, envs))
end

# Transfer matrices
# -----------------
# function expectation_value(ψ::InfiniteMPS, mpo::DenseMPO)
#     return expectation_value(convert(MPSMultiline, ψ), convert(MPOMultiline, mpo))
# end
# function expectation_value(ψ::MPSMultiline, mpo::MPOMultiline)
#     return expectation_value(ψ, environments(ψ, mpo))
# end
# function expectation_value(ψ::InfiniteMPS, ca::PerMPOInfEnv)
#     return expectation_value(convert(MPSMultiline, ψ), ca)
# end
# function expectation_value(ψ::MPSMultiline, O::MPOMultiline, ca::PerMPOInfEnv)
#     retval = PeriodicMatrix{scalartype(ψ)}(undef, size(ψ, 1), size(ψ, 2))
#     for (i, j) in product(1:size(ψ, 1), 1:size(ψ, 2))
#         retval[i, j] = @plansor leftenv(ca, i, j, ψ)[1 2; 3] * O[i, j][2 4; 6 5] *
#                                 ψ.AC[i, j][3 6; 7] * rightenv(ca, i, j, ψ)[7 5; 8] *
#                                 conj(ψ.AC[i + 1, j][1 4; 8])
#     end
#     return retval
# end

# for now we also have LinearCombination
function expectation_value(ψ, H::LinearCombination, envs::LazyLincoCache=environments(ψ, H))
    return sum(((c, op, env),) -> c * expectation_value(ψ, op, env),
               zip(H.coeffs, H.opps, envs.envs))
end

# ProjectionOperator
# ------------------
function expectation_value(ψ::FiniteMPS, O::ProjectionOperator,
                           envs::FinEnv=environments(ψ, O))
    ens = zeros(scalartype(ψ), length(ψ))
    for i in 1:length(ψ)
        operator = ∂∂AC(i, ψ, O, envs)
        ens[i] = dot(ψ.AC[i], operator * ψ.AC[i])
    end
    n = norm(ψ.AC[end])^2
    return sum(ens) / n
end
