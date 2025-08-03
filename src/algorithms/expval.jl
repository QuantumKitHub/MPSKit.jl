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
* `environments::AbstractMPSEnvironments` : the environments to use for the calculation. If not given, they will be calculated.

# Examples
```jldoctest
julia> ψ = FiniteMPS(ones(Float64, (ℂ^2)^4));

julia> S_x = TensorMap(Float64[0 1; 1 0], ℂ^2, ℂ^2);

julia> round(expectation_value(ψ, 2 => S_x))
1.0

julia> round(expectation_value(ψ, (2, 3) => S_x ⊗ S_x))
1.0
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
            physicalspace(ψ, site) == physicalspace(o) ||
                throw(SpaceMismatch("physical space does not match at site $site"))
        end
    end

    Ut = fill_data!(similar(local_mpo[1], _firstspace(first(local_mpo))), one)

    # some special cases that avoid using transfer matrices
    if length(sites) == 1
        AC = ψ.AC[sites[1]]
        E = @plansor conj(AC[4 5; 6]) * conj(Ut[1]) * local_mpo[1][1 5; 3 2] * Ut[2] *
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
        @plansor Vl[-1 -2; -3] := isomorphism(
            T, left_virtualspace(ψ, sites[1]), left_virtualspace(ψ, sites[1])
        )[-1; -3] *
            conj(Ut[-2])

        # middle
        M = foldl(zip(sites, local_mpo); init = Vl) do v, (site, o)
            if o isa Number
                return scale!(v * TransferMatrix(ψ.AL[site], ψ.AL[site]), o)
            else
                return v * TransferMatrix(ψ.AL[site], o, ψ.AL[site])
            end
        end

        # right side
        E = @plansor M[1 2; 3] * Ut[2] * ψ.C[sites[end]][3; 4] *
            conj(ψ.C[sites[end]][1; 4])
    end

    return E / norm(ψ)^2
end

# MPOHamiltonian
# --------------
function contract_mpo_expval(
        AC::MPSTensor, GL::MPSTensor, O::MPOTensor, GR::MPSTensor, ACbar::MPSTensor = AC
    )
    return @plansor GL[1 2; 3] * AC[3 7; 5] * GR[5 8; 6] * O[2 4; 7 8] * conj(ACbar[1 4; 6])
end

function expectation_value(
        ψ::FiniteMPS, H::FiniteMPOHamiltonian,
        envs::AbstractMPSEnvironments = environments(ψ, H)
    )
    return dot(ψ, H, ψ, envs) / dot(ψ, ψ)
end

function expectation_value(
        ψ::InfiniteMPS, H::InfiniteMPOHamiltonian,
        envs::AbstractMPSEnvironments = environments(ψ, H)
    )
    return sum(1:length(ψ)) do i
        return contract_mpo_expval(
            ψ.AC[i], envs.GLs[i], H[i][:, 1, 1, end], envs.GRs[i][end]
        )
    end
end

# DenseMPO
# --------
function expectation_value(ψ::FiniteMPS, mpo::FiniteMPO)
    return dot(ψ, mpo, ψ) / dot(ψ, ψ)
end
function expectation_value(ψ::FiniteQP, mpo::FiniteMPO)
    return expectation_value(convert(FiniteMPS, ψ), mpo)
end
function expectation_value(ψ::InfiniteMPS, mpo::InfiniteMPO, envs...)
    return expectation_value(convert(MultilineMPS, ψ), convert(MultilineMPO, mpo), envs...)
end
function expectation_value(
        ψ::MultilineMPS, O::MultilineMPO{<:InfiniteMPO},
        envs::MultilineEnvironments = environments(ψ, O)
    )
    return prod(product(1:size(ψ, 1), 1:size(ψ, 2))) do (i, j)
        GL = envs[i].GLs[j]
        GR = envs[i].GRs[j]
        return contract_mpo_expval(ψ.AC[i, j], GL, O[i, j], GR, ψ.AC[i + 1, j])
    end
end
function expectation_value(ψ::MultilineMPS, mpo::MultilineMPO, envs...)
    # TODO: fix environments
    return prod(x -> expectation_value(x...), zip(parent(ψ), parent(mpo)))
end
# fallback
function expectation_value(ψ::AbstractMPS, mpo::AbstractMPO, envs...)
    return dot(ψ, mpo, ψ) / dot(ψ, ψ)
end

# Lazy operators
# --------------
function expectation_value(ψ, op::UntimedOperator, args...)
    return op.f * expectation_value(ψ, op.op, args...)
end

function expectation_value(ψ, ops::LazySum)
    return sum(op -> expectation_value(ψ, op), ops.ops)
end
function expectation_value(ψ, ops::LazySum, envs::MultipleEnvironments)
    return sum(((op, env),) -> expectation_value(ψ, op, env), zip(ops.ops, envs))
end

# for now we also have LinearCombination
function expectation_value(ψ, H::LinearCombination, envs::LazyLincoCache = environments(ψ, H))
    return sum(
        ((c, op, env),) -> c * expectation_value(ψ, op, env),
        zip(H.coeffs, H.opps, envs.envs)
    )
end

# ProjectionOperator
# ------------------
function expectation_value(
        ψ::FiniteMPS, O::ProjectionOperator,
        envs::FiniteEnvironments = environments(ψ, O)
    )
    ens = zeros(scalartype(ψ), length(ψ))
    for i in 1:length(ψ)
        operator = AC_hamiltonian(i, ψ, O, ψ, envs)
        ens[i] = dot(ψ.AC[i], operator * ψ.AC[i])
    end
    n = norm(ψ.AC[end])^2
    return sum(ens) / (n * length(ψ))
end
