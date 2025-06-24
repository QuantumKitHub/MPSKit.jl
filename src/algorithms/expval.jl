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

    # special cases that can be handled more efficiently
    if length(inds) == 1
        return local_expectation_value1(ψ, inds[1], O)
    elseif length(inds) == 2 && eltype(inds) <: Integer && inds[1] + 1 == inds[2]
        return local_expectation_value2(ψ, inds[1], O)
    end

    # generic case: convert to MPO and write as Vl * T^N * Vr
    sites, local_mpo = instantiate_operator(ψ, inds => O)

    # left side
    Vl = insertrightunit(l_LL(ψ, sites[1]), 1; dual=true)

    # middle
    M = foldl(zip(sites, local_mpo); init=Vl) do v, (site, o)
        if o isa Number
            return scale!(v * TransferMatrix(ψ.AL[site], ψ.AL[site]), o)
        else
            return v * TransferMatrix(ψ.AL[site], o, ψ.AL[site])
        end
    end

    # right side
    E = @plansor removeunit(M, 2)[1; 2] * ψ.C[sites[end]][2; 3] *
                 conj(ψ.C[sites[end]][1; 3])
    return E / dot(ψ, ψ)
end

function local_expectation_value1(ψ::AbstractMPS, site, O)
    E = contract_mpo_expval1(ψ.AC[site], O, ψ.AC[site])
    return E / dot(ψ, ψ)
end
function local_expectation_value2(ψ::AbstractMPS, site, O)
    AC = ψ.AC[site]
    AR = ψ.AR[site + 1]
    E = contract_mpo_expval2(AC, AR, O, AC, AR)
    return E / dot(ψ, ψ)
end

# MPOHamiltonian
# --------------
function contract_mpo_expval(
        AC::MPSTensor, GL::MPSTensor, O::MPOTensor, GR::MPSTensor, ACbar::MPSTensor = AC
    )
    return @plansor GL[1 2; 3] * AC[3 7; 5] * GR[5 8; 6] * O[2 4; 7 8] * conj(ACbar[1 4; 6])
end
# generic fallback
contract_mpo_expval(AC, GL, O, GR, ACbar=AC) = dot(ACbar, MPO_AC_Hamiltonian(GL, O, GR) * AC)

function contract_mpo_expval1(AC::MPSTensor, O::AbstractTensorMap, ACbar::MPSTensor=AC)
    numin(O) == numout(O) == 1 || throw(ArgumentError("O is not a single-site operator"))
    return @plansor conj(ACbar[2 3; 4]) * O[3; 1] * AC[2 1; 4]
end
function contract_mpo_expval1(AC::GenericMPSTensor{S,3}, O::AbstractTensorMap{<:Any,S},
                              ACbar::GenericMPSTensor{S,3}=AC) where {S}
    numin(O) == numout(O) == 1 || throw(ArgumentError("O is not a single-site operator"))
    return @plansor conj(ACbar[2 3 4; 5]) * O[3; 1] * AC[2 1 4; 5]
end

function contract_mpo_expval2(A1::MPSTensor, A2::MPSTensor, O::AbstractTensorMap,
                              A1bar::MPSTensor=A1, A2bar::MPSTensor=A2)
    numin(O) == numout(O) == 2 || throw(ArgumentError("O is not a two-site operator"))
    return @plansor conj(A1bar[4 5; 6]) * conj(A2bar[6 7; 8]) * O[5 7; 2 3] * A1[4 2; 1] *
                    A2[1 3; 8]
end
function contract_mpo_expval2(A1::GenericMPSTensor{S,3}, A2::GenericMPSTensor{S,3},
                              O::AbstractTensorMap{<:Any,S},
                              A1bar::GenericMPSTensor{S,3}=A1,
                              A2bar::GenericMPSTensor{S,3}=A2) where {S}
    numin(O) == numout(O) == 2 || throw(ArgumentError("O is not a two-site operator"))
    return @plansor conj(A1bar[8 3 4; 11]) * conj(A2bar[11 12 13; 14]) * τ[9 6; 1 2] *
                    τ[3 4; 9 10] * A1[8 1 2; 5] * A2[5 7 13; 14] * O[10 12; 6 7]
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
