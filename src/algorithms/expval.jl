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


"""
function expectation_value end

# Local operators
# ---------------
function expectation_value(ψ::AbstractMPS, (inds, O)::Pair)
    @boundscheck foreach(Base.Fix1(checkbounds, ψ), inds)

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
        return @plansor conj(AC[4 5; 6]) *
                        conj(Ut[1]) * local_mpo[1][1 5; 3 2] * Ut[2] *
                        AC[4 3; 6]
    end
    if length(sites) == 2
        AC = ψ.AC[sites[1]]
        AR = ψ.AR[sites[2]]
        O1, O2 = local_mpo
        return @plansor conj(AC[4 5; 10]) * conj(Ut[1]) * O1[1 5; 3 8] * AC[4 3; 6] *
                        conj(AR[10 9; 11]) * Ut[2] * O2[8 9; 7 2] * AR[6 7; 11]
    end

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
    return E / norm(ψ)^2
end

# MPOs
# ----

function expectation_value(ψ::AbstractFiniteMPS, H::MPOHamiltonian,
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
function expectation_value(ψ::InfiniteMPS, H::MPOHamiltonian,
                           envs::Cache=environments(ψ, H))
    # environments are renormalized per site -- we need to contract entire unitcell
    # TODO: this could be done slightly more efficiently, we do not need:
    # - the entire left environment
    # - the right environments GR[1][2:end]

    GL = leftenv(envs, 1, ψ)
    GR = rightenv(envs, 1, ψ)
    A = [i == 1 ? ψ.AC[i] : ψ.AR[i] for i in 1:length(ψ)]
    T = TransferMatrix(A, H[:], A)
    return @plansor GL[1][1 2; 3] * (T * GR)[1][3 2; 1] / norm(ψ)^2
end

# MPOHamiltonian
# --------------
# TODO: add section in documentation to explain convention
expectation_value(ψ, envs::Cache) = expectation_value(ψ, envs.opp, envs)
function expectation_value(ψ, H::MPOHamiltonian)
    return expectation_value(ψ, H, environments(ψ, H))
end

"""
    expectation_value(ψ::WindowMPS, H::MPOHAmiltonian, envs) -> vals, tot

TODO
"""
function expectation_value(ψ::WindowMPS, H::MPOHamiltonian, envs::FinEnv)
    vals = expectation_value_fimpl(ψ, H, envs)

    tot = 0.0 + 0im
    for i in 1:(H.odim), j in 1:(H.odim)
        tot += @plansor leftenv(envs, length(ψ), ψ)[i][1 2; 3] * ψ.AC[end][3 4; 5] *
                        rightenv(envs, length(ψ), ψ)[j][5 6; 7] *
                        H[length(ψ)][i, j][2 8; 4 6] * conj(ψ.AC[end][1 8; 7])
    end

    return vals, tot / (norm(ψ.AC[end])^2)
end

# function expectation_value(st::InfiniteMPS, H::MPOHamiltonian,
#                            prevca::Union{MPOHamInfEnv,IDMRGEnv})
#     #calculate energy density
#     len = length(st)
#     ens = PeriodicArray(zeros(scalartype(st.AR[1]), len))
#     for i in 1:len
#         util = fill_data!(similar(st.AL[1], space(prevca.lw[H.odim, i + 1], 2)), one)
#         for j in (H.odim):-1:1
#             apl = leftenv(prevca, i, st)[j] *
#                   TransferMatrix(st.AL[i], H[i][j, H.odim], st.AL[i])
#             ens[i] += @plansor apl[1 2; 3] * r_LL(st, i)[3; 1] * conj(util[2])
#         end
#     end
#     return ens
# end

#the mpo hamiltonian over n sites has energy f+n*edens, which is what we calculate here. f can then be found as this - n*edens
# function expectation_value(st::InfiniteMPS, prevca::MPOHamInfEnv,
#                            range::Union{UnitRange{Int},Int})
#     return expectation_value(st, prevca.opp, range, prevca)
# end
# function expectation_value(st::InfiniteMPS, H::MPOHamiltonian, range::Int,
#                            prevca=environments(st, H))
#     return expectation_value(st, H, 1:range, prevca)
# end
# function expectation_value(st::InfiniteMPS, H::MPOHamiltonian, range::UnitRange{Int},
#                            prevca=environments(st, H))
#     start = map(leftenv(prevca, range.start, st)) do y
#         @plansor x[-1 -2; -3] := y[1 -2; 3] * st.CR[range.start - 1][3; -3] *
#                                  conj(st.CR[range.start - 1][1; -1])
#     end
#
#     for i in range
#         start = start * TransferMatrix(st.AR[i], H[i], st.AR[i])
#     end
#
#     tot = 0.0 + 0im
#     for i in 1:(H.odim)
#         tot += @plansor start[i][1 2; 3] * rightenv(prevca, range.stop, st)[i][3 2; 1]
#     end
#
#     return tot
# end

# Transfer matrices
# -----------------
function expectation_value(ψ::InfiniteMPS, mpo::DenseMPO)
    return expectation_value(convert(MPSMultiline, ψ), convert(MPOMultiline, mpo))
end
function expectation_value(ψ::MPSMultiline, mpo::MPOMultiline)
    return expectation_value(ψ, environments(ψ, mpo))
end
function expectation_value(ψ::InfiniteMPS, ca::PerMPOInfEnv)
    return expectation_value(convert(MPSMultiline, ψ), ca)
end
function expectation_value(ψ::MPSMultiline, O::MPOMultiline, ca::PerMPOInfEnv)
    retval = PeriodicMatrix{scalartype(ψ)}(undef, size(ψ, 1), size(ψ, 2))
    for (i, j) in product(1:size(ψ, 1), 1:size(ψ, 2))
        retval[i, j] = @plansor leftenv(ca, i, j, ψ)[1 2; 3] * O[i, j][2 4; 6 5] *
                                ψ.AC[i, j][3 6; 7] * rightenv(ca, i, j, ψ)[7 5; 8] *
                                conj(ψ.AC[i + 1, j][1 4; 8])
    end
    return retval
end

expectation_value(ψ::FiniteQP, O) = expectation_value(convert(FiniteMPS, ψ), O)
function expectation_value(ψ::FiniteQP, O::MPOHamiltonian)
    return expectation_value(convert(FiniteMPS, ψ), O)
end

# more specific typing to account for onsite operators, array of operators, ...
# define expectation_value for MultipliedOperator as scalar multiplication of the non-multiplied result, instead of multiplying the operator itself
function expectation_value(ψ, op::UntimedOperator, args...)
    return op.f * expectation_value(ψ, op.op, args...)
end

# define expectation_value for LazySum
function expectation_value(ψ, ops::LazySum, at::Int)
    return sum(op -> expectation_value(ψ, op, at), ops)
end
function expectation_value(ψ, ops::LazySum, envs::MultipleEnvironments=environments(ψ, ops))
    return sum(((op, env),) -> expectation_value(ψ, op, env), zip(ops.ops, envs))
end

# for now we also have LinearCombination
function expectation_value(ψ, H::LinearCombination, envs::LazyLincoCache=environments(ψ, H))
    return return sum(((c, op, env),) -> c * expectation_value(ψ, op, env),
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
    return ens ./ n
end
