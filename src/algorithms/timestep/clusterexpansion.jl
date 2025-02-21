struct ClusterExpansion
    N::Int
end

function make_time_mpo(H::MPOHamiltonian, dt::Number, alg::ClusterExpansion)
    N = alg.N
    lmax = N ÷ 2 # largest level
    τ = -im * dt
    # spaces
    P = physicalspace(H)[1]
    D = dim(physicalspace(H)[1]) # physical dimension
    V = BlockTensorKit.oplus([ℂ^(D^2l) for l in 0:lmax]...)

    TT = tensormaptype(ComplexSpace, 2, 2, ComplexF64)
    O = SparseBlockTensorMap{TT}(undef, V ⊗ P ← P ⊗ V)

    for n in 1:N
        if n == 1
            O[1, 1, 1, 1] = _make_Hamterms(H, 1, τ)
        elseif n == 2
            U, S, V = tsvd(_make_b(H, O, 2, τ), ((1, 2, 4), (3, 5, 6)))
            O[1, 1, 1, 2] = permute(U * sqrt(S), ((1, 2), (3, 4)))
            O[2, 1, 1, 1] = permute(sqrt(S) * V, ((1, 2), (3, 4)))
        else
            nl = n ÷ 2 # new level
            jnl = nl + 1 # Julia indexing
            apply_function = _make_apply_functions(O, n)
            b = _make_b(H, O, n, τ)
            Onew, info = lssolve(apply_function, b; verbosity=5)

            (info.converged) != 1 &&
                @warn "Did not converge when constucting the $n cluster"

            if isodd(n) # linear problem
                # assign the new level to O
                O[jnl, 1, 1, jnl] = Onew
            elseif iseven(n) # linear problem + svd
                U, S, V = tsvd(Onew, (1, 2, 4), (3, 5, 6))
                # assign the new levels to O
                O[jnl - 1, 1, 1, jnl] = permute(U * sqrt(S), ((1, 2), (3, 4)))
                O[jnl, 1, 1, jnl - 1] = permute(sqrt(S) * V, ((1, 2), (3, 4)))
            end
        end
    end

    # return O
    return MPO(PeriodicVector([O]))
end

function _make_Hamterms(H::MPOHamiltonian, n::Int, τ::Number)
    return add_util_leg(convert(TensorMap,
                                scale(DenseMPO(open_boundary_conditions(H, n)), τ)))
end

function _make_b(H::MPOHamiltonian, O::SparseBlockTensorMap, n::Int, τ::Number)
    # be - bo
    be = permute(exp(permute(_make_Hamterms(H, n, τ), Tuple(1:(n + 1)),
                             Tuple(vcat([2n + 2], collect((n + 2):(2n + 1)))))),
                 Tuple(1:(n + 1)), Tuple(vcat(collect((n + 3):(2n + 2)), [n + 2])))
    bo = _fully_contract_O(O, n)
    return be - bo
end

function _fully_contract_O(O::SparseBlockTensorMap, n::Int)
    # Make projector onto the 0 subspace of the virtual level
    Pl = zeros(ComplexF64, ℂ^1 ← space(O, 1))
    Pl[1] = 1

    Pr = zeros(ComplexF64, space(O, 4)' ← ℂ^1)
    Pr[1] = 1

    O_inds = [[i, -(i + 1), -(i + 1 + n), i + 1] for i in 1:n]

    # Contract and permute to right structure
    return permute(ncon([Pl, fill(O, n)..., Pr], [[-1, 1], O_inds..., [n + 1, -(2n + 2)]]),
                   Tuple(1:(n + 1)), Tuple((n + 2):(2n + 2)))
end

function make_envs(O::SparseBlockTensorMap, n::Int)
    nt = (n - 1) ÷ 2 # amount of tensors in the environment
    return _make_envs(O, nt)
end

function _make_envs(O::SparseBlockTensorMap, nt::Int)
    if nt == 1
        return O[1, 1, 1, 2], O[2, 1, 1, 1]
    else
        Tl, Tr = _make_envs(O, nt - 1)
        Ol = O[nt, 1, 1, nt + 1]
        Or = O[nt + 1, 1, 1, nt]
        return _make_left_env(Tl, Ol), _make_right_env(Or, Tr)
    end
end

@generated function _make_left_env(Tl::AbstractTensorMap, Ol::AbstractTensorMap)
    N₁ = numin(Tl)
    N = numind(Tl)
    t_out = tensorexpr(:new_e_l, -(1:(N₁ + 1)), -((N₁ + 2):(N + 2)))
    t_left = tensorexpr(:Tl, -(1:N₁), ((-((N₁ + 2):N)...), 1))
    t_right = tensorexpr(:Ol, (1, -(N₁ + 1)), (-(N + 1), -(N + 2)))
    return macroexpand(@__MODULE__,
                       :(return @tensor $t_out := $t_left * $t_right))
end

@generated function _make_right_env(Or::AbstractTensorMap, Tr::AbstractTensorMap)
    N₁ = numin(Tr)
    N = numind(Tr)
    t_out = tensorexpr(:new_e_r, -(1:(N₁ + 1)), -((N₁ + 2):(N + 2)))
    t_left = tensorexpr(:Or, -(1:2), (-(N₁ + 2), 1))
    t_right = tensorexpr(:Tr, (1, (-(3:(N₁ + 1))...)), -((N₁ + 3):(N + 2)))
    return macroexpand(@__MODULE__,
                       :(return @tensor $t_out := $t_left * $t_right))
end

function _make_apply_functions(O::SparseBlockTensorMap, n::Int)
    nT = (n - 1) ÷ 2
    Al, Ar = make_envs(O, n)

    fun = if iseven(n)
        let
            function A(x::TensorMap, ::Val{false})
                Al_inds = vcat(-collect(1:(nT + 1)), -collect((2nT + 4):(3nT + 3)), [1])
                x_inds = [1, -(nT + 2), -(nT + 3), -(3nT + 4), -(3nT + 5), 2]
                Ar_inds = vcat([2], -collect((nT + 4):(2nT + 3)),
                               -collect((3nT + 6):(4nT + 6)))
                return permute(ncon([Al, x, Ar], [Al_inds, x_inds, Ar_inds]),
                               Tuple(1:(2nT + 3)), Tuple((2nT + 4):(4nT + 6)))
            end

            function A(b::TensorMap, ::Val{true})
                Al_inds = vcat(collect(3:(nT + 2)), [-1, 1], collect((nT + 3):(2nT + 2)))
                b_inds = vcat([1], collect((nT + 3):(2nT + 2)), [-2, -3],
                              collect((3nT + 3):(4nT + 2)), collect(3:(nT + 2)), [-4, -5],
                              collect((2nT + 3):(3nT + 2)), [2])
                Ar_inds = vcat(collect((2nT + 3):(3nT + 2)), [2, -6],
                               collect((3nT + 3):(4nT + 2)))
                return permute(ncon([adjoint(Al), b, adjoint(Ar)],
                                    [Al_inds, b_inds, Ar_inds]), (1, 2, 3), (4, 5, 6))
            end
        end
    elseif isodd(n)
        let
            function A(x::TensorMap, ::Val{false})
                Al_inds = vcat(-collect(1:(nT + 1)), -collect((2nT + 3):(3nT + 2)), [1])
                x_inds = [1, -(nT + 2), -(3nT + 3), 2]
                Ar_inds = vcat([2], -collect((nT + 3):(2nT + 2)),
                               -collect((3nT + 4):(4nT + 4)))
                return permute(ncon([Al, x, Ar], [Al_inds, x_inds, Ar_inds]),
                               Tuple(1:(2nT + 2)), Tuple((2nT + 3):(4nT + 4)))
            end
            function A(b::TensorMap, ::Val{true})
                Al_inds = vcat(collect(3:(nT + 2)), [-1, 1], collect((nT + 3):(2nT + 2)))
                b_inds = vcat([1], collect((nT + 3):(2nT + 2)), [-2],
                              collect((3nT + 3):(4nT + 2)), collect(3:(nT + 2)), [-3],
                              collect((2nT + 3):(3nT + 2)), [2])
                Ar_inds = vcat(collect((2nT + 3):(3nT + 2)), [2, -4],
                               collect((3nT + 3):(4nT + 2)))
                return permute(ncon([adjoint(Al), b, adjoint(Ar)],
                                    [Al_inds, b_inds, Ar_inds]), (1, 2), (3, 4))
            end
        end
    end
    return fun
end
