# Given a state and it's environments, we can act on it
"""
    DerivativeOperator

Abstract supertype for derivative operators acting on MPS. These operators are used to represent
the effective local operators obtained from taking the partial derivative of an MPS-MPO-MPS sandwich.
"""
abstract type DerivativeOperator end

Base.:*(h::DerivativeOperator, v) = h(v)
(h::DerivativeOperator)(v, ::Number) = h(v)

# Generic constructors
# --------------------

@doc """
    C_hamiltonian(site, above, operator, below, envs)::DerivativeOperator

Effective zero-site local operator acting at `site`.

```
 ┌─   ─┐ 
 │     │ 
┌┴┐   ┌┴┐
│ ├───┤ │
└┬┘   └┬┘
 │     │ 
 └─   ─┘ 
```

See also [`∂C`](@ref).
""" C_hamiltonian

@doc """
    AC_hamiltonian(site, mps, operator, mps, envs)::DerivativeOperator

Effective one-site local operator acting at `site`.

```
 ┌───   ───┐ 
 │    │    │ 
┌┴┐ ┌─┴─┐ ┌┴┐
│ ├─┤   ├─┤ │
└┬┘ └─┬─┘ └┬┘
 │    │    │ 
 └───   ───┘ 
```

See also [`∂AC`](@ref).
""" AC_hamiltonian

@doc """
    AC2_hamiltonian(site, above, operator, below, envs)

Effective two-site local operator acting at `site`.

```
 ┌──        ──┐ 
 │   │    │   │ 
┌┴┐┌─┴─┐┌─┴─┐┌┴┐
│ ├┤   ├┤   ├┤ │
└┬┘└─┬─┘└─┬─┘└┬┘
 │   │    │   │ 
 └──        ──┘ 
```

See also [`∂AC2`](@ref).
""" AC2_hamiltonian

# boilerplate for the derivative operators
for hamiltonian in (:C_hamiltonian, :AC_hamiltonian, :AC2_hamiltonian)
    @eval function $hamiltonian(site::CartesianIndex{2}, below, operator::MultilineMPO,
                                above, envs)
        row, col = Tuple(site)
        return $hamiltonian(col, below[row + 1], operator[row], above[row], envs[row])
    end
    @eval function $hamiltonian(col::Int, below, operator::MultilineMPO, above, envs)
        Hs = map(1:size(operator, 1)) do row
            return $hamiltonian(CartesianIndex(row, col), below, operator, above, envs)
        end
        return Multiline(Hs)
    end
    @eval function $hamiltonian(site::Int, below, operator::MultipliedOperator, above, envs)
        H = $hamiltonian(site, below, operator.op, above, envs)
        return MultipliedOperator(H, operator.f)
    end
    @eval function $hamiltonian(site::Int, below, operator::LinearCombination, above, envs)
        Hs = map(operator.opps, envs.envs) do o, env
            return $hamiltonian(site, below, o, above, env)
        end
        return LinearCombination(Hs, operator.coeffs)
    end
    @eval function $hamiltonian(site::Int, below, operator::LazySum, above,
                                envs::MultipleEnvironments)
        Hs = map(operator.ops, envs.envs) do o, env
            return $hamiltonian(site, below, o, above, env)
        end
        elT = Union{D,MultipliedOperator{D}} where {D<:DerivativeOperator}
        return LazySum{elT}(Hs)
    end
end

"""
    site_derivative(nsites, site, below, operator, above, envs)

Effective local `nsite`-derivative operator acting at `site`.
"""
Base.@constprop :aggressive function site_derivative(nsites::Int, site, below, operator,
                                                     above, envs)
    return site_derivative(Val(nsites), site, below, operator, above, envs)
end
function site_derivative(::Val{0}, site, below, operator, above, envs)
    return C_hamiltonian(site, below, operator, above, envs)
end
function site_derivative(::Val{1}, site, below, operator, above, envs)
    return AC_hamiltonian(site, below, operator, above, envs)
end
function site_derivative(::Val{2}, site, below, operator, above, envs)
    return AC2_hamiltonian(site, below, operator, above, envs)
end
function site_derivative(::Val{N}, site, below, operator, above, envs) where {N}
    throw(ArgumentError("site derivative not implemented for $N sites"))
end

# Generic actions
# ---------------

"""
    ∂C(x, leftenv, rightenv)

Application of the effective zero-site local operator on a local tensor `x`.

```
   ┌─┐   
 ┌─┤ ├─┐ 
 │ └─┘ │ 
┌┴┐   ┌┴┐
│ ├───┤ │
└┬┘   └┬┘
 │     │ 
 └─   ─┘ 
```

See also [`C_hamiltonian`](@ref).
"""
∂C(x, leftenv, rightenv) = MPO_C_Hamiltonian(leftenv, rightenv)(x)

"""
    ∂AC(x, operator, leftenv, rightenv)

Application of the effective one-site local operator on a local tensor `x`.

```
    ┌───┐    
 ┌──┤   ├──┐ 
 │  └─┬─┘  │ 
┌┴┐ ┌─┴─┐ ┌┴┐
│ ├─┤   ├─┤ │
└┬┘ └─┬─┘ └┬┘
 │    │    │ 
 └──     ──┘ 
```

See also [`AC_hamiltonian`](@ref).
"""
∂AC(x, operator, leftenv, rightenv) = MPO_AC_Hamiltonian(leftenv, operator, rightenv)(x)

"""
    ∂AC2(x, operator1, operator2, leftenv, rightenv)

Application of the effective two-site local operator on a local tensor `x`.

```
    ┌──────┐    
 ┌──┤      ├──┐ 
 │  └┬────┬┘  │ 
┌┴┐┌─┴─┐┌─┴─┐┌┴┐
│ ├┤   ├┤   ├┤ │
└┬┘└─┬─┘└─┬─┘└┬┘
 │   │    │   │ 
 └──        ──┘ 
```

See also [`AC2_hamiltonian`](@ref).
"""
∂AC2(x, O₁, O₂, leftenv, rightenv) = MPO_AC2_Hamiltonian(leftenv, O₁, O₂, rightenv)(x)

# Projection operators
# --------------------
function c_proj(pos::Int, ψ, (operator, ϕ)::Tuple, envs)
    return C_hamiltonian(pos, ψ, operator, ϕ, envs) * ϕ.C[pos]
end
function c_proj(pos::Int, ψ, ϕ::AbstractMPS, envs)
    return C_hamiltonian(pos, ψ, nothing, ϕ, envs) * ϕ.C[pos]
end
function c_proj(pos::Int, ψ, Oϕs::LazySum, envs)
    return sum(zip(Oϕs.ops, envs.envs)) do x
        return c_proj(pos, ψ, x...)
    end
end
function c_proj(row::Int, col::Int, ψ::MultilineMPS, (O, ϕ)::Tuple, envs)
    return c_proj(col, ψ[row], (O[row], ϕ[row]), envs[row])
end

ac_proj(pos::Int, ψ, (O, ϕ)::Tuple, envs) = AC_hamiltonian(pos, ψ, O, ϕ, envs) * ϕ.AC[pos]
function ac_proj(pos::Int, ψ, ϕ::AbstractMPS, envs)
    return AC_hamiltonian(pos, ψ, nothing, ϕ, envs) * ϕ.AC[pos]
end

function ac_proj(pos::Int, ψ, Oϕs::LazySum, envs)
    return sum(zip(Oϕs.ops, envs.envs)) do x
        return ac_proj(pos, ψ, x...)
    end
end
function ac_proj(row::Int, col::Int, ψ::MultilineMPS, (O, ϕ)::Tuple, envs)
    return ac_proj(col, ψ[row], (O[row], ϕ[row]), envs[row])
end

function ac2_proj(pos::Int, ψ, (O, ϕ)::Tuple, envs)
    AC2 = ϕ.AC[pos] * _transpose_tail(ϕ.AR[pos + 1])
    return AC2_hamiltonian(pos, ψ, O, ϕ, envs) * AC2
end
function ac2_proj(pos::Int, ψ, ϕ::AbstractMPS, envs)
    AC2 = ϕ.AC[pos] * _transpose_tail(ϕ.AR[pos + 1])
    return AC2_hamiltonian(pos, ψ, nothing, ϕ, envs) * AC2
end
function ac2_proj(row::Int, col::Int, ψ::MultilineMPS, (O, ϕ)::Tuple, envs)
    return ac2_proj(col, ψ[row], (O[row], ϕ[row]), envs[row])
end

# Varia
# -----

# Multiline
function (H::Multiline{<:DerivativeOperator})(x::AbstractVector)
    return [H[row - 1] * x[mod1(row - 1, end)] for row in 1:size(H, 1)]
end
Base.:*(H::Multiline{<:DerivativeOperator}, x::AbstractVector) = H(x)

# time dependent derivative operators
(h::UntimedOperator{<:DerivativeOperator})(y, args...) = h.f * h.op(y)
(h::TimedOperator{<:DerivativeOperator})(y, t::Number) = h.f(t) * h.op(y)
function (x::LazySum{<:Union{MultipliedOperator{D},D} where {D<:DerivativeOperator}})(y,
                                                                                      t::Number)
    return sum(O -> O(y, t), x)
end
function (x::LazySum{<:Union{MultipliedOperator{D},D} where {D<:DerivativeOperator}})(y)
    return sum(O -> O(y), x)
end
function Base.:*(h::LazySum{<:Union{D,MultipliedOperator{D}} where {D<:DerivativeOperator}},
                 v)
    return h(v)
end
