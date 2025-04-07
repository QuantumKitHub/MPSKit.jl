# Given a state and it's environments, we can act on it
"""
    DerivativeOperator

Abstract supertype for derivative operators acting on MPS. These operators are used to represent
the effective local operators obtained from taking the partial derivative of an MPS-MPO-MPS sandwich.
"""
abstract type DerivativeOperator end

Base.:*(h::DerivativeOperator, v) = h(v)
(h::DerivativeOperator)(v, ::Number) = h(v)

@doc """
    ∂∂C(site::Int, mps, operator, envs)
    ∂∂C(row::Int, col::Int, mps, operator, envs)

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
""" ∂∂C

∂∂C(pos::CartesianIndex, mps, operator, envs) = ∂∂C(Tuple(pos)..., mps, operator, envs)
function ∂∂C(row::Int, col::Int, mps::MultilineMPS, operator::MultilineMPO, envs)
    return ∂∂C(col, mps[row], operator[row], envs[row])
end
function ∂∂C(site::Int, mps, operator::MultilineMPO, envs)
    return Multiline([∂∂C(row, site, mps, operator, envs) for row in 1:size(operator, 1)])
end
function ∂∂C(pos::Int, mps, operator::MultipliedOperator, cache)
    return MultipliedOperator(∂∂C(pos, mps, operator.op, cache), operator.f)
end
function ∂∂C(pos::Int, mps, operator::LinearCombination, cache)
    return LinearCombination(broadcast((h, e) -> ∂∂C(pos, mps, h, e), operator.opps,
                                       cache.envs),
                             operator.coeffs)
end
function ∂∂C(pos::Int, mps, operator::LazySum, cache::MultipleEnvironments)
    suboperators = map((op, openv) -> ∂∂C(pos, mps, op, openv), operator.ops, cache.envs)
    return LazySum{Union{MPO_∂∂C,MultipliedOperator{<:MPO_∂∂C}}}(suboperators)
end

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

See also [`∂∂C`](@ref).
"""
∂C(x, leftenv, rightenv) = MPO_∂∂C(leftenv, rightenv)(x)

@doc """
    ∂∂AC(site::Int, mps, operator, envs)
    ∂∂AC(row::Int, col::Int, mps, operator, envs)

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
""" ∂∂AC
∂∂AC(pos::CartesianIndex, mps, operator, envs) = ∂∂AC(Tuple(pos)..., mps, operator, envs)
function ∂∂AC(row::Int, col::Int, mps::MultilineMPS, operator::MultilineMPO, envs)
    return ∂∂AC(col, mps[row], operator[row], envs[row])
end
function ∂∂AC(site::Int, mps, operator::MultilineMPO, envs)
    return Multiline([∂∂AC(row, site, mps, operator, envs) for row in 1:size(operator, 1)])
end
function ∂∂AC(pos::Int, mps, operator::MultipliedOperator, cache)
    return MultipliedOperator(∂∂AC(pos, mps, operator.op, cache), operator.f)
end
function ∂∂AC(pos::Int, mps, operator::LinearCombination, cache)
    return LinearCombination(broadcast((h, e) -> ∂∂AC(pos, mps, h, e), operator.opps,
                                       cache.envs), operator.coeffs)
end
function ∂∂AC(pos::Int, mps, operator::LazySum, cache::MultipleEnvironments)
    suboperators = map((op, openv) -> ∂∂AC(pos, mps, op, openv), operator.ops, cache.envs)
    elT = Union{D,MultipliedOperator{D}} where {D<:Union{MPO_∂∂AC,JordanMPO_∂∂AC}}
    return LazySum{elT}(suboperators)
end

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

See also [`∂∂AC`](@ref).
"""
∂AC(x, operator, leftenv, rightenv) = MPO_∂∂AC(leftenv, operator, rightenv)(x)

@doc """
    ∂∂AC2(site::Int, mps, operator, envs)
    ∂∂AC2(row::Int, col::Int, mps, operator, envs)

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
""" ∂∂AC2

∂∂AC2(pos::CartesianIndex, mps, operator, envs) = ∂∂AC2(Tuple(pos)..., mps, operator, envs)
function ∂∂AC2(row::Int, col::Int, mps::MultilineMPS, operator::MultilineMPO, envs)
    return ∂∂AC2(col, mps[row], operator[row], envs[row])
end
function ∂∂AC2(site::Int, mps, operator::MultilineMPO, envs)
    return Multiline([∂∂AC2(row, site, mps, operator, envs) for row in 1:size(operator, 1)])
end
function ∂∂AC2(pos::Int, mps, operator::MultipliedOperator, cache)
    return MultipliedOperator(∂∂AC2(pos, mps, operator.op, cache), operator.f)
end
function ∂∂AC2(pos::Int, mps, operator::LinearCombination, cache)
    return LinearCombination(broadcast((h, e) -> ∂∂AC2(pos, mps, h, e), operator.opps,
                                       cache.envs), operator.coeffs)
end
function ∂∂AC2(pos::Int, mps, operator::LazySum, cache::MultipleEnvironments)
    suboperators = map((op, openv) -> ∂∂AC2(pos, mps, op, openv), operator.ops, cache.envs)
    elT = Union{D,MultipliedOperator{D}} where {D<:Union{MPO_∂∂AC2,JordanMPO_∂∂AC2}}
    return LazySum{elT}(suboperators)
end

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

See also [`∂∂AC2`](@ref).
"""
∂AC2(x, O₁, O₂, leftenv, rightenv) = MPO_∂∂AC2(leftenv, O₁, O₂, rightenv)(x)

# Projection operators
# --------------------
c_proj(pos::Int, ψ, (operator, ϕ)::Tuple, envs) = ∂∂C(pos, ψ, operator, envs) * ϕ.C[pos]
c_proj(pos::Int, ψ, ϕ::AbstractMPS, envs) = ∂∂C(pos, ψ, nothing, envs) * ϕ.C[pos]
function c_proj(pos::Int, ψ, Oϕs::LazySum, envs)
    return sum(zip(Oϕs.ops, envs.envs)) do x
        return c_proj(pos, ψ, x...)
    end
end
function c_proj(row::Int, col::Int, ψ::MultilineMPS, (O, ϕ)::Tuple, envs)
    return c_proj(col, ψ[row], (O[row], ϕ[row]), envs[row])
end

ac_proj(pos::Int, ψ, (O, ϕ)::Tuple, envs) = ∂∂AC(pos, ψ, O, envs) * ϕ.AC[pos]
ac_proj(pos::Int, ψ, ϕ::AbstractMPS, envs) = ∂∂AC(pos, ψ, nothing, envs) * ϕ.AC[pos]

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
    return ∂∂AC2(pos, ψ, O, envs) * AC2
end
function ac2_proj(pos::Int, ψ, ϕ::AbstractMPS, envs)
    AC2 = ϕ.AC[pos] * _transpose_tail(ϕ.AR[pos + 1])
    return ∂∂AC2(pos, ψ, nothing, envs) * AC2
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
