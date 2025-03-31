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

@doc """
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
""" ∂C

function ∂C(x::MPSBondTensor, leftenv::MPSBondTensor, rightenv::MPSBondTensor)
    @plansor y[-1; -2] ≔ leftenv[-1; 1] * x[1; 2] * rightenv[2; -2]
    return y isa AbstractBlockTensorMap ? only(y) : y
end
function ∂C(x::MPSBondTensor, leftenv::MPSTensor, rightenv::MPSTensor)
    @plansor y[-1; -2] ≔ leftenv[-1 3; 1] * x[1; 2] * rightenv[2 3; -2]
    return y isa AbstractBlockTensorMap ? only(y) : y
end

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

@doc """
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
""" ∂AC

function ∂AC(x::MPSTensor, operator::MPOTensor, leftenv::MPSTensor, rightenv::MPSTensor)
    @plansor y[-1 -2; -3] := leftenv[-1 5; 4] * x[4 2; 1] * operator[5 -2; 2 3] *
                             rightenv[1 3; -3]
    return y isa AbstractBlockTensorMap ? only(y) : y
end
function ∂AC(x::MPSTensor, operator::Number, leftenv::MPSTensor, rightenv::MPSTensor)
    @plansor y[-1 -2; -3] := operator * (leftenv[-1 5; 4] * x[4 6; 1] * τ[6 5; 7 -2] *
                                         rightenv[1 7; -3])
    return y isa AbstractBlockTensorMap ? only(y) : y
end
function ∂AC(x::MPSTensor, ::Nothing, leftenv::MPSBondTensor, rightenv::MPSBondTensor)
    @plansor y[-1 -2; -3] := leftenv[-1; 2] * x[2 -2; 1] * rightenv[1; -3]
end
function ∂AC(x::GenericMPSTensor{<:Any,3}, operator::MPOTensor,
             leftenv::MPSTensor, rightenv::MPSTensor)
    @plansor y[-1 -2 -3; -4] ≔ leftenv[-1 7; 6] * x[6 4 2; 1] * operator[7 -2; 4 5] *
                               τ[5 -3; 2 3] * rightenv[1 3; -4]
    return y isa AbstractBlockTensorMap ? only(y) : y
end

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

@doc """
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
""" ∂AC2

function ∂AC2(x::MPOTensor, ::Nothing, ::Nothing, leftenv::MPSBondTensor,
              rightenv::MPSBondTensor)
    @plansor y[-1 -2; -3 -4] ≔ leftenv[-1; 1] * x[1 -2; 2 -4] * rightenv[2 -3]
    return y isa AbstractBlockTensorMap ? only(y) : y
end
function ∂AC2(x::MPOTensor, operator1::MPOTensor, operator2::MPOTensor, leftenv::MPSTensor,
              rightenv::MPSTensor)
    @plansor y[-1 -2; -3 -4] ≔ leftenv[-1 7; 6] * x[6 5; 1 3] *
                               operator1[7 -2; 5 4] * operator2[4 -4; 3 2] *
                               rightenv[1 2; -3]
    return y isa AbstractBlockTensorMap ? only(y) : y
end
function ∂AC2(x::AbstractTensorMap{<:Any,<:Any,3,3}, operator1::MPOTensor,
              operator2::MPOTensor, leftenv::MPSTensor, rightenv::MPSTensor)
    @plansor y[-1 -2 -3; -4 -5 -6] ≔ leftenv[-1 11; 10] * x[10 8 6; 1 2 4] *
                                     rightenv[1 3; -4] *
                                     operator1[11 -2; 8 9] * τ[9 -3; 6 7] *
                                     operator2[7 -6; 4 5] * τ[5 -5; 2 3]
    return y isa AbstractBlockTensorMap ? only(y) : y
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
