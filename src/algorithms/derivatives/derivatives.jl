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
    C_hamiltonian(site, below, operator, above, envs)::DerivativeOperator

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

See also [`C_projection`](@ref).
""" C_hamiltonian

@doc """
    AC_hamiltonian(site, below, operator, above, envs)::DerivativeOperator

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

See also [`AC_projection`](@ref).
""" AC_hamiltonian

@doc """
    AC2_hamiltonian(site, below, operator, above, envs)

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

See also [`AC2_projection`](@ref).
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

# Projection operators
# --------------------
@doc """
    C_projection(site, below, operator, above, envs)

Application of the effective zero-site local operator at a given site.

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
""" C_projection

@doc """
    AC_projection(site, below, operator, above, envs)

Application of the effective one-site local operator at a given site.

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
""" AC_projection

@doc """
    AC2_projection(site, below, operator, above, envs)

Application of the effective two-site local operator at a given site.

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
""" AC2_projection

# boilerplate for the projection actions
for kind in (:C, :AC, :AC2)
    projection = Symbol(kind, "_projection")
    hamiltonian = Symbol(kind, "_hamiltonian")

    @eval function $projection(site, below, above::Tuple, envs)
        return $projection(site, below, above..., envs)
    end
    @eval function $projection(site, below, above::AbstractMPS, envs)
        return $projection(site, below, nothing, above, envs)
    end
    @eval function $projection(site::CartesianIndex{2}, below::MultilineMPS, operator,
                               above::MultilineMPS, envs)
        row, col = Tuple(site)
        return $projection(col, below[row + 1], operator[row], above[row], envs[row])
    end
    @eval function $projection(site, below, above::LazySum, envs)
        return sum(zip(above.ops, envs.envs)) do x
            return $projection(site, below, x...)
        end
    end
end
function C_projection(site, below, operator, above, envs)
    return C_hamiltonian(site, below, operator, above, envs) * above.C[site]
end
function AC_projection(site, below, operator, above, envs)
    return AC_hamiltonian(site, below, operator, above, envs) * above.AC[site]
end
function AC2_projection(site::Int, below, operator, above, envs)
    AC2 = above.AC[site] * _transpose_tail(above.AR[site + 1])
    return AC2_hamiltonian(site, below, operator, above, envs) * AC2
end

# Multiline
# ---------
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
