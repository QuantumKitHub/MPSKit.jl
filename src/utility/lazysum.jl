"""
Structure representing a sum of operators. Consists of
    - A vector of operators (MPO, Hamiltonian, TimedOperator, ...)
"""
struct LazySum{O} <: AbstractVector{O}
    ops::Vector{O}

end

Base.size(x::LazySum) = size(x.ops)
Base.getindex(x::LazySum, i) = x.ops[i]
#iteration and summation gets automatically implementend thanks to subtyping

Base.length(x::LazySum) = prod(size(x))

# constructors
LazySum(x) = LazySum([x])

function LazySum(ops::AbstractVector, fs::AbstractVector)
    return LazySum(map(MultipliedOperator, ops, fs))
end

# For users
# evaluating at t should return UntimedOperators
(x::LazySum{<:UntimedOperator})() = sum(ConvertOperator,x)
(x::LazySum{<:UntimedOperator})(::Number) = x
(x::LazySum{<:MultipliedOperator})(t::Number) = LazySum{UntimedOperator}( map(y -> ConvertOperator(y,t), x))
#(x::LazySum{MultipliedOperator{S}})(t::Number) where {S} = LazySum{UntimedOperator}( map(y -> ConvertOperator(y,t), x))
#(x::LazySum{MultipliedOperator{S,T}})(t::Number) where {S,T} = LazySum{UntimedOperator}( map(y -> ConvertOperator(y,t), x))
evalat(x::LazySum{<:UntimedOperator}) = x()
evalat(x::LazySum{<:MultipliedOperator},t::Number) = sum(y->ConvertOperator(y,t)(),x)
evalat(x::LazySum{<:UntimedOperator},::Number) = x
# we define the addition for LazySum and we do the rest with promote
function Base.:+(SumOfOps1::LazySum, SumOfOps2::LazySum)
    return LazySum([SumOfOps1...,SumOfOps2...])
end

Base.promote_rule(::Type{<:LazySum},::Type{T}) where {T} = LazySum
Base.convert(::Type{<:LazySum},x::O) where {O} = LazySum(x)
Base.convert(::Type{T}, x::T) where {T<:LazySum} = x

Base.:+(op1::Union{MultipliedOperator,LazySum}, op2::MultipliedOperator) = +(promote(op1,op2)...)
Base.:+(op1::MultipliedOperator, op2::LazySum) = +(promote(op1,op2)...)