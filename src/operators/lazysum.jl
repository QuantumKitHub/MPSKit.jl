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

# Holy traits
struct TimeDependent end
struct NotTimeDependent end

TimeDependence(x::T) where {T} = TimeDependence(T)
TimeDependence(::Type) = NotTimeDependent
#TimeDependence(::Type{<:TimedOperator}) = TimeDependent
TimeDependence(x::LazySum) = promote_type(TimeDependence.(x)...)

Base.promote_rule(::Type{TimeDependent},::Type{NotTimeDependent}) = TimeDependent

# constructors
LazySum(x) = LazySum([x])


#function LazySum(ops::AbstractVector, fs::AbstractVector)
#    return LazySum(map(MultipliedOperator, ops, fs))
#end

# For internal use only
_eval_at(x::LazySum,args...) = _eval_at(TimeDependence(x),x,args...)
_eval_at(::Type{TimeDependent},x::LazySum,t::Number) = LazySum( map(y -> _eval_at(y,t), x))
_eval_at(::Type{TimeDependent},x::LazySum) = error("A time dependent LazySum needs a specified time to be evaluated at")
_eval_at(::Type{NotTimeDependent},x) = sum(_eval_at,x)
_eval_at(::Type{NotTimeDependent},x,::Number) =  error("A time independent LazySum cannot be evaluated at a time t")

# For users
# using (t) should return LazySum{UntimedOperators}
# using ConvertOperator should do explicit multiplication
(x::LazySum)() = _eval_at(x)
ConvertOperator(x::LazySum) = _eval_at(x,t)
(x::LazySum)(t::Number) = _eval_at(x,t)
ConvertOperator(x::LazySum,t) = x(t)()

# we define the addition for LazySum and we do the rest with promote
function Base.:+(SumOfOps1::LazySum, SumOfOps2::LazySum)
    return LazySum([SumOfOps1...,SumOfOps2...])
end

#Base.promote_rule(::Type{<:MultipliedOperator},::Type{T}) where {T} = LazySum
Base.promote_rule(::Type{<:LazySum},::Type{T}) where {T} = LazySum
Base.convert(::Type{<:LazySum},x::O) where {O} = LazySum(x)
Base.convert(::Type{T}, x::T) where {T<:LazySum} = x

# still a bit of a mess, idealy we would only need promote I think
Base.:+(op1, op2::MultipliedOperator) = +(LazySum(op1),LazySum(op2))
Base.:+(op1::MultipliedOperator, op2) = +(LazySum(op1),LazySum(op2))
Base.:+(op1::LazySum, op2) = +(promote(op1,op2)...)
Base.:+(op1::MultipliedOperator, op2::LazySum) = +(LazySum(op1),op2)
Base.:+(op1::LazySum, op2::MultipliedOperator) = +(op1,LazySum(op2))