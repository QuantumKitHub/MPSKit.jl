"""
    LazySum{O} <: AbstractVector{O}

Type that represents a lazy sum i.e explicit summation is only done when needed. 
This type is basically an AbstractVector with some extra functionality to calcaulate things efficiently.

## Fields
- ops -- Vector of summable objects

---

## Constructors
    LazySum(x::Vector)

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

# For internal use only
_eval_at(x::O,args...) where {O} = x

_eval_at(x::LazySum,args...) = _eval_at(TimeDependence(x),x,args...)
_eval_at(::Type{NotTimeDependent},x) = sum(_eval_at,x)
_eval_at(::Type{NotTimeDependent},x,::Number) =  ArgumentError("A time independent LazySum cannot be evaluated at a time t")

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

Base.promote_rule(::Type{<:LazySum},::Type{T}) where {T} = LazySum
Base.convert(::Type{<:LazySum},x::O) where {O} = LazySum(x)
Base.convert(::Type{T}, x::T) where {T<:LazySum} = x

Base.:+(op1::LazySum, op2) = +(promote(op1,op2)...)

Base.repeat(x::LazySum, args...) = LazySum(repeat.(x,args...))