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

TimeDependence(x) = NotTimeDependent()
TimeDependence(x::LazySum) = istimed(x) ? TimeDependent() : NotTimeDependent()

istimed(::TimeDependent) = true
istimed(::NotTimeDependent) = false
istimed(x) = istimed(TimeDependence(x))
istimed(x::LazySum) = any(istimed, x)

# constructors
LazySum(x) = LazySum([x])
LazySum(f::Function,x) = LazySum(map(y->f(y),x))

# Internal use only, works always
_eval_at(x,args...) = x # -> this is what you should define for your custom structs inside a LazySum
#see derivatives.jl for x::DerivativeOperator

# wrapper around _eval_at
eval_at(x,args...) = eval_at(TimeDependence(x),x,args...)
eval_at(::TimeDependent,x::LazySum,t::Number) = LazySum(O -> _eval_at(O,t),x) 
eval_at(::TimeDependent,x::LazySum) = throw(ArgumentError("attempting to evaluate time-dependent LazySum without specifiying a time"))
eval_at(::NotTimeDependent,x::LazySum) = sum(_eval_at,x)
eval_at(::NotTimeDependent,x::LazySum,t::Number) = throw(ArgumentError("attempting to evaluate time-independent LazySum at time"))

# For users
(x::LazySum)() = eval_at(x)
ConvertOperator(x::LazySum) = eval_at(x) # using ConvertOperator should do explicit multiplication
(x::LazySum)(t::Number) = eval_at(x,t) # using (t) should return NotTimeDependent LazySum
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