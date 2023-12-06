"""
    LazySum{O} <: AbstractVector{O}

Type that represents a lazy sum i.e explicit summation is only done when needed. 
This type is basically an `AbstractVector` with some extra functionality to calculate things efficiently.

## Fields
- ops -- Vector of summable objects

---

## Constructors
    LazySum(x::Vector)

"""
struct LazySum{O} <: AbstractVector{O}
    ops::Vector{O}
end

# for the AbstractArray interface
Base.size(x::LazySum) = size(x.ops)
Base.getindex(x::LazySum, i) = x.ops[i]
Base.length(x::LazySum) = prod(size(x))
Base.similar(x::LazySum, ::Type{S}, dims::Dims) where {S} = LazySum(similar(x.ops, S, dims))

# Holy traits
TimeDependence(x::LazySum) = istimed(x) ? TimeDependent() : NotTimeDependent()
istimed(x::LazySum) = any(istimed, x)

# constructors
LazySum(x) = LazySum([x])
LazySum(f::Function, x) = LazySum(map(y -> f(y), x))

# wrapper around _eval_at
safe_eval(::TimeDependent, x::LazySum, t::Number) = LazySum(O -> _eval_at(O, t), x)
function safe_eval(::TimeDependent, x::LazySum)
    throw(ArgumentError("attempting to evaluate time-dependent LazySum without specifiying a time"))
end
safe_eval(::NotTimeDependent, x::LazySum) = sum(O -> _eval_at(O), x)
function safe_eval(::NotTimeDependent, x::LazySum, t::Number)
    throw(ArgumentError("attempting to evaluate time-independent LazySum at time"))
end

# For users
# using (t) should return NotTimeDependent LazySum
(x::LazySum)(t::Number) = safe_eval(x, t) 

# we define the addition for LazySum and we do the rest with this
function Base.:+(SumOfOps1::LazySum, SumOfOps2::LazySum)
    return LazySum([SumOfOps1..., SumOfOps2...])
end

Base.:+(op1::LazySum, op2) = op1 + LazySum(op2)
Base.:+(op1, op2::LazySum) = LazySum(op1) + op2

Base.repeat(x::LazySum, args...) = LazySum(repeat.(x, args...))
