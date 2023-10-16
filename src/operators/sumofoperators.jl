"""
Structure representing a sum of operators. Consists of
    - A vector of operators (MPO, Hamiltonian, TimedOperator, ...)
"""
struct SumOfOperators{O} <: AbstractVector{O}
    ops::Vector{O}
end

Base.size(x::SumOfOperators) = size(x.ops)
Base.getindex(x::SumOfOperators, i) = x.ops[i]
#iteration gets automatically implementend thanks to subtyping

Base.length(x::SumOfOperators) = prod(size(x))

# singleton constructor
SumOfOperators(x) = SumOfOperators([x])

# handy constructor for SumOfOperators{MultipliedOperator} and backwards compatibility for LinearCombination
function SumOfOperators(ops::AbstractVector, fs::AbstractVector)
    return SumOfOperators(map((op, f) -> MultipliedOperator(op, f), ops, fs))
end

# we can add operators to SumOfOperators by using +
function Base.:+(op1::MultipliedOperator{O,F}, op2::MultipliedOperator{O,G}) where {O,F,G}
    return SumOfOperators([op1, op2])
end

# this we could also do with promote
function Base.:+(op1::TimedOperator{O}, op2::Union{O,UntimedOperator{O}}) where {O}
    return SumOfOperators(TimedOperator{O}[op1, TimedOperator(op2)])
end
function Base.:+(op1::Union{O,UntimedOperator{O}}, op2::TimedOperator{O}) where {O}
    return SumOfOperators(TimedOperator{O}[TimedOperator(op1), op2])
end
function Base.:+(op1::UntimedOperator{O}, op2::O) where {O}
    return SumOfOperators(UntimedOperator{O}[op1, UntimedOperator(op2)])
end
function Base.:+(op1::O, op2::UntimedOperator{O}) where {O}
    return SumOfOperators(UntimedOperator{O}[UntimedOperator(op1), op2])
end

function Base.:+(SumOfOps::SumOfOperators{O}, op::O) where {O}
    return SumOfOperators(vcat(SumOfOps.ops, op))
end
function Base.:+(op::O, SumOfOps::SumOfOperators{O}) where {O}
    return SumOfOperators(vcat(op, SumOfOps.ops))
end

function Base.:+(SumOfOps1::SumOfOperators{O}, SumOfOps2::SumOfOperators{O}) where {O}
    return SumOfOperators(vcat(SumOfOps1.ops, SumOfOps2.ops))
end

function Base.:+(SumOfOps::SumOfOperators{T}, op::O) where {O,T<:MultipliedOperator{O}}
    return SumOfOps + SumOfOperators(UntimedOperator(op))
end
function Base.:+(op::O, SumOfOps::SumOfOperators{T}) where {O,T<:MultipliedOperator{O}}
    return SumOfOperators(UntimedOperator(op)) + SumOfOps
end

# (x::SumOfOperators{<:TimedOperator})(t::Number) = SumOfOperators(map(op -> op(t), x)) #will convert to SumOfOperators{UnTimedOperator}

# logic for derivatives
(x::SumOfOperators{<:TimedOperator})(y, t::Number) = sum(O -> O(y, t), x)
(x::SumOfOperators)(y, ::Number) = sum(O -> O(y), x)
(x::SumOfOperators{<:TimedOperator})(t::Number) = sum(O -> O(t), x)
(x::SumOfOperators)(t::Number) = sum(x)
(x::SumOfOperators)(y) = sum(O -> O(y), x)

Base.:*(x::SumOfOperators, v) = x(v)

# (x::SumOfOperators)(y) = sum(op -> op(y), x)

# (x::SumOfOperators)(y, t::Number) = x(t)(y)
