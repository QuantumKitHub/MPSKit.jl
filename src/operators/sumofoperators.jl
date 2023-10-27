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

function SumOfOperators(ops::AbstractVector, fs::AbstractVector)
    return SumOfOperators(map(MultipliedOperator, ops, fs))
end

# evaluating at t should return object of type(object.opp)
(x::SumOfOperators)(t::Number) = SumOfOperators(map(O -> O(t), x))

# we can add operators to SumOfOperators by using +
function Base.:+(op1::MultipliedOperator{O,F}, op2::MultipliedOperator{O,G}) where {O,F,G}
    return SumOfOperators([op1, op2])
end

# under review
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

# logic for derivatives

#(x::SumOfOperators{<:MultipliedOperator})(t::Number) = SumOfOperators(map(op -> op(t), x))

#(x::SumOfOperators)(y) = SumOfOperators(map(op -> op(y), x))

#Base.:*(x::SumOfOperators, v) = x(v)
