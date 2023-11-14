"""
Structure representing a sum of operators. Consists of
    - A vector of operators (MPO, Hamiltonian, TimedOperator, ...)
"""
struct SumOfOperators{O} <: AbstractVector{O}
    ops::Vector{O}
end

Base.size(x::SumOfOperators) = size(x.ops)
Base.getindex(x::SumOfOperators, i) = x.ops[i]
#iteration and summation gets automatically implementend thanks to subtyping

Base.length(x::SumOfOperators) = prod(size(x))

# singleton constructor
SumOfOperators(x) = SumOfOperators([x])

function SumOfOperators(ops::AbstractVector, fs::AbstractVector)
    return SumOfOperators(map(MultipliedOperator, ops, fs))
end

# evaluating at t should return UntimedOperators
(x::SumOfOperators)(t::Number) = SumOfOperators( map(O -> ConvertOperator(O,t), x))
(x::SumOfOperators{O})() where {A,O<:Union{A,UntimedOperator{A}}}= sum(ConvertOperator,x)

# we define the addition for SumOfOperators and we do the rest with promote
function Base.:+(SumOfOps1::SumOfOperators, SumOfOps2::SumOfOperators)
    return SumOfOperators([SumOfOps1...,SumOfOps2...])
end

Base.promote_rule(::Type{<:SumOfOperators},::Type{O}) where {O} = SumOfOperators
Base.convert(::Type{<:SumOfOperators},x::O) where {O} = SumOfOperators(x)
Base.convert(::Type{T}, x::T) where {T<:SumOfOperators} = x

Base.:+(op1::Union{MultipliedOperator,SumOfOperators}, op2::MultipliedOperator) = +(promote(op1,op2)...)
Base.:+(op1::MultipliedOperator, op2::SumOfOperators) = +(promote(op1,op2)...)