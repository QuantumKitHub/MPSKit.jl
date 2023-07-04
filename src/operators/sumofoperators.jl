"""
Structure representing a sum of operators. Consists of
    - A vector of operators (MPO, Hamiltonian, TimedOperator, ...)
"""
struct SumOfOperators{O} <: AbstractVector{O}
ops::Vector{O}
end

Base.size(x::SumOfOperators) = size(x.ops)
Base.getindex(x::SumOfOperators,i) = x.ops[i]
#iteration gets automatically implementend thanks to subtyping

Base.length(x::SumOfOperators) = prod(size(x))

# singleton constructor
SumOfOperators(x) = SumOfOperators([x])

# handy constructor for SumOfOperators{TimedOperator}
SumOfOperators(ops::AbstractVector,fs::AbstractVector) = SumOfOperators( map( (op,f)->TimedOperator(op,f),ops,fs) )

#define repeat, not sure if this is wat we want
#Base.repeat(x::SumOfOperators,n::Int) = SumOfOperators(repeat(x.ops,n))

# we can add operators to SumOfOperators by using +
Base.:+(op1::TimedOperator{O,F},op2::TimedOperator{O,G}) where {O,G,F} = SumOfOperators([op1,op2])

Base.:+(op1::TimedOperator{O,F},op2::O) where {O,F} = SumOfOperators([op1,TimedOperator(op2)])
Base.:+(op1::O,op2::TimedOperator{O,F}) where {O,F} =  SumOfOperators([TimedOperator(op1),op2])

Base.:+(SumOfOps::SumOfOperators{O},op::O) where {O} = SumOfOperators(vcat(SumOfOps.ops,op))
Base.:+(op::O,SumOfOps::SumOfOperators{O}) where {O} = SumOfOps + op

Base.:+(SumOfOps1::SumOfOperators{O},SumOfOps2::SumOfOperators{O}) where {O} = SumOfOperators(vcat(SumOfOps1.ops,SumOfOps2.ops))

Base.:+(SumOfOps::SumOfOperators{T},op::O) where {O,F,T<:TimedOperator{O,F}} = SumOfOps + SumOfOperators(TimedOperator(op))
Base.:+(op::O,SumOfOps::SumOfOperators{T}) where {O,F,T<:TimedOperator{O,F}} = SumOfOps + op

# logic for derivatives
(x::SumOfOperators)(y) = sum(op-> op(y),x)

#(x::SumOfOperators)(t::Number) = sum(x)

(x::SumOfOperators{O})(y) where {O <: TimedOperator} = SumOfOperators( map(op-> op(y),x))

(x::SumOfOperators{O})(t::Number) where {O <: TimedOperator} = sum(op -> op(t), x)

(x::SumOfOperators{O})(y,t::Number) where {O <: TimedOperator} = x(y)(t)

# timeselector function that handles wether to apply t or not
#(x::SumOfOperators)(t::Number) = sum(op -> timeselect(op,t), x)

