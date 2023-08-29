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

# handy constructor for SumOfOperators{MultipliedOperator} and backwards compatibility for LinearCombination
SumOfOperators(ops::AbstractVector,fs::AbstractVector) = SumOfOperators( map( (op,f)->MultipliedOperator(op,f),ops,fs))
LinearCombination(ops::Tuple,fs::Tuple) = SumOfOperators( collect(ops), collect(fs))


# we can add operators to SumOfOperators by using +
Base.:+(op1::MultipliedOperator{O,F},op2::MultipliedOperator{O,G}) where {O,F,G} = SumOfOperators([op1,op2])

# this we can also do with promote
Base.:+(op1::TimedOperator{O},op2::Union{O,UntimedOperator{O}}) where O   = SumOfOperators(TimedOperator{O}[op1, TimedOperator(op2)  ])
Base.:+(op1::Union{O,UntimedOperator{O}},op2::TimedOperator{O}) where O   = SumOfOperators(TimedOperator{O}[TimedOperator(op1), op2  ])
Base.:+(op1::UntimedOperator{O},op2::O) where O = SumOfOperators(UntimedOperator{O}[op1,UntimedOperator( op2)])
Base.:+(op1::O,op2::UntimedOperator{O}) where O = SumOfOperators(UntimedOperator{O}[UntimedOperator(op1), op2])

Base.:+(SumOfOps::SumOfOperators{O},op::O) where {O} = SumOfOperators(vcat(SumOfOps.ops,op))
Base.:+(op::O,SumOfOps::SumOfOperators{O}) where {O} = SumOfOperators(vcat(op,SumOfOps.ops))

Base.:+(SumOfOps1::SumOfOperators{O},SumOfOps2::SumOfOperators{O}) where {O} = SumOfOperators(vcat(SumOfOps1.ops,SumOfOps2.ops))

Base.:+(SumOfOps::SumOfOperators{T},op::O) where {O,T<:MultipliedOperator{O}} = SumOfOps + SumOfOperators(UntimedOperator(op))
Base.:+(op::O,SumOfOps::SumOfOperators{T}) where {O,T<:MultipliedOperator{O}} = SumOfOperators(UntimedOperator(op)) + SumOfOps

(x::SumOfOperators{<: UntimedOperator})() = sum(op->op(),x)

#ignore time-dependence by default
(x::SumOfOperators)(t::Number) = x

(x::SumOfOperators{<: MultipliedOperator})(t::Number) = SumOfOperators(map(op->op(t),x)) #will convert to SumOfOperators{UnTimedOperator}

# logic for derivatives
Base.:*(x::SumOfOperators,v) = x(v);

(x::SumOfOperators)(y) = sum(op-> op(y),x)

(x::SumOfOperators)(y,t::Number) = x(t)(y)

