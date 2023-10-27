"""
    Structure representing a multiplied operator. Consists of
        - An operator op (MPO, Hamiltonian, ...)
        - An object f that gets multiplied with the operator (Number, function, ...) 
"""
struct MultipliedOperator{O,F}
    op::O
    f::F
end

"""
    Structure representing a time-dependent operator. Consists of
        - An operator op (MPO, Hamiltonian, ...)
        - An function f that gives the time-dependence according to op(t) = f(t)*op
"""
const TimedOperator{O} = MultipliedOperator{O,<:Function}

"""
    Structure representing a time-independent operator that will be multiplied with a constant coefficient. Consists of
        - An operator (MPO, Hamiltonian, ...)
        - A number f that gets multiplied with the operator
"""
const UntimedOperator{O} = MultipliedOperator{O,<:Real}

#constructors for (un)TimedOperator
TimedOperator(x::O, f::F) where {F<:Function,O} = MultipliedOperator(x, f)
UntimedOperator(x::O, c::C) where {C<:Real,O} = MultipliedOperator(x, c)

TimedOperator(x) = TimedOperator(x, t -> 1)
UntimedOperator(x) = UntimedOperator(x, 1)

# evaluating at t should return object of type(object.opp)
(x::TimedOperator)(t::Number) = x.f(t) * x.op
(x::MultipliedOperator)(t::Number) = x.f * x.op
(x::UntimedOperator)() = x.f * x.op

# what to do when we multiply by a scalar
function Base.:*(op::UntimedOperator, b::Number)
    return UntimedOperator(op.op, b * op.f)
end
function Base.:*(op::TimedOperator, b::Number)
    return TimedOperator(op.op, t -> b * op.f(t))
end
Base.:*(b::Number, op::MultipliedOperator) = op * b


# logic for derivatives

#(x::MultipliedOperator{<:Any,<:Number})(y, ::Number) = x.f * x.op(y)

#(x::MultipliedOperator{<:Any,<:Function})(t::Number) = UntimedOperator(x.op, x.f(t))
#(x::MultipliedOperator{<:Any,<:Function})(y) = t -> x.f(t) * x.op(y)
#(x::MultipliedOperator{<:Any,<:Number})(::Number) = x.f * x.op
#(x::MultipliedOperator{<:Any,<:Number})(y) = t -> x.f * x.op(y)

#Base.:*(x::MultipliedOperator, v) = x(v)

# don't know a better place to put this
# environment for MultipliedOperator
function environments(st, x::MultipliedOperator, args...; kwargs...)
    return environments(st, x.op, args...; kwargs...)
end
