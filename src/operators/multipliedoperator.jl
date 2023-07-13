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
TimedOperator(x::O,f::F) where {F<:Function,O} = MultipliedOperator(x,f);
UntimedOperator(x::O,c::C) where {C<:Real,O} = MultipliedOperator(x,c)

TimedOperator(x) = TimedOperator(x,t->1)
UntimedOperator(x) = UntimedOperator(x,1)

TimedOperator(x::UntimedOperator) = TimedOperator(x.op,t->x.f)
UntimedOperator(x::TimedOperator) = throw(ArgumentError("Cannot make a UntimedOperator from a TimedOperator without a time argument"))

# what to do when we multiply by a scalar
function Base.:*(op::UntimedOperator, b::Number)
    UntimedOperator(op.op,b*op.f)
end
function Base.:*(op::TimedOperator, b::Number)
    UntimedOperator(op.op,t->b*op.f(t))
end
Base.:*(b::Number, op::MultipliedOperator) = op * b

(x::UntimedOperator)() = x.f*x.op

#ignore time-dependence by default
(x::MultipliedOperator)(t::Number) = x

(x::TimedOperator)(t::Number) = UntimedOperator(x.op,x.f(t))

# logic for derivatives
Base.:*(x::MultipliedOperator,v) = x(v);

(x::MultipliedOperator)(y) = x.f*x.op(y)

(x::MultipliedOperator)(y,t::Number) = x(t)(y)


# don't know a better place to put this
# environment for MultipliedOperator
environments(st,x::MultipliedOperator) = environments(st,x.op)