"""
    Structure representing a time dependent operator. Consists of
        - An operator (MPO, Hamiltonian, ...)
        - A function f giving the time dependence i.e O(t) = f(t)*O
"""
struct TimedOperator{O,F <: Function}
    op::O
    fun::F
end

#constructors for TimedOperator
TimedOperator(x) = TimedOperator(x,t->1);
TimedOperator(x::TimedOperator) = x

# what to do when we multiply by a scalar
function Base.:*(op::TimedOperator, b::Number)
    TimedOperator(b*op.op,op.fun)
end
Base.:*(b::Number, op::TimedOperator) = op * b

#do we need to define * for two TimedOperators?

#perhaps we define an convert?

#=
# Time selector function, needed if we allow mixing of operators and TimedOperators
timeselect(x,t::Number) = x
timeselect(x::TimedOperator,t::Number) = x(t)
=#

#define so we can act on it with (x,t) and obtain normal operator
(x::TimedOperator)(t::Number) = x.fun(t)*x.op
(x::TimedOperator)(y) = TimedOperator(x.op(y),x.fun)
(x::TimedOperator)(y,t::Number) = x(y)(t)


# don't know a better place to put this
# environment for TimedOperator
environments(st,x::TimedOperator) = environments(st,x.op)