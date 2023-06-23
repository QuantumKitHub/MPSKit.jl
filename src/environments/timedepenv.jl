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

#define so we can act on it with (x,t) and obtain normal operator
(x::TimedOperator)(t::Number) = x.fun(t)*x.op
(x::TimedOperator)(y,t::Number) = x.fun(t)*x.op(y) #this is for derivatives

#define repeat please

# environment for TimedOperator
environments(st,x::TimedOperator) = environments(st,x.op)

#define how derivative should work
∂∂C(pos::Int,mps,opp::TimedOperator,cache) =
    TimedOperator(∂∂C(pos::Int,mps,opp.op,cache),opp.fun)

∂∂AC(pos::Int,mps,opp::TimedOperator,cache) =
    TimedOperator(∂∂AC(pos::Int,mps,opp.op,cache),opp.fun)

∂∂AC2(pos::Int,mps,opp::TimedOperator,cache) =
    TimedOperator(∂∂AC2(pos::Int,mps,opp.op,cache),opp.fun)


#define expectation_value and the like
expectation_value(state,op::TimedOperator,t::Number,at::Int64) = expectation_value(state,op(t),at::Int64)

expectation_value(state,op::TimedOperator,t::Number,envs::Cache=environments(state,op)) = expectation_value(state,op(t),envs)

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

#define repeat please

#add definition of sum, returns SumOfOperators object
# should we only allow same type SumOfOperators/TimedOperator to sum?
Base.:+(op1::TimedOperator,op2::TimedOperator) = SumOfOperators([op1,op2])

Base.:+(op1::SumOfOperators,op2::TimedOperator) = SumOfOperators(vcat(op1.ops,op2))

Base.:+(op1::SumOfOperators,op2::SumOfOperators) = SumOfOperators(vcat(op1.ops,op2.ops))

Base.:+(op1::Union{TimedOperator,SumOfOperators},op2::SumOfOperators) = op2 + op1

#define so we can act on it with (y,t) and obtain normal operator
# there should not be any confusion about TimedOperator or not, but type parameter in case
(x::SumOfOperators{O})(t::Number) where O <: TimedOperator = 
        sum(map( top -> top(t), x))

#this is for derivatives
(x::SumOfOperators)(y) = sum(map( top -> top(y), x))

(x::SumOfOperators{O})(y,t::Number) where O <: TimedOperator = 
        sum(map( top -> top(y,t), x)) 


# should this just be a alias for AbstractVector{C} where C <: Cache ?
struct MultipleEnvironments{C}
    envs::Vector{C}
end

Base.size(x::MultipleEnvironments) = size(x.envs)
Base.getindex(x::MultipleEnvironments,i) = x.envs[i]
Base.length(x::MultipleEnvironments) = prod(size(x))

Base.iterate(x::MultipleEnvironments) = iterate(x.envs)
Base.iterate(x::MultipleEnvironments,i) = iterate(x.envs,i)

# we need constructor, agnostic of particular MPS?
environments(st,ham::SumOfOperators) = MultipleEnvironments( map(op->environments(st,op),ham.ops) )

# we need to define how to recalculate
"""
    Recalculate in-place each sub-env in MultipleEnvironments
"""
function recalculate!(env::MultipleEnvironments,args...)
    for subenv in env.envs
        recalculate!(subenv,args...)
    end
    env
end

# derivatives
∂∂C(pos::Int,mps,opp::SumOfOperators,cache::MultipleEnvironments) =
    SumOfOperators( map((op,openv)->∂∂C(pos,mps,op,openv),opp.ops,cache.envs) )

∂∂AC(pos::Int,mps,opp::SumOfOperators,cache::MultipleEnvironments) =
    SumOfOperators( map((op,openv)->∂∂AC(pos,mps,op,openv),opp.ops,cache.envs) )

∂∂AC2(pos::Int,mps,opp::SumOfOperators,cache::MultipleEnvironments) =
    SumOfOperators( map((op,openv)->∂∂AC2(pos,mps,op,openv),opp.ops,cache.envs) )

#define expectation_value and the like
expectation_value(state,ops::SumOfOperators,t::Number,at::Int64) = sum(map(top->expectation_value(state,top(t),at::Int64),ops))

expectation_value(state,ops::SumOfOperators,t::Number,envs::MultipleEnvironments=environments(state,ops)) = sum(map( (top,tenv)->expectation_value(state,top(t),tenv),ops.ops,envs))

#where do we put this in MPSKit.jl?

environments(state::WindowMPS,win::Window) = Window([environments(state.left_gs,win.left), environments(state.window,win.middle), environments(state.right_gs,win.right)] )