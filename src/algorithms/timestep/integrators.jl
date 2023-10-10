"""
    integrate(f, y₀, t₀, a, dt, algorithm)

Integrate the differential equation ``dy/dt = a*f(y,t)`` over a time step 'dt' starting from ``y(t₀)=y₀``, using the provided algorithm. 

# Arguments
- `f::Function`: driving function
- `y₀`: object to integrate
- `t₀`: time f is evaluated at
- `a` : scalar prefactor
- `dt`: timestep
- `algorithm`: integration scheme
"""
function integrate end

# make time evaluation dispatchable
# user provides f(y,t) that can be called with two arguments
Eval(f,x,t::Number) = f(x)
Eval(f::F,x,t::Number) where {O<:TimedOperator,F<:Union{O,SumOfOperators{O}}} = f(x,t)


"""
    ExpIntegrator

Method that solves ``dy/dt = a*f(y,t)`` by exponentiating the action of f(y,t).

# Fields
- `krylovmethod::Union{Arnoldi,Lanczos}`` KrylovKit method for exponentiation. For options such as tolerance, see Lanczos/Arnoldi in KrylovKit.
"""
@kwdef struct ExpIntegrator
    krylovmethod::Union{Arnoldi,Lanczos} = Lanczos();
end

#original integrator in iTDVP, namely exponentiation
function integrate(
    f, y₀, t₀::Number, a::Number, dt::Number, method::ExpIntegrator
)
    sol, convhist = exponentiate( x->Eval_t(f,x,t₀), a*dt, y₀, method.krylovmethod)
    return sol, convhist.converged, convhist
end
