"""
    integrate(f, y₀, t₀, a, dt, algorithm)

Integrate the differential equation ``dy/dt = a f(y,t)`` over a time step 'dt' starting from ``y(t₀)=y₀``, using the provided algorithm. 
For time-independent operators (i.e. not a TimedOperator) t₀ is ingored.

# Arguments
- `f::Function`: driving function
- `y₀`: object to integrate
- `t₀`: time f is evaluated at
- `a` : scalar prefactor
- `dt`: timestep
- `method`: method to integrate TDVP equations
"""
function integrate end

# for backwards compatibility
integrate(f, y₀, a, dt, method) = integrate(f, y₀, 0.0, a, dt, method)

# wrap function into UntimedOperator by default
integrate(f, y₀, t₀, a, dt, method) = integrate(UntimedOperator(f), y₀, t₀, a, dt, method)

#original integrator in iTDVP, namely exponentiation
function integrate(
    f::F, y₀, t₀, a, dt, method::Union{Arnoldi,Lanczos}
) where {F<:Union{MultipliedOperator,SumOfOperators}}
    sol, convhist = exponentiate(x -> f(x, t₀ + dt / 2), a * dt, y₀, method)
    return sol, convhist.converged, convhist
end
