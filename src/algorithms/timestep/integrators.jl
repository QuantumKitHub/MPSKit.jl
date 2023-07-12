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
integrate(f,y₀,a,dt,method) = integrate(f,y₀,0.,a,dt,method) 

# wrap function into UntimedOperator by default
integrate(f,y₀,t₀,a,dt,method) = integrate(UntimedOperator(f),y₀,t₀,a,dt,method)

#original integrator in iTDVP, namely exponentiation
function integrate(f::F,y₀,t₀,a,dt,method::Union{Arnoldi,Lanczos}) where {F <: Union{MultipliedOperator,SumOfOperators}}
    sol, convhist = exponentiate(x->f(x,t₀+dt/2),a*dt,y₀,method)
    sol, convhist.converged, convhist
end

"""
    ImplicitMidpoint

Second order and time-reversible method that preserves norm, even for time-dependent driving functions f.

# Fields
- `tol::Float64`: desired tolerance for solving the implicit step via a linsolve
"""
@kwdef struct ImplicitMidpoint
    tol::Float64 = MPSKit.Defaults.tol;
end

function integrate(f::F,y₀,t₀,a,dt,method::ImplicitMidpoint) where {F <: Union{MultipliedOperator,SumOfOperators}}
    y1, info = linsolve(x->f(x,t₀+dt/2),y₀,y₀,1,-0.5*a*dt;tol=method.tol) #solve implicit problem
    y1+0.5*a*dt*f(y1,t₀+dt/2), info.converged, info
end


"""
    Taylor

Taylor series approximation of exp( a*dt*f(y,t) ). Currently only first order is implemented.

# Fields
- `order::Int64`: order of the approximation
"""
@kwdef struct Taylor
    order::Int64 = 1
end

function integrate(f::F,y₀,t₀,a,dt,method::Taylor) where {F <: Union{MultipliedOperator,SumOfOperators}}
    if method.order == 1
        return y₀+a*dt*f(y₀,t₀+dt/2), 1, nothing
    end
end

"""
    RK4

Standard Runge-Kutta 4 numerical integrator

# Fields
- `nh::Int64`: number of time sub-intervals
"""
@kwdef struct RK4
    nh::Int64 = 1
end

function integrate(f::F,y₀,t₀,a,dt,method::RK4) where {F <: Union{MultipliedOperator,SumOfOperators}}
    h = dt/method.nh;

    y = y₀
    t = t₀;
    for i in 1:method.nh
        k₁ = a*f(y,t);
        k₂ = a*f(y + k₁ * h/2,t+h/2);
        k₃ = a*f(y + k₂ * h/2,t+h/2);
        k₄ = a*f(y + k₃ * h,t+h);

        t+=h;
        y+= 1/6*h*(k₁+2*k₂+2*k₃+k₄);
    end
    normalize(y), 1, nothing # normalize because RK4 does not conserve norm
end

