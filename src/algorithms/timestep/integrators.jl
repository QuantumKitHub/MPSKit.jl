"""
    integrate(f, y₀, dt, algorithm)
    integrate(f, y₀, t₀, a, dt, algorithm)

Integrate the differential equation ``dy/dt = a f(y,t)`` over a time step 'dt' starting from ``y(t₀)=y₀``, using the provided algorithm. 
For time-independent 'f' the pre-factor 'a' can be absorbed in the 'dt' part.

# Arguments
- `f::Function`: driving function
- `y₀`: object to integrate
- `dt::Number`: timestep
- `algorithm`: evolution algorithm
"""
function integrate end

#original integrator in iTDVP, namely exponentiation
function integrate(f,y₀,t₀,a,dt,method::Union{Arnoldi,Lanczos})
    sol, convhist = exponentiate(x->f(x,t₀+dt/2),a*dt,y₀,method)
    sol, convhist.converged, convhist
end

"""
    ImplicitMidpoint <: Algorithm

Second order and time-reversible method that preserves norm, even for time-dependent driving functions f.

# Fields
- `tol::Float64`: desired tolerance for the linear problem solution
"""

@kwdef struct ImplicitMidpoint <: MPSKit.Algorithm
    tol::Float64 = MPSKit.Defaults.tol;
end

function integrate(f,y₀,t₀,a,dt,method::ImplicitMidpoint)
    y1, info = linsolve(x->f(x,t₀+dt/2),y₀,y₀,1,-0.5*a*dt;tol=method.tol) #solve implicit problem
    y1+0.5*a*dt*f(y1,t₀+dt/2), info.converged, info
end


"""
    Taylor <: Algorithm

Taylor series approximation of exp( a*dt*f(y,t) ). Currently only first order is implemented.

# Fields
- `order::Int64`: order of the approximation
"""

#Taylor series integrator
@kwdef struct Taylor <: MPSKit.Algorithm
    order::Int64 = 1
end

function integrate(f,y₀,t₀,a,dt,method::Taylor)
    if method.order == 1
        return y₀+a*dt*f(y₀,t₀+dt/2), 1, nothing
    end
end

"""
    RK4 <: Algorithm

Standard Runge-Kutta 4 numerical integrator

# Fields
- `nh::Int64`: number of time sub-intervals
"""

@kwdef struct RK4 <: MPSKit.Algorithm
    nh::Int64 = 1
end

function integrate(f,y₀,t₀,a,dt,method::RK4)
    # nh == number of function evaluations; more should be more accurate.
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
    normalize(y), 1, nothing #because RK4 does not conserve norm
end

# allow time-independence
integrate(f,y₀,dt,method) = integrate((x,t)->f(x),y₀,NaN,1,dt,method)