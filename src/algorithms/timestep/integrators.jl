#original integrator in iTDVP, namely exponentiation
function integrate(f,y₀,t₀,a,dt,method::Union{Arnoldi,Lanczos})
    exponentiate(x->f(x,t₀+dt/2),a*dt,y₀,method)[1]
end

#Taylor series integrator
@with_kw struct Taylor <: MPSKit.Algorithm
    order::Int64 = 1
end

function integrate(f,y₀,t₀,a,dt,method::Taylor)
    if method.order == 1
        return y₀+a*dt*f(y₀,t₀+dt/2)
    end
end

#Implicit midpoint method is second order and time-reversible and preserves norm, even for time-dependent H
@with_kw struct IM <: MPSKit.Algorithm
    tol::Float64 = MPSKit.Defaults.tol;
end

function integrate(f,y₀,t₀,a,dt,method::IM) #implicit midpoint method
    y1,info = linsolve(x->f(x,t₀+dt/2),y₀,y₀,1,-0.5*a*dt;tol=method.tol) #solve implicit problem
    if iszero(info.converged) @show info.converged end
    y1+0.5*a*dt*f(y1,t₀+dt/2)
end

#%% runge kutta 4 is a generic numerical integrator
@with_kw struct RK4 <: MPSKit.Algorithm
    tol::Float64 = MPSKit.Defaults.tol;
end

function integrate(f,y₀,t₀,a,dt,method::RK4)
    # nh == number of function evaluations; more should be more accurate.
    # here I kind of guestimate the necessary number
    h = (method.tol/abs(dt))^(1/4)
    nh = ceil(abs(dt)/h);
    h = dt/nh;

    y = y₀
    t = t₀;
    for i in 1:nh
        k₁ = a*f(y,t);
        k₂ = a*f(y + k₁ * h/2,t+h/2);
        k₃ = a*f(y + k₂ * h/2,t+h/2);
        k₄ = a*f(y + k₃ * h,t+h);

        t+=h;
        y+= 1/6*h*(k₁+2*k₂+2*k₃+k₄);
    end
    normalize(y) #because RK4 does not conserve norm
end

#=
using DifferentialEquations

# struct that holds a DifferentialEquations solver, to be used during integration of the tdvp equations
# not debugged and not benchmarked, use at own risk
@with_kw struct DEIntegrator{O}
    integrator::O=nothing
end


function _map2vec(y::TensorMap)
    if y.data isa Array
        reshape(y.data,length(y.data))
    else
        reduce(vcat,map(x->reshape(x,length(x)),values(y.data)))
    end
end

function _map2vec!(out::Vector,y::TensorMap)
    if y.data isa Array
        out[:] .= y.data[:]
    else
        i = 1;
        for d in values(t.data)
            v[i:length(d)+i-1].=d[:];
            i+=length(d)
        end
    end
    out
end

function _map2tens!(t::TensorMap,v::Vector)
    if t.data isa Array
        t.data[:].=v[:]
    else
        i = 1;
        for d in values(t.data)
            d[:].=v[i:length(d)+i-1];
            i+=length(d)
        end
    end
    t
end

function integrate(f,y₀,t₀,a,dt,method::DEIntegrator)
    wrapped_f!(dv,v,p,t) = _map2vec!(dv,a*f(_map2tens!(similar(y₀),v),t))

    problem = ODEProblem(wrapped_f!,_map2vec(y₀),(t₀,t₀+dt))
    solution = solve(problem,method.integrator)
    _map2tens!(similar(y₀),solution[end])
end
=#