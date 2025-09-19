"""
    integrate(f, y₀, t, dt, alg)

Integrate the differential equation ``i dy/dt = f(y, t)`` over a time step 'dt' starting from
``y(t₀)=y₀``, using the provided algorithm.

# Arguments
- `f`: driving function
- `y₀`: object to integrate
- `t::Number`: starting time of time-step
- `dt::Number`: time-step magnitude
- `alg`: integration scheme
"""
function integrate end

# default for things that are callable on two arguments
_eval_t(f, t::Number) = Base.Fix2(f, t)
_eval_x(f, x) = Base.Fix1(f, x)

function integrate(
        f, y₀, t::Number, dt::Number, alg::Union{Arnoldi, Lanczos};
        imaginary_evolution::Bool = false
    )
    dt′ = imaginary_evolution ? -dt : -1im * dt
    y, convhist = exponentiate(_eval_t(f, t + dt / 2), dt′, y₀, alg)
    convhist.converged == 0 &&
        @warn "integrator failed to converge: normres = $(convhist.normres)"
    return y
end
