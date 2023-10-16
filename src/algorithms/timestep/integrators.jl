"""
    integrate(f, y₀, t, dt, alg)

Integrate the differential equation ``dy/dt = f(y, t)`` over a time step 'dt' starting from
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

# TODO: properly clean up this mess
const DerivativeOperator = Union{MPO_∂∂C,MPO_∂∂AC,MPO_∂∂AC2}
(h::DerivativeOperator)(x, ::Number) = h(x)

_eval_t(h::DerivativeOperator, t::Number) = h
_eval_x(h::DerivativeOperator, x) = t -> h(x)

# _eval_x(h::SumOfOperators, x) = h(x)

function integrate(f, y₀, t::Number, dt::Number, alg::Union{Arnoldi,Lanczos})
    y, convhist = exponentiate(_eval_t(f, t), dt, y₀, alg)
    convhist.converged == 0 && @warn "integration failed $(convhist.normres)"
    return y
end
