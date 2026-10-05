module IterativeLoggers

export IterLog
export loginit!, logiter!, logfinish!, logcancel!
export @log_initialization, @log_iteration, @log_convergence, @log_nonconvergence

export format_time
export with_verbosity

using Printf: @printf, @sprintf
using LoggingExtras: EarlyFilteredLogger, current_logger, with_logger
import LoggingExtras

const VERBOSITY_LEVELS = (
    mpskit_initialization = 2,
    mpskit_iteration = 3,
    mpskit_convergence = 2,
    mpskit_nonconvergence = 1,
    mpskit_warning = 1,
    mpskit_timing = 4,
)

"""
    with_verbosity(f; verbosity::Integer)

Run `f` with MPSKit's numeric verbosity filter. Messages use standard logging macros
with symbolic groups so suppressed messages are filtered before evaluating their contents.
Messages in other groups pass through unchanged. As with `LoggingExtras.withlevel`,
the current logger's severity threshold is temporarily overridden to `Info`.
"""
function with_verbosity(f; verbosity::Integer)
    return LoggingExtras.withlevel() do
        logger = EarlyFilteredLogger(current_logger()) do args
            level = get(VERBOSITY_LEVELS, args.group, nothing)
            return isnothing(level) || verbosity >= level
        end
        return with_logger(f, logger)
    end
end

@enum LogState INIT ITER CONV CANCEL

mutable struct IterLog
    name::AbstractString
    iter::Int
    error::Float64
    objective::Union{Nothing, Number}

    t_init::Float64
    t_prev::Float64
    t_last::Float64

    state::LogState
end
function IterLog(name = "")
    t = Base.time()
    return IterLog(name, 0, NaN, nothing, t, t, t, INIT)
end

# Input
# -----

isapproxreal(x::Number) = isreal(x) || isapprox(imag(x), 0; atol = eps(abs(x))^(3 / 4))
warnapproxreal(x::Number) = isapproxreal(x) || @warn "Objective has imaginary part: $x"

function loginit!(
        log::IterLog, error::Float64, objective::Union{Nothing, Number} = nothing
    )
    log.iter = 0
    log.error = error
    log.objective = objective

    log.t_init = log.t_prev = log.t_last = Base.time()
    log.state = INIT

    return log
end

function logiter!(
        log::IterLog, iter::Int, error::Float64, objective::Union{Nothing, Number} = nothing
    )
    log.iter = iter
    log.error = error
    log.objective = objective

    log.t_prev = log.t_last
    log.t_last = Base.time()
    log.state = ITER

    return log
end

function logfinish!(
        log::IterLog, iter::Int, error::Float64, objective::Union{Nothing, Number} = nothing
    )
    log.iter = iter
    log.error = error
    log.objective = objective

    log.t_prev = log.t_last
    log.t_last = Base.time()
    log.state = CONV

    return log
end

function logcancel!(
        log::IterLog, iter::Int, error::Float64, objective::Union{Nothing, Number} = nothing
    )
    log.iter = iter
    log.error = error
    log.objective = objective

    log.t_prev = log.t_last
    log.t_last = Base.time()
    log.state = CANCEL

    return log
end

# Standard algorithm messages
# ---------------------------

# Expand directly to a standard logging macro at the caller's source location.
function phase_log_expr(source, severity, group, message, args...)
    return Expr(
        :macrocall, GlobalRef(Base, severity), source, message,
        Expr(:(=), :_group, QuoteNode(group)), args...
    )
end

"""
    @log_initialization message args...

Log `message` at `Info` severity in the `:mpskit_initialization` group (verbosity 2).
The message is evaluated only when enabled; standard logging metadata is supported.

```julia
@log_initialization loginit!(log, error, expectation_value(ψ, H, envs))
```
"""
macro log_initialization(message, args...)
    return esc(phase_log_expr(__source__, Symbol("@info"), :mpskit_initialization, message, args...))
end

"""
    @log_iteration message args...

Log `message` at `Info` severity in the `:mpskit_iteration` group (verbosity 3).
The message is evaluated only when enabled; standard logging metadata is supported.
"""
macro log_iteration(message, args...)
    return esc(phase_log_expr(__source__, Symbol("@info"), :mpskit_iteration, message, args...))
end

"""
    @log_convergence message args...

Log `message` at `Info` severity in the `:mpskit_convergence` group (verbosity 2).
The message is evaluated only when enabled; standard logging metadata is supported.
"""
macro log_convergence(message, args...)
    return esc(phase_log_expr(__source__, Symbol("@info"), :mpskit_convergence, message, args...))
end

"""
    @log_nonconvergence message args...

Log `message` at `Warn` severity in the `:mpskit_nonconvergence` group (verbosity 1).
The message is evaluated only when enabled; standard logging metadata is supported.
"""
macro log_nonconvergence(message, args...)
    return esc(phase_log_expr(__source__, Symbol("@warn"), :mpskit_nonconvergence, message, args...))
end

# Output
# ------

function format_time(t::Float64)
    return t < 60 ? @sprintf("%0.2f sec", t) :
        t < 3600 ? @sprintf("%0.2f min", t / 60) :
        @sprintf("%0.2f hr", t / 3600)
end

function format_objective(t::Number)
    if isapproxreal(t)
        return @sprintf("%+0.12e", real(t))
    else
        return @sprintf("%+0.12e %+0.12eim", real(t), imag(t))
    end
end

# defined to make standard logging behave nicely
function Base.show(io::IO, log::IterLog)
    if log.state === INIT
        if isnothing(log.objective)
            return @printf io "%s init:\terr = %0.4e" log.name log.error
        else
            obj_str = format_objective(log.objective)
            return @printf io "%s init:\tobj = %s\terr = %0.4e" log.name obj_str log.error
        end
    elseif log.state === CONV
        Δt_str = format_time(log.t_last - log.t_init)
        if isnothing(log.objective)
            return @printf io "%s conv %d:\terr = %0.10e\ttime = %s" log.name log.iter log.error Δt_str
        else
            obj_str = format_objective(log.objective)
            return @printf io "%s conv %d:\tobj = %s\terr = %0.10e\ttime = %s" log.name log.iter obj_str log.error Δt_str
        end
    elseif log.state === ITER
        Δt_str = format_time(log.t_last - log.t_prev)
        if isnothing(log.objective)
            return @printf io "%s %3d:\terr = %0.10e\ttime = %s" log.name log.iter log.error Δt_str
        else
            obj_str = format_objective(log.objective)
            return @printf io "%s %3d:\tobj = %s\terr = %0.10e\ttime = %s" log.name log.iter obj_str log.error Δt_str
        end
    elseif log.state === CANCEL
        Δt_str = format_time(log.t_last - log.t_init)
        if isnothing(log.objective)
            return @printf io "%s cancel %d:\terr = %0.10e\ttime = %s" log.name log.iter log.error Δt_str
        else
            obj_str = format_objective(log.objective)
            return @printf io "%s cancel %d:\tobj = %s\terr = %0.10e\ttime = %s" log.name log.iter obj_str log.error Δt_str
        end
    end
    return nothing
end

end
