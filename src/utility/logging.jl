module IterativeLoggers

export IterLog
export loginit!, logiter!, logfinish!, logcancel!

export format_time

using Printf: @printf, @sprintf

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

# Output
# ------

function format_time(t::Float64)
    return t < 60 ? @sprintf("%0.2f sec", t) :
        t < 2600 ? @sprintf("%0.2f min", t / 60) :
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
