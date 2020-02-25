#credits : https://github.com/ffreyer/MonteCarlo.jl

"""
    @bm function ... end
    @bm foo(args...) = ...

Wraps the body of a function with `@timeit_debug <function name> begin ... end`.

The `@timeit_debug` macro can be disabled. When it is, it should come with zero
overhead. To enable timing, use `TimerOutputs.enable_debug_timings(<module>)`.
See TimerOutputs.jl for more details.

Benchmarks/Timings can be retrieved using `print_timer()` and reset with
`reset_timer!()`. It should be no problem to add additonal `@timeit_debug` to
a function.
"""
macro bm(func)
    esc(Expr(
        func.head,     # function or =
        func.args[1],  # function name w/ args
        quote                                  # name of function
            MPSKit.@timeit_debug $(string(func.args[1].args[1])) begin
                $(func.args[2]) # function body
            end
        end
    ))
end

timeit_debug_enabled() = false

"""
    enable_benchmarks()

Enables benchmarking for `MPSKit`.

This affects every function with the `MPSKit.@bm` macro as well as any
`TimerOutputs.@timeit_debug` blocks. Benchmarks are recorded to the default
TimerOutput `TimerOutputs.DEFAULT_TIMER`. Results can be printed via
`TimerOutputs.print_timer()`.

[`disable_benchmarks`](@ref)
"""
enable_benchmarks() = TimerOutputs.enable_debug_timings(MPSKit)

"""
    disable_benchmarks()

Disables benchmarking for `MPSKit`.

This affects every function with the `MonteCarlo.@bm` macro as well as any
`TimerOutputs.@timeit_debug` blocks.

[`enable_benchmarks`](@ref)
"""
disable_benchmarks() = TimerOutputs.disable_debug_timings(MPSKit)
