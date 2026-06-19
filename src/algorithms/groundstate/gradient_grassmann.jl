"""
$(TYPEDEF)

Variational gradient-based optimization algorithm that keeps the MPS in left-canonical form,
as points on a Grassmann manifold. The optimization is then a Riemannian gradient descent 
with a preconditioner to induce the metric from the Hilbert space inner product.

## Fields

$(TYPEDFIELDS)

## References

* [Hauru et al. SciPost Phys. 10 (2021)](@cite hauru2021)

---

## Constructors
    GradientGrassmann(; kwargs...)

### Keywords
- `method=ConjugateGradient`: instance of optimization algorithm, or type of optimization
    algorithm to construct
- `finalize!`: finalizer algorithm
- `tol::Float64`: tolerance for convergence criterium
- `maxiter::Int`: maximum amount of iterations
- `verbosity::Int`: level of information display
"""
struct GradientGrassmann{O <: OptimKit.OptimizationAlgorithm, F} <: Algorithm
    "optimization algorithm"
    method::O
    "callback function applied after each iteration, of signature `finalize!(x, f, g, numiter) -> x, f, g`"
    finalize!::F

    function GradientGrassmann(;
            method = ConjugateGradient, (finalize!) = OptimKit._finalize!,
            tol = Defaults.tol, maxiter = Defaults.maxiter,
            verbosity = Defaults.verbosity - 1
        )
        if isa(method, OptimKit.OptimizationAlgorithm)
            # We were given an optimisation method, just use it.
            m = method
        elseif method <: OptimKit.OptimizationAlgorithm
            # We were given an optimisation method type, construct an instance of it.
            # restrict linesearch maxiter
            linesearch = OptimKit.HagerZhangLineSearch(;
                verbosity = verbosity - 2, maxiter = 100
            )
            m = method(; maxiter, verbosity, gradtol = tol, linesearch)
        else
            msg = "method should be either an instance or a subtype of `OptimKit.OptimizationAlgorithm`."
            throw(ArgumentError(msg))
        end
        return new{typeof(m), typeof(finalize!)}(m, finalize!)
    end
end

function find_groundstate(
        style::GeometryStyle, ψ::S, H, alg::GradientGrassmann,
        envs::P = environments(ψ, H)
    )::Tuple{S, P, Float64} where {S, P}
    !(style isa FiniteChainStyle) || dim(ψ.C[end]) == 1 ||
        @warn "This is not fully supported - split the mps up in a sum of mps's and optimize separately"
    normalize!(ψ)

    timeroutput = TimerOutput("GradientGrassmann")
    method_verbosity = hasproperty(alg.method, :verbosity) ? alg.method.verbosity : 0
    method_verbosity > 3 || disable_timer!(timeroutput)

    fg(x) = timeit(() -> GrassmannMPS.fg(x, H, envs; timeroutput), timeroutput, "fg")
    retract(state, g, α) = timeit(
        () -> GrassmannMPS.retract(state, g, α), timeroutput, "retract",
    )
    transport!(h, state, g, α, state′) = timeit(
        () -> GrassmannMPS.transport!(h, state, g, α, state′), timeroutput, "transport!",
    )
    precondition(state, g) = timeit(
        () -> GrassmannMPS.precondition(state, g), timeroutput, "precondition",
    )

    x, _, _, _, normgradhistory = optimize(
        fg, ψ, alg.method;
        retract, transport!, precondition,
        GrassmannMPS.inner,
        GrassmannMPS.scale!,
        GrassmannMPS.add!,
        alg.finalize!,
        isometrictransport = true,
    )

    LoggingExtras.withlevel(; verbosity = method_verbosity) do
        @infov 4 timeroutput
    end

    return x, envs, normgradhistory[end]
end
