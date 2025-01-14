"""
    GradientGrassmann <: Algorithm

Variational gradient-based optimization algorithm that keeps the MPS in left-canonical form,
as points on a Grassmann manifold. The optimization is then a Riemannian gradient descent 
with a preconditioner to induce the metric from the Hilbert space inner product.

## Fields
- `method::OptimKit.OptimizationAlgorithm`: algorithm to perform the gradient search
- `finalize!`: user-supplied callable which is applied after each iteration, with
    signature `finalize!(x::GrassmannMPS.ManifoldPoint, f, g, numiter) -> x, f, g`

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
struct GradientGrassmann{O<:OptimKit.OptimizationAlgorithm,F} <: Algorithm
    method::O
    finalize!::F

    function GradientGrassmann(; method=ConjugateGradient, (finalize!)=OptimKit._finalize!,
                               tol=Defaults.tol, maxiter=Defaults.maxiter,
                               verbosity=Defaults.verbosity - 1)
        if isa(method, OptimKit.OptimizationAlgorithm)
            # We were given an optimisation method, just use it.
            m = method
        elseif method <: OptimKit.OptimizationAlgorithm
            # We were given an optimisation method type, construct an instance of it.
            # restrict linesearch maxiter
            linesearch = OptimKit.HagerZhangLineSearch(; verbosity=verbosity - 2,
                                                       maxiter=100)
            m = method(; maxiter, verbosity, gradtol=tol, linesearch)
        else
            msg = "method should be either an instance or a subtype of `OptimKit.OptimizationAlgorithm`."
            throw(ArgumentError(msg))
        end
        return new{typeof(m),typeof(finalize!)}(m, finalize!)
    end
end

function find_groundstate(ψ::S, H, alg::GradientGrassmann,
                          envs::P=environments(ψ, H))::Tuple{S,P,Float64} where {S,P}
    !isa(ψ, FiniteMPS) ||
        dim(ψ.C[end]) == 1 ||
        @warn "This is not fully supported - split the mps up in a sum of mps's and optimize seperately"
    normalize!(ψ)

    fg(x) = GrassmannMPS.fg(x, H, envs)
    x, _, _, _, normgradhistory = optimize(fg, ψ,
                                           alg.method;
                                           GrassmannMPS.transport!,
                                           GrassmannMPS.retract,
                                           GrassmannMPS.inner,
                                           GrassmannMPS.scale!,
                                           GrassmannMPS.add!,
                                           GrassmannMPS.precondition,
                                           alg.finalize!,
                                           isometrictransport=true)
    return x, envs, normgradhistory[end]
end
