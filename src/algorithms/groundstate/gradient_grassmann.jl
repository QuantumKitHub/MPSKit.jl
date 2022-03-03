"""
GradientGrassmann is an optimisation methdod that keeps the MPS in left-canonical form, and
treats the tensors as points on Grassmann manifolds. It then applies one of the standard
gradient optimisation methods, e.g. conjugate gradient, to the MPS, making use of the
Riemannian manifold structure. A preconditioner is used, so that effectively the metric used
on the manifold is that given by the Hilbert space inner product.

The arguments to the constructor are
method = OptimKit.ConjugateGradient
    The gradient optimisation method to be used. Should either be an instance or a subtype
    of `OptimKit.OptimizationAlgorithm`. If it's an instance, this `method` is simply used
    to do the optimisation. If it's a subtype, then an instance is constructed as
    `method(; maxiter=maxiter, verbosity=verbosity, gradtol=tol)`

finalize! = OptimKit._finalize!
    A function that gets called once each iteration. See OptimKit for details.

tol = Defaults.tol
maxiter = Defaults.maxiter
verbosity = 2
    Arguments passed to the `method` constructor. If `method` is an instance of
    `OptimKit.OptimizationAlgorithm`, these argument are ignored.

In other words, by default conjugate gradient is used. One can easily set `tol`, `maxiter`
and `verbosity` for it, or switch to LBFGS or gradient descent by setting `method`. If more
control is wanted over things like specifics of the linesearch, CG flavor or the `m`
parameter of LBFGS, then the user should create the `OptimKit.OptimizationAlgorithm`
instance manually and pass it as `method`.
"""
struct GradientGrassmann <: Algorithm
    method::OptimKit.OptimizationAlgorithm
    finalize!::Function

    function GradientGrassmann(; method = ConjugateGradient,
                               finalize! = OptimKit._finalize!,
                               tol = Defaults.tol,
                               maxiter = Defaults.maxiter,
                               verbosity = 2)
        if isa(method, OptimKit.OptimizationAlgorithm)
            # We were given an optimisation method, just use it.
            m = method
        elseif method <: OptimKit.OptimizationAlgorithm
            # We were given an optimisation method type, construct an instance of it.
            m = method(; maxiter=maxiter, verbosity=verbosity, gradtol=tol)
        else
            msg = "method should be either an instance or a subtype of OptimKit.OptimizationAlgorithm."
            throw(ArgumentError(msg))
        end
        return new(m, finalize!)
    end
end

function find_groundstate(state::S, H::HT, alg::GradientGrassmann,
                          envs::P=environments(state, H))::Tuple{S,P,Float64} where {S,HT,P}

    !isa(state,FiniteMPS) || dim(state.CR[end]) == 1 || @warn "This is not fully supported - split the mps up in a sum of mps's and optimize seperately"
    normalize!(state)

    #optimtest(GrassmannMPS.fg,(state,envs);alpha=-0.01:0.001:0.01,retract=GrassmannMPS.retract,inner=GrassmannMPS.inner)
    res = optimize(GrassmannMPS.fg, GrassmannMPS.ManifoldPoint(state, envs), alg.method;
                   transport! = GrassmannMPS.transport!,
                   retract = GrassmannMPS.retract,
                   inner = GrassmannMPS.inner,
                   scale! = GrassmannMPS.scale!,
                   add! = GrassmannMPS.add!,
                   finalize! = alg.finalize!,
                   #precondition = GrassmannMPS.precondition,
                   isometrictransport = true)
    (x, fx, gx, numfg, normgradhistory) = res
    return x.state, x.envs, normgradhistory[end]
end
