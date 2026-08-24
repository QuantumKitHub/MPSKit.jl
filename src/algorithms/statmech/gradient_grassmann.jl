function leading_boundary(
        state::MultilineMPS,
        operator::MultilineMPO,
        alg::GradientGrassmann,
        envs::MultilineEnvironments = environments(state, operator, state)
    )
    # read the scheduler here rather than in `fg`, so that the allocator it selects is inferable
    scheduler = Defaults.scheduler[]
    fg(x) = GrassmannMPS.fg(x, operator, envs; alg.backend, scheduler)
    x, _, _, _, normgradhistory = optimize(
        fg, state,
        alg.method;
        GrassmannMPS.transport!,
        GrassmannMPS.retract,
        GrassmannMPS.inner,
        GrassmannMPS.scale!,
        GrassmannMPS.add!,
        GrassmannMPS.precondition,
        alg.finalize!,
        isometrictransport = true
    )

    normres = normgradhistory[end, 2] # full history returned as [fhistory normgradhistory]
    info = AlgorithmInfo(;
        converged = normres <= alg.method.gradtol, normres,
        numiter = size(normgradhistory, 1) - 1 # history starts with initial point before first iteration
    )
    return x, envs, info
end
