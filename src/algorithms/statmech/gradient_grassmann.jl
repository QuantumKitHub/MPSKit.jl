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
    return x, envs, normgradhistory[end]
end
