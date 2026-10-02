function leading_boundary(
        state::InfiniteMultilineMPS,
        operator::InfiniteMultilineMPO,
        alg::GradientGrassmann,
        envs::MultilineEnvironments = environments(state, operator, state)
    )
    # read the scheduler here rather than in `fg`, so that the allocator it selects is inferable
    scheduler = Defaults.scheduler[]
    fg(x) = GrassmannMPS.fg(x, operator, envs; alg.backend, scheduler)
    x, f, g, _, normgradhistory = optimize(
        fg, state,
        alg.method;
        GrassmannMPS.transport!,
        GrassmannMPS.retract,
        GrassmannMPS.inner,
        GrassmannMPS.scale!,
        GrassmannMPS.add!,
        GrassmannMPS.precondition,
        alg.finalize!,
        alg.hasconverged,
        alg.shouldstop,
        isometrictransport = true
    )

    info = _optimkit_info(alg, x, f, g, normgradhistory)
    return x, envs, info
end
