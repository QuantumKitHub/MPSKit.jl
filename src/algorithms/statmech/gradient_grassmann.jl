function leading_boundary(
        state::MultilineMPS,
        operator::MultilineMPO,
        alg::GradientGrassmann,
        envs::MultilineEnvironments = environments(state, H)
    )
    fg(x) = GrassmannMPS.fg(x, operator, envs)
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
