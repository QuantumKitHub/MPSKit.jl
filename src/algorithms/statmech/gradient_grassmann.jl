function leading_boundary(state::InfiniteMPS, H::DenseMPO, alg::GradientGrassmann,
                          envs=environments(state, H))
    (multi, envs, err) = leading_boundary(convert(MultilineMPS, state),
                                          convert(MultilineMPO, H), alg, envs)
    state = convert(InfiniteMPS, multi)
    return (state, envs, err)
end

function leading_boundary(state::MultilineMPS, H, alg::GradientGrassmann,
                          envs=environments(state, H))
    fg(x) = GrassmannMPS.fg(x, H, envs)
    x, _, _, _, normgradhistory = optimize(fg, state,
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
