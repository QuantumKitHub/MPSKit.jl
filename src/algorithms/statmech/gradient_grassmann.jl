function leading_boundary(state::InfiniteMPS, H::DenseMPO, alg::GradientGrassmann,
                          envs=environments(state, H))
    (multi, envs, err) = leading_boundary(convert(MPSMultiline, state),
                                          convert(MPOMultiline, H), alg, envs)
    state = convert(InfiniteMPS, multi)
    return (state, envs, err)
end

function leading_boundary(state::MPSMultiline, H, alg::GradientGrassmann,
                          envs=environments(state, H))
    res = optimize(GrassmannMPS.fg,
                   GrassmannMPS.ManifoldPoint(state, envs),
                   alg.method;
                   (transport!)=GrassmannMPS.transport!,
                   retract=GrassmannMPS.retract,
                   inner=GrassmannMPS.inner,
                   (scale!)=GrassmannMPS.scale!,
                   (add!)=GrassmannMPS.add!,
                   (finalize!)=alg.finalize!,
                   #precondition = GrassmannMPS.precondition,
                   isometrictransport=true)
    (x, fx, gx, numfg, normgradhistory) = res
    return x.state, x.envs, normgradhistory[end]
end
