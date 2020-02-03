struct DoNothing<:Algorithm
end

changebonds(state,alg::DoNothing) = state
changebonds(state,H,alg::DoNothing,pars=params(state,H))=(state,pars)
