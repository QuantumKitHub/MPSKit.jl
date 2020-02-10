struct DoNothing<:Algorithm
end

"
    Change the bond dimension of state using alg
"
changebonds(state,alg::DoNothing) = state
changebonds(state,H,alg::DoNothing,pars=params(state,H))=(state,pars)
