function correlation_length(state::InfiniteMPS; otherstate=state, tol = 1e-8, sector = one(sectortype(virtualspace(state,1))), tol_angle = 0.1, max_sinvals = 20)

   #the first eigenvalue tells us what angle to search at for the leading nontrivial eigenvalue
    init = TensorMap(rand, eltype(state), virtualspace(otherstate,0)*ℂ[sector => 1],virtualspace(state,0))

    #this is potentially still wrong. The idea is that we can at most ask D evals, with D the dimension of the transfer matrix
    numvals = min(dim(virtualspace(state,0)*virtualspace(state,0)),max_sinvals);

    eigenvals, eigenvecs,convhist = eigsolve(init, numvals, :LM, tol=tol) do x
		return transfer_left(x, state.AL, otherstate.AL)
    end
    convhist.converged < numvals && @warn "correlation length failed to converge $(convhist.normres)"

    (state === otherstate) && (eigenvals = eigenvals[2:end])

    best_angle = mod1(angle(eigenvals[1]), 2*pi)
    ind_at_angle = findall(x->x<tol_angle || abs(x-2*pi)<tol_angle, mod1.(angle.(eigenvals).-best_angle, 2*pi))
    eigenvals_at_angle = eigenvals[ind_at_angle]

    lambda2 = -log(abs(eigenvals_at_angle[1]))

    lambda3 = Inf;
	if length(eigenvals_at_angle) > 1
        lambda3 = -log(abs(eigenvals_at_angle[2]))
    end

    return 1/lambda2, lambda3-lambda2, best_angle,eigenvals
end
