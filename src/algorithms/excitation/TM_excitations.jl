function correlation_length(state::InfiniteMPS; otherstate=state, tol = 1e-8, exi_space = ComplexSpace(1), tol_angle = 0.1, triv=true)
    ALs1 = state.AL
    ALs2 = otherstate.AL

   #the first eigenvalue tells us what angle to search at for the leading nontrivial eigenvalue
    init = TensorMap(rand, ComplexF64, space(ALs2[1],1)*exi_space,space(ALs1[1],1))

    eigenvals, _ = eigsolve(init, 20, :LM, tol=tol) do x
		return transfer_left(x, ALs1, ALs2)
    end

	triv && (eigenvals = eigenvals[2:end])

    best_angle = mod1(angle(eigenvals[1]), 2*pi)
    ind_at_angle = findall(x->x<0.1 || abs(x-2*pi)<0.1, abs.(mod1.(angle.(eigenvals), 2*pi).-best_angle))
    eigenvals_at_angle = eigenvals[ind_at_angle]

	if length(eigenvals_at_angle) == 1
    	lambda2 = -log(norm(eigenvals_at_angle[1]))
    	return 1/lambda2, Inf,best_angle, eigenvals
	else
    	lambda2 = -log(norm(eigenvals_at_angle[1]))
    	lambda3 = -log(norm(eigenvals_at_angle[2]))
    	return 1/lambda2, lambda3-lambda2,best_angle, eigenvals
	end
end
