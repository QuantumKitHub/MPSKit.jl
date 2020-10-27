using MPSKit,MPSKitModels,TensorKit,Test

#we pass this function to dmrg, it gets executed once per iteration
function finalize!(iter,state,ham,envs)
    final_bonddim = 12;

    upperbound = max_Ds(state);
    shouldincrease = reduce((a,i) -> a && dim(virtualspace(state,i)) >= final_bonddim || upperbound[i+1] == dim(virtualspace(state,i)),1:length(state),init=true);

    if (shouldincrease)
        (state,envs) = changebonds!(state, ham, OptimalExpand(),envs);
    end

    return (state,envs,true)
end

#defining the hamiltonian
th = nonsym_ising_ham(lambda = 4.0);
szt = TensorMap([1 0;0 -1],ℂ^2,ℂ^2)

#finite mps (dmrg)
ts = FiniteMPS(fill(TensorMap(rand,ComplexF64,ℂ^1*ℂ^2,ℂ^1),10));
(ts,envs,_) = find_groundstate!(ts,th,Dmrg(finalize! = finalize!));

szval_finite = sum(expectation_value(ts,szt))/length(ts)
@test szval_finite ≈ 0 atol=1e-12

#twosite dmrg
ts = FiniteMPS(fill(TensorMap(rand,ComplexF64,ℂ^1*ℂ^2,ℂ^1),10));
(ts,envs,_) = find_groundstate!(ts,th,Dmrg2(trscheme = truncdim(15)));

szval_finite = sum(expectation_value(ts,szt))/length(ts)
@test szval_finite ≈ 0 atol=1e-12

#uniform mps
ts=InfiniteMPS([ℂ^2],[ℂ^50]);
(ts,envs,_) = find_groundstate!(ts,th,Vumps(maxiter=400));

szval_infinite = sum(expectation_value(ts,szt))/length(ts)
@test szval_infinite ≈ 0 atol=1e-12

# gradient optimisation using the Grassmann manifold structure
ts = InfiniteMPS([ℂ^2], [ℂ^5]);
(ts, envs, _) = find_groundstate(ts, th, GradientGrassmann(maxiter=400));

szval_infinite = sum(expectation_value(ts,szt))/length(ts)
@test szval_infinite ≈ 0 atol=1e-12
