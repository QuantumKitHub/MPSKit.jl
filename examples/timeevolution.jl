using MPSKit,MPSKitModels,TensorKit,Test

#local redefinition
function nonsym_ising(; J = 1, g = 0.5)
    (sx, _, sz) = nonsym_spintensors(1//2)

    MPOHamiltonian(LocalOperator(-4*J * sz ⊗ sz, (1, 2)) +
                   LocalOperator(-2*g * sx, (1,)))
end

# for now this uses the old MPSKitModels code

(sx, _, sz) = nonsym_spintensors(1//2)
σₓ = 2*sx; σz = 2*sz  #factor 2 to get pauli matrices
J = 1; g = 0.5

HJ = MPOHamiltonian(LocalOperator(-J*σz ⊗ σz, (1, 2)))
Hg = MPOHamiltonian(LocalOperator(-g*σₓ, (1,)))
H₀ = HJ + Hg

gs = InfiniteMPS([2],[50]); #create MPS with physical bond dimension d=2 and virtual D=50
(gs,envs,_) = find_groundstate(gs,H₀,VUMPS(maxiter=400));

sx_gs = sum(expectation_value(gs,σₓ))/length(gs)
@show sx_gs
E_gs = sum(expectation_value(gs,H₀,envs))/length(gs)
@show E_gs

# some time function that slowly ramps
f    = t -> t==0. ? 1 : 1+2*min(0.1*t,2.)

# time dependent Hamiltonian = HJ+f(t)*Hg
Hₜ    = TimeDepProblem((HJ,Hg),(t->1,f));

# the corresponding environment
envs = environments(gs,Hₜ);

# the algorithm we will use for time evolution
# expalg performs the integration of the tdvp equations
# here we choose the implicit midpoint method IM() (others are availible/ easily implemented see integrators.jl)
alg = TDVP(expalg=IM())

# timestep and how many steps
dt  = 0.001
N   = 100

# containers for observables
sxs = zeros(ComplexF64,N+1)
Es  = zeros(ComplexF64,N+1)

# the actual time evolution
let nstate = copy(gs), nenvs = environments(gs,Hₜ), t=0.
    for i in 1:N+1
        sxs[i] = expectation_value(nstate,σₓ,1) 
        Es[i]  = expectation_value(nstate,Hₜ,t,nenvs)[1]
        nstate,nenvs = timestep(nstate,Hₜ,t,dt,alg,nenvs);
        t += dt
    end
end

#check that for t=0 before timestep everything is same as in gs
@test real(sxs[1]) ≈ real(sx_gs)
@test real(Es[1])  ≈ real(E_gs)

# numerical free fermion solution @ t=0.1
sxt = 0.25869106309840484
Et  = -1.0661310695057724

@show abs(real(sxs[end])/sxt-1)
@show abs(real(Es[end])/Et-1)
