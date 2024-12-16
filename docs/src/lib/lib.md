# Library documentation

## [States](@id lib_states)
```@docs
FiniteMPS
InfiniteMPS
WindowMPS
MultilineMPS
```

## Operators
```@docs
AbstractMPO
MPO
MPOHamiltonian
```

## Environments
```@docs
MPSKit.AbstractMPSEnvironments
MPSKit.AbstractInfiniteEnvironments
MPSKit.InfiniteMPOEnvironments
MPSKit.InfiniteMPOHamiltonianEnvironments
MPSKit.FiniteEnvironments
MPSKit.IDMRGEnvironments
```

## Algorithms
```@docs
find_groundstate
timestep
leading_boundary
dynamicaldmrg
changebonds
excitations
approximate
```

### [Groundstate algorithms](@id lib_gs_alg)
```@docs
VUMPS
IDMRG1
IDMRG2
DMRG
DMRG2
GradientGrassmann
```

### [Time evolution algorithms](@id lib_time_alg)
```@docs
TDVP
TDVP2
TaylorCluster
WII
```

### [Leading boundary algorithms](@id lib_bound_alg)
```@docs
VUMPS
VOMPS
GradientGrassmann
```

### [Bond change algorithms](@id lib_bc_alg)
```@docs
OptimalExpand
RandExpand
VUMPSSvdCut
SvdCut
```

### [Excitations](@id lib_ex_alg)
```@docs
QuasiparticleAnsatz
FiniteExcited
```

## Utility
```@docs
left_virtualspace
right_virtualspace
physicalspace
add_util_leg
expectation_value
variance
entanglement_spectrum
entropy
transfer_spectrum
correlation_length
entanglementplot
transferplot
```
