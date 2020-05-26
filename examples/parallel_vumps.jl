using TensorOperations,MPSKit,TensorKit,Test
using LinearAlgebra;
LinearAlgebra.BLAS.set_num_threads(1);

#by default the vumps we use is parallelized - try running this with different values of JULIA_NUM_THREADS
function a()
TensorOperations.enable_cache()

th = nonsym_ising_ham();
th = repeat(th,10);
ts = InfiniteMPS(fill(ℂ^2,10),fill(ℂ^50,10));
@time find_groundstate(ts,th,Vumps(maxiter=100));

end

function b()
TensorOperations.disable_cache()

th = nonsym_ising_ham();
th = repeat(th,10);
ts = InfiniteMPS(fill(ℂ^2,10),fill(ℂ^50,10));
@time find_groundstate(ts,th,Vumps(maxiter=100));

end

a()
b()
