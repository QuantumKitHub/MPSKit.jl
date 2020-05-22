using MPSKit,TensorKit,Test

using LinearAlgebra;
LinearAlgebra.BLAS.set_num_threads(1);

#by default the vumps we use is parallelized - try running this with different values of JULIA_NUM_THREADS

th = nonsym_ising_ham(lambda=4);
ts = InfiniteMPS([ℂ^2],[ℂ^50]);
@time find_groundstate(ts,th,Vumps(maxiter=400));

th = repeat(th,100);
ts = InfiniteMPS(fill(ℂ^2,100),fill(ℂ^50,100));
@time find_groundstate(ts,th,Vumps(maxiter=400));
