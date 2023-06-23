using MPSKit,MPSKitModels,TensorKit,Test
using Revise


# some general things

D = 20
g = 0.5

th = transverse_field_ising(;J=1,hx = g);
szt = TensorMap([1 0;0 -1],ℂ^2,ℂ^2)
sxt = TensorMap([0 1;1 0],ℂ^2,ℂ^2)

ts = InfiniteMPS([ℂ^2],[ℂ^D]);

#########################
# Linearity test
#########################
#tests for timedoperator

O1 = TensorMap(rand,ComplexF64,ℂ^4*ℂ^4,ℂ^4*ℂ^4);
O2 = TensorMap(rand,ComplexF64,ℂ^4*ℂ^4,ℂ^4*ℂ^4);
f1(t) = 3*exp(t) #take nicer behaved function
timedo1 = TimedOperator(O1,f1)

@test timedo1(5) == f1(5) * O1

## time-dependence of operator

## time-dependence of hamiltonian
H0 =  transverse_field_ising(;J=1,hx = 0);
Ht =  TimedOperator(transverse_field_ising(;J=1,hx = 0),f1);

# the corresponding environment
envs = environments(ts,H0);
envst = environments(ts,Ht);

# test linearity here
expectation_value(ts,H0,envs)
expectation_value(ts,Ht,0.,envs)
expectation_value(ts,Ht(0.),envs) #should be the same

#add here other derivatives
hc = MPSKit.∂∂C(1,ts,H0,envs);
hct = MPSKit.∂∂C(1,ts,Ht,envst);

@test norm(hct(ts.CR[1],0) - f1(0)*hc(ts.CR[1])) < 1e-12

##########################
#tests for sumofoperators
##########################

f2(t) = 3/t #take nicer behaved function
timedo2 = TimedOperator(O2,f2);

timedos = SumOfOperators([timedo1,timedo2])

@test timedos(5) == f1(5) * O1 + f2(5) * O2 #write a bigger test case

# test expectation_value
summedH = Ht + TimedOperator(transverse_field_ising(;J=1,hx = 0),f2); # == ( f1(t)+f2(t) ) H0
summedEnvs = environments(ts,summedH);
@test abs(expectation_value(ts, summedH,5,summedEnvs)[1] - (f1(5)+f2(5) ) * expectation_value(ts, H0,envs)[1]) < 1e-12

# test derivatives
summedhct = MPSKit.∂∂C(1,ts,summedH,summedEnvs);

@test norm(summedhct(ts.CR[1],0.1) - (f1(0.1)+f2(0.1) )*hc(ts.CR[1])) < 1e-10


#######################
# time evolution
#######################

# the algorithm we will use for time evolution
# expalg performs the integration of the tdvp equations
# here we choose the implicit midpoint method IM() (others are availible/ easily implemented see integrators.jl)
alg = TDVP(expalg=ImplicitMidpoint())
dt  = 0.001

#first check that for trivial time-dependence we get the same
H0 =  transverse_field_ising(;J=1,hx = 0);
envs = environments(ts,H0);
summedH =  TimedOperator(transverse_field_ising(;J=0.5,hx = 0)) +  TimedOperator(transverse_field_ising(;J=0.5,hx = 0));
summedEnvs = environments(ts,summedH);

ts_dt_H0,envs_dt0 = timestep(ts,H0,dt,alg,envs);

Ht =  TimedOperator(transverse_field_ising(;J=1,hx = 0));
envst = environments(ts,Ht);
ts_dt,envst = timestep(ts,Ht,0.,dt,alg,envst);

@test abs(expectation_value(ts_dt, Ht,0.,envst)[1]- expectation_value(ts_dt_H0, H0,envs_dt0)[1]) < 1e-12

ts_dt_H ,summedEnvs  = timestep(ts,summedH,0.,dt,alg,summedEnvs);

@test abs(sum(expectation_value(ts_dt_H, summedH,0.,summedEnvs))- sum(expectation_value(ts_dt_H0, H0,envs_dt0))) < 1e-12

@assert false

##################
# Copy tests
##################

ts = InfiniteMPS([ℂ^2],[ℂ^20]);
ts_copied = copy(ts);

norm(ts)
ts.AC[1] *= 2;
norm(ts)
@test abs(norm(ts_copied) - norm(ts)) > 1e-05

tswindow = WindowMPS(InfiniteMPS([ℂ^2],[ℂ^20]),10);

@test tswindow.left_gs !== tswindow.right_gs
@test tswindow.left_gs == tswindow.right_gs #this should be right behaviour I think

tswindow.left_gs.AC[1] *= 2;
@test abs(norm(tswindow.left_gs) - norm(tswindow.right_gs)) > 1e-05

norm(tswindow.left_gs)
norm(tswindow.right_gs)

##################
# WindowMPS
##################
# first let's try to put all different Hs and envs together into a Window
Hleft =  TimedOperator(transverse_field_ising(;J=1,hx = 0.5));
Hmiddle =  TimedOperator(repeat(transverse_field_ising(;J=1,hx = 0.5),20));
Hright = TimedOperator(transverse_field_ising(;J=1,hx = 0.5));

gs,envs = find_groundstate(ts,Hleft(0.),VUMPS(maxiter=400));
expectation_value(gs,Hleft,0.,envs)
gswindow = WindowMPS(gs,20);

Hwindow = Window(Hleft,Hmiddle,Hright)
WindEnvs = environments(gswindow,Hwindow)
typeof(WindEnvs)


# we should also test this with SumOfOperators



