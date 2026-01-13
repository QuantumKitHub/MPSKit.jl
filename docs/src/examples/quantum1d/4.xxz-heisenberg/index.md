```@meta
EditURL = "../../../../../examples/quantum1d/4.xxz-heisenberg/main.jl"
```

[![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QuantumKitHub/MPSKit.jl/gh-pages?filepath=dev/examples/quantum1d/4.xxz-heisenberg/main.ipynb)
[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](https://nbviewer.jupyter.org/github/QuantumKitHub/MPSKit.jl/blob/gh-pages/dev/examples/quantum1d/4.xxz-heisenberg/main.ipynb)
[![](https://img.shields.io/badge/download-project-orange)](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/QuantumKitHub/MPSKit.jl/examples/tree/gh-pages/dev/examples/quantum1d/4.xxz-heisenberg)

# The XXZ model

In this file we will give step by step instructions on how to analyze the spin 1/2 XXZ model.
The necessary packages to follow this tutorial are:

````julia
using MPSKit, MPSKitModels, TensorKit, Plots
````

## Failure

First we should define the hamiltonian we want to work with.
Then we specify an initial guess, which we then further optimize.
Working directly in the thermodynamic limit, this is achieved as follows:

````julia
H = heisenberg_XXX(; spin = 1 // 2)
````

````
single site InfiniteMPOHamiltonian{MPSKit.JordanMPOTensor{ComplexF64, TensorKit.ComplexSpace, Union{TensorKit.BraidingTensor{ComplexF64, TensorKit.ComplexSpace}, TensorKit.TensorMap{ComplexF64, TensorKit.ComplexSpace, 2, 2, Vector{ComplexF64}}}, TensorKit.TensorMap{ComplexF64, TensorKit.ComplexSpace, 2, 1, Vector{ComplexF64}}, TensorKit.TensorMap{ComplexF64, TensorKit.ComplexSpace, 1, 2, Vector{ComplexF64}}, TensorKit.TensorMap{ComplexF64, TensorKit.ComplexSpace, 1, 1, Vector{ComplexF64}}}}:
╷  ⋮
┼ W[1]: 3×1×1×3 JordanMPOTensor(((ℂ^1 ⊞ ℂ^3 ⊞ ℂ^1) ⊗ ⊞(ℂ^2)) ← (⊞(ℂ^2) ⊗ (ℂ^1 ⊞ ℂ^3 ⊞ ℂ^1)))
╵  ⋮

````

We then need an intial state, which we shall later optimize. In this example we work directly in the thermodynamic limit.

````julia
state = InfiniteMPS(2, 20)
````

````
single site InfiniteMPS:
│   ⋮
│ C[1]: TensorMap(ℂ^20 ← ℂ^20)
├── AL[1]: TensorMap((ℂ^20 ⊗ ℂ^2) ← ℂ^20)
│   ⋮

````

The ground state can then be found by calling `find_groundstate`.

````julia
groundstate, cache, delta = find_groundstate(state, H, VUMPS());
````

````
[ Info: VUMPS init:	obj = +2.499995549467e-01	err = 1.8944e-03
[ Info: VUMPS   1:	obj = -1.637326006220e-01	err = 3.4380167886e-01	time = 0.02 sec
[ Info: VUMPS   2:	obj = +4.022469296605e-02	err = 3.6124904819e-01	time = 0.02 sec
[ Info: VUMPS   3:	obj = -1.451781276155e-01	err = 3.7525843588e-01	time = 0.02 sec
[ Info: VUMPS   4:	obj = -3.297874086946e-01	err = 3.0974732609e-01	time = 0.03 sec
[ Info: VUMPS   5:	obj = +8.480531607776e-02	err = 4.2311149602e-01	time = 0.02 sec
[ Info: VUMPS   6:	obj = -4.682900333525e-02	err = 3.9435625697e-01	time = 0.03 sec
[ Info: VUMPS   7:	obj = -2.038899540195e-01	err = 3.7392529829e-01	time = 0.02 sec
[ Info: VUMPS   8:	obj = -1.567891704735e-01	err = 3.9440419459e-01	time = 0.02 sec
[ Info: VUMPS   9:	obj = +1.798128191908e-02	err = 3.8563879385e-01	time = 0.02 sec
[ Info: VUMPS  10:	obj = -1.627417051509e-02	err = 4.0179419114e-01	time = 0.02 sec
[ Info: VUMPS  11:	obj = +1.971444286808e-01	err = 3.1358642027e-01	time = 0.02 sec
[ Info: VUMPS  12:	obj = -1.544732012749e-01	err = 3.9113131940e-01	time = 0.03 sec
[ Info: VUMPS  13:	obj = -5.143757541054e-02	err = 3.6158026015e-01	time = 0.02 sec
[ Info: VUMPS  14:	obj = -2.378431775937e-01	err = 3.6684775265e-01	time = 0.15 sec
[ Info: VUMPS  15:	obj = -2.335972628792e-01	err = 3.5833112567e-01	time = 0.02 sec
[ Info: VUMPS  16:	obj = -3.513837130591e-01	err = 3.1447160955e-01	time = 0.03 sec
[ Info: VUMPS  17:	obj = -1.269547785447e-01	err = 4.0560571685e-01	time = 0.03 sec
[ Info: VUMPS  18:	obj = -5.526651111523e-02	err = 3.9877496947e-01	time = 0.02 sec
[ Info: VUMPS  19:	obj = -1.736900085076e-01	err = 3.7193659518e-01	time = 0.02 sec
[ Info: VUMPS  20:	obj = -2.041914181357e-01	err = 3.5751923396e-01	time = 0.02 sec
[ Info: VUMPS  21:	obj = -3.693165536045e-01	err = 2.7182712254e-01	time = 0.03 sec
[ Info: VUMPS  22:	obj = -2.011720710870e-01	err = 3.6676865727e-01	time = 0.02 sec
[ Info: VUMPS  23:	obj = -1.633950485920e-01	err = 4.1193224347e-01	time = 0.02 sec
[ Info: VUMPS  24:	obj = -4.626612529510e-02	err = 4.0924314691e-01	time = 0.03 sec
[ Info: VUMPS  25:	obj = -1.281611280890e-01	err = 3.9044251366e-01	time = 0.02 sec
[ Info: VUMPS  26:	obj = -3.368232288231e-01	err = 3.2045176746e-01	time = 0.02 sec
[ Info: VUMPS  27:	obj = -5.824695053770e-02	err = 3.9279884760e-01	time = 0.02 sec
[ Info: VUMPS  28:	obj = -1.584622209274e-01	err = 3.9976472773e-01	time = 0.11 sec
[ Info: VUMPS  29:	obj = -1.002822977314e-01	err = 3.8803754727e-01	time = 0.03 sec
[ Info: VUMPS  30:	obj = +7.400348453278e-03	err = 3.9371296126e-01	time = 0.03 sec
[ Info: VUMPS  31:	obj = -1.177431901110e-01	err = 3.9678690292e-01	time = 0.02 sec
[ Info: VUMPS  32:	obj = -1.648756258411e-01	err = 3.8382308262e-01	time = 0.03 sec
[ Info: VUMPS  33:	obj = -1.486770429634e-01	err = 3.6066618432e-01	time = 0.02 sec
[ Info: VUMPS  34:	obj = -3.115577559394e-01	err = 3.4055409376e-01	time = 0.03 sec
[ Info: VUMPS  35:	obj = -3.431437826768e-01	err = 3.1792959691e-01	time = 0.03 sec
[ Info: VUMPS  36:	obj = -4.164170397906e-01	err = 2.0323232703e-01	time = 0.03 sec
[ Info: VUMPS  37:	obj = -4.347019355186e-01	err = 1.2218577552e-01	time = 0.04 sec
[ Info: VUMPS  38:	obj = +1.126876972824e-02	err = 3.8542612776e-01	time = 0.03 sec
[ Info: VUMPS  39:	obj = -1.054351314040e-01	err = 4.0525492793e-01	time = 0.06 sec
[ Info: VUMPS  40:	obj = -1.414301602807e-01	err = 4.0916217647e-01	time = 0.03 sec
[ Info: VUMPS  41:	obj = -1.893357319314e-01	err = 3.7913998013e-01	time = 0.02 sec
[ Info: VUMPS  42:	obj = -2.638139371875e-01	err = 3.5727613224e-01	time = 0.03 sec
[ Info: VUMPS  43:	obj = -1.087944796522e-01	err = 3.5097164414e-01	time = 0.02 sec
[ Info: VUMPS  44:	obj = -1.601735087524e-01	err = 3.9348691903e-01	time = 0.02 sec
[ Info: VUMPS  45:	obj = -3.599764269854e-01	err = 2.9117042431e-01	time = 0.03 sec
[ Info: VUMPS  46:	obj = -4.307479018240e-01	err = 1.4590877399e-01	time = 0.03 sec
[ Info: VUMPS  47:	obj = +1.409551265290e-01	err = 3.7003839916e-01	time = 0.02 sec
[ Info: VUMPS  48:	obj = -1.237917373142e-01	err = 3.8254863769e-01	time = 0.02 sec
[ Info: VUMPS  49:	obj = -1.627580205525e-01	err = 3.7367727861e-01	time = 0.02 sec
[ Info: VUMPS  50:	obj = -2.769089084247e-01	err = 3.5236198797e-01	time = 0.05 sec
[ Info: VUMPS  51:	obj = -2.343888370451e-01	err = 3.6427105579e-01	time = 0.02 sec
[ Info: VUMPS  52:	obj = -2.248449151603e-01	err = 3.7977874636e-01	time = 0.03 sec
[ Info: VUMPS  53:	obj = -7.550510266759e-02	err = 4.0150761461e-01	time = 0.02 sec
[ Info: VUMPS  54:	obj = -1.157957101584e-01	err = 4.1227979674e-01	time = 0.03 sec
[ Info: VUMPS  55:	obj = -3.546254055503e-02	err = 3.9957978395e-01	time = 0.03 sec
[ Info: VUMPS  56:	obj = -1.129978671789e-01	err = 4.0034284390e-01	time = 0.02 sec
[ Info: VUMPS  57:	obj = -2.329800814496e-01	err = 3.7187892563e-01	time = 0.03 sec
[ Info: VUMPS  58:	obj = +2.751650323643e-02	err = 4.0247044284e-01	time = 0.03 sec
[ Info: VUMPS  59:	obj = +1.917009780601e-01	err = 3.3210927196e-01	time = 0.02 sec
[ Info: VUMPS  60:	obj = -1.263549808391e-01	err = 3.9315953588e-01	time = 0.07 sec
[ Info: VUMPS  61:	obj = -8.215407597511e-02	err = 3.9533232172e-01	time = 0.02 sec
[ Info: VUMPS  62:	obj = -2.064545516441e-01	err = 3.7374649705e-01	time = 0.02 sec
[ Info: VUMPS  63:	obj = -2.034027935096e-01	err = 3.9761679200e-01	time = 0.03 sec
[ Info: VUMPS  64:	obj = -2.846911108688e-03	err = 3.7662169289e-01	time = 0.03 sec
[ Info: VUMPS  65:	obj = +7.553415672703e-02	err = 3.7835378290e-01	time = 0.02 sec
[ Info: VUMPS  66:	obj = -1.171790490724e-01	err = 3.5935068701e-01	time = 0.03 sec
[ Info: VUMPS  67:	obj = -1.497117506752e-01	err = 3.7395334449e-01	time = 0.03 sec
[ Info: VUMPS  68:	obj = -2.039793547678e-01	err = 3.7667227814e-01	time = 0.03 sec
[ Info: VUMPS  69:	obj = -2.089140931903e-01	err = 3.6415207135e-01	time = 0.02 sec
[ Info: VUMPS  70:	obj = -1.134944918293e-01	err = 3.6436812433e-01	time = 0.06 sec
[ Info: VUMPS  71:	obj = -1.157762777756e-01	err = 3.8567331411e-01	time = 0.02 sec
[ Info: VUMPS  72:	obj = -2.394261860644e-01	err = 3.6057044079e-01	time = 0.02 sec
[ Info: VUMPS  73:	obj = -3.342400580410e-01	err = 3.1317816493e-01	time = 0.02 sec
[ Info: VUMPS  74:	obj = -3.904134305503e-01	err = 2.5856894641e-01	time = 0.03 sec
[ Info: VUMPS  75:	obj = +8.449256685246e-02	err = 3.8483086625e-01	time = 0.03 sec
[ Info: VUMPS  76:	obj = -3.933130622801e-02	err = 4.0208977801e-01	time = 0.03 sec
[ Info: VUMPS  77:	obj = +8.653869040795e-02	err = 3.8802124228e-01	time = 0.03 sec
[ Info: VUMPS  78:	obj = +9.587848560625e-02	err = 3.5722735528e-01	time = 0.02 sec
[ Info: VUMPS  79:	obj = -5.563236275270e-02	err = 4.1092165670e-01	time = 0.03 sec
[ Info: VUMPS  80:	obj = -6.486296834251e-03	err = 3.9951995891e-01	time = 0.06 sec
[ Info: VUMPS  81:	obj = -1.209081100544e-01	err = 3.6666868713e-01	time = 0.02 sec
[ Info: VUMPS  82:	obj = -2.521478349167e-01	err = 3.5309480836e-01	time = 0.02 sec
[ Info: VUMPS  83:	obj = -3.517052113063e-01	err = 3.1461835111e-01	time = 0.03 sec
[ Info: VUMPS  84:	obj = -3.299898797334e-01	err = 3.1486702189e-01	time = 0.03 sec
[ Info: VUMPS  85:	obj = -3.493479897924e-01	err = 3.1357877281e-01	time = 0.03 sec
[ Info: VUMPS  86:	obj = -9.705041624306e-02	err = 3.8584904079e-01	time = 0.02 sec
[ Info: VUMPS  87:	obj = -6.719503369858e-02	err = 3.8802177181e-01	time = 0.02 sec
[ Info: VUMPS  88:	obj = +7.083873868280e-02	err = 3.8329406568e-01	time = 0.02 sec
[ Info: VUMPS  89:	obj = +1.427527172763e-01	err = 3.6625276364e-01	time = 0.02 sec
[ Info: VUMPS  90:	obj = -2.155800529197e-02	err = 4.3111181536e-01	time = 0.03 sec
[ Info: VUMPS  91:	obj = -3.535552416184e-02	err = 4.1031265505e-01	time = 0.05 sec
[ Info: VUMPS  92:	obj = +4.038017896823e-02	err = 3.8143678194e-01	time = 0.02 sec
[ Info: VUMPS  93:	obj = -1.670680877582e-01	err = 3.8295252948e-01	time = 0.02 sec
[ Info: VUMPS  94:	obj = -9.389358669508e-02	err = 4.0264966465e-01	time = 0.04 sec
[ Info: VUMPS  95:	obj = -6.539125878566e-02	err = 3.6997274129e-01	time = 0.03 sec
[ Info: VUMPS  96:	obj = -2.076204063368e-01	err = 3.8010942600e-01	time = 0.02 sec
[ Info: VUMPS  97:	obj = -2.793821365956e-01	err = 3.4935730821e-01	time = 0.02 sec
[ Info: VUMPS  98:	obj = -7.458031879980e-02	err = 3.8897663409e-01	time = 0.02 sec
[ Info: VUMPS  99:	obj = -2.819439484489e-01	err = 3.6706683600e-01	time = 0.03 sec
[ Info: VUMPS 100:	obj = -1.001823219839e-01	err = 3.7624054699e-01	time = 0.02 sec
[ Info: VUMPS 101:	obj = -6.394432799715e-02	err = 3.9177673956e-01	time = 0.06 sec
[ Info: VUMPS 102:	obj = -1.465694002120e-01	err = 3.8378109832e-01	time = 0.02 sec
[ Info: VUMPS 103:	obj = -2.405698991621e-01	err = 3.7543291978e-01	time = 0.02 sec
[ Info: VUMPS 104:	obj = -3.513218895760e-01	err = 3.0159411514e-01	time = 0.02 sec
[ Info: VUMPS 105:	obj = -1.023723784151e-01	err = 3.6048545911e-01	time = 0.03 sec
[ Info: VUMPS 106:	obj = -1.254960641829e-01	err = 3.5705227068e-01	time = 0.03 sec
[ Info: VUMPS 107:	obj = -2.439341345938e-01	err = 3.6666868249e-01	time = 0.03 sec
[ Info: VUMPS 108:	obj = -1.822440707693e-01	err = 3.9830893422e-01	time = 0.03 sec
[ Info: VUMPS 109:	obj = -2.501473062742e-01	err = 3.5974890197e-01	time = 0.02 sec
[ Info: VUMPS 110:	obj = -2.907188314950e-01	err = 3.3385674670e-01	time = 0.03 sec
[ Info: VUMPS 111:	obj = -3.880803577729e-01	err = 2.6012194034e-01	time = 0.05 sec
[ Info: VUMPS 112:	obj = -3.135177544690e-01	err = 3.5859183499e-01	time = 0.03 sec
[ Info: VUMPS 113:	obj = -3.239598468980e-01	err = 3.4821622589e-01	time = 0.03 sec
[ Info: VUMPS 114:	obj = -4.092283711908e-01	err = 2.1862198400e-01	time = 0.03 sec
[ Info: VUMPS 115:	obj = -4.220328070787e-01	err = 1.8801072163e-01	time = 0.03 sec
[ Info: VUMPS 116:	obj = +1.547122686700e-02	err = 4.0231050978e-01	time = 0.03 sec
[ Info: VUMPS 117:	obj = -7.971332985234e-02	err = 3.7654696853e-01	time = 0.02 sec
[ Info: VUMPS 118:	obj = +1.462167544480e-02	err = 3.7464919429e-01	time = 0.02 sec
[ Info: VUMPS 119:	obj = -2.507716805320e-01	err = 3.7118480512e-01	time = 0.02 sec
[ Info: VUMPS 120:	obj = -3.534787471790e-01	err = 3.1688927832e-01	time = 0.05 sec
[ Info: VUMPS 121:	obj = -1.108707448425e-01	err = 4.1451059429e-01	time = 0.02 sec
[ Info: VUMPS 122:	obj = -1.842124515165e-01	err = 3.8528792301e-01	time = 0.02 sec
[ Info: VUMPS 123:	obj = -2.998975814447e-01	err = 3.4023194006e-01	time = 0.02 sec
[ Info: VUMPS 124:	obj = -1.186804385815e-01	err = 4.1899473179e-01	time = 0.04 sec
[ Info: VUMPS 125:	obj = -1.948725735956e-01	err = 3.6431703182e-01	time = 0.02 sec
[ Info: VUMPS 126:	obj = +3.273407063701e-02	err = 3.4643539632e-01	time = 0.02 sec
[ Info: VUMPS 127:	obj = -2.266211833470e-01	err = 3.4825725987e-01	time = 0.03 sec
[ Info: VUMPS 128:	obj = -3.116731680576e-01	err = 3.4010142993e-01	time = 0.03 sec
[ Info: VUMPS 129:	obj = -2.826698629538e-01	err = 3.6929476876e-01	time = 0.03 sec
[ Info: VUMPS 130:	obj = -1.572495813700e-01	err = 3.9971170739e-01	time = 0.06 sec
[ Info: VUMPS 131:	obj = -1.732268711440e-02	err = 3.6936890832e-01	time = 0.02 sec
[ Info: VUMPS 132:	obj = -4.775357014229e-02	err = 3.9962098427e-01	time = 0.03 sec
[ Info: VUMPS 133:	obj = -2.431698382557e-01	err = 3.6404028082e-01	time = 0.02 sec
[ Info: VUMPS 134:	obj = -2.821932941813e-01	err = 3.5445708250e-01	time = 0.02 sec
[ Info: VUMPS 135:	obj = -9.024883449741e-02	err = 3.7670364392e-01	time = 0.03 sec
[ Info: VUMPS 136:	obj = -2.262433195377e-01	err = 3.8532078673e-01	time = 0.03 sec
[ Info: VUMPS 137:	obj = -8.163381818312e-02	err = 3.8947952861e-01	time = 0.02 sec
[ Info: VUMPS 138:	obj = -7.989801779312e-03	err = 3.8868571809e-01	time = 0.02 sec
[ Info: VUMPS 139:	obj = +2.634778860890e-02	err = 3.6599239785e-01	time = 0.02 sec
[ Info: VUMPS 140:	obj = -1.307675815427e-01	err = 3.4665317823e-01	time = 0.03 sec
[ Info: VUMPS 141:	obj = -3.217088121498e-01	err = 3.2313971501e-01	time = 0.05 sec
[ Info: VUMPS 142:	obj = -4.246143494691e-01	err = 1.5885779904e-01	time = 0.03 sec
[ Info: VUMPS 143:	obj = -8.774044006206e-02	err = 3.8014515500e-01	time = 0.03 sec
[ Info: VUMPS 144:	obj = -8.327043512249e-02	err = 3.7969588944e-01	time = 0.02 sec
[ Info: VUMPS 145:	obj = -1.325167744344e-01	err = 3.8919621062e-01	time = 0.02 sec
[ Info: VUMPS 146:	obj = -1.662916750343e-01	err = 3.4606874768e-01	time = 0.03 sec
[ Info: VUMPS 147:	obj = -1.901394048799e-01	err = 3.8421371861e-01	time = 0.03 sec
[ Info: VUMPS 148:	obj = -2.601825005552e-02	err = 3.9297149298e-01	time = 0.03 sec
[ Info: VUMPS 149:	obj = -2.376758303565e-01	err = 3.6086607615e-01	time = 0.03 sec
[ Info: VUMPS 150:	obj = -2.272499924592e-01	err = 3.7479126222e-01	time = 0.05 sec
[ Info: VUMPS 151:	obj = -2.036843938804e-01	err = 3.9925741694e-01	time = 0.03 sec
[ Info: VUMPS 152:	obj = -2.313892960788e-01	err = 3.7972643403e-01	time = 0.03 sec
[ Info: VUMPS 153:	obj = -1.986998610932e-01	err = 3.9191236797e-01	time = 0.03 sec
[ Info: VUMPS 154:	obj = -2.108517047947e-01	err = 3.7580583074e-01	time = 0.03 sec
[ Info: VUMPS 155:	obj = -3.176605702081e-01	err = 3.2923368977e-01	time = 0.03 sec
[ Info: VUMPS 156:	obj = +9.435660504635e-02	err = 3.8427690796e-01	time = 0.02 sec
[ Info: VUMPS 157:	obj = -7.759368642086e-02	err = 3.9643333120e-01	time = 0.02 sec
[ Info: VUMPS 158:	obj = -3.957699631929e-02	err = 4.0189632984e-01	time = 0.02 sec
[ Info: VUMPS 159:	obj = -1.217059048082e-01	err = 3.7324188583e-01	time = 0.03 sec
[ Info: VUMPS 160:	obj = -3.009908896146e-02	err = 3.8322255280e-01	time = 0.05 sec
[ Info: VUMPS 161:	obj = -7.237850128759e-02	err = 4.0889926487e-01	time = 0.03 sec
[ Info: VUMPS 162:	obj = -1.373382697223e-01	err = 4.2014320380e-01	time = 0.02 sec
[ Info: VUMPS 163:	obj = -1.099154797758e-01	err = 4.1417886345e-01	time = 0.03 sec
[ Info: VUMPS 164:	obj = -1.374913376578e-01	err = 3.5906146231e-01	time = 0.02 sec
[ Info: VUMPS 165:	obj = -1.464572634014e-01	err = 3.8610039285e-01	time = 0.03 sec
[ Info: VUMPS 166:	obj = -3.497532054532e-01	err = 3.2149526676e-01	time = 0.03 sec
[ Info: VUMPS 167:	obj = -3.538682309356e-01	err = 3.0937161510e-01	time = 0.03 sec
[ Info: VUMPS 168:	obj = -1.174426478371e-01	err = 4.0570387888e-01	time = 0.02 sec
[ Info: VUMPS 169:	obj = -1.003314337439e-01	err = 3.9330106579e-01	time = 0.03 sec
[ Info: VUMPS 170:	obj = -1.397014985899e-01	err = 3.8327251771e-01	time = 0.05 sec
[ Info: VUMPS 171:	obj = -1.485590666911e-02	err = 3.6193610363e-01	time = 0.02 sec
[ Info: VUMPS 172:	obj = -2.841425862542e-01	err = 3.5517293193e-01	time = 0.03 sec
[ Info: VUMPS 173:	obj = -3.175669710557e-01	err = 3.2886842202e-01	time = 0.02 sec
[ Info: VUMPS 174:	obj = -4.215876012708e-01	err = 1.8272144933e-01	time = 0.03 sec
[ Info: VUMPS 175:	obj = +1.080077314978e-01	err = 3.8650612296e-01	time = 0.03 sec
[ Info: VUMPS 176:	obj = -1.105754803548e-01	err = 3.7491324259e-01	time = 0.02 sec
[ Info: VUMPS 177:	obj = -1.534253812824e-01	err = 3.6869415832e-01	time = 0.02 sec
[ Info: VUMPS 178:	obj = -1.418091290806e-01	err = 3.7395180342e-01	time = 0.02 sec
[ Info: VUMPS 179:	obj = -2.897274308719e-01	err = 3.5651563652e-01	time = 0.02 sec
[ Info: VUMPS 180:	obj = -2.391169842001e-02	err = 3.7325894327e-01	time = 0.05 sec
[ Info: VUMPS 181:	obj = -1.721332943936e-01	err = 3.7821121884e-01	time = 0.03 sec
[ Info: VUMPS 182:	obj = -1.052070523338e-01	err = 3.8451236128e-01	time = 0.02 sec
[ Info: VUMPS 183:	obj = -6.354898914006e-02	err = 4.3678211247e-01	time = 0.02 sec
[ Info: VUMPS 184:	obj = +5.549252056570e-02	err = 3.6090995884e-01	time = 0.03 sec
[ Info: VUMPS 185:	obj = +5.565063849145e-02	err = 4.0136597605e-01	time = 0.03 sec
[ Info: VUMPS 186:	obj = +1.768269685046e-01	err = 3.2651611381e-01	time = 0.02 sec
[ Info: VUMPS 187:	obj = +1.593690655282e-04	err = 3.8841746150e-01	time = 0.02 sec
[ Info: VUMPS 188:	obj = +4.130061860625e-02	err = 3.7222674602e-01	time = 0.02 sec
[ Info: VUMPS 189:	obj = -1.930146741248e-01	err = 3.8117696997e-01	time = 0.03 sec
[ Info: VUMPS 190:	obj = -3.519361182836e-01	err = 3.0263989022e-01	time = 0.02 sec
[ Info: VUMPS 191:	obj = -2.601579677807e-01	err = 3.6061622733e-01	time = 0.05 sec
[ Info: VUMPS 192:	obj = -1.002975257021e-01	err = 4.1048517065e-01	time = 0.02 sec
[ Info: VUMPS 193:	obj = -1.563988174307e-01	err = 3.9315686700e-01	time = 0.02 sec
[ Info: VUMPS 194:	obj = -9.911970259716e-02	err = 3.7832099886e-01	time = 0.02 sec
[ Info: VUMPS 195:	obj = -1.044710830190e-01	err = 4.0631216492e-01	time = 0.03 sec
[ Info: VUMPS 196:	obj = -1.858694684809e-01	err = 3.9336892502e-01	time = 0.02 sec
[ Info: VUMPS 197:	obj = -3.090115170517e-01	err = 3.5367444183e-01	time = 0.03 sec
[ Info: VUMPS 198:	obj = -1.946258306654e-02	err = 3.6965265025e-01	time = 0.02 sec
[ Info: VUMPS 199:	obj = -2.238429310745e-01	err = 3.7577524599e-01	time = 0.03 sec
┌ Warning: VUMPS cancel 200:	obj = -3.951359685829e-01	err = 2.5736930769e-01	time = 5.62 sec
└ @ MPSKit ~/Projects/MPSKit.jl/src/algorithms/groundstate/vumps.jl:76

````

As you can see, VUMPS struggles to converge.
On it's own, that is already quite curious.
Maybe we can do better using another algorithm, such as gradient descent.

````julia
groundstate, cache, delta = find_groundstate(state, H, GradientGrassmann(; maxiter = 20));
````

````
[ Info: CG: initializing with f = 0.249999554947, ‖∇f‖ = 1.3395e-03
┌ Warning: CG: not converged to requested tol after 20 iterations and time 6.30 s: f = -0.441802293068, ‖∇f‖ = 1.0800e-02
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/cg.jl:172

````

Convergence is quite slow and even fails after sufficiently many iterations.
To understand why, we can look at the transfer matrix spectrum.

````julia
transferplot(groundstate, groundstate)
````

```@raw html
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAIAAAD9V4nPAAAABmJLR0QA/wD/AP+gvaeTAAAgAElEQVR4nO3dd2BUZbrH8WdKemISQULHhBIIJYAIC2xcMBCQDoJisODGxX7B61WwAqLsckXBuyCyyEIo0hGQ3iUIGJpICYihJiGQkD5pk5lz/zgwO4Ig7k4yCe/389d73vPmzDOT98xv5pwzMwZN0wQAAFUZ3V0AAADuRBACAJRGEAIAlEYQAgCURhACAJRGEAIAlEYQAgCURhACAJRGEAIAlEYQAgCU5sognDhxYmZm5p2Pt1qtLrz1qstut9tsNndXUSkwJXQ2m81ut7u7ikqBKaErKytzdwmVRXlMCVcG4dKlS1NTU+98fHFxsQtvveqy2Wzs7TqmhM5qtfLaSMeU0JWUlPDaSFceU4JDowAApRGEAAClEYQAAKURhAAApRGEAAClEYQAAKURhAAApRGEAAClEYQAAKURhAAApZndXQAAKMpqtX744YclJSV3MtJsNhsMhgqoqnKqWbPmqFGjymnjBCEAuEdGRsbUqVPffvttdxdS2RUVFY0fP54gBIC7kL+//+jRo91dRWWXk5Pz2Wefld/2OUcIAFAaQQgAUBpBCABQGkEIAFAaQQgAUBpBCABVXnKe1mlNmburqKr4+AQAVHlFNjl8VXPV1hISEpKSkm7orF+/fs+ePfV2bm5u//79d+7ceastvPTSS0899VTHjh1/dW1SUtKHH364cOFCfTE+Pj4gIGDQoEEHDx5ct27d+++/74L78HsQhACAX7h48eIPP/wgIvv27bNYLNHR0SJSVvavd5xms7lFixa32UJYWNg999xzq7V5eXkHDhzQ27m5uWPHjj1y5IiItGnTZvjw4UOGDGnWrJlL7sgdIggBAL8QGxsbGxsrIqNHj05NTf38889FZNu2bUVFRbt27SopKenVq9fjjz8uIsnJySUlJREREfofpqSkZGRktGnTpnv37rVq1RIRu91+6NChpKSke++9t0uXLn5+fjfc1vz587t27RoYGCgiRqPxqaeemj59+rRp0yry/nKOEADw22JiYvr27btw4cKDBw/m5OR06dJFRJKTkwcMGKBp147Kvvnmmxs2bBCRF198ce/evSIyZ86c8ePH//jjj3Pnzo2MjMzOzr5hs6tWrerRo4fzraxcubJi7pED7wgBoOo5fFX76w92x2KuVYwGeWybzXnMrChToKcrb/SJJ56Ii4sTkczMTL2nW7duxcXF+/bt69ixY15e3jfffDNp0iTnP4mLi9P/RET+/Oc/z5s3b+TIkc4DDh48+OmnnzoWIyIi0tPT09PTa9as6crSb4sgBICqJ9BTHqj+rx+juFyk7Uj7RY+ImF19yK979+439OgHM+Pj4zt27Lh06dKOHTvWq1fPecCVK1fGjRt34MCBq1ev5uTkBAQEOK+12+25ubn6cVGdp6enj49PVlYWQQgAuJ2wAMPoyH/F3rFsbUaSfXRk+Z7t8vX1vblz+PDhDz744JQpU+Lj41944YUb1r700kuhoaGbN28OCgp69913s7KynNcajcZ77rknLy/P0WO1WouKioKCgsqj/lvhHCEA4N/XuHHj5s2bf/bZZ0ePHh04cOANa5OSkvr06RMUFGS32zdt2nTzn7du3frEiRPO40NCQmrXrl2+Rf8SQQgA+I8MHz78vffeGzJkyM1vGQcNGvT888+/++67Xbt29fDwuPlvBwwYsHXrVsfitm3b+vXrV77l3oRDowCAX/fCCy+Ulpbq7c2bNztO5gUFBTl/mj42NjY0NLR58+aOnhkzZugnCydMmBAdHX327Nknn3zS29u7uLhYRCIiIhyfpn/66afbtGmTn58fEBCgaVp8fPy8efMq5M79C0EIAFXevV4SF+76I3yhoaGOtv6xep3ZbI6KinIs+vn5devWzfkPIyMjHe0uXbron7VwCAgIaNeund6+9957R48evX79+scff3z//v3dunVr1aqVS+/EbyMIAaDKq+1rmNbJ5O4q/k0vvfSS3mjfvn379u0rvgDOEQIAlEYQAgCURhACAJRGEAIAlEYQAgCUxlWjAOAeBoMhJyfn+eefd3chlZ3js4zlhCAEAPcICQn54osvioqKfnNkSUmJp6enwWD4zZF3qyFDhpTfxglCAHAP/acb7mSkxWLx8fExGjmZVS54WAEASiMIAQBKIwgBAEojCAEASiMIAQBKIwgBAEojCAEASiMIAQBKIwgBAEojCAEASiMIAQBKIwgBAEojCAEASiMIAQBKIwgBAEojCAEASrsxCLOysrKystxSCgAAFc84cuRITdNEZPPmzQ0bNmzUqFGTJk3atm178uRJfcTo0aMNTgoKCtxaMAAArmRctWrVpk2b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xl0yMzM//vjjoUOHdu/e/YZVV69eHTZsmGPYk08+KSLvvffeH/7wh8aNG3fr1m3Dhg0VXW55mjJlSnR0dHh4eI8ePbZv3+68auXKlf/4xz/09vLly7/88su0tLTHHnusZcuWzZs3f+aZZy5cuOCOksvF+fPnJ06c+Oijj8bGxt6w6uTJk6NGjdLbSUlJr732moisXLny1ZX/zSMAAA0bSURBVFdf7dGjxzfffFPRtVaIsrKy2NjYl19+2blz7dq106ZN09vffPPN9OnTMzIynnjiiZYtW0ZERDz55JNnzpxxR7HlIjExsbuTffv2OValp6cPHz5cb6elpf35z38WkTFjxnTo0KFJkyYxMTFbt251S83lITc396233urYsWOzZs2eeOKJ06dPO6/98MMPHd+zOmHChO++++7y5cuTJk167LHH+vTp45ICXB+EO3bsGDNmzOeffz5//vy///3vS5YscaxavHhxjRo19PaiRYv0dkJCQr169U6dOnX58mWXF+NG6enpp06daty48c3zdd26dT4+Pnp77dq1ertp06bTp0/ftGnTsGHDHn300ZMnT1Z0xeXmyJEj//M//7N27dqBAwf26dPn1KlTjlWzZ8++//77ndtGo3HgwIFLly5dtmyZiAwaNMgtNZeHs2fPpqSk1K1b9+ZvT16xYkVQUJDeXr58uf753++++65GjRoXL15071fRlp///d///eGHH274gtk5c+bUr19fb//zn/9s0KCBiPTp02fJkiUrV6709fXt16+fG2otH5mZmampqaOva9SokWPVmjVr/Pz8bmg3b958xowZGzZsGDJkSL9+/e6a1wRXrlyxWCyTJ09etWpVtWrVunfvbrVe+7JHm802ffr0Zs2a6e0ZM2ZERESkpKQkJyeHhYXt2LHDJQW4/nOEjz/+eHh4+AcffCAi06dPX758uaPW8PDwjRs36h+Pbdiw4bZt2xxPgq1bt37rrbcef/xxETl//vyMGTOct2k0GidOnOjaOitGUlJSRETEDQ/yo48+Onz48L59+4rIwIED4+LibnhdExERMX78+P79++tvmp299tprISEh5V12+XnggQdeeeWVZ599VkQKCgoaNWp04cIFT09P57Zj8NGjR9u2bVtSUnLo0KHly5c7byckJER/21TlbNmyJS4u7oZ3uu3bt585c2abNm1EpF27drNn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" />
```

We can clearly see multiple eigenvalues close to the unit circle.
Our state is close to being non-injective, and represents the sum of multiple injective states.
This is numerically very problematic, but also indicates that we used an incorrect ansatz to approximate the groundstate.
We should retry with a larger unit cell.

## Success

Let's initialize a different initial state, this time with a 2-site unit cell:

````julia
state = InfiniteMPS(fill(2, 2), fill(20, 2))
````

````
2-site InfiniteMPS:
│   ⋮
│ C[2]: TensorMap(ℂ^20 ← ℂ^20)
├── AL[2]: TensorMap((ℂ^20 ⊗ ℂ^2) ← ℂ^20)
├── AL[1]: TensorMap((ℂ^20 ⊗ ℂ^2) ← ℂ^20)
│   ⋮

````

In MPSKit, we require that the periodicity of the hamiltonian equals that of the state it is applied to.
This is not a big obstacle, you can simply repeat the original hamiltonian.
Alternatively, the hamiltonian can be constructed directly on a two-site unitcell by making use of MPSKitModels.jl's `@mpoham`.

````julia
# H2 = repeat(H, 2); -- copies the one-site version
H2 = heisenberg_XXX(ComplexF64, Trivial, InfiniteChain(2); spin = 1 // 2)
groundstate, envs, delta = find_groundstate(
    state, H2, VUMPS(; maxiter = 100, tol = 1.0e-12)
);
````

````
[ Info: VUMPS init:	obj = +4.988938408716e-01	err = 6.6329e-02
[ Info: VUMPS   1:	obj = -4.739248140491e-01	err = 3.3816850792e-01	time = 0.03 sec
[ Info: VUMPS   2:	obj = -8.743783163834e-01	err = 8.7782281077e-02	time = 0.03 sec
[ Info: VUMPS   3:	obj = -8.852289662056e-01	err = 1.1261717077e-02	time = 0.03 sec
[ Info: VUMPS   4:	obj = -8.859341540347e-01	err = 6.0581249357e-03	time = 0.12 sec
[ Info: VUMPS   5:	obj = -8.861158625188e-01	err = 4.1548303785e-03	time = 0.03 sec
[ Info: VUMPS   6:	obj = -8.861822061550e-01	err = 2.8484872369e-03	time = 0.03 sec
[ Info: VUMPS   7:	obj = -8.862103323128e-01	err = 2.1781142069e-03	time = 0.03 sec
[ Info: VUMPS   8:	obj = -8.862234788867e-01	err = 1.6559901778e-03	time = 0.03 sec
[ Info: VUMPS   9:	obj = -8.862298897521e-01	err = 1.3551274661e-03	time = 0.03 sec
[ Info: VUMPS  10:	obj = -8.862331118301e-01	err = 1.1002219058e-03	time = 0.04 sec
[ Info: VUMPS  11:	obj = -8.862347480843e-01	err = 9.4695562130e-04	time = 0.09 sec
[ Info: VUMPS  12:	obj = -8.862355951169e-01	err = 8.0634081582e-04	time = 0.04 sec
[ Info: VUMPS  13:	obj = -8.862360364673e-01	err = 7.1458186619e-04	time = 0.03 sec
[ Info: VUMPS  14:	obj = -8.862362718663e-01	err = 6.3113350829e-04	time = 0.04 sec
[ Info: VUMPS  15:	obj = -8.862363996891e-01	err = 5.6139764728e-04	time = 0.04 sec
[ Info: VUMPS  16:	obj = -8.862364715964e-01	err = 5.0511718667e-04	time = 0.04 sec
[ Info: VUMPS  17:	obj = -8.862365136698e-01	err = 4.5105971790e-04	time = 0.06 sec
[ Info: VUMPS  18:	obj = -8.862365394867e-01	err = 4.0773238562e-04	time = 0.03 sec
[ Info: VUMPS  19:	obj = -8.862365562591e-01	err = 3.6503807001e-04	time = 0.04 sec
[ Info: VUMPS  20:	obj = -8.862365676678e-01	err = 3.3014036465e-04	time = 0.03 sec
[ Info: VUMPS  21:	obj = -8.862365758645e-01	err = 2.9598570227e-04	time = 0.04 sec
[ Info: VUMPS  22:	obj = -8.862365819255e-01	err = 2.6762287043e-04	time = 0.07 sec
[ Info: VUMPS  23:	obj = -8.862365865819e-01	err = 2.4021341209e-04	time = 0.03 sec
[ Info: VUMPS  24:	obj = -8.862365902034e-01	err = 2.1707752879e-04	time = 0.03 sec
[ Info: VUMPS  25:	obj = -8.862365930830e-01	err = 1.9507319179e-04	time = 0.03 sec
[ Info: VUMPS  26:	obj = -8.862365953820e-01	err = 1.7616283550e-04	time = 0.03 sec
[ Info: VUMPS  27:	obj = -8.862365972387e-01	err = 1.5851786128e-04	time = 0.03 sec
[ Info: VUMPS  28:	obj = -8.862365987416e-01	err = 1.4302884556e-04	time = 0.07 sec
[ Info: VUMPS  29:	obj = -8.862365999644e-01	err = 1.2889703895e-04	time = 0.03 sec
[ Info: VUMPS  30:	obj = -8.862366009623e-01	err = 1.1617835138e-04	time = 0.03 sec
[ Info: VUMPS  31:	obj = -8.862366017775e-01	err = 1.0488359393e-04	time = 0.03 sec
[ Info: VUMPS  32:	obj = -8.862366024472e-01	err = 9.4408469738e-05	time = 0.03 sec
[ Info: VUMPS  33:	obj = -8.862366029961e-01	err = 8.5404044738e-05	time = 0.03 sec
[ Info: VUMPS  34:	obj = -8.862366034498e-01	err = 7.6748331848e-05	time = 0.03 sec
[ Info: VUMPS  35:	obj = -8.862366038231e-01	err = 6.9594627182e-05	time = 0.06 sec
[ Info: VUMPS  36:	obj = -8.862366041338e-01	err = 6.2415914486e-05	time = 0.03 sec
[ Info: VUMPS  37:	obj = -8.862366043907e-01	err = 5.6758538061e-05	time = 0.03 sec
[ Info: VUMPS  38:	obj = -8.862366046064e-01	err = 5.0780009127e-05	time = 0.03 sec
[ Info: VUMPS  39:	obj = -8.862366047860e-01	err = 4.6333072643e-05	time = 0.03 sec
[ Info: VUMPS  40:	obj = -8.862366049382e-01	err = 4.1331207549e-05	time = 0.03 sec
[ Info: VUMPS  41:	obj = -8.862366050662e-01	err = 3.7863899058e-05	time = 0.06 sec
[ Info: VUMPS  42:	obj = -8.862366051761e-01	err = 3.3658336653e-05	time = 0.03 sec
[ Info: VUMPS  43:	obj = -8.862366052696e-01	err = 3.0983965257e-05	time = 0.03 sec
[ Info: VUMPS  44:	obj = -8.862366053511e-01	err = 2.7523802341e-05	time = 0.04 sec
[ Info: VUMPS  45:	obj = -8.862366054215e-01	err = 2.5397047125e-05	time = 0.03 sec
[ Info: VUMPS  46:	obj = -8.862366054839e-01	err = 2.2595519406e-05	time = 0.03 sec
[ Info: VUMPS  47:	obj = -8.862366055388e-01	err = 2.0862977631e-05	time = 0.06 sec
[ Info: VUMPS  48:	obj = -8.862366055883e-01	err = 1.8599136674e-05	time = 0.03 sec
[ Info: VUMPS  49:	obj = -8.862366056327e-01	err = 1.7188042263e-05	time = 0.03 sec
[ Info: VUMPS  50:	obj = -8.862366056736e-01	err = 1.5364204857e-05	time = 0.04 sec
[ Info: VUMPS  51:	obj = -8.862366057110e-01	err = 1.4215375363e-05	time = 0.03 sec
[ Info: VUMPS  52:	obj = -8.862366057460e-01	err = 1.2752194421e-05	time = 0.04 sec
[ Info: VUMPS  53:	obj = -8.862366057787e-01	err = 1.1817843549e-05	time = 0.06 sec
[ Info: VUMPS  54:	obj = -8.862366058097e-01	err = 1.0651271784e-05	time = 0.03 sec
[ Info: VUMPS  55:	obj = -8.862366058392e-01	err = 9.8929892467e-06	time = 0.04 sec
[ Info: VUMPS  56:	obj = -8.862366058675e-01	err = 8.9706820132e-06	time = 0.03 sec
[ Info: VUMPS  57:	obj = -8.862366058948e-01	err = 8.3567048894e-06	time = 0.04 sec
[ Info: VUMPS  58:	obj = -8.862366059214e-01	err = 7.6361746845e-06	time = 0.04 sec
[ Info: VUMPS  59:	obj = -8.862366059472e-01	err = 7.1407423616e-06	time = 0.07 sec
[ Info: VUMPS  60:	obj = -8.862366059725e-01	err = 6.5866324690e-06	time = 0.04 sec
[ Info: VUMPS  61:	obj = -8.862366059973e-01	err = 6.1886994474e-06	time = 0.04 sec
[ Info: VUMPS  62:	obj = -8.862366060219e-01	err = 5.7708691344e-06	time = 0.04 sec
[ Info: VUMPS  63:	obj = -8.862366060460e-01	err = 5.4526696713e-06	time = 0.04 sec
[ Info: VUMPS  64:	obj = -8.862366060700e-01	err = 5.1458563859e-06	time = 0.07 sec
[ Info: VUMPS  65:	obj = -8.862366060938e-01	err = 4.8920859232e-06	time = 0.03 sec
[ Info: VUMPS  66:	obj = -8.862366061174e-01	err = 4.6741015849e-06	time = 0.03 sec
[ Info: VUMPS  67:	obj = -8.862366061409e-01	err = 4.4720328824e-06	time = 0.03 sec
[ Info: VUMPS  68:	obj = -8.862366061643e-01	err = 4.3236853227e-06	time = 0.04 sec
[ Info: VUMPS  69:	obj = -8.862366061876e-01	err = 4.1624035800e-06	time = 0.04 sec
[ Info: VUMPS  70:	obj = -8.862366062109e-01	err = 4.0670654185e-06	time = 0.06 sec
[ Info: VUMPS  71:	obj = -8.862366062341e-01	err = 3.9372522985e-06	time = 0.04 sec
[ Info: VUMPS  72:	obj = -8.862366062573e-01	err = 3.8814470100e-06	time = 0.04 sec
[ Info: VUMPS  73:	obj = -8.862366062805e-01	err = 3.7755570661e-06	time = 0.04 sec
[ Info: VUMPS  74:	obj = -8.862366063037e-01	err = 3.7485088165e-06	time = 0.04 sec
[ Info: VUMPS  75:	obj = -8.862366063269e-01	err = 3.6605911620e-06	time = 0.04 sec
[ Info: VUMPS  76:	obj = -8.862366063501e-01	err = 3.6540367594e-06	time = 0.06 sec
[ Info: VUMPS  77:	obj = -8.862366063733e-01	err = 3.5795056881e-06	time = 0.04 sec
[ Info: VUMPS  78:	obj = -8.862366063965e-01	err = 3.5873001441e-06	time = 0.04 sec
[ Info: VUMPS  79:	obj = -8.862366064197e-01	err = 3.5226970394e-06	time = 0.04 sec
[ Info: VUMPS  80:	obj = -8.862366064429e-01	err = 3.5403874329e-06	time = 0.04 sec
[ Info: VUMPS  81:	obj = -8.862366064662e-01	err = 3.4831436727e-06	time = 0.07 sec
[ Info: VUMPS  82:	obj = -8.862366064895e-01	err = 3.5075667400e-06	time = 0.03 sec
[ Info: VUMPS  83:	obj = -8.862366065128e-01	err = 3.4557926414e-06	time = 0.04 sec
[ Info: VUMPS  84:	obj = -8.862366065361e-01	err = 3.4847339467e-06	time = 0.04 sec
[ Info: VUMPS  85:	obj = -8.862366065595e-01	err = 3.4370450912e-06	time = 0.04 sec
[ Info: VUMPS  86:	obj = -8.862366065829e-01	err = 3.4689709047e-06	time = 0.04 sec
[ Info: VUMPS  87:	obj = -8.862366066063e-01	err = 3.4243527295e-06	time = 0.06 sec
[ Info: VUMPS  88:	obj = -8.862366066297e-01	err = 3.4582089166e-06	time = 0.04 sec
[ Info: VUMPS  89:	obj = -8.862366066532e-01	err = 3.4159150211e-06	time = 0.03 sec
[ Info: VUMPS  90:	obj = -8.862366066767e-01	err = 3.4509811256e-06	time = 0.04 sec
[ Info: VUMPS  91:	obj = -8.862366067002e-01	err = 3.4104587388e-06	time = 0.04 sec
[ Info: VUMPS  92:	obj = -8.862366067238e-01	err = 3.4462444730e-06	time = 0.07 sec
[ Info: VUMPS  93:	obj = -8.862366067473e-01	err = 3.4093962434e-06	time = 0.03 sec
[ Info: VUMPS  94:	obj = -8.862366067709e-01	err = 3.4432535858e-06	time = 0.03 sec
[ Info: VUMPS  95:	obj = -8.862366067945e-01	err = 3.4135991515e-06	time = 0.03 sec
[ Info: VUMPS  96:	obj = -8.862366068182e-01	err = 3.4414722655e-06	time = 0.04 sec
[ Info: VUMPS  97:	obj = -8.862366068418e-01	err = 3.4175821732e-06	time = 0.04 sec
[ Info: VUMPS  98:	obj = -8.862366068655e-01	err = 3.4405116799e-06	time = 0.06 sec
[ Info: VUMPS  99:	obj = -8.862366068892e-01	err = 3.4212410568e-06	time = 0.04 sec
┌ Warning: VUMPS cancel 100:	obj = -8.862366069129e-01	err = 3.4400872838e-06	time = 4.02 sec
└ @ MPSKit ~/Projects/MPSKit.jl/src/algorithms/groundstate/vumps.jl:76

````

We get convergence, but it takes an enormous amount of iterations.
The reason behind this becomes more obvious at higher bond dimensions:

````julia
groundstate, envs, delta = find_groundstate(
    state, H2, IDMRG2(; trscheme = truncrank(50), maxiter = 20, tol = 1.0e-12)
);
entanglementplot(groundstate)
````

```@raw html
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAIAAAD9V4nPAAAABmJLR0QA/wD/AP+gvaeTAAAgAElEQVR4nOzdZ0ATWdsG4JNKCR1EmjRFQUFRQERUQFERRV27rqti730tu2v3tfeuq651bWtXsCBYsNEERBFsSO+9JiT5fgxfjBEVWwbIff0iT04yz4RyMzNnZhhisZgAAAAoKibdDQAAANAJQQiK6ObNm2PGjAkMDKS7EbmqrKx89uzZ/fv3Y2JisrKy6G4HoLZAEMIneXh4cD/BxcXla9/t9u3b+/btS01N/Rmtfq2YmJiDBw8+f/6c7ka+3enTp/ft2ycUCmsyuKioaMaMGTo6Ora2th06dLCzs9PX1zc3N/f19a2oqPjZrX7KnTt3as+PBCgyNt0NQO0lEAgEAkHLli11dHRknmrWrNnXvtvhw4cPHToUFBRkZGT0gxpUaEuXLo2NjR0xYgSLxfr8SD6f7+npGRISoqurO3ny5MaNGxcVFb169SogIODQoUNbtmxRUlKST88yDh8+/M8//wQGBuJHAuiFIIQvWLt2rZeXF91dwLc7duxYSEiItbX1gwcPtLW1JfXKykp/f3+6UhCg9kAQwveKjY0tLS21t7cXi8U3btyIi4tTU1Pz9PS0sLCQjAkPD8/OziaExMfHq6urU8VWrVqx2WxCiFAoDA8Pj4uLy8jI0NTUbNu2batWrWSW8vz587KyMpmldO3a1dzc/OOW3r59e/PmzdLS0qZNm3bt2lUkEsXExGhoaFhZWX1+Xfh8/p07d168eCEUCq2trTt37szlcqUHhIeHc7lcOzu7wsJCPz+/tLQ0MzOzXr16UcPEYvHdu3ejoqK4XK6Xl1e1vZWXl9++fTs+Pl4sFtvY2Hh4eHA4HMmzYrE4IiJCWVm5RYsWxcXFV69eTUpKMjIy8vLykmyXl5SUvHjxory8nBDy5MkTatFKSkq2trbVrlRwcDAhZOTIkdIpSAhhs9k+Pj7Vrl1+fr6/v39qaqqJiYm3t7fkWyatrKwsKCjo1atXhJAWLVq4ublR300ZOTk5QUFBycnJysrKFhYWbm5uysrKhJCIiAjJj4SGhgY1uGXLlhwOJyUlJT093cLCQkdH58mTJ48ePSovLx8/fjyHw3n69Km6unrTpk2lF5GWlpaammpubq6rq0tVYmJiKioq2rRpw+fz/f393759a2Bg4OPjo6amRg0ICwsLCQkRiUQeHh4tWrSo9nMDBSIG+ARXV1dCiL+//+eH2dvbE0IiIiKk/xCz2ewdO3ZIxjCZ1RyNzsjIEIvFQUFBenp6Mk95eXnl5uZKL8XOzo4Q8uTJk+bNm0uGcTic3bt3y/SzdOlS6b2FNjY2169fp95TMmbTpk2EkO3bt0u/8Nq1a6amptJtWFhYhIaGSo9hMpmmpqa3bt2S/M0lhFhbW6ekpGRkZFCfGEVJSenkyZMyvV26dMnY2Fh6EVZWVpGRkZIBVLw1a9YsMDCwQYMGkmFaWlq3b9+mxoSEhHz8YVpYWHzqGzR+/HhCyLJlyz79Pfxg7a5evaqpqSl554YNG0oWLXHmzJmGDRtKN2BjY/Ps2TPpMUKhcNGiRSoqKtLD1NXVX79+LRaLq92jm56eLhaLFyxYQAg5ePBgnz59JE8lJSW9ffuWENKlSxeZZpYvX04I+eeffyQV6j+eiIgI6f9FDAwMoqOji4uL+/XrJykymcz169d/8ZOB+g1BCJ/0VUFoYWHh7e199erVsLCwNWvWcLlcNpsdFxdHjQkICOjWrRshZMOGDTf/X0VFhVgsPnPmTM+ePY8dO/b48eO4uDh/f39PT09CyIABA6SXQgWhhYVFr169/Pz8wsLCVq1axeVyORzOq1evJMP2799PCGnUqNHZs2cTExNDQ0N79epFHYL6fBAGBQWx2WwNDY1169aFhYU9efJk5cqVSkpKOjo6SUlJkmFMJlNDQ0NHR2fGjBl37969c+eOt7c3IaR///4eHh4uLi4XL14MCwv73//+x2KxNDQ0pOPc39+fxWJpa2tv2rQpPDw8IiJiyZIlHA5HX1+f+p9A/P9BqKOjo6WlNXXq1MDAwIcPH06YMIEQYmxsTH1iBQUFN2/epDL76tWr1IcZHBz8qW/Qrl27qPe8cOFCZWXlZ76VTCZTXV1dXV199uzZsbGxr169WrZsGYvFUldXf/funWTY+fPnmUymnp7e9u3bnzx5Eh4evnDhQhaLZWJikpOTIxk2depUQoi5ufmhQ4devHgRGRl54sQJb2/v58+fUz8S3bt3r/ZHggpCU1NTKyurPXv2PHjw4NSpU/n5+V8bhGZmZsOGDbt58+ajR498fX0JIQ4ODqNGjbK2tj558mRERMSuXbt4PB6LxYqNjf3MxwL1HoIQPokKQmtr63YfOXTokGQYFYTe3t4ikUhSnDNnDiFk3bp1ksqoUaMIIUFBQV9cLp/Pb9OmDYPBSExMlBSpIPTx8ZFeysyZMwkhGzdupB4KBAIq80JCQiRjBAKBjY3N54NQKBQ2a9aMyWTeuXNHupMdO3YQQqZNmyapUJu2f/31l6RSVFSkpaVFCGnevDmfz5fUhw8fTgiRbBQKBAJzc3M2m/348WPpRaxfv54QMn/+fOohFYSEkKVLl0oP69ixIyFEuj1qpcrKyj73UYrFYrG4uLi4TZs21Ntqamp6e3svW7YsNDRU+pOUXruRI0dKF+fPn08IGTdunKRDQ0NDJSUl6Q1ZsVi8bNky6e3OiIgIamsyNTX1U41R4RQYGChTp4KQx+PJvPZrg3DYsGGSikgkomZ46enpSf93smjRIkLIqlWrPtUkKAKcPgFfkJiY+OwjH5+FNnfuXAaDIXlIbf9Rf7m+FofD8fb2FovFYWFhX7WU8PDw1NRUV1dXJycnyRg2mz1t2rTPLzE0NDQuLs7Nza1Tp07S9XHjxnE4HH9/f5nxs2bNknytpqbm4OBACJk6dar00T43Nzfp3oKDgxMSErp37962bVvpt5o0aRKTyZRZBJPJpP6TkOjatSsh5M2bN59fkWrxeLyHDx+uWrWqadOmBQUFfn5+S5YscXJyatKkybVr1z4eL7PomTNnMpnM8+fPi8ViQsitW7fS0tJ8fHxkjuNOnjyZEOLn50c9PH78OCFk6tSphoaG39AzIWTEiBHf/FrK7NmzJV8zGAzqmztq1CjpY6Uy3yZQTJgsA19w9uzZmswalTmhgjqAlJ6eXpNF3LhxY8eOHc+fP09JSZFsEhFCqMkUNV9KXFwcIUT6ICLli7Mhnjx5QggpLCyktkWkqaqqUnsFJQGsp6cncz4JdTBPZgYHVczIyJBeRHZ29seLUFZWTkhIkK4YGxtLpnVIr6nk3b4Wl8tduHDhwoULX7169ejRo9u3b1+4cOHNmzc+Pj4BAQFUGFDYbLbMB2hgYGBoaJiSkpKZmdmwYUNqRdLT0z9eES6XK1mRqKgoQkjr1q2/rWFS3ffxa8nMjarJtwkUE4IQfgxVVVXph9RONnENrmS7Z8+eSZMmqaur9+jRY/DgwdQEwrt37/r5+VVWVn7VUqgQlZkbWW1FRl5eHiHk+fPnH29yMZlMNTU1gUAgmT4q04OkDZlZIVRwSnqjFvH06dP4+HiZlyspKcnMt/zUIkQi0edX5IuaNGnSpEmT4cOHr1u3zsfH58GDBytXrpQOQl1d3Y+nsejr66ekpBQVFTVs2JBakYiIiGfPnskM4/F4kk+psLCQEPI9Jwh+PIXqa1X70yLzbar5DyrUYwhCoJNAIPjjjz/U1NTCw8Ol/3/PysqS7GSrOSrwPr5SSUpKyudfSJ0eMHbs2G3btn3tQmuIWsTUqVPXrl37kxbxVXR0dBYvXuzl5RUeHi5dz8nJEQqFMllIbTBR/6NQKzJv3rwlS5Z85v2p46YpKSnfs1Eogwqtjy+mU1JS8qMWAYoJxwhBTqjjZzJ/xd69e5eXl+fo6CizF4uaavG1qCkhwcHBMpuSt2/f/vwLqT/WDx48+IaF1tAPXwT1eX680Vxz1Pl8Mme2VFZWRkdHS1dSU1PT09P19fX19fVJjVeEGiaTsjKq/ZH4jIYNGzIYjI93Y8bGxtbwHQCq9b1BuHz58q86ziwQCL5ziVBHUXvJkpOTpYvU7q/k5GTpv4ZBQUFBQUHfsIjGjRu7uromJCTs3btXUkxISNizZ8/nX9iuXTsbG5vw8PAzZ858/GxxcfE3NCPDzc3N0tIyODj48uXLP2QR1X6e1bp69Wq128QHDhwghDg6OsrUN27cKP1w06ZNIpGof//+1MNu3boZGxsHBAQEBAR8/J6SFRk+fDiDwdixY8dnOqRWISkp6YurQFFSUjI2No6Pj3/9+rWkGBUV9Q07DwCkfe+u0cDAQDc3N+lriHxeRUUFm82WnvgHtdyuXbuuXLkiU2SxWFu3bv2q96Fmci5atOjdu3cGBgaEkN9++01LS6t169ZPnjwZMWLE9OnTNTQ0AgICFi9ebGlp+W0zJHfs2NGhQ4dp06bdu3evXbt2qampBw8edHZ2vnHjxmdexWKxDhw40Llz52HDhgUHB3fr1s3U1DQjIyMuLu7EiROtWrXauXPnNzQjjcPhHDhwoHv37v379582bZqnp6eJiUlaWlpcXNy///7r6uq6YcOGr3pDJyena9eujRgxYuDAgZqamurq6kOHDq125Pnz53/55RcvL6+uXbs2bdqUw+EkJCQcO3YsKCiIw+H89ddf0oM1NDSuXr06efLksWPHstnsEydObN68WUtL688//6QGKCsr79+/38fHp1evXjNmzPDw8DA2Nk5NTY2NjT1+/HiPHj2okxlsbW1///33devWtWvX7q+//nJxcamoqHjx4sWRI0e2bdtGTYShfiSWLFmSmJhITRD97bffZI7hyRgyZMiGDRt8fHyWL1/esGHD0NDQFStWWFpafnzkFeArfOfpF25ubh9fdeIzioqKPj57CWon6eukyGCz2ZJh1HmEeXl50q+ldq/17dtXUhGJRAsWLJC+ZAl1FnlMTIyZmZmkyGAw5syZs27dOkLIrl27JC+nziMsLCyUXgo1g7F///7SxZCQEGdnZ+rdNDU1FyxYQO3Hkz5Dv9oryzx+/JhaF2mGhob79++XjKGuvSLzQQ0bNowQcv/+fenixYsXCSEzZsyQLt67d+/jC6EZGxsfPXqUGiC5sozMIv7++2/y4eluhYWFgwYNklwp9DNXljlx4oSjo+PHF/exsbG5ceOG9Ehq7W7evCk9LdbExOThw4cy73nr1i3qREZppqamp0+flowRiUSrVq3i8XjSYxo0aJCQkCAZsHDhQukfCekry5w4ceLjdSkqKqKut0BhMplLly791HmEAoFA+rVU5Es+asrTp08JIT4+Pp/69EARMMTfN13K3d192bJl0rPOPq+4uJjH42GLsE5ITk4uKyur9ikGg9GkSRPJMD6fb25uLv2nls/nJycn83g8mQtxEUJyc3Pz8/MJIWZmZtSkjIqKiuDg4Ldv36qqqnbs2LFRo0YFBQU5OTl6enqSq1B+7VLy8vJKSkoMDAyozZphw4bNmTNHstWVn5+flZWlr68v/VeYECIWi58+fRoZGVlaWtqwYUMzMzN7e3vpJb5584bNZstciS0zM7O4uNjIyIg66kYpLS1NT0/X1NSUvhgbIUQkEkVHR0dHR5eWlhoYGJibm7dq1UryGyEWi9++fcvhcBo1aiT9qqKioqysLOqKMzINp6enl5WVfdyVjIyMjGfPnqWkpJSUlGhqatrZ2X0cydTVYd69e1dcXBwQEJCRkWFkZOTp6VntVppIJHry5ElMTExZWZmhoaGFhYWdnd3Hv9qFhYX37t1LTk5WVVW1sLBwcXH5eFaqzI8E9VBfX1/mHBLJKj948CAmJkZVVdXd3b1Ro0Z5eXl5eXkNGjSQXBM1KSlJIBBYWFhI91Pt21I/QqqqqtReClBMCEKo58RisaenZ2Bg4NWrV6nLocGnSIKQ7kYA5AqnT0C9Ql1S2dfX187OTlVVNS4ubseOHYGBgU5OTtSVLQEAZHwuCM+fP//y5ctevXp9/yUeAOSDwWDcu3fv5s2b0sUuXbocPXr0izewBQDF9MkgXLduXVRU1ODBg4cMGfLff//JXJcIoHbi8Xi5ubkPHz5MSkrKy8vT0tJycnLCDedqaPXq1dXeehCgfvtkEO7YsSMqKkpbW5uagL5mzRp5tgXwzVRUVDp37kx3F3XSvHnz6G4BgAbVn1BfXFwsFoupC1bZ2NjgHB0AAKivqg9CgUAgOaDCZrP5fL4cWwIAAJCf90FYWlp648aNe/fuCYVCLS2t0tLSiooKQkhiYuLnz08CAACou6qOESYmJnbs2NHa2jo3N5fL5QYEBPTp02fHjh2+vr579uyhLtwAAABQ/1RtEa5du9bT0/P69euPHj0SCoVHjx7dunVrenq6r6/vhAkTan6+PAAAQN1SdWUZQ0PD48ePU3PttmzZcv36dX9//5q83sjIqKioSHJbUVNT0+Dg4M+MLykpUVVVxZVlAABADpSVlakbfn0GmxAiEAgyMzMlBwJNTU1rcm8XipWV1e+//y65OrOqqqrkKsDVYjAYuMQaAADUHmxCCJ/PF4lEkszkcDjUJfBrgsFgqKurUydaAAAA1DlMQgiPx9PQ0MjOzqZKWVlZ1L3BFFBpaWlaWhrdXQAAgPxUTZZp37695J7gt2/f/syN6Oqrp0+ftu7U1cKle+tfxhk0bbV+647vvC8HAADUCWw+n8/lcufOnTtgwAB1dfWsrKzLly9HRkbS3ZhcvX371nPQqMyh/xAjG0IIEZQtv/hHZvaK9SsW090aAAD8XFVbhF26dLlw4cKTJ09yc3MfPHggfcdwRbB03ZasbkuqUpAQwlEp7rfh8Klzn7otLQAA1BtsLpdLfeXm5qaw5wuGPYkSD1/yQYnJIqb2r169srOzo6kpAACQh+qvNapolJWVSUWJTJFRUaKsrExLPwAAIDcIQkII6efdTTni1Aeloixu7psmTZrQ1BEAAMgJgpAQQmZPm9Tk9WVewHpSnE0q+Yy4O3r7+vy9eQ1O/AcAqPc+eWNehaKiohJxL2Dzzj3/nhtZXFTcxr7l6isnGzduTHdfAADw0yEIq3A4nHkzp82bOY3uRgAAQK6waxQAABQaghAAABQaghAAABQaghAAABQaghAAABQaghAAABQaghAAABQaghAAABQaghAAABQaghAAABQaghAAABQaghAAABQaghAAABQaghAAABQaghAAABQagrCmhEKhWCymuwsAAPjBEIRfdvfePVsXD6OWLoYtnJw9e0VHR9PdEQAA/DC4Q/0XXPbzH/XH+tyh+4muKSEkIyXGc8jYG//us7e3p7s1AAD4AbBF+AWz/lyeO/IolYKEEGJsmzVoz7Q/ltPaFAAA/DAIws/h8/mFfDFR1/+gamL3JiGRpo4AAOAHQxB+DpPJJGLhx3WxWCT/ZgAA4GdAEH4Om83W1+SR7ATpIuNlcOuWtjR1BAAAPxiC8Av2b1mrd3goef2IEELEYmbMtYYXZ29fvZTmtgAA4AdBEH5BO2fnB5dOdI/fY7zZpdG2DgMK/SKC/CwtLenuCwAAfgycPvFlVlZW184ep7sLAAD4KbBFCAAACg1BCAAACg1BCAAACg1BCAAACg1BCAAACg1BCAAACg1B+L2ys7ODg4Nfv34tEuG6awAAdQ/OI/x2BQUFIyfPevD0pbBRa3ZhqlZZ+vG9Wx0dHOjuCwAAvgKC8Nt5DxweYjm0cvpe6mFm1puewwZH3blmYGBAb2MAAFBz2DX6jeLi4l4WMSsdB74vNbDM6TRz59//0NcUAAB8NQThN4qLiyszlr1JvbBRm9DoWFr6AQCAb4Mg/EaamprcslzZanG2no4WHe0AAMA3QhB+I2dnZ+6re6Q0T7qo/Wj/qEF96GoJAAC+AYLwGykrK+/bvLrBbm9GxDmSnUBeP9T5Z2hvW33PLl3obg0AAL4CZo1+Ox/vHpH2rdZv3xMe/F8jI8MJa2Z06tiR7qYAAODrIAi/i5GR0ebVy+nuAgAAvh12jQIAgEJDEP5c5eXldLcAAACfgyD8KfLy8oaPn2pg3dqsraeRdeuV6zYJBAK6mwIAgGogCH88Pp/frkvPUyzXjLkhmdOD0mYGr36UN2T0JLr7AgCAaiAIf7zjJ04mW3StdBhQ9ZjNLfVefDc26dWrV7T2BQAA1UAQ/ngBD8JKrTrLFIusuoSGhtLSDwAAfAaC8Mfjstmkki9TZFZWsNk4WQUAoNZBEP54fbu7az67+EFJLFKL9e/QoQNNHQEAwCchCH+83j4+LYTv1PxXkIpiQgjJT9U6PMK3v7ehoSHdrQEAgCzsrPvxGAzGHb/zW3ft/ftQn+LSMn09nRV/zOzp3YPuvgAAoBoIwp+CzWbPmT5lzvQpdDcCAABfgF2jAACg0BCEAACg0BCEAACg0BCEAACg0BCEAACg0BCEAACg0BCEAACg0BCEAACg0BCEAACg0BCEAACg0BCEAACg0BCEAACg0BCEAACg0BCEAACg0BCEAACg0BCEAACg0BCEAACg0BCEAACg0BCE8lZeXr5k1Tr7Tt2aO7uPmTYnIyOD7o4AABQaglCucnNzbdu5r4tViup3JHbEhcPEtWWn7hFPntDdFwCA4mLT3YBiWbBs9TvnqZVOg6mHwlY+mYY2I6dMfvogiN7GAAAUFrYI5epG4J3KNv0+KOk3ySwWlJSU0NQRAICiQxDKlVAkIiyOTJGhxCsrK6OlHwAAQBDKVWNLC5IU/UFJUMYszNDV1aWpIwAARYcglKsNS+brnp1GcpOqHpcXaZ6YtHDWFAaDQWtfAACKC5Nl5MrR0fH8nnVjZo4oFCsTrgorP2Xp7zPGjR5Jd18AAIoLQShvHTt0iA8LzsvLKy8vNzQ0pLsdAABFhyCkh7a2Nt0tAAAAIThGCAAACg5BCAAACg1BCAAACg1BCAAACg1BCAAACg1BCAAACg1BCAAACg1BWIu8evWqz7DRFq3a2Ti7z1+8orS0lO6OAADqPwRhbXHzVqBL72GXGv2WMCXoxYjzWxN1W7p45Ofn090XAEA9hyCsLcbPWpA99hyx6kCYbKKkVtFhXKLThGVrN9HdFwBAPYcgrBVSU1PLVPSIRkPpoqBN/6s3AulqCQBAQeBao7UCn88nHBXZKltJUCmgvhSLxREREa9fvzY1NXV0dGSz8Y0DAPgx8Pe0VmjUqBHJfEkqKwhb6X01/p6DfStCyMuXL/v+Ni6TZ1bSwEY1/5ZG2owTf+9wbutEW7sAAPUIgrBWYLFYsyePX3ViYsGArURFgxBCUp7pX1mw+vJJPp/frd+whP57iYkdIaSMkJzcpL6/9Xv2MFBHR4fmvgEA6j4EYW0xb+ZUPW2tpeu685W1CL/URE/znzOHrKysrl27lmfWgUrBKjqNch1Hnjj935SJ4+nrFwCgnkAQ1iKjRw4fPXJ4Xl4ej8fjcrlU8fWbt4V6NjIj+QbNo+MwjwYA4AfArNFaR1tbW5KChJCG+g1UilNlxjAL0hoZNJBvXwAA9ROCsLbz9PRUi7lISqXOrK+s0Hn899ABv9DXFABA/YFdo7WdlpbWvg0rJ/zule08TmjYnJGdoPdw95IZ4xo3bkx3awAA9QG2COuAPr16xty7vrpVxbDc/5Y3yQzzOzVl3GjqqZycnGFjJhs3d2xo7dCkdftjJ07R2yoAQJ2DLcK6QU9P7/dZM2SKRUVFDm7dk90XCGdvJYRkluZN3fd77Ms3/1u8kI4eAQDqJGwR1mFbd+9LazNSaN+n6rGqdsHQvfuOnyksLKS1LwCAugRBWIfdvPuIb9P9gxKTJbLqGBkZSVNHAAB1D4KwDmMymUQslK2KhUwmvq0AADWFv5h1WM/OHZSfXv6gJBSwXga3bt2apo4AAOoeBGEdNmXC2EZx55Ue/ENEQkIIyU/V+mfo71PG8Xg8ulsDAKgzMGu0DlNRUXlyL2Dh8tUXt3cSiER62pobli/o1q0r3X0BANQlCMK6jcfjbVu7ctvalXQ3AgBQV2HXKAAAKDQEIQAAKDQEIQAAKDQEIQAAKDQEIQAAKDQEIQAAKDQEIQAAKDQEYb2VmZn528Tpje1dLO1dho+flp6eTndHAAC1EYKwfnr58mWrTt1PcD3eTLr1dtKtEypd7N28Xrx4QXdfAAC1DoKwfho/+4/0/juFLXsRFoewOCK7nhmD942bhRv2AgDIQhDWT7HxL4mF0wcl09bxbxLo6QYAoBZDENZTTFZ1RXy7AQBk4S9j/dRAW4PkJn1Qyk/TVVejqR0AgNoLQVg/bf3fEt1jviQnsepxbrLO0RFbVy2mtSkAgNoIt2Gqnzp7uJ/buXLi3DE5ZQIGYWgrs3ZuWdbZw53uvgAAah0EYb3VqWPH549vl5eXi8ViFRUVutsBAKilEIT1nLKyMt0tAADUajhGCAAACg1BCAAACg1BCAAACg1BCAAACg1BCAAACg1BCAAACg1BCAAACg1BqIjevn3724Tptu07u/kMOnj4mFgsprsjAADaIAgVzqWrfs69hhxX9Xo29Mzd9itm/fekQ/feQqGQ7r4AAOiBIFQsQqFw4uw/siZcFrfoSlQ0SAPLwj5rolVtDx4+SndrAAD0QBAqlujo6MpGrYiqtnSx2Om3ExevUV/fvnPHZ9ho2/ad+40YHxYWRkePAAByhWuNKpbS0lKRkoZsVVm9qLiIEDJ93l/H78fmdplPXBo/S4u9O+HPmUO8/vp9Fg2NAgDIC7YIFYu1tTXznex2HuPNo7atW0ZHR/8bFJ476gQxbU1UNIilc86EC1uP/JeYmEgIycrKGjpmskkLR6Pmjq07db177x4d7QMA/HgIQsWiq6vbrb2DyrVVRPT/s2PSXjQIWLVw5pTzV6/l2g8jDMb70Ux2QcsBAQG3MjMz27h1P8PrmjLrYdrsh5E9d/8yY/mRf0/RsgoAAD8Wdo0qnH92bjZYtovr1Z0AACAASURBVOrw+rZME1tGUZa+kvDYmcMmJiYFRSViFdm9pgIlzfyi4iWrN6a5/S5s2auqqmuWO+b0gmUew4cMZDLxvxQA1G34K6ZwOBzOhpVL0mIj7u5Z8vzGqej7gS1btiSEOLdqoZYUIjNYO/lR65a2gcEPhLY9PnhCSU2o3yQhIUFeXQMA/CwIQgXFZrObNWumo6MjqfzyS1/9NwGMF0GSCivyYqOSV25ubmKR+INdplUYIpFILs0CAPxECEKowuVy7/mf93i5v8E2D/0TYxts7tArzy/w0hkmk+newYX1/MYHoytKmJkvLS0taWoWAOCHwTFCeM/IyOjWxdOFhYVJSUkWFhaqqqpUffnCOVfde6RzVUXNuxFCSF6K9qlJK/+YiwOEAFAPIAhBloaGRosWLaQrBgYGYUF+k+b+GXptiZAQPU31LWv+8vTsQleHAAA/EIIQasTQ0PDC8YN0dwEA8ONh1xYAACg0BCEAACg0BCF8r8TExP4jxlm0atekjeuUOQvz8/Pp7ggA4CsgCOG7hISGOnb75bz+wIQpQa/HX/+72Maufee0tDS6+wIAqCkEIXwX32m/Z406Kbb2IEw2YSsJnIamdl08Z9FKuvsCAKgpBCF8u/Ly8qziCqJnLl0U2fa49/AxTR0BAHw1BCF8O4FAwGBzZasMpkgspqMdAIBv8ckg3LRpk52dnaOjY8eOHd+9eyfPnqCuUFdXVxIUk7KCD6pJUU0b49JrAFBnfDIIvby8IiMjw8LCBg4cuGLFCnn2BHXI//6cp33UlxRlVT1Oj9c7M2Xzij9pbQoA4Ct88soyzZs3p75QV1dns3EBGqjeb8MGq6upzlncv5hwGUJBQ03Vg0d32dvb090XAEBNsV+/fv38+XPpkqWlpeRSk6mpqWvXrvXz86OjN6gb+vb26dvbp7i4mMvlcrkfHDKMiopatHbri7h4QyOjSSMGDRk4QPrZpKSk9PR0KysrLS0t+bYMAPAeOz09PTw8XLrE4XCoIMzMzPTx8dm7dy/utgNfpKamJlPZse/A0t3Hc3osJ53sX+YmR/+9+fh/ly6fOkIIiYmJGTJ2ahZDQ6RlxEiK7tSmxYHtGzQ1NeloHAAUHUP8iQl+2dnZXl5ey5cv9/b2/szr3d3dly1b5ubmVsPlFRcX83g8RjV3eYV6JT8/v2m7zlkz7xK2kqSodWL8iXlDHB0c7Dp4pv96lBhaU3V26Km2b0/dv3GZpmYBQKF9crLM1KlT8/PzDx06NGjQoGXLlsmzJ6gH7t+/X2HjJZ2ChJD8Nr+eunR978HD2e0mSVKQEFLpNPhlCUdmFz0AgHywg4ODO3To8PETW7ZsKS0tpb5WUVH51Otzc3OPHTt2//596qGenp6vr+9nlicQCAQCAbYI672ioiIBV3ZnKVFWz08vCol6XtlkkswzJSYOT58+tbKyklN/AKAYWCzWF28hzo6Kiqo2CA0MDGqyjMrKyqKiory8POohg8EQiUSfGS8SiUQiEYKw3mvevDlv87GyD4tKb+67OtlFx70mJbky4zmluWpqatQPT35+fmRkJJ/Pb9WqVcOGDeXVMgDUQ19MQUIIOzMz83uWoa+vP2nSpJofIxQIBEpKSgjCeq9ly5b2hry7d/fwO04gDAYhhPHqgX7k8Ql7boeFhV1Zsi/P2uP96LJC5fhbHh4rlJSU1m3ZsXHvIUFTDyGLo/RyzS9dXHdtXM1isWhbEwCo79iYqgc/ycV//5nz57L/1jmxDJuJ81OtTfSP+J1TV1f38PDwOX3hyqFfc91mEO1GjHfheoFrtq9eqqqqeuzEqdXnHuTPvEtYHEIIEYuP+S1XX7Jyw8oldK8NANRbjLNnz/br1++bX49Zo/B5lZWV7969MzQ0VFVVla4HBt3efeRUUkpqG7vm86dPNDMzI4TYung8G3iUaEjtDhUJG250To+LknPbAKA42D169KC7B6jP2Gx248aNP6539nDv7OEuU8wrKPogBQkhTBbh6RYWFmpoaPy0HgFAoTE/MyMUQM64bBYRlMsURSV5PB6Pln4AQBHgNkxQiwz+xUc5eJ90hfnsukPLFpgsAwA/D66mDbXI0oVz7/cdHHM6Lt+uP+Eoq8XdMEm6c9jvHN19AUB9hiCEWkRZWfnetYtXr/pdCggsLajoPtB56ODl2BwEgJ8KQQi1Ts+e3j17VnOF26dPn85evCru5Ssul9unh+fyP+bh2CEAfD8cI4S64fylKx5DJwbYzUqa+eD1+Os704xatvcoKCgghAiFws07dlu0dDZo7mhm1/Z/6zYJBAK6+wWAOgNbhFAHiMXiafMX5Uy6TlS1CSGErVTRfnQyYfxvw9Z1Kxb3HeYbVG5cMukm4aiQSv6qWxtv9Bl0x+883V0DQN2ALUKoA96+fSvQtahKwf/Hb93PLyDoyZMnj5KLS3yWE44KIYSwuaXdFz6t0Lx9+zYtrQJAnYMghDqgsrJS5o5OhBDC5goElffuP8ix8pJ5Jq9Zz4C7D+TUHADUcQhCqAMsLCxI6jNSyZcuMmID2zs7EkJINVfsE3/qjtMAADIQhFAHcDicOZPHa/47jpRW3fCLvA1peGPZioVzOrR30Y2/JjNeO86vq5urvLsEgLoJk2Wgbpg3c6pRwwZ/repVxuAyhPxm5o0OXDljYmJiYmLSzoR3+8qS4u4LqMkyqrc22SkVuLu7090yANQNCEKoM4YPHTx86OCysjIlJSXpm21e+PefLTv3bN/lWVYpVGazxg4fPH/WKRr7BIC6BUEIdczHl4lnsVhzpk+ZM30KLf0AQF2HY4QAAKDQsEUI9dzZ8xeOnL2SmZXt3Npu4aypDRs2/PJrAECRIAih3hKJRF79hoaWaeU7TyCtGoS+eXjCrcepvVvc3TrR3RoA1CLYNQr11pFj/z4SmuQP3EFM7Ym2sdBhQOa4iyMmzxKJRISQtLS0oWMmW7ZqZ2nvMnLSjMzMTLr7BQB6IAih3jpy9kqRs+8HJU2Dcn3r2NjY58+ft3bvcVrD++2UoLeTbh1nu7Xq1P3169c0dQoAdEIQQr1VWFRIVLVkiiJV7YKCgjEz5mcMPShq0Z0w2YTFEbbySf9l28Q5f9LSJwDQC0EI9VarFjaMhHCZIjPxSbNmzV4nJhETuw+eaNwu5kWc/JoDgFoDQQj11h8zJ+vdXEFyk6oei8XKgVvdHVro6OgQRjV3vReTj69ZCgD1H2aNQr3VuHHjCwe3j5j8axHPSKSmx3gX0beb+/Z12xkMho66alZBGtE0fD86O8GggQ59zQIAbRCEUJ+1d3F5GfEgISEhJyfH2tpaTU2Nqm9euWj4/FG5vx4k2saEEJKdoHt89Pbdq6lnCwsL9x44FBIda9RQb8TAvg4ODnT1DwBygCCEeo7BYFhYWFhYWEgXe3TvdobDmTx/RH6FiBCxHo+7Z8+aDq6uhJBHjx/3GzU5u62voNFIUpR5fNLiAa62ezavpal9APjpEISgoDp39ngR6lFeXs5gMJSUqu76KxKJBo2enDb6bNWWIiE5dj1OHRnZ99p1L6/u9DULAD8RJsuAQlNWVpakICEkKiqqzMBWkoKU/I5T9/17Vu6tAYCcIAgB3svJyalU++hipJoGGVnZdLQDAPKAIAR4r3Hjxpy0GNlqcrSdtRUd7QCAPCAIAd6zsLBoqs1mRV16XyrO1gtYNWfyWPqaAoCfC5NlAD5w+eThQb4Tox7vLzNtq1ySoZwYtm/LaisrbBEC1FsIQoAPaGtr37xw6s2bNzExMQYGBq1abZWeTQMA9Q+CEKAalpaWlpaWdHcBAPKAY4QAAKDQEIQAAKDQEIQAX+HBw4eDx0x27tp79JTZL168oLsdAPgBEIQANTXrjyW9Z60+rT8kpPuOf7ieHQeN273/H7qbAoDvhckyADXy5MmTozdDc8ZfIAwGIYRoGWY37bhkU+cBfXo1aNCA7u4A4NthixCgRk5fvJrrOLIqBSkc5SLbvoGBgfQ1BQA/AIIQoEZyC4rEPG2ZIl9FJz+/gJZ+AOBHQRAC1IijnbVKUphMUSsltEWL5rT0AwA/CoIQoEaGDR6kF3OWJIRLKszoqyZlCa6urjR2BQDfD5NlAGqEx+MFXTo9dOzUdzfEIl0LRuozx6amRy6eZjAYhJDKysode/ef9QsoKips7+SwbMFszKABqCsQhAA11bhx45Ag//T09MTERCsrK23tqkOGJSUl7Tx7JjTqXNxpDVFWj4kLOtuh69V/Dzg6ONDbMADUBIIQ4OsYGBgYGBhIV/63fku8VX9+p4nUQ2Gb/pmNWg+fOPZF6D06GgSAr4NjhADf6+zVa3zn4R+UGljmiZSys3Ffe4A6AEEI8L0qyisIlydTZKhoFhcX09IPAHwVBCHA97KxbkYSPjyzQlQpznxlYmIiKeTl5ZWVlcm7MwCoAQQhwPda/eccvYtzSV5K1ePKCvVzc8ePGMZmswkhx06cMm3haOM50MK5a8v2ncPCZE9GBAB6YbIMwPeyt7c/s3PNmBlDSpT1xMrqJPXZjPGjF86ZQQjZue/An0duFEzwJ6pahJCMjJc9RvgGntpvZ2dHd9cAUIUhFou/5/Xu7u7Lli1zc3Or4fji4mIej8eQvmAjQH2RlpZWXFxsaWnJYrEIIWKx2Ni6ddqMO4Sr+n7Q29BuL3ZeP3ucti4B4EPYIgT4YQwNDaUfZmdnizQNPkhBQoiFU+zFeLm2BQCfhWOEAD8Ll8sV8z+aICMUsJj4vQOoRfALCfCzaGpqanNEJCdRush+cr5b55oeSgAAOUAQAvxEh3ZsaPDPQMbzm0QoIBXFSg8OmNzfsnrxArr7AoD3EIQAP1E7Z+fwmxcHFVyx3ONpc/SX6SZZMY9u6+jo0N0XALyHyTIAP1ejRo1OHtj1mQHl5eXKyspy6wcAZGCLEIAeAoFgxdqNRtatzdp6GjSzHzlxel5eHt1NASgibBEC0GOQ74QbZY1KZwYTNpcQ8m/oqceevZ4+us3hcOhuDUCxYIsQgAZxcXHBcWmlPf6iUpAQUuk0ONms84lTp+ltDEABIQgBaBAeHl7UxEOmWGLV+daDqiuRJiQkbN+1e9YfS06cPFlRUSH3BgEUCIIQgAZsNpspEshWK/lcDpsQsm7rzra9hs0MYW4pcRh7+rmVg2t0dDQNXQIoBgQhAA1cXV3VnvuRD6/0q/nsQp9u7qGhoWuPXMyaHijqMJrY9Sj1XpQ07FifX8eIRCK6ugWo3xCEADQwNjYe0aer1pGRJD+VEELKi9T8lrckqT29vXcfPpnrPpswWe9H6zcubmgbERFBV7cA9RtmjQLQY8PKpZ2uXF2yblxWTp4aT3XCiCHTJy1lMBhJqemkTSOZwRVapmlpabT0CVDvIQgBaNO7V8/evXrKFJtamAZkviQGTaWLytkvzcz6E0LKy8vXb93pF3iPz6/w7Nj+z7kzNDQ05NcxQH2EXaMAtcvUsSN0A9eRimJJhfE2RK8kyc7OLisrq4Wz2+oY1qOuWyN679+Samjj7Pb69WsauwWoB7BFCFC72NjY7Fgyd+aiziUtfErVDLTTIgyLXl357xiDwZixcGlCx3ki+97USH67kakNm4+aOvee/3l6ewao0xCEALXOkIH9enTrcvfu3dS0dNsWvu3bt2cwGISQO/cfimZv+mCohVP82XdisZgaAADfAEEIUBtpamr6+PjIFMUMBmHIHs4Qs5UEAgGXy5VXawD1DY4RAtQZejpaJDf5g1J5kbIYKQjwXRCEAHXG+sULtE9OIMU5VY8rijVPTlo8dzqtTQHUedg1ClBndO/W9WB5+bQF3gJtM8JiMzNeLl8we6zvCLr7AqjbEIQAdUnf3j59e/skJiZWVlZaWFhIz5ERCoWXLl26Hx5toKfdy6ubtbU1jX0C1CEIQoC6x9TUVKby7t27rr8MTTfpUGTWnpGWu/bw5OE9Om5etYyW9gDqFgQhQH3Q59cxr7w3is0dCSFiQrLbDT901Nft0uW+vWWnngKADEyWAajzUlJS0vhsKgWrMBj5nvO3H/yXvqYA6gwEIUCdl5GRIdYyka3qNsJ1ugFqAkEIUOeZmJgwsj664mjGS3NzMzraAahjEIQAdZ6+vn4zQy3W06vvS5V8Hb8lC6aOJYRUVlau2bTV3K6tQXNHc7u2azZtFQgEtPUKUPtgsgxAfXD+6P5eQ0a+ijyVb9peqTyPF3PpzxkTO3XsSAjxHvDrA1azkimBhK1EBOUrA9YH3h1+48Ip6oWZmZl3797Nzctv1dLO2dmZ1pUAoAeCEKA+0NXVfXjzSnh4eGRklI6OSadOU3R1dQkhjx49ishjlfy2qGocR7mkx6KIY6MePnzo4uKyeefeNTv3F9v1rVDW1vp3hyVjqd/po3p6etTYkpKS2NhYFotlY2OjrKxM16oB/GwIQoD6w8HBwcHBQboSdO9BTlMvmWE5TXsE3ntQVl6x4vDlvBl3CItDCMkh4/KfXes7fEzwtYuEkBXrNu04eExk7sgQCVnvwpfOmzlhzCg5rQaAfCEIARSPWCwWizfsPpjntZhKQYqwhdfLO1syMzMPnzi9PiCuaFZw1bP80gX7fbW1NAb170dbzwA/DSbLANRnHh3b68b5yxR14/27dHJ9l5hI9JvIPCXSb5yUlLRlz4GiXza8z0iuav6g7UvXbZNDwwDyhyAEqM/atWvnoCvm+a8ggnJCCBGU8/xXOOiIXFxcDI2MSG6SzHhmbqKurm4lS5lwPjwoqK6fX1Qsr64B5ApBCFDPXT1zbFEHPbNdXfQ3tjPb7bmog97VM8cIIdN8h2ndXEPEYslIxpvHxsoiMzMzMb9U9l1EQiYRyxYB6gUcIwSo59hs9vxZ0+fPkr1tYR+fXmMehx/e2S23zXART1cjIdgg9dHFCycZDIZ9i+YBL4LE1h7v3yTslFcXd7n2DSAvDLH4u/7Lc3d3X7ZsmZubWw3HFxcX83g86XvHAACN4uPj/a7fTMvKdXVs1atXLyaTSQhJTU3t4NU3vUX/subeRChQe3reLPnu/RuXNDU16e4X4MfDFiGAQmvatGnTpk1likZGRrGh93buO3D9zjquErd3z06jRy5lsVjUs+np6Rt37I14FmdsoD92WH/qtH2AugtbhADwFfyv3xg1849st9ki09akIF0neGevlsaHd2+luy+Ab4fJMgBQU3w+f/T0eZkTrogcBxL9JsSqQ67viUvPc67fuEENuHDpcuc+Q5o5dezz65iIiAh6uwWoIQQhANRUSEgI37I9UdOVLua7jDt0+hIhZNCo8aO3ngtyWhQ/+uol89Fdx/y+eccemjoF+AoIQgCoqYKCAoGqrmxVTS87N+/27dsBb0vyhu4jBk0JR4VYtM2dcGn1rgNZWVmSgSUlJTk5OXLtGKAGEIQAUFNNmzZVTomUKTKTnjjYWp++fD2v9ZAPnmBzS1r43L17lxASEhraop27ZfseLbyGmdi0OXT0uNx6BvgizBoFgJqysrJqpsXMCztd6TioqpTzTu/2xulBV+ctX0u0NGTGC7jqJSUlUVFRPUdNyx5+mDSwJISQsoKZ+6YVlZROmziOGhYdHR0WFq6urubq6mpkZCS/9QEghGCLEAC+ypVTR3oWBjTY4alzcZ7+0d+anBpx6eheIyOjDm3slN8+kBmsmfiwZcuWvy9bm/3LlqoUJISoaBYM27t6806xWFxUVNSl98Auk5ZNCCoe+d9L+679lqxaL+9VAoWHLUIA+AqampoX/v0nMzPzxYsXhoaGjRs3ps7B/+3Xoau3uSWZtxNbdSCEELGY8/CwNa/C3t7+RVw88f7g5lCEoyLSMsrKyho/c/49w96C3kMIIZWElHWZve2or531+QH9fqFh3UBRYYsQAL6avr5+p06drKysqBQkhPB4vHt+5zo926G/1V3/2Ai9Dc6/Kj/zO3OMEMJis0glX+YdxBUlIpHoQWSMwEnqyCKTld9r5bqdB+S1HgCEYIsQAH4UU1PT21fOlpSUpKSkWFhYcDhVd3Hq5uF28Mn5SqfB74dmJ+hwSXl5OVPHVPZddE0zMjKoL4+fPL1y086C4mJVZaVJo36dMXkCm40/WfDj4acKAH4kHo8nc8221YsX3HT3Sq0orHAYTDjKjLggPb9Fh4/t09PTExekyb6+IF1bW4sQMnbanP9iCwqGnSJquoRfuuTWxkvXBt7xOy+3FQHFgV2jAPBz6ejoxDy6PdMst8W/Axrv6zas5HrErcttnZzU1NRszAyZsQHSgzUC1k7xHf769euL9yMLhuyuOnmfq1rSY1G0QO/G/1/CJiwsbPq8v/qPmrhu09aCggL5rxTUJ7jWKADQJjMzs0ufQUladgWNPRgVJbqR//Zqa31wx6ajR4+OvZ4t8Jj2weiYa9OUHm9bu3LCzHlnH8bmuIwn6vqcxDDdx/vPHdrt0q4dTSsBdR52jQIAbfT19aMfBPn5+d19HKGtoeY9bW3Lli0JIZVCoYjBkh3NYAkqhf7+104/Sckfd46qCczapNv2HDymf8LTUGrmTklJybNnz8RicfPmzdXV1eW7QlAnIQgBgE4MBqNnz549e/aULrZ1ctLe+0e2+2TposbLgG5jO/194lx+hw/qRNu4zNAuMjKyTZs2m7bvXrfzb6FFWzGDyXr7eKrvr4vmzZYMFAqFb9++1dXV1dbW/pnrBHUMjhECQK1ja2vrYq6t6reCVFYQQohYxL1/wCIvsk/v3lk5uUSjocx4gbpBdnb2gUNHl/93P2Pm3ez+23P6bc2ceW/D9Wdbd+0lhFRWVi5YstLQunWHMX9adxlg194jKipK/usFtROCEABqo7NH9//pom28zU1/c0ejza7j9BKDr19iMpktmjZmpMTIDOamxVhZWa3etrug/2bCVqqqsjiF/TZu2rWfEDJ2+tztT/lZvz/O+O1Y5uRrMd7buw0alZiYKOeVgtoJQQgAtRGHw/nj91nJz8NTooJTnoftWL9KTU2NEDJn8ljdgFWk6P1NLViRF5tqMi0sLIrLKoiq1gfvosQrE5Hc3Fy/Ow9LvRcR5v8fDDJoltX1r1Wbd8pvfaAWwzFCAKjVZE6it7KyOrp11biZvctN2lSoGygnhtoba546eZgQwhSLiFhEGB/8f8+o5MfHx4vMPrzGGyHixi6hl/b/7OahTkAQAkAd49XV81XE/ejo6PT0dFtbXwsLC6ru6d7xRPh/7++MQQgz+qqrs5OqqiqzvEj2XcoKeao86svU1NTjp848e/WuVTPL4UMHN2jQQC7rAbUFghAA6h4lJSUnJyeZ4rY1yyO693mXGVts25swmLznV41fX997/aKOjg4nNYYU51Sdnk8IIYQXdnxo3x6EkEPHTsxftTm73XhRAx9W+Is1u712rV7Sv29vua4P0Aon1ANA/SEUCg8dPX4lMFgsFnu5uYwdNYLas3rF/9roOYuzuy8VN2lPirM17u9rXvrs3rWLaWlpbboPyJ52k3BUqt6irFB/Z9fn92/q6up+bklQj2CLEADqDxaLNWbUiDGjRsjUe/XwCrGxXrR6U/iRdXoNGozo7zN65Bomk3n2wqV8p1HvU5AQoqJRaD/Iz9//t+HD5dk50AhBCAAKwdzc/OjebTLFpPTMSs3WMsVydePktEx59QX0w+kTAKC4WlhZKme+kCmqZcfaWFnS0g/QAkEIAIqr/y99tSJPkZx370vpcVpx17p3705fUyBv2DUKAIpLU1Pz8vH9g0b/WmDYuljHSj07Vjf3xbkzR1VUVL78YqgvEIQAoNAcHRziw+8/fvw4ISGhSZMOTk5O1F0sQHEgCAFA0bHZbFdXV1dXV7obAXrgHx8AAFBoCEIAAFBo2DUKAPBJkZGRF/1vZOUWdGpr3++XX2SuAA71A7YIAQCqIRaLR0+d3XX8n0vfNdpZ2X70sQhrxw7v3r378iuhrsF/NwAA1Tj678mzL0sLx1+gHpa06PbmbUi/EePD71yntzH44bBFCABQjd2HTxZ6zJGuiC3aJhcLMzNx9bX6BluEAADVyMrOItrGMkWxtnFmZqa+vr5YLL527drth2FKXE6PLm4uLi60NAk/BIIQAKAa5mZmr9PjiLGtdJGR8crExCQ/P79Ln0FvVJrkW3Ullfxd8za7GO0+d+wAh8Ohq1v4Htg1CgBQjYVTx2lf+YtUVkgqnLBTjtYWWlpavlPnRNmOzR+wlbTqRRz65Yw8Hii0XL1xK43dwvfAFiEAQDW6dOm8dMyb/211K2vuXaGkqfH2Xit9pRNH94tEovshYcJ5e6QHl3aedeTvHosXzCWElJSUnDx1OuxZvKWJwYC+vS0sLGhaA6gpBCEAQPWmTxz768BfHj16VFBQYG+/unnz5oSQoqIiBk9bdihXtayCTwh59Phxf9/JOS0HVRi7MWJS1x/4dc7YYfNnTqVGJScnX7xy9VViqkOLpv379cOlvWsJBCEAwCfp6ur27NlTuqKmpkZKcolYRBhSh5ZKcjXUeAKBYIDvpFTfc9QsGzEhWc5D1+3u1c3NtXXr1jv2Hli+bV+u40ihloPKhcgF/+tw/sheJ0dHOa8RfAxBCADwFRgMRr+ePQ4FbSvvPLOqJBZpXP5z1kTfhw8flpm7fDDXlMnO7TRj//EzkzicpbuO5EwPIiwOIaTMrkeK0/B+Iwa8jnzE5XLpWA94D5NlAAC+zta1K3qy4xvs9VG6tUX15nr9bZ0ntDcdP3pUZmZmubqR7Ghtk8TU9APHT+d2nEGlYBWdRqXm7R8+fCjPzqFa2CIEAPg6XC73vyN/x8fHh4WFcblcF5eRxsbGhBAzMzPV7MulHw5mpMe1aGIRl5AkNjeReZ9STdO0tDR5dQ2fhCAEAPgWTZs2bdq0qXTFwcFBN/9VzttQsYVTVak0T/fOxon+/+09dIyR/kps2lp6PC8n3syso9wahk9BEAIA/BhMJvP6vNRmzgAACSlJREFUuX99ho5Ke2Sab9CaV5zCexm4b/Mqc3Pz8SN/3e89KLtFN6KiWTU6KVorM6Zt27a0tgyEIAgBAH4gMzOzqPuBISEhL168MDa2bd9+maqqKiHEwsLi73VLJ//erbi5d4m6sVZGlH7Os8v/HWOxWHS3DAhCAIAfisFgODs7Ozs7y9T7+vTs4t7p7t27ySmpNtbDOnTowGRiumKtgCAEAJATdXV1mbMSoTbA/yMAAKDQsEUIAEAzoVC49+Chi9dvl5WXd27v9PuMKTwej+6mFAiCEACAToWFha7deycYdSq2n0c4yiGxNw44dbx9+Uzjxo0JIUKh8OLFi3dDI3U11Xv36NaqVSu6+62HEIQAAHT6Y8WauObDBS4jqYcVHccnN3IYPnHmw5uXk5OTu/QZnGbsWmTZkeQWbRu3sJdTs4M7NjEYDEJIRETE6m374l+/NjczmzNhZKeOOCXxG+EYIQAAna5cCxA4Df2gZO7wJiWDz+f3HznhZddVRb3/R2y9iNPA7PEXz70sO3zsX0LIqo1bu41b8J/hr9EDj18yH/vLvI0zFy6mZwXqPgQhAACdKkUiwpa97jZDRT05OTkht0TcpL10vdBz3u7DJ5OSkrYcOp0z4SJp3I6oahNzh9zRp4/dDHn69KkcG68/EIQAAHQy0NcnWW8/KFVWMIpz+Hw+0fr4Et7GmZkZN28G5Nv1J0ypY1sMRm6b4eeu+P/0dusjBCEAAJ1W/zFb579ppDS/6nElX/3c3GnjRhkbG8sGJCEkPd7U1KywuESgrCnzjFhFI6+w+Of3Ww9hsgwAAJ26dvXcXVg066/ulQbWYo4K41349LEjF86ZwWAwnFpY3Qg7JXAcXDVUKNC+8ufC5VOVuBztKyfyyAjp91FLfOwypGpO6YsXL85f9kvJzHGxbzFw4ADc8vDzGGKx+Hte7+7uvmzZMjc3txqOLy4u5vF41JQnAACgVFZWvn79uqysrHnz5pLcKioq+mX4mKhMfqFFR66gSPXZ1XmTRs+ZNkkkEtm7dnnmMFVk35sayXhx2+LmX7Gh97hc7ty/lh2+eifb0Zeo6agmhjSI97t25qi1tTV9K1fbYYsQAIB+bDa7WbNmMkV1dfWAi6djYmIiIyPV1Zu4uk7V09MjhDCZzMDLZ0ZPm/to83qGYTNx5htbc4Ojfue4XO716zcO3n6WN/kaYTAIIaUtur1r1b/Pr2Piwu/TsFZ1BIIQAKBWs7W1tbW1lSnq6eldOnGouLg4ISHB1NRUQ0ODqu88fDLPfTaR3utmZJOvahQXF/dx0AIFk2UAAOoqNTU1W1tbSQoSQtIyMoi2scwwgZZJRkaGfFurS7BFCABQf1hZWoRlxBNNA+kiJyPO3Nyc+jogIODm3YcMBqObW/vOnTvT0GLtgy1CAID6Y+6k0TrXlhN+qaTCfH7DQoNhampaUlLSoXvvgWtOrMuzXZvTfMD/jnTq0besrIzGbmsJbBECANQfbdq02TR/0vwV7mXNvUtVGmglPWrMLbp84hAhZObCJaHGvfjtfamRefY+j+/tnfvX8p0bVxNCQkND5y5b++Ztgpqa2rB+PvNnTVOcky5w+gQAQH2Tn5//8OHDrKwsOzu71q1bU0XDZvbpcx8ThtSOQJHQaHP7lNiIff8cWbj9SG6fDcS4BakoVg3e2zjhWtidGwqShdgiBACob7S0tHr06CFTFDJYH6QgIYTJqiSM8vLyRas35s66RzgqhBCipFbaZc7r6/y//zk8ZcI4ebVMJxwjBABQCEpMQiorPigJypRZzMjISKGlS1UK/r/SVv3P+QfKtT/6IAgBABTCmOGDef4rieRwmFisdnXphFG/8vl8MVtJdjRHuYLPl3OHdMGuUQAAhbB4wdzUWfMv7OhabN2dQQjvxbUBXdovmD29oKCA+XoOEYulT8NnxwW6t3OgsVt5QhACACgEJpO5b+v6P9+9CwkJYTAYzs6/NWrUiBCira09uGfXI+fmFvVeSe0gZcTdafho95yNt+huWU4QhAAACsTMzMzMzEymuG3dyia7963f5i5U1iT8kpbNGh+4flFbW5uWDuUPQQgAoOiYTObMKRNnTplYVFTE4/GYTMWaPoIgBACAKurq6nS3QAPFin0AAAAZCEIAAFBoCEIAAFBo33uMkM/nx8fHq6mp1XB8RkaGnp4ei8X6zuUCAAB8kaWl5Renv37vRbfbtWtXUlKipPTRVQk+4dWrV/r6+tK3kQQAAPhJ+vXr98cff3x+zPcG4dfq0aPHjBkzvLy85LlQAACAT8ExQgAAUGgIQgAAUGgIQgAAUGjyDsJx48bZ29vLeaEAAACfIu/JMgAAALUKdo0CAIBCQxACAIBCQxACAIBC++lBmJKSYm9vP3LkyNu3b//sZQEAAHytnx6ExsbGISEhXbt2nThxYufOnV+9evWzlwgAAFBzPzEIIyIiDh06RAjhcrnDhw+Piopq3bq1s7NzSEjIz1soAADAV/mJQaikpHTw4EHphxs3bly4cGHv3r3T09N/3nIBAABq7ueeR9i8efMDBw64uLhIF319fcViMbWxCAAAQK8fv0UYGBiYnJxMfT1r1qwpU6YIBALpAZs2bbp27VpKSsoPXzQAAMDX+vFBePv2bff/a99+XRYGwjiAH56MRVfEaLwgaFoaOOT+BzlwZaLdYhNk07D/YMU0TKIg2CyazAaxXrEIFlkQZWAYjPdt7ysnGr6f/BzPtYfv/bDtdBa6rqtpWrfb/Zk7DcPodDqz2Ux5awAAgP9SPwg9zxNCWJYlpaSUzufz7XYrhIjjOKtxHCdLjQAAAB/0rjvCwWAQRdFmsymXy+fz2XXdw+EwGo2azWY+nyeEJElCKX1HawAAgL9716tR3/cdx7FtW0pZLBZXq9VkMlkul4yxxWJBCMEUBACAb6AyEd5utzAMT6dTq9WqVqvkdy5Ma+73ey6XS0MhAADAxylLhHEc1+v14/FYKBQ451EUEUJ83xdCcM6v12tapmkapiAAAHwPOhwOX178eDyyE85er2eaZhAEl8tlv99Pp9NSqVSr1TjnlUqFMaZmvwAAAEq9ngillIyx7PGnruv9fn+323met16vx+Nxu91Of803Gg0lewUAAFDuCa2QA0p4Xr5eAAAAAElFTkSuQmCC" />
```

We see that some eigenvalues clearly belong to a group, and are almost degenerate.
This implies 2 things:
- there is superfluous information, if those eigenvalues are the same anyway
- poor convergence if we cut off within such a subspace

It are precisely those problems that we can solve by using symmetries.

## Symmetries

The XXZ Heisenberg hamiltonian is SU(2) symmetric and we can exploit this to greatly speed up the simulation.

It is cumbersome to construct symmetric hamiltonians, but luckily su(2) symmetric XXZ is already implemented:

````julia
H2 = heisenberg_XXX(ComplexF64, SU2Irrep, InfiniteChain(2); spin = 1 // 2);
````

Our initial state should also be SU(2) symmetric.
It now becomes apparent why we have to use a two-site periodic state.
The physical space carries a half-integer charge and the first tensor maps the first `virtual_space ⊗ the physical_space` to the second `virtual_space`.
Half-integer virtual charges will therefore map only to integer charges, and vice versa.
The staggering thus happens on the virtual level.

An alternative constructor for the initial state is

````julia
P = Rep[SU₂](1 // 2 => 1)
V1 = Rep[SU₂](1 // 2 => 10, 3 // 2 => 5, 5 // 2 => 2)
V2 = Rep[SU₂](0 => 15, 1 => 10, 2 => 5)
state = InfiniteMPS([P, P], [V1, V2]);
````

````
┌ Warning: Constructing an MPS from tensors that are not full rank
└ @ MPSKit ~/Projects/MPSKit.jl/src/states/infinitemps.jl:160

````

Even though the bond dimension is higher than in the example without symmetry, convergence is reached much faster:

````julia
println(dim(V1))
println(dim(V2))
groundstate, cache, delta = find_groundstate(state, H2, VUMPS(; maxiter = 400, tol = 1.0e-12));
````

````
52
70
[ Info: VUMPS init:	obj = -1.141245853330e-02	err = 4.0215e-01
[ Info: VUMPS   1:	obj = -8.788989897232e-01	err = 8.3126786202e-02	time = 0.13 sec
[ Info: VUMPS   2:	obj = -8.857995903945e-01	err = 6.7432032291e-03	time = 0.03 sec
[ Info: VUMPS   3:	obj = -8.861329058794e-01	err = 3.4413406908e-03	time = 0.03 sec
[ Info: VUMPS   4:	obj = -8.862249298781e-01	err = 2.0932139560e-03	time = 0.03 sec
[ Info: VUMPS   5:	obj = -8.862609030986e-01	err = 1.0275091935e-03	time = 0.03 sec
[ Info: VUMPS   6:	obj = -8.862754866598e-01	err = 7.4461572015e-04	time = 0.04 sec
[ Info: VUMPS   7:	obj = -8.862819270462e-01	err = 5.8510697988e-04	time = 0.05 sec
[ Info: VUMPS   8:	obj = -8.862849475005e-01	err = 4.8578532406e-04	time = 0.04 sec
[ Info: VUMPS   9:	obj = -8.862864073427e-01	err = 4.0315541644e-04	time = 0.05 sec
[ Info: VUMPS  10:	obj = -8.862871279225e-01	err = 3.5445010854e-04	time = 0.05 sec
[ Info: VUMPS  11:	obj = -8.862874891630e-01	err = 3.1822823463e-04	time = 0.05 sec
[ Info: VUMPS  12:	obj = -8.862876773439e-01	err = 2.7443998839e-04	time = 0.05 sec
[ Info: VUMPS  13:	obj = -8.862877793817e-01	err = 2.2054254501e-04	time = 0.12 sec
[ Info: VUMPS  14:	obj = -8.862878353269e-01	err = 1.6605215489e-04	time = 0.04 sec
[ Info: VUMPS  15:	obj = -8.862878654264e-01	err = 1.1979575296e-04	time = 0.04 sec
[ Info: VUMPS  16:	obj = -8.862878812303e-01	err = 8.4769125422e-05	time = 0.05 sec
[ Info: VUMPS  17:	obj = -8.862878894026e-01	err = 5.9778171991e-05	time = 0.05 sec
[ Info: VUMPS  18:	obj = -8.862878936100e-01	err = 4.2285093952e-05	time = 0.05 sec
[ Info: VUMPS  19:	obj = -8.862878957743e-01	err = 3.0076620714e-05	time = 0.05 sec
[ Info: VUMPS  20:	obj = -8.862878968901e-01	err = 2.1497977400e-05	time = 0.05 sec
[ Info: VUMPS  21:	obj = -8.862878974666e-01	err = 1.5431141918e-05	time = 0.05 sec
[ Info: VUMPS  22:	obj = -8.862878977650e-01	err = 1.1107227296e-05	time = 0.05 sec
[ Info: VUMPS  23:	obj = -8.862878979198e-01	err = 8.0121943412e-06	time = 0.05 sec
[ Info: VUMPS  24:	obj = -8.862878980003e-01	err = 5.7884597378e-06	time = 0.05 sec
[ Info: VUMPS  25:	obj = -8.862878980421e-01	err = 4.1862569480e-06	time = 0.10 sec
[ Info: VUMPS  26:	obj = -8.862878980639e-01	err = 3.0295966217e-06	time = 0.05 sec
[ Info: VUMPS  27:	obj = -8.862878980752e-01	err = 2.1934508934e-06	time = 0.05 sec
[ Info: VUMPS  28:	obj = -8.862878980811e-01	err = 1.5884579833e-06	time = 0.05 sec
[ Info: VUMPS  29:	obj = -8.862878980842e-01	err = 1.1504627834e-06	time = 0.05 sec
[ Info: VUMPS  30:	obj = -8.862878980858e-01	err = 8.3325912425e-07	time = 0.05 sec
[ Info: VUMPS  31:	obj = -8.862878980867e-01	err = 6.0349319385e-07	time = 0.05 sec
[ Info: VUMPS  32:	obj = -8.862878980871e-01	err = 4.3705144218e-07	time = 0.05 sec
[ Info: VUMPS  33:	obj = -8.862878980874e-01	err = 3.1648278136e-07	time = 0.09 sec
[ Info: VUMPS  34:	obj = -8.862878980875e-01	err = 2.2914950477e-07	time = 0.04 sec
[ Info: VUMPS  35:	obj = -8.862878980876e-01	err = 1.6589609004e-07	time = 0.04 sec
[ Info: VUMPS  36:	obj = -8.862878980876e-01	err = 1.2008839249e-07	time = 0.04 sec
[ Info: VUMPS  37:	obj = -8.862878980876e-01	err = 8.6918909626e-08	time = 0.05 sec
[ Info: VUMPS  38:	obj = -8.862878980876e-01	err = 6.2903889958e-08	time = 0.05 sec
[ Info: VUMPS  39:	obj = -8.862878980876e-01	err = 4.5519032022e-08	time = 0.05 sec
[ Info: VUMPS  40:	obj = -8.862878980877e-01	err = 3.2940004401e-08	time = 0.05 sec
[ Info: VUMPS  41:	obj = -8.862878980877e-01	err = 2.3832479903e-08	time = 0.09 sec
[ Info: VUMPS  42:	obj = -8.862878980877e-01	err = 1.7241026505e-08	time = 0.05 sec
[ Info: VUMPS  43:	obj = -8.862878980877e-01	err = 1.2471412997e-08	time = 0.05 sec
[ Info: VUMPS  44:	obj = -8.862878980877e-01	err = 9.0205419450e-09	time = 0.05 sec
[ Info: VUMPS  45:	obj = -8.862878980877e-01	err = 6.5240508358e-09	time = 0.05 sec
[ Info: VUMPS  46:	obj = -8.862878980877e-01	err = 4.7181601548e-09	time = 0.05 sec
[ Info: VUMPS  47:	obj = -8.862878980877e-01	err = 3.4119389042e-09	time = 0.05 sec
[ Info: VUMPS  48:	obj = -8.862878980877e-01	err = 2.4672054622e-09	time = 0.05 sec
[ Info: VUMPS  49:	obj = -8.862878980877e-01	err = 1.7839684924e-09	time = 0.05 sec
[ Info: VUMPS  50:	obj = -8.862878980877e-01	err = 1.2898741454e-09	time = 0.09 sec
[ Info: VUMPS  51:	obj = -8.862878980877e-01	err = 9.3258970310e-10	time = 0.05 sec
[ Info: VUMPS  52:	obj = -8.862878980877e-01	err = 6.7423602771e-10	time = 0.04 sec
[ Info: VUMPS  53:	obj = -8.862878980877e-01	err = 4.8743673761e-10	time = 0.04 sec
[ Info: VUMPS  54:	obj = -8.862878980877e-01	err = 3.5238100286e-10	time = 0.04 sec
[ Info: VUMPS  55:	obj = -8.862878980877e-01	err = 2.5473951258e-10	time = 0.05 sec
[ Info: VUMPS  56:	obj = -8.862878980878e-01	err = 1.8414828185e-10	time = 0.05 sec
[ Info: VUMPS  57:	obj = -8.862878980878e-01	err = 1.3311459508e-10	time = 0.05 sec
[ Info: VUMPS  58:	obj = -8.862878980878e-01	err = 9.6221597943e-11	time = 0.05 sec
[ Info: VUMPS  59:	obj = -8.862878980878e-01	err = 6.9553567834e-11	time = 0.09 sec
[ Info: VUMPS  60:	obj = -8.862878980878e-01	err = 5.0274813406e-11	time = 0.05 sec
[ Info: VUMPS  61:	obj = -8.862878980878e-01	err = 3.6335883698e-11	time = 0.05 sec
[ Info: VUMPS  62:	obj = -8.862878980878e-01	err = 2.6254468114e-11	time = 0.04 sec
[ Info: VUMPS  63:	obj = -8.862878980878e-01	err = 1.8980171184e-11	time = 0.04 sec
[ Info: VUMPS  64:	obj = -8.862878980878e-01	err = 1.3721774296e-11	time = 0.04 sec
[ Info: VUMPS  65:	obj = -8.862878980878e-01	err = 9.9185056725e-12	time = 0.04 sec
[ Info: VUMPS  66:	obj = -8.862878980878e-01	err = 7.1729564828e-12	time = 0.04 sec
[ Info: VUMPS  67:	obj = -8.862878980878e-01	err = 5.1845841611e-12	time = 0.04 sec
[ Info: VUMPS  68:	obj = -8.862878980878e-01	err = 3.7451968982e-12	time = 0.04 sec
[ Info: VUMPS  69:	obj = -8.862878980878e-01	err = 2.7085301699e-12	time = 0.08 sec
[ Info: VUMPS  70:	obj = -8.862878980878e-01	err = 1.9547335304e-12	time = 0.04 sec
[ Info: VUMPS  71:	obj = -8.862878980878e-01	err = 1.4119541453e-12	time = 0.04 sec
[ Info: VUMPS  72:	obj = -8.862878980878e-01	err = 1.0222255848e-12	time = 0.03 sec
[ Info: VUMPS conv 73:	obj = -8.862878980878e-01	err = 7.3834786497e-13	time = 3.71 sec

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

