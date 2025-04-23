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
H = heisenberg_XXX(; spin=1 // 2)
````

````
single site InfiniteMPOHamiltonian{MPSKit.JordanMPOTensor{ComplexF64, TensorKit.ComplexSpace, Union{TensorKit.BraidingTensor{ComplexF64, TensorKit.ComplexSpace}, TensorKit.TensorMap{ComplexF64, TensorKit.ComplexSpace, 2, 2, Vector{ComplexF64}}}, TensorKit.TensorMap{ComplexF64, TensorKit.ComplexSpace, 2, 1, Vector{ComplexF64}}, TensorKit.TensorMap{ComplexF64, TensorKit.ComplexSpace, 1, 2, Vector{ComplexF64}}, TensorKit.TensorMap{ComplexF64, TensorKit.ComplexSpace, 1, 1, Vector{ComplexF64}}}}:
╷  ⋮
┼ W[1]: 3×1×1×3 JordanMPOTensor(((ℂ^1 ⊕ ℂ^3 ⊕ ℂ^1) ⊗ ⊕(ℂ^2)) ← (⊕(ℂ^2) ⊗ (ℂ^1 ⊕ ℂ^3 ⊕ ℂ^1)))
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

The groundstate can then be found by calling `find_groundstate`.

````julia
groundstate, cache, delta = find_groundstate(state, H, VUMPS());
````

````
[ Info: VUMPS init:	obj = +2.499997151608e-01	err = 1.5076e-03
[ Info: VUMPS   1:	obj = -1.102936647693e-01	err = 3.8393870916e-01	time = 0.02 sec
[ Info: VUMPS   2:	obj = -2.518819686637e-03	err = 3.8247870042e-01	time = 0.02 sec
[ Info: VUMPS   3:	obj = -1.283698632877e-01	err = 4.5231528318e-01	time = 0.02 sec
[ Info: VUMPS   4:	obj = -1.318247939270e-01	err = 3.9194798602e-01	time = 0.02 sec
[ Info: VUMPS   5:	obj = -1.563893197526e-01	err = 3.6030425585e-01	time = 0.02 sec
[ Info: VUMPS   6:	obj = -1.275381556576e-01	err = 3.7241063143e-01	time = 0.09 sec
[ Info: VUMPS   7:	obj = -2.001728737220e-01	err = 3.5694925283e-01	time = 0.09 sec
[ Info: VUMPS   8:	obj = -4.148514795033e-01	err = 1.9978091568e-01	time = 0.02 sec
[ Info: VUMPS   9:	obj = -8.948544839910e-02	err = 3.8051752551e-01	time = 0.02 sec
[ Info: VUMPS  10:	obj = +3.907102087904e-03	err = 4.0778106525e-01	time = 0.02 sec
[ Info: VUMPS  11:	obj = -4.210510632748e-02	err = 3.9552473178e-01	time = 0.02 sec
[ Info: VUMPS  12:	obj = -1.143304888714e-01	err = 3.8847677538e-01	time = 0.02 sec
[ Info: VUMPS  13:	obj = -2.684102122959e-01	err = 3.5839329032e-01	time = 0.02 sec
[ Info: VUMPS  14:	obj = -3.140375987930e-01	err = 3.5220148478e-01	time = 0.07 sec
[ Info: VUMPS  15:	obj = -3.494719807888e-01	err = 3.0216614383e-01	time = 0.03 sec
[ Info: VUMPS  16:	obj = -2.600823546903e-01	err = 3.4812671114e-01	time = 0.02 sec
[ Info: VUMPS  17:	obj = -2.378681434544e-01	err = 3.7837254224e-01	time = 0.02 sec
[ Info: VUMPS  18:	obj = -1.165503147031e-01	err = 3.9507067038e-01	time = 0.02 sec
[ Info: VUMPS  19:	obj = -3.371493327212e-01	err = 3.2991102430e-01	time = 0.02 sec
[ Info: VUMPS  20:	obj = -2.694806272840e-01	err = 3.6205665137e-01	time = 0.05 sec
[ Info: VUMPS  21:	obj = -2.381146966165e-01	err = 3.6628398837e-01	time = 0.03 sec
[ Info: VUMPS  22:	obj = -3.533056802218e-01	err = 3.0627687902e-01	time = 0.02 sec
[ Info: VUMPS  23:	obj = -3.323562393144e-01	err = 3.2310539990e-01	time = 0.03 sec
[ Info: VUMPS  24:	obj = -2.930878664518e-01	err = 3.5261734042e-01	time = 0.03 sec
[ Info: VUMPS  25:	obj = -3.195260603801e-01	err = 3.4213237114e-01	time = 0.02 sec
[ Info: VUMPS  26:	obj = -3.009445857195e-01	err = 3.6267895477e-01	time = 0.05 sec
[ Info: VUMPS  27:	obj = -1.945482035700e-01	err = 3.8036687114e-01	time = 0.02 sec
[ Info: VUMPS  28:	obj = -3.129608668588e-01	err = 3.3946739811e-01	time = 0.02 sec
[ Info: VUMPS  29:	obj = -2.685329650730e-01	err = 3.6719306482e-01	time = 0.02 sec
[ Info: VUMPS  30:	obj = -1.255622850827e-01	err = 3.9830069704e-01	time = 0.02 sec
[ Info: VUMPS  31:	obj = -2.259105658052e-01	err = 3.5495588739e-01	time = 0.02 sec
[ Info: VUMPS  32:	obj = -2.961189169588e-01	err = 3.4982483151e-01	time = 0.02 sec
[ Info: VUMPS  33:	obj = -3.982754765049e-01	err = 2.4610327930e-01	time = 0.05 sec
[ Info: VUMPS  34:	obj = -3.868968141119e-01	err = 2.6461805850e-01	time = 0.08 sec
[ Info: VUMPS  35:	obj = -1.455777740261e-01	err = 4.0452794718e-01	time = 0.02 sec
[ Info: VUMPS  36:	obj = -2.384731273529e-01	err = 3.7152623297e-01	time = 0.02 sec
[ Info: VUMPS  37:	obj = -2.655148561526e-01	err = 3.5351438817e-01	time = 0.02 sec
[ Info: VUMPS  38:	obj = -1.824684314123e-01	err = 3.8362950466e-01	time = 0.02 sec
[ Info: VUMPS  39:	obj = -4.340710641979e-02	err = 3.6536524706e-01	time = 0.06 sec
[ Info: VUMPS  40:	obj = -7.432642869747e-02	err = 4.1533450888e-01	time = 0.02 sec
[ Info: VUMPS  41:	obj = -1.277746081261e-01	err = 4.0325440215e-01	time = 0.02 sec
[ Info: VUMPS  42:	obj = -1.637433441229e-01	err = 3.9644561915e-01	time = 0.02 sec
[ Info: VUMPS  43:	obj = -2.494122089360e-02	err = 3.9811228669e-01	time = 0.01 sec
[ Info: VUMPS  44:	obj = -5.437621624074e-02	err = 4.1086160744e-01	time = 0.02 sec
[ Info: VUMPS  45:	obj = -9.499156814866e-02	err = 3.9556960135e-01	time = 0.02 sec
[ Info: VUMPS  46:	obj = -2.664160174156e-01	err = 3.4236577100e-01	time = 0.06 sec
[ Info: VUMPS  47:	obj = -1.712858760526e-01	err = 3.9929081714e-01	time = 0.02 sec
[ Info: VUMPS  48:	obj = -2.278081372696e-01	err = 3.6618881136e-01	time = 0.02 sec
[ Info: VUMPS  49:	obj = -3.229675684963e-01	err = 3.1862991140e-01	time = 0.02 sec
[ Info: VUMPS  50:	obj = -2.110632456326e-01	err = 3.8859303819e-01	time = 0.02 sec
[ Info: VUMPS  51:	obj = -1.101060326910e-01	err = 3.3901651620e-01	time = 0.03 sec
[ Info: VUMPS  52:	obj = +7.333338049546e-02	err = 3.6898787075e-01	time = 0.04 sec
[ Info: VUMPS  53:	obj = -2.159728827455e-01	err = 3.4788128440e-01	time = 0.02 sec
[ Info: VUMPS  54:	obj = -3.781744951044e-01	err = 2.8734162232e-01	time = 0.02 sec
[ Info: VUMPS  55:	obj = -2.498878569904e-01	err = 3.7437122967e-01	time = 0.02 sec
[ Info: VUMPS  56:	obj = -9.182753559612e-02	err = 4.1119786611e-01	time = 0.02 sec
[ Info: VUMPS  57:	obj = -9.896252196567e-02	err = 3.9430375805e-01	time = 0.01 sec
[ Info: VUMPS  58:	obj = +9.467977464259e-02	err = 3.8040243513e-01	time = 0.02 sec
[ Info: VUMPS  59:	obj = -4.749770024744e-02	err = 3.6638080888e-01	time = 0.05 sec
[ Info: VUMPS  60:	obj = -2.530337073792e-01	err = 3.5986314085e-01	time = 0.01 sec
[ Info: VUMPS  61:	obj = -1.569761362938e-01	err = 3.8385844092e-01	time = 0.02 sec
[ Info: VUMPS  62:	obj = -2.259009259309e-01	err = 3.7057427131e-01	time = 0.02 sec
[ Info: VUMPS  63:	obj = +6.333173021014e-02	err = 4.1117754784e-01	time = 0.02 sec
[ Info: VUMPS  64:	obj = -1.342383442709e-01	err = 3.6454520924e-01	time = 0.01 sec
[ Info: VUMPS  65:	obj = -2.731276613159e-01	err = 3.5374290954e-01	time = 0.02 sec
[ Info: VUMPS  66:	obj = -1.467205296305e-02	err = 3.7407661717e-01	time = 0.05 sec
[ Info: VUMPS  67:	obj = -7.117743661345e-02	err = 3.5898879533e-01	time = 0.02 sec
[ Info: VUMPS  68:	obj = -2.109524282916e-01	err = 3.7190622381e-01	time = 0.02 sec
[ Info: VUMPS  69:	obj = -2.429823175649e-01	err = 3.5944789711e-01	time = 0.02 sec
[ Info: VUMPS  70:	obj = -2.401276795251e-01	err = 3.5294125102e-01	time = 0.02 sec
[ Info: VUMPS  71:	obj = -3.770033736284e-01	err = 2.7476469956e-01	time = 0.02 sec
[ Info: VUMPS  72:	obj = +6.405996956936e-02	err = 3.5202436851e-01	time = 0.05 sec
[ Info: VUMPS  73:	obj = -2.333733284165e-01	err = 3.6882340641e-01	time = 0.02 sec
[ Info: VUMPS  74:	obj = -8.840535012858e-02	err = 3.8308266948e-01	time = 0.02 sec
[ Info: VUMPS  75:	obj = -3.397593157665e-01	err = 3.2422050821e-01	time = 0.02 sec
[ Info: VUMPS  76:	obj = -1.577057931895e-01	err = 3.7953340218e-01	time = 0.02 sec
[ Info: VUMPS  77:	obj = -1.283119074775e-01	err = 3.8701219784e-01	time = 0.02 sec
[ Info: VUMPS  78:	obj = -2.838119695877e-01	err = 3.5153675863e-01	time = 0.02 sec
[ Info: VUMPS  79:	obj = -3.773014424183e-01	err = 2.8096984820e-01	time = 0.05 sec
[ Info: VUMPS  80:	obj = -4.248680541774e-01	err = 1.6995008124e-01	time = 0.02 sec
[ Info: VUMPS  81:	obj = -3.898082089496e-01	err = 2.6625454039e-01	time = 0.03 sec
[ Info: VUMPS  82:	obj = -9.519451168265e-03	err = 4.2221971320e-01	time = 0.02 sec
[ Info: VUMPS  83:	obj = -5.231292682682e-02	err = 3.6959419670e-01	time = 0.02 sec
[ Info: VUMPS  84:	obj = -1.866826486661e-01	err = 3.8258698851e-01	time = 0.01 sec
[ Info: VUMPS  85:	obj = -1.692405363751e-01	err = 3.8770899838e-01	time = 0.05 sec
[ Info: VUMPS  86:	obj = -2.083324644893e-01	err = 3.5431521899e-01	time = 0.02 sec
[ Info: VUMPS  87:	obj = -2.939979120876e-01	err = 3.5532871233e-01	time = 0.02 sec
[ Info: VUMPS  88:	obj = -2.828666406513e-01	err = 3.4873331725e-01	time = 0.02 sec
[ Info: VUMPS  89:	obj = -1.692982578604e-01	err = 3.8168691441e-01	time = 0.02 sec
[ Info: VUMPS  90:	obj = -2.363440425785e-01	err = 3.4723691699e-01	time = 0.02 sec
[ Info: VUMPS  91:	obj = -1.414396078859e-01	err = 4.0303952514e-01	time = 0.05 sec
[ Info: VUMPS  92:	obj = -9.111717266883e-02	err = 3.9613002109e-01	time = 0.02 sec
[ Info: VUMPS  93:	obj = +3.874548352332e-02	err = 3.6730004996e-01	time = 0.02 sec
[ Info: VUMPS  94:	obj = -7.194785728285e-02	err = 3.9110410465e-01	time = 0.02 sec
[ Info: VUMPS  95:	obj = -1.043822044301e-01	err = 3.9084477906e-01	time = 0.02 sec
[ Info: VUMPS  96:	obj = -1.029644744149e-01	err = 4.1397652279e-01	time = 0.02 sec
[ Info: VUMPS  97:	obj = -2.861705477721e-01	err = 3.5209989794e-01	time = 0.01 sec
[ Info: VUMPS  98:	obj = -3.470187676038e-01	err = 3.0920621601e-01	time = 0.05 sec
[ Info: VUMPS  99:	obj = -5.822199548527e-02	err = 3.5959218485e-01	time = 0.02 sec
[ Info: VUMPS 100:	obj = -1.847189630830e-01	err = 3.6268585762e-01	time = 0.02 sec
[ Info: VUMPS 101:	obj = -1.998247793394e-01	err = 3.7026241762e-01	time = 0.02 sec
[ Info: VUMPS 102:	obj = -2.845861059390e-01	err = 3.5015115387e-01	time = 0.02 sec
[ Info: VUMPS 103:	obj = +1.001145171552e-01	err = 3.7897389825e-01	time = 0.02 sec
[ Info: VUMPS 104:	obj = +5.388940615771e-03	err = 3.8837861849e-01	time = 0.02 sec
[ Info: VUMPS 105:	obj = +6.174310736557e-02	err = 3.6429568298e-01	time = 0.05 sec
[ Info: VUMPS 106:	obj = -1.953458747936e-02	err = 4.1105781305e-01	time = 0.02 sec
[ Info: VUMPS 107:	obj = -1.227404516478e-01	err = 3.9527799477e-01	time = 0.02 sec
[ Info: VUMPS 108:	obj = -2.249890803810e-01	err = 3.7797056287e-01	time = 0.02 sec
[ Info: VUMPS 109:	obj = -6.728916323045e-02	err = 3.6671401448e-01	time = 0.03 sec
[ Info: VUMPS 110:	obj = -1.503938692777e-01	err = 3.7432233900e-01	time = 0.02 sec
[ Info: VUMPS 111:	obj = -5.369841176091e-02	err = 4.2078297478e-01	time = 0.02 sec
[ Info: VUMPS 112:	obj = +3.503395811604e-02	err = 4.0241955891e-01	time = 0.05 sec
[ Info: VUMPS 113:	obj = -1.641757316788e-01	err = 3.8361173302e-01	time = 0.01 sec
[ Info: VUMPS 114:	obj = -2.285888795528e-01	err = 3.6399759156e-01	time = 0.02 sec
[ Info: VUMPS 115:	obj = -3.184239125421e-01	err = 3.5620987147e-01	time = 0.03 sec
[ Info: VUMPS 116:	obj = -4.004343201842e-01	err = 2.3427592725e-01	time = 0.02 sec
[ Info: VUMPS 117:	obj = +1.250815154369e-01	err = 3.8171291086e-01	time = 0.02 sec
[ Info: VUMPS 118:	obj = -8.382309493919e-02	err = 3.9885688763e-01	time = 0.05 sec
[ Info: VUMPS 119:	obj = -1.480032425391e-01	err = 3.7961317585e-01	time = 0.01 sec
[ Info: VUMPS 120:	obj = -2.603417820356e-01	err = 3.5989698304e-01	time = 0.02 sec
[ Info: VUMPS 121:	obj = -3.231635462838e-01	err = 3.3803029927e-01	time = 0.02 sec
[ Info: VUMPS 122:	obj = -3.960892923915e-01	err = 2.5894024392e-01	time = 0.02 sec
[ Info: VUMPS 123:	obj = +1.383960399671e-01	err = 3.2560793126e-01	time = 0.02 sec
[ Info: VUMPS 124:	obj = +2.266107848362e-03	err = 4.1080709820e-01	time = 0.02 sec
[ Info: VUMPS 125:	obj = -2.283058118711e-01	err = 3.7356440605e-01	time = 0.05 sec
[ Info: VUMPS 126:	obj = -3.418107299565e-01	err = 3.1899399020e-01	time = 0.02 sec
[ Info: VUMPS 127:	obj = -2.802812290403e-01	err = 3.5828911261e-01	time = 0.02 sec
[ Info: VUMPS 128:	obj = -1.362964889519e-01	err = 3.9593008281e-01	time = 0.02 sec
[ Info: VUMPS 129:	obj = -6.715310983028e-02	err = 3.8272455592e-01	time = 0.02 sec
[ Info: VUMPS 130:	obj = -2.592620383388e-01	err = 3.5195567454e-01	time = 0.02 sec
[ Info: VUMPS 131:	obj = -4.252097482878e-01	err = 1.6873141588e-01	time = 0.05 sec
[ Info: VUMPS 132:	obj = +3.863457166648e-02	err = 3.2307300405e-01	time = 0.02 sec
[ Info: VUMPS 133:	obj = +2.300685431813e-03	err = 3.8827309741e-01	time = 0.02 sec
[ Info: VUMPS 134:	obj = -1.800971988749e-01	err = 3.9325052328e-01	time = 0.02 sec
[ Info: VUMPS 135:	obj = +2.655829501197e-03	err = 3.8180278781e-01	time = 0.02 sec
[ Info: VUMPS 136:	obj = -1.960012423358e-01	err = 3.7736036559e-01	time = 0.02 sec
[ Info: VUMPS 137:	obj = -3.945976063716e-01	err = 2.5776349595e-01	time = 0.05 sec
[ Info: VUMPS 138:	obj = +9.835127345556e-02	err = 3.9127467384e-01	time = 0.02 sec
[ Info: VUMPS 139:	obj = -1.960607007481e-01	err = 3.9818909217e-01	time = 0.02 sec
[ Info: VUMPS 140:	obj = -2.076438228078e-01	err = 3.6858724855e-01	time = 0.01 sec
[ Info: VUMPS 141:	obj = -2.752417182485e-01	err = 3.5409227604e-01	time = 0.02 sec
[ Info: VUMPS 142:	obj = -2.323570171993e-01	err = 3.8479850876e-01	time = 0.03 sec
[ Info: VUMPS 143:	obj = -2.478431862183e-01	err = 3.6601199783e-01	time = 0.02 sec
[ Info: VUMPS 144:	obj = -1.303161187004e-01	err = 3.9306929113e-01	time = 0.05 sec
[ Info: VUMPS 145:	obj = -1.753964431937e-01	err = 3.8947518565e-01	time = 0.02 sec
[ Info: VUMPS 146:	obj = +4.823037317739e-02	err = 3.7010938139e-01	time = 0.02 sec
[ Info: VUMPS 147:	obj = -1.720633754678e-01	err = 3.8439582822e-01	time = 0.02 sec
[ Info: VUMPS 148:	obj = -1.497805800054e-01	err = 3.8708211808e-01	time = 0.02 sec
[ Info: VUMPS 149:	obj = -1.574371937148e-01	err = 3.9621587227e-01	time = 0.02 sec
[ Info: VUMPS 150:	obj = -4.188596075056e-02	err = 4.0489025033e-01	time = 0.05 sec
[ Info: VUMPS 151:	obj = -2.931751248451e-01	err = 3.3701305583e-01	time = 0.02 sec
[ Info: VUMPS 152:	obj = -3.542385510115e-01	err = 3.0922083054e-01	time = 0.02 sec
[ Info: VUMPS 153:	obj = -3.153636373307e-01	err = 3.2780719969e-01	time = 0.02 sec
[ Info: VUMPS 154:	obj = -2.099485771080e-01	err = 3.7583374896e-01	time = 0.03 sec
[ Info: VUMPS 155:	obj = -2.325036131152e-01	err = 3.6785157612e-01	time = 0.02 sec
[ Info: VUMPS 156:	obj = -3.306527885138e-01	err = 3.1451420802e-01	time = 0.05 sec
[ Info: VUMPS 157:	obj = -3.683042109253e-01	err = 3.0098744465e-01	time = 0.03 sec
[ Info: VUMPS 158:	obj = -2.899970230712e-01	err = 3.4912426816e-01	time = 0.03 sec
[ Info: VUMPS 159:	obj = -1.548982258892e-01	err = 3.9634701519e-01	time = 0.02 sec
[ Info: VUMPS 160:	obj = -1.807853460354e-01	err = 4.0347363337e-01	time = 0.02 sec
[ Info: VUMPS 161:	obj = -9.508228951537e-02	err = 3.7574701420e-01	time = 0.02 sec
[ Info: VUMPS 162:	obj = -1.993563335295e-01	err = 3.5850713945e-01	time = 0.05 sec
[ Info: VUMPS 163:	obj = +9.436575352874e-02	err = 3.6482840550e-01	time = 0.02 sec
[ Info: VUMPS 164:	obj = +7.903937847467e-02	err = 3.8855241138e-01	time = 0.02 sec
[ Info: VUMPS 165:	obj = -5.495752329074e-02	err = 3.8048454765e-01	time = 0.01 sec
[ Info: VUMPS 166:	obj = -2.977368905239e-01	err = 3.5841635043e-01	time = 0.02 sec
[ Info: VUMPS 167:	obj = -2.285161464463e-01	err = 3.6155361498e-01	time = 0.02 sec
[ Info: VUMPS 168:	obj = -2.969068468462e-01	err = 3.3763604978e-01	time = 0.02 sec
[ Info: VUMPS 169:	obj = -3.372311890964e-01	err = 3.1700106891e-01	time = 0.05 sec
[ Info: VUMPS 170:	obj = -3.045611491085e-01	err = 3.0628902149e-01	time = 0.03 sec
[ Info: VUMPS 171:	obj = -3.017393279199e-01	err = 3.6408415132e-01	time = 0.02 sec
[ Info: VUMPS 172:	obj = -1.249057059582e-01	err = 3.8003500781e-01	time = 0.02 sec
[ Info: VUMPS 173:	obj = -1.804070758425e-01	err = 3.8048914783e-01	time = 0.02 sec
[ Info: VUMPS 174:	obj = -1.587740528232e-01	err = 3.8521882028e-01	time = 0.02 sec
[ Info: VUMPS 175:	obj = -2.851791605376e-01	err = 3.3829945088e-01	time = 0.05 sec
[ Info: VUMPS 176:	obj = -6.876942688524e-02	err = 4.0447417286e-01	time = 0.02 sec
[ Info: VUMPS 177:	obj = +4.088424378843e-02	err = 3.9727430522e-01	time = 0.02 sec
[ Info: VUMPS 178:	obj = +4.349001316215e-02	err = 3.6572321243e-01	time = 0.02 sec
[ Info: VUMPS 179:	obj = -2.273704295370e-01	err = 3.8731200334e-01	time = 0.02 sec
[ Info: VUMPS 180:	obj = -2.616628489407e-01	err = 3.4696217488e-01	time = 0.02 sec
[ Info: VUMPS 181:	obj = -2.694623492030e-01	err = 3.4874627834e-01	time = 0.02 sec
[ Info: VUMPS 182:	obj = -2.783882521737e-01	err = 3.5076502054e-01	time = 0.05 sec
[ Info: VUMPS 183:	obj = -2.136144754439e-01	err = 3.8565666056e-01	time = 0.02 sec
[ Info: VUMPS 184:	obj = -2.172901485715e-01	err = 3.8072126413e-01	time = 0.02 sec
[ Info: VUMPS 185:	obj = +8.912737400891e-02	err = 3.5445943853e-01	time = 0.03 sec
[ Info: VUMPS 186:	obj = -4.405574529414e-02	err = 4.0903072415e-01	time = 0.02 sec
[ Info: VUMPS 187:	obj = -1.606066852805e-01	err = 3.8640765878e-01	time = 0.02 sec
[ Info: VUMPS 188:	obj = +2.554966161756e-02	err = 3.7067877383e-01	time = 0.05 sec
[ Info: VUMPS 189:	obj = -1.520562341146e-01	err = 3.7246997774e-01	time = 0.01 sec
[ Info: VUMPS 190:	obj = -5.455120357931e-02	err = 3.7880083957e-01	time = 0.02 sec
[ Info: VUMPS 191:	obj = -1.821662375375e-01	err = 3.9326693657e-01	time = 0.02 sec
[ Info: VUMPS 192:	obj = -1.577721407955e-01	err = 4.1429770418e-01	time = 0.03 sec
[ Info: VUMPS 193:	obj = +2.586895357233e-02	err = 3.5544800306e-01	time = 0.02 sec
[ Info: VUMPS 194:	obj = -2.583240143294e-01	err = 3.7987694829e-01	time = 0.02 sec
[ Info: VUMPS 195:	obj = -2.224656299792e-01	err = 3.6394963764e-01	time = 0.05 sec
[ Info: VUMPS 196:	obj = -1.254311298138e-01	err = 3.7548601279e-01	time = 0.02 sec
[ Info: VUMPS 197:	obj = -1.723763414272e-01	err = 3.5494740522e-01	time = 0.02 sec
[ Info: VUMPS 198:	obj = -2.762110763835e-01	err = 3.4218268756e-01	time = 0.02 sec
[ Info: VUMPS 199:	obj = -3.239089102881e-01	err = 3.2528270621e-01	time = 0.02 sec
┌ Warning: VUMPS cancel 200:	obj = -2.870558938604e-01	err = 3.3497651674e-01	time = 5.07 sec
└ @ MPSKit ~/Projects/MPSKit.jl/src/algorithms/groundstate/vumps.jl:73

````

As you can see, VUMPS struggles to converge.
On it's own, that is already quite curious.
Maybe we can do better using another algorithm, such as gradient descent.

````julia
groundstate, cache, delta = find_groundstate(state, H, GradientGrassmann(; maxiter=20));
````

````
[ Info: CG: initializing with f = 0.249999715161, ‖∇f‖ = 1.0660e-03
┌ Warning: CG: not converged to requested tol after 20 iterations and time 4.69 s: f = -0.442407384926, ‖∇f‖ = 8.5132e-03
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/cg.jl:172

````

Convergence is quite slow and even fails after sufficiently many iterations.
To understand why, we can look at the transfer matrix spectrum.

````julia
transferplot(groundstate, groundstate)
````

```@raw html
<?xml version="1.0" encoding="utf-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="600" height="400" viewBox="0 0 2400 1600">
<defs>
  <clipPath id="clip950">
    <rect x="0" y="0" width="2400" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip950)" d="M0 1600 L2400 1600 L2400 0 L0 0  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip951">
    <rect x="480" y="0" width="1681" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip950)" d="M102.74 1505.26 L2352.76 1505.26 L2352.76 47.2441 L102.74 47.2441  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip952">
    <rect x="102" y="47" width="2251" height="1459"/>
  </clipPath>
</defs>
<polyline clip-path="url(#clip952)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="166.42,1505.26 166.42,47.2441 "/>
<polyline clip-path="url(#clip952)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="520.196,1505.26 520.196,47.2441 "/>
<polyline clip-path="url(#clip952)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="873.972,1505.26 873.972,47.2441 "/>
<polyline clip-path="url(#clip952)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1227.75,1505.26 1227.75,47.2441 "/>
<polyline clip-path="url(#clip952)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1581.52,1505.26 1581.52,47.2441 "/>
<polyline clip-path="url(#clip952)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1935.3,1505.26 1935.3,47.2441 "/>
<polyline clip-path="url(#clip952)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="2289.08,1505.26 2289.08,47.2441 "/>
<polyline clip-path="url(#clip952)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="102.74,1262.72 2352.76,1262.72 "/>
<polyline clip-path="url(#clip952)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="102.74,1019.63 2352.76,1019.63 "/>
<polyline clip-path="url(#clip952)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="102.74,776.531 2352.76,776.531 "/>
<polyline clip-path="url(#clip952)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="102.74,533.436 2352.76,533.436 "/>
<polyline clip-path="url(#clip952)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="102.74,290.34 2352.76,290.34 "/>
<polyline clip-path="url(#clip950)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="102.74,1505.26 2352.76,1505.26 "/>
<polyline clip-path="url(#clip950)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="166.42,1505.26 166.42,1486.36 "/>
<polyline clip-path="url(#clip950)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="520.196,1505.26 520.196,1486.36 "/>
<polyline clip-path="url(#clip950)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="873.972,1505.26 873.972,1486.36 "/>
<polyline clip-path="url(#clip950)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1227.75,1505.26 1227.75,1486.36 "/>
<polyline clip-path="url(#clip950)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1581.52,1505.26 1581.52,1486.36 "/>
<polyline clip-path="url(#clip950)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1935.3,1505.26 1935.3,1486.36 "/>
<polyline clip-path="url(#clip950)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="2289.08,1505.26 2289.08,1486.36 "/>
<path clip-path="url(#clip950)" d="M120.182 1536.18 Q116.571 1536.18 114.742 1539.74 Q112.936 1543.28 112.936 1550.41 Q112.936 1557.52 114.742 1561.09 Q116.571 1564.63 120.182 1564.63 Q123.816 1564.63 125.621 1561.09 Q127.45 1557.52 127.45 1550.41 Q127.45 1543.28 125.621 1539.74 Q123.816 1536.18 120.182 1536.18 M120.182 1532.47 Q125.992 1532.47 129.047 1537.08 Q132.126 1541.66 132.126 1550.41 Q132.126 1559.14 129.047 1563.75 Q125.992 1568.33 120.182 1568.33 Q114.372 1568.33 111.293 1563.75 Q108.237 1559.14 108.237 1550.41 Q108.237 1541.66 111.293 1537.08 Q114.372 1532.47 120.182 1532.47 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M147.311 1533.1 L151.246 1533.1 L139.209 1572.06 L135.274 1572.06 L147.311 1533.1 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M163.283 1533.1 L167.218 1533.1 L155.181 1572.06 L151.246 1572.06 L163.283 1533.1 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M173.098 1563.72 L180.737 1563.72 L180.737 1537.36 L172.427 1539.03 L172.427 1534.77 L180.691 1533.1 L185.367 1533.1 L185.367 1563.72 L193.005 1563.72 L193.005 1567.66 L173.098 1567.66 L173.098 1563.72 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M199.093 1541.73 L223.931 1541.73 L223.931 1545.99 L220.667 1545.99 L220.667 1561.92 Q220.667 1563.59 221.223 1564.33 Q221.801 1565.04 223.075 1565.04 Q223.422 1565.04 223.931 1565 Q224.44 1564.93 224.602 1564.91 L224.602 1567.98 Q223.792 1568.28 222.936 1568.42 Q222.079 1568.56 221.223 1568.56 Q218.445 1568.56 217.38 1567.06 Q216.315 1565.53 216.315 1561.46 L216.315 1545.99 L206.755 1545.99 L206.755 1567.66 L202.403 1567.66 L202.403 1545.99 L199.093 1545.99 L199.093 1541.73 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M463.726 1563.72 L471.365 1563.72 L471.365 1537.36 L463.055 1539.03 L463.055 1534.77 L471.319 1533.1 L475.995 1533.1 L475.995 1563.72 L483.634 1563.72 L483.634 1567.66 L463.726 1567.66 L463.726 1563.72 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M500.046 1533.1 L503.981 1533.1 L491.944 1572.06 L488.009 1572.06 L500.046 1533.1 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M516.018 1533.1 L519.953 1533.1 L507.916 1572.06 L503.981 1572.06 L516.018 1533.1 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M539.189 1549.03 Q542.545 1549.74 544.42 1552.01 Q546.318 1554.28 546.318 1557.61 Q546.318 1562.73 542.8 1565.53 Q539.281 1568.33 532.8 1568.33 Q530.624 1568.33 528.309 1567.89 Q526.018 1567.47 523.564 1566.62 L523.564 1562.1 Q525.508 1563.24 527.823 1563.82 Q530.138 1564.4 532.661 1564.4 Q537.059 1564.4 539.351 1562.66 Q541.666 1560.92 541.666 1557.61 Q541.666 1554.56 539.513 1552.85 Q537.383 1551.11 533.564 1551.11 L529.536 1551.11 L529.536 1547.27 L533.749 1547.27 Q537.198 1547.27 539.027 1545.9 Q540.856 1544.51 540.856 1541.92 Q540.856 1539.26 538.957 1537.85 Q537.082 1536.41 533.564 1536.41 Q531.643 1536.41 529.444 1536.83 Q527.244 1537.24 524.606 1538.12 L524.606 1533.96 Q527.268 1533.22 529.582 1532.85 Q531.92 1532.47 533.981 1532.47 Q539.305 1532.47 542.406 1534.91 Q545.508 1537.31 545.508 1541.43 Q545.508 1544.3 543.865 1546.29 Q542.221 1548.26 539.189 1549.03 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M551.828 1541.73 L576.666 1541.73 L576.666 1545.99 L573.402 1545.99 L573.402 1561.92 Q573.402 1563.59 573.957 1564.33 Q574.536 1565.04 575.809 1565.04 Q576.156 1565.04 576.666 1565 Q577.175 1564.93 577.337 1564.91 L577.337 1567.98 Q576.527 1568.28 575.67 1568.42 Q574.814 1568.56 573.957 1568.56 Q571.179 1568.56 570.115 1567.06 Q569.05 1565.53 569.05 1561.46 L569.05 1545.99 L559.49 1545.99 L559.49 1567.66 L555.138 1567.66 L555.138 1545.99 L551.828 1545.99 L551.828 1541.73 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M821.588 1563.72 L837.907 1563.72 L837.907 1567.66 L815.963 1567.66 L815.963 1563.72 Q818.625 1560.97 823.208 1556.34 Q827.815 1551.69 828.995 1550.35 Q831.241 1547.82 832.12 1546.09 Q833.023 1544.33 833.023 1542.64 Q833.023 1539.88 831.079 1538.15 Q829.157 1536.41 826.056 1536.41 Q823.856 1536.41 821.403 1537.17 Q818.972 1537.94 816.195 1539.49 L816.195 1534.77 Q819.019 1533.63 821.472 1533.05 Q823.926 1532.47 825.963 1532.47 Q831.333 1532.47 834.528 1535.16 Q837.722 1537.85 837.722 1542.34 Q837.722 1544.47 836.912 1546.39 Q836.125 1548.28 834.018 1550.88 Q833.44 1551.55 830.338 1554.77 Q827.236 1557.96 821.588 1563.72 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M854.69 1533.1 L858.625 1533.1 L846.588 1572.06 L842.653 1572.06 L854.69 1533.1 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M870.662 1533.1 L874.597 1533.1 L862.56 1572.06 L858.625 1572.06 L870.662 1533.1 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M893.833 1549.03 Q897.189 1549.74 899.064 1552.01 Q900.963 1554.28 900.963 1557.61 Q900.963 1562.73 897.444 1565.53 Q893.926 1568.33 887.444 1568.33 Q885.268 1568.33 882.953 1567.89 Q880.662 1567.47 878.208 1566.62 L878.208 1562.1 Q880.152 1563.24 882.467 1563.82 Q884.782 1564.4 887.305 1564.4 Q891.703 1564.4 893.995 1562.66 Q896.31 1560.92 896.31 1557.61 Q896.31 1554.56 894.157 1552.85 Q892.027 1551.11 888.208 1551.11 L884.18 1551.11 L884.18 1547.27 L888.393 1547.27 Q891.842 1547.27 893.671 1545.9 Q895.5 1544.51 895.5 1541.92 Q895.5 1539.26 893.601 1537.85 Q891.727 1536.41 888.208 1536.41 Q886.287 1536.41 884.088 1536.83 Q881.889 1537.24 879.25 1538.12 L879.25 1533.96 Q881.912 1533.22 884.227 1532.85 Q886.564 1532.47 888.625 1532.47 Q893.949 1532.47 897.051 1534.91 Q900.152 1537.31 900.152 1541.43 Q900.152 1544.3 898.509 1546.29 Q896.865 1548.26 893.833 1549.03 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M906.472 1541.73 L931.31 1541.73 L931.31 1545.99 L928.046 1545.99 L928.046 1561.92 Q928.046 1563.59 928.601 1564.33 Q929.18 1565.04 930.453 1565.04 Q930.8 1565.04 931.31 1565 Q931.819 1564.93 931.981 1564.91 L931.981 1567.98 Q931.171 1568.28 930.314 1568.42 Q929.458 1568.56 928.601 1568.56 Q925.824 1568.56 924.759 1567.06 Q923.694 1565.53 923.694 1561.46 L923.694 1545.99 L914.134 1545.99 L914.134 1567.66 L909.782 1567.66 L909.782 1545.99 L906.472 1545.99 L906.472 1541.73 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M1171.28 1563.72 L1178.92 1563.72 L1178.92 1537.36 L1170.61 1539.03 L1170.61 1534.77 L1178.87 1533.1 L1183.55 1533.1 L1183.55 1563.72 L1191.19 1563.72 L1191.19 1567.66 L1171.28 1567.66 L1171.28 1563.72 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M1207.6 1533.1 L1211.53 1533.1 L1199.5 1572.06 L1195.56 1572.06 L1207.6 1533.1 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M1223.57 1533.1 L1227.5 1533.1 L1215.47 1572.06 L1211.53 1572.06 L1223.57 1533.1 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M1233.38 1563.72 L1241.02 1563.72 L1241.02 1537.36 L1232.71 1539.03 L1232.71 1534.77 L1240.98 1533.1 L1245.65 1533.1 L1245.65 1563.72 L1253.29 1563.72 L1253.29 1567.66 L1233.38 1567.66 L1233.38 1563.72 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M1259.38 1541.73 L1284.22 1541.73 L1284.22 1545.99 L1280.95 1545.99 L1280.95 1561.92 Q1280.95 1563.59 1281.51 1564.33 Q1282.09 1565.04 1283.36 1565.04 Q1283.71 1565.04 1284.22 1565 Q1284.73 1564.93 1284.89 1564.91 L1284.89 1567.98 Q1284.08 1568.28 1283.22 1568.42 Q1282.37 1568.56 1281.51 1568.56 Q1278.73 1568.56 1277.67 1567.06 Q1276.6 1565.53 1276.6 1561.46 L1276.6 1545.99 L1267.04 1545.99 L1267.04 1567.66 L1262.69 1567.66 L1262.69 1545.99 L1259.38 1545.99 L1259.38 1541.73 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M1538.54 1537.17 L1526.73 1555.62 L1538.54 1555.62 L1538.54 1537.17 M1537.31 1533.1 L1543.19 1533.1 L1543.19 1555.62 L1548.12 1555.62 L1548.12 1559.51 L1543.19 1559.51 L1543.19 1567.66 L1538.54 1567.66 L1538.54 1559.51 L1522.94 1559.51 L1522.94 1555 L1537.31 1533.1 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M1562.82 1533.1 L1566.76 1533.1 L1554.72 1572.06 L1550.78 1572.06 L1562.82 1533.1 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M1578.79 1533.1 L1582.73 1533.1 L1570.69 1572.06 L1566.76 1572.06 L1578.79 1533.1 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M1601.96 1549.03 Q1605.32 1549.74 1607.2 1552.01 Q1609.09 1554.28 1609.09 1557.61 Q1609.09 1562.73 1605.57 1565.53 Q1602.06 1568.33 1595.57 1568.33 Q1593.4 1568.33 1591.08 1567.89 Q1588.79 1567.47 1586.34 1566.62 L1586.34 1562.1 Q1588.28 1563.24 1590.6 1563.82 Q1592.91 1564.4 1595.44 1564.4 Q1599.83 1564.4 1602.13 1562.66 Q1604.44 1560.92 1604.44 1557.61 Q1604.44 1554.56 1602.29 1552.85 Q1600.16 1551.11 1596.34 1551.11 L1592.31 1551.11 L1592.31 1547.27 L1596.52 1547.27 Q1599.97 1547.27 1601.8 1545.9 Q1603.63 1544.51 1603.63 1541.92 Q1603.63 1539.26 1601.73 1537.85 Q1599.86 1536.41 1596.34 1536.41 Q1594.42 1536.41 1592.22 1536.83 Q1590.02 1537.24 1587.38 1538.12 L1587.38 1533.96 Q1590.04 1533.22 1592.36 1532.85 Q1594.7 1532.47 1596.76 1532.47 Q1602.08 1532.47 1605.18 1534.91 Q1608.28 1537.31 1608.28 1541.43 Q1608.28 1544.3 1606.64 1546.29 Q1605 1548.26 1601.96 1549.03 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M1614.6 1541.73 L1639.44 1541.73 L1639.44 1545.99 L1636.18 1545.99 L1636.18 1561.92 Q1636.18 1563.59 1636.73 1564.33 Q1637.31 1565.04 1638.58 1565.04 Q1638.93 1565.04 1639.44 1565 Q1639.95 1564.93 1640.11 1564.91 L1640.11 1567.98 Q1639.3 1568.28 1638.45 1568.42 Q1637.59 1568.56 1636.73 1568.56 Q1633.95 1568.56 1632.89 1567.06 Q1631.82 1565.53 1631.82 1561.46 L1631.82 1545.99 L1622.26 1545.99 L1622.26 1567.66 L1617.91 1567.66 L1617.91 1545.99 L1614.6 1545.99 L1614.6 1541.73 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M1878.84 1533.1 L1897.2 1533.1 L1897.2 1537.04 L1883.12 1537.04 L1883.12 1545.51 Q1884.14 1545.16 1885.16 1545 Q1886.18 1544.81 1887.2 1544.81 Q1892.99 1544.81 1896.37 1547.98 Q1899.74 1551.16 1899.74 1556.57 Q1899.74 1562.15 1896.27 1565.25 Q1892.8 1568.33 1886.48 1568.33 Q1884.31 1568.33 1882.04 1567.96 Q1879.79 1567.59 1877.38 1566.85 L1877.38 1562.15 Q1879.47 1563.28 1881.69 1563.84 Q1883.91 1564.4 1886.39 1564.4 Q1890.39 1564.4 1892.73 1562.29 Q1895.07 1560.18 1895.07 1556.57 Q1895.07 1552.96 1892.73 1550.85 Q1890.39 1548.75 1886.39 1548.75 Q1884.51 1548.75 1882.64 1549.16 Q1880.79 1549.58 1878.84 1550.46 L1878.84 1533.1 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M1915.93 1533.1 L1919.86 1533.1 L1907.82 1572.06 L1903.89 1572.06 L1915.93 1533.1 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M1931.9 1533.1 L1935.83 1533.1 L1923.8 1572.06 L1919.86 1572.06 L1931.9 1533.1 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M1955.07 1549.03 Q1958.43 1549.74 1960.3 1552.01 Q1962.2 1554.28 1962.2 1557.61 Q1962.2 1562.73 1958.68 1565.53 Q1955.16 1568.33 1948.68 1568.33 Q1946.5 1568.33 1944.19 1567.89 Q1941.9 1567.47 1939.44 1566.62 L1939.44 1562.1 Q1941.39 1563.24 1943.7 1563.82 Q1946.02 1564.4 1948.54 1564.4 Q1952.94 1564.4 1955.23 1562.66 Q1957.55 1560.92 1957.55 1557.61 Q1957.55 1554.56 1955.39 1552.85 Q1953.26 1551.11 1949.44 1551.11 L1945.42 1551.11 L1945.42 1547.27 L1949.63 1547.27 Q1953.08 1547.27 1954.91 1545.9 Q1956.74 1544.51 1956.74 1541.92 Q1956.74 1539.26 1954.84 1537.85 Q1952.96 1536.41 1949.44 1536.41 Q1947.52 1536.41 1945.32 1536.83 Q1943.12 1537.24 1940.49 1538.12 L1940.49 1533.96 Q1943.15 1533.22 1945.46 1532.85 Q1947.8 1532.47 1949.86 1532.47 Q1955.18 1532.47 1958.29 1534.91 Q1961.39 1537.31 1961.39 1541.43 Q1961.39 1544.3 1959.74 1546.29 Q1958.1 1548.26 1955.07 1549.03 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M1967.71 1541.73 L1992.55 1541.73 L1992.55 1545.99 L1989.28 1545.99 L1989.28 1561.92 Q1989.28 1563.59 1989.84 1564.33 Q1990.42 1565.04 1991.69 1565.04 Q1992.04 1565.04 1992.55 1565 Q1993.05 1564.93 1993.22 1564.91 L1993.22 1567.98 Q1992.41 1568.28 1991.55 1568.42 Q1990.69 1568.56 1989.84 1568.56 Q1987.06 1568.56 1985.99 1567.06 Q1984.93 1565.53 1984.93 1561.46 L1984.93 1545.99 L1975.37 1545.99 L1975.37 1567.66 L1971.02 1567.66 L1971.02 1545.99 L1967.71 1545.99 L1967.71 1541.73 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M2236.69 1563.72 L2253.01 1563.72 L2253.01 1567.66 L2231.07 1567.66 L2231.07 1563.72 Q2233.73 1560.97 2238.31 1556.34 Q2242.92 1551.69 2244.1 1550.35 Q2246.34 1547.82 2247.22 1546.09 Q2248.13 1544.33 2248.13 1542.64 Q2248.13 1539.88 2246.18 1538.15 Q2244.26 1536.41 2241.16 1536.41 Q2238.96 1536.41 2236.51 1537.17 Q2234.08 1537.94 2231.3 1539.49 L2231.3 1534.77 Q2234.12 1533.63 2236.58 1533.05 Q2239.03 1532.47 2241.07 1532.47 Q2246.44 1532.47 2249.63 1535.16 Q2252.83 1537.85 2252.83 1542.34 Q2252.83 1544.47 2252.02 1546.39 Q2251.23 1548.28 2249.12 1550.88 Q2248.54 1551.55 2245.44 1554.77 Q2242.34 1557.96 2236.69 1563.72 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M2269.79 1533.1 L2273.73 1533.1 L2261.69 1572.06 L2257.76 1572.06 L2269.79 1533.1 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M2285.77 1533.1 L2289.7 1533.1 L2277.66 1572.06 L2273.73 1572.06 L2285.77 1533.1 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M2295.58 1563.72 L2303.22 1563.72 L2303.22 1537.36 L2294.91 1539.03 L2294.91 1534.77 L2303.17 1533.1 L2307.85 1533.1 L2307.85 1563.72 L2315.49 1563.72 L2315.49 1567.66 L2295.58 1567.66 L2295.58 1563.72 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M2321.58 1541.73 L2346.41 1541.73 L2346.41 1545.99 L2343.15 1545.99 L2343.15 1561.92 Q2343.15 1563.59 2343.71 1564.33 Q2344.28 1565.04 2345.56 1565.04 Q2345.9 1565.04 2346.41 1565 Q2346.92 1564.93 2347.09 1564.91 L2347.09 1567.98 Q2346.27 1568.28 2345.42 1568.42 Q2344.56 1568.56 2343.71 1568.56 Q2340.93 1568.56 2339.86 1567.06 Q2338.8 1565.53 2338.8 1561.46 L2338.8 1545.99 L2329.24 1545.99 L2329.24 1567.66 L2324.89 1567.66 L2324.89 1545.99 L2321.58 1545.99 L2321.58 1541.73 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M1237.87 1627.53 L1217.59 1627.53 Q1218.07 1637.04 1220.27 1641.08 Q1223.01 1646.05 1227.75 1646.05 Q1232.52 1646.05 1235.16 1641.05 Q1237.49 1636.66 1237.87 1627.53 M1237.77 1622.11 Q1236.88 1613.08 1235.16 1609.89 Q1232.43 1604.86 1227.75 1604.86 Q1222.88 1604.86 1220.3 1609.83 Q1218.26 1613.84 1217.66 1622.11 L1237.77 1622.11 M1227.75 1600.09 Q1235.39 1600.09 1239.75 1606.84 Q1244.11 1613.55 1244.11 1625.46 Q1244.11 1637.33 1239.75 1644.08 Q1235.39 1650.86 1227.75 1650.86 Q1220.08 1650.86 1215.75 1644.08 Q1211.39 1637.33 1211.39 1625.46 Q1211.39 1613.55 1215.75 1606.84 Q1220.08 1600.09 1227.75 1600.09 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><polyline clip-path="url(#clip950)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="102.74,1505.26 102.74,47.2441 "/>
<polyline clip-path="url(#clip950)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="102.74,1262.72 121.638,1262.72 "/>
<polyline clip-path="url(#clip950)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="102.74,1019.63 121.638,1019.63 "/>
<polyline clip-path="url(#clip950)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="102.74,776.531 121.638,776.531 "/>
<polyline clip-path="url(#clip950)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="102.74,533.436 121.638,533.436 "/>
<polyline clip-path="url(#clip950)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="102.74,290.34 121.638,290.34 "/>
<path clip-path="url(#clip950)" d="M-20.5973 1248.52 Q-24.2084 1248.52 -26.0371 1252.09 Q-27.8427 1255.63 -27.8427 1262.76 Q-27.8427 1269.86 -26.0371 1273.43 Q-24.2084 1276.97 -20.5973 1276.97 Q-16.9631 1276.97 -15.1576 1273.43 Q-13.3289 1269.86 -13.3289 1262.76 Q-13.3289 1255.63 -15.1576 1252.09 Q-16.9631 1248.52 -20.5973 1248.52 M-20.5973 1244.82 Q-14.7872 1244.82 -11.7316 1249.42 Q-8.65296 1254.01 -8.65296 1262.76 Q-8.65296 1271.48 -11.7316 1276.09 Q-14.7872 1280.67 -20.5973 1280.67 Q-26.4075 1280.67 -29.4862 1276.09 Q-32.5417 1271.48 -32.5417 1262.76 Q-32.5417 1254.01 -29.4862 1249.42 Q-26.4075 1244.82 -20.5973 1244.82 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M-0.435408 1274.12 L4.44882 1274.12 L4.44882 1280 L-0.435408 1280 L-0.435408 1274.12 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M24.6339 1263.59 Q21.3006 1263.59 19.3793 1265.37 Q17.4812 1267.16 17.4812 1270.28 Q17.4812 1273.41 19.3793 1275.19 Q21.3006 1276.97 24.6339 1276.97 Q27.9672 1276.97 29.8885 1275.19 Q31.8098 1273.38 31.8098 1270.28 Q31.8098 1267.16 29.8885 1265.37 Q27.9904 1263.59 24.6339 1263.59 M19.958 1261.6 Q16.9488 1260.86 15.259 1258.8 Q13.5923 1256.74 13.5923 1253.78 Q13.5923 1249.63 16.5321 1247.23 Q19.495 1244.82 24.6339 1244.82 Q29.7959 1244.82 32.7357 1247.23 Q35.6755 1249.63 35.6755 1253.78 Q35.6755 1256.74 33.9857 1258.8 Q32.319 1260.86 29.333 1261.6 Q32.7126 1262.39 34.5876 1264.68 Q36.4857 1266.97 36.4857 1270.28 Q36.4857 1275.3 33.407 1277.99 Q30.3515 1280.67 24.6339 1280.67 Q18.9163 1280.67 15.8377 1277.99 Q12.7821 1275.3 12.7821 1270.28 Q12.7821 1266.97 14.6803 1264.68 Q16.5784 1262.39 19.958 1261.6 M18.245 1254.22 Q18.245 1256.9 19.9117 1258.41 Q21.6015 1259.91 24.6339 1259.91 Q27.6431 1259.91 29.333 1258.41 Q31.0459 1256.9 31.0459 1254.22 Q31.0459 1251.53 29.333 1250.03 Q27.6431 1248.52 24.6339 1248.52 Q21.6015 1248.52 19.9117 1250.03 Q18.245 1251.53 18.245 1254.22 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M54.7958 1248.52 Q51.1847 1248.52 49.356 1252.09 Q47.5504 1255.63 47.5504 1262.76 Q47.5504 1269.86 49.356 1273.43 Q51.1847 1276.97 54.7958 1276.97 Q58.43 1276.97 60.2356 1273.43 Q62.0643 1269.86 62.0643 1262.76 Q62.0643 1255.63 60.2356 1252.09 Q58.43 1248.52 54.7958 1248.52 M54.7958 1244.82 Q60.6059 1244.82 63.6615 1249.42 Q66.7402 1254.01 66.7402 1262.76 Q66.7402 1271.48 63.6615 1276.09 Q60.6059 1280.67 54.7958 1280.67 Q48.9856 1280.67 45.9069 1276.09 Q42.8514 1271.48 42.8514 1262.76 Q42.8514 1254.01 45.9069 1249.42 Q48.9856 1244.82 54.7958 1244.82 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M-19.602 1005.43 Q-23.2131 1005.43 -25.0418 1008.99 Q-26.8473 1012.53 -26.8473 1019.66 Q-26.8473 1026.77 -25.0418 1030.33 Q-23.2131 1033.87 -19.602 1033.87 Q-15.9677 1033.87 -14.1622 1030.33 Q-12.3335 1026.77 -12.3335 1019.66 Q-12.3335 1012.53 -14.1622 1008.99 Q-15.9677 1005.43 -19.602 1005.43 M-19.602 1001.72 Q-13.7918 1001.72 -10.7363 1006.33 Q-7.65759 1010.91 -7.65759 1019.66 Q-7.65759 1028.39 -10.7363 1033 Q-13.7918 1037.58 -19.602 1037.58 Q-25.4121 1037.58 -28.4908 1033 Q-31.5464 1028.39 -31.5464 1019.66 Q-31.5464 1010.91 -28.4908 1006.33 Q-25.4121 1001.72 -19.602 1001.72 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M0.559957 1031.03 L5.44419 1031.03 L5.44419 1036.91 L0.559957 1036.91 L0.559957 1031.03 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M25.6293 1020.5 Q22.296 1020.5 20.3747 1022.28 Q18.4765 1024.06 18.4765 1027.18 Q18.4765 1030.31 20.3747 1032.09 Q22.296 1033.87 25.6293 1033.87 Q28.9626 1033.87 30.8839 1032.09 Q32.8052 1030.29 32.8052 1027.18 Q32.8052 1024.06 30.8839 1022.28 Q28.9857 1020.5 25.6293 1020.5 M20.9534 1018.5 Q17.9441 1017.76 16.2543 1015.7 Q14.5877 1013.64 14.5877 1010.68 Q14.5877 1006.54 17.5275 1004.13 Q20.4904 1001.72 25.6293 1001.72 Q30.7913 1001.72 33.7311 1004.13 Q36.6709 1006.54 36.6709 1010.68 Q36.6709 1013.64 34.9811 1015.7 Q33.3144 1017.76 30.3283 1018.5 Q33.7079 1019.29 35.5829 1021.58 Q37.4811 1023.87 37.4811 1027.18 Q37.4811 1032.21 34.4024 1034.89 Q31.3468 1037.58 25.6293 1037.58 Q19.9117 1037.58 16.833 1034.89 Q13.7775 1032.21 13.7775 1027.18 Q13.7775 1023.87 15.6756 1021.58 Q17.5738 1019.29 20.9534 1018.5 M19.2404 1011.12 Q19.2404 1013.81 20.9071 1015.31 Q22.5969 1016.81 25.6293 1016.81 Q28.6385 1016.81 30.3283 1015.31 Q32.0413 1013.81 32.0413 1011.12 Q32.0413 1008.44 30.3283 1006.93 Q28.6385 1005.43 25.6293 1005.43 Q22.5969 1005.43 20.9071 1006.93 Q19.2404 1008.44 19.2404 1011.12 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M45.8375 1002.35 L64.1939 1002.35 L64.1939 1006.28 L50.1199 1006.28 L50.1199 1014.75 Q51.1384 1014.41 52.1569 1014.25 Q53.1754 1014.06 54.1939 1014.06 Q59.9809 1014.06 63.3605 1017.23 Q66.7402 1020.4 66.7402 1025.82 Q66.7402 1031.4 63.268 1034.5 Q59.7958 1037.58 53.4763 1037.58 Q51.3004 1037.58 49.0319 1037.21 Q46.7866 1036.84 44.3792 1036.1 L44.3792 1031.4 Q46.4625 1032.53 48.6847 1033.09 Q50.9069 1033.64 53.3837 1033.64 Q57.3884 1033.64 59.7263 1031.54 Q62.0643 1029.43 62.0643 1025.82 Q62.0643 1022.21 59.7263 1020.1 Q57.3884 1018 53.3837 1018 Q51.5088 1018 49.6338 1018.41 Q47.7819 1018.83 45.8375 1019.71 L45.8375 1002.35 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M-20.5973 762.33 Q-24.2084 762.33 -26.0371 765.895 Q-27.8427 769.436 -27.8427 776.566 Q-27.8427 783.673 -26.0371 787.237 Q-24.2084 790.779 -20.5973 790.779 Q-16.9631 790.779 -15.1576 787.237 Q-13.3289 783.673 -13.3289 776.566 Q-13.3289 769.436 -15.1576 765.895 Q-16.9631 762.33 -20.5973 762.33 M-20.5973 758.626 Q-14.7872 758.626 -11.7316 763.233 Q-8.65296 767.816 -8.65296 776.566 Q-8.65296 785.293 -11.7316 789.899 Q-14.7872 794.483 -20.5973 794.483 Q-26.4075 794.483 -29.4862 789.899 Q-32.5417 785.293 -32.5417 776.566 Q-32.5417 767.816 -29.4862 763.233 Q-26.4075 758.626 -20.5973 758.626 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M-0.435408 787.932 L4.44882 787.932 L4.44882 793.811 L-0.435408 793.811 L-0.435408 787.932 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M14.7728 793.094 L14.7728 788.835 Q16.5321 789.668 18.3376 790.108 Q20.1432 790.547 21.8793 790.547 Q26.5089 790.547 28.9394 787.446 Q31.3931 784.321 31.7403 777.978 Q30.3978 779.969 28.3376 781.034 Q26.2774 782.098 23.7774 782.098 Q18.5923 782.098 15.5599 778.973 Q12.5506 775.825 12.5506 770.386 Q12.5506 765.062 15.6988 761.844 Q18.8469 758.626 24.0783 758.626 Q30.0737 758.626 33.2218 763.233 Q36.3931 767.816 36.3931 776.566 Q36.3931 784.737 32.5042 789.622 Q28.6385 794.483 22.0876 794.483 Q20.3284 794.483 18.5228 794.135 Q16.7173 793.788 14.7728 793.094 M24.0783 778.441 Q27.2265 778.441 29.0552 776.288 Q30.907 774.136 30.907 770.386 Q30.907 766.659 29.0552 764.506 Q27.2265 762.33 24.0783 762.33 Q20.9302 762.33 19.0784 764.506 Q17.2497 766.659 17.2497 770.386 Q17.2497 774.136 19.0784 776.288 Q20.9302 778.441 24.0783 778.441 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M54.7958 762.33 Q51.1847 762.33 49.356 765.895 Q47.5504 769.436 47.5504 776.566 Q47.5504 783.673 49.356 787.237 Q51.1847 790.779 54.7958 790.779 Q58.43 790.779 60.2356 787.237 Q62.0643 783.673 62.0643 776.566 Q62.0643 769.436 60.2356 765.895 Q58.43 762.33 54.7958 762.33 M54.7958 758.626 Q60.6059 758.626 63.6615 763.233 Q66.7402 767.816 66.7402 776.566 Q66.7402 785.293 63.6615 789.899 Q60.6059 794.483 54.7958 794.483 Q48.9856 794.483 45.9069 789.899 Q42.8514 785.293 42.8514 776.566 Q42.8514 767.816 45.9069 763.233 Q48.9856 758.626 54.7958 758.626 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M-19.602 519.234 Q-23.2131 519.234 -25.0418 522.799 Q-26.8473 526.341 -26.8473 533.47 Q-26.8473 540.577 -25.0418 544.142 Q-23.2131 547.683 -19.602 547.683 Q-15.9677 547.683 -14.1622 544.142 Q-12.3335 540.577 -12.3335 533.47 Q-12.3335 526.341 -14.1622 522.799 Q-15.9677 519.234 -19.602 519.234 M-19.602 515.531 Q-13.7918 515.531 -10.7363 520.137 Q-7.65759 524.72 -7.65759 533.47 Q-7.65759 542.197 -10.7363 546.804 Q-13.7918 551.387 -19.602 551.387 Q-25.4121 551.387 -28.4908 546.804 Q-31.5464 542.197 -31.5464 533.47 Q-31.5464 524.72 -28.4908 520.137 Q-25.4121 515.531 -19.602 515.531 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M0.559957 544.836 L5.44419 544.836 L5.44419 550.716 L0.559957 550.716 L0.559957 544.836 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M15.7682 549.998 L15.7682 545.739 Q17.5275 546.572 19.333 547.012 Q21.1385 547.452 22.8747 547.452 Q27.5043 547.452 29.9348 544.35 Q32.3885 541.225 32.7357 534.882 Q31.3931 536.873 29.333 537.938 Q27.2728 539.003 24.7728 539.003 Q19.5876 539.003 16.5552 535.878 Q13.546 532.73 13.546 527.29 Q13.546 521.966 16.6941 518.748 Q19.8423 515.531 25.0737 515.531 Q31.0691 515.531 34.2172 520.137 Q37.3885 524.72 37.3885 533.47 Q37.3885 541.642 33.4996 546.526 Q29.6339 551.387 23.083 551.387 Q21.3237 551.387 19.5182 551.04 Q17.7126 550.692 15.7682 549.998 M25.0737 535.345 Q28.2218 535.345 30.0505 533.193 Q31.9024 531.04 31.9024 527.29 Q31.9024 523.563 30.0505 521.41 Q28.2218 519.234 25.0737 519.234 Q21.9256 519.234 20.0737 521.41 Q18.245 523.563 18.245 527.29 Q18.245 531.04 20.0737 533.193 Q21.9256 535.345 25.0737 535.345 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M45.8375 516.156 L64.1939 516.156 L64.1939 520.091 L50.1199 520.091 L50.1199 528.563 Q51.1384 528.216 52.1569 528.054 Q53.1754 527.869 54.1939 527.869 Q59.9809 527.869 63.3605 531.04 Q66.7402 534.211 66.7402 539.628 Q66.7402 545.206 63.268 548.308 Q59.7958 551.387 53.4763 551.387 Q51.3004 551.387 49.0319 551.017 Q46.7866 550.646 44.3792 549.905 L44.3792 545.206 Q46.4625 546.341 48.6847 546.896 Q50.9069 547.452 53.3837 547.452 Q57.3884 547.452 59.7263 545.345 Q62.0643 543.239 62.0643 539.628 Q62.0643 536.017 59.7263 533.91 Q57.3884 531.804 53.3837 531.804 Q51.5088 531.804 49.6338 532.22 Q47.7819 532.637 45.8375 533.517 L45.8375 516.156 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M-29.7871 303.685 L-22.1483 303.685 L-22.1483 277.319 L-30.4584 278.986 L-30.4584 274.727 L-22.1946 273.06 L-17.5187 273.06 L-17.5187 303.685 L-9.8798 303.685 L-9.8798 307.62 L-29.7871 307.62 L-29.7871 303.685 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M-0.435408 301.74 L4.44882 301.74 L4.44882 307.62 L-0.435408 307.62 L-0.435408 301.74 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M24.6339 276.139 Q21.0228 276.139 19.1941 279.703 Q17.3886 283.245 17.3886 290.375 Q17.3886 297.481 19.1941 301.046 Q21.0228 304.587 24.6339 304.587 Q28.2681 304.587 30.0737 301.046 Q31.9024 297.481 31.9024 290.375 Q31.9024 283.245 30.0737 279.703 Q28.2681 276.139 24.6339 276.139 M24.6339 272.435 Q30.4441 272.435 33.4996 277.041 Q36.5783 281.625 36.5783 290.375 Q36.5783 299.101 33.4996 303.708 Q30.4441 308.291 24.6339 308.291 Q18.8237 308.291 15.7451 303.708 Q12.6895 299.101 12.6895 290.375 Q12.6895 281.625 15.7451 277.041 Q18.8237 272.435 24.6339 272.435 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M54.7958 276.139 Q51.1847 276.139 49.356 279.703 Q47.5504 283.245 47.5504 290.375 Q47.5504 297.481 49.356 301.046 Q51.1847 304.587 54.7958 304.587 Q58.43 304.587 60.2356 301.046 Q62.0643 297.481 62.0643 290.375 Q62.0643 283.245 60.2356 279.703 Q58.43 276.139 54.7958 276.139 M54.7958 272.435 Q60.6059 272.435 63.6615 277.041 Q66.7402 281.625 66.7402 290.375 Q66.7402 299.101 63.6615 303.708 Q60.6059 308.291 54.7958 308.291 Q48.9856 308.291 45.9069 303.708 Q42.8514 299.101 42.8514 290.375 Q42.8514 281.625 45.9069 277.041 Q48.9856 272.435 54.7958 272.435 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M-112.971 765.812 Q-113.544 766.799 -113.799 767.977 Q-114.085 769.122 -114.085 770.523 Q-114.085 775.488 -110.839 778.162 Q-107.624 780.803 -101.577 780.803 L-82.7977 780.803 L-82.7977 786.692 L-118.446 786.692 L-118.446 780.803 L-112.908 780.803 Q-116.154 778.957 -117.714 775.997 Q-119.305 773.037 -119.305 768.804 Q-119.305 768.199 -119.21 767.467 Q-119.146 766.735 -118.987 765.844 L-112.971 765.812 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><line clip-path="url(#clip952)" x1="166.42" y1="290.34" x2="166.42" y2="274.34" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="166.42" y1="290.34" x2="150.42" y2="290.34" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="166.42" y1="290.34" x2="166.42" y2="306.34" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="166.42" y1="290.34" x2="182.42" y2="290.34" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1227.75" y1="290.481" x2="1227.75" y2="274.481" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1227.75" y1="290.481" x2="1211.75" y2="290.481" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1227.75" y1="290.481" x2="1227.75" y2="306.481" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1227.75" y1="290.481" x2="1243.75" y2="290.481" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1749.59" y1="397.278" x2="1749.59" y2="381.278" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1749.59" y1="397.278" x2="1733.59" y2="397.278" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1749.59" y1="397.278" x2="1749.59" y2="413.278" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1749.59" y1="397.278" x2="1765.59" y2="397.278" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="705.901" y1="397.278" x2="705.901" y2="381.278" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="705.901" y1="397.278" x2="689.901" y2="397.278" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="705.901" y1="397.278" x2="705.901" y2="413.278" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="705.901" y1="397.278" x2="721.901" y2="397.278" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="688.267" y1="397.291" x2="688.267" y2="381.291" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="688.267" y1="397.291" x2="672.267" y2="397.291" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="688.267" y1="397.291" x2="688.267" y2="413.291" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="688.267" y1="397.291" x2="704.267" y2="397.291" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1767.23" y1="397.291" x2="1767.23" y2="381.291" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1767.23" y1="397.291" x2="1751.23" y2="397.291" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1767.23" y1="397.291" x2="1767.23" y2="413.291" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1767.23" y1="397.291" x2="1783.23" y2="397.291" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="2289.08" y1="754.876" x2="2289.08" y2="738.876" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="2289.08" y1="754.876" x2="2273.08" y2="754.876" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="2289.08" y1="754.876" x2="2289.08" y2="770.876" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="2289.08" y1="754.876" x2="2305.08" y2="754.876" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1227.75" y1="754.877" x2="1227.75" y2="738.877" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1227.75" y1="754.877" x2="1211.75" y2="754.877" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1227.75" y1="754.877" x2="1227.75" y2="770.877" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1227.75" y1="754.877" x2="1243.75" y2="754.877" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="181.496" y1="773.03" x2="181.496" y2="757.03" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="181.496" y1="773.03" x2="165.496" y2="773.03" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="181.496" y1="773.03" x2="181.496" y2="789.03" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="181.496" y1="773.03" x2="197.496" y2="773.03" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="2274" y1="773.03" x2="2274" y2="757.03" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="2274" y1="773.03" x2="2258" y2="773.03" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="2274" y1="773.03" x2="2274" y2="789.03" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="2274" y1="773.03" x2="2290" y2="773.03" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1212.67" y1="773.037" x2="1212.67" y2="757.037" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1212.67" y1="773.037" x2="1196.67" y2="773.037" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1212.67" y1="773.037" x2="1212.67" y2="789.037" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1212.67" y1="773.037" x2="1228.67" y2="773.037" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1242.82" y1="773.037" x2="1242.82" y2="757.037" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1242.82" y1="773.037" x2="1226.82" y2="773.037" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1242.82" y1="773.037" x2="1242.82" y2="789.037" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1242.82" y1="773.037" x2="1258.82" y2="773.037" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="692.264" y1="1023.96" x2="692.264" y2="1007.96" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="692.264" y1="1023.96" x2="676.264" y2="1023.96" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="692.264" y1="1023.96" x2="692.264" y2="1039.96" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="692.264" y1="1023.96" x2="708.264" y2="1023.96" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1763.23" y1="1023.96" x2="1763.23" y2="1007.96" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1763.23" y1="1023.96" x2="1747.23" y2="1023.96" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1763.23" y1="1023.96" x2="1763.23" y2="1039.96" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1763.23" y1="1023.96" x2="1779.23" y2="1023.96" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1753.59" y1="1024.01" x2="1753.59" y2="1008.01" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1753.59" y1="1024.01" x2="1737.59" y2="1024.01" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1753.59" y1="1024.01" x2="1753.59" y2="1040.01" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1753.59" y1="1024.01" x2="1769.59" y2="1024.01" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="701.904" y1="1024.01" x2="701.904" y2="1008.01" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="701.904" y1="1024.01" x2="685.904" y2="1024.01" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="701.904" y1="1024.01" x2="701.904" y2="1040.01" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="701.904" y1="1024.01" x2="717.904" y2="1024.01" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="704.812" y1="1467.1" x2="704.812" y2="1451.1" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="704.812" y1="1467.1" x2="688.812" y2="1467.1" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="704.812" y1="1467.1" x2="704.812" y2="1483.1" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="704.812" y1="1467.1" x2="720.812" y2="1467.1" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1750.68" y1="1467.1" x2="1750.68" y2="1451.1" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1750.68" y1="1467.1" x2="1734.68" y2="1467.1" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1750.68" y1="1467.1" x2="1750.68" y2="1483.1" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1750.68" y1="1467.1" x2="1766.68" y2="1467.1" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="689.355" y1="1467.11" x2="689.355" y2="1451.11" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="689.355" y1="1467.11" x2="673.355" y2="1467.11" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="689.355" y1="1467.11" x2="689.355" y2="1483.11" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="689.355" y1="1467.11" x2="705.355" y2="1467.11" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1766.14" y1="1467.11" x2="1766.14" y2="1451.11" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1766.14" y1="1467.11" x2="1750.14" y2="1467.11" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1766.14" y1="1467.11" x2="1766.14" y2="1483.11" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1766.14" y1="1467.11" x2="1782.14" y2="1467.11" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="679.248" y1="1505.23" x2="679.248" y2="1489.23" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="679.248" y1="1505.23" x2="663.248" y2="1505.23" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="679.248" y1="1505.23" x2="679.248" y2="1521.23" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="679.248" y1="1505.23" x2="695.248" y2="1505.23" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1776.25" y1="1505.23" x2="1776.25" y2="1489.23" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1776.25" y1="1505.23" x2="1760.25" y2="1505.23" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1776.25" y1="1505.23" x2="1776.25" y2="1521.23" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1776.25" y1="1505.23" x2="1792.25" y2="1505.23" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="714.919" y1="1505.26" x2="714.919" y2="1489.26" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="714.919" y1="1505.26" x2="698.919" y2="1505.26" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="714.919" y1="1505.26" x2="714.919" y2="1521.26" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="714.919" y1="1505.26" x2="730.919" y2="1505.26" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1740.58" y1="1505.26" x2="1740.58" y2="1489.26" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1740.58" y1="1505.26" x2="1724.58" y2="1505.26" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1740.58" y1="1505.26" x2="1740.58" y2="1521.26" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip952)" x1="1740.58" y1="1505.26" x2="1756.58" y2="1505.26" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<path clip-path="url(#clip950)" d="M1881.6 199.525 L2277.76 199.525 L2277.76 95.8446 L1881.6 95.8446  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<polyline clip-path="url(#clip950)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1881.6,199.525 2277.76,199.525 2277.76,95.8446 1881.6,95.8446 1881.6,199.525 "/>
<line clip-path="url(#clip950)" x1="1981.6" y1="147.685" x2="1981.6" y2="124.929" style="stroke:#009af9; stroke-width:4.55111; stroke-opacity:1"/>
<line clip-path="url(#clip950)" x1="1981.6" y1="147.685" x2="1958.84" y2="147.685" style="stroke:#009af9; stroke-width:4.55111; stroke-opacity:1"/>
<line clip-path="url(#clip950)" x1="1981.6" y1="147.685" x2="1981.6" y2="170.44" style="stroke:#009af9; stroke-width:4.55111; stroke-opacity:1"/>
<line clip-path="url(#clip950)" x1="1981.6" y1="147.685" x2="2004.35" y2="147.685" style="stroke:#009af9; stroke-width:4.55111; stroke-opacity:1"/>
<path clip-path="url(#clip950)" d="M2081.6 130.405 L2110.83 130.405 L2110.83 134.34 L2098.57 134.34 L2098.57 164.965 L2093.87 164.965 L2093.87 134.34 L2081.6 134.34 L2081.6 130.405 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M2123.22 143.02 Q2122.5 142.604 2121.64 142.418 Q2120.81 142.21 2119.79 142.21 Q2116.18 142.21 2114.24 144.571 Q2112.32 146.909 2112.32 151.307 L2112.32 164.965 L2108.03 164.965 L2108.03 139.039 L2112.32 139.039 L2112.32 143.067 Q2113.66 140.705 2115.81 139.571 Q2117.96 138.414 2121.04 138.414 Q2121.48 138.414 2122.02 138.483 Q2122.55 138.53 2123.2 138.645 L2123.22 143.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M2127.69 139.039 L2131.95 139.039 L2131.95 164.965 L2127.69 164.965 L2127.69 139.039 M2127.69 128.946 L2131.95 128.946 L2131.95 134.34 L2127.69 134.34 L2127.69 128.946 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M2137.8 139.039 L2142.32 139.039 L2150.42 160.798 L2158.52 139.039 L2163.03 139.039 L2153.31 164.965 L2147.52 164.965 L2137.8 139.039 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M2168.91 139.039 L2173.17 139.039 L2173.17 164.965 L2168.91 164.965 L2168.91 139.039 M2168.91 128.946 L2173.17 128.946 L2173.17 134.34 L2168.91 134.34 L2168.91 128.946 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M2193.87 151.932 Q2188.7 151.932 2186.71 153.113 Q2184.72 154.293 2184.72 157.141 Q2184.72 159.409 2186.2 160.752 Q2187.71 162.071 2190.28 162.071 Q2193.82 162.071 2195.95 159.571 Q2198.1 157.048 2198.1 152.881 L2198.1 151.932 L2193.87 151.932 M2202.36 150.173 L2202.36 164.965 L2198.1 164.965 L2198.1 161.029 Q2196.64 163.391 2194.47 164.525 Q2192.29 165.636 2189.14 165.636 Q2185.16 165.636 2182.8 163.414 Q2180.46 161.168 2180.46 157.418 Q2180.46 153.043 2183.38 150.821 Q2186.32 148.599 2192.13 148.599 L2198.1 148.599 L2198.1 148.182 Q2198.1 145.243 2196.16 143.645 Q2194.24 142.025 2190.74 142.025 Q2188.52 142.025 2186.41 142.557 Q2184.31 143.09 2182.36 144.155 L2182.36 140.219 Q2184.7 139.317 2186.9 138.877 Q2189.1 138.414 2191.18 138.414 Q2196.81 138.414 2199.58 141.33 Q2202.36 144.247 2202.36 150.173 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M2211.14 128.946 L2215.39 128.946 L2215.39 164.965 L2211.14 164.965 L2211.14 128.946 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M2234.54 128.993 Q2231.44 134.317 2229.93 139.525 Q2228.43 144.733 2228.43 150.08 Q2228.43 155.428 2229.93 160.682 Q2231.46 165.914 2234.54 171.215 L2230.83 171.215 Q2227.36 165.775 2225.63 160.52 Q2223.91 155.266 2223.91 150.08 Q2223.91 144.918 2225.63 139.687 Q2227.34 134.456 2230.83 128.993 L2234.54 128.993 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip950)" d="M2242.13 128.993 L2245.83 128.993 Q2249.31 134.456 2251.02 139.687 Q2252.76 144.918 2252.76 150.08 Q2252.76 155.266 2251.02 160.52 Q2249.31 165.775 2245.83 171.215 L2242.13 171.215 Q2245.21 165.914 2246.71 160.682 Q2248.24 155.428 2248.24 150.08 Q2248.24 144.733 2246.71 139.525 Q2245.21 134.317 2242.13 128.993 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /></svg>

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
H2 = heisenberg_XXX(ComplexF64, Trivial, InfiniteChain(2); spin=1 // 2)
groundstate, envs, delta = find_groundstate(state, H2,
                                            VUMPS(; maxiter=100, tol=1e-12));
````

````
[ Info: VUMPS init:	obj = +4.991752933537e-01	err = 6.3268e-02
[ Info: VUMPS   1:	obj = -3.177585801487e-01	err = 3.4457002483e-01	time = 0.26 sec
[ Info: VUMPS   2:	obj = -8.783277426699e-01	err = 7.2859563338e-02	time = 0.02 sec
[ Info: VUMPS   3:	obj = -8.852412984334e-01	err = 1.2557144867e-02	time = 0.01 sec
[ Info: VUMPS   4:	obj = -8.859163712655e-01	err = 6.9408088858e-03	time = 0.02 sec
[ Info: VUMPS   5:	obj = -8.861199316380e-01	err = 3.8910203290e-03	time = 0.09 sec
[ Info: VUMPS   6:	obj = -8.861865553641e-01	err = 2.8480322641e-03	time = 0.02 sec
[ Info: VUMPS   7:	obj = -8.862145191872e-01	err = 2.0944450809e-03	time = 0.02 sec
[ Info: VUMPS   8:	obj = -8.862266281943e-01	err = 1.7300547846e-03	time = 0.02 sec
[ Info: VUMPS   9:	obj = -8.862324798521e-01	err = 1.4474994746e-03	time = 0.04 sec
[ Info: VUMPS  10:	obj = -8.862352100866e-01	err = 1.4187844649e-03	time = 0.10 sec
[ Info: VUMPS  11:	obj = -8.862366322409e-01	err = 1.3017638159e-03	time = 0.05 sec
[ Info: VUMPS  12:	obj = -8.862372899823e-01	err = 1.4040262285e-03	time = 0.05 sec
[ Info: VUMPS  13:	obj = -8.862377268299e-01	err = 1.3307584406e-03	time = 0.05 sec
[ Info: VUMPS  14:	obj = -8.862379533217e-01	err = 1.4146443881e-03	time = 0.05 sec
[ Info: VUMPS  15:	obj = -8.862382038030e-01	err = 1.3272177810e-03	time = 0.09 sec
[ Info: VUMPS  16:	obj = -8.862383890237e-01	err = 1.3377148402e-03	time = 0.04 sec
[ Info: VUMPS  17:	obj = -8.862386470981e-01	err = 1.1956124852e-03	time = 0.03 sec
[ Info: VUMPS  18:	obj = -8.862388507153e-01	err = 1.1254536444e-03	time = 0.03 sec
[ Info: VUMPS  19:	obj = -8.862391024178e-01	err = 9.3067565334e-04	time = 0.06 sec
[ Info: VUMPS  20:	obj = -8.862392792626e-01	err = 8.0592942148e-04	time = 0.03 sec
[ Info: VUMPS  21:	obj = -8.862394408212e-01	err = 6.3751094232e-04	time = 0.03 sec
[ Info: VUMPS  22:	obj = -8.862395447014e-01	err = 5.2641943438e-04	time = 0.08 sec
[ Info: VUMPS  23:	obj = -8.862396202592e-01	err = 4.2357063364e-04	time = 0.05 sec
[ Info: VUMPS  24:	obj = -8.862396677820e-01	err = 3.5346208123e-04	time = 0.05 sec
[ Info: VUMPS  25:	obj = -8.862396999787e-01	err = 2.9779376807e-04	time = 0.05 sec
[ Info: VUMPS  26:	obj = -8.862397209167e-01	err = 2.5591990952e-04	time = 0.09 sec
[ Info: VUMPS  27:	obj = -8.862397356027e-01	err = 2.2322838649e-04	time = 0.05 sec
[ Info: VUMPS  28:	obj = -8.862397458518e-01	err = 1.9539015992e-04	time = 0.05 sec
[ Info: VUMPS  29:	obj = -8.862397536040e-01	err = 1.7307555129e-04	time = 0.09 sec
[ Info: VUMPS  30:	obj = -8.862397594558e-01	err = 1.5260211513e-04	time = 0.05 sec
[ Info: VUMPS  31:	obj = -8.862397642025e-01	err = 1.3574708036e-04	time = 0.05 sec
[ Info: VUMPS  32:	obj = -8.862397680164e-01	err = 1.2004395690e-04	time = 0.05 sec
[ Info: VUMPS  33:	obj = -8.862397712590e-01	err = 1.0674955980e-04	time = 0.11 sec
[ Info: VUMPS  34:	obj = -8.862397739766e-01	err = 9.4662033971e-05	time = 0.05 sec
[ Info: VUMPS  35:	obj = -8.862397763530e-01	err = 8.4025399031e-05	time = 0.05 sec
[ Info: VUMPS  36:	obj = -8.862397784016e-01	err = 7.4857517793e-05	time = 0.09 sec
[ Info: VUMPS  37:	obj = -8.862397802247e-01	err = 6.6308156905e-05	time = 0.03 sec
[ Info: VUMPS  38:	obj = -8.862397818290e-01	err = 5.9532927397e-05	time = 0.02 sec
[ Info: VUMPS  39:	obj = -8.862397832750e-01	err = 5.2635281822e-05	time = 0.03 sec
[ Info: VUMPS  40:	obj = -8.862397845683e-01	err = 4.7812478221e-05	time = 0.05 sec
[ Info: VUMPS  41:	obj = -8.862397857460e-01	err = 4.2209834688e-05	time = 0.05 sec
[ Info: VUMPS  42:	obj = -8.862397868137e-01	err = 3.8962795037e-05	time = 0.04 sec
[ Info: VUMPS  43:	obj = -8.862397877944e-01	err = 3.4360884968e-05	time = 0.05 sec
[ Info: VUMPS  44:	obj = -8.862397886936e-01	err = 3.2362735927e-05	time = 0.08 sec
[ Info: VUMPS  45:	obj = -8.862397895258e-01	err = 2.8521727631e-05	time = 0.05 sec
[ Info: VUMPS  46:	obj = -8.862397902957e-01	err = 2.7488241281e-05	time = 0.05 sec
[ Info: VUMPS  47:	obj = -8.862397910129e-01	err = 2.4218467594e-05	time = 0.09 sec
[ Info: VUMPS  48:	obj = -8.862397916814e-01	err = 2.3904114480e-05	time = 0.05 sec
[ Info: VUMPS  49:	obj = -8.862397923073e-01	err = 2.1059493401e-05	time = 0.05 sec
[ Info: VUMPS  50:	obj = -8.862397928943e-01	err = 2.1258345276e-05	time = 0.05 sec
[ Info: VUMPS  51:	obj = -8.862397934463e-01	err = 1.8738335007e-05	time = 0.07 sec
[ Info: VUMPS  52:	obj = -8.862397939664e-01	err = 1.9277045662e-05	time = 0.02 sec
[ Info: VUMPS  53:	obj = -8.862397944573e-01	err = 1.7514627759e-05	time = 0.04 sec
[ Info: VUMPS  54:	obj = -8.862397949216e-01	err = 1.7756990313e-05	time = 0.08 sec
[ Info: VUMPS  55:	obj = -8.862397953612e-01	err = 1.6507778424e-05	time = 0.05 sec
[ Info: VUMPS  56:	obj = -8.862397957782e-01	err = 1.6553773913e-05	time = 0.05 sec
[ Info: VUMPS  57:	obj = -8.862397961741e-01	err = 1.5646923962e-05	time = 0.05 sec
[ Info: VUMPS  58:	obj = -8.862397965506e-01	err = 1.5568425529e-05	time = 0.09 sec
[ Info: VUMPS  59:	obj = -8.862397969088e-01	err = 1.4888553209e-05	time = 0.05 sec
[ Info: VUMPS  60:	obj = -8.862397972502e-01	err = 1.4734924565e-05	time = 0.05 sec
[ Info: VUMPS  61:	obj = -8.862397975756e-01	err = 1.4205866057e-05	time = 0.09 sec
[ Info: VUMPS  62:	obj = -8.862397978863e-01	err = 1.4009885396e-05	time = 0.05 sec
[ Info: VUMPS  63:	obj = -8.862397981830e-01	err = 1.3582081159e-05	time = 0.05 sec
[ Info: VUMPS  64:	obj = -8.862397984666e-01	err = 1.3364902771e-05	time = 0.05 sec
[ Info: VUMPS  65:	obj = -8.862397987379e-01	err = 1.3006347133e-05	time = 0.11 sec
[ Info: VUMPS  66:	obj = -8.862397989976e-01	err = 1.2781264172e-05	time = 0.04 sec
[ Info: VUMPS  67:	obj = -8.862397992464e-01	err = 1.2471321790e-05	time = 0.05 sec
[ Info: VUMPS  68:	obj = -8.862397994848e-01	err = 1.2246441659e-05	time = 0.09 sec
[ Info: VUMPS  69:	obj = -8.862397997134e-01	err = 1.1971757049e-05	time = 0.03 sec
[ Info: VUMPS  70:	obj = -8.862397999328e-01	err = 1.1751836617e-05	time = 0.04 sec
[ Info: VUMPS  71:	obj = -8.862398001434e-01	err = 1.1503687228e-05	time = 0.05 sec
[ Info: VUMPS  72:	obj = -8.862398003457e-01	err = 1.1291356616e-05	time = 0.08 sec
[ Info: VUMPS  73:	obj = -8.862398005401e-01	err = 1.1063958815e-05	time = 0.05 sec
[ Info: VUMPS  74:	obj = -8.862398007271e-01	err = 1.0860522804e-05	time = 0.05 sec
[ Info: VUMPS  75:	obj = -8.862398009068e-01	err = 1.0649959354e-05	time = 0.05 sec
[ Info: VUMPS  76:	obj = -8.862398010799e-01	err = 1.0455912668e-05	time = 0.09 sec
[ Info: VUMPS  77:	obj = -8.862398012464e-01	err = 1.0259455807e-05	time = 0.05 sec
[ Info: VUMPS  78:	obj = -8.862398014068e-01	err = 1.0074813285e-05	time = 0.04 sec
[ Info: VUMPS  79:	obj = -8.862398015614e-01	err = 9.8905015821e-06	time = 0.09 sec
[ Info: VUMPS  80:	obj = -8.862398017103e-01	err = 9.7150036409e-06	time = 0.05 sec
[ Info: VUMPS  81:	obj = -8.862398018540e-01	err = 9.5413743747e-06	time = 0.05 sec
[ Info: VUMPS  82:	obj = -8.862398019925e-01	err = 9.3746148885e-06	time = 0.05 sec
[ Info: VUMPS  83:	obj = -8.862398021261e-01	err = 9.2105357844e-06	time = 0.07 sec
[ Info: VUMPS  84:	obj = -8.862398022551e-01	err = 9.0520382611e-06	time = 0.04 sec
[ Info: VUMPS  85:	obj = -8.862398023797e-01	err = 8.8966029204e-06	time = 0.05 sec
[ Info: VUMPS  86:	obj = -8.862398025000e-01	err = 8.7458737419e-06	time = 0.09 sec
[ Info: VUMPS  87:	obj = -8.862398026162e-01	err = 8.5983314572e-06	time = 0.05 sec
[ Info: VUMPS  88:	obj = -8.862398027285e-01	err = 8.4548765300e-06	time = 0.04 sec
[ Info: VUMPS  89:	obj = -8.862398028370e-01	err = 8.3145903294e-06	time = 0.05 sec
[ Info: VUMPS  90:	obj = -8.862398029420e-01	err = 8.1779367252e-06	time = 0.11 sec
[ Info: VUMPS  91:	obj = -8.862398030435e-01	err = 8.0443547655e-06	time = 0.05 sec
[ Info: VUMPS  92:	obj = -8.862398031417e-01	err = 7.9140564546e-06	time = 0.05 sec
[ Info: VUMPS  93:	obj = -8.862398032367e-01	err = 7.7866927647e-06	time = 0.09 sec
[ Info: VUMPS  94:	obj = -8.862398033287e-01	err = 7.6623336924e-06	time = 0.04 sec
[ Info: VUMPS  95:	obj = -8.862398034177e-01	err = 7.5407551359e-06	time = 0.05 sec
[ Info: VUMPS  96:	obj = -8.862398035039e-01	err = 7.4219497671e-06	time = 0.05 sec
[ Info: VUMPS  97:	obj = -8.862398035875e-01	err = 7.3057668246e-06	time = 0.08 sec
[ Info: VUMPS  98:	obj = -8.862398036684e-01	err = 7.1921592390e-06	time = 0.05 sec
[ Info: VUMPS  99:	obj = -8.862398037467e-01	err = 7.0810192783e-06	time = 0.05 sec
┌ Warning: VUMPS cancel 100:	obj = -8.862398038227e-01	err = 6.9722815663e-06	time = 5.64 sec
└ @ MPSKit ~/Projects/MPSKit.jl/src/algorithms/groundstate/vumps.jl:73

````

We get convergence, but it takes an enormous amount of iterations.
The reason behind this becomes more obvious at higher bond dimensions:

````julia
groundstate, envs, delta = find_groundstate(state, H2,
                                            IDMRG2(; trscheme=truncdim(50), maxiter=20,
                                                   tol=1e-12));
entanglementplot(groundstate)
````

```@raw html
<?xml version="1.0" encoding="utf-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="600" height="400" viewBox="0 0 2400 1600">
<defs>
  <clipPath id="clip980">
    <rect x="0" y="0" width="2400" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip980)" d="M0 1600 L2400 1600 L2400 0 L0 0  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip981">
    <rect x="480" y="0" width="1681" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip980)" d="M86.9921 1505.26 L2352.76 1505.26 L2352.76 62.9921 L86.9921 62.9921  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip982">
    <rect x="86" y="62" width="2267" height="1443"/>
  </clipPath>
</defs>
<polyline clip-path="url(#clip982)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="151.118,1505.26 151.118,62.9921 "/>
<polyline clip-path="url(#clip982)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="86.9921,1501.73 2352.76,1501.73 "/>
<polyline clip-path="url(#clip982)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="86.9921,1155.83 2352.76,1155.83 "/>
<polyline clip-path="url(#clip982)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="86.9921,809.931 2352.76,809.931 "/>
<polyline clip-path="url(#clip982)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="86.9921,464.03 2352.76,464.03 "/>
<polyline clip-path="url(#clip982)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="86.9921,118.129 2352.76,118.129 "/>
<polyline clip-path="url(#clip980)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="86.9921,1505.26 2352.76,1505.26 "/>
<polyline clip-path="url(#clip980)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="151.118,1505.26 151.118,1524.16 "/>
<path clip-path="url(#clip980)" d="M17.8728 1661.2 L38.5457 1640.52 L41.3283 1643.3 L32.6532 1651.98 L54.3082 1673.63 L50.9855 1676.96 L29.3305 1655.3 L20.6554 1663.98 L17.8728 1661.2 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M56.2233 1640.69 Q55.4213 1640.9 54.6847 1641.37 Q53.9481 1641.82 53.2279 1642.54 Q50.6745 1645.09 50.9691 1648.13 Q51.2638 1651.15 54.3737 1654.26 L64.0309 1663.91 L61.0028 1666.94 L42.6705 1648.61 L45.6986 1645.58 L48.5467 1648.43 Q47.8265 1645.81 48.5467 1643.48 Q49.2505 1641.14 51.4275 1638.97 Q51.7384 1638.66 52.164 1638.33 Q52.5732 1637.99 53.1134 1637.61 L56.2233 1640.69 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M56.567 1634.71 L59.5788 1631.7 L77.9111 1650.03 L74.8993 1653.04 L56.567 1634.71 M49.4305 1627.58 L52.4423 1624.56 L56.256 1628.38 L53.2443 1631.39 L49.4305 1627.58 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M63.7199 1627.56 L66.9117 1624.37 L88.0266 1634.02 L78.3694 1612.91 L81.5612 1609.72 L93.0189 1634.92 L88.9268 1639.02 L63.7199 1627.56 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M85.7187 1605.56 L88.7304 1602.55 L107.063 1620.88 L104.051 1623.89 L85.7187 1605.56 M78.5822 1598.42 L81.5939 1595.41 L85.4077 1599.23 L82.3959 1602.24 L78.5822 1598.42 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M112.481 1597.03 Q108.83 1600.68 108.258 1602.92 Q107.685 1605.17 109.698 1607.18 Q111.302 1608.78 113.299 1608.69 Q115.296 1608.56 117.113 1606.74 Q119.617 1604.23 119.355 1600.96 Q119.093 1597.65 116.147 1594.71 L115.476 1594.04 L112.481 1597.03 M117.244 1589.78 L127.703 1600.24 L124.691 1603.25 L121.909 1600.47 Q122.547 1603.17 121.81 1605.51 Q121.057 1607.84 118.831 1610.06 Q116.016 1612.88 112.775 1612.97 Q109.534 1613.04 106.883 1610.39 Q103.789 1607.29 104.28 1603.66 Q104.788 1600.01 108.896 1595.9 L113.119 1591.68 L112.824 1591.39 Q110.746 1589.31 108.241 1589.55 Q105.737 1589.76 103.265 1592.24 Q101.694 1593.81 100.581 1595.67 Q99.4679 1597.54 98.8459 1599.67 L96.0633 1596.88 Q97.0782 1594.59 98.3221 1592.73 Q99.5497 1590.84 101.023 1589.37 Q105 1585.39 109.027 1585.49 Q113.053 1585.59 117.244 1589.78 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M108.438 1568.57 L111.449 1565.56 L136.918 1591.03 L133.906 1594.04 L108.438 1568.57 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M125.019 1552.05 Q126.59 1558.01 129.209 1562.76 Q131.828 1567.5 135.609 1571.29 Q139.39 1575.07 144.169 1577.72 Q148.949 1580.34 154.874 1581.91 L152.255 1584.53 Q145.953 1583.14 141.01 1580.65 Q136.083 1578.14 132.417 1574.48 Q128.767 1570.83 126.279 1565.92 Q123.791 1561.01 122.4 1554.67 L125.019 1552.05 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M130.387 1546.68 L133.006 1544.06 Q139.324 1545.47 144.235 1547.96 Q149.162 1550.43 152.812 1554.08 Q156.478 1557.75 158.966 1562.69 Q161.47 1567.62 162.862 1573.92 L160.243 1576.54 Q158.671 1570.61 156.036 1565.85 Q153.401 1561.05 149.62 1557.27 Q145.839 1553.49 141.076 1550.89 Q136.329 1548.27 130.387 1546.68 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M1136.07 1763.92 L1131.33 1751.73 L1121.56 1769.56 L1114.65 1769.56 L1128.46 1744.35 L1122.67 1729.36 Q1121.11 1725.35 1116.21 1725.35 L1114.65 1725.35 L1114.65 1720.32 L1116.88 1720.38 Q1125.09 1720.6 1127.16 1725.92 L1131.87 1738.11 L1141.64 1720.29 L1148.55 1720.29 L1134.73 1745.49 L1140.53 1760.49 Q1142.09 1764.5 1146.99 1764.5 L1148.55 1764.5 L1148.55 1769.53 L1146.32 1769.46 Q1138.11 1769.24 1136.07 1763.92 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M1178.05 1726.37 L1218.86 1726.37 L1218.86 1731.71 L1178.05 1731.71 L1178.05 1726.37 M1178.05 1739.35 L1218.86 1739.35 L1218.86 1744.76 L1178.05 1744.76 L1178.05 1739.35 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M1253.52 1708.45 L1278.76 1708.45 L1278.76 1713.86 L1259.41 1713.86 L1259.41 1725.51 Q1260.81 1725.03 1262.21 1724.81 Q1263.61 1724.55 1265.01 1724.55 Q1272.96 1724.55 1277.61 1728.91 Q1282.26 1733.27 1282.26 1740.72 Q1282.26 1748.39 1277.48 1752.66 Q1272.71 1756.89 1264.02 1756.89 Q1261.03 1756.89 1257.91 1756.38 Q1254.82 1755.87 1251.51 1754.85 L1251.51 1748.39 Q1254.38 1749.95 1257.43 1750.71 Q1260.49 1751.48 1263.89 1751.48 Q1269.4 1751.48 1272.61 1748.58 Q1275.83 1745.69 1275.83 1740.72 Q1275.83 1735.76 1272.61 1732.86 Q1269.4 1729.96 1263.89 1729.96 Q1261.31 1729.96 1258.74 1730.54 Q1256.19 1731.11 1253.52 1732.32 L1253.52 1708.45 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M1308.68 1712.68 Q1303.71 1712.68 1301.2 1717.58 Q1298.71 1722.45 1298.71 1732.25 Q1298.71 1742.03 1301.2 1746.93 Q1303.71 1751.8 1308.68 1751.8 Q1313.67 1751.8 1316.16 1746.93 Q1318.67 1742.03 1318.67 1732.25 Q1318.67 1722.45 1316.16 1717.58 Q1313.67 1712.68 1308.68 1712.68 M1308.68 1707.59 Q1316.66 1707.59 1320.87 1713.92 Q1325.1 1720.22 1325.1 1732.25 Q1325.1 1744.25 1320.87 1750.59 Q1316.66 1756.89 1308.68 1756.89 Q1300.69 1756.89 1296.45 1750.59 Q1292.25 1744.25 1292.25 1732.25 Q1292.25 1720.22 1296.45 1713.92 Q1300.69 1707.59 1308.68 1707.59 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><polyline clip-path="url(#clip980)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="86.9921,1505.26 86.9921,62.9921 "/>
<polyline clip-path="url(#clip980)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="86.9921,1501.73 105.89,1501.73 "/>
<polyline clip-path="url(#clip980)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="86.9921,1155.83 105.89,1155.83 "/>
<polyline clip-path="url(#clip980)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="86.9921,809.931 105.89,809.931 "/>
<polyline clip-path="url(#clip980)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="86.9921,464.03 105.89,464.03 "/>
<polyline clip-path="url(#clip980)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="86.9921,118.129 105.89,118.129 "/>
<path clip-path="url(#clip980)" d="M-50.8404 1521.52 L-43.2015 1521.52 L-43.2015 1495.16 L-51.5117 1496.83 L-51.5117 1492.57 L-43.2478 1490.9 L-38.5719 1490.9 L-38.5719 1521.52 L-30.9331 1521.52 L-30.9331 1525.46 L-50.8404 1525.46 L-50.8404 1521.52 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M-11.4887 1493.98 Q-15.0998 1493.98 -16.9285 1497.54 Q-18.7341 1501.09 -18.7341 1508.21 Q-18.7341 1515.32 -16.9285 1518.89 Q-15.0998 1522.43 -11.4887 1522.43 Q-7.85449 1522.43 -6.04895 1518.89 Q-4.22025 1515.32 -4.22025 1508.21 Q-4.22025 1501.09 -6.04895 1497.54 Q-7.85449 1493.98 -11.4887 1493.98 M-11.4887 1490.28 Q-5.67858 1490.28 -2.62304 1494.88 Q0.455649 1499.46 0.455649 1508.21 Q0.455649 1516.94 -2.62304 1521.55 Q-5.67858 1526.13 -11.4887 1526.13 Q-17.2989 1526.13 -20.3776 1521.55 Q-23.4331 1516.94 -23.4331 1508.21 Q-23.4331 1499.46 -20.3776 1494.88 Q-17.2989 1490.28 -11.4887 1490.28 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M0.455649 1484.38 L24.5672 1484.38 L24.5672 1487.57 L0.455649 1487.57 L0.455649 1484.38 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M43.2057 1473.28 L33.6137 1488.27 L43.2057 1488.27 L43.2057 1473.28 M42.2089 1469.97 L46.9861 1469.97 L46.9861 1488.27 L50.9921 1488.27 L50.9921 1491.43 L46.9861 1491.43 L46.9861 1498.05 L43.2057 1498.05 L43.2057 1491.43 L30.5293 1491.43 L30.5293 1487.76 L42.2089 1469.97 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M-49.9188 1175.62 L-42.2799 1175.62 L-42.2799 1149.26 L-50.5901 1150.93 L-50.5901 1146.67 L-42.3262 1145 L-37.6503 1145 L-37.6503 1175.62 L-30.0115 1175.62 L-30.0115 1179.56 L-49.9188 1179.56 L-49.9188 1175.62 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M-10.5672 1148.08 Q-14.1782 1148.08 -16.0069 1151.64 Q-17.8125 1155.18 -17.8125 1162.31 Q-17.8125 1169.42 -16.0069 1172.99 Q-14.1782 1176.53 -10.5672 1176.53 Q-6.93291 1176.53 -5.12736 1172.99 Q-3.29867 1169.42 -3.29867 1162.31 Q-3.29867 1155.18 -5.12736 1151.64 Q-6.93291 1148.08 -10.5672 1148.08 M-10.5672 1144.37 Q-4.757 1144.37 -1.70146 1148.98 Q1.37723 1153.56 1.37723 1162.31 Q1.37723 1171.04 -1.70146 1175.65 Q-4.757 1180.23 -10.5672 1180.23 Q-16.3773 1180.23 -19.456 1175.65 Q-22.5115 1171.04 -22.5115 1162.31 Q-22.5115 1153.56 -19.456 1148.98 Q-16.3773 1144.37 -10.5672 1144.37 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M1.37723 1138.48 L25.4888 1138.48 L25.4888 1141.67 L1.37723 1141.67 L1.37723 1138.48 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M45.1993 1137.01 Q47.9265 1137.59 49.4499 1139.43 Q50.9921 1141.28 50.9921 1143.99 Q50.9921 1148.14 48.1333 1150.42 Q45.2746 1152.69 40.0084 1152.69 Q38.2405 1152.69 36.3597 1152.34 Q34.4977 1152 32.5041 1151.3 L32.5041 1147.63 Q34.0839 1148.56 35.9647 1149.03 Q37.8455 1149.5 39.8955 1149.5 Q43.469 1149.5 45.331 1148.09 Q47.2118 1146.68 47.2118 1143.99 Q47.2118 1141.5 45.4626 1140.11 Q43.7323 1138.7 40.629 1138.7 L37.3565 1138.7 L37.3565 1135.58 L40.7795 1135.58 Q43.5819 1135.58 45.0677 1134.47 Q46.5535 1133.34 46.5535 1131.23 Q46.5535 1129.07 45.0113 1127.92 Q43.4878 1126.76 40.629 1126.76 Q39.068 1126.76 37.2813 1127.1 Q35.4945 1127.44 33.3504 1128.15 L33.3504 1124.76 Q35.5133 1124.16 37.3941 1123.86 Q39.2937 1123.56 40.9676 1123.56 Q45.2934 1123.56 47.8136 1125.54 Q50.3339 1127.49 50.3339 1130.84 Q50.3339 1133.17 48.9985 1134.79 Q47.6632 1136.39 45.1993 1137.01 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M-49.1477 829.723 L-41.5088 829.723 L-41.5088 803.357 L-49.819 805.024 L-49.819 800.765 L-41.5551 799.098 L-36.8792 799.098 L-36.8792 829.723 L-29.2404 829.723 L-29.2404 833.658 L-49.1477 833.658 L-49.1477 829.723 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M-9.79603 802.177 Q-13.4071 802.177 -15.2358 805.742 Q-17.0414 809.283 -17.0414 816.413 Q-17.0414 823.519 -15.2358 827.084 Q-13.4071 830.626 -9.79603 830.626 Q-6.16179 830.626 -4.35625 827.084 Q-2.52755 823.519 -2.52755 816.413 Q-2.52755 809.283 -4.35625 805.742 Q-6.16179 802.177 -9.79603 802.177 M-9.79603 798.473 Q-3.98588 798.473 -0.930339 803.08 Q2.14835 807.663 2.14835 816.413 Q2.14835 825.14 -0.930339 829.746 Q-3.98588 834.33 -9.79603 834.33 Q-15.6062 834.33 -18.6849 829.746 Q-21.7404 825.14 -21.7404 816.413 Q-21.7404 807.663 -18.6849 803.08 Q-15.6062 798.473 -9.79603 798.473 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M2.14835 792.575 L26.2599 792.575 L26.2599 795.772 L2.14835 795.772 L2.14835 792.575 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M37.7326 803.051 L50.9921 803.051 L50.9921 806.248 L33.1624 806.248 L33.1624 803.051 Q35.3253 800.812 39.0492 797.051 Q42.7919 793.27 43.7511 792.18 Q45.5755 790.13 46.2902 788.719 Q47.0237 787.29 47.0237 785.917 Q47.0237 783.678 45.4438 782.268 Q43.8828 780.857 41.3625 780.857 Q39.5758 780.857 37.5822 781.478 Q35.6074 782.099 33.3504 783.359 L33.3504 779.522 Q35.645 778.6 37.6386 778.13 Q39.6322 777.66 41.2873 777.66 Q45.6507 777.66 48.2462 779.842 Q50.8417 782.023 50.8417 785.672 Q50.8417 787.402 50.1834 788.963 Q49.5439 790.506 47.8324 792.612 Q47.3622 793.158 44.842 795.772 Q42.3217 798.367 37.7326 803.051 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M-49.4486 483.822 L-41.8097 483.822 L-41.8097 457.457 L-50.1199 459.123 L-50.1199 454.864 L-41.856 453.197 L-37.1801 453.197 L-37.1801 483.822 L-29.5413 483.822 L-29.5413 487.757 L-49.4486 487.757 L-49.4486 483.822 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M-10.097 456.276 Q-13.708 456.276 -15.5367 459.841 Q-17.3423 463.382 -17.3423 470.512 Q-17.3423 477.618 -15.5367 481.183 Q-13.708 484.725 -10.097 484.725 Q-6.46272 484.725 -4.65717 481.183 Q-2.82848 477.618 -2.82848 470.512 Q-2.82848 463.382 -4.65717 459.841 Q-6.46272 456.276 -10.097 456.276 M-10.097 452.572 Q-4.2868 452.572 -1.23126 457.179 Q1.84742 461.762 1.84742 470.512 Q1.84742 479.239 -1.23126 483.845 Q-4.2868 488.429 -10.097 488.429 Q-15.9071 488.429 -18.9858 483.845 Q-22.0413 479.239 -22.0413 470.512 Q-22.0413 461.762 -18.9858 457.179 Q-15.9071 452.572 -10.097 452.572 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M1.84742 446.674 L25.959 446.674 L25.959 449.871 L1.84742 449.871 L1.84742 446.674 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M34.8174 457.15 L41.024 457.15 L41.024 435.728 L34.272 437.082 L34.272 433.621 L40.9864 432.267 L44.7856 432.267 L44.7856 457.15 L50.9921 457.15 L50.9921 460.347 L34.8174 460.347 L34.8174 457.15 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M-19.7135 137.921 L-12.0747 137.921 L-12.0747 111.556 L-20.3848 113.222 L-20.3848 108.963 L-12.121 107.296 L-7.44506 107.296 L-7.44506 137.921 L0.193787 137.921 L0.193787 141.856 L-19.7135 141.856 L-19.7135 137.921 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M19.6381 110.375 Q16.027 110.375 14.1983 113.94 Q12.3928 117.482 12.3928 124.611 Q12.3928 131.718 14.1983 135.282 Q16.027 138.824 19.6381 138.824 Q23.2724 138.824 25.0779 135.282 Q26.9066 131.718 26.9066 124.611 Q26.9066 117.482 25.0779 113.94 Q23.2724 110.375 19.6381 110.375 M19.6381 106.671 Q25.4483 106.671 28.5038 111.278 Q31.5825 115.861 31.5825 124.611 Q31.5825 133.338 28.5038 137.944 Q25.4483 142.528 19.6381 142.528 Q13.828 142.528 10.7493 137.944 Q7.69375 133.338 7.69375 124.611 Q7.69375 115.861 10.7493 111.278 Q13.828 106.671 19.6381 106.671 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M41.2873 88.8674 Q38.3533 88.8674 36.8675 91.7638 Q35.4005 94.6414 35.4005 100.434 Q35.4005 106.208 36.8675 109.105 Q38.3533 111.982 41.2873 111.982 Q44.2401 111.982 45.7071 109.105 Q47.193 106.208 47.193 100.434 Q47.193 94.6414 45.7071 91.7638 Q44.2401 88.8674 41.2873 88.8674 M41.2873 85.8582 Q46.0081 85.8582 48.4907 89.6009 Q50.9921 93.3249 50.9921 100.434 Q50.9921 107.525 48.4907 111.267 Q46.0081 114.991 41.2873 114.991 Q36.5666 114.991 34.0651 111.267 Q31.5825 107.525 31.5825 100.434 Q31.5825 93.3249 34.0651 89.6009 Q36.5666 85.8582 41.2873 85.8582 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M720.944 12.096 L759.185 12.096 L759.185 18.9825 L729.127 18.9825 L729.127 36.8875 L757.929 36.8875 L757.929 43.7741 L729.127 43.7741 L729.127 65.6895 L759.914 65.6895 L759.914 72.576 L720.944 72.576 L720.944 12.096 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M810.753 45.1919 L810.753 72.576 L803.299 72.576 L803.299 45.4349 Q803.299 38.994 800.788 35.7938 Q798.276 32.5936 793.253 32.5936 Q787.217 32.5936 783.733 36.4419 Q780.25 40.2903 780.25 46.9338 L780.25 72.576 L772.755 72.576 L772.755 27.2059 L780.25 27.2059 L780.25 34.2544 Q782.923 30.163 786.529 28.1376 Q790.174 26.1121 794.914 26.1121 Q802.732 26.1121 806.743 30.9732 Q810.753 35.7938 810.753 45.1919 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M832.992 14.324 L832.992 27.2059 L848.345 27.2059 L848.345 32.9987 L832.992 32.9987 L832.992 57.6282 Q832.992 63.1779 834.491 64.7578 Q836.031 66.3376 840.689 66.3376 L848.345 66.3376 L848.345 72.576 L840.689 72.576 Q832.061 72.576 828.779 69.3758 Q825.498 66.1351 825.498 57.6282 L825.498 32.9987 L820.03 32.9987 L820.03 27.2059 L825.498 27.2059 L825.498 14.324 L832.992 14.324 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M878.768 49.7694 Q869.734 49.7694 866.25 51.8354 Q862.767 53.9013 862.767 58.8839 Q862.767 62.8538 865.359 65.2034 Q867.992 67.5124 872.489 67.5124 Q878.687 67.5124 882.413 63.1374 Q886.181 58.7219 886.181 51.4303 L886.181 49.7694 L878.768 49.7694 M893.634 46.6907 L893.634 72.576 L886.181 72.576 L886.181 65.6895 Q883.629 69.8214 879.821 71.8063 Q876.013 73.7508 870.504 73.7508 Q863.536 73.7508 859.404 69.8619 Q855.313 65.9325 855.313 59.3701 Q855.313 51.7138 860.417 47.825 Q865.562 43.9361 875.729 43.9361 L886.181 43.9361 L886.181 43.2069 Q886.181 38.0623 882.778 35.2672 Q879.416 32.4315 873.299 32.4315 Q869.41 32.4315 865.724 33.3632 Q862.037 34.295 858.635 36.1584 L858.635 29.2718 Q862.726 27.692 866.574 26.9223 Q870.423 26.1121 874.069 26.1121 Q883.912 26.1121 888.773 31.2163 Q893.634 36.3204 893.634 46.6907 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M946.701 45.1919 L946.701 72.576 L939.248 72.576 L939.248 45.4349 Q939.248 38.994 936.736 35.7938 Q934.225 32.5936 929.201 32.5936 Q923.166 32.5936 919.682 36.4419 Q916.198 40.2903 916.198 46.9338 L916.198 72.576 L908.704 72.576 L908.704 27.2059 L916.198 27.2059 L916.198 34.2544 Q918.872 30.163 922.477 28.1376 Q926.123 26.1121 930.862 26.1121 Q938.68 26.1121 942.691 30.9732 Q946.701 35.7938 946.701 45.1919 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M991.423 49.3643 Q991.423 41.2625 988.061 36.8065 Q984.739 32.3505 978.703 32.3505 Q972.708 32.3505 969.346 36.8065 Q966.024 41.2625 966.024 49.3643 Q966.024 57.4256 969.346 61.8816 Q972.708 66.3376 978.703 66.3376 Q984.739 66.3376 988.061 61.8816 Q991.423 57.4256 991.423 49.3643 M998.877 66.9452 Q998.877 78.5308 993.732 84.1616 Q988.588 89.8329 977.974 89.8329 Q974.045 89.8329 970.561 89.2252 Q967.077 88.6581 963.796 87.4428 L963.796 80.1917 Q967.077 81.9741 970.278 82.8248 Q973.478 83.6755 976.8 83.6755 Q984.132 83.6755 987.777 79.8271 Q991.423 76.0193 991.423 68.282 L991.423 64.5957 Q989.114 68.6061 985.509 70.5911 Q981.904 72.576 976.881 72.576 Q968.536 72.576 963.432 66.2161 Q958.327 59.8562 958.327 49.3643 Q958.327 38.832 963.432 32.472 Q968.536 26.1121 976.881 26.1121 Q981.904 26.1121 985.509 28.0971 Q989.114 30.082 991.423 34.0924 L991.423 27.2059 L998.877 27.2059 L998.877 66.9452 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M1014.23 9.54393 L1021.68 9.54393 L1021.68 72.576 L1014.23 72.576 L1014.23 9.54393 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M1076.09 48.0275 L1076.09 51.6733 L1041.82 51.6733 Q1042.3 59.3701 1046.43 63.421 Q1050.61 67.4314 1058.02 67.4314 Q1062.31 67.4314 1066.32 66.3781 Q1070.38 65.3249 1074.35 63.2184 L1074.35 70.267 Q1070.33 71.9684 1066.12 72.8596 Q1061.91 73.7508 1057.57 73.7508 Q1046.72 73.7508 1040.36 67.4314 Q1034.04 61.1119 1034.04 50.3365 Q1034.04 39.1965 1040.03 32.6746 Q1046.07 26.1121 1056.28 26.1121 Q1065.43 26.1121 1070.74 32.0264 Q1076.09 37.9003 1076.09 48.0275 M1068.63 45.84 Q1068.55 39.7232 1065.19 36.0774 Q1061.87 32.4315 1056.36 32.4315 Q1050.12 32.4315 1046.35 35.9558 Q1042.63 39.4801 1042.06 45.8805 L1068.63 45.84 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M1123.64 35.9153 Q1126.44 30.8922 1130.33 28.5022 Q1134.22 26.1121 1139.48 26.1121 Q1146.57 26.1121 1150.42 31.0947 Q1154.27 36.0368 1154.27 45.1919 L1154.27 72.576 L1146.78 72.576 L1146.78 45.4349 Q1146.78 38.913 1144.47 35.7533 Q1142.16 32.5936 1137.42 32.5936 Q1131.63 32.5936 1128.26 36.4419 Q1124.9 40.2903 1124.9 46.9338 L1124.9 72.576 L1117.41 72.576 L1117.41 45.4349 Q1117.41 38.8725 1115.1 35.7533 Q1112.79 32.5936 1107.97 32.5936 Q1102.26 32.5936 1098.89 36.4824 Q1095.53 40.3308 1095.53 46.9338 L1095.53 72.576 L1088.04 72.576 L1088.04 27.2059 L1095.53 27.2059 L1095.53 34.2544 Q1098.08 30.082 1101.65 28.0971 Q1105.21 26.1121 1110.11 26.1121 Q1115.06 26.1121 1118.5 28.6237 Q1121.98 31.1352 1123.64 35.9153 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M1207.94 48.0275 L1207.94 51.6733 L1173.67 51.6733 Q1174.16 59.3701 1178.29 63.421 Q1182.46 67.4314 1189.88 67.4314 Q1194.17 67.4314 1198.18 66.3781 Q1202.23 65.3249 1206.2 63.2184 L1206.2 70.267 Q1202.19 71.9684 1197.98 72.8596 Q1193.77 73.7508 1189.43 73.7508 Q1178.58 73.7508 1172.22 67.4314 Q1165.9 61.1119 1165.9 50.3365 Q1165.9 39.1965 1171.89 32.6746 Q1177.93 26.1121 1188.14 26.1121 Q1197.29 26.1121 1202.6 32.0264 Q1207.94 37.9003 1207.94 48.0275 M1200.49 45.84 Q1200.41 39.7232 1197.05 36.0774 Q1193.73 32.4315 1188.22 32.4315 Q1181.98 32.4315 1178.21 35.9558 Q1174.48 39.4801 1173.92 45.8805 L1200.49 45.84 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M1257.89 45.1919 L1257.89 72.576 L1250.44 72.576 L1250.44 45.4349 Q1250.44 38.994 1247.93 35.7938 Q1245.41 32.5936 1240.39 32.5936 Q1234.36 32.5936 1230.87 36.4419 Q1227.39 40.2903 1227.39 46.9338 L1227.39 72.576 L1219.89 72.576 L1219.89 27.2059 L1227.39 27.2059 L1227.39 34.2544 Q1230.06 30.163 1233.67 28.1376 Q1237.31 26.1121 1242.05 26.1121 Q1249.87 26.1121 1253.88 30.9732 Q1257.89 35.7938 1257.89 45.1919 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M1280.13 14.324 L1280.13 27.2059 L1295.48 27.2059 L1295.48 32.9987 L1280.13 32.9987 L1280.13 57.6282 Q1280.13 63.1779 1281.63 64.7578 Q1283.17 66.3376 1287.83 66.3376 L1295.48 66.3376 L1295.48 72.576 L1287.83 72.576 Q1279.2 72.576 1275.92 69.3758 Q1272.64 66.1351 1272.64 57.6282 L1272.64 32.9987 L1267.17 32.9987 L1267.17 27.2059 L1272.64 27.2059 L1272.64 14.324 L1280.13 14.324 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M1368.24 14.0809 L1368.24 22.0612 Q1363.58 19.8332 1359.45 18.7395 Q1355.32 17.6457 1351.47 17.6457 Q1344.78 17.6457 1341.14 20.2383 Q1337.53 22.8309 1337.53 27.611 Q1337.53 31.6214 1339.92 33.6873 Q1342.35 35.7128 1349.08 36.9686 L1354.02 37.9813 Q1363.17 39.7232 1367.51 44.1387 Q1371.88 48.5136 1371.88 55.8863 Q1371.88 64.6767 1365.97 69.2137 Q1360.1 73.7508 1348.71 73.7508 Q1344.42 73.7508 1339.56 72.7785 Q1334.74 71.8063 1329.55 69.9024 L1329.55 61.4765 Q1334.53 64.2716 1339.31 65.6895 Q1344.09 67.1073 1348.71 67.1073 Q1355.72 67.1073 1359.53 64.3527 Q1363.34 61.598 1363.34 56.4939 Q1363.34 52.0379 1360.58 49.5264 Q1357.87 47.0148 1351.63 45.759 L1346.65 44.7868 Q1337.49 42.9639 1333.4 39.075 Q1329.31 35.1862 1329.31 28.2591 Q1329.31 20.2383 1334.94 15.6203 Q1340.61 11.0023 1350.54 11.0023 Q1354.79 11.0023 1359.2 11.7719 Q1363.62 12.5416 1368.24 14.0809 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M1391.53 65.7705 L1391.53 89.8329 L1384.04 89.8329 L1384.04 27.2059 L1391.53 27.2059 L1391.53 34.0924 Q1393.88 30.0415 1397.45 28.0971 Q1401.05 26.1121 1406.03 26.1121 Q1414.3 26.1121 1419.44 32.6746 Q1424.63 39.2371 1424.63 49.9314 Q1424.63 60.6258 1419.44 67.1883 Q1414.3 73.7508 1406.03 73.7508 Q1401.05 73.7508 1397.45 71.8063 Q1393.88 69.8214 1391.53 65.7705 M1416.89 49.9314 Q1416.89 41.7081 1413.49 37.0496 Q1410.12 32.3505 1404.21 32.3505 Q1398.3 32.3505 1394.89 37.0496 Q1391.53 41.7081 1391.53 49.9314 Q1391.53 58.1548 1394.89 62.8538 Q1398.3 67.5124 1404.21 67.5124 Q1410.12 67.5124 1413.49 62.8538 Q1416.89 58.1548 1416.89 49.9314 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M1475.79 48.0275 L1475.79 51.6733 L1441.52 51.6733 Q1442.01 59.3701 1446.14 63.421 Q1450.31 67.4314 1457.72 67.4314 Q1462.02 67.4314 1466.03 66.3781 Q1470.08 65.3249 1474.05 63.2184 L1474.05 70.267 Q1470.04 71.9684 1465.82 72.8596 Q1461.61 73.7508 1457.28 73.7508 Q1446.42 73.7508 1440.06 67.4314 Q1433.74 61.1119 1433.74 50.3365 Q1433.74 39.1965 1439.74 32.6746 Q1445.77 26.1121 1455.98 26.1121 Q1465.14 26.1121 1470.44 32.0264 Q1475.79 37.9003 1475.79 48.0275 M1468.34 45.84 Q1468.26 39.7232 1464.89 36.0774 Q1461.57 32.4315 1456.06 32.4315 Q1449.82 32.4315 1446.06 35.9558 Q1442.33 39.4801 1441.76 45.8805 L1468.34 45.84 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M1520.67 28.9478 L1520.67 35.9153 Q1517.51 34.1734 1514.31 33.3227 Q1511.15 32.4315 1507.91 32.4315 Q1500.66 32.4315 1496.65 37.0496 Q1492.64 41.6271 1492.64 49.9314 Q1492.64 58.2358 1496.65 62.8538 Q1500.66 67.4314 1507.91 67.4314 Q1511.15 67.4314 1514.31 66.5807 Q1517.51 65.6895 1520.67 63.9476 L1520.67 70.8341 Q1517.55 72.2924 1514.19 73.0216 Q1510.87 73.7508 1507.1 73.7508 Q1496.85 73.7508 1490.82 67.3098 Q1484.78 60.8689 1484.78 49.9314 Q1484.78 38.832 1490.86 32.472 Q1496.98 26.1121 1507.59 26.1121 Q1511.03 26.1121 1514.31 26.8413 Q1517.6 27.5299 1520.67 28.9478 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M1541.01 14.324 L1541.01 27.2059 L1556.36 27.2059 L1556.36 32.9987 L1541.01 32.9987 L1541.01 57.6282 Q1541.01 63.1779 1542.51 64.7578 Q1544.05 66.3376 1548.71 66.3376 L1556.36 66.3376 L1556.36 72.576 L1548.71 72.576 Q1540.08 72.576 1536.8 69.3758 Q1533.52 66.1351 1533.52 57.6282 L1533.52 32.9987 L1528.05 32.9987 L1528.05 27.2059 L1533.52 27.2059 L1533.52 14.324 L1541.01 14.324 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M1592.46 34.1734 Q1591.2 33.4443 1589.7 33.1202 Q1588.24 32.7556 1586.46 32.7556 Q1580.14 32.7556 1576.74 36.8875 Q1573.38 40.9789 1573.38 48.6757 L1573.38 72.576 L1565.88 72.576 L1565.88 27.2059 L1573.38 27.2059 L1573.38 34.2544 Q1575.73 30.1225 1579.49 28.1376 Q1583.26 26.1121 1588.65 26.1121 Q1589.42 26.1121 1590.35 26.2337 Q1591.28 26.3147 1592.42 26.5172 L1592.46 34.1734 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M1599.5 54.671 L1599.5 27.2059 L1606.96 27.2059 L1606.96 54.3874 Q1606.96 60.8284 1609.47 64.0691 Q1611.98 67.2693 1617 67.2693 Q1623.04 67.2693 1626.52 63.421 Q1630.05 59.5726 1630.05 52.9291 L1630.05 27.2059 L1637.5 27.2059 L1637.5 72.576 L1630.05 72.576 L1630.05 65.6084 Q1627.33 69.7404 1623.73 71.7658 Q1620.16 73.7508 1615.42 73.7508 Q1607.61 73.7508 1603.56 68.8897 Q1599.5 64.0286 1599.5 54.671 M1618.26 26.1121 L1618.26 26.1121 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip980)" d="M1688.18 35.9153 Q1690.97 30.8922 1694.86 28.5022 Q1698.75 26.1121 1704.02 26.1121 Q1711.11 26.1121 1714.96 31.0947 Q1718.8 36.0368 1718.8 45.1919 L1718.8 72.576 L1711.31 72.576 L1711.31 45.4349 Q1711.31 38.913 1709 35.7533 Q1706.69 32.5936 1701.95 32.5936 Q1696.16 32.5936 1692.8 36.4419 Q1689.43 40.2903 1689.43 46.9338 L1689.43 72.576 L1681.94 72.576 L1681.94 45.4349 Q1681.94 38.8725 1679.63 35.7533 Q1677.32 32.5936 1672.5 32.5936 Q1666.79 32.5936 1663.43 36.4824 Q1660.07 40.3308 1660.07 46.9338 L1660.07 72.576 L1652.57 72.576 L1652.57 27.2059 L1660.07 27.2059 L1660.07 34.2544 Q1662.62 30.082 1666.18 28.0971 Q1669.75 26.1121 1674.65 26.1121 Q1679.59 26.1121 1683.03 28.6237 Q1686.52 31.1352 1688.18 35.9153 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><circle clip-path="url(#clip982)" cx="364.869" cy="146.527" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="399.767" cy="261.591" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="434.665" cy="300.924" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="469.563" cy="331.803" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="504.462" cy="481.626" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="539.36" cy="524.619" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="574.258" cy="549.62" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="609.156" cy="554.315" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="644.054" cy="666.73" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="678.952" cy="711.419" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="713.851" cy="744.889" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="748.749" cy="746.817" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="783.647" cy="753.041" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="818.545" cy="801.643" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="853.443" cy="803.89" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="888.341" cy="824.742" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="923.24" cy="835.596" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="958.138" cy="839.17" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="993.036" cy="867.909" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="1027.93" cy="891.716" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="1062.83" cy="948.101" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="1097.73" cy="1000.61" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="1132.63" cy="1033.56" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="1167.53" cy="1036.14" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="1202.42" cy="1043.9" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="1237.32" cy="1098.62" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="1272.22" cy="1104.63" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="1307.12" cy="1139.14" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="1342.02" cy="1157.02" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="1376.92" cy="1168.6" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="1411.81" cy="1171.32" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="1446.71" cy="1189.96" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="1481.61" cy="1193.11" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="1516.51" cy="1195.34" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="1551.41" cy="1202.88" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="1586.3" cy="1207.57" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="1621.2" cy="1229.5" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="1656.1" cy="1244.27" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="1691" cy="1275.24" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="1725.9" cy="1283.09" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="1760.8" cy="1327.07" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="1795.69" cy="1341.37" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="1830.59" cy="1380.92" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="1865.49" cy="1388.71" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="1900.39" cy="1404.2" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="1935.29" cy="1411.78" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="1970.18" cy="1424.62" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="2005.08" cy="1436.16" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="2039.98" cy="1446.76" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip982)" cx="2074.88" cy="1464.44" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
</svg>

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
H2 = heisenberg_XXX(ComplexF64, SU2Irrep, InfiniteChain(2); spin=1 // 2);
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
└ @ MPSKit ~/Projects/MPSKit.jl/src/states/infinitemps.jl:149

````

Even though the bond dimension is higher than in the example without symmetry, convergence is reached much faster:

````julia
println(dim(V1))
println(dim(V2))
groundstate, cache, delta = find_groundstate(state, H2,
                                             VUMPS(; maxiter=400, tol=1e-12));
````

````
52
70
[ Info: VUMPS init:	obj = +1.757765749207e-02	err = 3.4286e-01
[ Info: VUMPS   1:	obj = -8.696634007330e-01	err = 1.3207614531e-01	time = 0.21 sec
[ Info: VUMPS   2:	obj = -8.857455218686e-01	err = 7.9112939847e-03	time = 0.12 sec
[ Info: VUMPS   3:	obj = -8.861028581917e-01	err = 3.6033827124e-03	time = 0.03 sec
[ Info: VUMPS   4:	obj = -8.862158212696e-01	err = 1.6926177755e-03	time = 0.03 sec
[ Info: VUMPS   5:	obj = -8.862577312672e-01	err = 1.0611468264e-03	time = 0.03 sec
[ Info: VUMPS   6:	obj = -8.862746027064e-01	err = 7.7042891777e-04	time = 0.03 sec
[ Info: VUMPS   7:	obj = -8.862817300286e-01	err = 6.2709123174e-04	time = 0.04 sec
[ Info: VUMPS   8:	obj = -8.862849391575e-01	err = 5.0493868777e-04	time = 0.05 sec
[ Info: VUMPS   9:	obj = -8.862864498235e-01	err = 3.9930999383e-04	time = 0.03 sec
[ Info: VUMPS  10:	obj = -8.862871831010e-01	err = 3.0800857369e-04	time = 0.03 sec
[ Info: VUMPS  11:	obj = -8.862875441594e-01	err = 2.3263846933e-04	time = 0.05 sec
[ Info: VUMPS  12:	obj = -8.862877229110e-01	err = 1.7276003533e-04	time = 0.10 sec
[ Info: VUMPS  13:	obj = -8.862878114835e-01	err = 1.2669078880e-04	time = 0.05 sec
[ Info: VUMPS  14:	obj = -8.862878553110e-01	err = 9.2023000727e-05	time = 0.13 sec
[ Info: VUMPS  15:	obj = -8.862878769711e-01	err = 6.6386757063e-05	time = 0.05 sec
[ Info: VUMPS  16:	obj = -8.862878876657e-01	err = 4.7658401917e-05	time = 0.05 sec
[ Info: VUMPS  17:	obj = -8.862878929424e-01	err = 3.4092229469e-05	time = 0.05 sec
[ Info: VUMPS  18:	obj = -8.862878955462e-01	err = 2.4325841163e-05	time = 0.05 sec
[ Info: VUMPS  19:	obj = -8.862878968315e-01	err = 1.7324877324e-05	time = 0.06 sec
[ Info: VUMPS  20:	obj = -8.862878974663e-01	err = 1.2323307122e-05	time = 0.06 sec
[ Info: VUMPS  21:	obj = -8.862878977801e-01	err = 8.7563035932e-06	time = 0.05 sec
[ Info: VUMPS  22:	obj = -8.862878979353e-01	err = 6.2166861423e-06	time = 0.04 sec
[ Info: VUMPS  23:	obj = -8.862878980121e-01	err = 4.4107365461e-06	time = 0.04 sec
[ Info: VUMPS  24:	obj = -8.862878980502e-01	err = 3.1277152106e-06	time = 0.04 sec
[ Info: VUMPS  25:	obj = -8.862878980690e-01	err = 2.2170142160e-06	time = 0.11 sec
[ Info: VUMPS  26:	obj = -8.862878980784e-01	err = 1.5707437234e-06	time = 0.05 sec
[ Info: VUMPS  27:	obj = -8.862878980830e-01	err = 1.1124442525e-06	time = 0.04 sec
[ Info: VUMPS  28:	obj = -8.862878980853e-01	err = 7.8759729582e-07	time = 0.05 sec
[ Info: VUMPS  29:	obj = -8.862878980865e-01	err = 5.5743614600e-07	time = 0.06 sec
[ Info: VUMPS  30:	obj = -8.862878980870e-01	err = 3.9442227022e-07	time = 0.05 sec
[ Info: VUMPS  31:	obj = -8.862878980873e-01	err = 2.7900505779e-07	time = 0.04 sec
[ Info: VUMPS  32:	obj = -8.862878980875e-01	err = 1.9731276183e-07	time = 0.06 sec
[ Info: VUMPS  33:	obj = -8.862878980875e-01	err = 1.3950769360e-07	time = 0.06 sec
[ Info: VUMPS  34:	obj = -8.862878980876e-01	err = 9.8611263682e-08	time = 0.11 sec
[ Info: VUMPS  35:	obj = -8.862878980876e-01	err = 6.9693039590e-08	time = 0.05 sec
[ Info: VUMPS  36:	obj = -8.862878980876e-01	err = 4.9246283392e-08	time = 0.05 sec
[ Info: VUMPS  37:	obj = -8.862878980876e-01	err = 3.4792297123e-08	time = 0.05 sec
[ Info: VUMPS  38:	obj = -8.862878980876e-01	err = 2.4576688585e-08	time = 0.04 sec
[ Info: VUMPS  39:	obj = -8.862878980876e-01	err = 1.7357975423e-08	time = 0.05 sec
[ Info: VUMPS  40:	obj = -8.862878980877e-01	err = 1.2257865533e-08	time = 0.04 sec
[ Info: VUMPS  41:	obj = -8.862878980877e-01	err = 8.6551624916e-09	time = 0.06 sec
[ Info: VUMPS  42:	obj = -8.862878980877e-01	err = 6.1106026828e-09	time = 0.08 sec
[ Info: VUMPS  43:	obj = -8.862878980877e-01	err = 4.3136802051e-09	time = 0.06 sec
[ Info: VUMPS  44:	obj = -8.862878980877e-01	err = 3.0449401934e-09	time = 0.05 sec
[ Info: VUMPS  45:	obj = -8.862878980877e-01	err = 2.1490894708e-09	time = 0.06 sec
[ Info: VUMPS  46:	obj = -8.862878980877e-01	err = 1.5166673693e-09	time = 0.06 sec
[ Info: VUMPS  47:	obj = -8.862878980877e-01	err = 1.0702667096e-09	time = 0.06 sec
[ Info: VUMPS  48:	obj = -8.862878980877e-01	err = 7.5519801704e-10	time = 0.05 sec
[ Info: VUMPS  49:	obj = -8.862878980877e-01	err = 5.3284409175e-10	time = 0.08 sec
[ Info: VUMPS  50:	obj = -8.862878980877e-01	err = 3.7593665457e-10	time = 0.05 sec
[ Info: VUMPS  51:	obj = -8.862878980877e-01	err = 2.6521797987e-10	time = 0.04 sec
[ Info: VUMPS  52:	obj = -8.862878980877e-01	err = 1.8709489707e-10	time = 0.05 sec
[ Info: VUMPS  53:	obj = -8.862878980877e-01	err = 1.3197998674e-10	time = 0.05 sec
[ Info: VUMPS  54:	obj = -8.862878980877e-01	err = 9.3098181427e-11	time = 0.05 sec
[ Info: VUMPS  55:	obj = -8.862878980877e-01	err = 6.5667849471e-11	time = 0.05 sec
[ Info: VUMPS  56:	obj = -8.862878980877e-01	err = 4.6315648967e-11	time = 0.04 sec
[ Info: VUMPS  57:	obj = -8.862878980877e-01	err = 3.2665823891e-11	time = 0.10 sec
[ Info: VUMPS  58:	obj = -8.862878980878e-01	err = 2.3038786394e-11	time = 0.03 sec
[ Info: VUMPS  59:	obj = -8.862878980878e-01	err = 1.6249291109e-11	time = 0.05 sec
[ Info: VUMPS  60:	obj = -8.862878980878e-01	err = 1.1461841113e-11	time = 0.04 sec
[ Info: VUMPS  61:	obj = -8.862878980878e-01	err = 8.0811426926e-12	time = 0.03 sec
[ Info: VUMPS  62:	obj = -8.862878980878e-01	err = 5.6989473760e-12	time = 0.03 sec
[ Info: VUMPS  63:	obj = -8.862878980878e-01	err = 4.0199684636e-12	time = 0.02 sec
[ Info: VUMPS  64:	obj = -8.862878980878e-01	err = 2.8323491405e-12	time = 0.03 sec
[ Info: VUMPS  65:	obj = -8.862878980878e-01	err = 1.9985430493e-12	time = 0.03 sec
[ Info: VUMPS  66:	obj = -8.862878980878e-01	err = 1.4098856438e-12	time = 0.07 sec
[ Info: VUMPS conv 67:	obj = -8.862878980878e-01	err = 9.9583426980e-13	time = 3.67 sec

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

