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
single site InfiniteMPOHamiltonian{BlockTensorKit.SparseBlockTensorMap{TensorKit.AbstractTensorMap{ComplexF64, TensorKit.ComplexSpace, 2, 2}, ComplexF64, TensorKit.ComplexSpace, 2, 2, 4}}:
╷  ⋮
┼ W[1]: 3×1×1×3 SparseBlockTensorMap(((ℂ^1 ⊕ ℂ^3 ⊕ ℂ^1) ⊗ ⊕(ℂ^2)) ← (⊕(ℂ^2) ⊗ (ℂ^1 ⊕ ℂ^3 ⊕ ℂ^1)))
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
[ Info: VUMPS init:	obj = +2.499965731027e-01	err = 5.1788e-03
[ Info: VUMPS   1:	obj = -2.870739839017e-01	err = 3.4821394246e-01	time = 0.07 sec
[ Info: VUMPS   2:	obj = -2.656232620998e-01	err = 3.6581492102e-01	time = 0.02 sec
[ Info: VUMPS   3:	obj = -2.487873444302e-01	err = 3.7086723391e-01	time = 0.03 sec
[ Info: VUMPS   4:	obj = -3.355593086868e-01	err = 3.2314100893e-01	time = 0.08 sec
[ Info: VUMPS   5:	obj = -2.087227092608e-01	err = 3.8879465954e-01	time = 0.04 sec
[ Info: VUMPS   6:	obj = -1.686721826675e-01	err = 3.9999392741e-01	time = 0.05 sec
[ Info: VUMPS   7:	obj = -2.122667343815e-01	err = 3.7272920359e-01	time = 0.02 sec
[ Info: VUMPS   8:	obj = -2.958500792621e-01	err = 3.4380638961e-01	time = 0.04 sec
[ Info: VUMPS   9:	obj = -1.691665113603e-01	err = 3.8405954237e-01	time = 0.09 sec
[ Info: VUMPS  10:	obj = -1.346143177404e-01	err = 3.4883977446e-01	time = 0.02 sec
[ Info: VUMPS  11:	obj = -1.092808638757e-01	err = 3.7461357204e-01	time = 0.02 sec
[ Info: VUMPS  12:	obj = -2.991631481281e-01	err = 3.4180870323e-01	time = 0.02 sec
[ Info: VUMPS  13:	obj = -4.041455140465e-01	err = 2.3211278498e-01	time = 0.02 sec
[ Info: VUMPS  14:	obj = -1.211412747111e-01	err = 4.1647647567e-01	time = 0.06 sec
[ Info: VUMPS  15:	obj = -2.509458091277e-01	err = 3.5092248009e-01	time = 0.02 sec
[ Info: VUMPS  16:	obj = -4.071002050793e-01	err = 2.2247393653e-01	time = 0.03 sec
[ Info: VUMPS  17:	obj = -4.021042105328e-01	err = 2.6546423344e-01	time = 0.03 sec
[ Info: VUMPS  18:	obj = -3.358698009449e-01	err = 3.3467169172e-01	time = 0.04 sec
[ Info: VUMPS  19:	obj = -1.524895669284e-01	err = 3.8798422141e-01	time = 0.03 sec
[ Info: VUMPS  20:	obj = -1.265048042203e-01	err = 3.7215805361e-01	time = 0.02 sec
[ Info: VUMPS  21:	obj = -1.770558280076e-01	err = 3.7736964997e-01	time = 0.05 sec
[ Info: VUMPS  22:	obj = -1.758824588510e-01	err = 3.3384682264e-01	time = 0.03 sec
[ Info: VUMPS  23:	obj = -3.552464652427e-01	err = 3.0791686047e-01	time = 0.02 sec
[ Info: VUMPS  24:	obj = -2.973745735943e-01	err = 3.4433290332e-01	time = 0.02 sec
[ Info: VUMPS  25:	obj = -2.126053698440e-02	err = 3.5744948778e-01	time = 0.04 sec
[ Info: VUMPS  26:	obj = -1.665582239538e-01	err = 3.9735917776e-01	time = 0.02 sec
[ Info: VUMPS  27:	obj = -7.756732754444e-02	err = 3.7945618304e-01	time = 0.02 sec
[ Info: VUMPS  28:	obj = -2.817156455223e-01	err = 3.4406533403e-01	time = 0.04 sec
[ Info: VUMPS  29:	obj = -3.230554787336e-01	err = 3.2705394977e-01	time = 0.02 sec
[ Info: VUMPS  30:	obj = -2.018081391878e-01	err = 3.7406357547e-01	time = 0.03 sec
[ Info: VUMPS  31:	obj = -1.786630693499e-01	err = 3.9234894919e-01	time = 0.03 sec
[ Info: VUMPS  32:	obj = -3.529035366014e-01	err = 3.1633431759e-01	time = 0.04 sec
[ Info: VUMPS  33:	obj = -3.702246762737e-01	err = 2.8428281676e-01	time = 0.03 sec
[ Info: VUMPS  34:	obj = -1.602113590704e-01	err = 3.7520993925e-01	time = 0.02 sec
[ Info: VUMPS  35:	obj = -3.388604134509e-01	err = 3.1762942849e-01	time = 0.04 sec
[ Info: VUMPS  36:	obj = -3.723405977388e-01	err = 2.8408621360e-01	time = 0.03 sec
[ Info: VUMPS  37:	obj = -3.745747522467e-01	err = 2.8304308386e-01	time = 0.04 sec
[ Info: VUMPS  38:	obj = -2.440623140951e-01	err = 3.7638448206e-01	time = 0.04 sec
[ Info: VUMPS  39:	obj = -8.072775592287e-02	err = 3.8612921006e-01	time = 0.02 sec
[ Info: VUMPS  40:	obj = -2.763116458716e-01	err = 3.5555871103e-01	time = 0.02 sec
[ Info: VUMPS  41:	obj = +6.862231264718e-02	err = 3.8982985734e-01	time = 0.03 sec
[ Info: VUMPS  42:	obj = -1.805816132678e-01	err = 3.6731056390e-01	time = 0.06 sec
[ Info: VUMPS  43:	obj = -3.371670549031e-01	err = 3.1767372334e-01	time = 0.03 sec
[ Info: VUMPS  44:	obj = -2.873802128753e-01	err = 3.4305615474e-01	time = 0.03 sec
[ Info: VUMPS  45:	obj = -2.828733180854e-01	err = 3.5497882766e-01	time = 0.04 sec
[ Info: VUMPS  46:	obj = -3.151181003076e-01	err = 3.3975637221e-01	time = 0.02 sec
[ Info: VUMPS  47:	obj = -1.937831420783e-01	err = 3.8657017576e-01	time = 0.02 sec
[ Info: VUMPS  48:	obj = -2.425587959565e-01	err = 3.7189637072e-01	time = 0.05 sec
[ Info: VUMPS  49:	obj = -2.049463548852e-01	err = 3.8722815080e-01	time = 0.02 sec
[ Info: VUMPS  50:	obj = -7.564228735972e-02	err = 4.1948815295e-01	time = 0.02 sec
[ Info: VUMPS  51:	obj = -4.707106116205e-02	err = 3.6253733610e-01	time = 0.02 sec
[ Info: VUMPS  52:	obj = -9.650543009270e-02	err = 3.5444792918e-01	time = 0.03 sec
[ Info: VUMPS  53:	obj = -1.331392128923e-01	err = 3.8560843085e-01	time = 0.02 sec
[ Info: VUMPS  54:	obj = -1.124516916412e-02	err = 3.7947301174e-01	time = 0.03 sec
[ Info: VUMPS  55:	obj = -2.418907023334e-01	err = 3.4329126203e-01	time = 0.04 sec
[ Info: VUMPS  56:	obj = -4.151433495252e-01	err = 2.0818325174e-01	time = 0.02 sec
[ Info: VUMPS  57:	obj = -1.786030756917e-01	err = 3.7115660915e-01	time = 0.02 sec
[ Info: VUMPS  58:	obj = -1.960401941740e-01	err = 3.5572579801e-01	time = 0.02 sec
[ Info: VUMPS  59:	obj = -9.436717674245e-02	err = 3.8916533587e-01	time = 0.05 sec
[ Info: VUMPS  60:	obj = -2.228295802945e-02	err = 3.6463933659e-01	time = 0.02 sec
[ Info: VUMPS  61:	obj = -2.194464875714e-01	err = 3.6427298916e-01	time = 0.02 sec
[ Info: VUMPS  62:	obj = -3.743425585067e-02	err = 3.7149006612e-01	time = 0.03 sec
[ Info: VUMPS  63:	obj = -2.350124327064e-01	err = 3.6291145387e-01	time = 0.02 sec
[ Info: VUMPS  64:	obj = -2.739779809191e-01	err = 3.5784309114e-01	time = 0.03 sec
[ Info: VUMPS  65:	obj = -3.412560948766e-01	err = 2.8174464505e-01	time = 0.02 sec
[ Info: VUMPS  66:	obj = -1.758996225495e-01	err = 3.8520609290e-01	time = 0.06 sec
[ Info: VUMPS  67:	obj = -1.699372158198e-01	err = 3.8218080957e-01	time = 0.02 sec
[ Info: VUMPS  68:	obj = -2.497078022290e-01	err = 3.6231171007e-01	time = 0.02 sec
[ Info: VUMPS  69:	obj = -2.229561060642e-02	err = 4.3692162324e-01	time = 0.04 sec
[ Info: VUMPS  70:	obj = -9.716163248661e-02	err = 3.7768230860e-01	time = 0.02 sec
[ Info: VUMPS  71:	obj = -2.105380545631e-01	err = 3.6840628633e-01	time = 0.03 sec
[ Info: VUMPS  72:	obj = -1.431191811942e-01	err = 4.1983529888e-01	time = 0.04 sec
[ Info: VUMPS  73:	obj = -2.456278256482e-01	err = 3.7213774591e-01	time = 0.03 sec
[ Info: VUMPS  74:	obj = +1.660831572369e-02	err = 3.7310912630e-01	time = 0.02 sec
[ Info: VUMPS  75:	obj = -1.294046229502e-01	err = 3.8966645809e-01	time = 0.02 sec
[ Info: VUMPS  76:	obj = -1.346478227103e-01	err = 4.0775279451e-01	time = 0.06 sec
[ Info: VUMPS  77:	obj = -1.434894134037e-01	err = 3.8830957632e-01	time = 0.02 sec
[ Info: VUMPS  78:	obj = -8.436938171198e-02	err = 3.9453735993e-01	time = 0.02 sec
[ Info: VUMPS  79:	obj = -1.568793636450e-01	err = 3.8139578446e-01	time = 0.03 sec
[ Info: VUMPS  80:	obj = -2.509613995100e-01	err = 3.4621772914e-01	time = 0.02 sec
[ Info: VUMPS  81:	obj = -3.071362100710e-01	err = 3.3272198605e-01	time = 0.04 sec
[ Info: VUMPS  82:	obj = -2.489354707357e-01	err = 3.6535737689e-01	time = 0.04 sec
[ Info: VUMPS  83:	obj = -3.233159718003e-01	err = 3.2698048226e-01	time = 0.04 sec
[ Info: VUMPS  84:	obj = -2.357289864043e-01	err = 3.6156474727e-01	time = 0.03 sec
[ Info: VUMPS  85:	obj = -2.890021533721e-01	err = 3.1412109066e-01	time = 0.03 sec
[ Info: VUMPS  86:	obj = -3.253485615272e-01	err = 3.3031102623e-01	time = 0.04 sec
[ Info: VUMPS  87:	obj = -3.281142405411e-01	err = 3.2969999722e-01	time = 0.04 sec
[ Info: VUMPS  88:	obj = -1.359597453577e-01	err = 3.8793501932e-01	time = 0.04 sec
[ Info: VUMPS  89:	obj = -6.631801194315e-03	err = 3.7340996710e-01	time = 0.02 sec
[ Info: VUMPS  90:	obj = +1.832809680496e-02	err = 3.5792326056e-01	time = 0.03 sec
[ Info: VUMPS  91:	obj = -6.856316008337e-02	err = 3.5994692545e-01	time = 0.02 sec
[ Info: VUMPS  92:	obj = -2.608632232008e-01	err = 3.6605391192e-01	time = 0.03 sec
[ Info: VUMPS  93:	obj = -1.670986937074e-01	err = 3.7964023113e-01	time = 0.02 sec
[ Info: VUMPS  94:	obj = -2.651431497593e-01	err = 3.5258576130e-01	time = 0.02 sec
[ Info: VUMPS  95:	obj = -2.706093035257e-01	err = 3.7025672893e-01	time = 0.05 sec
[ Info: VUMPS  96:	obj = -3.768785600541e-01	err = 2.9383773407e-01	time = 0.10 sec
[ Info: VUMPS  97:	obj = +5.342752195057e-02	err = 3.8002070543e-01	time = 0.02 sec
[ Info: VUMPS  98:	obj = -1.838597150381e-01	err = 3.7961859777e-01	time = 0.05 sec
[ Info: VUMPS  99:	obj = -1.969454185642e-02	err = 3.8200442771e-01	time = 0.02 sec
[ Info: VUMPS 100:	obj = -1.441225062613e-01	err = 3.8706264330e-01	time = 0.02 sec
[ Info: VUMPS 101:	obj = -2.446765603311e-01	err = 3.6532621105e-01	time = 0.03 sec
[ Info: VUMPS 102:	obj = -2.204402257144e-01	err = 3.7315648492e-01	time = 0.02 sec
[ Info: VUMPS 103:	obj = -1.431272767610e-01	err = 3.7376490092e-01	time = 0.07 sec
[ Info: VUMPS 104:	obj = -1.745092283897e-01	err = 3.6882026711e-01	time = 0.03 sec
[ Info: VUMPS 105:	obj = -2.701937658952e-01	err = 3.7388543029e-01	time = 0.02 sec
[ Info: VUMPS 106:	obj = -9.291696774226e-02	err = 4.2850757556e-01	time = 0.02 sec
[ Info: VUMPS 107:	obj = +6.504133167110e-03	err = 3.5675440826e-01	time = 0.02 sec
[ Info: VUMPS 108:	obj = -2.138818248344e-01	err = 3.6354195786e-01	time = 0.02 sec
[ Info: VUMPS 109:	obj = -3.093345419137e-01	err = 3.3070663384e-01	time = 0.05 sec
[ Info: VUMPS 110:	obj = -8.268913071079e-02	err = 3.1550696115e-01	time = 0.02 sec
[ Info: VUMPS 111:	obj = -2.107997932376e-01	err = 3.6503535129e-01	time = 0.02 sec
[ Info: VUMPS 112:	obj = +3.958261022456e-02	err = 3.8220064662e-01	time = 0.05 sec
[ Info: VUMPS 113:	obj = -3.653049942074e-02	err = 3.5885348161e-01	time = 0.02 sec
[ Info: VUMPS 114:	obj = -3.176006942346e-03	err = 3.6522463390e-01	time = 0.02 sec
[ Info: VUMPS 115:	obj = +1.297902475997e-02	err = 3.8081187995e-01	time = 0.03 sec
[ Info: VUMPS 116:	obj = -2.257182864704e-01	err = 3.7847439472e-01	time = 0.03 sec
[ Info: VUMPS 117:	obj = -3.783877829550e-01	err = 2.8575736900e-01	time = 0.02 sec
[ Info: VUMPS 118:	obj = -3.596364279567e-01	err = 2.9860727922e-01	time = 0.03 sec
[ Info: VUMPS 119:	obj = -2.160654550015e-01	err = 3.6037175690e-01	time = 0.05 sec
[ Info: VUMPS 120:	obj = -2.998281716528e-01	err = 3.5085343179e-01	time = 0.02 sec
[ Info: VUMPS 121:	obj = -1.253233018610e-01	err = 4.1186792204e-01	time = 0.04 sec
[ Info: VUMPS 122:	obj = -8.495205449513e-02	err = 3.9116475794e-01	time = 0.04 sec
[ Info: VUMPS 123:	obj = -1.239031607531e-01	err = 4.0538357627e-01	time = 0.02 sec
[ Info: VUMPS 124:	obj = -7.476131348428e-02	err = 3.5680372948e-01	time = 0.02 sec
[ Info: VUMPS 125:	obj = -1.235003950709e-01	err = 3.9220147366e-01	time = 0.02 sec
[ Info: VUMPS 126:	obj = -2.142196900619e-01	err = 3.7704232167e-01	time = 0.06 sec
[ Info: VUMPS 127:	obj = -2.645003835192e-01	err = 3.5928863949e-01	time = 0.03 sec
[ Info: VUMPS 128:	obj = -3.730167168070e-01	err = 2.5026625392e-01	time = 0.02 sec
[ Info: VUMPS 129:	obj = -1.804106020775e-01	err = 3.7738108204e-01	time = 0.05 sec
[ Info: VUMPS 130:	obj = -1.863936298624e-01	err = 3.8284559321e-01	time = 0.03 sec
[ Info: VUMPS 131:	obj = -2.608923525741e-01	err = 3.6106626386e-01	time = 0.02 sec
[ Info: VUMPS 132:	obj = -2.994754938380e-01	err = 3.3311683635e-01	time = 0.03 sec
[ Info: VUMPS 133:	obj = -5.788570503693e-02	err = 4.0709722403e-01	time = 0.03 sec
[ Info: VUMPS 134:	obj = +2.913059661603e-02	err = 3.6531531721e-01	time = 0.02 sec
[ Info: VUMPS 135:	obj = -2.081011636612e-01	err = 3.6392169999e-01	time = 0.04 sec
[ Info: VUMPS 136:	obj = -1.603476385814e-01	err = 3.6771314632e-01	time = 0.02 sec
[ Info: VUMPS 137:	obj = -1.636479309807e-01	err = 3.7125737211e-01	time = 0.02 sec
[ Info: VUMPS 138:	obj = -2.750467527686e-01	err = 3.2863136849e-01	time = 0.02 sec
[ Info: VUMPS 139:	obj = -2.572103038456e-01	err = 3.6211867903e-01	time = 0.04 sec
[ Info: VUMPS 140:	obj = -2.665360022075e-01	err = 3.7940809379e-01	time = 0.03 sec
[ Info: VUMPS 141:	obj = -2.593337474320e-01	err = 3.6454803784e-01	time = 0.02 sec
[ Info: VUMPS 142:	obj = -3.411862689499e-01	err = 3.1808026920e-01	time = 0.04 sec
[ Info: VUMPS 143:	obj = -2.205125208998e-01	err = 3.7092175963e-01	time = 0.03 sec
[ Info: VUMPS 144:	obj = -9.963125397700e-02	err = 3.8898556137e-01	time = 0.03 sec
[ Info: VUMPS 145:	obj = -1.681774371269e-01	err = 3.9335580811e-01	time = 0.05 sec
[ Info: VUMPS 146:	obj = -9.287161404206e-02	err = 3.6603524853e-01	time = 0.02 sec
[ Info: VUMPS 147:	obj = -1.207255601478e-01	err = 4.0912304096e-01	time = 0.02 sec
[ Info: VUMPS 148:	obj = +8.790587245207e-02	err = 3.8342414897e-01	time = 0.04 sec
[ Info: VUMPS 149:	obj = +4.241349649169e-02	err = 4.0748001878e-01	time = 0.02 sec
[ Info: VUMPS 150:	obj = -7.338008049140e-02	err = 4.0063354761e-01	time = 0.02 sec
[ Info: VUMPS 151:	obj = +7.002734934922e-02	err = 3.5204488593e-01	time = 0.02 sec
[ Info: VUMPS 152:	obj = -3.083362829704e-01	err = 3.3188484423e-01	time = 0.04 sec
[ Info: VUMPS 153:	obj = -3.675718114070e-01	err = 2.8359701887e-01	time = 0.03 sec
[ Info: VUMPS 154:	obj = -4.071243488800e-01	err = 2.2886009056e-01	time = 0.03 sec
[ Info: VUMPS 155:	obj = +8.104308654096e-02	err = 4.2286813728e-01	time = 0.04 sec
[ Info: VUMPS 156:	obj = -4.108264081814e-02	err = 3.9576637883e-01	time = 0.02 sec
[ Info: VUMPS 157:	obj = -2.228817853977e-01	err = 3.7200506666e-01	time = 0.02 sec
[ Info: VUMPS 158:	obj = -2.992198820305e-01	err = 3.5158845413e-01	time = 0.04 sec
[ Info: VUMPS 159:	obj = -1.686415612476e-01	err = 3.7247278990e-01	time = 0.03 sec
[ Info: VUMPS 160:	obj = -1.152367792890e-01	err = 3.9917974018e-01	time = 0.02 sec
[ Info: VUMPS 161:	obj = -5.129393700677e-02	err = 3.9558994709e-01	time = 0.04 sec
[ Info: VUMPS 162:	obj = -2.199516890574e-01	err = 3.6782710459e-01	time = 0.02 sec
[ Info: VUMPS 163:	obj = -1.374857584824e-01	err = 3.8304471567e-01	time = 0.02 sec
[ Info: VUMPS 164:	obj = -2.956465748635e-01	err = 3.3927124072e-01	time = 0.02 sec
[ Info: VUMPS 165:	obj = -3.595732073797e-01	err = 3.0506560533e-01	time = 0.06 sec
[ Info: VUMPS 166:	obj = -3.325684373442e-01	err = 3.2318606295e-01	time = 0.03 sec
[ Info: VUMPS 167:	obj = -1.905447843258e-01	err = 4.1110901471e-01	time = 0.04 sec
[ Info: VUMPS 168:	obj = -2.669703131832e-01	err = 3.6976363194e-01	time = 0.03 sec
[ Info: VUMPS 169:	obj = -1.597064130916e-01	err = 3.6921850572e-01	time = 0.02 sec
[ Info: VUMPS 170:	obj = -1.076798062759e-01	err = 3.9137688080e-01	time = 0.02 sec
[ Info: VUMPS 171:	obj = -1.495699984471e-01	err = 3.9462681427e-01	time = 0.04 sec
[ Info: VUMPS 172:	obj = -3.056242070874e-01	err = 3.3910827717e-01	time = 0.03 sec
[ Info: VUMPS 173:	obj = -2.149102529179e-01	err = 3.7052305875e-01	time = 0.02 sec
[ Info: VUMPS 174:	obj = -3.100783748777e-01	err = 3.3925710609e-01	time = 0.03 sec
[ Info: VUMPS 175:	obj = -1.982933749807e-01	err = 3.8222445788e-01	time = 0.03 sec
[ Info: VUMPS 176:	obj = -3.747586662326e-01	err = 2.9445949733e-01	time = 0.02 sec
[ Info: VUMPS 177:	obj = -4.336548038953e-01	err = 1.2024784368e-01	time = 0.05 sec
[ Info: VUMPS 178:	obj = -3.303643165085e-01	err = 3.4720074617e-01	time = 0.03 sec
[ Info: VUMPS 179:	obj = -1.844376842541e-01	err = 4.0787730097e-01	time = 0.03 sec
[ Info: VUMPS 180:	obj = -8.308432235064e-02	err = 3.4796859272e-01	time = 0.04 sec
[ Info: VUMPS 181:	obj = -3.005047543784e-01	err = 3.3064737468e-01	time = 0.03 sec
[ Info: VUMPS 182:	obj = -3.790757971290e-01	err = 2.6638705368e-01	time = 0.03 sec
[ Info: VUMPS 183:	obj = -3.126399844497e-01	err = 3.6411380953e-01	time = 0.06 sec
[ Info: VUMPS 184:	obj = -2.913111452974e-01	err = 3.4128172182e-01	time = 0.04 sec
[ Info: VUMPS 185:	obj = -3.787535706892e-01	err = 2.8558578489e-01	time = 0.04 sec
[ Info: VUMPS 186:	obj = -1.517299512336e-01	err = 3.9021203067e-01	time = 0.05 sec
[ Info: VUMPS 187:	obj = -3.238252514845e-01	err = 3.5845397916e-01	time = 0.04 sec
[ Info: VUMPS 188:	obj = -3.838215783698e-01	err = 2.8427969456e-01	time = 0.06 sec
[ Info: VUMPS 189:	obj = -3.604325689641e-01	err = 3.0848700945e-01	time = 0.04 sec
[ Info: VUMPS 190:	obj = -1.766620216419e-01	err = 3.9605496185e-01	time = 0.04 sec
[ Info: VUMPS 191:	obj = -3.254438660309e-01	err = 3.3887948532e-01	time = 0.05 sec
[ Info: VUMPS 192:	obj = -5.932188106595e-02	err = 4.0571620185e-01	time = 0.03 sec
[ Info: VUMPS 193:	obj = -3.373680767382e-02	err = 4.2662944124e-01	time = 0.02 sec
[ Info: VUMPS 194:	obj = -1.413975733022e-01	err = 4.0245218432e-01	time = 0.03 sec
[ Info: VUMPS 195:	obj = -2.568746931960e-01	err = 3.7519798391e-01	time = 0.05 sec
[ Info: VUMPS 196:	obj = +9.058385119732e-02	err = 3.3634529804e-01	time = 0.02 sec
[ Info: VUMPS 197:	obj = -4.005399235058e-02	err = 3.8729320160e-01	time = 0.03 sec
[ Info: VUMPS 198:	obj = -1.799865406032e-01	err = 3.8680476800e-01	time = 0.04 sec
[ Info: VUMPS 199:	obj = -2.761370055456e-01	err = 3.4505915073e-01	time = 0.02 sec
┌ Warning: VUMPS cancel 200:	obj = -3.540268003523e-01	err = 3.1337698402e-01	time = 6.34 sec
└ @ MPSKit ~/Projects/MPSKit.jl-1/src/algorithms/groundstate/vumps.jl:71

````

As you can see, VUMPS struggles to converge.
On it's own, that is already quite curious.
Maybe we can do better using another algorithm, such as gradient descent.

````julia
groundstate, cache, delta = find_groundstate(state, H, GradientGrassmann(; maxiter=20));
````

````
[ Info: CG: initializing with f = 0.249996573103, ‖∇f‖ = 3.6623e-03
┌ Warning: resorting to η
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/cg.jl:207
┌ Warning: CG: not converged to requested tol after 20 iterations and time 3.89 s: f = -0.442870660226, ‖∇f‖ = 2.9059e-03
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
  <clipPath id="clip500">
    <rect x="0" y="0" width="2400" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip500)" d="M0 1600 L2400 1600 L2400 0 L0 0  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip501">
    <rect x="480" y="0" width="1681" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip500)" d="M249.542 1423.18 L2352.76 1423.18 L2352.76 47.2441 L249.542 47.2441  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip502">
    <rect x="249" y="47" width="2104" height="1377"/>
  </clipPath>
</defs>
<polyline clip-path="url(#clip502)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="309.067,1423.18 309.067,47.2441 "/>
<polyline clip-path="url(#clip502)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="639.761,1423.18 639.761,47.2441 "/>
<polyline clip-path="url(#clip502)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="970.455,1423.18 970.455,47.2441 "/>
<polyline clip-path="url(#clip502)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1301.15,1423.18 1301.15,47.2441 "/>
<polyline clip-path="url(#clip502)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1631.84,1423.18 1631.84,47.2441 "/>
<polyline clip-path="url(#clip502)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1962.54,1423.18 1962.54,47.2441 "/>
<polyline clip-path="url(#clip502)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="2293.23,1423.18 2293.23,47.2441 "/>
<polyline clip-path="url(#clip502)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="249.542,1304.27 2352.76,1304.27 "/>
<polyline clip-path="url(#clip502)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="249.542,1052.87 2352.76,1052.87 "/>
<polyline clip-path="url(#clip502)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="249.542,801.462 2352.76,801.462 "/>
<polyline clip-path="url(#clip502)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="249.542,550.056 2352.76,550.056 "/>
<polyline clip-path="url(#clip502)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="249.542,298.65 2352.76,298.65 "/>
<polyline clip-path="url(#clip500)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="249.542,1423.18 2352.76,1423.18 "/>
<polyline clip-path="url(#clip500)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="309.067,1423.18 309.067,1404.28 "/>
<polyline clip-path="url(#clip500)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="639.761,1423.18 639.761,1404.28 "/>
<polyline clip-path="url(#clip500)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="970.455,1423.18 970.455,1404.28 "/>
<polyline clip-path="url(#clip500)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1301.15,1423.18 1301.15,1404.28 "/>
<polyline clip-path="url(#clip500)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1631.84,1423.18 1631.84,1404.28 "/>
<polyline clip-path="url(#clip500)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1962.54,1423.18 1962.54,1404.28 "/>
<polyline clip-path="url(#clip500)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="2293.23,1423.18 2293.23,1404.28 "/>
<path clip-path="url(#clip500)" d="M262.829 1454.1 Q259.218 1454.1 257.389 1457.66 Q255.583 1461.2 255.583 1468.33 Q255.583 1475.44 257.389 1479.01 Q259.218 1482.55 262.829 1482.55 Q266.463 1482.55 268.269 1479.01 Q270.097 1475.44 270.097 1468.33 Q270.097 1461.2 268.269 1457.66 Q266.463 1454.1 262.829 1454.1 M262.829 1450.39 Q268.639 1450.39 271.694 1455 Q274.773 1459.58 274.773 1468.33 Q274.773 1477.06 271.694 1481.67 Q268.639 1486.25 262.829 1486.25 Q257.019 1486.25 253.94 1481.67 Q250.884 1477.06 250.884 1468.33 Q250.884 1459.58 253.94 1455 Q257.019 1450.39 262.829 1450.39 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M289.958 1451.02 L293.893 1451.02 L281.856 1489.98 L277.921 1489.98 L289.958 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M305.93 1451.02 L309.866 1451.02 L297.829 1489.98 L293.893 1489.98 L305.93 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M315.745 1481.64 L323.384 1481.64 L323.384 1455.28 L315.074 1456.95 L315.074 1452.69 L323.338 1451.02 L328.014 1451.02 L328.014 1481.64 L335.652 1481.64 L335.652 1485.58 L315.745 1485.58 L315.745 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M341.74 1459.65 L366.578 1459.65 L366.578 1463.91 L363.314 1463.91 L363.314 1479.84 Q363.314 1481.51 363.87 1482.25 Q364.449 1482.96 365.722 1482.96 Q366.069 1482.96 366.578 1482.92 Q367.087 1482.85 367.25 1482.83 L367.25 1485.9 Q366.439 1486.2 365.583 1486.34 Q364.726 1486.48 363.87 1486.48 Q361.092 1486.48 360.027 1484.98 Q358.963 1483.45 358.963 1479.38 L358.963 1463.91 L349.402 1463.91 L349.402 1485.58 L345.051 1485.58 L345.051 1463.91 L341.74 1463.91 L341.74 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M583.291 1481.64 L590.93 1481.64 L590.93 1455.28 L582.62 1456.95 L582.62 1452.69 L590.884 1451.02 L595.56 1451.02 L595.56 1481.64 L603.199 1481.64 L603.199 1485.58 L583.291 1485.58 L583.291 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M619.611 1451.02 L623.546 1451.02 L611.509 1489.98 L607.574 1489.98 L619.611 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M635.583 1451.02 L639.518 1451.02 L627.481 1489.98 L623.546 1489.98 L635.583 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M658.754 1466.95 Q662.11 1467.66 663.985 1469.93 Q665.884 1472.2 665.884 1475.53 Q665.884 1480.65 662.365 1483.45 Q658.847 1486.25 652.365 1486.25 Q650.189 1486.25 647.874 1485.81 Q645.583 1485.39 643.129 1484.54 L643.129 1480.02 Q645.073 1481.16 647.388 1481.74 Q649.703 1482.32 652.226 1482.32 Q656.624 1482.32 658.916 1480.58 Q661.231 1478.84 661.231 1475.53 Q661.231 1472.48 659.078 1470.77 Q656.948 1469.03 653.129 1469.03 L649.101 1469.03 L649.101 1465.19 L653.314 1465.19 Q656.763 1465.19 658.592 1463.82 Q660.421 1462.43 660.421 1459.84 Q660.421 1457.18 658.522 1455.77 Q656.647 1454.33 653.129 1454.33 Q651.208 1454.33 649.009 1454.75 Q646.81 1455.16 644.171 1456.04 L644.171 1451.88 Q646.833 1451.14 649.147 1450.77 Q651.485 1450.39 653.546 1450.39 Q658.87 1450.39 661.971 1452.83 Q665.073 1455.23 665.073 1459.35 Q665.073 1462.22 663.43 1464.21 Q661.786 1466.18 658.754 1466.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M671.393 1459.65 L696.231 1459.65 L696.231 1463.91 L692.967 1463.91 L692.967 1479.84 Q692.967 1481.51 693.522 1482.25 Q694.101 1482.96 695.374 1482.96 Q695.721 1482.96 696.231 1482.92 Q696.74 1482.85 696.902 1482.83 L696.902 1485.9 Q696.092 1486.2 695.235 1486.34 Q694.379 1486.48 693.522 1486.48 Q690.744 1486.48 689.68 1484.98 Q688.615 1483.45 688.615 1479.38 L688.615 1463.91 L679.055 1463.91 L679.055 1485.58 L674.703 1485.58 L674.703 1463.91 L671.393 1463.91 L671.393 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M918.071 1481.64 L934.39 1481.64 L934.39 1485.58 L912.446 1485.58 L912.446 1481.64 Q915.108 1478.89 919.691 1474.26 Q924.298 1469.61 925.478 1468.27 Q927.724 1465.74 928.603 1464.01 Q929.506 1462.25 929.506 1460.56 Q929.506 1457.8 927.562 1456.07 Q925.64 1454.33 922.539 1454.33 Q920.339 1454.33 917.886 1455.09 Q915.455 1455.86 912.677 1457.41 L912.677 1452.69 Q915.502 1451.55 917.955 1450.97 Q920.409 1450.39 922.446 1450.39 Q927.816 1450.39 931.011 1453.08 Q934.205 1455.77 934.205 1460.26 Q934.205 1462.39 933.395 1464.31 Q932.608 1466.2 930.501 1468.8 Q929.923 1469.47 926.821 1472.69 Q923.719 1475.88 918.071 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M951.173 1451.02 L955.108 1451.02 L943.071 1489.98 L939.136 1489.98 L951.173 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M967.145 1451.02 L971.08 1451.02 L959.043 1489.98 L955.108 1489.98 L967.145 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M990.316 1466.95 Q993.672 1467.66 995.547 1469.93 Q997.446 1472.2 997.446 1475.53 Q997.446 1480.65 993.927 1483.45 Q990.409 1486.25 983.927 1486.25 Q981.751 1486.25 979.436 1485.81 Q977.145 1485.39 974.691 1484.54 L974.691 1480.02 Q976.635 1481.16 978.95 1481.74 Q981.265 1482.32 983.788 1482.32 Q988.186 1482.32 990.478 1480.58 Q992.793 1478.84 992.793 1475.53 Q992.793 1472.48 990.64 1470.77 Q988.51 1469.03 984.691 1469.03 L980.663 1469.03 L980.663 1465.19 L984.876 1465.19 Q988.325 1465.19 990.154 1463.82 Q991.983 1462.43 991.983 1459.84 Q991.983 1457.18 990.084 1455.77 Q988.209 1454.33 984.691 1454.33 Q982.77 1454.33 980.571 1454.75 Q978.372 1455.16 975.733 1456.04 L975.733 1451.88 Q978.395 1451.14 980.71 1450.77 Q983.047 1450.39 985.108 1450.39 Q990.432 1450.39 993.534 1452.83 Q996.635 1455.23 996.635 1459.35 Q996.635 1462.22 994.992 1464.21 Q993.348 1466.18 990.316 1466.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M1002.95 1459.65 L1027.79 1459.65 L1027.79 1463.91 L1024.53 1463.91 L1024.53 1479.84 Q1024.53 1481.51 1025.08 1482.25 Q1025.66 1482.96 1026.94 1482.96 Q1027.28 1482.96 1027.79 1482.92 Q1028.3 1482.85 1028.46 1482.83 L1028.46 1485.9 Q1027.65 1486.2 1026.8 1486.34 Q1025.94 1486.48 1025.08 1486.48 Q1022.31 1486.48 1021.24 1484.98 Q1020.18 1483.45 1020.18 1479.38 L1020.18 1463.91 L1010.62 1463.91 L1010.62 1485.58 L1006.26 1485.58 L1006.26 1463.91 L1002.95 1463.91 L1002.95 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M1244.68 1481.64 L1252.32 1481.64 L1252.32 1455.28 L1244.01 1456.95 L1244.01 1452.69 L1252.27 1451.02 L1256.95 1451.02 L1256.95 1481.64 L1264.59 1481.64 L1264.59 1485.58 L1244.68 1485.58 L1244.68 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M1281 1451.02 L1284.93 1451.02 L1272.9 1489.98 L1268.96 1489.98 L1281 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M1296.97 1451.02 L1300.91 1451.02 L1288.87 1489.98 L1284.93 1489.98 L1296.97 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M1306.79 1481.64 L1314.42 1481.64 L1314.42 1455.28 L1306.11 1456.95 L1306.11 1452.69 L1314.38 1451.02 L1319.05 1451.02 L1319.05 1481.64 L1326.69 1481.64 L1326.69 1485.58 L1306.79 1485.58 L1306.79 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M1332.78 1459.65 L1357.62 1459.65 L1357.62 1463.91 L1354.35 1463.91 L1354.35 1479.84 Q1354.35 1481.51 1354.91 1482.25 Q1355.49 1482.96 1356.76 1482.96 Q1357.11 1482.96 1357.62 1482.92 Q1358.13 1482.85 1358.29 1482.83 L1358.29 1485.9 Q1357.48 1486.2 1356.62 1486.34 Q1355.77 1486.48 1354.91 1486.48 Q1352.13 1486.48 1351.07 1484.98 Q1350 1483.45 1350 1479.38 L1350 1463.91 L1340.44 1463.91 L1340.44 1485.58 L1336.09 1485.58 L1336.09 1463.91 L1332.78 1463.91 L1332.78 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M1588.86 1455.09 L1577.05 1473.54 L1588.86 1473.54 L1588.86 1455.09 M1587.63 1451.02 L1593.51 1451.02 L1593.51 1473.54 L1598.44 1473.54 L1598.44 1477.43 L1593.51 1477.43 L1593.51 1485.58 L1588.86 1485.58 L1588.86 1477.43 L1573.26 1477.43 L1573.26 1472.92 L1587.63 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M1613.14 1451.02 L1617.07 1451.02 L1605.04 1489.98 L1601.1 1489.98 L1613.14 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M1629.11 1451.02 L1633.05 1451.02 L1621.01 1489.98 L1617.07 1489.98 L1629.11 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M1652.28 1466.95 Q1655.64 1467.66 1657.51 1469.93 Q1659.41 1472.2 1659.41 1475.53 Q1659.41 1480.65 1655.89 1483.45 Q1652.38 1486.25 1645.89 1486.25 Q1643.72 1486.25 1641.4 1485.81 Q1639.11 1485.39 1636.66 1484.54 L1636.66 1480.02 Q1638.6 1481.16 1640.92 1481.74 Q1643.23 1482.32 1645.75 1482.32 Q1650.15 1482.32 1652.44 1480.58 Q1654.76 1478.84 1654.76 1475.53 Q1654.76 1472.48 1652.61 1470.77 Q1650.48 1469.03 1646.66 1469.03 L1642.63 1469.03 L1642.63 1465.19 L1646.84 1465.19 Q1650.29 1465.19 1652.12 1463.82 Q1653.95 1462.43 1653.95 1459.84 Q1653.95 1457.18 1652.05 1455.77 Q1650.18 1454.33 1646.66 1454.33 Q1644.74 1454.33 1642.54 1454.75 Q1640.34 1455.16 1637.7 1456.04 L1637.7 1451.88 Q1640.36 1451.14 1642.68 1450.77 Q1645.01 1450.39 1647.07 1450.39 Q1652.4 1450.39 1655.5 1452.83 Q1658.6 1455.23 1658.6 1459.35 Q1658.6 1462.22 1656.96 1464.21 Q1655.32 1466.18 1652.28 1466.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M1664.92 1459.65 L1689.76 1459.65 L1689.76 1463.91 L1686.5 1463.91 L1686.5 1479.84 Q1686.5 1481.51 1687.05 1482.25 Q1687.63 1482.96 1688.9 1482.96 Q1689.25 1482.96 1689.76 1482.92 Q1690.27 1482.85 1690.43 1482.83 L1690.43 1485.9 Q1689.62 1486.2 1688.76 1486.34 Q1687.91 1486.48 1687.05 1486.48 Q1684.27 1486.48 1683.21 1484.98 Q1682.14 1483.45 1682.14 1479.38 L1682.14 1463.91 L1672.58 1463.91 L1672.58 1485.58 L1668.23 1485.58 L1668.23 1463.91 L1664.92 1463.91 L1664.92 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M1906.08 1451.02 L1924.44 1451.02 L1924.44 1454.96 L1910.36 1454.96 L1910.36 1463.43 Q1911.38 1463.08 1912.4 1462.92 Q1913.42 1462.73 1914.44 1462.73 Q1920.22 1462.73 1923.6 1465.9 Q1926.98 1469.08 1926.98 1474.49 Q1926.98 1480.07 1923.51 1483.17 Q1920.04 1486.25 1913.72 1486.25 Q1911.54 1486.25 1909.27 1485.88 Q1907.03 1485.51 1904.62 1484.77 L1904.62 1480.07 Q1906.7 1481.2 1908.93 1481.76 Q1911.15 1482.32 1913.63 1482.32 Q1917.63 1482.32 1919.97 1480.21 Q1922.31 1478.1 1922.31 1474.49 Q1922.31 1470.88 1919.97 1468.77 Q1917.63 1466.67 1913.63 1466.67 Q1911.75 1466.67 1909.88 1467.08 Q1908.02 1467.5 1906.08 1468.38 L1906.08 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M1943.16 1451.02 L1947.1 1451.02 L1935.06 1489.98 L1931.13 1489.98 L1943.16 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M1959.13 1451.02 L1963.07 1451.02 L1951.03 1489.98 L1947.1 1489.98 L1959.13 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M1982.31 1466.95 Q1985.66 1467.66 1987.54 1469.93 Q1989.43 1472.2 1989.43 1475.53 Q1989.43 1480.65 1985.92 1483.45 Q1982.4 1486.25 1975.92 1486.25 Q1973.74 1486.25 1971.43 1485.81 Q1969.13 1485.39 1966.68 1484.54 L1966.68 1480.02 Q1968.62 1481.16 1970.94 1481.74 Q1973.25 1482.32 1975.78 1482.32 Q1980.18 1482.32 1982.47 1480.58 Q1984.78 1478.84 1984.78 1475.53 Q1984.78 1472.48 1982.63 1470.77 Q1980.5 1469.03 1976.68 1469.03 L1972.65 1469.03 L1972.65 1465.19 L1976.87 1465.19 Q1980.31 1465.19 1982.14 1463.82 Q1983.97 1462.43 1983.97 1459.84 Q1983.97 1457.18 1982.07 1455.77 Q1980.2 1454.33 1976.68 1454.33 Q1974.76 1454.33 1972.56 1454.75 Q1970.36 1455.16 1967.72 1456.04 L1967.72 1451.88 Q1970.38 1451.14 1972.7 1450.77 Q1975.04 1450.39 1977.1 1450.39 Q1982.42 1450.39 1985.52 1452.83 Q1988.62 1455.23 1988.62 1459.35 Q1988.62 1462.22 1986.98 1464.21 Q1985.34 1466.18 1982.31 1466.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M1994.94 1459.65 L2019.78 1459.65 L2019.78 1463.91 L2016.52 1463.91 L2016.52 1479.84 Q2016.52 1481.51 2017.07 1482.25 Q2017.65 1482.96 2018.93 1482.96 Q2019.27 1482.96 2019.78 1482.92 Q2020.29 1482.85 2020.45 1482.83 L2020.45 1485.9 Q2019.64 1486.2 2018.79 1486.34 Q2017.93 1486.48 2017.07 1486.48 Q2014.3 1486.48 2013.23 1484.98 Q2012.17 1483.45 2012.17 1479.38 L2012.17 1463.91 L2002.61 1463.91 L2002.61 1485.58 L1998.25 1485.58 L1998.25 1463.91 L1994.94 1463.91 L1994.94 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M2240.85 1481.64 L2257.17 1481.64 L2257.17 1485.58 L2235.22 1485.58 L2235.22 1481.64 Q2237.88 1478.89 2242.47 1474.26 Q2247.07 1469.61 2248.25 1468.27 Q2250.5 1465.74 2251.38 1464.01 Q2252.28 1462.25 2252.28 1460.56 Q2252.28 1457.8 2250.34 1456.07 Q2248.42 1454.33 2245.31 1454.33 Q2243.12 1454.33 2240.66 1455.09 Q2238.23 1455.86 2235.45 1457.41 L2235.45 1452.69 Q2238.28 1451.55 2240.73 1450.97 Q2243.18 1450.39 2245.22 1450.39 Q2250.59 1450.39 2253.79 1453.08 Q2256.98 1455.77 2256.98 1460.26 Q2256.98 1462.39 2256.17 1464.31 Q2255.38 1466.2 2253.28 1468.8 Q2252.7 1469.47 2249.6 1472.69 Q2246.5 1475.88 2240.85 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M2273.95 1451.02 L2277.88 1451.02 L2265.85 1489.98 L2261.91 1489.98 L2273.95 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M2289.92 1451.02 L2293.86 1451.02 L2281.82 1489.98 L2277.88 1489.98 L2289.92 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M2299.74 1481.64 L2307.37 1481.64 L2307.37 1455.28 L2299.06 1456.95 L2299.06 1452.69 L2307.33 1451.02 L2312 1451.02 L2312 1481.64 L2319.64 1481.64 L2319.64 1485.58 L2299.74 1485.58 L2299.74 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M2325.73 1459.65 L2350.57 1459.65 L2350.57 1463.91 L2347.3 1463.91 L2347.3 1479.84 Q2347.3 1481.51 2347.86 1482.25 Q2348.44 1482.96 2349.71 1482.96 Q2350.06 1482.96 2350.57 1482.92 Q2351.08 1482.85 2351.24 1482.83 L2351.24 1485.9 Q2350.43 1486.2 2349.57 1486.34 Q2348.72 1486.48 2347.86 1486.48 Q2345.08 1486.48 2344.02 1484.98 Q2342.95 1483.45 2342.95 1479.38 L2342.95 1463.91 L2333.39 1463.91 L2333.39 1485.58 L2329.04 1485.58 L2329.04 1463.91 L2325.73 1463.91 L2325.73 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M1311.27 1545.45 L1291 1545.45 Q1291.47 1554.96 1293.67 1559 Q1296.41 1563.97 1301.15 1563.97 Q1305.92 1563.97 1308.57 1558.97 Q1310.89 1554.58 1311.27 1545.45 M1311.17 1540.03 Q1310.28 1531 1308.57 1527.81 Q1305.83 1522.78 1301.15 1522.78 Q1296.28 1522.78 1293.7 1527.75 Q1291.66 1531.76 1291.06 1540.03 L1311.17 1540.03 M1301.15 1518.01 Q1308.79 1518.01 1313.15 1524.76 Q1317.51 1531.47 1317.51 1543.38 Q1317.51 1555.25 1313.15 1562 Q1308.79 1568.78 1301.15 1568.78 Q1293.48 1568.78 1289.15 1562 Q1284.79 1555.25 1284.79 1543.38 Q1284.79 1531.47 1289.15 1524.76 Q1293.48 1518.01 1301.15 1518.01 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><polyline clip-path="url(#clip500)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="249.542,1423.18 249.542,47.2441 "/>
<polyline clip-path="url(#clip500)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="249.542,1304.27 268.44,1304.27 "/>
<polyline clip-path="url(#clip500)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="249.542,1052.87 268.44,1052.87 "/>
<polyline clip-path="url(#clip500)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="249.542,801.462 268.44,801.462 "/>
<polyline clip-path="url(#clip500)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="249.542,550.056 268.44,550.056 "/>
<polyline clip-path="url(#clip500)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="249.542,298.65 268.44,298.65 "/>
<path clip-path="url(#clip500)" d="M126.205 1290.07 Q122.593 1290.07 120.765 1293.64 Q118.959 1297.18 118.959 1304.31 Q118.959 1311.41 120.765 1314.98 Q122.593 1318.52 126.205 1318.52 Q129.839 1318.52 131.644 1314.98 Q133.473 1311.41 133.473 1304.31 Q133.473 1297.18 131.644 1293.64 Q129.839 1290.07 126.205 1290.07 M126.205 1286.37 Q132.015 1286.37 135.07 1290.97 Q138.149 1295.56 138.149 1304.31 Q138.149 1313.03 135.07 1317.64 Q132.015 1322.22 126.205 1322.22 Q120.394 1322.22 117.316 1317.64 Q114.26 1313.03 114.26 1304.31 Q114.26 1295.56 117.316 1290.97 Q120.394 1286.37 126.205 1286.37 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M146.366 1315.67 L151.251 1315.67 L151.251 1321.55 L146.366 1321.55 L146.366 1315.67 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M171.436 1305.14 Q168.102 1305.14 166.181 1306.92 Q164.283 1308.71 164.283 1311.83 Q164.283 1314.96 166.181 1316.74 Q168.102 1318.52 171.436 1318.52 Q174.769 1318.52 176.69 1316.74 Q178.612 1314.93 178.612 1311.83 Q178.612 1308.71 176.69 1306.92 Q174.792 1305.14 171.436 1305.14 M166.76 1303.15 Q163.751 1302.41 162.061 1300.35 Q160.394 1298.29 160.394 1295.33 Q160.394 1291.18 163.334 1288.78 Q166.297 1286.37 171.436 1286.37 Q176.598 1286.37 179.538 1288.78 Q182.477 1291.18 182.477 1295.33 Q182.477 1298.29 180.788 1300.35 Q179.121 1302.41 176.135 1303.15 Q179.514 1303.94 181.389 1306.23 Q183.288 1308.52 183.288 1311.83 Q183.288 1316.85 180.209 1319.54 Q177.153 1322.22 171.436 1322.22 Q165.718 1322.22 162.64 1319.54 Q159.584 1316.85 159.584 1311.83 Q159.584 1308.52 161.482 1306.23 Q163.38 1303.94 166.76 1303.15 M165.047 1295.77 Q165.047 1298.45 166.714 1299.96 Q168.403 1301.46 171.436 1301.46 Q174.445 1301.46 176.135 1299.96 Q177.848 1298.45 177.848 1295.77 Q177.848 1293.08 176.135 1291.58 Q174.445 1290.07 171.436 1290.07 Q168.403 1290.07 166.714 1291.58 Q165.047 1293.08 165.047 1295.77 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M201.598 1290.07 Q197.987 1290.07 196.158 1293.64 Q194.352 1297.18 194.352 1304.31 Q194.352 1311.41 196.158 1314.98 Q197.987 1318.52 201.598 1318.52 Q205.232 1318.52 207.037 1314.98 Q208.866 1311.41 208.866 1304.31 Q208.866 1297.18 207.037 1293.64 Q205.232 1290.07 201.598 1290.07 M201.598 1286.37 Q207.408 1286.37 210.463 1290.97 Q213.542 1295.56 213.542 1304.31 Q213.542 1313.03 210.463 1317.64 Q207.408 1322.22 201.598 1322.22 Q195.787 1322.22 192.709 1317.64 Q189.653 1313.03 189.653 1304.31 Q189.653 1295.56 192.709 1290.97 Q195.787 1286.37 201.598 1286.37 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M127.2 1038.67 Q123.589 1038.67 121.76 1042.23 Q119.955 1045.77 119.955 1052.9 Q119.955 1060.01 121.76 1063.57 Q123.589 1067.12 127.2 1067.12 Q130.834 1067.12 132.64 1063.57 Q134.468 1060.01 134.468 1052.9 Q134.468 1045.77 132.64 1042.23 Q130.834 1038.67 127.2 1038.67 M127.2 1034.96 Q133.01 1034.96 136.066 1039.57 Q139.144 1044.15 139.144 1052.9 Q139.144 1061.63 136.066 1066.24 Q133.01 1070.82 127.2 1070.82 Q121.39 1070.82 118.311 1066.24 Q115.256 1061.63 115.256 1052.9 Q115.256 1044.15 118.311 1039.57 Q121.39 1034.96 127.2 1034.96 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M147.362 1064.27 L152.246 1064.27 L152.246 1070.15 L147.362 1070.15 L147.362 1064.27 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M172.431 1053.74 Q169.098 1053.74 167.177 1055.52 Q165.278 1057.3 165.278 1060.43 Q165.278 1063.55 167.177 1065.33 Q169.098 1067.12 172.431 1067.12 Q175.764 1067.12 177.686 1065.33 Q179.607 1063.53 179.607 1060.43 Q179.607 1057.3 177.686 1055.52 Q175.788 1053.74 172.431 1053.74 M167.755 1051.74 Q164.746 1051 163.056 1048.94 Q161.39 1046.88 161.39 1043.92 Q161.39 1039.78 164.329 1037.37 Q167.292 1034.96 172.431 1034.96 Q177.593 1034.96 180.533 1037.37 Q183.473 1039.78 183.473 1043.92 Q183.473 1046.88 181.783 1048.94 Q180.116 1051 177.13 1051.74 Q180.51 1052.53 182.385 1054.82 Q184.283 1057.12 184.283 1060.43 Q184.283 1065.45 181.204 1068.13 Q178.149 1070.82 172.431 1070.82 Q166.714 1070.82 163.635 1068.13 Q160.579 1065.45 160.579 1060.43 Q160.579 1057.12 162.477 1054.82 Q164.376 1052.53 167.755 1051.74 M166.042 1044.36 Q166.042 1047.05 167.709 1048.55 Q169.399 1050.05 172.431 1050.05 Q175.44 1050.05 177.13 1048.55 Q178.843 1047.05 178.843 1044.36 Q178.843 1041.68 177.13 1040.17 Q175.44 1038.67 172.431 1038.67 Q169.399 1038.67 167.709 1040.17 Q166.042 1041.68 166.042 1044.36 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M192.639 1035.59 L210.996 1035.59 L210.996 1039.52 L196.922 1039.52 L196.922 1047.99 Q197.94 1047.65 198.959 1047.49 Q199.977 1047.3 200.996 1047.3 Q206.783 1047.3 210.162 1050.47 Q213.542 1053.64 213.542 1059.06 Q213.542 1064.64 210.07 1067.74 Q206.598 1070.82 200.278 1070.82 Q198.102 1070.82 195.834 1070.45 Q193.588 1070.08 191.181 1069.34 L191.181 1064.64 Q193.264 1065.77 195.487 1066.33 Q197.709 1066.88 200.186 1066.88 Q204.19 1066.88 206.528 1064.78 Q208.866 1062.67 208.866 1059.06 Q208.866 1055.45 206.528 1053.34 Q204.19 1051.24 200.186 1051.24 Q198.311 1051.24 196.436 1051.65 Q194.584 1052.07 192.639 1052.95 L192.639 1035.59 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M126.205 787.26 Q122.593 787.26 120.765 790.825 Q118.959 794.367 118.959 801.496 Q118.959 808.603 120.765 812.168 Q122.593 815.709 126.205 815.709 Q129.839 815.709 131.644 812.168 Q133.473 808.603 133.473 801.496 Q133.473 794.367 131.644 790.825 Q129.839 787.26 126.205 787.26 M126.205 783.557 Q132.015 783.557 135.07 788.163 Q138.149 792.746 138.149 801.496 Q138.149 810.223 135.07 814.83 Q132.015 819.413 126.205 819.413 Q120.394 819.413 117.316 814.83 Q114.26 810.223 114.26 801.496 Q114.26 792.746 117.316 788.163 Q120.394 783.557 126.205 783.557 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M146.366 812.862 L151.251 812.862 L151.251 818.742 L146.366 818.742 L146.366 812.862 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M161.575 818.024 L161.575 813.765 Q163.334 814.598 165.14 815.038 Q166.945 815.478 168.681 815.478 Q173.311 815.478 175.741 812.376 Q178.195 809.251 178.542 802.908 Q177.2 804.899 175.139 805.964 Q173.079 807.029 170.579 807.029 Q165.394 807.029 162.362 803.904 Q159.353 800.756 159.353 795.316 Q159.353 789.992 162.501 786.774 Q165.649 783.557 170.88 783.557 Q176.876 783.557 180.024 788.163 Q183.195 792.746 183.195 801.496 Q183.195 809.668 179.306 814.552 Q175.44 819.413 168.889 819.413 Q167.13 819.413 165.325 819.066 Q163.519 818.718 161.575 818.024 M170.88 803.371 Q174.028 803.371 175.857 801.219 Q177.709 799.066 177.709 795.316 Q177.709 791.589 175.857 789.436 Q174.028 787.26 170.88 787.26 Q167.732 787.26 165.88 789.436 Q164.052 791.589 164.052 795.316 Q164.052 799.066 165.88 801.219 Q167.732 803.371 170.88 803.371 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M201.598 787.26 Q197.987 787.26 196.158 790.825 Q194.352 794.367 194.352 801.496 Q194.352 808.603 196.158 812.168 Q197.987 815.709 201.598 815.709 Q205.232 815.709 207.037 812.168 Q208.866 808.603 208.866 801.496 Q208.866 794.367 207.037 790.825 Q205.232 787.26 201.598 787.26 M201.598 783.557 Q207.408 783.557 210.463 788.163 Q213.542 792.746 213.542 801.496 Q213.542 810.223 210.463 814.83 Q207.408 819.413 201.598 819.413 Q195.787 819.413 192.709 814.83 Q189.653 810.223 189.653 801.496 Q189.653 792.746 192.709 788.163 Q195.787 783.557 201.598 783.557 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M127.2 535.854 Q123.589 535.854 121.76 539.419 Q119.955 542.961 119.955 550.09 Q119.955 557.197 121.76 560.762 Q123.589 564.303 127.2 564.303 Q130.834 564.303 132.64 560.762 Q134.468 557.197 134.468 550.09 Q134.468 542.961 132.64 539.419 Q130.834 535.854 127.2 535.854 M127.2 532.151 Q133.01 532.151 136.066 536.757 Q139.144 541.341 139.144 550.09 Q139.144 558.817 136.066 563.424 Q133.01 568.007 127.2 568.007 Q121.39 568.007 118.311 563.424 Q115.256 558.817 115.256 550.09 Q115.256 541.341 118.311 536.757 Q121.39 532.151 127.2 532.151 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M147.362 561.456 L152.246 561.456 L152.246 567.336 L147.362 567.336 L147.362 561.456 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M162.57 566.618 L162.57 562.359 Q164.329 563.192 166.135 563.632 Q167.94 564.072 169.677 564.072 Q174.306 564.072 176.737 560.97 Q179.19 557.845 179.538 551.503 Q178.195 553.493 176.135 554.558 Q174.075 555.623 171.575 555.623 Q166.39 555.623 163.357 552.498 Q160.348 549.35 160.348 543.91 Q160.348 538.586 163.496 535.368 Q166.644 532.151 171.876 532.151 Q177.871 532.151 181.019 536.757 Q184.19 541.341 184.19 550.09 Q184.19 558.262 180.301 563.146 Q176.436 568.007 169.885 568.007 Q168.126 568.007 166.32 567.66 Q164.515 567.313 162.57 566.618 M171.876 551.965 Q175.024 551.965 176.852 549.813 Q178.704 547.66 178.704 543.91 Q178.704 540.183 176.852 538.03 Q175.024 535.854 171.876 535.854 Q168.727 535.854 166.876 538.03 Q165.047 540.183 165.047 543.91 Q165.047 547.66 166.876 549.813 Q168.727 551.965 171.876 551.965 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M192.639 532.776 L210.996 532.776 L210.996 536.711 L196.922 536.711 L196.922 545.183 Q197.94 544.836 198.959 544.674 Q199.977 544.489 200.996 544.489 Q206.783 544.489 210.162 547.66 Q213.542 550.831 213.542 556.248 Q213.542 561.827 210.07 564.928 Q206.598 568.007 200.278 568.007 Q198.102 568.007 195.834 567.637 Q193.588 567.266 191.181 566.526 L191.181 561.827 Q193.264 562.961 195.487 563.516 Q197.709 564.072 200.186 564.072 Q204.19 564.072 206.528 561.965 Q208.866 559.859 208.866 556.248 Q208.866 552.637 206.528 550.53 Q204.19 548.424 200.186 548.424 Q198.311 548.424 196.436 548.841 Q194.584 549.257 192.639 550.137 L192.639 532.776 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M117.015 311.995 L124.654 311.995 L124.654 285.629 L116.343 287.296 L116.343 283.037 L124.607 281.37 L129.283 281.37 L129.283 311.995 L136.922 311.995 L136.922 315.93 L117.015 315.93 L117.015 311.995 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M146.366 310.05 L151.251 310.05 L151.251 315.93 L146.366 315.93 L146.366 310.05 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M171.436 284.449 Q167.825 284.449 165.996 288.013 Q164.19 291.555 164.19 298.685 Q164.19 305.791 165.996 309.356 Q167.825 312.898 171.436 312.898 Q175.07 312.898 176.876 309.356 Q178.704 305.791 178.704 298.685 Q178.704 291.555 176.876 288.013 Q175.07 284.449 171.436 284.449 M171.436 280.745 Q177.246 280.745 180.301 285.351 Q183.38 289.935 183.38 298.685 Q183.38 307.411 180.301 312.018 Q177.246 316.601 171.436 316.601 Q165.626 316.601 162.547 312.018 Q159.491 307.411 159.491 298.685 Q159.491 289.935 162.547 285.351 Q165.626 280.745 171.436 280.745 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M201.598 284.449 Q197.987 284.449 196.158 288.013 Q194.352 291.555 194.352 298.685 Q194.352 305.791 196.158 309.356 Q197.987 312.898 201.598 312.898 Q205.232 312.898 207.037 309.356 Q208.866 305.791 208.866 298.685 Q208.866 291.555 207.037 288.013 Q205.232 284.449 201.598 284.449 M201.598 280.745 Q207.408 280.745 210.463 285.351 Q213.542 289.935 213.542 298.685 Q213.542 307.411 210.463 312.018 Q207.408 316.601 201.598 316.601 Q195.787 316.601 192.709 312.018 Q189.653 307.411 189.653 298.685 Q189.653 289.935 192.709 285.351 Q195.787 280.745 201.598 280.745 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M33.8307 724.772 Q33.2578 725.759 33.0032 726.937 Q32.7167 728.082 32.7167 729.483 Q32.7167 734.448 35.9632 737.122 Q39.1779 739.763 45.2253 739.763 L64.0042 739.763 L64.0042 745.652 L28.3562 745.652 L28.3562 739.763 L33.8944 739.763 Q30.6479 737.917 29.0883 734.957 Q27.4968 731.997 27.4968 727.764 Q27.4968 727.159 27.5923 726.427 Q27.656 725.695 27.8151 724.804 L33.8307 724.772 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><line clip-path="url(#clip502)" x1="2293.23" y1="298.65" x2="2293.23" y2="282.65" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="2293.23" y1="298.65" x2="2277.23" y2="298.65" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="2293.23" y1="298.65" x2="2293.23" y2="314.65" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="2293.23" y1="298.65" x2="2309.23" y2="298.65" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1301.15" y1="298.679" x2="1301.15" y2="282.679" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1301.15" y1="298.679" x2="1285.15" y2="298.679" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1301.15" y1="298.679" x2="1301.15" y2="314.679" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1301.15" y1="298.679" x2="1317.15" y2="298.679" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1801.25" y1="421.021" x2="1801.25" y2="405.021" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1801.25" y1="421.021" x2="1785.25" y2="421.021" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1801.25" y1="421.021" x2="1801.25" y2="437.021" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1801.25" y1="421.021" x2="1817.25" y2="421.021" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="801.052" y1="421.021" x2="801.052" y2="405.021" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="801.052" y1="421.021" x2="785.052" y2="421.021" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="801.052" y1="421.021" x2="801.052" y2="437.021" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="801.052" y1="421.021" x2="817.052" y2="421.021" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1793.13" y1="421.025" x2="1793.13" y2="405.025" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1793.13" y1="421.025" x2="1777.13" y2="421.025" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1793.13" y1="421.025" x2="1793.13" y2="437.025" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1793.13" y1="421.025" x2="1809.13" y2="421.025" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="809.164" y1="421.025" x2="809.164" y2="405.025" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="809.164" y1="421.025" x2="793.164" y2="421.025" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="809.164" y1="421.025" x2="809.164" y2="437.025" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="809.164" y1="421.025" x2="825.164" y2="421.025" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1308.18" y1="798.841" x2="1308.18" y2="782.841" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1308.18" y1="798.841" x2="1292.18" y2="798.841" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1308.18" y1="798.841" x2="1308.18" y2="814.841" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1308.18" y1="798.841" x2="1324.18" y2="798.841" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1294.12" y1="798.841" x2="1294.12" y2="782.841" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1294.12" y1="798.841" x2="1278.12" y2="798.841" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1294.12" y1="798.841" x2="1294.12" y2="814.841" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1294.12" y1="798.841" x2="1310.12" y2="798.841" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="316.097" y1="798.85" x2="316.097" y2="782.85" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="316.097" y1="798.85" x2="300.097" y2="798.85" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="316.097" y1="798.85" x2="316.097" y2="814.85" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="316.097" y1="798.85" x2="332.097" y2="798.85" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="2286.2" y1="798.85" x2="2286.2" y2="782.85" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="2286.2" y1="798.85" x2="2270.2" y2="798.85" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="2286.2" y1="798.85" x2="2286.2" y2="814.85" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="2286.2" y1="798.85" x2="2302.2" y2="798.85" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1301.15" y1="823.704" x2="1301.15" y2="807.704" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1301.15" y1="823.704" x2="1285.15" y2="823.704" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1301.15" y1="823.704" x2="1301.15" y2="839.704" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1301.15" y1="823.704" x2="1317.15" y2="823.704" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="2293.23" y1="823.706" x2="2293.23" y2="807.706" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="2293.23" y1="823.706" x2="2277.23" y2="823.706" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="2293.23" y1="823.706" x2="2293.23" y2="839.706" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="2293.23" y1="823.706" x2="2309.23" y2="823.706" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1792.44" y1="980.406" x2="1792.44" y2="964.406" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1792.44" y1="980.406" x2="1776.44" y2="980.406" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1792.44" y1="980.406" x2="1792.44" y2="996.406" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1792.44" y1="980.406" x2="1808.44" y2="980.406" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="809.861" y1="980.406" x2="809.861" y2="964.406" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="809.861" y1="980.406" x2="793.861" y2="980.406" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="809.861" y1="980.406" x2="809.861" y2="996.406" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="809.861" y1="980.406" x2="825.861" y2="980.406" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="800.356" y1="980.413" x2="800.356" y2="964.413" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="800.356" y1="980.413" x2="784.356" y2="980.413" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="800.356" y1="980.413" x2="800.356" y2="996.413" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="800.356" y1="980.413" x2="816.356" y2="980.413" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1801.94" y1="980.413" x2="1801.94" y2="964.413" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1801.94" y1="980.413" x2="1785.94" y2="980.413" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1801.94" y1="980.413" x2="1801.94" y2="996.413" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1801.94" y1="980.413" x2="1817.94" y2="980.413" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="2293.23" y1="1362.94" x2="2293.23" y2="1346.94" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="2293.23" y1="1362.94" x2="2277.23" y2="1362.94" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="2293.23" y1="1362.94" x2="2293.23" y2="1378.94" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="2293.23" y1="1362.94" x2="2309.23" y2="1362.94" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1301.15" y1="1363.26" x2="1301.15" y2="1347.26" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1301.15" y1="1363.26" x2="1285.15" y2="1363.26" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1301.15" y1="1363.26" x2="1301.15" y2="1379.26" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1301.15" y1="1363.26" x2="1317.15" y2="1363.26" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="812.673" y1="1423.17" x2="812.673" y2="1407.17" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="812.673" y1="1423.17" x2="796.673" y2="1423.17" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="812.673" y1="1423.17" x2="812.673" y2="1439.17" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="812.673" y1="1423.17" x2="828.673" y2="1423.17" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1789.63" y1="1423.17" x2="1789.63" y2="1407.17" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1789.63" y1="1423.17" x2="1773.63" y2="1423.17" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1789.63" y1="1423.17" x2="1789.63" y2="1439.17" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="1789.63" y1="1423.17" x2="1805.63" y2="1423.17" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="797.546" y1="1423.18" x2="797.546" y2="1407.18" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="797.546" y1="1423.18" x2="781.546" y2="1423.18" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="797.546" y1="1423.18" x2="797.546" y2="1439.18" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip502)" x1="797.546" y1="1423.18" x2="813.546" y2="1423.18" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<path clip-path="url(#clip500)" d="M319.649 196.789 L701.127 196.789 L701.127 93.1086 L319.649 93.1086  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<polyline clip-path="url(#clip500)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="319.649,196.789 701.127,196.789 701.127,93.1086 319.649,93.1086 319.649,196.789 "/>
<line clip-path="url(#clip500)" x1="413.125" y1="144.949" x2="413.125" y2="122.193" style="stroke:#009af9; stroke-width:4.55111; stroke-opacity:1"/>
<line clip-path="url(#clip500)" x1="413.125" y1="144.949" x2="390.37" y2="144.949" style="stroke:#009af9; stroke-width:4.55111; stroke-opacity:1"/>
<line clip-path="url(#clip500)" x1="413.125" y1="144.949" x2="413.125" y2="167.704" style="stroke:#009af9; stroke-width:4.55111; stroke-opacity:1"/>
<line clip-path="url(#clip500)" x1="413.125" y1="144.949" x2="435.881" y2="144.949" style="stroke:#009af9; stroke-width:4.55111; stroke-opacity:1"/>
<path clip-path="url(#clip500)" d="M506.602 127.669 L535.837 127.669 L535.837 131.604 L523.569 131.604 L523.569 162.229 L518.87 162.229 L518.87 131.604 L506.602 131.604 L506.602 127.669 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M548.222 140.284 Q547.504 139.868 546.648 139.682 Q545.814 139.474 544.796 139.474 Q541.185 139.474 539.24 141.835 Q537.319 144.173 537.319 148.571 L537.319 162.229 L533.037 162.229 L533.037 136.303 L537.319 136.303 L537.319 140.331 Q538.662 137.969 540.814 136.835 Q542.967 135.678 546.046 135.678 Q546.486 135.678 547.018 135.747 Q547.55 135.794 548.199 135.909 L548.222 140.284 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M552.689 136.303 L556.948 136.303 L556.948 162.229 L552.689 162.229 L552.689 136.303 M552.689 126.21 L556.948 126.21 L556.948 131.604 L552.689 131.604 L552.689 126.21 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M562.805 136.303 L567.319 136.303 L575.421 158.062 L583.522 136.303 L588.036 136.303 L578.314 162.229 L572.527 162.229 L562.805 136.303 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M593.916 136.303 L598.175 136.303 L598.175 162.229 L593.916 162.229 L593.916 136.303 M593.916 126.21 L598.175 126.21 L598.175 131.604 L593.916 131.604 L593.916 126.21 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M618.869 149.196 Q613.707 149.196 611.717 150.377 Q609.726 151.557 609.726 154.405 Q609.726 156.673 611.207 158.016 Q612.712 159.335 615.281 159.335 Q618.823 159.335 620.953 156.835 Q623.106 154.312 623.106 150.145 L623.106 149.196 L618.869 149.196 M627.365 147.437 L627.365 162.229 L623.106 162.229 L623.106 158.293 Q621.647 160.655 619.471 161.789 Q617.295 162.9 614.147 162.9 Q610.166 162.9 607.805 160.678 Q605.467 158.432 605.467 154.682 Q605.467 150.307 608.383 148.085 Q611.323 145.863 617.133 145.863 L623.106 145.863 L623.106 145.446 Q623.106 142.507 621.161 140.909 Q619.24 139.289 615.744 139.289 Q613.522 139.289 611.416 139.821 Q609.309 140.354 607.365 141.419 L607.365 137.483 Q609.703 136.581 611.902 136.141 Q614.101 135.678 616.184 135.678 Q621.809 135.678 624.587 138.594 Q627.365 141.511 627.365 147.437 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M636.138 126.21 L640.397 126.21 L640.397 162.229 L636.138 162.229 L636.138 126.21 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M659.541 126.257 Q656.439 131.581 654.934 136.789 Q653.429 141.997 653.429 147.344 Q653.429 152.692 654.934 157.946 Q656.462 163.178 659.541 168.479 L655.837 168.479 Q652.365 163.039 650.629 157.784 Q648.916 152.53 648.916 147.344 Q648.916 142.182 650.629 136.951 Q652.341 131.72 655.837 126.257 L659.541 126.257 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip500)" d="M667.133 126.257 L670.837 126.257 Q674.309 131.72 676.022 136.951 Q677.758 142.182 677.758 147.344 Q677.758 152.53 676.022 157.784 Q674.309 163.039 670.837 168.479 L667.133 168.479 Q670.212 163.178 671.716 157.946 Q673.244 152.692 673.244 147.344 Q673.244 141.997 671.716 136.789 Q670.212 131.581 667.133 126.257 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /></svg>

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
[ Info: VUMPS init:	obj = +4.997341808366e-01	err = 3.3223e-02
[ Info: VUMPS   1:	obj = -1.594112352349e-02	err = 3.6251118556e-01	time = 0.03 sec
[ Info: VUMPS   2:	obj = -5.893204055544e-01	err = 3.2586517671e-01	time = 0.02 sec
[ Info: VUMPS   3:	obj = -8.788089887974e-01	err = 6.5358945399e-02	time = 0.07 sec
[ Info: VUMPS   4:	obj = -8.854760112229e-01	err = 9.5922833638e-03	time = 0.02 sec
[ Info: VUMPS   5:	obj = -8.859842744868e-01	err = 6.1204840284e-03	time = 0.02 sec
[ Info: VUMPS   6:	obj = -8.861337420714e-01	err = 3.7792318952e-03	time = 0.03 sec
[ Info: VUMPS   7:	obj = -8.861881017311e-01	err = 3.1317124490e-03	time = 0.07 sec
[ Info: VUMPS   8:	obj = -8.862137350933e-01	err = 2.3975568202e-03	time = 0.03 sec
[ Info: VUMPS   9:	obj = -8.862251327762e-01	err = 2.2106727485e-03	time = 0.03 sec
[ Info: VUMPS  10:	obj = -8.862312162660e-01	err = 1.9293076851e-03	time = 0.03 sec
[ Info: VUMPS  11:	obj = -8.862342619256e-01	err = 1.6050437593e-03	time = 0.06 sec
[ Info: VUMPS  12:	obj = -8.862359505088e-01	err = 1.4816545608e-03	time = 0.03 sec
[ Info: VUMPS  13:	obj = -8.862368764449e-01	err = 1.2781514392e-03	time = 0.03 sec
[ Info: VUMPS  14:	obj = -8.862374129179e-01	err = 1.1624293554e-03	time = 0.05 sec
[ Info: VUMPS  15:	obj = -8.862377281077e-01	err = 1.0938599286e-03	time = 0.03 sec
[ Info: VUMPS  16:	obj = -8.862379338567e-01	err = 1.0425844398e-03	time = 0.05 sec
[ Info: VUMPS  17:	obj = -8.862380883538e-01	err = 1.0373569708e-03	time = 0.03 sec
[ Info: VUMPS  18:	obj = -8.862382378385e-01	err = 9.8984757496e-04	time = 0.05 sec
[ Info: VUMPS  19:	obj = -8.862383889196e-01	err = 9.6785742572e-04	time = 0.03 sec
[ Info: VUMPS  20:	obj = -8.862385564718e-01	err = 9.0837049328e-04	time = 0.06 sec
[ Info: VUMPS  21:	obj = -8.862387352126e-01	err = 8.6293965923e-04	time = 0.03 sec
[ Info: VUMPS  22:	obj = -8.862389214616e-01	err = 7.8828706493e-04	time = 0.04 sec
[ Info: VUMPS  23:	obj = -8.862390971611e-01	err = 7.2607660817e-04	time = 0.03 sec
[ Info: VUMPS  24:	obj = -8.862392532642e-01	err = 6.5364673431e-04	time = 0.04 sec
[ Info: VUMPS  25:	obj = -8.862393778810e-01	err = 5.9147681449e-04	time = 0.03 sec
[ Info: VUMPS  26:	obj = -8.862394733887e-01	err = 5.3636880130e-04	time = 0.05 sec
[ Info: VUMPS  27:	obj = -8.862395423805e-01	err = 4.8504268918e-04	time = 0.03 sec
[ Info: VUMPS  28:	obj = -8.862395923548e-01	err = 4.4441379777e-04	time = 0.05 sec
[ Info: VUMPS  29:	obj = -8.862396285847e-01	err = 4.0231105256e-04	time = 0.03 sec
[ Info: VUMPS  30:	obj = -8.862396556013e-01	err = 3.6916035879e-04	time = 0.06 sec
[ Info: VUMPS  31:	obj = -8.862396763722e-01	err = 3.3334930196e-04	time = 0.03 sec
[ Info: VUMPS  32:	obj = -8.862396927533e-01	err = 3.0425391783e-04	time = 0.05 sec
[ Info: VUMPS  33:	obj = -8.862397061121e-01	err = 2.7382663121e-04	time = 0.03 sec
[ Info: VUMPS  34:	obj = -8.862397171221e-01	err = 2.4782892892e-04	time = 0.06 sec
[ Info: VUMPS  35:	obj = -8.862397264511e-01	err = 2.2202730622e-04	time = 0.03 sec
[ Info: VUMPS  36:	obj = -8.862397343406e-01	err = 1.9955510110e-04	time = 0.05 sec
[ Info: VUMPS  37:	obj = -8.862397411686e-01	err = 1.7802753933e-04	time = 0.03 sec
[ Info: VUMPS  38:	obj = -8.862397470270e-01	err = 1.5915087928e-04	time = 0.06 sec
[ Info: VUMPS  39:	obj = -8.862397521603e-01	err = 1.4153078177e-04	time = 0.03 sec
[ Info: VUMPS  40:	obj = -8.862397566078e-01	err = 1.2605332798e-04	time = 0.06 sec
[ Info: VUMPS  41:	obj = -8.862397605405e-01	err = 1.1189472872e-04	time = 0.04 sec
[ Info: VUMPS  42:	obj = -8.862397639777e-01	err = 9.9739776101e-05	time = 0.05 sec
[ Info: VUMPS  43:	obj = -8.862397670421e-01	err = 8.8453658840e-05	time = 0.05 sec
[ Info: VUMPS  44:	obj = -8.862397697451e-01	err = 7.9189013692e-05	time = 0.03 sec
[ Info: VUMPS  45:	obj = -8.862397721752e-01	err = 7.0330402694e-05	time = 0.05 sec
[ Info: VUMPS  46:	obj = -8.862397743393e-01	err = 6.3267090579e-05	time = 0.03 sec
[ Info: VUMPS  47:	obj = -8.862397763011e-01	err = 5.6426725698e-05	time = 0.04 sec
[ Info: VUMPS  48:	obj = -8.862397780652e-01	err = 5.1128241760e-05	time = 0.04 sec
[ Info: VUMPS  49:	obj = -8.862397796770e-01	err = 4.5926995872e-05	time = 0.05 sec
[ Info: VUMPS  50:	obj = -8.862397811397e-01	err = 4.2009900376e-05	time = 0.04 sec
[ Info: VUMPS  51:	obj = -8.862397824859e-01	err = 3.8108448367e-05	time = 0.05 sec
[ Info: VUMPS  52:	obj = -8.862397837179e-01	err = 3.5242350449e-05	time = 0.03 sec
[ Info: VUMPS  53:	obj = -8.862397848587e-01	err = 3.2344986224e-05	time = 0.04 sec
[ Info: VUMPS  54:	obj = -8.862397859106e-01	err = 3.0255243200e-05	time = 0.03 sec
[ Info: VUMPS  55:	obj = -8.862397868899e-01	err = 2.8111274901e-05	time = 0.04 sec
[ Info: VUMPS  56:	obj = -8.862397877986e-01	err = 2.6577136483e-05	time = 0.04 sec
[ Info: VUMPS  57:	obj = -8.862397886484e-01	err = 2.4981878609e-05	time = 0.03 sec
[ Info: VUMPS  58:	obj = -8.862397894411e-01	err = 2.3834074979e-05	time = 0.04 sec
[ Info: VUMPS  59:	obj = -8.862397901854e-01	err = 2.2628267055e-05	time = 0.03 sec
[ Info: VUMPS  60:	obj = -8.862397908827e-01	err = 2.1743572332e-05	time = 0.04 sec
[ Info: VUMPS  61:	obj = -8.862397915395e-01	err = 2.0809374787e-05	time = 0.04 sec
[ Info: VUMPS  62:	obj = -8.862397921573e-01	err = 2.0102389279e-05	time = 0.05 sec
[ Info: VUMPS  63:	obj = -8.862397927408e-01	err = 1.9356349937e-05	time = 0.03 sec
[ Info: VUMPS  64:	obj = -8.862397932914e-01	err = 1.8770105199e-05	time = 0.05 sec
[ Info: VUMPS  65:	obj = -8.862397938127e-01	err = 1.8155382194e-05	time = 0.03 sec
[ Info: VUMPS  66:	obj = -8.862397943059e-01	err = 1.7653085312e-05	time = 0.06 sec
[ Info: VUMPS  67:	obj = -8.862397947738e-01	err = 1.7131405641e-05	time = 0.05 sec
[ Info: VUMPS  68:	obj = -8.862397952176e-01	err = 1.6689554407e-05	time = 0.03 sec
[ Info: VUMPS  69:	obj = -8.862397956395e-01	err = 1.6235648134e-05	time = 0.04 sec
[ Info: VUMPS  70:	obj = -8.862397960404e-01	err = 1.5839190591e-05	time = 0.03 sec
[ Info: VUMPS  71:	obj = -8.862397964222e-01	err = 1.5436281430e-05	time = 0.04 sec
[ Info: VUMPS  72:	obj = -8.862397967857e-01	err = 1.5075544448e-05	time = 0.03 sec
[ Info: VUMPS  73:	obj = -8.862397971324e-01	err = 1.4712383704e-05	time = 0.04 sec
[ Info: VUMPS  74:	obj = -8.862397974630e-01	err = 1.4380852941e-05	time = 0.03 sec
[ Info: VUMPS  75:	obj = -8.862397977788e-01	err = 1.4049723154e-05	time = 0.06 sec
[ Info: VUMPS  76:	obj = -8.862397980804e-01	err = 1.3742830620e-05	time = 0.03 sec
[ Info: VUMPS  77:	obj = -8.862397983688e-01	err = 1.3438395364e-05	time = 0.04 sec
[ Info: VUMPS  78:	obj = -8.862397986447e-01	err = 1.3152907419e-05	time = 0.03 sec
[ Info: VUMPS  79:	obj = -8.862397989088e-01	err = 1.2871159581e-05	time = 0.05 sec
[ Info: VUMPS  80:	obj = -8.862397991617e-01	err = 1.2604597907e-05	time = 0.04 sec
[ Info: VUMPS  81:	obj = -8.862397994042e-01	err = 1.2342570951e-05	time = 0.06 sec
[ Info: VUMPS  82:	obj = -8.862397996366e-01	err = 1.2092951155e-05	time = 0.03 sec
[ Info: VUMPS  83:	obj = -8.862397998596e-01	err = 1.1848436903e-05	time = 0.05 sec
[ Info: VUMPS  84:	obj = -8.862398000737e-01	err = 1.1614244222e-05	time = 0.03 sec
[ Info: VUMPS  85:	obj = -8.862398002793e-01	err = 1.1385220089e-05	time = 0.06 sec
[ Info: VUMPS  86:	obj = -8.862398004768e-01	err = 1.1164991638e-05	time = 0.03 sec
[ Info: VUMPS  87:	obj = -8.862398006667e-01	err = 1.0949929482e-05	time = 0.05 sec
[ Info: VUMPS  88:	obj = -8.862398008493e-01	err = 1.0742498408e-05	time = 0.03 sec
[ Info: VUMPS  89:	obj = -8.862398010250e-01	err = 1.0540072845e-05	time = 0.04 sec
[ Info: VUMPS  90:	obj = -8.862398011941e-01	err = 1.0344399470e-05	time = 0.02 sec
[ Info: VUMPS  91:	obj = -8.862398013569e-01	err = 1.0153410007e-05	time = 0.04 sec
[ Info: VUMPS  92:	obj = -8.862398015137e-01	err = 9.9686216240e-06	time = 0.02 sec
[ Info: VUMPS  93:	obj = -8.862398016649e-01	err = 9.7882693852e-06	time = 0.06 sec
[ Info: VUMPS  94:	obj = -8.862398018106e-01	err = 9.6133335102e-06	time = 0.03 sec
[ Info: VUMPS  95:	obj = -8.862398019510e-01	err = 9.4426461882e-06	time = 0.06 sec
[ Info: VUMPS  96:	obj = -8.862398020866e-01	err = 9.2768957464e-06	time = 0.03 sec
[ Info: VUMPS  97:	obj = -8.862398022174e-01	err = 9.1151525598e-06	time = 0.05 sec
[ Info: VUMPS  98:	obj = -8.862398023436e-01	err = 8.9578781596e-06	time = 0.03 sec
[ Info: VUMPS  99:	obj = -8.862398024655e-01	err = 8.8043839642e-06	time = 0.04 sec
┌ Warning: VUMPS cancel 100:	obj = -8.862398025833e-01	err = 8.6549648360e-06	time = 3.95 sec
└ @ MPSKit ~/Projects/MPSKit.jl-1/src/algorithms/groundstate/vumps.jl:71

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
  <clipPath id="clip530">
    <rect x="0" y="0" width="2400" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip530)" d="M0 1600 L2400 1600 L2400 0 L0 0  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip531">
    <rect x="480" y="0" width="1681" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip530)" d="M187.803 1352.62 L2352.76 1352.62 L2352.76 123.472 L187.803 123.472  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip532">
    <rect x="187" y="123" width="2166" height="1230"/>
  </clipPath>
</defs>
<polyline clip-path="url(#clip532)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="249.075,1352.62 249.075,123.472 "/>
<polyline clip-path="url(#clip532)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="187.803,767.409 2352.76,767.409 "/>
<polyline clip-path="url(#clip532)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="187.803,170.611 2352.76,170.611 "/>
<polyline clip-path="url(#clip530)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="187.803,1352.62 2352.76,1352.62 "/>
<polyline clip-path="url(#clip530)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="249.075,1352.62 249.075,1371.52 "/>
<path clip-path="url(#clip530)" d="M115.831 1508.55 L136.504 1487.88 L139.286 1490.66 L130.611 1499.34 L152.266 1520.99 L148.943 1524.32 L127.288 1502.66 L118.613 1511.34 L115.831 1508.55 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M154.181 1488.04 Q153.379 1488.26 152.643 1488.73 Q151.906 1489.17 151.186 1489.89 Q148.632 1492.45 148.927 1495.49 Q149.222 1498.5 152.332 1501.61 L161.989 1511.27 L158.961 1514.3 L140.628 1495.97 L143.657 1492.94 L146.505 1495.79 Q145.784 1493.17 146.505 1490.84 Q147.208 1488.5 149.385 1486.33 Q149.696 1486.01 150.122 1485.69 Q150.531 1485.34 151.071 1484.97 L154.181 1488.04 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M154.525 1482.07 L157.537 1479.06 L175.869 1497.39 L172.857 1500.4 L154.525 1482.07 M147.388 1474.93 L150.4 1471.92 L154.214 1475.74 L151.202 1478.75 L147.388 1474.93 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M161.678 1474.92 L164.87 1471.73 L185.985 1481.38 L176.327 1460.27 L179.519 1457.08 L190.977 1482.28 L186.885 1486.37 L161.678 1474.92 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M183.677 1452.92 L186.688 1449.91 L205.021 1468.24 L202.009 1471.25 L183.677 1452.92 M176.54 1445.78 L179.552 1442.77 L183.366 1446.58 L180.354 1449.6 L176.54 1445.78 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M210.439 1444.39 Q206.788 1448.04 206.216 1450.28 Q205.643 1452.53 207.656 1454.54 Q209.26 1456.14 211.257 1456.04 Q213.254 1455.91 215.071 1454.1 Q217.575 1451.59 217.313 1448.32 Q217.051 1445.01 214.105 1442.07 L213.434 1441.4 L210.439 1444.39 M215.202 1437.14 L225.661 1447.6 L222.649 1450.61 L219.867 1447.83 Q220.505 1450.53 219.768 1452.87 Q219.015 1455.19 216.789 1457.42 Q213.974 1460.23 210.733 1460.33 Q207.492 1460.4 204.841 1457.75 Q201.747 1454.65 202.238 1451.02 Q202.745 1447.37 206.854 1443.26 L211.077 1439.04 L210.782 1438.74 Q208.703 1436.66 206.199 1436.91 Q203.695 1437.12 201.223 1439.59 Q199.652 1441.17 198.539 1443.03 Q197.426 1444.9 196.804 1447.03 L194.021 1444.24 Q195.036 1441.95 196.28 1440.09 Q197.508 1438.2 198.981 1436.73 Q202.958 1432.75 206.985 1432.85 Q211.011 1432.95 215.202 1437.14 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M206.396 1415.93 L209.407 1412.91 L234.876 1438.38 L231.864 1441.4 L206.396 1415.93 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M222.976 1399.41 Q224.548 1405.37 227.167 1410.12 Q229.786 1414.86 233.567 1418.64 Q237.348 1422.42 242.127 1425.08 Q246.907 1427.7 252.832 1429.27 L250.213 1431.89 Q243.911 1430.49 238.968 1428.01 Q234.041 1425.5 230.375 1421.84 Q226.725 1418.19 224.237 1413.27 Q221.749 1408.36 220.358 1402.03 L222.976 1399.41 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M228.345 1394.04 L230.964 1391.42 Q237.282 1392.83 242.193 1395.32 Q247.119 1397.79 250.77 1401.44 Q254.436 1405.11 256.924 1410.05 Q259.428 1414.98 260.82 1421.28 L258.201 1423.9 Q256.629 1417.97 253.994 1413.21 Q251.359 1408.41 247.578 1404.63 Q243.797 1400.85 239.034 1398.25 Q234.287 1395.63 228.345 1394.04 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M1186.48 1611.28 L1181.73 1599.09 L1171.96 1616.92 L1165.05 1616.92 L1178.87 1591.71 L1173.08 1576.72 Q1171.52 1572.71 1166.61 1572.71 L1165.05 1572.71 L1165.05 1567.68 L1167.28 1567.74 Q1175.49 1567.96 1177.56 1573.28 L1182.27 1585.47 L1192.05 1567.64 L1198.95 1567.64 L1185.14 1592.85 L1190.93 1607.84 Q1192.49 1611.85 1197.39 1611.85 L1198.95 1611.85 L1198.95 1616.88 L1196.72 1616.82 Q1188.51 1616.6 1186.48 1611.28 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M1228.46 1573.72 L1269.26 1573.72 L1269.26 1579.07 L1228.46 1579.07 L1228.46 1573.72 M1228.46 1586.71 L1269.26 1586.71 L1269.26 1592.12 L1228.46 1592.12 L1228.46 1586.71 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M1303.92 1555.8 L1329.16 1555.8 L1329.16 1561.22 L1309.81 1561.22 L1309.81 1572.86 Q1311.21 1572.39 1312.61 1572.16 Q1314.01 1571.91 1315.41 1571.91 Q1323.37 1571.91 1328.02 1576.27 Q1332.66 1580.63 1332.66 1588.08 Q1332.66 1595.75 1327.89 1600.01 Q1323.11 1604.25 1314.43 1604.25 Q1311.43 1604.25 1308.31 1603.74 Q1305.23 1603.23 1301.92 1602.21 L1301.92 1595.75 Q1304.78 1597.31 1307.84 1598.07 Q1310.89 1598.84 1314.3 1598.84 Q1319.8 1598.84 1323.02 1595.94 Q1326.23 1593.04 1326.23 1588.08 Q1326.23 1583.11 1323.02 1580.22 Q1319.8 1577.32 1314.3 1577.32 Q1311.72 1577.32 1309.14 1577.89 Q1306.6 1578.47 1303.92 1579.68 L1303.92 1555.8 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M1359.08 1560.04 Q1354.12 1560.04 1351.6 1564.94 Q1349.12 1569.81 1349.12 1579.61 Q1349.12 1589.38 1351.6 1594.29 Q1354.12 1599.16 1359.08 1599.16 Q1364.08 1599.16 1366.56 1594.29 Q1369.08 1589.38 1369.08 1579.61 Q1369.08 1569.81 1366.56 1564.94 Q1364.08 1560.04 1359.08 1560.04 M1359.08 1554.95 Q1367.07 1554.95 1371.27 1561.28 Q1375.5 1567.58 1375.5 1579.61 Q1375.5 1591.61 1371.27 1597.95 Q1367.07 1604.25 1359.08 1604.25 Q1351.09 1604.25 1346.86 1597.95 Q1342.66 1591.61 1342.66 1579.61 Q1342.66 1567.58 1346.86 1561.28 Q1351.09 1554.95 1359.08 1554.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><polyline clip-path="url(#clip530)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="187.803,1352.62 187.803,123.472 "/>
<polyline clip-path="url(#clip530)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="187.803,767.409 206.701,767.409 "/>
<polyline clip-path="url(#clip530)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="187.803,170.611 206.701,170.611 "/>
<path clip-path="url(#clip530)" d="M51.6634 787.202 L59.3023 787.202 L59.3023 760.836 L50.9921 762.503 L50.9921 758.244 L59.256 756.577 L63.9319 756.577 L63.9319 787.202 L71.5707 787.202 L71.5707 791.137 L51.6634 791.137 L51.6634 787.202 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M91.0151 759.656 Q87.404 759.656 85.5753 763.22 Q83.7697 766.762 83.7697 773.892 Q83.7697 780.998 85.5753 784.563 Q87.404 788.105 91.0151 788.105 Q94.6493 788.105 96.4548 784.563 Q98.2835 780.998 98.2835 773.892 Q98.2835 766.762 96.4548 763.22 Q94.6493 759.656 91.0151 759.656 M91.0151 755.952 Q96.8252 755.952 99.8808 760.558 Q102.959 765.142 102.959 773.892 Q102.959 782.619 99.8808 787.225 Q96.8252 791.808 91.0151 791.808 Q85.2049 791.808 82.1262 787.225 Q79.0707 782.619 79.0707 773.892 Q79.0707 765.142 82.1262 760.558 Q85.2049 755.952 91.0151 755.952 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M102.959 750.053 L127.071 750.053 L127.071 753.251 L102.959 753.251 L102.959 750.053 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M138.544 760.529 L151.803 760.529 L151.803 763.727 L133.973 763.727 L133.973 760.529 Q136.136 758.291 139.86 754.53 Q143.603 750.749 144.562 749.658 Q146.387 747.608 147.101 746.198 Q147.835 744.768 147.835 743.395 Q147.835 741.157 146.255 739.747 Q144.694 738.336 142.174 738.336 Q140.387 738.336 138.393 738.957 Q136.418 739.577 134.162 740.838 L134.162 737.001 Q136.456 736.079 138.45 735.609 Q140.443 735.139 142.098 735.139 Q146.462 735.139 149.057 737.32 Q151.653 739.502 151.653 743.151 Q151.653 744.881 150.994 746.442 Q150.355 747.984 148.644 750.091 Q148.173 750.636 145.653 753.251 Q143.133 755.846 138.544 760.529 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M81.0976 190.403 L88.7364 190.403 L88.7364 164.038 L80.4263 165.704 L80.4263 161.445 L88.6901 159.778 L93.366 159.778 L93.366 190.403 L101.005 190.403 L101.005 194.338 L81.0976 194.338 L81.0976 190.403 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M120.449 162.857 Q116.838 162.857 115.009 166.422 Q113.204 169.964 113.204 177.093 Q113.204 184.2 115.009 187.764 Q116.838 191.306 120.449 191.306 Q124.083 191.306 125.889 187.764 Q127.718 184.2 127.718 177.093 Q127.718 169.964 125.889 166.422 Q124.083 162.857 120.449 162.857 M120.449 159.153 Q126.259 159.153 129.315 163.76 Q132.394 168.343 132.394 177.093 Q132.394 185.82 129.315 190.426 Q126.259 195.01 120.449 195.01 Q114.639 195.01 111.56 190.426 Q108.505 185.82 108.505 177.093 Q108.505 168.343 111.56 163.76 Q114.639 159.153 120.449 159.153 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M142.098 141.349 Q139.164 141.349 137.679 144.246 Q136.212 147.123 136.212 152.916 Q136.212 158.69 137.679 161.587 Q139.164 164.464 142.098 164.464 Q145.051 164.464 146.518 161.587 Q148.004 158.69 148.004 152.916 Q148.004 147.123 146.518 144.246 Q145.051 141.349 142.098 141.349 M142.098 138.34 Q146.819 138.34 149.302 142.083 Q151.803 145.807 151.803 152.916 Q151.803 160.007 149.302 163.75 Q146.819 167.473 142.098 167.473 Q137.378 167.473 134.876 163.75 Q132.394 160.007 132.394 152.916 Q132.394 145.807 134.876 142.083 Q137.378 138.34 142.098 138.34 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M771.35 12.096 L809.59 12.096 L809.59 18.9825 L779.533 18.9825 L779.533 36.8875 L808.335 36.8875 L808.335 43.7741 L779.533 43.7741 L779.533 65.6895 L810.32 65.6895 L810.32 72.576 L771.35 72.576 L771.35 12.096 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M861.158 45.1919 L861.158 72.576 L853.705 72.576 L853.705 45.4349 Q853.705 38.994 851.193 35.7938 Q848.682 32.5936 843.659 32.5936 Q837.623 32.5936 834.139 36.4419 Q830.655 40.2903 830.655 46.9338 L830.655 72.576 L823.161 72.576 L823.161 27.2059 L830.655 27.2059 L830.655 34.2544 Q833.329 30.163 836.934 28.1376 Q840.58 26.1121 845.319 26.1121 Q853.138 26.1121 857.148 30.9732 Q861.158 35.7938 861.158 45.1919 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M883.398 14.324 L883.398 27.2059 L898.751 27.2059 L898.751 32.9987 L883.398 32.9987 L883.398 57.6282 Q883.398 63.1779 884.897 64.7578 Q886.436 66.3376 891.095 66.3376 L898.751 66.3376 L898.751 72.576 L891.095 72.576 Q882.466 72.576 879.185 69.3758 Q875.904 66.1351 875.904 57.6282 L875.904 32.9987 L870.435 32.9987 L870.435 27.2059 L875.904 27.2059 L875.904 14.324 L883.398 14.324 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M929.173 49.7694 Q920.14 49.7694 916.656 51.8354 Q913.172 53.9013 913.172 58.8839 Q913.172 62.8538 915.765 65.2034 Q918.398 67.5124 922.894 67.5124 Q929.092 67.5124 932.819 63.1374 Q936.586 58.7219 936.586 51.4303 L936.586 49.7694 L929.173 49.7694 M944.04 46.6907 L944.04 72.576 L936.586 72.576 L936.586 65.6895 Q934.034 69.8214 930.226 71.8063 Q926.419 73.7508 920.909 73.7508 Q913.942 73.7508 909.81 69.8619 Q905.718 65.9325 905.718 59.3701 Q905.718 51.7138 910.823 47.825 Q915.967 43.9361 926.135 43.9361 L936.586 43.9361 L936.586 43.2069 Q936.586 38.0623 933.184 35.2672 Q929.821 32.4315 923.704 32.4315 Q919.816 32.4315 916.129 33.3632 Q912.443 34.295 909.04 36.1584 L909.04 29.2718 Q913.132 27.692 916.98 26.9223 Q920.828 26.1121 924.474 26.1121 Q934.318 26.1121 939.179 31.2163 Q944.04 36.3204 944.04 46.6907 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M997.107 45.1919 L997.107 72.576 L989.653 72.576 L989.653 45.4349 Q989.653 38.994 987.142 35.7938 Q984.63 32.5936 979.607 32.5936 Q973.571 32.5936 970.087 36.4419 Q966.604 40.2903 966.604 46.9338 L966.604 72.576 L959.109 72.576 L959.109 27.2059 L966.604 27.2059 L966.604 34.2544 Q969.277 30.163 972.882 28.1376 Q976.528 26.1121 981.268 26.1121 Q989.086 26.1121 993.096 30.9732 Q997.107 35.7938 997.107 45.1919 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M1041.83 49.3643 Q1041.83 41.2625 1038.47 36.8065 Q1035.14 32.3505 1029.11 32.3505 Q1023.11 32.3505 1019.75 36.8065 Q1016.43 41.2625 1016.43 49.3643 Q1016.43 57.4256 1019.75 61.8816 Q1023.11 66.3376 1029.11 66.3376 Q1035.14 66.3376 1038.47 61.8816 Q1041.83 57.4256 1041.83 49.3643 M1049.28 66.9452 Q1049.28 78.5308 1044.14 84.1616 Q1038.99 89.8329 1028.38 89.8329 Q1024.45 89.8329 1020.97 89.2252 Q1017.48 88.6581 1014.2 87.4428 L1014.2 80.1917 Q1017.48 81.9741 1020.68 82.8248 Q1023.88 83.6755 1027.21 83.6755 Q1034.54 83.6755 1038.18 79.8271 Q1041.83 76.0193 1041.83 68.282 L1041.83 64.5957 Q1039.52 68.6061 1035.91 70.5911 Q1032.31 72.576 1027.29 72.576 Q1018.94 72.576 1013.84 66.2161 Q1008.73 59.8562 1008.73 49.3643 Q1008.73 38.832 1013.84 32.472 Q1018.94 26.1121 1027.29 26.1121 Q1032.31 26.1121 1035.91 28.0971 Q1039.52 30.082 1041.83 34.0924 L1041.83 27.2059 L1049.28 27.2059 L1049.28 66.9452 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M1064.64 9.54393 L1072.09 9.54393 L1072.09 72.576 L1064.64 72.576 L1064.64 9.54393 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M1126.49 48.0275 L1126.49 51.6733 L1092.22 51.6733 Q1092.71 59.3701 1096.84 63.421 Q1101.01 67.4314 1108.43 67.4314 Q1112.72 67.4314 1116.73 66.3781 Q1120.78 65.3249 1124.75 63.2184 L1124.75 70.267 Q1120.74 71.9684 1116.53 72.8596 Q1112.31 73.7508 1107.98 73.7508 Q1097.12 73.7508 1090.76 67.4314 Q1084.44 61.1119 1084.44 50.3365 Q1084.44 39.1965 1090.44 32.6746 Q1096.48 26.1121 1106.68 26.1121 Q1115.84 26.1121 1121.15 32.0264 Q1126.49 37.9003 1126.49 48.0275 M1119.04 45.84 Q1118.96 39.7232 1115.6 36.0774 Q1112.27 32.4315 1106.76 32.4315 Q1100.53 32.4315 1096.76 35.9558 Q1093.03 39.4801 1092.47 45.8805 L1119.04 45.84 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M1174.05 35.9153 Q1176.85 30.8922 1180.73 28.5022 Q1184.62 26.1121 1189.89 26.1121 Q1196.98 26.1121 1200.83 31.0947 Q1204.68 36.0368 1204.68 45.1919 L1204.68 72.576 L1197.18 72.576 L1197.18 45.4349 Q1197.18 38.913 1194.87 35.7533 Q1192.56 32.5936 1187.82 32.5936 Q1182.03 32.5936 1178.67 36.4419 Q1175.31 40.2903 1175.31 46.9338 L1175.31 72.576 L1167.81 72.576 L1167.81 45.4349 Q1167.81 38.8725 1165.5 35.7533 Q1163.19 32.5936 1158.37 32.5936 Q1152.66 32.5936 1149.3 36.4824 Q1145.94 40.3308 1145.94 46.9338 L1145.94 72.576 L1138.44 72.576 L1138.44 27.2059 L1145.94 27.2059 L1145.94 34.2544 Q1148.49 30.082 1152.05 28.0971 Q1155.62 26.1121 1160.52 26.1121 Q1165.46 26.1121 1168.91 28.6237 Q1172.39 31.1352 1174.05 35.9153 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M1258.35 48.0275 L1258.35 51.6733 L1224.08 51.6733 Q1224.57 59.3701 1228.7 63.421 Q1232.87 67.4314 1240.28 67.4314 Q1244.58 67.4314 1248.59 66.3781 Q1252.64 65.3249 1256.61 63.2184 L1256.61 70.267 Q1252.6 71.9684 1248.38 72.8596 Q1244.17 73.7508 1239.84 73.7508 Q1228.98 73.7508 1222.62 67.4314 Q1216.3 61.1119 1216.3 50.3365 Q1216.3 39.1965 1222.3 32.6746 Q1228.33 26.1121 1238.54 26.1121 Q1247.7 26.1121 1253 32.0264 Q1258.35 37.9003 1258.35 48.0275 M1250.9 45.84 Q1250.81 39.7232 1247.45 36.0774 Q1244.13 32.4315 1238.62 32.4315 Q1232.38 32.4315 1228.62 35.9558 Q1224.89 39.4801 1224.32 45.8805 L1250.9 45.84 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M1308.3 45.1919 L1308.3 72.576 L1300.84 72.576 L1300.84 45.4349 Q1300.84 38.994 1298.33 35.7938 Q1295.82 32.5936 1290.8 32.5936 Q1284.76 32.5936 1281.28 36.4419 Q1277.79 40.2903 1277.79 46.9338 L1277.79 72.576 L1270.3 72.576 L1270.3 27.2059 L1277.79 27.2059 L1277.79 34.2544 Q1280.47 30.163 1284.07 28.1376 Q1287.72 26.1121 1292.46 26.1121 Q1300.28 26.1121 1304.29 30.9732 Q1308.3 35.7938 1308.3 45.1919 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M1330.54 14.324 L1330.54 27.2059 L1345.89 27.2059 L1345.89 32.9987 L1330.54 32.9987 L1330.54 57.6282 Q1330.54 63.1779 1332.04 64.7578 Q1333.57 66.3376 1338.23 66.3376 L1345.89 66.3376 L1345.89 72.576 L1338.23 72.576 Q1329.61 72.576 1326.32 69.3758 Q1323.04 66.1351 1323.04 57.6282 L1323.04 32.9987 L1317.57 32.9987 L1317.57 27.2059 L1323.04 27.2059 L1323.04 14.324 L1330.54 14.324 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M1418.64 14.0809 L1418.64 22.0612 Q1413.99 19.8332 1409.85 18.7395 Q1405.72 17.6457 1401.87 17.6457 Q1395.19 17.6457 1391.54 20.2383 Q1387.94 22.8309 1387.94 27.611 Q1387.94 31.6214 1390.33 33.6873 Q1392.76 35.7128 1399.48 36.9686 L1404.43 37.9813 Q1413.58 39.7232 1417.91 44.1387 Q1422.29 48.5136 1422.29 55.8863 Q1422.29 64.6767 1416.38 69.2137 Q1410.5 73.7508 1399.12 73.7508 Q1394.82 73.7508 1389.96 72.7785 Q1385.14 71.8063 1379.96 69.9024 L1379.96 61.4765 Q1384.94 64.2716 1389.72 65.6895 Q1394.5 67.1073 1399.12 67.1073 Q1406.13 67.1073 1409.93 64.3527 Q1413.74 61.598 1413.74 56.4939 Q1413.74 52.0379 1410.99 49.5264 Q1408.27 47.0148 1402.04 45.759 L1397.05 44.7868 Q1387.9 42.9639 1383.81 39.075 Q1379.71 35.1862 1379.71 28.2591 Q1379.71 20.2383 1385.35 15.6203 Q1391.02 11.0023 1400.94 11.0023 Q1405.19 11.0023 1409.61 11.7719 Q1414.03 12.5416 1418.64 14.0809 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M1441.94 65.7705 L1441.94 89.8329 L1434.44 89.8329 L1434.44 27.2059 L1441.94 27.2059 L1441.94 34.0924 Q1444.29 30.0415 1447.85 28.0971 Q1451.46 26.1121 1456.44 26.1121 Q1464.7 26.1121 1469.85 32.6746 Q1475.03 39.2371 1475.03 49.9314 Q1475.03 60.6258 1469.85 67.1883 Q1464.7 73.7508 1456.44 73.7508 Q1451.46 73.7508 1447.85 71.8063 Q1444.29 69.8214 1441.94 65.7705 M1467.3 49.9314 Q1467.3 41.7081 1463.89 37.0496 Q1460.53 32.3505 1454.62 32.3505 Q1448.7 32.3505 1445.3 37.0496 Q1441.94 41.7081 1441.94 49.9314 Q1441.94 58.1548 1445.3 62.8538 Q1448.7 67.5124 1454.62 67.5124 Q1460.53 67.5124 1463.89 62.8538 Q1467.3 58.1548 1467.3 49.9314 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M1526.2 48.0275 L1526.2 51.6733 L1491.92 51.6733 Q1492.41 59.3701 1496.54 63.421 Q1500.72 67.4314 1508.13 67.4314 Q1512.42 67.4314 1516.43 66.3781 Q1520.48 65.3249 1524.45 63.2184 L1524.45 70.267 Q1520.44 71.9684 1516.23 72.8596 Q1512.02 73.7508 1507.68 73.7508 Q1496.83 73.7508 1490.47 67.4314 Q1484.15 61.1119 1484.15 50.3365 Q1484.15 39.1965 1490.14 32.6746 Q1496.18 26.1121 1506.39 26.1121 Q1515.54 26.1121 1520.85 32.0264 Q1526.2 37.9003 1526.2 48.0275 M1518.74 45.84 Q1518.66 39.7232 1515.3 36.0774 Q1511.98 32.4315 1506.47 32.4315 Q1500.23 32.4315 1496.46 35.9558 Q1492.73 39.4801 1492.17 45.8805 L1518.74 45.84 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M1571.08 28.9478 L1571.08 35.9153 Q1567.92 34.1734 1564.72 33.3227 Q1561.56 32.4315 1558.32 32.4315 Q1551.07 32.4315 1547.06 37.0496 Q1543.05 41.6271 1543.05 49.9314 Q1543.05 58.2358 1547.06 62.8538 Q1551.07 67.4314 1558.32 67.4314 Q1561.56 67.4314 1564.72 66.5807 Q1567.92 65.6895 1571.08 63.9476 L1571.08 70.8341 Q1567.96 72.2924 1564.6 73.0216 Q1561.28 73.7508 1557.51 73.7508 Q1547.26 73.7508 1541.22 67.3098 Q1535.19 60.8689 1535.19 49.9314 Q1535.19 38.832 1541.26 32.472 Q1547.38 26.1121 1558 26.1121 Q1561.44 26.1121 1564.72 26.8413 Q1568 27.5299 1571.08 28.9478 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M1591.41 14.324 L1591.41 27.2059 L1606.77 27.2059 L1606.77 32.9987 L1591.41 32.9987 L1591.41 57.6282 Q1591.41 63.1779 1592.91 64.7578 Q1594.45 66.3376 1599.11 66.3376 L1606.77 66.3376 L1606.77 72.576 L1599.11 72.576 Q1590.48 72.576 1587.2 69.3758 Q1583.92 66.1351 1583.92 57.6282 L1583.92 32.9987 L1578.45 32.9987 L1578.45 27.2059 L1583.92 27.2059 L1583.92 14.324 L1591.41 14.324 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M1642.86 34.1734 Q1641.61 33.4443 1640.11 33.1202 Q1638.65 32.7556 1636.87 32.7556 Q1630.55 32.7556 1627.14 36.8875 Q1623.78 40.9789 1623.78 48.6757 L1623.78 72.576 L1616.29 72.576 L1616.29 27.2059 L1623.78 27.2059 L1623.78 34.2544 Q1626.13 30.1225 1629.9 28.1376 Q1633.67 26.1121 1639.05 26.1121 Q1639.82 26.1121 1640.76 26.2337 Q1641.69 26.3147 1642.82 26.5172 L1642.86 34.1734 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M1649.91 54.671 L1649.91 27.2059 L1657.36 27.2059 L1657.36 54.3874 Q1657.36 60.8284 1659.88 64.0691 Q1662.39 67.2693 1667.41 67.2693 Q1673.45 67.2693 1676.93 63.421 Q1680.45 59.5726 1680.45 52.9291 L1680.45 27.2059 L1687.91 27.2059 L1687.91 72.576 L1680.45 72.576 L1680.45 65.6084 Q1677.74 69.7404 1674.13 71.7658 Q1670.57 73.7508 1665.83 73.7508 Q1658.01 73.7508 1653.96 68.8897 Q1649.91 64.0286 1649.91 54.671 M1668.67 26.1121 L1668.67 26.1121 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip530)" d="M1738.58 35.9153 Q1741.38 30.8922 1745.27 28.5022 Q1749.16 26.1121 1754.42 26.1121 Q1761.51 26.1121 1765.36 31.0947 Q1769.21 36.0368 1769.21 45.1919 L1769.21 72.576 L1761.72 72.576 L1761.72 45.4349 Q1761.72 38.913 1759.41 35.7533 Q1757.1 32.5936 1752.36 32.5936 Q1746.56 32.5936 1743.2 36.4419 Q1739.84 40.2903 1739.84 46.9338 L1739.84 72.576 L1732.35 72.576 L1732.35 45.4349 Q1732.35 38.8725 1730.04 35.7533 Q1727.73 32.5936 1722.91 32.5936 Q1717.2 32.5936 1713.83 36.4824 Q1710.47 40.3308 1710.47 46.9338 L1710.47 72.576 L1702.98 72.576 L1702.98 27.2059 L1710.47 27.2059 L1710.47 34.2544 Q1713.02 30.082 1716.59 28.0971 Q1720.15 26.1121 1725.05 26.1121 Q1730 26.1121 1733.44 28.6237 Q1736.92 31.1352 1738.58 35.9153 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><circle clip-path="url(#clip532)" cx="453.316" cy="194.848" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="486.662" cy="316.47" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="520.007" cy="321.728" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="553.353" cy="327.066" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="586.698" cy="508.682" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="620.043" cy="513.794" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="653.389" cy="519.977" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="686.734" cy="544.592" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="720.08" cy="670.62" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="753.425" cy="676.548" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="786.771" cy="681.989" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="820.116" cy="747.184" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="853.462" cy="755.011" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="886.807" cy="760.191" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="920.152" cy="762.118" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="953.498" cy="767.47" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="986.843" cy="772.963" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="1020.19" cy="778.567" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="1053.53" cy="782.953" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="1086.88" cy="784.397" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="1120.23" cy="918.781" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="1153.57" cy="925.02" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="1186.92" cy="932.113" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="1220.26" cy="971.908" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="1253.61" cy="1017.48" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="1286.95" cy="1024.38" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="1320.3" cy="1030.71" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="1353.64" cy="1035.53" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="1386.99" cy="1041.85" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="1420.33" cy="1065.12" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="1453.68" cy="1074.04" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="1487.02" cy="1079.18" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="1520.37" cy="1097.93" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="1553.72" cy="1101.19" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="1587.06" cy="1102.65" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="1620.41" cy="1106.89" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="1653.75" cy="1129.11" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="1687.1" cy="1135.56" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="1720.44" cy="1143.65" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="1753.79" cy="1224.4" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="1787.13" cy="1233.98" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="1820.48" cy="1240.63" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="1853.82" cy="1245.58" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="1887.17" cy="1250.11" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="1920.52" cy="1266.09" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="1953.86" cy="1277.33" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="1987.21" cy="1283.85" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="2020.55" cy="1300.67" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="2053.9" cy="1309.44" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip532)" cx="2087.24" cy="1317.83" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
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
└ @ MPSKit ~/Projects/MPSKit.jl-1/src/states/infinitemps.jl:149

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
[ Info: VUMPS init:	obj = +1.052533493982e-02	err = 3.9322e-01
[ Info: VUMPS   1:	obj = -8.586142442258e-01	err = 1.7404583507e-01	time = 0.04 sec
[ Info: VUMPS   2:	obj = -8.855341287075e-01	err = 1.1479354239e-02	time = 0.03 sec
[ Info: VUMPS   3:	obj = -8.860676957691e-01	err = 4.4513971533e-03	time = 0.02 sec
[ Info: VUMPS   4:	obj = -8.862113959862e-01	err = 1.9960263824e-03	time = 0.02 sec
[ Info: VUMPS   5:	obj = -8.862570223250e-01	err = 1.0545641300e-03	time = 0.03 sec
[ Info: VUMPS   6:	obj = -8.862741272987e-01	err = 7.9318206354e-04	time = 0.03 sec
[ Info: VUMPS   7:	obj = -8.862813606933e-01	err = 6.7352140608e-04	time = 0.04 sec
[ Info: VUMPS   8:	obj = -8.862847056809e-01	err = 5.0986510421e-04	time = 0.03 sec
[ Info: VUMPS   9:	obj = -8.862863139592e-01	err = 4.2947960248e-04	time = 0.03 sec
[ Info: VUMPS  10:	obj = -8.862871098320e-01	err = 3.3780968122e-04	time = 0.04 sec
[ Info: VUMPS  11:	obj = -8.862875059167e-01	err = 2.5480879095e-04	time = 0.04 sec
[ Info: VUMPS  12:	obj = -8.862877030835e-01	err = 1.8904912725e-04	time = 0.03 sec
[ Info: VUMPS  13:	obj = -8.862878012433e-01	err = 1.3863285016e-04	time = 0.12 sec
[ Info: VUMPS  14:	obj = -8.862878500377e-01	err = 1.0077734696e-04	time = 0.04 sec
[ Info: VUMPS  15:	obj = -8.862878742666e-01	err = 7.2802908546e-05	time = 0.04 sec
[ Info: VUMPS  16:	obj = -8.862878862821e-01	err = 5.2349798928e-05	time = 0.04 sec
[ Info: VUMPS  17:	obj = -8.862878922369e-01	err = 3.7516355006e-05	time = 0.04 sec
[ Info: VUMPS  18:	obj = -8.862878951873e-01	err = 2.6821162126e-05	time = 0.04 sec
[ Info: VUMPS  19:	obj = -8.862878966493e-01	err = 1.9143092105e-05	time = 0.04 sec
[ Info: VUMPS  20:	obj = -8.862878973739e-01	err = 1.3644291541e-05	time = 0.04 sec
[ Info: VUMPS  21:	obj = -8.862878977333e-01	err = 9.7155403469e-06	time = 0.04 sec
[ Info: VUMPS  22:	obj = -8.862878979116e-01	err = 6.9124233678e-06	time = 0.04 sec
[ Info: VUMPS  23:	obj = -8.862878980002e-01	err = 4.9147579920e-06	time = 0.19 sec
[ Info: VUMPS  24:	obj = -8.862878980441e-01	err = 3.4923740292e-06	time = 0.04 sec
[ Info: VUMPS  25:	obj = -8.862878980660e-01	err = 2.4803303656e-06	time = 0.04 sec
[ Info: VUMPS  26:	obj = -8.862878980768e-01	err = 1.7607651460e-06	time = 0.04 sec
[ Info: VUMPS  27:	obj = -8.862878980822e-01	err = 1.2493874817e-06	time = 0.04 sec
[ Info: VUMPS  28:	obj = -8.862878980849e-01	err = 8.8615845213e-07	time = 0.04 sec
[ Info: VUMPS  29:	obj = -8.862878980863e-01	err = 6.2828472988e-07	time = 0.04 sec
[ Info: VUMPS  30:	obj = -8.862878980869e-01	err = 4.4528979699e-07	time = 0.04 sec
[ Info: VUMPS  31:	obj = -8.862878980873e-01	err = 3.1548565613e-07	time = 0.04 sec
[ Info: VUMPS  32:	obj = -8.862878980875e-01	err = 2.2344785798e-07	time = 0.04 sec
[ Info: VUMPS  33:	obj = -8.862878980875e-01	err = 1.5821262228e-07	time = 0.47 sec
[ Info: VUMPS  34:	obj = -8.862878980876e-01	err = 1.1199335816e-07	time = 0.04 sec
[ Info: VUMPS  35:	obj = -8.862878980876e-01	err = 7.9253223063e-08	time = 0.04 sec
[ Info: VUMPS  36:	obj = -8.862878980876e-01	err = 5.6070812789e-08	time = 0.04 sec
[ Info: VUMPS  37:	obj = -8.862878980876e-01	err = 3.9660398566e-08	time = 0.04 sec
[ Info: VUMPS  38:	obj = -8.862878980876e-01	err = 2.8046869966e-08	time = 0.04 sec
[ Info: VUMPS  39:	obj = -8.862878980877e-01	err = 1.9830105743e-08	time = 0.04 sec
[ Info: VUMPS  40:	obj = -8.862878980877e-01	err = 1.4017963099e-08	time = 0.04 sec
[ Info: VUMPS  41:	obj = -8.862878980877e-01	err = 9.9076364459e-09	time = 0.04 sec
[ Info: VUMPS  42:	obj = -8.862878980877e-01	err = 7.0014191617e-09	time = 0.04 sec
[ Info: VUMPS  43:	obj = -8.862878980877e-01	err = 4.9469527783e-09	time = 0.04 sec
[ Info: VUMPS  44:	obj = -8.862878980877e-01	err = 3.4948630965e-09	time = 0.04 sec
[ Info: VUMPS  45:	obj = -8.862878980877e-01	err = 2.4686947541e-09	time = 0.04 sec
[ Info: VUMPS  46:	obj = -8.862878980877e-01	err = 1.7436290502e-09	time = 0.04 sec
[ Info: VUMPS  47:	obj = -8.862878980877e-01	err = 1.2313805149e-09	time = 0.04 sec
[ Info: VUMPS  48:	obj = -8.862878980877e-01	err = 8.6953674698e-10	time = 0.03 sec
[ Info: VUMPS  49:	obj = -8.862878980877e-01	err = 6.1396305019e-10	time = 0.04 sec
[ Info: VUMPS  50:	obj = -8.862878980877e-01	err = 4.3347347551e-10	time = 0.04 sec
[ Info: VUMPS  51:	obj = -8.862878980877e-01	err = 3.0601329055e-10	time = 0.03 sec
[ Info: VUMPS  52:	obj = -8.862878980877e-01	err = 2.1601337605e-10	time = 0.04 sec
[ Info: VUMPS  53:	obj = -8.862878980877e-01	err = 1.5247304678e-10	time = 0.11 sec
[ Info: VUMPS  54:	obj = -8.862878980877e-01	err = 1.0761819228e-10	time = 0.03 sec
[ Info: VUMPS  55:	obj = -8.862878980877e-01	err = 7.5952215110e-11	time = 0.03 sec
[ Info: VUMPS  56:	obj = -8.862878980878e-01	err = 5.3599539397e-11	time = 0.04 sec
[ Info: VUMPS  57:	obj = -8.862878980878e-01	err = 3.7819901320e-11	time = 0.03 sec
[ Info: VUMPS  58:	obj = -8.862878980878e-01	err = 2.6684106703e-11	time = 0.03 sec
[ Info: VUMPS  59:	obj = -8.862878980878e-01	err = 1.8825942788e-11	time = 0.03 sec
[ Info: VUMPS  60:	obj = -8.862878980878e-01	err = 1.3281993723e-11	time = 0.03 sec
[ Info: VUMPS  61:	obj = -8.862878980878e-01	err = 9.3662459703e-12	time = 0.03 sec
[ Info: VUMPS  62:	obj = -8.862878980878e-01	err = 6.6072758692e-12	time = 0.03 sec
[ Info: VUMPS  63:	obj = -8.862878980878e-01	err = 4.6599206957e-12	time = 0.03 sec
[ Info: VUMPS  64:	obj = -8.862878980878e-01	err = 3.2839000337e-12	time = 0.02 sec
[ Info: VUMPS  65:	obj = -8.862878980878e-01	err = 2.3144827812e-12	time = 0.03 sec
[ Info: VUMPS  66:	obj = -8.862878980878e-01	err = 1.6297721015e-12	time = 0.03 sec
[ Info: VUMPS  67:	obj = -8.862878980878e-01	err = 1.1503316449e-12	time = 0.02 sec
[ Info: VUMPS conv 68:	obj = -8.862878980878e-01	err = 8.0783658769e-13	time = 3.08 sec

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

