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
[ Info: VUMPS init:	obj = +2.499932270895e-01	err = 7.3488e-03
[ Info: VUMPS   1:	obj = -2.684294662883e-01	err = 3.4887991618e-01	time = 0.04 sec
[ Info: VUMPS   2:	obj = -3.908732509075e-01	err = 2.4346308901e-01	time = 0.09 sec
[ Info: VUMPS   3:	obj = -4.103272462718e-01	err = 2.2211707874e-01	time = 0.04 sec
[ Info: VUMPS   4:	obj = +2.763771623319e-02	err = 4.0601070717e-01	time = 0.06 sec
[ Info: VUMPS   5:	obj = -1.278912388343e-01	err = 4.0024753668e-01	time = 0.04 sec
[ Info: VUMPS   6:	obj = -6.128684560655e-02	err = 3.9554377744e-01	time = 0.05 sec
[ Info: VUMPS   7:	obj = -2.019229524257e-01	err = 3.9507954388e-01	time = 0.03 sec
[ Info: VUMPS   8:	obj = -3.385836978728e-01	err = 3.1206810658e-01	time = 0.06 sec
[ Info: VUMPS   9:	obj = -4.217712628117e-01	err = 1.9266010859e-01	time = 0.05 sec
[ Info: VUMPS  10:	obj = +3.119439956568e-02	err = 4.0307569557e-01	time = 0.05 sec
[ Info: VUMPS  11:	obj = -1.453987450329e-01	err = 3.7892383130e-01	time = 0.03 sec
[ Info: VUMPS  12:	obj = -3.122541983387e-01	err = 3.4361928372e-01	time = 0.05 sec
[ Info: VUMPS  13:	obj = -3.175415058698e-01	err = 3.4372788072e-01	time = 0.06 sec
[ Info: VUMPS  14:	obj = -9.281281368249e-02	err = 3.8082690832e-01	time = 0.04 sec
[ Info: VUMPS  15:	obj = -1.118289260232e-01	err = 3.9703529501e-01	time = 0.04 sec
[ Info: VUMPS  16:	obj = -3.392299177529e-02	err = 3.8653278445e-01	time = 0.04 sec
[ Info: VUMPS  17:	obj = -8.859386904872e-02	err = 3.7559265639e-01	time = 0.06 sec
[ Info: VUMPS  18:	obj = -2.660652072169e-01	err = 3.4535690909e-01	time = 0.06 sec
[ Info: VUMPS  19:	obj = +1.581362017845e-01	err = 3.6728199421e-01	time = 0.02 sec
[ Info: VUMPS  20:	obj = -2.614275413781e-03	err = 3.5210479173e-01	time = 0.04 sec
[ Info: VUMPS  21:	obj = -1.404557974268e-01	err = 3.8727147114e-01	time = 0.02 sec
[ Info: VUMPS  22:	obj = -1.922542821356e-01	err = 3.6768505428e-01	time = 0.05 sec
[ Info: VUMPS  23:	obj = +8.174391358782e-02	err = 3.9158252120e-01	time = 0.02 sec
[ Info: VUMPS  24:	obj = +4.347184217158e-02	err = 3.3940461711e-01	time = 0.05 sec
[ Info: VUMPS  25:	obj = -2.122928828839e-01	err = 3.6816674483e-01	time = 0.02 sec
[ Info: VUMPS  26:	obj = -3.750381325917e-02	err = 4.4510656286e-01	time = 0.06 sec
[ Info: VUMPS  27:	obj = -9.698985317742e-02	err = 4.0266067497e-01	time = 0.05 sec
[ Info: VUMPS  28:	obj = -2.160914808205e-01	err = 3.6590508687e-01	time = 0.03 sec
[ Info: VUMPS  29:	obj = +4.936490579149e-02	err = 3.7234044270e-01	time = 0.05 sec
[ Info: VUMPS  30:	obj = -1.148676233851e-01	err = 4.1462641635e-01	time = 0.02 sec
[ Info: VUMPS  31:	obj = -1.519618595135e-01	err = 3.7492975708e-01	time = 0.05 sec
[ Info: VUMPS  32:	obj = -1.890862522082e-01	err = 3.7659388392e-01	time = 0.03 sec
[ Info: VUMPS  33:	obj = -7.477276890967e-02	err = 3.9194272124e-01	time = 0.05 sec
[ Info: VUMPS  34:	obj = -1.940989007410e-01	err = 3.9403330192e-01	time = 0.03 sec
[ Info: VUMPS  35:	obj = -3.502934869819e-01	err = 3.0111969128e-01	time = 0.05 sec
[ Info: VUMPS  36:	obj = -3.442085541276e-01	err = 3.2611257359e-01	time = 0.04 sec
[ Info: VUMPS  37:	obj = -2.312483202161e-01	err = 3.7396977836e-01	time = 0.06 sec
[ Info: VUMPS  38:	obj = -2.282920202884e-01	err = 3.5801388290e-01	time = 0.06 sec
[ Info: VUMPS  39:	obj = -3.434233746212e-01	err = 3.3005413456e-01	time = 0.05 sec
[ Info: VUMPS  40:	obj = -3.700311228608e-01	err = 2.8698807789e-01	time = 0.05 sec
[ Info: VUMPS  41:	obj = -2.287419146432e-01	err = 3.7652136991e-01	time = 0.08 sec
[ Info: VUMPS  42:	obj = -1.848386897474e-01	err = 3.9410789712e-01	time = 0.07 sec
[ Info: VUMPS  43:	obj = -4.151049475911e-02	err = 3.6614428192e-01	time = 0.04 sec
[ Info: VUMPS  44:	obj = +4.278404791371e-02	err = 3.8999971963e-01	time = 0.06 sec
[ Info: VUMPS  45:	obj = +4.133615949384e-02	err = 3.7259192609e-01	time = 0.04 sec
[ Info: VUMPS  46:	obj = -1.755417704686e-01	err = 3.8281160487e-01	time = 0.06 sec
[ Info: VUMPS  47:	obj = -3.433301511715e-01	err = 3.1797681990e-01	time = 0.03 sec
[ Info: VUMPS  48:	obj = +8.038768523644e-02	err = 3.5190647487e-01	time = 0.05 sec
[ Info: VUMPS  49:	obj = -6.209071290954e-02	err = 3.4368443495e-01	time = 0.07 sec
[ Info: VUMPS  50:	obj = +1.373233695943e-01	err = 3.4584780766e-01	time = 0.04 sec
[ Info: VUMPS  51:	obj = -1.298395809866e-02	err = 3.7394444037e-01	time = 0.06 sec
[ Info: VUMPS  52:	obj = -1.849662152303e-01	err = 4.0186739319e-01	time = 0.04 sec
[ Info: VUMPS  53:	obj = -2.535592561522e-01	err = 3.6315928824e-01	time = 0.05 sec
[ Info: VUMPS  54:	obj = -2.078073506292e-01	err = 3.8862703529e-01	time = 0.03 sec
[ Info: VUMPS  55:	obj = -2.648807481312e-01	err = 3.5935081624e-01	time = 0.05 sec
[ Info: VUMPS  56:	obj = +3.508321588679e-02	err = 3.5155300157e-01	time = 0.06 sec
[ Info: VUMPS  57:	obj = +2.587094014595e-02	err = 3.7426723561e-01	time = 0.05 sec
[ Info: VUMPS  58:	obj = +6.099768411641e-02	err = 3.6261231989e-01	time = 0.06 sec
[ Info: VUMPS  59:	obj = +6.085060881519e-02	err = 3.5718570541e-01	time = 0.02 sec
[ Info: VUMPS  60:	obj = -4.856039331256e-02	err = 3.9347118222e-01	time = 0.06 sec
[ Info: VUMPS  61:	obj = -5.072268826867e-02	err = 3.8271040330e-01	time = 0.04 sec
[ Info: VUMPS  62:	obj = +1.769142598093e-02	err = 4.0183932016e-01	time = 0.05 sec
[ Info: VUMPS  63:	obj = +5.913223876388e-02	err = 4.0964563195e-01	time = 0.06 sec
[ Info: VUMPS  64:	obj = -2.478892754727e-01	err = 3.8032380630e-01	time = 0.04 sec
[ Info: VUMPS  65:	obj = -3.071655369627e-01	err = 3.4522086404e-01	time = 0.05 sec
[ Info: VUMPS  66:	obj = -3.204977019062e-01	err = 3.2905910740e-01	time = 0.05 sec
[ Info: VUMPS  67:	obj = -2.546816866134e-01	err = 3.8561093133e-01	time = 0.07 sec
[ Info: VUMPS  68:	obj = -2.393554685021e-01	err = 3.5880361408e-01	time = 0.05 sec
[ Info: VUMPS  69:	obj = -1.327133696952e-01	err = 4.0101733932e-01	time = 0.06 sec
[ Info: VUMPS  70:	obj = -1.041902383608e-01	err = 4.1285042628e-01	time = 0.05 sec
[ Info: VUMPS  71:	obj = -1.993915263334e-01	err = 3.6003918734e-01	time = 0.04 sec
[ Info: VUMPS  72:	obj = -1.940520154241e-01	err = 3.4570202542e-01	time = 0.02 sec
[ Info: VUMPS  73:	obj = -8.071173596559e-02	err = 3.9919344455e-01	time = 0.04 sec
[ Info: VUMPS  74:	obj = -1.370928589038e-01	err = 3.8043789146e-01	time = 0.02 sec
[ Info: VUMPS  75:	obj = -2.590488646429e-01	err = 3.6030703598e-01	time = 0.05 sec
[ Info: VUMPS  76:	obj = -4.074845294173e-01	err = 2.2747635384e-01	time = 0.03 sec
[ Info: VUMPS  77:	obj = -4.083741658015e-01	err = 2.2604491079e-01	time = 0.05 sec
[ Info: VUMPS  78:	obj = +1.005627309383e-01	err = 3.4879269264e-01	time = 0.05 sec
[ Info: VUMPS  79:	obj = -1.534111629484e-01	err = 4.0398980845e-01	time = 0.04 sec
[ Info: VUMPS  80:	obj = -1.185244700305e-01	err = 4.1527071676e-01	time = 0.05 sec
[ Info: VUMPS  81:	obj = -2.203348020820e-01	err = 3.4143151318e-01	time = 0.04 sec
[ Info: VUMPS  82:	obj = -5.889431350950e-02	err = 3.9822663932e-01	time = 0.05 sec
[ Info: VUMPS  83:	obj = -1.737018741265e-01	err = 3.8538020144e-01	time = 0.04 sec
[ Info: VUMPS  84:	obj = -1.582331643179e-01	err = 3.6506684092e-01	time = 0.02 sec
[ Info: VUMPS  85:	obj = -2.020317867236e-01	err = 3.7807390527e-01	time = 0.05 sec
[ Info: VUMPS  86:	obj = -6.233681285915e-02	err = 3.7406473722e-01	time = 0.02 sec
[ Info: VUMPS  87:	obj = -1.314218634692e-01	err = 3.8818132852e-01	time = 0.06 sec
[ Info: VUMPS  88:	obj = -2.302777507565e-01	err = 3.6235120973e-01	time = 0.04 sec
[ Info: VUMPS  89:	obj = +1.777592295373e-02	err = 3.1886054992e-01	time = 0.06 sec
[ Info: VUMPS  90:	obj = -9.137389133038e-02	err = 3.6650533950e-01	time = 0.05 sec
[ Info: VUMPS  91:	obj = -1.522621453585e-01	err = 3.7520570789e-01	time = 0.04 sec
[ Info: VUMPS  92:	obj = -2.067248068564e-01	err = 3.8667875904e-01	time = 0.05 sec
[ Info: VUMPS  93:	obj = +2.047786471459e-02	err = 3.7528645946e-01	time = 0.04 sec
[ Info: VUMPS  94:	obj = -2.196126737735e-01	err = 3.7361837770e-01	time = 0.06 sec
[ Info: VUMPS  95:	obj = -3.148681836053e-01	err = 3.3877001456e-01	time = 0.03 sec
[ Info: VUMPS  96:	obj = -3.632549896064e-01	err = 3.1063135089e-01	time = 0.05 sec
[ Info: VUMPS  97:	obj = -2.359487524961e-01	err = 3.4983505936e-01	time = 0.05 sec
[ Info: VUMPS  98:	obj = -2.085648662363e-01	err = 3.6646074907e-01	time = 0.04 sec
[ Info: VUMPS  99:	obj = -3.378709599647e-01	err = 3.1090195525e-01	time = 0.03 sec
[ Info: VUMPS 100:	obj = +1.697054331465e-01	err = 3.3318043865e-01	time = 0.03 sec
[ Info: VUMPS 101:	obj = -1.372336728300e-01	err = 3.9831002380e-01	time = 0.03 sec
[ Info: VUMPS 102:	obj = -2.025796302451e-01	err = 3.6198000343e-01	time = 0.05 sec
[ Info: VUMPS 103:	obj = -2.078176354072e-01	err = 3.8150634396e-01	time = 0.03 sec
[ Info: VUMPS 104:	obj = -2.923528623190e-01	err = 3.3199471479e-01	time = 0.09 sec
[ Info: VUMPS 105:	obj = -2.895261890513e-02	err = 3.9381156810e-01	time = 0.06 sec
[ Info: VUMPS 106:	obj = +1.102282311350e-01	err = 3.6753101837e-01	time = 0.02 sec
[ Info: VUMPS 107:	obj = -3.457075224669e-03	err = 3.8935262670e-01	time = 0.05 sec
[ Info: VUMPS 108:	obj = -1.401802303936e-01	err = 3.5477482436e-01	time = 0.07 sec
[ Info: VUMPS 109:	obj = -2.773177515742e-01	err = 3.5618814609e-01	time = 0.04 sec
[ Info: VUMPS 110:	obj = -2.980497929052e-01	err = 3.3973067915e-01	time = 0.07 sec
[ Info: VUMPS 111:	obj = -3.666221447505e-01	err = 2.9612511089e-01	time = 0.04 sec
[ Info: VUMPS 112:	obj = -3.406894034780e-01	err = 3.1838335674e-01	time = 0.07 sec
[ Info: VUMPS 113:	obj = -4.117451756841e-01	err = 2.1229269280e-01	time = 0.05 sec
[ Info: VUMPS 114:	obj = +3.472393021263e-02	err = 4.2138653263e-01	time = 0.04 sec
[ Info: VUMPS 115:	obj = +6.929635845720e-03	err = 4.0042995319e-01	time = 0.02 sec
[ Info: VUMPS 116:	obj = +1.198686632180e-01	err = 3.5017433469e-01	time = 0.04 sec
[ Info: VUMPS 117:	obj = -1.271079129807e-01	err = 3.7101933225e-01	time = 0.02 sec
[ Info: VUMPS 118:	obj = -1.834088729116e-01	err = 3.6678418731e-01	time = 0.03 sec
[ Info: VUMPS 119:	obj = -3.363021880354e-01	err = 3.1503549193e-01	time = 0.02 sec
[ Info: VUMPS 120:	obj = -4.275043892741e-01	err = 1.5610761856e-01	time = 0.06 sec
[ Info: VUMPS 121:	obj = +3.978743301000e-02	err = 3.1230305054e-01	time = 0.08 sec
[ Info: VUMPS 122:	obj = -1.990612171666e-01	err = 3.6200810172e-01	time = 0.04 sec
[ Info: VUMPS 123:	obj = +1.050122944992e-01	err = 3.4198140893e-01	time = 0.04 sec
[ Info: VUMPS 124:	obj = -5.398579365216e-02	err = 4.1624422030e-01	time = 0.04 sec
[ Info: VUMPS 125:	obj = -1.003705443385e-01	err = 3.7636491379e-01	time = 0.02 sec
[ Info: VUMPS 126:	obj = +8.454616496648e-02	err = 3.5803981901e-01	time = 0.04 sec
[ Info: VUMPS 127:	obj = -1.128523326267e-01	err = 3.8181570938e-01	time = 0.03 sec
[ Info: VUMPS 128:	obj = -2.150635416320e-01	err = 3.5780210657e-01	time = 0.04 sec
[ Info: VUMPS 129:	obj = -1.312465126627e-01	err = 3.8405159732e-01	time = 0.06 sec
[ Info: VUMPS 130:	obj = -3.846997189431e-02	err = 4.2623757858e-01	time = 0.05 sec
[ Info: VUMPS 131:	obj = +2.473200047618e-02	err = 3.5905352543e-01	time = 0.05 sec
[ Info: VUMPS 132:	obj = +7.006948564206e-02	err = 3.4845271565e-01	time = 0.04 sec
[ Info: VUMPS 133:	obj = -2.383461944993e-01	err = 3.7052130221e-01	time = 0.06 sec
[ Info: VUMPS 134:	obj = -2.062217898227e-01	err = 3.6937256423e-01	time = 0.04 sec
[ Info: VUMPS 135:	obj = -2.274384652936e-01	err = 3.4724295244e-01	time = 0.07 sec
[ Info: VUMPS 136:	obj = -2.622435856918e-01	err = 3.4143263011e-01	time = 0.05 sec
[ Info: VUMPS 137:	obj = -1.368393705006e-01	err = 3.9505761386e-01	time = 0.04 sec
[ Info: VUMPS 138:	obj = -2.365952891363e-01	err = 3.6070782003e-01	time = 0.05 sec
[ Info: VUMPS 139:	obj = -3.285892419662e-01	err = 3.2510337702e-01	time = 0.06 sec
[ Info: VUMPS 140:	obj = -2.897155176623e-01	err = 3.4971272209e-01	time = 0.05 sec
[ Info: VUMPS 141:	obj = -5.613172601920e-02	err = 4.0280146205e-01	time = 0.06 sec
[ Info: VUMPS 142:	obj = -1.912055510339e-01	err = 3.8086840086e-01	time = 0.06 sec
[ Info: VUMPS 143:	obj = -2.522493416812e-01	err = 3.7780461889e-01	time = 0.03 sec
[ Info: VUMPS 144:	obj = -1.448388260442e-01	err = 3.9735778286e-01	time = 0.06 sec
[ Info: VUMPS 145:	obj = +2.133811571892e-02	err = 3.6767706542e-01	time = 0.05 sec
[ Info: VUMPS 146:	obj = -1.021149430836e-01	err = 4.1518428940e-01	time = 0.04 sec
[ Info: VUMPS 147:	obj = +3.345423157498e-02	err = 3.7487854814e-01	time = 0.06 sec
[ Info: VUMPS 148:	obj = -1.519027769061e-01	err = 3.7323297896e-01	time = 0.04 sec
[ Info: VUMPS 149:	obj = -1.015157315769e-01	err = 3.9037495750e-01	time = 0.07 sec
[ Info: VUMPS 150:	obj = -7.525226965218e-02	err = 4.0412444789e-01	time = 0.05 sec
[ Info: VUMPS 151:	obj = -1.435426260194e-01	err = 4.0959006551e-01	time = 0.08 sec
[ Info: VUMPS 152:	obj = -1.234168209435e-01	err = 4.0252461632e-01	time = 0.02 sec
[ Info: VUMPS 153:	obj = -1.582356991859e-01	err = 4.0856159774e-01	time = 0.07 sec
[ Info: VUMPS 154:	obj = -2.611502661937e-02	err = 3.7361316068e-01	time = 0.02 sec
[ Info: VUMPS 155:	obj = -1.903742563432e-02	err = 3.6268799414e-01	time = 0.05 sec
[ Info: VUMPS 156:	obj = -1.660075702137e-01	err = 3.6001499475e-01	time = 0.04 sec
[ Info: VUMPS 157:	obj = -1.750801644509e-01	err = 3.5799477869e-01	time = 0.03 sec
[ Info: VUMPS 158:	obj = -3.185357562769e-01	err = 3.4039580564e-01	time = 0.04 sec
[ Info: VUMPS 159:	obj = -3.462765635183e-01	err = 3.0750748489e-01	time = 0.03 sec
[ Info: VUMPS 160:	obj = -3.906343752662e-01	err = 2.7025958372e-01	time = 0.05 sec
[ Info: VUMPS 161:	obj = +6.478821525323e-02	err = 4.1194192280e-01	time = 0.05 sec
[ Info: VUMPS 162:	obj = -2.303254431617e-02	err = 3.8071990310e-01	time = 0.04 sec
[ Info: VUMPS 163:	obj = -2.010080764751e-01	err = 3.7801881819e-01	time = 0.04 sec
[ Info: VUMPS 164:	obj = -3.545532748718e-01	err = 2.8772339262e-01	time = 0.04 sec
[ Info: VUMPS 165:	obj = -3.269217578981e-01	err = 3.3116005000e-01	time = 0.05 sec
[ Info: VUMPS 166:	obj = -7.763273380566e-02	err = 4.4233452292e-01	time = 0.05 sec
[ Info: VUMPS 167:	obj = -6.297628554394e-02	err = 4.0291320901e-01	time = 0.02 sec
[ Info: VUMPS 168:	obj = -6.509753260918e-02	err = 3.7453292998e-01	time = 0.04 sec
[ Info: VUMPS 169:	obj = -1.479501367088e-01	err = 3.7646685662e-01	time = 0.06 sec
[ Info: VUMPS 170:	obj = -1.936550011338e-01	err = 3.9968563965e-01	time = 0.05 sec
[ Info: VUMPS 171:	obj = -2.407761747630e-01	err = 3.8299383049e-01	time = 0.05 sec
[ Info: VUMPS 172:	obj = +2.261073305309e-02	err = 4.0104933806e-01	time = 0.03 sec
[ Info: VUMPS 173:	obj = -7.166260900992e-02	err = 3.9840210145e-01	time = 0.04 sec
[ Info: VUMPS 174:	obj = -1.839330523396e-01	err = 3.7559526845e-01	time = 0.03 sec
[ Info: VUMPS 175:	obj = -3.059547394952e-01	err = 3.3999396097e-01	time = 0.04 sec
[ Info: VUMPS 176:	obj = -2.995029313597e-01	err = 3.4151945139e-01	time = 0.06 sec
[ Info: VUMPS 177:	obj = -1.628200467586e-01	err = 3.9581213200e-01	time = 0.03 sec
[ Info: VUMPS 178:	obj = -2.771745987181e-02	err = 3.7799012327e-01	time = 0.04 sec
[ Info: VUMPS 179:	obj = -2.450722764418e-01	err = 3.6283636173e-01	time = 0.03 sec
[ Info: VUMPS 180:	obj = +2.249435492935e-02	err = 3.8247790534e-01	time = 0.07 sec
[ Info: VUMPS 181:	obj = -2.354825221975e-02	err = 4.1422828465e-01	time = 0.03 sec
[ Info: VUMPS 182:	obj = -1.142934591710e-01	err = 3.8986670495e-01	time = 0.04 sec
[ Info: VUMPS 183:	obj = -1.567308697369e-01	err = 3.9700243050e-01	time = 0.03 sec
[ Info: VUMPS 184:	obj = -1.842499532379e-01	err = 3.6532838941e-01	time = 0.05 sec
[ Info: VUMPS 185:	obj = -2.514877837893e-01	err = 3.6637072380e-01	time = 0.03 sec
[ Info: VUMPS 186:	obj = -5.058982255378e-02	err = 4.0946736692e-01	time = 0.06 sec
[ Info: VUMPS 187:	obj = -3.684615096066e-03	err = 4.0659110148e-01	time = 0.03 sec
[ Info: VUMPS 188:	obj = -1.225767651180e-01	err = 4.2689031391e-01	time = 0.05 sec
[ Info: VUMPS 189:	obj = -2.461679320853e-01	err = 3.6120729961e-01	time = 0.04 sec
[ Info: VUMPS 190:	obj = +6.763190603494e-02	err = 3.8401937650e-01	time = 0.03 sec
[ Info: VUMPS 191:	obj = +6.358079111108e-02	err = 3.8794532554e-01	time = 0.04 sec
[ Info: VUMPS 192:	obj = -2.591627687434e-01	err = 3.4498062399e-01	time = 0.03 sec
[ Info: VUMPS 193:	obj = -3.469647117223e-01	err = 3.1920950928e-01	time = 0.07 sec
[ Info: VUMPS 194:	obj = -2.328255401437e-01	err = 3.7087112305e-01	time = 0.05 sec
[ Info: VUMPS 195:	obj = -1.059181306961e-01	err = 3.9644832676e-01	time = 0.06 sec
[ Info: VUMPS 196:	obj = +3.165492219560e-02	err = 3.4664317312e-01	time = 0.04 sec
[ Info: VUMPS 197:	obj = -1.060919702177e-01	err = 4.0404073514e-01	time = 0.06 sec
[ Info: VUMPS 198:	obj = -7.551485865858e-02	err = 4.1482067043e-01	time = 0.03 sec
[ Info: VUMPS 199:	obj = -8.128843893232e-02	err = 3.8193109836e-01	time = 0.07 sec
┌ Warning: VUMPS cancel 200:	obj = -1.297030723726e-01	err = 3.8203051738e-01	time = 9.08 sec
└ @ MPSKit ~/Projects/MPSKit.jl/src/algorithms/groundstate/vumps.jl:71

````

As you can see, VUMPS struggles to converge.
On it's own, that is already quite curious.
Maybe we can do better using another algorithm, such as gradient descent.

````julia
groundstate, cache, delta = find_groundstate(state, H, GradientGrassmann(; maxiter=20));
````

````
[ Info: CG: initializing with f = 0.249993227089, ‖∇f‖ = 5.1973e-03
[ Info: CG: iter    1: f = -0.104179418829, ‖∇f‖ = 3.1103e-01, α = 1.34e+04, β = 0.00e+00, nfg = 5
[ Info: CG: iter    2: f = -0.168230521916, ‖∇f‖ = 3.1218e-01, α = 1.73e+00, β = 4.36e+01, nfg = 25
[ Info: CG: iter    3: f = -0.279139021454, ‖∇f‖ = 2.5424e-01, α = 1.12e+00, β = 6.09e-01, nfg = 3
[ Info: CG: iter    4: f = -0.351436956991, ‖∇f‖ = 1.8762e-01, α = 8.32e-01, β = 2.88e-01, nfg = 3
[ Info: CG: iter    5: f = -0.394810706473, ‖∇f‖ = 1.5263e-01, α = 7.39e-01, β = 2.23e-01, nfg = 2
[ Info: CG: iter    6: f = -0.421934998173, ‖∇f‖ = 1.1117e-01, α = 6.51e-01, β = 3.64e-01, nfg = 2
[ Info: CG: iter    7: f = -0.432301255729, ‖∇f‖ = 6.0604e-02, α = 5.15e-01, β = 3.63e-01, nfg = 2
[ Info: CG: iter    8: f = -0.436376560912, ‖∇f‖ = 4.3154e-02, α = 3.66e-01, β = 3.53e-01, nfg = 2
[ Info: CG: iter    9: f = -0.438323587926, ‖∇f‖ = 3.4866e-02, α = 2.49e-01, β = 5.20e-01, nfg = 2
[ Info: CG: iter   10: f = -0.439871747915, ‖∇f‖ = 2.4460e-02, α = 3.84e-01, β = 4.02e-01, nfg = 2
[ Info: CG: iter   11: f = -0.440833392787, ‖∇f‖ = 1.7872e-02, α = 3.89e-01, β = 3.60e-01, nfg = 2
[ Info: CG: iter   12: f = -0.441105694369, ‖∇f‖ = 1.6493e-02, α = 1.27e-01, β = 6.83e-01, nfg = 2
[ Info: CG: iter   13: f = -0.441491345325, ‖∇f‖ = 1.4925e-02, α = 1.90e-01, β = 8.48e-01, nfg = 2
[ Info: CG: iter   14: f = -0.441792424830, ‖∇f‖ = 1.2359e-02, α = 2.98e-01, β = 3.23e-01, nfg = 2
[ Info: CG: iter   15: f = -0.442153471150, ‖∇f‖ = 1.0004e-02, α = 4.72e-01, β = 3.45e-01, nfg = 2
[ Info: CG: iter   16: f = -0.442265625493, ‖∇f‖ = 9.2135e-03, α = 1.70e-01, β = 6.20e-01, nfg = 2
[ Info: CG: iter   17: f = -0.442409712129, ‖∇f‖ = 7.1718e-03, α = 3.16e-01, β = 5.69e-01, nfg = 2
[ Info: CG: iter   18: f = -0.442533970670, ‖∇f‖ = 5.9509e-03, α = 3.59e-01, β = 4.16e-01, nfg = 2
[ Info: CG: iter   19: f = -0.442582601670, ‖∇f‖ = 5.6643e-03, α = 1.19e-01, β = 8.47e-01, nfg = 2
┌ Warning: CG: not converged to requested tol: f = -0.442655552067, ‖∇f‖ = 6.1012e-03
└ @ OptimKit ~/.julia/packages/OptimKit/xpmbV/src/cg.jl:103

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
  <clipPath id="clip330">
    <rect x="0" y="0" width="2400" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip330)" d="M0 1600 L2400 1600 L2400 0 L0 0  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip331">
    <rect x="480" y="0" width="1681" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip330)" d="M219.288 1423.18 L2352.76 1423.18 L2352.76 47.2441 L219.288 47.2441  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip332">
    <rect x="219" y="47" width="2134" height="1377"/>
  </clipPath>
</defs>
<polyline clip-path="url(#clip332)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="279.669,1423.18 279.669,47.2441 "/>
<polyline clip-path="url(#clip332)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="615.12,1423.18 615.12,47.2441 "/>
<polyline clip-path="url(#clip332)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="950.571,1423.18 950.571,47.2441 "/>
<polyline clip-path="url(#clip332)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1286.02,1423.18 1286.02,47.2441 "/>
<polyline clip-path="url(#clip332)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1621.47,1423.18 1621.47,47.2441 "/>
<polyline clip-path="url(#clip332)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1956.92,1423.18 1956.92,47.2441 "/>
<polyline clip-path="url(#clip332)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="2292.37,1423.18 2292.37,47.2441 "/>
<polyline clip-path="url(#clip332)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="219.288,1299.67 2352.76,1299.67 "/>
<polyline clip-path="url(#clip332)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="219.288,941.831 2352.76,941.831 "/>
<polyline clip-path="url(#clip332)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="219.288,583.996 2352.76,583.996 "/>
<polyline clip-path="url(#clip332)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="219.288,226.161 2352.76,226.161 "/>
<polyline clip-path="url(#clip330)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="219.288,1423.18 2352.76,1423.18 "/>
<polyline clip-path="url(#clip330)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="279.669,1423.18 279.669,1404.28 "/>
<polyline clip-path="url(#clip330)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="615.12,1423.18 615.12,1404.28 "/>
<polyline clip-path="url(#clip330)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="950.571,1423.18 950.571,1404.28 "/>
<polyline clip-path="url(#clip330)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1286.02,1423.18 1286.02,1404.28 "/>
<polyline clip-path="url(#clip330)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1621.47,1423.18 1621.47,1404.28 "/>
<polyline clip-path="url(#clip330)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1956.92,1423.18 1956.92,1404.28 "/>
<polyline clip-path="url(#clip330)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="2292.37,1423.18 2292.37,1404.28 "/>
<path clip-path="url(#clip330)" d="M233.431 1454.1 Q229.819 1454.1 227.991 1457.66 Q226.185 1461.2 226.185 1468.33 Q226.185 1475.44 227.991 1479.01 Q229.819 1482.55 233.431 1482.55 Q237.065 1482.55 238.87 1479.01 Q240.699 1475.44 240.699 1468.33 Q240.699 1461.2 238.87 1457.66 Q237.065 1454.1 233.431 1454.1 M233.431 1450.39 Q239.241 1450.39 242.296 1455 Q245.375 1459.58 245.375 1468.33 Q245.375 1477.06 242.296 1481.67 Q239.241 1486.25 233.431 1486.25 Q227.62 1486.25 224.542 1481.67 Q221.486 1477.06 221.486 1468.33 Q221.486 1459.58 224.542 1455 Q227.62 1450.39 233.431 1450.39 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M260.56 1451.02 L264.495 1451.02 L252.458 1489.98 L248.523 1489.98 L260.56 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M276.532 1451.02 L280.467 1451.02 L268.43 1489.98 L264.495 1489.98 L276.532 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M286.347 1481.64 L293.986 1481.64 L293.986 1455.28 L285.676 1456.95 L285.676 1452.69 L293.94 1451.02 L298.615 1451.02 L298.615 1481.64 L306.254 1481.64 L306.254 1485.58 L286.347 1485.58 L286.347 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M312.342 1459.65 L337.18 1459.65 L337.18 1463.91 L333.916 1463.91 L333.916 1479.84 Q333.916 1481.51 334.472 1482.25 Q335.05 1482.96 336.324 1482.96 Q336.671 1482.96 337.18 1482.92 Q337.689 1482.85 337.851 1482.83 L337.851 1485.9 Q337.041 1486.2 336.185 1486.34 Q335.328 1486.48 334.472 1486.48 Q331.694 1486.48 330.629 1484.98 Q329.564 1483.45 329.564 1479.38 L329.564 1463.91 L320.004 1463.91 L320.004 1485.58 L315.652 1485.58 L315.652 1463.91 L312.342 1463.91 L312.342 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M558.65 1481.64 L566.289 1481.64 L566.289 1455.28 L557.979 1456.95 L557.979 1452.69 L566.243 1451.02 L570.919 1451.02 L570.919 1481.64 L578.557 1481.64 L578.557 1485.58 L558.65 1485.58 L558.65 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M594.969 1451.02 L598.905 1451.02 L586.868 1489.98 L582.932 1489.98 L594.969 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M610.942 1451.02 L614.877 1451.02 L602.84 1489.98 L598.905 1489.98 L610.942 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M634.113 1466.95 Q637.469 1467.66 639.344 1469.93 Q641.242 1472.2 641.242 1475.53 Q641.242 1480.65 637.724 1483.45 Q634.205 1486.25 627.724 1486.25 Q625.548 1486.25 623.233 1485.81 Q620.941 1485.39 618.488 1484.54 L618.488 1480.02 Q620.432 1481.16 622.747 1481.74 Q625.062 1482.32 627.585 1482.32 Q631.983 1482.32 634.275 1480.58 Q636.59 1478.84 636.59 1475.53 Q636.59 1472.48 634.437 1470.77 Q632.307 1469.03 628.488 1469.03 L624.46 1469.03 L624.46 1465.19 L628.673 1465.19 Q632.122 1465.19 633.951 1463.82 Q635.779 1462.43 635.779 1459.84 Q635.779 1457.18 633.881 1455.77 Q632.006 1454.33 628.488 1454.33 Q626.566 1454.33 624.367 1454.75 Q622.168 1455.16 619.529 1456.04 L619.529 1451.88 Q622.191 1451.14 624.506 1450.77 Q626.844 1450.39 628.904 1450.39 Q634.228 1450.39 637.33 1452.83 Q640.432 1455.23 640.432 1459.35 Q640.432 1462.22 638.789 1464.21 Q637.145 1466.18 634.113 1466.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M646.752 1459.65 L671.589 1459.65 L671.589 1463.91 L668.325 1463.91 L668.325 1479.84 Q668.325 1481.51 668.881 1482.25 Q669.46 1482.96 670.733 1482.96 Q671.08 1482.96 671.589 1482.92 Q672.099 1482.85 672.261 1482.83 L672.261 1485.9 Q671.45 1486.2 670.594 1486.34 Q669.738 1486.48 668.881 1486.48 Q666.103 1486.48 665.038 1484.98 Q663.974 1483.45 663.974 1479.38 L663.974 1463.91 L654.414 1463.91 L654.414 1485.58 L650.062 1485.58 L650.062 1463.91 L646.752 1463.91 L646.752 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M898.187 1481.64 L914.506 1481.64 L914.506 1485.58 L892.562 1485.58 L892.562 1481.64 Q895.224 1478.89 899.807 1474.26 Q904.414 1469.61 905.594 1468.27 Q907.839 1465.74 908.719 1464.01 Q909.622 1462.25 909.622 1460.56 Q909.622 1457.8 907.677 1456.07 Q905.756 1454.33 902.654 1454.33 Q900.455 1454.33 898.002 1455.09 Q895.571 1455.86 892.793 1457.41 L892.793 1452.69 Q895.617 1451.55 898.071 1450.97 Q900.525 1450.39 902.562 1450.39 Q907.932 1450.39 911.127 1453.08 Q914.321 1455.77 914.321 1460.26 Q914.321 1462.39 913.511 1464.31 Q912.724 1466.2 910.617 1468.8 Q910.039 1469.47 906.937 1472.69 Q903.835 1475.88 898.187 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M931.288 1451.02 L935.224 1451.02 L923.187 1489.98 L919.251 1489.98 L931.288 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M947.261 1451.02 L951.196 1451.02 L939.159 1489.98 L935.224 1489.98 L947.261 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M970.432 1466.95 Q973.788 1467.66 975.663 1469.93 Q977.561 1472.2 977.561 1475.53 Q977.561 1480.65 974.043 1483.45 Q970.524 1486.25 964.043 1486.25 Q961.867 1486.25 959.552 1485.81 Q957.261 1485.39 954.807 1484.54 L954.807 1480.02 Q956.751 1481.16 959.066 1481.74 Q961.381 1482.32 963.904 1482.32 Q968.302 1482.32 970.594 1480.58 Q972.909 1478.84 972.909 1475.53 Q972.909 1472.48 970.756 1470.77 Q968.626 1469.03 964.807 1469.03 L960.779 1469.03 L960.779 1465.19 L964.992 1465.19 Q968.441 1465.19 970.27 1463.82 Q972.098 1462.43 972.098 1459.84 Q972.098 1457.18 970.2 1455.77 Q968.325 1454.33 964.807 1454.33 Q962.885 1454.33 960.686 1454.75 Q958.487 1455.16 955.848 1456.04 L955.848 1451.88 Q958.511 1451.14 960.825 1450.77 Q963.163 1450.39 965.223 1450.39 Q970.547 1450.39 973.649 1452.83 Q976.751 1455.23 976.751 1459.35 Q976.751 1462.22 975.108 1464.21 Q973.464 1466.18 970.432 1466.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M983.071 1459.65 L1007.91 1459.65 L1007.91 1463.91 L1004.64 1463.91 L1004.64 1479.84 Q1004.64 1481.51 1005.2 1482.25 Q1005.78 1482.96 1007.05 1482.96 Q1007.4 1482.96 1007.91 1482.92 Q1008.42 1482.85 1008.58 1482.83 L1008.58 1485.9 Q1007.77 1486.2 1006.91 1486.34 Q1006.06 1486.48 1005.2 1486.48 Q1002.42 1486.48 1001.36 1484.98 Q1000.29 1483.45 1000.29 1479.38 L1000.29 1463.91 L990.733 1463.91 L990.733 1485.58 L986.381 1485.58 L986.381 1463.91 L983.071 1463.91 L983.071 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M1229.55 1481.64 L1237.19 1481.64 L1237.19 1455.28 L1228.88 1456.95 L1228.88 1452.69 L1237.14 1451.02 L1241.82 1451.02 L1241.82 1481.64 L1249.46 1481.64 L1249.46 1485.58 L1229.55 1485.58 L1229.55 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M1265.87 1451.02 L1269.81 1451.02 L1257.77 1489.98 L1253.83 1489.98 L1265.87 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M1281.84 1451.02 L1285.78 1451.02 L1273.74 1489.98 L1269.81 1489.98 L1281.84 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M1291.66 1481.64 L1299.3 1481.64 L1299.3 1455.28 L1290.99 1456.95 L1290.99 1452.69 L1299.25 1451.02 L1303.93 1451.02 L1303.93 1481.64 L1311.57 1481.64 L1311.57 1485.58 L1291.66 1485.58 L1291.66 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M1317.65 1459.65 L1342.49 1459.65 L1342.49 1463.91 L1339.23 1463.91 L1339.23 1479.84 Q1339.23 1481.51 1339.78 1482.25 Q1340.36 1482.96 1341.63 1482.96 Q1341.98 1482.96 1342.49 1482.92 Q1343 1482.85 1343.16 1482.83 L1343.16 1485.9 Q1342.35 1486.2 1341.5 1486.34 Q1340.64 1486.48 1339.78 1486.48 Q1337.01 1486.48 1335.94 1484.98 Q1334.88 1483.45 1334.88 1479.38 L1334.88 1463.91 L1325.32 1463.91 L1325.32 1485.58 L1320.96 1485.58 L1320.96 1463.91 L1317.65 1463.91 L1317.65 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M1578.49 1455.09 L1566.68 1473.54 L1578.49 1473.54 L1578.49 1455.09 M1577.26 1451.02 L1583.14 1451.02 L1583.14 1473.54 L1588.07 1473.54 L1588.07 1477.43 L1583.14 1477.43 L1583.14 1485.58 L1578.49 1485.58 L1578.49 1477.43 L1562.89 1477.43 L1562.89 1472.92 L1577.26 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M1602.77 1451.02 L1606.7 1451.02 L1594.67 1489.98 L1590.73 1489.98 L1602.77 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M1618.74 1451.02 L1622.68 1451.02 L1610.64 1489.98 L1606.7 1489.98 L1618.74 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M1641.91 1466.95 Q1645.27 1467.66 1647.14 1469.93 Q1649.04 1472.2 1649.04 1475.53 Q1649.04 1480.65 1645.52 1483.45 Q1642.01 1486.25 1635.52 1486.25 Q1633.35 1486.25 1631.03 1485.81 Q1628.74 1485.39 1626.29 1484.54 L1626.29 1480.02 Q1628.23 1481.16 1630.55 1481.74 Q1632.86 1482.32 1635.38 1482.32 Q1639.78 1482.32 1642.07 1480.58 Q1644.39 1478.84 1644.39 1475.53 Q1644.39 1472.48 1642.24 1470.77 Q1640.11 1469.03 1636.29 1469.03 L1632.26 1469.03 L1632.26 1465.19 L1636.47 1465.19 Q1639.92 1465.19 1641.75 1463.82 Q1643.58 1462.43 1643.58 1459.84 Q1643.58 1457.18 1641.68 1455.77 Q1639.81 1454.33 1636.29 1454.33 Q1634.37 1454.33 1632.17 1454.75 Q1629.97 1455.16 1627.33 1456.04 L1627.33 1451.88 Q1629.99 1451.14 1632.31 1450.77 Q1634.64 1450.39 1636.7 1450.39 Q1642.03 1450.39 1645.13 1452.83 Q1648.23 1455.23 1648.23 1459.35 Q1648.23 1462.22 1646.59 1464.21 Q1644.94 1466.18 1641.91 1466.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M1654.55 1459.65 L1679.39 1459.65 L1679.39 1463.91 L1676.13 1463.91 L1676.13 1479.84 Q1676.13 1481.51 1676.68 1482.25 Q1677.26 1482.96 1678.53 1482.96 Q1678.88 1482.96 1679.39 1482.92 Q1679.9 1482.85 1680.06 1482.83 L1680.06 1485.9 Q1679.25 1486.2 1678.39 1486.34 Q1677.54 1486.48 1676.68 1486.48 Q1673.9 1486.48 1672.84 1484.98 Q1671.77 1483.45 1671.77 1479.38 L1671.77 1463.91 L1662.21 1463.91 L1662.21 1485.58 L1657.86 1485.58 L1657.86 1463.91 L1654.55 1463.91 L1654.55 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M1900.47 1451.02 L1918.82 1451.02 L1918.82 1454.96 L1904.75 1454.96 L1904.75 1463.43 Q1905.77 1463.08 1906.79 1462.92 Q1907.8 1462.73 1908.82 1462.73 Q1914.61 1462.73 1917.99 1465.9 Q1921.37 1469.08 1921.37 1474.49 Q1921.37 1480.07 1917.9 1483.17 Q1914.42 1486.25 1908.1 1486.25 Q1905.93 1486.25 1903.66 1485.88 Q1901.41 1485.51 1899.01 1484.77 L1899.01 1480.07 Q1901.09 1481.2 1903.31 1481.76 Q1905.54 1482.32 1908.01 1482.32 Q1912.02 1482.32 1914.35 1480.21 Q1916.69 1478.1 1916.69 1474.49 Q1916.69 1470.88 1914.35 1468.77 Q1912.02 1466.67 1908.01 1466.67 Q1906.14 1466.67 1904.26 1467.08 Q1902.41 1467.5 1900.47 1468.38 L1900.47 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M1937.55 1451.02 L1941.48 1451.02 L1929.45 1489.98 L1925.51 1489.98 L1937.55 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M1953.52 1451.02 L1957.46 1451.02 L1945.42 1489.98 L1941.48 1489.98 L1953.52 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M1976.69 1466.95 Q1980.05 1467.66 1981.92 1469.93 Q1983.82 1472.2 1983.82 1475.53 Q1983.82 1480.65 1980.3 1483.45 Q1976.78 1486.25 1970.3 1486.25 Q1968.13 1486.25 1965.81 1485.81 Q1963.52 1485.39 1961.07 1484.54 L1961.07 1480.02 Q1963.01 1481.16 1965.33 1481.74 Q1967.64 1482.32 1970.16 1482.32 Q1974.56 1482.32 1976.85 1480.58 Q1979.17 1478.84 1979.17 1475.53 Q1979.17 1472.48 1977.02 1470.77 Q1974.89 1469.03 1971.07 1469.03 L1967.04 1469.03 L1967.04 1465.19 L1971.25 1465.19 Q1974.7 1465.19 1976.53 1463.82 Q1978.36 1462.43 1978.36 1459.84 Q1978.36 1457.18 1976.46 1455.77 Q1974.59 1454.33 1971.07 1454.33 Q1969.15 1454.33 1966.95 1454.75 Q1964.75 1455.16 1962.11 1456.04 L1962.11 1451.88 Q1964.77 1451.14 1967.09 1450.77 Q1969.42 1450.39 1971.48 1450.39 Q1976.81 1450.39 1979.91 1452.83 Q1983.01 1455.23 1983.01 1459.35 Q1983.01 1462.22 1981.37 1464.21 Q1979.72 1466.18 1976.69 1466.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M1989.33 1459.65 L2014.17 1459.65 L2014.17 1463.91 L2010.9 1463.91 L2010.9 1479.84 Q2010.9 1481.51 2011.46 1482.25 Q2012.04 1482.96 2013.31 1482.96 Q2013.66 1482.96 2014.17 1482.92 Q2014.68 1482.85 2014.84 1482.83 L2014.84 1485.9 Q2014.03 1486.2 2013.17 1486.34 Q2012.32 1486.48 2011.46 1486.48 Q2008.68 1486.48 2007.62 1484.98 Q2006.55 1483.45 2006.55 1479.38 L2006.55 1463.91 L1996.99 1463.91 L1996.99 1485.58 L1992.64 1485.58 L1992.64 1463.91 L1989.33 1463.91 L1989.33 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M2239.99 1481.64 L2256.31 1481.64 L2256.31 1485.58 L2234.37 1485.58 L2234.37 1481.64 Q2237.03 1478.89 2241.61 1474.26 Q2246.22 1469.61 2247.4 1468.27 Q2249.64 1465.74 2250.52 1464.01 Q2251.43 1462.25 2251.43 1460.56 Q2251.43 1457.8 2249.48 1456.07 Q2247.56 1454.33 2244.46 1454.33 Q2242.26 1454.33 2239.81 1455.09 Q2237.38 1455.86 2234.6 1457.41 L2234.6 1452.69 Q2237.42 1451.55 2239.88 1450.97 Q2242.33 1450.39 2244.37 1450.39 Q2249.74 1450.39 2252.93 1453.08 Q2256.12 1455.77 2256.12 1460.26 Q2256.12 1462.39 2255.31 1464.31 Q2254.53 1466.2 2252.42 1468.8 Q2251.84 1469.47 2248.74 1472.69 Q2245.64 1475.88 2239.99 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M2273.09 1451.02 L2277.03 1451.02 L2264.99 1489.98 L2261.06 1489.98 L2273.09 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M2289.06 1451.02 L2293 1451.02 L2280.96 1489.98 L2277.03 1489.98 L2289.06 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M2298.88 1481.64 L2306.52 1481.64 L2306.52 1455.28 L2298.21 1456.95 L2298.21 1452.69 L2306.47 1451.02 L2311.15 1451.02 L2311.15 1481.64 L2318.79 1481.64 L2318.79 1485.58 L2298.88 1485.58 L2298.88 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M2324.87 1459.65 L2349.71 1459.65 L2349.71 1463.91 L2346.45 1463.91 L2346.45 1479.84 Q2346.45 1481.51 2347 1482.25 Q2347.58 1482.96 2348.86 1482.96 Q2349.2 1482.96 2349.71 1482.92 Q2350.22 1482.85 2350.38 1482.83 L2350.38 1485.9 Q2349.57 1486.2 2348.72 1486.34 Q2347.86 1486.48 2347 1486.48 Q2344.23 1486.48 2343.16 1484.98 Q2342.1 1483.45 2342.1 1479.38 L2342.1 1463.91 L2332.54 1463.91 L2332.54 1485.58 L2328.18 1485.58 L2328.18 1463.91 L2324.87 1463.91 L2324.87 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M1296.14 1545.45 L1275.87 1545.45 Q1276.35 1554.96 1278.54 1559 Q1281.28 1563.97 1286.02 1563.97 Q1290.8 1563.97 1293.44 1558.97 Q1295.76 1554.58 1296.14 1545.45 M1296.05 1540.03 Q1295.16 1531 1293.44 1527.81 Q1290.7 1522.78 1286.02 1522.78 Q1281.15 1522.78 1278.57 1527.75 Q1276.54 1531.76 1275.93 1540.03 L1296.05 1540.03 M1286.02 1518.01 Q1293.66 1518.01 1298.02 1524.76 Q1302.38 1531.47 1302.38 1543.38 Q1302.38 1555.25 1298.02 1562 Q1293.66 1568.78 1286.02 1568.78 Q1278.35 1568.78 1274.02 1562 Q1269.66 1555.25 1269.66 1543.38 Q1269.66 1531.47 1274.02 1524.76 Q1278.35 1518.01 1286.02 1518.01 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><polyline clip-path="url(#clip330)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="219.288,1423.18 219.288,47.2441 "/>
<polyline clip-path="url(#clip330)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="219.288,1299.67 238.185,1299.67 "/>
<polyline clip-path="url(#clip330)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="219.288,941.831 238.185,941.831 "/>
<polyline clip-path="url(#clip330)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="219.288,583.996 238.185,583.996 "/>
<polyline clip-path="url(#clip330)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="219.288,226.161 238.185,226.161 "/>
<path clip-path="url(#clip330)" d="M127.015 1285.46 Q123.404 1285.46 121.575 1289.03 Q119.769 1292.57 119.769 1299.7 Q119.769 1306.81 121.575 1310.37 Q123.404 1313.91 127.015 1313.91 Q130.649 1313.91 132.455 1310.37 Q134.283 1306.81 134.283 1299.7 Q134.283 1292.57 132.455 1289.03 Q130.649 1285.46 127.015 1285.46 M127.015 1281.76 Q132.825 1281.76 135.88 1286.37 Q138.959 1290.95 138.959 1299.7 Q138.959 1308.43 135.88 1313.03 Q132.825 1317.62 127.015 1317.62 Q121.205 1317.62 118.126 1313.03 Q115.07 1308.43 115.07 1299.7 Q115.07 1290.95 118.126 1286.37 Q121.205 1281.76 127.015 1281.76 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M147.177 1311.07 L152.061 1311.07 L152.061 1316.95 L147.177 1316.95 L147.177 1311.07 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M161.065 1282.39 L183.288 1282.39 L183.288 1284.38 L170.741 1316.95 L165.857 1316.95 L177.663 1286.32 L161.065 1286.32 L161.065 1282.39 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M126.205 927.629 Q122.593 927.629 120.765 931.194 Q118.959 934.736 118.959 941.865 Q118.959 948.972 120.765 952.537 Q122.593 956.078 126.205 956.078 Q129.839 956.078 131.644 952.537 Q133.473 948.972 133.473 941.865 Q133.473 934.736 131.644 931.194 Q129.839 927.629 126.205 927.629 M126.205 923.926 Q132.015 923.926 135.07 928.532 Q138.149 933.116 138.149 941.865 Q138.149 950.592 135.07 955.199 Q132.015 959.782 126.205 959.782 Q120.394 959.782 117.316 955.199 Q114.26 950.592 114.26 941.865 Q114.26 933.116 117.316 928.532 Q120.394 923.926 126.205 923.926 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M146.366 953.231 L151.251 953.231 L151.251 959.111 L146.366 959.111 L146.366 953.231 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M171.436 942.699 Q168.102 942.699 166.181 944.481 Q164.283 946.264 164.283 949.389 Q164.283 952.514 166.181 954.296 Q168.102 956.078 171.436 956.078 Q174.769 956.078 176.69 954.296 Q178.612 952.49 178.612 949.389 Q178.612 946.264 176.69 944.481 Q174.792 942.699 171.436 942.699 M166.76 940.708 Q163.751 939.967 162.061 937.907 Q160.394 935.847 160.394 932.884 Q160.394 928.741 163.334 926.333 Q166.297 923.926 171.436 923.926 Q176.598 923.926 179.538 926.333 Q182.477 928.741 182.477 932.884 Q182.477 935.847 180.788 937.907 Q179.121 939.967 176.135 940.708 Q179.514 941.495 181.389 943.787 Q183.288 946.078 183.288 949.389 Q183.288 954.412 180.209 957.097 Q177.153 959.782 171.436 959.782 Q165.718 959.782 162.64 957.097 Q159.584 954.412 159.584 949.389 Q159.584 946.078 161.482 943.787 Q163.38 941.495 166.76 940.708 M165.047 933.324 Q165.047 936.009 166.714 937.514 Q168.403 939.018 171.436 939.018 Q174.445 939.018 176.135 937.514 Q177.848 936.009 177.848 933.324 Q177.848 930.639 176.135 929.134 Q174.445 927.629 171.436 927.629 Q168.403 927.629 166.714 929.134 Q165.047 930.639 165.047 933.324 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M126.297 569.795 Q122.686 569.795 120.857 573.36 Q119.052 576.901 119.052 584.031 Q119.052 591.137 120.857 594.702 Q122.686 598.244 126.297 598.244 Q129.931 598.244 131.737 594.702 Q133.566 591.137 133.566 584.031 Q133.566 576.901 131.737 573.36 Q129.931 569.795 126.297 569.795 M126.297 566.091 Q132.107 566.091 135.163 570.698 Q138.242 575.281 138.242 584.031 Q138.242 592.758 135.163 597.364 Q132.107 601.947 126.297 601.947 Q120.487 601.947 117.408 597.364 Q114.353 592.758 114.353 584.031 Q114.353 575.281 117.408 570.698 Q120.487 566.091 126.297 566.091 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M146.459 595.396 L151.343 595.396 L151.343 601.276 L146.459 601.276 L146.459 595.396 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M161.667 600.558 L161.667 596.299 Q163.427 597.133 165.232 597.572 Q167.038 598.012 168.774 598.012 Q173.403 598.012 175.834 594.91 Q178.288 591.785 178.635 585.443 Q177.292 587.434 175.232 588.498 Q173.172 589.563 170.672 589.563 Q165.487 589.563 162.454 586.438 Q159.445 583.29 159.445 577.85 Q159.445 572.526 162.593 569.309 Q165.741 566.091 170.973 566.091 Q176.968 566.091 180.116 570.698 Q183.288 575.281 183.288 584.031 Q183.288 592.202 179.399 597.086 Q175.533 601.947 168.982 601.947 Q167.223 601.947 165.417 601.6 Q163.612 601.253 161.667 600.558 M170.973 585.906 Q174.121 585.906 175.95 583.753 Q177.801 581.6 177.801 577.85 Q177.801 574.123 175.95 571.971 Q174.121 569.795 170.973 569.795 Q167.825 569.795 165.973 571.971 Q164.144 574.123 164.144 577.85 Q164.144 581.6 165.973 583.753 Q167.825 585.906 170.973 585.906 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M116.922 239.506 L124.561 239.506 L124.561 213.141 L116.251 214.807 L116.251 210.548 L124.515 208.881 L129.191 208.881 L129.191 239.506 L136.829 239.506 L136.829 243.441 L116.922 243.441 L116.922 239.506 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M146.274 237.562 L151.158 237.562 L151.158 243.441 L146.274 243.441 L146.274 237.562 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M171.343 211.96 Q167.732 211.96 165.903 215.525 Q164.098 219.067 164.098 226.196 Q164.098 233.303 165.903 236.867 Q167.732 240.409 171.343 240.409 Q174.977 240.409 176.783 236.867 Q178.612 233.303 178.612 226.196 Q178.612 219.067 176.783 215.525 Q174.977 211.96 171.343 211.96 M171.343 208.256 Q177.153 208.256 180.209 212.863 Q183.288 217.446 183.288 226.196 Q183.288 234.923 180.209 239.529 Q177.153 244.113 171.343 244.113 Q165.533 244.113 162.454 239.529 Q159.399 234.923 159.399 226.196 Q159.399 217.446 162.454 212.863 Q165.533 208.256 171.343 208.256 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M33.8307 724.772 Q33.2578 725.759 33.0032 726.937 Q32.7167 728.082 32.7167 729.483 Q32.7167 734.448 35.9632 737.122 Q39.1779 739.763 45.2253 739.763 L64.0042 739.763 L64.0042 745.652 L28.3562 745.652 L28.3562 739.763 L33.8944 739.763 Q30.6479 737.917 29.0883 734.957 Q27.4968 731.997 27.4968 727.764 Q27.4968 727.159 27.5923 726.427 Q27.656 725.695 27.8151 724.804 L33.8307 724.772 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><line clip-path="url(#clip332)" x1="279.669" y1="226.161" x2="279.669" y2="210.161" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="279.669" y1="226.161" x2="263.669" y2="226.161" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="279.669" y1="226.161" x2="279.669" y2="242.161" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="279.669" y1="226.161" x2="295.669" y2="226.161" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1286.02" y1="226.234" x2="1286.02" y2="210.234" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1286.02" y1="226.234" x2="1270.02" y2="226.234" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1286.02" y1="226.234" x2="1286.02" y2="242.234" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1286.02" y1="226.234" x2="1302.02" y2="226.234" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="780.88" y1="367.905" x2="780.88" y2="351.905" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="780.88" y1="367.905" x2="764.88" y2="367.905" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="780.88" y1="367.905" x2="780.88" y2="383.905" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="780.88" y1="367.905" x2="796.88" y2="367.905" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1791.16" y1="367.905" x2="1791.16" y2="351.905" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1791.16" y1="367.905" x2="1775.16" y2="367.905" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1791.16" y1="367.905" x2="1791.16" y2="383.905" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1791.16" y1="367.905" x2="1807.16" y2="367.905" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="784.81" y1="367.912" x2="784.81" y2="351.912" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="784.81" y1="367.912" x2="768.81" y2="367.912" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="784.81" y1="367.912" x2="784.81" y2="383.912" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="784.81" y1="367.912" x2="800.81" y2="367.912" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1787.23" y1="367.912" x2="1787.23" y2="351.912" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1787.23" y1="367.912" x2="1771.23" y2="367.912" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1787.23" y1="367.912" x2="1787.23" y2="383.912" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1787.23" y1="367.912" x2="1803.23" y2="367.912" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="280.884" y1="765.449" x2="280.884" y2="749.449" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="280.884" y1="765.449" x2="264.884" y2="765.449" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="280.884" y1="765.449" x2="280.884" y2="781.449" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="280.884" y1="765.449" x2="296.884" y2="765.449" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="2291.16" y1="765.449" x2="2291.16" y2="749.449" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="2291.16" y1="765.449" x2="2275.16" y2="765.449" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="2291.16" y1="765.449" x2="2291.16" y2="781.449" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="2291.16" y1="765.449" x2="2307.16" y2="765.449" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1284.81" y1="765.457" x2="1284.81" y2="749.457" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1284.81" y1="765.457" x2="1268.81" y2="765.457" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1284.81" y1="765.457" x2="1284.81" y2="781.457" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1284.81" y1="765.457" x2="1300.81" y2="765.457" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1287.24" y1="765.457" x2="1287.24" y2="749.457" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1287.24" y1="765.457" x2="1271.24" y2="765.457" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1287.24" y1="765.457" x2="1287.24" y2="781.457" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1287.24" y1="765.457" x2="1303.24" y2="765.457" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1286.02" y1="857.365" x2="1286.02" y2="841.365" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1286.02" y1="857.365" x2="1270.02" y2="857.365" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1286.02" y1="857.365" x2="1286.02" y2="873.365" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1286.02" y1="857.365" x2="1302.02" y2="857.365" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="279.669" y1="857.453" x2="279.669" y2="841.453" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="279.669" y1="857.453" x2="263.669" y2="857.453" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="279.669" y1="857.453" x2="279.669" y2="873.453" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="279.669" y1="857.453" x2="295.669" y2="857.453" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="785.255" y1="962.274" x2="785.255" y2="946.274" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="785.255" y1="962.274" x2="769.255" y2="962.274" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="785.255" y1="962.274" x2="785.255" y2="978.274" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="785.255" y1="962.274" x2="801.255" y2="962.274" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1786.79" y1="962.274" x2="1786.79" y2="946.274" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1786.79" y1="962.274" x2="1770.79" y2="962.274" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1786.79" y1="962.274" x2="1786.79" y2="978.274" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1786.79" y1="962.274" x2="1802.79" y2="962.274" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1791.61" y1="962.316" x2="1791.61" y2="946.316" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1791.61" y1="962.316" x2="1775.61" y2="962.316" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1791.61" y1="962.316" x2="1791.61" y2="978.316" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1791.61" y1="962.316" x2="1807.61" y2="962.316" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="780.429" y1="962.316" x2="780.429" y2="946.316" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="780.429" y1="962.316" x2="764.429" y2="962.316" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="780.429" y1="962.316" x2="780.429" y2="978.316" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="780.429" y1="962.316" x2="796.429" y2="962.316" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="2292.37" y1="1312.15" x2="2292.37" y2="1296.15" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="2292.37" y1="1312.15" x2="2276.37" y2="1312.15" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="2292.37" y1="1312.15" x2="2292.37" y2="1328.15" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="2292.37" y1="1312.15" x2="2308.37" y2="1312.15" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1286.02" y1="1313.07" x2="1286.02" y2="1297.07" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1286.02" y1="1313.07" x2="1270.02" y2="1313.07" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1286.02" y1="1313.07" x2="1286.02" y2="1329.07" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1286.02" y1="1313.07" x2="1302.02" y2="1313.07" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="784.895" y1="1423.07" x2="784.895" y2="1407.07" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="784.895" y1="1423.07" x2="768.895" y2="1423.07" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="784.895" y1="1423.07" x2="784.895" y2="1439.07" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="784.895" y1="1423.07" x2="800.895" y2="1423.07" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1787.15" y1="1423.07" x2="1787.15" y2="1407.07" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1787.15" y1="1423.07" x2="1771.15" y2="1423.07" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1787.15" y1="1423.07" x2="1787.15" y2="1439.07" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1787.15" y1="1423.07" x2="1803.15" y2="1423.07" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="780.837" y1="1423.18" x2="780.837" y2="1407.18" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="780.837" y1="1423.18" x2="764.837" y2="1423.18" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="780.837" y1="1423.18" x2="780.837" y2="1439.18" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="780.837" y1="1423.18" x2="796.837" y2="1423.18" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<path clip-path="url(#clip330)" d="M1897.14 196.789 L2281.64 196.789 L2281.64 93.1086 L1897.14 93.1086  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<polyline clip-path="url(#clip330)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1897.14,196.789 2281.64,196.789 2281.64,93.1086 1897.14,93.1086 1897.14,196.789 "/>
<line clip-path="url(#clip330)" x1="1991.96" y1="144.949" x2="1991.96" y2="122.193" style="stroke:#009af9; stroke-width:4.55111; stroke-opacity:1"/>
<line clip-path="url(#clip330)" x1="1991.96" y1="144.949" x2="1969.2" y2="144.949" style="stroke:#009af9; stroke-width:4.55111; stroke-opacity:1"/>
<line clip-path="url(#clip330)" x1="1991.96" y1="144.949" x2="1991.96" y2="167.704" style="stroke:#009af9; stroke-width:4.55111; stroke-opacity:1"/>
<line clip-path="url(#clip330)" x1="1991.96" y1="144.949" x2="2014.71" y2="144.949" style="stroke:#009af9; stroke-width:4.55111; stroke-opacity:1"/>
<path clip-path="url(#clip330)" d="M2086.78 127.669 L2116.01 127.669 L2116.01 131.604 L2103.75 131.604 L2103.75 162.229 L2099.05 162.229 L2099.05 131.604 L2086.78 131.604 L2086.78 127.669 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M2128.4 140.284 Q2127.68 139.868 2126.82 139.682 Q2125.99 139.474 2124.97 139.474 Q2121.36 139.474 2119.42 141.835 Q2117.5 144.173 2117.5 148.571 L2117.5 162.229 L2113.21 162.229 L2113.21 136.303 L2117.5 136.303 L2117.5 140.331 Q2118.84 137.969 2120.99 136.835 Q2123.14 135.678 2126.22 135.678 Q2126.66 135.678 2127.2 135.747 Q2127.73 135.794 2128.38 135.909 L2128.4 140.284 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M2132.87 136.303 L2137.13 136.303 L2137.13 162.229 L2132.87 162.229 L2132.87 136.303 M2132.87 126.21 L2137.13 126.21 L2137.13 131.604 L2132.87 131.604 L2132.87 126.21 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M2142.98 136.303 L2147.5 136.303 L2155.6 158.062 L2163.7 136.303 L2168.21 136.303 L2158.49 162.229 L2152.7 162.229 L2142.98 136.303 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M2174.09 136.303 L2178.35 136.303 L2178.35 162.229 L2174.09 162.229 L2174.09 136.303 M2174.09 126.21 L2178.35 126.21 L2178.35 131.604 L2174.09 131.604 L2174.09 126.21 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M2199.05 149.196 Q2193.88 149.196 2191.89 150.377 Q2189.9 151.557 2189.9 154.405 Q2189.9 156.673 2191.38 158.016 Q2192.89 159.335 2195.46 159.335 Q2199 159.335 2201.13 156.835 Q2203.28 154.312 2203.28 150.145 L2203.28 149.196 L2199.05 149.196 M2207.54 147.437 L2207.54 162.229 L2203.28 162.229 L2203.28 158.293 Q2201.82 160.655 2199.65 161.789 Q2197.47 162.9 2194.32 162.9 Q2190.34 162.9 2187.98 160.678 Q2185.64 158.432 2185.64 154.682 Q2185.64 150.307 2188.56 148.085 Q2191.5 145.863 2197.31 145.863 L2203.28 145.863 L2203.28 145.446 Q2203.28 142.507 2201.34 140.909 Q2199.42 139.289 2195.92 139.289 Q2193.7 139.289 2191.59 139.821 Q2189.49 140.354 2187.54 141.419 L2187.54 137.483 Q2189.88 136.581 2192.08 136.141 Q2194.28 135.678 2196.36 135.678 Q2201.99 135.678 2204.76 138.594 Q2207.54 141.511 2207.54 147.437 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M2216.31 126.21 L2220.57 126.21 L2220.57 162.229 L2216.31 162.229 L2216.31 126.21 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M2239.72 126.257 Q2236.62 131.581 2235.11 136.789 Q2233.61 141.997 2233.61 147.344 Q2233.61 152.692 2235.11 157.946 Q2236.64 163.178 2239.72 168.479 L2236.01 168.479 Q2232.54 163.039 2230.81 157.784 Q2229.09 152.53 2229.09 147.344 Q2229.09 142.182 2230.81 136.951 Q2232.52 131.72 2236.01 126.257 L2239.72 126.257 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M2247.31 126.257 L2251.01 126.257 Q2254.49 131.72 2256.2 136.951 Q2257.94 142.182 2257.94 147.344 Q2257.94 152.53 2256.2 157.784 Q2254.49 163.039 2251.01 168.479 L2247.31 168.479 Q2250.39 163.178 2251.89 157.946 Q2253.42 152.692 2253.42 147.344 Q2253.42 141.997 2251.89 136.789 Q2250.39 131.581 2247.31 126.257 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /></svg>

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
[ Info: VUMPS init:	obj = +4.994527818488e-01	err = 4.7959e-02
[ Info: VUMPS   1:	obj = -5.224213331967e-01	err = 3.1869497986e-01	time = 0.07 sec
[ Info: VUMPS   2:	obj = -8.703292449817e-01	err = 9.7881092516e-02	time = 0.04 sec
[ Info: VUMPS   3:	obj = -8.853872271509e-01	err = 1.1633880994e-02	time = 0.05 sec
[ Info: VUMPS   4:	obj = -8.859759226381e-01	err = 5.0303292009e-03	time = 0.06 sec
[ Info: VUMPS   5:	obj = -8.861272827748e-01	err = 4.0977568854e-03	time = 0.03 sec
[ Info: VUMPS   6:	obj = -8.861876892553e-01	err = 2.8143569786e-03	time = 0.06 sec
[ Info: VUMPS   7:	obj = -8.862118265423e-01	err = 2.4768815055e-03	time = 0.07 sec
[ Info: VUMPS   8:	obj = -8.862240634280e-01	err = 2.1529722171e-03	time = 0.07 sec
[ Info: VUMPS   9:	obj = -8.862295952770e-01	err = 2.0143660733e-03	time = 0.08 sec
[ Info: VUMPS  10:	obj = -8.862325590423e-01	err = 1.9696799082e-03	time = 0.07 sec
[ Info: VUMPS  11:	obj = -8.862339539915e-01	err = 1.9072844137e-03	time = 0.07 sec
[ Info: VUMPS  12:	obj = -8.862347209414e-01	err = 1.9329517027e-03	time = 0.04 sec
[ Info: VUMPS  13:	obj = -8.862350788779e-01	err = 1.8912383017e-03	time = 0.08 sec
[ Info: VUMPS  14:	obj = -8.862353072621e-01	err = 1.9119217581e-03	time = 0.07 sec
[ Info: VUMPS  15:	obj = -8.862354080095e-01	err = 1.8651910333e-03	time = 0.07 sec
[ Info: VUMPS  16:	obj = -8.862354997154e-01	err = 1.8780338112e-03	time = 0.07 sec
[ Info: VUMPS  17:	obj = -8.862355500667e-01	err = 1.8132834599e-03	time = 0.06 sec
[ Info: VUMPS  18:	obj = -8.862356079518e-01	err = 1.8176447019e-03	time = 0.07 sec
[ Info: VUMPS  19:	obj = -8.862356026529e-01	err = 1.7897346650e-03	time = 0.06 sec
[ Info: VUMPS  20:	obj = -8.862356494807e-01	err = 1.7869452100e-03	time = 0.05 sec
[ Info: VUMPS  21:	obj = -8.862357156491e-01	err = 1.7171781823e-03	time = 0.07 sec
[ Info: VUMPS  22:	obj = -8.862356705107e-01	err = 1.7618627080e-03	time = 0.05 sec
[ Info: VUMPS  23:	obj = -8.862356847446e-01	err = 1.7345893329e-03	time = 0.06 sec
[ Info: VUMPS  24:	obj = -8.862356597631e-01	err = 1.7698180368e-03	time = 0.06 sec
[ Info: VUMPS  25:	obj = -8.862356369663e-01	err = 1.7832515198e-03	time = 0.07 sec
[ Info: VUMPS  26:	obj = -8.862355639238e-01	err = 1.8610433054e-03	time = 0.07 sec
[ Info: VUMPS  27:	obj = -8.862355966939e-01	err = 1.8414491439e-03	time = 0.07 sec
[ Info: VUMPS  28:	obj = -8.862354448304e-01	err = 1.9601069224e-03	time = 0.07 sec
[ Info: VUMPS  29:	obj = -8.862353997121e-01	err = 2.0323064685e-03	time = 0.04 sec
[ Info: VUMPS  30:	obj = -8.862352867417e-01	err = 2.0852298701e-03	time = 0.06 sec
[ Info: VUMPS  31:	obj = -8.862350688513e-01	err = 2.3296901858e-03	time = 0.08 sec
[ Info: VUMPS  32:	obj = -8.862348370300e-01	err = 2.4308916811e-03	time = 0.08 sec
[ Info: VUMPS  33:	obj = -8.862348650705e-01	err = 2.5264912559e-03	time = 0.07 sec
[ Info: VUMPS  34:	obj = -8.862341340353e-01	err = 2.8806808793e-03	time = 0.07 sec
[ Info: VUMPS  35:	obj = -8.862340510957e-01	err = 3.1043746978e-03	time = 0.07 sec
[ Info: VUMPS  36:	obj = -8.862334585911e-01	err = 3.2535241129e-03	time = 0.07 sec
[ Info: VUMPS  37:	obj = -8.862322313415e-01	err = 4.0963820560e-03	time = 0.05 sec
[ Info: VUMPS  38:	obj = -8.862328105793e-01	err = 3.5663760522e-03	time = 0.05 sec
[ Info: VUMPS  39:	obj = -8.862317650974e-01	err = 4.3666283036e-03	time = 0.08 sec
[ Info: VUMPS  40:	obj = -8.862334734959e-01	err = 3.2172539857e-03	time = 0.06 sec
[ Info: VUMPS  41:	obj = -8.862328031496e-01	err = 4.0602562223e-03	time = 0.07 sec
[ Info: VUMPS  42:	obj = -8.862348928428e-01	err = 2.8096230226e-03	time = 0.06 sec
[ Info: VUMPS  43:	obj = -8.862354347226e-01	err = 2.9469451252e-03	time = 0.07 sec
[ Info: VUMPS  44:	obj = -8.862369582324e-01	err = 2.0065666373e-03	time = 0.03 sec
[ Info: VUMPS  45:	obj = -8.862376214330e-01	err = 1.5332753632e-03	time = 0.07 sec
[ Info: VUMPS  46:	obj = -8.862381576736e-01	err = 9.5235621400e-04	time = 0.07 sec
[ Info: VUMPS  47:	obj = -8.862383675148e-01	err = 5.9064456193e-04	time = 0.06 sec
[ Info: VUMPS  48:	obj = -8.862384608233e-01	err = 3.4391997505e-04	time = 0.08 sec
[ Info: VUMPS  49:	obj = -8.862384963189e-01	err = 1.9914113331e-04	time = 0.07 sec
[ Info: VUMPS  50:	obj = -8.862385106659e-01	err = 1.1648442343e-04	time = 0.06 sec
[ Info: VUMPS  51:	obj = -8.862385173677e-01	err = 7.0216183061e-05	time = 0.05 sec
[ Info: VUMPS  52:	obj = -8.862385211516e-01	err = 5.4936268184e-05	time = 0.07 sec
[ Info: VUMPS  53:	obj = -8.862385239261e-01	err = 5.0775181524e-05	time = 0.05 sec
[ Info: VUMPS  54:	obj = -8.862385263536e-01	err = 4.9093824356e-05	time = 0.06 sec
[ Info: VUMPS  55:	obj = -8.862385287510e-01	err = 4.7160784987e-05	time = 0.05 sec
[ Info: VUMPS  56:	obj = -8.862385312350e-01	err = 4.7356857246e-05	time = 0.06 sec
[ Info: VUMPS  57:	obj = -8.862385338978e-01	err = 4.6554878190e-05	time = 0.07 sec
[ Info: VUMPS  58:	obj = -8.862385367826e-01	err = 4.7893015679e-05	time = 0.05 sec
[ Info: VUMPS  59:	obj = -8.862385399464e-01	err = 4.7866639440e-05	time = 0.08 sec
[ Info: VUMPS  60:	obj = -8.862385434317e-01	err = 5.0108148399e-05	time = 0.08 sec
[ Info: VUMPS  61:	obj = -8.862385472946e-01	err = 5.0701719642e-05	time = 0.06 sec
[ Info: VUMPS  62:	obj = -8.862385515924e-01	err = 5.3747205154e-05	time = 0.06 sec
[ Info: VUMPS  63:	obj = -8.862385563928e-01	err = 5.4917836680e-05	time = 0.07 sec
[ Info: VUMPS  64:	obj = -8.862385617762e-01	err = 5.8736952243e-05	time = 0.07 sec
[ Info: VUMPS  65:	obj = -8.862385678319e-01	err = 6.0517251626e-05	time = 0.04 sec
[ Info: VUMPS  66:	obj = -8.862385746724e-01	err = 6.5122279370e-05	time = 0.05 sec
[ Info: VUMPS  67:	obj = -8.862385824209e-01	err = 6.7603807255e-05	time = 0.08 sec
[ Info: VUMPS  68:	obj = -8.862385912344e-01	err = 7.3037975820e-05	time = 0.06 sec
[ Info: VUMPS  69:	obj = -8.862386012855e-01	err = 7.6356824733e-05	time = 0.06 sec
[ Info: VUMPS  70:	obj = -8.862386127926e-01	err = 8.2675024543e-05	time = 0.08 sec
[ Info: VUMPS  71:	obj = -8.862386259979e-01	err = 8.7080551030e-05	time = 0.08 sec
[ Info: VUMPS  72:	obj = -8.862386412013e-01	err = 9.4240413691e-05	time = 0.07 sec
[ Info: VUMPS  73:	obj = -8.862386587357e-01	err = 1.0001061678e-04	time = 0.06 sec
[ Info: VUMPS  74:	obj = -8.862386790008e-01	err = 1.0787560414e-04	time = 0.05 sec
[ Info: VUMPS  75:	obj = -8.862387024327e-01	err = 1.1491628128e-04	time = 0.07 sec
[ Info: VUMPS  76:	obj = -8.862387295295e-01	err = 1.2350014063e-04	time = 0.07 sec
[ Info: VUMPS  77:	obj = -8.862387608070e-01	err = 1.3154045663e-04	time = 0.08 sec
[ Info: VUMPS  78:	obj = -8.862387968010e-01	err = 1.4054217709e-04	time = 0.08 sec
[ Info: VUMPS  79:	obj = -8.862388379935e-01	err = 1.4918434598e-04	time = 0.08 sec
[ Info: VUMPS  80:	obj = -8.862388847728e-01	err = 1.5756496092e-04	time = 0.06 sec
[ Info: VUMPS  81:	obj = -8.862389373185e-01	err = 1.6556206078e-04	time = 0.05 sec
[ Info: VUMPS  82:	obj = -8.862389955195e-01	err = 1.7207041444e-04	time = 0.05 sec
[ Info: VUMPS  83:	obj = -8.862390588490e-01	err = 1.7709969160e-04	time = 0.06 sec
[ Info: VUMPS  84:	obj = -8.862391263089e-01	err = 1.8017689240e-04	time = 0.07 sec
[ Info: VUMPS  85:	obj = -8.862391964083e-01	err = 1.8083364727e-04	time = 0.06 sec
[ Info: VUMPS  86:	obj = -8.862392672643e-01	err = 1.7896061249e-04	time = 0.06 sec
[ Info: VUMPS  87:	obj = -8.862393367843e-01	err = 1.7435281981e-04	time = 0.07 sec
[ Info: VUMPS  88:	obj = -8.862394029305e-01	err = 1.6733550307e-04	time = 0.07 sec
[ Info: VUMPS  89:	obj = -8.862394639791e-01	err = 1.5822580553e-04	time = 0.08 sec
[ Info: VUMPS  90:	obj = -8.862395187139e-01	err = 1.4763647565e-04	time = 0.05 sec
[ Info: VUMPS  91:	obj = -8.862395665083e-01	err = 1.3613348343e-04	time = 0.07 sec
[ Info: VUMPS  92:	obj = -8.862396072866e-01	err = 1.2433109567e-04	time = 0.06 sec
[ Info: VUMPS  93:	obj = -8.862396414074e-01	err = 1.1271049944e-04	time = 0.09 sec
[ Info: VUMPS  94:	obj = -8.862396695130e-01	err = 1.0166193044e-04	time = 0.08 sec
[ Info: VUMPS  95:	obj = -8.862396923871e-01	err = 9.1428430976e-05	time = 0.09 sec
[ Info: VUMPS  96:	obj = -8.862397108436e-01	err = 8.2152207117e-05	time = 0.08 sec
[ Info: VUMPS  97:	obj = -8.862397256537e-01	err = 7.3876229899e-05	time = 0.06 sec
[ Info: VUMPS  98:	obj = -8.862397375053e-01	err = 6.6584753895e-05	time = 0.07 sec
[ Info: VUMPS  99:	obj = -8.862397469874e-01	err = 6.0216703928e-05	time = 0.07 sec
┌ Warning: VUMPS cancel 100:	obj = -8.862397545895e-01	err = 5.4690768674e-05	time = 6.50 sec
└ @ MPSKit ~/Projects/MPSKit.jl/src/algorithms/groundstate/vumps.jl:71

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
  <clipPath id="clip360">
    <rect x="0" y="0" width="2400" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip360)" d="M0 1600 L2400 1600 L2400 0 L0 0  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip361">
    <rect x="480" y="0" width="1681" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip360)" d="M189.496 1352.62 L2352.76 1352.62 L2352.76 123.472 L189.496 123.472  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip362">
    <rect x="189" y="123" width="2164" height="1230"/>
  </clipPath>
</defs>
<polyline clip-path="url(#clip362)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="250.72,1352.62 250.72,123.472 "/>
<polyline clip-path="url(#clip362)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="189.496,1330.99 2352.76,1330.99 "/>
<polyline clip-path="url(#clip362)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="189.496,1040.81 2352.76,1040.81 "/>
<polyline clip-path="url(#clip362)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="189.496,750.632 2352.76,750.632 "/>
<polyline clip-path="url(#clip362)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="189.496,460.451 2352.76,460.451 "/>
<polyline clip-path="url(#clip362)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="189.496,170.271 2352.76,170.271 "/>
<polyline clip-path="url(#clip360)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="189.496,1352.62 2352.76,1352.62 "/>
<polyline clip-path="url(#clip360)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="250.72,1352.62 250.72,1371.52 "/>
<path clip-path="url(#clip360)" d="M117.476 1508.55 L138.148 1487.88 L140.931 1490.66 L132.256 1499.34 L153.911 1520.99 L150.588 1524.32 L128.933 1502.66 L120.258 1511.34 L117.476 1508.55 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M155.826 1488.04 Q155.024 1488.26 154.287 1488.73 Q153.551 1489.17 152.831 1489.89 Q150.277 1492.45 150.572 1495.49 Q150.867 1498.5 153.976 1501.61 L163.634 1511.27 L160.606 1514.3 L142.273 1495.97 L145.301 1492.94 L148.149 1495.79 Q147.429 1493.17 148.149 1490.84 Q148.853 1488.5 151.03 1486.33 Q151.341 1486.01 151.767 1485.69 Q152.176 1485.34 152.716 1484.97 L155.826 1488.04 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M156.17 1482.07 L159.182 1479.06 L177.514 1497.39 L174.502 1500.4 L156.17 1482.07 M149.033 1474.93 L152.045 1471.92 L155.859 1475.74 L152.847 1478.75 L149.033 1474.93 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M163.323 1474.92 L166.514 1471.73 L187.629 1481.38 L177.972 1460.27 L181.164 1457.08 L192.622 1482.28 L188.53 1486.37 L163.323 1474.92 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M185.321 1452.92 L188.333 1449.91 L206.665 1468.24 L203.654 1471.25 L185.321 1452.92 M178.185 1445.78 L181.197 1442.77 L185.01 1446.58 L181.999 1449.6 L178.185 1445.78 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M212.083 1444.39 Q208.433 1448.04 207.86 1450.28 Q207.287 1452.53 209.301 1454.54 Q210.905 1456.14 212.902 1456.04 Q214.899 1455.91 216.715 1454.1 Q219.22 1451.59 218.958 1448.32 Q218.696 1445.01 215.75 1442.07 L215.079 1441.4 L212.083 1444.39 M216.846 1437.14 L227.306 1447.6 L224.294 1450.61 L221.511 1447.83 Q222.15 1450.53 221.413 1452.87 Q220.66 1455.19 218.434 1457.42 Q215.619 1460.23 212.378 1460.33 Q209.137 1460.4 206.485 1457.75 Q203.392 1454.65 203.883 1451.02 Q204.39 1447.37 208.499 1443.26 L212.722 1439.04 L212.427 1438.74 Q210.348 1436.66 207.844 1436.91 Q205.34 1437.12 202.868 1439.59 Q201.297 1441.17 200.184 1443.03 Q199.071 1444.9 198.449 1447.03 L195.666 1444.24 Q196.681 1441.95 197.925 1440.09 Q199.152 1438.2 200.626 1436.73 Q204.603 1432.75 208.63 1432.85 Q212.656 1432.95 216.846 1437.14 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M208.04 1415.93 L211.052 1412.91 L236.521 1438.38 L233.509 1441.4 L208.04 1415.93 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M224.621 1399.41 Q226.193 1405.37 228.812 1410.12 Q231.43 1414.86 235.211 1418.64 Q238.992 1422.42 243.772 1425.08 Q248.551 1427.7 254.477 1429.27 L251.858 1431.89 Q245.556 1430.49 240.613 1428.01 Q235.686 1425.5 232.02 1421.84 Q228.37 1418.19 225.882 1413.27 Q223.394 1408.36 222.002 1402.03 L224.621 1399.41 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M229.99 1394.04 L232.609 1391.42 Q238.927 1392.83 243.837 1395.32 Q248.764 1397.79 252.414 1401.44 Q256.081 1405.11 258.569 1410.05 Q261.073 1414.98 262.464 1421.28 L259.845 1423.9 Q258.274 1417.97 255.639 1413.21 Q253.004 1408.41 249.223 1404.63 Q245.442 1400.85 240.678 1398.25 Q235.932 1395.63 229.99 1394.04 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1187.32 1611.28 L1182.58 1599.09 L1172.81 1616.92 L1165.9 1616.92 L1179.71 1591.71 L1173.92 1576.72 Q1172.36 1572.71 1167.46 1572.71 L1165.9 1572.71 L1165.9 1567.68 L1168.13 1567.74 Q1176.34 1567.96 1178.41 1573.28 L1183.12 1585.47 L1192.89 1567.64 L1199.8 1567.64 L1185.98 1592.85 L1191.78 1607.84 Q1193.34 1611.85 1198.24 1611.85 L1199.8 1611.85 L1199.8 1616.88 L1197.57 1616.82 Q1189.36 1616.6 1187.32 1611.28 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1229.3 1573.72 L1270.11 1573.72 L1270.11 1579.07 L1229.3 1579.07 L1229.3 1573.72 M1229.3 1586.71 L1270.11 1586.71 L1270.11 1592.12 L1229.3 1592.12 L1229.3 1586.71 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1304.77 1555.8 L1330.01 1555.8 L1330.01 1561.22 L1310.66 1561.22 L1310.66 1572.86 Q1312.06 1572.39 1313.46 1572.16 Q1314.86 1571.91 1316.26 1571.91 Q1324.22 1571.91 1328.86 1576.27 Q1333.51 1580.63 1333.51 1588.08 Q1333.51 1595.75 1328.74 1600.01 Q1323.96 1604.25 1315.27 1604.25 Q1312.28 1604.25 1309.16 1603.74 Q1306.07 1603.23 1302.76 1602.21 L1302.76 1595.75 Q1305.63 1597.31 1308.68 1598.07 Q1311.74 1598.84 1315.14 1598.84 Q1320.65 1598.84 1323.87 1595.94 Q1327.08 1593.04 1327.08 1588.08 Q1327.08 1583.11 1323.87 1580.22 Q1320.65 1577.32 1315.14 1577.32 Q1312.57 1577.32 1309.99 1577.89 Q1307.44 1578.47 1304.77 1579.68 L1304.77 1555.8 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1359.93 1560.04 Q1354.96 1560.04 1352.45 1564.94 Q1349.97 1569.81 1349.97 1579.61 Q1349.97 1589.38 1352.45 1594.29 Q1354.96 1599.16 1359.93 1599.16 Q1364.92 1599.16 1367.41 1594.29 Q1369.92 1589.38 1369.92 1579.61 Q1369.92 1569.81 1367.41 1564.94 Q1364.92 1560.04 1359.93 1560.04 M1359.93 1554.95 Q1367.92 1554.95 1372.12 1561.28 Q1376.35 1567.58 1376.35 1579.61 Q1376.35 1591.61 1372.12 1597.95 Q1367.92 1604.25 1359.93 1604.25 Q1351.94 1604.25 1347.71 1597.95 Q1343.5 1591.61 1343.5 1579.61 Q1343.5 1567.58 1347.71 1561.28 Q1351.94 1554.95 1359.93 1554.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><polyline clip-path="url(#clip360)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="189.496,1352.62 189.496,123.472 "/>
<polyline clip-path="url(#clip360)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="189.496,1330.99 208.394,1330.99 "/>
<polyline clip-path="url(#clip360)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="189.496,1040.81 208.394,1040.81 "/>
<polyline clip-path="url(#clip360)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="189.496,750.632 208.394,750.632 "/>
<polyline clip-path="url(#clip360)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="189.496,460.451 208.394,460.451 "/>
<polyline clip-path="url(#clip360)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="189.496,170.271 208.394,170.271 "/>
<path clip-path="url(#clip360)" d="M51.6634 1350.79 L59.3023 1350.79 L59.3023 1324.42 L50.9921 1326.09 L50.9921 1321.83 L59.256 1320.16 L63.9319 1320.16 L63.9319 1350.79 L71.5707 1350.79 L71.5707 1354.72 L51.6634 1354.72 L51.6634 1350.79 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M91.0151 1323.24 Q87.404 1323.24 85.5753 1326.8 Q83.7697 1330.35 83.7697 1337.48 Q83.7697 1344.58 85.5753 1348.15 Q87.404 1351.69 91.0151 1351.69 Q94.6493 1351.69 96.4548 1348.15 Q98.2835 1344.58 98.2835 1337.48 Q98.2835 1330.35 96.4548 1326.8 Q94.6493 1323.24 91.0151 1323.24 M91.0151 1319.54 Q96.8252 1319.54 99.8808 1324.14 Q102.959 1328.73 102.959 1337.48 Q102.959 1346.2 99.8808 1350.81 Q96.8252 1355.39 91.0151 1355.39 Q85.2049 1355.39 82.1262 1350.81 Q79.0707 1346.2 79.0707 1337.48 Q79.0707 1328.73 82.1262 1324.14 Q85.2049 1319.54 91.0151 1319.54 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M102.959 1313.64 L127.071 1313.64 L127.071 1316.83 L102.959 1316.83 L102.959 1313.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M145.71 1302.54 L136.118 1317.53 L145.71 1317.53 L145.71 1302.54 M144.713 1299.23 L149.49 1299.23 L149.49 1317.53 L153.496 1317.53 L153.496 1320.69 L149.49 1320.69 L149.49 1327.31 L145.71 1327.31 L145.71 1320.69 L133.033 1320.69 L133.033 1317.02 L144.713 1299.23 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M52.585 1060.61 L60.2238 1060.61 L60.2238 1034.24 L51.9137 1035.91 L51.9137 1031.65 L60.1776 1029.98 L64.8535 1029.98 L64.8535 1060.61 L72.4923 1060.61 L72.4923 1064.54 L52.585 1064.54 L52.585 1060.61 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M91.9366 1033.06 Q88.3255 1033.06 86.4969 1036.62 Q84.6913 1040.17 84.6913 1047.29 Q84.6913 1054.4 86.4969 1057.97 Q88.3255 1061.51 91.9366 1061.51 Q95.5709 1061.51 97.3764 1057.97 Q99.2051 1054.4 99.2051 1047.29 Q99.2051 1040.17 97.3764 1036.62 Q95.5709 1033.06 91.9366 1033.06 M91.9366 1029.36 Q97.7468 1029.36 100.802 1033.96 Q103.881 1038.55 103.881 1047.29 Q103.881 1056.02 100.802 1060.63 Q97.7468 1065.21 91.9366 1065.21 Q86.1265 1065.21 83.0478 1060.63 Q79.9923 1056.02 79.9923 1047.29 Q79.9923 1038.55 83.0478 1033.96 Q86.1265 1029.36 91.9366 1029.36 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M103.881 1023.46 L127.993 1023.46 L127.993 1026.65 L103.881 1026.65 L103.881 1023.46 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M147.703 1021.99 Q150.43 1022.57 151.954 1024.42 Q153.496 1026.26 153.496 1028.97 Q153.496 1033.12 150.637 1035.4 Q147.778 1037.68 142.512 1037.68 Q140.744 1037.68 138.863 1037.32 Q137.002 1036.98 135.008 1036.28 L135.008 1032.62 Q136.588 1033.54 138.469 1034.01 Q140.349 1034.48 142.399 1034.48 Q145.973 1034.48 147.835 1033.07 Q149.716 1031.66 149.716 1028.97 Q149.716 1026.48 147.966 1025.09 Q146.236 1023.68 143.133 1023.68 L139.86 1023.68 L139.86 1020.56 L143.283 1020.56 Q146.086 1020.56 147.571 1019.45 Q149.057 1018.32 149.057 1016.22 Q149.057 1014.05 147.515 1012.91 Q145.992 1011.74 143.133 1011.74 Q141.572 1011.74 139.785 1012.08 Q137.998 1012.42 135.854 1013.13 L135.854 1009.75 Q138.017 1009.14 139.898 1008.84 Q141.797 1008.54 143.471 1008.54 Q147.797 1008.54 150.317 1010.52 Q152.838 1012.47 152.838 1015.82 Q152.838 1018.15 151.502 1019.77 Q150.167 1021.37 147.703 1021.99 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M53.3561 770.424 L60.995 770.424 L60.995 744.059 L52.6848 745.725 L52.6848 741.466 L60.9487 739.8 L65.6246 739.8 L65.6246 770.424 L73.2634 770.424 L73.2634 774.36 L53.3561 774.36 L53.3561 770.424 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M92.7078 742.878 Q89.0967 742.878 87.268 746.443 Q85.4624 749.985 85.4624 757.114 Q85.4624 764.221 87.268 767.786 Q89.0967 771.327 92.7078 771.327 Q96.342 771.327 98.1475 767.786 Q99.9762 764.221 99.9762 757.114 Q99.9762 749.985 98.1475 746.443 Q96.342 742.878 92.7078 742.878 M92.7078 739.175 Q98.5179 739.175 101.573 743.781 Q104.652 748.364 104.652 757.114 Q104.652 765.841 101.573 770.448 Q98.5179 775.031 92.7078 775.031 Q86.8976 775.031 83.8189 770.448 Q80.7634 765.841 80.7634 757.114 Q80.7634 748.364 83.8189 743.781 Q86.8976 739.175 92.7078 739.175 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M104.652 733.276 L128.764 733.276 L128.764 736.473 L104.652 736.473 L104.652 733.276 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M140.236 743.752 L153.496 743.752 L153.496 746.949 L135.666 746.949 L135.666 743.752 Q137.829 741.514 141.553 737.752 Q145.296 733.972 146.255 732.881 Q148.079 730.831 148.794 729.42 Q149.527 727.991 149.527 726.618 Q149.527 724.38 147.948 722.969 Q146.387 721.559 143.866 721.559 Q142.08 721.559 140.086 722.179 Q138.111 722.8 135.854 724.06 L135.854 720.223 Q138.149 719.302 140.142 718.832 Q142.136 718.361 143.791 718.361 Q148.155 718.361 150.75 720.543 Q153.345 722.725 153.345 726.373 Q153.345 728.104 152.687 729.665 Q152.048 731.207 150.336 733.314 Q149.866 733.859 147.346 736.473 Q144.826 739.069 140.236 743.752 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M53.0552 480.244 L60.694 480.244 L60.694 453.878 L52.3839 455.545 L52.3839 451.286 L60.6477 449.619 L65.3236 449.619 L65.3236 480.244 L72.9625 480.244 L72.9625 484.179 L53.0552 484.179 L53.0552 480.244 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M92.4068 452.698 Q88.7957 452.698 86.967 456.262 Q85.1615 459.804 85.1615 466.934 Q85.1615 474.04 86.967 477.605 Q88.7957 481.147 92.4068 481.147 Q96.0411 481.147 97.8466 477.605 Q99.6753 474.04 99.6753 466.934 Q99.6753 459.804 97.8466 456.262 Q96.0411 452.698 92.4068 452.698 M92.4068 448.994 Q98.217 448.994 101.273 453.6 Q104.351 458.184 104.351 466.934 Q104.351 475.66 101.273 480.267 Q98.217 484.85 92.4068 484.85 Q86.5967 484.85 83.518 480.267 Q80.4625 475.66 80.4625 466.934 Q80.4625 458.184 83.518 453.6 Q86.5967 448.994 92.4068 448.994 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M104.351 443.095 L128.463 443.095 L128.463 446.293 L104.351 446.293 L104.351 443.095 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M137.321 453.571 L143.528 453.571 L143.528 432.149 L136.776 433.503 L136.776 430.043 L143.49 428.689 L147.289 428.689 L147.289 453.571 L153.496 453.571 L153.496 456.769 L137.321 456.769 L137.321 453.571 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M82.7903 190.063 L90.4291 190.063 L90.4291 163.697 L82.119 165.364 L82.119 161.105 L90.3828 159.438 L95.0587 159.438 L95.0587 190.063 L102.698 190.063 L102.698 193.998 L82.7903 193.998 L82.7903 190.063 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M122.142 162.517 Q118.531 162.517 116.702 166.082 Q114.897 169.623 114.897 176.753 Q114.897 183.859 116.702 187.424 Q118.531 190.966 122.142 190.966 Q125.776 190.966 127.582 187.424 Q129.41 183.859 129.41 176.753 Q129.41 169.623 127.582 166.082 Q125.776 162.517 122.142 162.517 M122.142 158.813 Q127.952 158.813 131.008 163.42 Q134.086 168.003 134.086 176.753 Q134.086 185.48 131.008 190.086 Q127.952 194.67 122.142 194.67 Q116.332 194.67 113.253 190.086 Q110.198 185.48 110.198 176.753 Q110.198 168.003 113.253 163.42 Q116.332 158.813 122.142 158.813 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M143.791 141.009 Q140.857 141.009 139.371 143.906 Q137.904 146.783 137.904 152.576 Q137.904 158.35 139.371 161.246 Q140.857 164.124 143.791 164.124 Q146.744 164.124 148.211 161.246 Q149.697 158.35 149.697 152.576 Q149.697 146.783 148.211 143.906 Q146.744 141.009 143.791 141.009 M143.791 138 Q148.512 138 150.994 141.743 Q153.496 145.467 153.496 152.576 Q153.496 159.667 150.994 163.409 Q148.512 167.133 143.791 167.133 Q139.07 167.133 136.569 163.409 Q134.086 159.667 134.086 152.576 Q134.086 145.467 136.569 141.743 Q139.07 138 143.791 138 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M772.196 12.096 L810.437 12.096 L810.437 18.9825 L780.379 18.9825 L780.379 36.8875 L809.181 36.8875 L809.181 43.7741 L780.379 43.7741 L780.379 65.6895 L811.166 65.6895 L811.166 72.576 L772.196 72.576 L772.196 12.096 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M862.005 45.1919 L862.005 72.576 L854.551 72.576 L854.551 45.4349 Q854.551 38.994 852.04 35.7938 Q849.528 32.5936 844.505 32.5936 Q838.469 32.5936 834.985 36.4419 Q831.502 40.2903 831.502 46.9338 L831.502 72.576 L824.007 72.576 L824.007 27.2059 L831.502 27.2059 L831.502 34.2544 Q834.175 30.163 837.78 28.1376 Q841.426 26.1121 846.166 26.1121 Q853.984 26.1121 857.994 30.9732 Q862.005 35.7938 862.005 45.1919 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M884.244 14.324 L884.244 27.2059 L899.597 27.2059 L899.597 32.9987 L884.244 32.9987 L884.244 57.6282 Q884.244 63.1779 885.743 64.7578 Q887.282 66.3376 891.941 66.3376 L899.597 66.3376 L899.597 72.576 L891.941 72.576 Q883.313 72.576 880.031 69.3758 Q876.75 66.1351 876.75 57.6282 L876.75 32.9987 L871.281 32.9987 L871.281 27.2059 L876.75 27.2059 L876.75 14.324 L884.244 14.324 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M930.02 49.7694 Q920.986 49.7694 917.502 51.8354 Q914.018 53.9013 914.018 58.8839 Q914.018 62.8538 916.611 65.2034 Q919.244 67.5124 923.741 67.5124 Q929.939 67.5124 933.665 63.1374 Q937.433 58.7219 937.433 51.4303 L937.433 49.7694 L930.02 49.7694 M944.886 46.6907 L944.886 72.576 L937.433 72.576 L937.433 65.6895 Q934.881 69.8214 931.073 71.8063 Q927.265 73.7508 921.756 73.7508 Q914.788 73.7508 910.656 69.8619 Q906.565 65.9325 906.565 59.3701 Q906.565 51.7138 911.669 47.825 Q916.814 43.9361 926.981 43.9361 L937.433 43.9361 L937.433 43.2069 Q937.433 38.0623 934.03 35.2672 Q930.668 32.4315 924.551 32.4315 Q920.662 32.4315 916.976 33.3632 Q913.289 34.295 909.887 36.1584 L909.887 29.2718 Q913.978 27.692 917.826 26.9223 Q921.675 26.1121 925.32 26.1121 Q935.164 26.1121 940.025 31.2163 Q944.886 36.3204 944.886 46.6907 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M997.953 45.1919 L997.953 72.576 L990.5 72.576 L990.5 45.4349 Q990.5 38.994 987.988 35.7938 Q985.476 32.5936 980.453 32.5936 Q974.417 32.5936 970.934 36.4419 Q967.45 40.2903 967.45 46.9338 L967.45 72.576 L959.956 72.576 L959.956 27.2059 L967.45 27.2059 L967.45 34.2544 Q970.123 30.163 973.729 28.1376 Q977.375 26.1121 982.114 26.1121 Q989.932 26.1121 993.943 30.9732 Q997.953 35.7938 997.953 45.1919 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1042.68 49.3643 Q1042.68 41.2625 1039.31 36.8065 Q1035.99 32.3505 1029.96 32.3505 Q1023.96 32.3505 1020.6 36.8065 Q1017.28 41.2625 1017.28 49.3643 Q1017.28 57.4256 1020.6 61.8816 Q1023.96 66.3376 1029.96 66.3376 Q1035.99 66.3376 1039.31 61.8816 Q1042.68 57.4256 1042.68 49.3643 M1050.13 66.9452 Q1050.13 78.5308 1044.98 84.1616 Q1039.84 89.8329 1029.23 89.8329 Q1025.3 89.8329 1021.81 89.2252 Q1018.33 88.6581 1015.05 87.4428 L1015.05 80.1917 Q1018.33 81.9741 1021.53 82.8248 Q1024.73 83.6755 1028.05 83.6755 Q1035.38 83.6755 1039.03 79.8271 Q1042.68 76.0193 1042.68 68.282 L1042.68 64.5957 Q1040.37 68.6061 1036.76 70.5911 Q1033.16 72.576 1028.13 72.576 Q1019.79 72.576 1014.68 66.2161 Q1009.58 59.8562 1009.58 49.3643 Q1009.58 38.832 1014.68 32.472 Q1019.79 26.1121 1028.13 26.1121 Q1033.16 26.1121 1036.76 28.0971 Q1040.37 30.082 1042.68 34.0924 L1042.68 27.2059 L1050.13 27.2059 L1050.13 66.9452 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1065.48 9.54393 L1072.94 9.54393 L1072.94 72.576 L1065.48 72.576 L1065.48 9.54393 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1127.34 48.0275 L1127.34 51.6733 L1093.07 51.6733 Q1093.55 59.3701 1097.69 63.421 Q1101.86 67.4314 1109.27 67.4314 Q1113.57 67.4314 1117.58 66.3781 Q1121.63 65.3249 1125.6 63.2184 L1125.6 70.267 Q1121.59 71.9684 1117.37 72.8596 Q1113.16 73.7508 1108.83 73.7508 Q1097.97 73.7508 1091.61 67.4314 Q1085.29 61.1119 1085.29 50.3365 Q1085.29 39.1965 1091.29 32.6746 Q1097.32 26.1121 1107.53 26.1121 Q1116.69 26.1121 1121.99 32.0264 Q1127.34 37.9003 1127.34 48.0275 M1119.89 45.84 Q1119.8 39.7232 1116.44 36.0774 Q1113.12 32.4315 1107.61 32.4315 Q1101.37 32.4315 1097.61 35.9558 Q1093.88 39.4801 1093.31 45.8805 L1119.89 45.84 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1174.9 35.9153 Q1177.69 30.8922 1181.58 28.5022 Q1185.47 26.1121 1190.74 26.1121 Q1197.82 26.1121 1201.67 31.0947 Q1205.52 36.0368 1205.52 45.1919 L1205.52 72.576 L1198.03 72.576 L1198.03 45.4349 Q1198.03 38.913 1195.72 35.7533 Q1193.41 32.5936 1188.67 32.5936 Q1182.88 32.5936 1179.51 36.4419 Q1176.15 40.2903 1176.15 46.9338 L1176.15 72.576 L1168.66 72.576 L1168.66 45.4349 Q1168.66 38.8725 1166.35 35.7533 Q1164.04 32.5936 1159.22 32.5936 Q1153.51 32.5936 1150.15 36.4824 Q1146.78 40.3308 1146.78 46.9338 L1146.78 72.576 L1139.29 72.576 L1139.29 27.2059 L1146.78 27.2059 L1146.78 34.2544 Q1149.34 30.082 1152.9 28.0971 Q1156.47 26.1121 1161.37 26.1121 Q1166.31 26.1121 1169.75 28.6237 Q1173.24 31.1352 1174.9 35.9153 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1259.2 48.0275 L1259.2 51.6733 L1224.93 51.6733 Q1225.41 59.3701 1229.54 63.421 Q1233.72 67.4314 1241.13 67.4314 Q1245.42 67.4314 1249.43 66.3781 Q1253.48 65.3249 1257.45 63.2184 L1257.45 70.267 Q1253.44 71.9684 1249.23 72.8596 Q1245.02 73.7508 1240.68 73.7508 Q1229.83 73.7508 1223.47 67.4314 Q1217.15 61.1119 1217.15 50.3365 Q1217.15 39.1965 1223.14 32.6746 Q1229.18 26.1121 1239.39 26.1121 Q1248.54 26.1121 1253.85 32.0264 Q1259.2 37.9003 1259.2 48.0275 M1251.74 45.84 Q1251.66 39.7232 1248.3 36.0774 Q1244.98 32.4315 1239.47 32.4315 Q1233.23 32.4315 1229.46 35.9558 Q1225.74 39.4801 1225.17 45.8805 L1251.74 45.84 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1309.14 45.1919 L1309.14 72.576 L1301.69 72.576 L1301.69 45.4349 Q1301.69 38.994 1299.18 35.7938 Q1296.67 32.5936 1291.64 32.5936 Q1285.61 32.5936 1282.12 36.4419 Q1278.64 40.2903 1278.64 46.9338 L1278.64 72.576 L1271.15 72.576 L1271.15 27.2059 L1278.64 27.2059 L1278.64 34.2544 Q1281.31 30.163 1284.92 28.1376 Q1288.57 26.1121 1293.3 26.1121 Q1301.12 26.1121 1305.13 30.9732 Q1309.14 35.7938 1309.14 45.1919 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1331.38 14.324 L1331.38 27.2059 L1346.74 27.2059 L1346.74 32.9987 L1331.38 32.9987 L1331.38 57.6282 Q1331.38 63.1779 1332.88 64.7578 Q1334.42 66.3376 1339.08 66.3376 L1346.74 66.3376 L1346.74 72.576 L1339.08 72.576 Q1330.45 72.576 1327.17 69.3758 Q1323.89 66.1351 1323.89 57.6282 L1323.89 32.9987 L1318.42 32.9987 L1318.42 27.2059 L1323.89 27.2059 L1323.89 14.324 L1331.38 14.324 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1419.49 14.0809 L1419.49 22.0612 Q1414.83 19.8332 1410.7 18.7395 Q1406.57 17.6457 1402.72 17.6457 Q1396.04 17.6457 1392.39 20.2383 Q1388.78 22.8309 1388.78 27.611 Q1388.78 31.6214 1391.17 33.6873 Q1393.61 35.7128 1400.33 36.9686 L1405.27 37.9813 Q1414.43 39.7232 1418.76 44.1387 Q1423.14 48.5136 1423.14 55.8863 Q1423.14 64.6767 1417.22 69.2137 Q1411.35 73.7508 1399.96 73.7508 Q1395.67 73.7508 1390.81 72.7785 Q1385.99 71.8063 1380.8 69.9024 L1380.8 61.4765 Q1385.79 64.2716 1390.57 65.6895 Q1395.35 67.1073 1399.96 67.1073 Q1406.97 67.1073 1410.78 64.3527 Q1414.59 61.598 1414.59 56.4939 Q1414.59 52.0379 1411.83 49.5264 Q1409.12 47.0148 1402.88 45.759 L1397.9 44.7868 Q1388.74 42.9639 1384.65 39.075 Q1380.56 35.1862 1380.56 28.2591 Q1380.56 20.2383 1386.19 15.6203 Q1391.86 11.0023 1401.79 11.0023 Q1406.04 11.0023 1410.46 11.7719 Q1414.87 12.5416 1419.49 14.0809 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1442.78 65.7705 L1442.78 89.8329 L1435.29 89.8329 L1435.29 27.2059 L1442.78 27.2059 L1442.78 34.0924 Q1445.13 30.0415 1448.7 28.0971 Q1452.3 26.1121 1457.29 26.1121 Q1465.55 26.1121 1470.69 32.6746 Q1475.88 39.2371 1475.88 49.9314 Q1475.88 60.6258 1470.69 67.1883 Q1465.55 73.7508 1457.29 73.7508 Q1452.3 73.7508 1448.7 71.8063 Q1445.13 69.8214 1442.78 65.7705 M1468.14 49.9314 Q1468.14 41.7081 1464.74 37.0496 Q1461.38 32.3505 1455.46 32.3505 Q1449.55 32.3505 1446.15 37.0496 Q1442.78 41.7081 1442.78 49.9314 Q1442.78 58.1548 1446.15 62.8538 Q1449.55 67.5124 1455.46 67.5124 Q1461.38 67.5124 1464.74 62.8538 Q1468.14 58.1548 1468.14 49.9314 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1527.04 48.0275 L1527.04 51.6733 L1492.77 51.6733 Q1493.26 59.3701 1497.39 63.421 Q1501.56 67.4314 1508.97 67.4314 Q1513.27 67.4314 1517.28 66.3781 Q1521.33 65.3249 1525.3 63.2184 L1525.3 70.267 Q1521.29 71.9684 1517.08 72.8596 Q1512.86 73.7508 1508.53 73.7508 Q1497.67 73.7508 1491.31 67.4314 Q1484.99 61.1119 1484.99 50.3365 Q1484.99 39.1965 1490.99 32.6746 Q1497.02 26.1121 1507.23 26.1121 Q1516.39 26.1121 1521.69 32.0264 Q1527.04 37.9003 1527.04 48.0275 M1519.59 45.84 Q1519.51 39.7232 1516.14 36.0774 Q1512.82 32.4315 1507.31 32.4315 Q1501.08 32.4315 1497.31 35.9558 Q1493.58 39.4801 1493.01 45.8805 L1519.59 45.84 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1571.93 28.9478 L1571.93 35.9153 Q1568.77 34.1734 1565.57 33.3227 Q1562.41 32.4315 1559.17 32.4315 Q1551.91 32.4315 1547.9 37.0496 Q1543.89 41.6271 1543.89 49.9314 Q1543.89 58.2358 1547.9 62.8538 Q1551.91 67.4314 1559.17 67.4314 Q1562.41 67.4314 1565.57 66.5807 Q1568.77 65.6895 1571.93 63.9476 L1571.93 70.8341 Q1568.81 72.2924 1565.44 73.0216 Q1562.12 73.7508 1558.36 73.7508 Q1548.11 73.7508 1542.07 67.3098 Q1536.03 60.8689 1536.03 49.9314 Q1536.03 38.832 1542.11 32.472 Q1548.23 26.1121 1558.84 26.1121 Q1562.28 26.1121 1565.57 26.8413 Q1568.85 27.5299 1571.93 28.9478 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1592.26 14.324 L1592.26 27.2059 L1607.61 27.2059 L1607.61 32.9987 L1592.26 32.9987 L1592.26 57.6282 Q1592.26 63.1779 1593.76 64.7578 Q1595.3 66.3376 1599.96 66.3376 L1607.61 66.3376 L1607.61 72.576 L1599.96 72.576 Q1591.33 72.576 1588.05 69.3758 Q1584.77 66.1351 1584.77 57.6282 L1584.77 32.9987 L1579.3 32.9987 L1579.3 27.2059 L1584.77 27.2059 L1584.77 14.324 L1592.26 14.324 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1643.71 34.1734 Q1642.45 33.4443 1640.95 33.1202 Q1639.49 32.7556 1637.71 32.7556 Q1631.39 32.7556 1627.99 36.8875 Q1624.63 40.9789 1624.63 48.6757 L1624.63 72.576 L1617.13 72.576 L1617.13 27.2059 L1624.63 27.2059 L1624.63 34.2544 Q1626.98 30.1225 1630.74 28.1376 Q1634.51 26.1121 1639.9 26.1121 Q1640.67 26.1121 1641.6 26.2337 Q1642.53 26.3147 1643.67 26.5172 L1643.71 34.1734 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1650.76 54.671 L1650.76 27.2059 L1658.21 27.2059 L1658.21 54.3874 Q1658.21 60.8284 1660.72 64.0691 Q1663.23 67.2693 1668.26 67.2693 Q1674.29 67.2693 1677.78 63.421 Q1681.3 59.5726 1681.3 52.9291 L1681.3 27.2059 L1688.75 27.2059 L1688.75 72.576 L1681.3 72.576 L1681.3 65.6084 Q1678.59 69.7404 1674.98 71.7658 Q1671.42 73.7508 1666.68 73.7508 Q1658.86 73.7508 1654.81 68.8897 Q1650.76 64.0286 1650.76 54.671 M1669.51 26.1121 L1669.51 26.1121 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1739.43 35.9153 Q1742.23 30.8922 1746.11 28.5022 Q1750 26.1121 1755.27 26.1121 Q1762.36 26.1121 1766.21 31.0947 Q1770.06 36.0368 1770.06 45.1919 L1770.06 72.576 L1762.56 72.576 L1762.56 45.4349 Q1762.56 38.913 1760.25 35.7533 Q1757.94 32.5936 1753.2 32.5936 Q1747.41 32.5936 1744.05 36.4419 Q1740.69 40.2903 1740.69 46.9338 L1740.69 72.576 L1733.19 72.576 L1733.19 45.4349 Q1733.19 38.8725 1730.88 35.7533 Q1728.57 32.5936 1723.75 32.5936 Q1718.04 32.5936 1714.68 36.4824 Q1711.32 40.3308 1711.32 46.9338 L1711.32 72.576 L1703.82 72.576 L1703.82 27.2059 L1711.32 27.2059 L1711.32 34.2544 Q1713.87 30.082 1717.43 28.0971 Q1721 26.1121 1725.9 26.1121 Q1730.84 26.1121 1734.29 28.6237 Q1737.77 31.1352 1739.43 35.9153 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><circle clip-path="url(#clip362)" cx="454.801" cy="194.334" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="488.121" cy="283.803" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="521.44" cy="327.858" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="554.759" cy="360.379" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="588.079" cy="466.576" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="621.398" cy="515.192" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="654.718" cy="531.492" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="688.037" cy="546.966" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="721.356" cy="619.244" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="754.676" cy="673.138" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="787.995" cy="680.237" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="821.314" cy="691.116" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="854.634" cy="709.417" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="887.953" cy="747.039" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="921.273" cy="748.962" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="954.592" cy="762.752" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="987.911" cy="783.417" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1021.23" cy="787.385" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1054.55" cy="816.784" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1087.87" cy="840.388" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1121.19" cy="848.1" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1154.51" cy="911.066" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1187.83" cy="920.178" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1221.15" cy="942.766" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1254.47" cy="957.752" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1287.79" cy="985.563" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1321.1" cy="997.531" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1354.42" cy="1031.1" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1387.74" cy="1042.05" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1421.06" cy="1053.76" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1454.38" cy="1054.74" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1487.7" cy="1073.77" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1521.02" cy="1075.87" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1554.34" cy="1079.81" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1587.66" cy="1097.2" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1620.98" cy="1101.62" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1654.3" cy="1113.52" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1687.62" cy="1115.71" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1720.94" cy="1123.57" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1754.26" cy="1156.84" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1787.58" cy="1169.46" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1820.9" cy="1200.49" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1854.21" cy="1212.31" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1887.53" cy="1227.57" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1920.85" cy="1241.81" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1954.17" cy="1258.11" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1987.49" cy="1262.32" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="2020.81" cy="1272.11" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="2054.13" cy="1298.17" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="2087.45" cy="1317.83" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
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
[ Info: VUMPS init:	obj = +8.571475316991e-02	err = 4.0869e-01
[ Info: VUMPS   1:	obj = -8.841768943048e-01	err = 3.0962778235e-02	time = 8.39 sec
[ Info: VUMPS   2:	obj = -8.858941441948e-01	err = 6.8861549505e-03	time = 0.08 sec
[ Info: VUMPS   3:	obj = -8.861591970700e-01	err = 3.1438981011e-03	time = 0.19 sec
[ Info: VUMPS   4:	obj = -8.862373586553e-01	err = 1.5694150974e-03	time = 0.08 sec
[ Info: VUMPS   5:	obj = -8.862665008570e-01	err = 9.6720118128e-04	time = 0.10 sec
[ Info: VUMPS   6:	obj = -8.862780790113e-01	err = 7.6876116694e-04	time = 0.11 sec
[ Info: VUMPS   7:	obj = -8.862831659742e-01	err = 6.6877783316e-04	time = 0.19 sec
[ Info: VUMPS   8:	obj = -8.862855860323e-01	err = 5.3599376124e-04	time = 0.11 sec
[ Info: VUMPS   9:	obj = -8.862867663190e-01	err = 4.0660286466e-04	time = 0.16 sec
[ Info: VUMPS  10:	obj = -8.862873424424e-01	err = 3.0566269459e-04	time = 0.32 sec
[ Info: VUMPS  11:	obj = -8.862876251526e-01	err = 2.2687597738e-04	time = 0.14 sec
[ Info: VUMPS  12:	obj = -8.862877641798e-01	err = 1.6630235595e-04	time = 0.28 sec
[ Info: VUMPS  13:	obj = -8.862878324776e-01	err = 1.2065962982e-04	time = 0.17 sec
[ Info: VUMPS  14:	obj = -8.862878659705e-01	err = 8.6847532670e-05	time = 0.13 sec
[ Info: VUMPS  15:	obj = -8.862878823733e-01	err = 6.2151682727e-05	time = 0.16 sec
[ Info: VUMPS  16:	obj = -8.862878903974e-01	err = 4.4290628543e-05	time = 0.18 sec
[ Info: VUMPS  17:	obj = -8.862878943223e-01	err = 3.1467696209e-05	time = 0.14 sec
[ Info: VUMPS  18:	obj = -8.862878962425e-01	err = 2.2309311401e-05	time = 0.17 sec
[ Info: VUMPS  19:	obj = -8.862878971826e-01	err = 1.5794437549e-05	time = 0.17 sec
[ Info: VUMPS  20:	obj = -8.862878976433e-01	err = 1.1169983677e-05	time = 0.15 sec
[ Info: VUMPS  21:	obj = -8.862878978692e-01	err = 7.8938125044e-06	time = 0.17 sec
[ Info: VUMPS  22:	obj = -8.862878979802e-01	err = 5.5754187069e-06	time = 0.14 sec
[ Info: VUMPS  23:	obj = -8.862878980347e-01	err = 3.9363165672e-06	time = 0.17 sec
[ Info: VUMPS  24:	obj = -8.862878980615e-01	err = 2.7782600761e-06	time = 0.18 sec
[ Info: VUMPS  25:	obj = -8.862878980747e-01	err = 1.9605416898e-06	time = 0.14 sec
[ Info: VUMPS  26:	obj = -8.862878980813e-01	err = 1.3831566356e-06	time = 0.17 sec
[ Info: VUMPS  27:	obj = -8.862878980845e-01	err = 9.7563836730e-07	time = 0.17 sec
[ Info: VUMPS  28:	obj = -8.862878980861e-01	err = 6.8808237712e-07	time = 0.14 sec
[ Info: VUMPS  29:	obj = -8.862878980868e-01	err = 4.8521326515e-07	time = 0.17 sec
[ Info: VUMPS  30:	obj = -8.862878980872e-01	err = 3.4211403050e-07	time = 0.18 sec
[ Info: VUMPS  31:	obj = -8.862878980874e-01	err = 2.4118986474e-07	time = 0.14 sec
[ Info: VUMPS  32:	obj = -8.862878980875e-01	err = 1.7001487151e-07	time = 0.17 sec
[ Info: VUMPS  33:	obj = -8.862878980876e-01	err = 1.1983516999e-07	time = 0.16 sec
[ Info: VUMPS  34:	obj = -8.862878980876e-01	err = 8.4458211885e-08	time = 0.14 sec
[ Info: VUMPS  35:	obj = -8.862878980876e-01	err = 5.9519779420e-08	time = 0.18 sec
[ Info: VUMPS  36:	obj = -8.862878980876e-01	err = 4.1941570268e-08	time = 0.14 sec
[ Info: VUMPS  37:	obj = -8.862878980877e-01	err = 2.9552481662e-08	time = 0.17 sec
[ Info: VUMPS  38:	obj = -8.862878980877e-01	err = 2.0821461973e-08	time = 0.17 sec
[ Info: VUMPS  39:	obj = -8.862878980877e-01	err = 1.4668928777e-08	time = 0.13 sec
[ Info: VUMPS  40:	obj = -8.862878980877e-01	err = 1.0333740629e-08	time = 0.16 sec
[ Info: VUMPS  41:	obj = -8.862878980877e-01	err = 7.2793133017e-09	time = 0.19 sec
[ Info: VUMPS  42:	obj = -8.862878980877e-01	err = 5.1274193990e-09	time = 0.14 sec
[ Info: VUMPS  43:	obj = -8.862878980877e-01	err = 3.6115696251e-09	time = 0.16 sec
[ Info: VUMPS  44:	obj = -8.862878980877e-01	err = 2.5436748324e-09	time = 0.18 sec
[ Info: VUMPS  45:	obj = -8.862878980877e-01	err = 1.7914400554e-09	time = 0.13 sec
[ Info: VUMPS  46:	obj = -8.862878980877e-01	err = 1.2616075130e-09	time = 0.17 sec
[ Info: VUMPS  47:	obj = -8.862878980877e-01	err = 8.8844294429e-10	time = 0.18 sec
[ Info: VUMPS  48:	obj = -8.862878980877e-01	err = 6.2563200343e-10	time = 0.14 sec
[ Info: VUMPS  49:	obj = -8.862878980877e-01	err = 4.4055009956e-10	time = 0.17 sec
[ Info: VUMPS  50:	obj = -8.862878980877e-01	err = 3.1021022428e-10	time = 0.17 sec
[ Info: VUMPS  51:	obj = -8.862878980877e-01	err = 2.1842377352e-10	time = 0.13 sec
[ Info: VUMPS  52:	obj = -8.862878980877e-01	err = 1.5378883830e-10	time = 0.17 sec
[ Info: VUMPS  53:	obj = -8.862878980877e-01	err = 1.0827914915e-10	time = 0.13 sec
[ Info: VUMPS  54:	obj = -8.862878980877e-01	err = 7.6233561627e-11	time = 0.17 sec
[ Info: VUMPS  55:	obj = -8.862878980878e-01	err = 5.3670361265e-11	time = 0.17 sec
[ Info: VUMPS  56:	obj = -8.862878980878e-01	err = 3.7785296901e-11	time = 0.12 sec
[ Info: VUMPS  57:	obj = -8.862878980878e-01	err = 2.6597183661e-11	time = 0.15 sec
[ Info: VUMPS  58:	obj = -8.862878980878e-01	err = 1.8724144528e-11	time = 0.15 sec
[ Info: VUMPS  59:	obj = -8.862878980878e-01	err = 1.3177752530e-11	time = 0.12 sec
[ Info: VUMPS  60:	obj = -8.862878980878e-01	err = 9.2726977215e-12	time = 0.15 sec
[ Info: VUMPS  61:	obj = -8.862878980878e-01	err = 6.5267922494e-12	time = 0.10 sec
[ Info: VUMPS  62:	obj = -8.862878980878e-01	err = 4.5890864081e-12	time = 0.14 sec
[ Info: VUMPS  63:	obj = -8.862878980878e-01	err = 3.2289295438e-12	time = 0.10 sec
[ Info: VUMPS  64:	obj = -8.862878980878e-01	err = 2.2706304983e-12	time = 0.13 sec
[ Info: VUMPS  65:	obj = -8.862878980878e-01	err = 1.5950561789e-12	time = 0.09 sec
[ Info: VUMPS  66:	obj = -8.862878980878e-01	err = 1.1189780150e-12	time = 0.09 sec
[ Info: VUMPS conv 67:	obj = -8.862878980878e-01	err = 7.7998640130e-13	time = 18.42 sec

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

