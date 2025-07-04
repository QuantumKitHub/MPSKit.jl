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
[ Info: VUMPS init:	obj = +2.499952940580e-01	err = 6.3441e-03
[ Info: VUMPS   1:	obj = -1.289444927303e-01	err = 3.6505534112e-01	time = 0.02 sec
[ Info: VUMPS   2:	obj = -3.046604131170e-01	err = 3.5438451443e-01	time = 0.03 sec
[ Info: VUMPS   3:	obj = -1.151380226837e-01	err = 3.7396849153e-01	time = 0.03 sec
[ Info: VUMPS   4:	obj = -7.859906223027e-02	err = 3.8101931668e-01	time = 0.02 sec
[ Info: VUMPS   5:	obj = -8.570311273681e-02	err = 4.5415600150e-01	time = 0.04 sec
[ Info: VUMPS   6:	obj = -1.607866955582e-01	err = 3.7328533114e-01	time = 0.02 sec
[ Info: VUMPS   7:	obj = -2.590055398932e-01	err = 3.6315039435e-01	time = 0.02 sec
[ Info: VUMPS   8:	obj = -3.985401345056e-01	err = 2.4797713082e-01	time = 0.03 sec
[ Info: VUMPS   9:	obj = -2.980269171709e-01	err = 3.5009326244e-01	time = 0.04 sec
[ Info: VUMPS  10:	obj = -7.766811941836e-02	err = 3.7146004460e-01	time = 0.04 sec
[ Info: VUMPS  11:	obj = -1.513433486022e-01	err = 3.9298119907e-01	time = 0.04 sec
[ Info: VUMPS  12:	obj = -1.636153457041e-01	err = 3.8567245083e-01	time = 0.02 sec
[ Info: VUMPS  13:	obj = -8.850847675954e-02	err = 3.8567080613e-01	time = 0.03 sec
[ Info: VUMPS  14:	obj = -1.933682984363e-01	err = 3.8578605637e-01	time = 0.08 sec
[ Info: VUMPS  15:	obj = -4.668791737320e-02	err = 4.0507831578e-01	time = 0.03 sec
[ Info: VUMPS  16:	obj = -6.685207964181e-02	err = 4.4181563160e-01	time = 0.03 sec
[ Info: VUMPS  17:	obj = -2.672203890897e-01	err = 3.3906178990e-01	time = 0.03 sec
[ Info: VUMPS  18:	obj = -3.428995365711e-01	err = 3.2686747553e-01	time = 0.03 sec
[ Info: VUMPS  19:	obj = -2.863734743969e-01	err = 3.5064087667e-01	time = 0.04 sec
[ Info: VUMPS  20:	obj = -1.067319756533e-01	err = 3.8187305789e-01	time = 0.03 sec
[ Info: VUMPS  21:	obj = -1.318548228297e-01	err = 3.6108954481e-01	time = 0.02 sec
[ Info: VUMPS  22:	obj = -4.044279153596e-02	err = 3.7520252599e-01	time = 0.02 sec
[ Info: VUMPS  23:	obj = -3.945104968675e-02	err = 3.9224571991e-01	time = 0.03 sec
[ Info: VUMPS  24:	obj = -2.671412452672e-01	err = 3.6585228539e-01	time = 0.03 sec
[ Info: VUMPS  25:	obj = -3.310347250642e-01	err = 3.1899906877e-01	time = 0.03 sec
[ Info: VUMPS  26:	obj = -2.168383952062e-01	err = 3.7166034087e-01	time = 0.03 sec
[ Info: VUMPS  27:	obj = -3.507015235665e-01	err = 3.0330946464e-01	time = 0.04 sec
[ Info: VUMPS  28:	obj = -2.810889490868e-01	err = 3.5384589143e-01	time = 0.04 sec
[ Info: VUMPS  29:	obj = -1.491679715343e-01	err = 3.7712897712e-01	time = 0.03 sec
[ Info: VUMPS  30:	obj = -2.103310942986e-01	err = 3.6804474254e-01	time = 0.02 sec
[ Info: VUMPS  31:	obj = -2.570407107876e-01	err = 3.5647467556e-01	time = 0.03 sec
[ Info: VUMPS  32:	obj = -2.622474733096e-01	err = 3.6359049960e-01	time = 0.03 sec
[ Info: VUMPS  33:	obj = -2.747408111088e-01	err = 3.5217438909e-01	time = 0.02 sec
[ Info: VUMPS  34:	obj = -2.795011381225e-01	err = 3.5577849785e-01	time = 0.03 sec
[ Info: VUMPS  35:	obj = +6.109686520960e-02	err = 3.7478423163e-01	time = 0.03 sec
[ Info: VUMPS  36:	obj = -3.235177003488e-01	err = 3.2501656960e-01	time = 0.03 sec
[ Info: VUMPS  37:	obj = -3.607718277418e-01	err = 3.0399210370e-01	time = 0.03 sec
[ Info: VUMPS  38:	obj = -1.522405023809e-01	err = 3.8460611720e-01	time = 0.03 sec
[ Info: VUMPS  39:	obj = -1.401486230679e-01	err = 4.2359144803e-01	time = 0.04 sec
[ Info: VUMPS  40:	obj = -3.016337945443e-01	err = 3.3760785136e-01	time = 0.02 sec
[ Info: VUMPS  41:	obj = -3.592630330471e-01	err = 2.8277667197e-01	time = 0.04 sec
[ Info: VUMPS  42:	obj = -9.648261609327e-02	err = 3.9800363224e-01	time = 0.03 sec
[ Info: VUMPS  43:	obj = -2.068120169738e-01	err = 3.4795790120e-01	time = 0.07 sec
[ Info: VUMPS  44:	obj = -3.320461487347e-01	err = 3.3031447114e-01	time = 0.03 sec
[ Info: VUMPS  45:	obj = -2.748266113721e-01	err = 3.3719047760e-01	time = 0.03 sec
[ Info: VUMPS  46:	obj = +5.825197478980e-02	err = 3.7299269867e-01	time = 0.03 sec
[ Info: VUMPS  47:	obj = -2.209737585465e-01	err = 3.6224489048e-01	time = 0.03 sec
[ Info: VUMPS  48:	obj = -3.495730972764e-01	err = 3.1370483925e-01	time = 0.03 sec
[ Info: VUMPS  49:	obj = -1.184295942341e-02	err = 3.8132184336e-01	time = 0.03 sec
[ Info: VUMPS  50:	obj = -2.253818587474e-01	err = 3.6351801471e-01	time = 0.02 sec
[ Info: VUMPS  51:	obj = -2.611723114106e-01	err = 3.6263568517e-01	time = 0.04 sec
[ Info: VUMPS  52:	obj = -2.391735319717e-01	err = 3.4034883778e-01	time = 0.03 sec
[ Info: VUMPS  53:	obj = -2.076637304355e-01	err = 3.7785371429e-01	time = 0.03 sec
[ Info: VUMPS  54:	obj = +1.233357556406e-01	err = 3.6817687888e-01	time = 0.03 sec
[ Info: VUMPS  55:	obj = -1.064547626837e-02	err = 3.6875802199e-01	time = 0.03 sec
[ Info: VUMPS  56:	obj = -1.051495232666e-01	err = 4.0131031658e-01	time = 0.03 sec
[ Info: VUMPS  57:	obj = -1.412941986925e-01	err = 3.8676020133e-01	time = 0.02 sec
[ Info: VUMPS  58:	obj = -5.105594856383e-02	err = 4.1918593555e-01	time = 0.03 sec
[ Info: VUMPS  59:	obj = -9.429180669034e-02	err = 3.5663676393e-01	time = 0.02 sec
[ Info: VUMPS  60:	obj = -3.211645595234e-01	err = 3.3556696008e-01	time = 0.03 sec
[ Info: VUMPS  61:	obj = -4.997897659120e-02	err = 3.6911454941e-01	time = 0.03 sec
[ Info: VUMPS  62:	obj = +5.061815735283e-02	err = 3.9240870402e-01	time = 0.03 sec
[ Info: VUMPS  63:	obj = -2.858216622151e-02	err = 3.9031720030e-01	time = 0.03 sec
[ Info: VUMPS  64:	obj = -3.066897558850e-01	err = 3.4219120029e-01	time = 0.02 sec
[ Info: VUMPS  65:	obj = -1.035726948011e-01	err = 3.9992774981e-01	time = 0.03 sec
[ Info: VUMPS  66:	obj = +9.359087106400e-02	err = 3.2551512023e-01	time = 0.03 sec
[ Info: VUMPS  67:	obj = -1.506323639493e-01	err = 3.8851758838e-01	time = 0.03 sec
[ Info: VUMPS  68:	obj = -2.694024194911e-01	err = 3.4908885270e-01	time = 0.02 sec
[ Info: VUMPS  69:	obj = -1.666459363907e-01	err = 3.7191916042e-01	time = 0.03 sec
[ Info: VUMPS  70:	obj = -3.295401946651e-01	err = 3.2485997665e-01	time = 0.02 sec
[ Info: VUMPS  71:	obj = -1.716492900593e-01	err = 3.4774685002e-01	time = 0.05 sec
[ Info: VUMPS  72:	obj = -2.062243894572e-01	err = 3.4367048189e-01	time = 0.03 sec
[ Info: VUMPS  73:	obj = -2.270916009662e-01	err = 3.4883289649e-01	time = 0.04 sec
[ Info: VUMPS  74:	obj = +6.972996650156e-03	err = 3.6664495072e-01	time = 0.03 sec
[ Info: VUMPS  75:	obj = -2.140897386871e-01	err = 3.6896873908e-01	time = 0.02 sec
[ Info: VUMPS  76:	obj = -2.776215003655e-01	err = 3.5162263908e-01	time = 0.03 sec
[ Info: VUMPS  77:	obj = -2.548517060346e-01	err = 3.5985886200e-01	time = 0.04 sec
[ Info: VUMPS  78:	obj = -3.627093129660e-01	err = 2.8522543561e-01	time = 0.04 sec
[ Info: VUMPS  79:	obj = -4.246979337066e-01	err = 1.8433888114e-01	time = 0.04 sec
[ Info: VUMPS  80:	obj = -2.768088632211e-01	err = 3.6889157455e-01	time = 0.04 sec
[ Info: VUMPS  81:	obj = -3.333529535371e-01	err = 3.2461308190e-01	time = 0.04 sec
[ Info: VUMPS  82:	obj = +9.739742280028e-02	err = 3.6140350665e-01	time = 0.04 sec
[ Info: VUMPS  83:	obj = -8.001219240663e-02	err = 3.3065484133e-01	time = 0.03 sec
[ Info: VUMPS  84:	obj = -1.494867091384e-01	err = 3.6515333400e-01	time = 0.02 sec
[ Info: VUMPS  85:	obj = -3.357585607378e-01	err = 3.2062204533e-01	time = 0.03 sec
[ Info: VUMPS  86:	obj = -1.214350882454e-01	err = 3.8756516414e-01	time = 0.03 sec
[ Info: VUMPS  87:	obj = -1.720890136641e-01	err = 3.7209498473e-01	time = 0.03 sec
[ Info: VUMPS  88:	obj = -2.513176623582e-01	err = 3.6428215490e-01	time = 0.03 sec
[ Info: VUMPS  89:	obj = -1.167825455975e-01	err = 3.7816700280e-01	time = 0.03 sec
[ Info: VUMPS  90:	obj = -1.487085246354e-01	err = 3.8452112070e-01	time = 0.03 sec
[ Info: VUMPS  91:	obj = -1.492511193594e-01	err = 3.6888008833e-01	time = 0.04 sec
[ Info: VUMPS  92:	obj = -2.942886788237e-01	err = 3.3926362090e-01	time = 0.02 sec
[ Info: VUMPS  93:	obj = -2.513546381842e-01	err = 3.5669021055e-01	time = 0.04 sec
[ Info: VUMPS  94:	obj = -2.592696224289e-01	err = 3.8742412819e-01	time = 0.04 sec
[ Info: VUMPS  95:	obj = -1.670969129733e-01	err = 4.0396565424e-01	time = 0.04 sec
[ Info: VUMPS  96:	obj = -8.982884784032e-02	err = 4.0673356787e-01	time = 0.04 sec
[ Info: VUMPS  97:	obj = -1.162574886965e-01	err = 4.0014599231e-01	time = 0.03 sec
[ Info: VUMPS  98:	obj = -9.485766256473e-02	err = 3.6567908225e-01	time = 0.02 sec
[ Info: VUMPS  99:	obj = -1.626897465348e-01	err = 3.9836027057e-01	time = 0.06 sec
[ Info: VUMPS 100:	obj = -2.820414385240e-01	err = 3.3860149095e-01	time = 0.02 sec
[ Info: VUMPS 101:	obj = +3.653164331064e-03	err = 3.8214630304e-01	time = 0.04 sec
[ Info: VUMPS 102:	obj = -1.488426870678e-01	err = 3.5921878796e-01	time = 0.02 sec
[ Info: VUMPS 103:	obj = -3.107550441980e-01	err = 3.3205507908e-01	time = 0.03 sec
[ Info: VUMPS 104:	obj = -2.991475468045e-01	err = 3.4083808929e-01	time = 0.04 sec
[ Info: VUMPS 105:	obj = -1.521003309052e-01	err = 4.0772463716e-01	time = 0.05 sec
[ Info: VUMPS 106:	obj = -8.123652136498e-02	err = 3.7546849297e-01	time = 0.03 sec
[ Info: VUMPS 107:	obj = -2.596284651709e-01	err = 3.4976630049e-01	time = 0.03 sec
[ Info: VUMPS 108:	obj = -2.935877769698e-01	err = 3.3783749995e-01	time = 0.04 sec
[ Info: VUMPS 109:	obj = -2.402370295781e-01	err = 3.6810295199e-01	time = 0.04 sec
[ Info: VUMPS 110:	obj = -3.150592845722e-01	err = 3.3233905128e-01	time = 0.04 sec
[ Info: VUMPS 111:	obj = -3.266942906048e-01	err = 3.3265129842e-01	time = 0.05 sec
[ Info: VUMPS 112:	obj = -3.089623364833e-02	err = 3.8736213578e-01	time = 0.03 sec
[ Info: VUMPS 113:	obj = +1.187921857755e-01	err = 3.5846597520e-01	time = 0.03 sec
[ Info: VUMPS 114:	obj = -6.812831791961e-02	err = 3.6601110780e-01	time = 0.03 sec
[ Info: VUMPS 115:	obj = -2.169545739645e-02	err = 3.7328578080e-01	time = 0.02 sec
[ Info: VUMPS 116:	obj = -6.651891606433e-02	err = 3.7799893055e-01	time = 0.03 sec
[ Info: VUMPS 117:	obj = -7.188213532364e-02	err = 3.9041626270e-01	time = 0.02 sec
[ Info: VUMPS 118:	obj = -1.858708677528e-01	err = 3.8364886262e-01	time = 0.02 sec
[ Info: VUMPS 119:	obj = -1.426092420792e-01	err = 4.0022614912e-01	time = 0.04 sec
[ Info: VUMPS 120:	obj = -5.966800560939e-02	err = 3.7475162153e-01	time = 0.03 sec
[ Info: VUMPS 121:	obj = -1.927011968813e-01	err = 3.7213373940e-01	time = 0.02 sec
[ Info: VUMPS 122:	obj = -1.332482420009e-01	err = 3.9823332335e-01	time = 0.03 sec
[ Info: VUMPS 123:	obj = -5.583604028721e-02	err = 3.8811845128e-01	time = 0.02 sec
[ Info: VUMPS 124:	obj = -1.173769997886e-01	err = 3.9758982598e-01	time = 0.03 sec
[ Info: VUMPS 125:	obj = -1.623302909431e-01	err = 3.9006196969e-01	time = 0.03 sec
[ Info: VUMPS 126:	obj = -1.945385790084e-01	err = 3.7801724719e-01	time = 0.06 sec
[ Info: VUMPS 127:	obj = -1.198866351263e-01	err = 3.8641158608e-01	time = 0.04 sec
[ Info: VUMPS 128:	obj = -4.982327612008e-02	err = 3.8983422860e-01	time = 0.02 sec
[ Info: VUMPS 129:	obj = -1.566708742911e-01	err = 3.6570433006e-01	time = 0.03 sec
[ Info: VUMPS 130:	obj = -3.746845720087e-01	err = 2.7790316765e-01	time = 0.03 sec
[ Info: VUMPS 131:	obj = -4.113990855397e-01	err = 2.1938734166e-01	time = 0.04 sec
[ Info: VUMPS 132:	obj = +2.987037930122e-02	err = 3.9820435991e-01	time = 0.03 sec
[ Info: VUMPS 133:	obj = -1.361451053253e-01	err = 3.5853656852e-01	time = 0.04 sec
[ Info: VUMPS 134:	obj = +1.459826659542e-02	err = 3.6344259860e-01	time = 0.02 sec
[ Info: VUMPS 135:	obj = -1.625462338965e-01	err = 3.8750540849e-01	time = 0.03 sec
[ Info: VUMPS 136:	obj = -1.945260498995e-01	err = 3.7322465363e-01	time = 0.03 sec
[ Info: VUMPS 137:	obj = -2.378874547137e-01	err = 3.5236618024e-01	time = 0.02 sec
[ Info: VUMPS 138:	obj = -3.397752713889e-01	err = 3.1681390396e-01	time = 0.04 sec
[ Info: VUMPS 139:	obj = +3.872021456876e-02	err = 3.9044746200e-01	time = 0.03 sec
[ Info: VUMPS 140:	obj = -2.931409673729e-02	err = 4.0519520800e-01	time = 0.03 sec
[ Info: VUMPS 141:	obj = -3.558928281290e-02	err = 3.9408216725e-01	time = 0.03 sec
[ Info: VUMPS 142:	obj = -7.419482234602e-02	err = 3.5450836272e-01	time = 0.03 sec
[ Info: VUMPS 143:	obj = -1.320472388492e-01	err = 3.8629858112e-01	time = 0.02 sec
[ Info: VUMPS 144:	obj = -9.540887842838e-02	err = 3.6612197907e-01	time = 0.03 sec
[ Info: VUMPS 145:	obj = -1.425512039944e-01	err = 3.8151598515e-01	time = 0.02 sec
[ Info: VUMPS 146:	obj = -1.889146812574e-01	err = 3.7124283218e-01	time = 0.03 sec
[ Info: VUMPS 147:	obj = -2.150431811539e-01	err = 3.5676583668e-01	time = 0.02 sec
[ Info: VUMPS 148:	obj = -5.190628673026e-02	err = 3.8934562140e-01	time = 0.02 sec
[ Info: VUMPS 149:	obj = -1.783612875181e-01	err = 3.6973710837e-01	time = 0.03 sec
[ Info: VUMPS 150:	obj = -1.973875305247e-01	err = 3.7126107705e-01	time = 0.02 sec
[ Info: VUMPS 151:	obj = -2.899275740844e-01	err = 3.6876702526e-01	time = 0.03 sec
[ Info: VUMPS 152:	obj = -1.958917623537e-01	err = 3.6026243367e-01	time = 0.04 sec
[ Info: VUMPS 153:	obj = -1.658053486005e-01	err = 3.7110180704e-01	time = 0.03 sec
[ Info: VUMPS 154:	obj = -1.826177780294e-01	err = 4.1264125075e-01	time = 0.06 sec
[ Info: VUMPS 155:	obj = -1.059120939251e-02	err = 3.9019355474e-01	time = 0.04 sec
[ Info: VUMPS 156:	obj = -1.197063315403e-01	err = 3.8266860150e-01	time = 0.03 sec
[ Info: VUMPS 157:	obj = -1.518053401692e-01	err = 3.9773151275e-01	time = 0.03 sec
[ Info: VUMPS 158:	obj = -3.627713872160e-01	err = 3.0031072551e-01	time = 0.03 sec
[ Info: VUMPS 159:	obj = -3.817082298432e-01	err = 2.7493910948e-01	time = 0.04 sec
[ Info: VUMPS 160:	obj = -3.859619689587e-01	err = 2.7048867198e-01	time = 0.04 sec
[ Info: VUMPS 161:	obj = -4.307827862211e-01	err = 1.2679659771e-01	time = 0.05 sec
[ Info: VUMPS 162:	obj = -4.409832181424e-01	err = 5.1946083987e-02	time = 0.05 sec
[ Info: VUMPS 163:	obj = -3.974418237321e-01	err = 2.6311596995e-01	time = 0.06 sec
[ Info: VUMPS 164:	obj = -1.705693106104e-01	err = 3.7489488650e-01	time = 0.04 sec
[ Info: VUMPS 165:	obj = -2.811377309143e-01	err = 3.5373076558e-01	time = 0.02 sec
[ Info: VUMPS 166:	obj = -3.580031548549e-01	err = 3.0296648457e-01	time = 0.03 sec
[ Info: VUMPS 167:	obj = -3.498256310994e-01	err = 3.0138171612e-01	time = 0.04 sec
[ Info: VUMPS 168:	obj = -3.961930709656e-01	err = 2.3106926668e-01	time = 0.05 sec
[ Info: VUMPS 169:	obj = -3.012322573839e-01	err = 3.4871517230e-01	time = 0.05 sec
[ Info: VUMPS 170:	obj = -3.236118252223e-01	err = 3.2683561230e-01	time = 0.03 sec
[ Info: VUMPS 171:	obj = -4.207960751090e-01	err = 1.9439910804e-01	time = 0.04 sec
[ Info: VUMPS 172:	obj = +6.486820083537e-02	err = 3.7997567109e-01	time = 0.04 sec
[ Info: VUMPS 173:	obj = -2.205989590288e-01	err = 3.5392631042e-01	time = 0.02 sec
[ Info: VUMPS 174:	obj = -3.040541044421e-01	err = 3.4795284083e-01	time = 0.02 sec
[ Info: VUMPS 175:	obj = -2.700316483862e-01	err = 3.5371268695e-01	time = 0.04 sec
[ Info: VUMPS 176:	obj = -1.451879669536e-01	err = 3.7441207912e-01	time = 0.03 sec
[ Info: VUMPS 177:	obj = -1.675392520421e-01	err = 3.5821939134e-01	time = 0.03 sec
[ Info: VUMPS 178:	obj = -1.067646782592e-01	err = 3.5902159365e-01	time = 0.02 sec
[ Info: VUMPS 179:	obj = +5.087477936752e-02	err = 3.8379007723e-01	time = 0.06 sec
[ Info: VUMPS 180:	obj = +2.700648055200e-02	err = 3.5560967889e-01	time = 0.03 sec
[ Info: VUMPS 181:	obj = -4.040983881654e-02	err = 3.1955366201e-01	time = 0.03 sec
[ Info: VUMPS 182:	obj = -2.974889920215e-01	err = 3.4443915929e-01	time = 0.03 sec
[ Info: VUMPS 183:	obj = -3.470958294750e-01	err = 3.1478705957e-01	time = 0.03 sec
[ Info: VUMPS 184:	obj = -2.257989223447e-01	err = 3.8360838997e-01	time = 0.04 sec
[ Info: VUMPS 185:	obj = -1.892095032963e-01	err = 3.9861950100e-01	time = 0.04 sec
[ Info: VUMPS 186:	obj = -1.178703030512e-01	err = 3.9409430451e-01	time = 0.03 sec
[ Info: VUMPS 187:	obj = -1.803058178275e-01	err = 3.8179418477e-01	time = 0.02 sec
[ Info: VUMPS 188:	obj = +2.026839055691e-02	err = 3.9195067790e-01	time = 0.02 sec
[ Info: VUMPS 189:	obj = -2.213025303557e-01	err = 3.6496697752e-01	time = 0.03 sec
[ Info: VUMPS 190:	obj = -3.308949018046e-01	err = 3.0825188354e-01	time = 0.03 sec
[ Info: VUMPS 191:	obj = -3.358030111876e-01	err = 3.2213471244e-01	time = 0.04 sec
[ Info: VUMPS 192:	obj = -3.879783512495e-01	err = 2.6554883101e-01	time = 0.04 sec
[ Info: VUMPS 193:	obj = -4.235762853319e-01	err = 1.6965146723e-01	time = 0.04 sec
[ Info: VUMPS 194:	obj = -2.904118238494e-01	err = 3.5667684109e-01	time = 0.06 sec
[ Info: VUMPS 195:	obj = -2.142437785647e-01	err = 3.8141868514e-01	time = 0.04 sec
[ Info: VUMPS 196:	obj = -7.074078690463e-02	err = 3.4787569806e-01	time = 0.04 sec
[ Info: VUMPS 197:	obj = -1.976927769100e-01	err = 3.7553265539e-01	time = 0.02 sec
[ Info: VUMPS 198:	obj = -2.828727739426e-01	err = 3.5757076377e-01	time = 0.03 sec
[ Info: VUMPS 199:	obj = -2.312961866796e-01	err = 3.7739801742e-01	time = 0.04 sec
┌ Warning: VUMPS cancel 200:	obj = -2.166543836900e-01	err = 3.7105307386e-01	time = 6.55 sec
└ @ MPSKit ~/git/MPSKit.jl/src/algorithms/groundstate/vumps.jl:73

````

As you can see, VUMPS struggles to converge.
On it's own, that is already quite curious.
Maybe we can do better using another algorithm, such as gradient descent.

````julia
groundstate, cache, delta = find_groundstate(state, H, GradientGrassmann(; maxiter=20));
````

````
[ Info: CG: initializing with f = 0.249995294058, ‖∇f‖ = 4.4866e-03
┌ Warning: CG: not converged to requested tol after 20 iterations and time 8.67 s: f = -0.441284034826, ‖∇f‖ = 1.1624e-02
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
  <clipPath id="clip180">
    <rect x="0" y="0" width="2400" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip180)" d="M0 1600 L2400 1600 L2400 0 L0 0  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip181">
    <rect x="480" y="0" width="1681" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip180)" d="M249.704 1423.18 L2352.76 1423.18 L2352.76 47.2441 L249.704 47.2441  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip182">
    <rect x="249" y="47" width="2104" height="1377"/>
  </clipPath>
</defs>
<polyline clip-path="url(#clip182)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="309.224,1423.18 309.224,47.2441 "/>
<polyline clip-path="url(#clip182)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="639.893,1423.18 639.893,47.2441 "/>
<polyline clip-path="url(#clip182)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="970.561,1423.18 970.561,47.2441 "/>
<polyline clip-path="url(#clip182)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1301.23,1423.18 1301.23,47.2441 "/>
<polyline clip-path="url(#clip182)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1631.9,1423.18 1631.9,47.2441 "/>
<polyline clip-path="url(#clip182)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1962.57,1423.18 1962.57,47.2441 "/>
<polyline clip-path="url(#clip182)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="2293.24,1423.18 2293.24,47.2441 "/>
<polyline clip-path="url(#clip182)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="249.704,1256.34 2352.76,1256.34 "/>
<polyline clip-path="url(#clip182)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="249.704,1014.52 2352.76,1014.52 "/>
<polyline clip-path="url(#clip182)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="249.704,772.701 2352.76,772.701 "/>
<polyline clip-path="url(#clip182)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="249.704,530.882 2352.76,530.882 "/>
<polyline clip-path="url(#clip182)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="249.704,289.063 2352.76,289.063 "/>
<polyline clip-path="url(#clip180)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="249.704,1423.18 2352.76,1423.18 "/>
<polyline clip-path="url(#clip180)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="309.224,1423.18 309.224,1404.28 "/>
<polyline clip-path="url(#clip180)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="639.893,1423.18 639.893,1404.28 "/>
<polyline clip-path="url(#clip180)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="970.561,1423.18 970.561,1404.28 "/>
<polyline clip-path="url(#clip180)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1301.23,1423.18 1301.23,1404.28 "/>
<polyline clip-path="url(#clip180)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1631.9,1423.18 1631.9,1404.28 "/>
<polyline clip-path="url(#clip180)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1962.57,1423.18 1962.57,1404.28 "/>
<polyline clip-path="url(#clip180)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="2293.24,1423.18 2293.24,1404.28 "/>
<path clip-path="url(#clip180)" d="M262.986 1454.1 Q259.375 1454.1 257.546 1457.66 Q255.741 1461.2 255.741 1468.33 Q255.741 1475.44 257.546 1479.01 Q259.375 1482.55 262.986 1482.55 Q266.62 1482.55 268.426 1479.01 Q270.255 1475.44 270.255 1468.33 Q270.255 1461.2 268.426 1457.66 Q266.62 1454.1 262.986 1454.1 M262.986 1450.39 Q268.796 1450.39 271.852 1455 Q274.931 1459.58 274.931 1468.33 Q274.931 1477.06 271.852 1481.67 Q268.796 1486.25 262.986 1486.25 Q257.176 1486.25 254.097 1481.67 Q251.042 1477.06 251.042 1468.33 Q251.042 1459.58 254.097 1455 Q257.176 1450.39 262.986 1450.39 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M290.116 1451.02 L294.051 1451.02 L282.014 1489.98 L278.079 1489.98 L290.116 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M306.088 1451.02 L310.023 1451.02 L297.986 1489.98 L294.051 1489.98 L306.088 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M315.903 1481.64 L323.541 1481.64 L323.541 1455.28 L315.231 1456.95 L315.231 1452.69 L323.495 1451.02 L328.171 1451.02 L328.171 1481.64 L335.81 1481.64 L335.81 1485.58 L315.903 1485.58 L315.903 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M341.898 1459.65 L366.736 1459.65 L366.736 1463.91 L363.472 1463.91 L363.472 1479.84 Q363.472 1481.51 364.027 1482.25 Q364.606 1482.96 365.879 1482.96 Q366.226 1482.96 366.736 1482.92 Q367.245 1482.85 367.407 1482.83 L367.407 1485.9 Q366.597 1486.2 365.74 1486.34 Q364.884 1486.48 364.027 1486.48 Q361.25 1486.48 360.185 1484.98 Q359.12 1483.45 359.12 1479.38 L359.12 1463.91 L349.56 1463.91 L349.56 1485.58 L345.208 1485.58 L345.208 1463.91 L341.898 1463.91 L341.898 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M583.423 1481.64 L591.062 1481.64 L591.062 1455.28 L582.752 1456.95 L582.752 1452.69 L591.016 1451.02 L595.692 1451.02 L595.692 1481.64 L603.331 1481.64 L603.331 1485.58 L583.423 1485.58 L583.423 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M619.743 1451.02 L623.678 1451.02 L611.641 1489.98 L607.706 1489.98 L619.743 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M635.715 1451.02 L639.65 1451.02 L627.613 1489.98 L623.678 1489.98 L635.715 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M658.886 1466.95 Q662.242 1467.66 664.117 1469.93 Q666.015 1472.2 666.015 1475.53 Q666.015 1480.65 662.497 1483.45 Q658.978 1486.25 652.497 1486.25 Q650.321 1486.25 648.006 1485.81 Q645.715 1485.39 643.261 1484.54 L643.261 1480.02 Q645.205 1481.16 647.52 1481.74 Q649.835 1482.32 652.358 1482.32 Q656.756 1482.32 659.048 1480.58 Q661.363 1478.84 661.363 1475.53 Q661.363 1472.48 659.21 1470.77 Q657.08 1469.03 653.261 1469.03 L649.233 1469.03 L649.233 1465.19 L653.446 1465.19 Q656.895 1465.19 658.724 1463.82 Q660.553 1462.43 660.553 1459.84 Q660.553 1457.18 658.654 1455.77 Q656.779 1454.33 653.261 1454.33 Q651.34 1454.33 649.141 1454.75 Q646.942 1455.16 644.303 1456.04 L644.303 1451.88 Q646.965 1451.14 649.279 1450.77 Q651.617 1450.39 653.678 1450.39 Q659.002 1450.39 662.103 1452.83 Q665.205 1455.23 665.205 1459.35 Q665.205 1462.22 663.562 1464.21 Q661.918 1466.18 658.886 1466.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M671.525 1459.65 L696.363 1459.65 L696.363 1463.91 L693.099 1463.91 L693.099 1479.84 Q693.099 1481.51 693.654 1482.25 Q694.233 1482.96 695.506 1482.96 Q695.853 1482.96 696.363 1482.92 Q696.872 1482.85 697.034 1482.83 L697.034 1485.9 Q696.224 1486.2 695.367 1486.34 Q694.511 1486.48 693.654 1486.48 Q690.876 1486.48 689.812 1484.98 Q688.747 1483.45 688.747 1479.38 L688.747 1463.91 L679.187 1463.91 L679.187 1485.58 L674.835 1485.58 L674.835 1463.91 L671.525 1463.91 L671.525 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M918.177 1481.64 L934.497 1481.64 L934.497 1485.58 L912.553 1485.58 L912.553 1481.64 Q915.215 1478.89 919.798 1474.26 Q924.404 1469.61 925.585 1468.27 Q927.83 1465.74 928.71 1464.01 Q929.613 1462.25 929.613 1460.56 Q929.613 1457.8 927.668 1456.07 Q925.747 1454.33 922.645 1454.33 Q920.446 1454.33 917.992 1455.09 Q915.562 1455.86 912.784 1457.41 L912.784 1452.69 Q915.608 1451.55 918.062 1450.97 Q920.515 1450.39 922.552 1450.39 Q927.923 1450.39 931.117 1453.08 Q934.312 1455.77 934.312 1460.26 Q934.312 1462.39 933.501 1464.31 Q932.714 1466.2 930.608 1468.8 Q930.029 1469.47 926.927 1472.69 Q923.826 1475.88 918.177 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M951.279 1451.02 L955.214 1451.02 L943.177 1489.98 L939.242 1489.98 L951.279 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M967.251 1451.02 L971.186 1451.02 L959.149 1489.98 L955.214 1489.98 L967.251 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M990.422 1466.95 Q993.779 1467.66 995.654 1469.93 Q997.552 1472.2 997.552 1475.53 Q997.552 1480.65 994.034 1483.45 Q990.515 1486.25 984.034 1486.25 Q981.858 1486.25 979.543 1485.81 Q977.251 1485.39 974.798 1484.54 L974.798 1480.02 Q976.742 1481.16 979.057 1481.74 Q981.372 1482.32 983.895 1482.32 Q988.293 1482.32 990.585 1480.58 Q992.899 1478.84 992.899 1475.53 Q992.899 1472.48 990.747 1470.77 Q988.617 1469.03 984.797 1469.03 L980.77 1469.03 L980.77 1465.19 L984.983 1465.19 Q988.432 1465.19 990.26 1463.82 Q992.089 1462.43 992.089 1459.84 Q992.089 1457.18 990.191 1455.77 Q988.316 1454.33 984.797 1454.33 Q982.876 1454.33 980.677 1454.75 Q978.478 1455.16 975.839 1456.04 L975.839 1451.88 Q978.501 1451.14 980.816 1450.77 Q983.154 1450.39 985.214 1450.39 Q990.538 1450.39 993.64 1452.83 Q996.742 1455.23 996.742 1459.35 Q996.742 1462.22 995.098 1464.21 Q993.455 1466.18 990.422 1466.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M1003.06 1459.65 L1027.9 1459.65 L1027.9 1463.91 L1024.64 1463.91 L1024.64 1479.84 Q1024.64 1481.51 1025.19 1482.25 Q1025.77 1482.96 1027.04 1482.96 Q1027.39 1482.96 1027.9 1482.92 Q1028.41 1482.85 1028.57 1482.83 L1028.57 1485.9 Q1027.76 1486.2 1026.9 1486.34 Q1026.05 1486.48 1025.19 1486.48 Q1022.41 1486.48 1021.35 1484.98 Q1020.28 1483.45 1020.28 1479.38 L1020.28 1463.91 L1010.72 1463.91 L1010.72 1485.58 L1006.37 1485.58 L1006.37 1463.91 L1003.06 1463.91 L1003.06 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M1244.76 1481.64 L1252.4 1481.64 L1252.4 1455.28 L1244.09 1456.95 L1244.09 1452.69 L1252.35 1451.02 L1257.03 1451.02 L1257.03 1481.64 L1264.67 1481.64 L1264.67 1485.58 L1244.76 1485.58 L1244.76 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M1281.08 1451.02 L1285.01 1451.02 L1272.98 1489.98 L1269.04 1489.98 L1281.08 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M1297.05 1451.02 L1300.99 1451.02 L1288.95 1489.98 L1285.01 1489.98 L1297.05 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M1306.87 1481.64 L1314.51 1481.64 L1314.51 1455.28 L1306.2 1456.95 L1306.2 1452.69 L1314.46 1451.02 L1319.13 1451.02 L1319.13 1481.64 L1326.77 1481.64 L1326.77 1485.58 L1306.87 1485.58 L1306.87 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M1332.86 1459.65 L1357.7 1459.65 L1357.7 1463.91 L1354.44 1463.91 L1354.44 1479.84 Q1354.44 1481.51 1354.99 1482.25 Q1355.57 1482.96 1356.84 1482.96 Q1357.19 1482.96 1357.7 1482.92 Q1358.21 1482.85 1358.37 1482.83 L1358.37 1485.9 Q1357.56 1486.2 1356.7 1486.34 Q1355.85 1486.48 1354.99 1486.48 Q1352.21 1486.48 1351.15 1484.98 Q1350.08 1483.45 1350.08 1479.38 L1350.08 1463.91 L1340.52 1463.91 L1340.52 1485.58 L1336.17 1485.58 L1336.17 1463.91 L1332.86 1463.91 L1332.86 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M1588.91 1455.09 L1577.11 1473.54 L1588.91 1473.54 L1588.91 1455.09 M1587.69 1451.02 L1593.57 1451.02 L1593.57 1473.54 L1598.5 1473.54 L1598.5 1477.43 L1593.57 1477.43 L1593.57 1485.58 L1588.91 1485.58 L1588.91 1477.43 L1573.31 1477.43 L1573.31 1472.92 L1587.69 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M1613.19 1451.02 L1617.13 1451.02 L1605.09 1489.98 L1601.16 1489.98 L1613.19 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M1629.17 1451.02 L1633.1 1451.02 L1621.07 1489.98 L1617.13 1489.98 L1629.17 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M1652.34 1466.95 Q1655.69 1467.66 1657.57 1469.93 Q1659.47 1472.2 1659.47 1475.53 Q1659.47 1480.65 1655.95 1483.45 Q1652.43 1486.25 1645.95 1486.25 Q1643.77 1486.25 1641.46 1485.81 Q1639.17 1485.39 1636.71 1484.54 L1636.71 1480.02 Q1638.66 1481.16 1640.97 1481.74 Q1643.29 1482.32 1645.81 1482.32 Q1650.21 1482.32 1652.5 1480.58 Q1654.82 1478.84 1654.82 1475.53 Q1654.82 1472.48 1652.66 1470.77 Q1650.53 1469.03 1646.71 1469.03 L1642.69 1469.03 L1642.69 1465.19 L1646.9 1465.19 Q1650.35 1465.19 1652.18 1463.82 Q1654 1462.43 1654 1459.84 Q1654 1457.18 1652.11 1455.77 Q1650.23 1454.33 1646.71 1454.33 Q1644.79 1454.33 1642.59 1454.75 Q1640.39 1455.16 1637.75 1456.04 L1637.75 1451.88 Q1640.42 1451.14 1642.73 1450.77 Q1645.07 1450.39 1647.13 1450.39 Q1652.45 1450.39 1655.56 1452.83 Q1658.66 1455.23 1658.66 1459.35 Q1658.66 1462.22 1657.01 1464.21 Q1655.37 1466.18 1652.34 1466.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M1664.98 1459.65 L1689.81 1459.65 L1689.81 1463.91 L1686.55 1463.91 L1686.55 1479.84 Q1686.55 1481.51 1687.11 1482.25 Q1687.69 1482.96 1688.96 1482.96 Q1689.31 1482.96 1689.81 1482.92 Q1690.32 1482.85 1690.49 1482.83 L1690.49 1485.9 Q1689.68 1486.2 1688.82 1486.34 Q1687.96 1486.48 1687.11 1486.48 Q1684.33 1486.48 1683.26 1484.98 Q1682.2 1483.45 1682.2 1479.38 L1682.2 1463.91 L1672.64 1463.91 L1672.64 1485.58 L1668.29 1485.58 L1668.29 1463.91 L1664.98 1463.91 L1664.98 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M1906.11 1451.02 L1924.47 1451.02 L1924.47 1454.96 L1910.39 1454.96 L1910.39 1463.43 Q1911.41 1463.08 1912.43 1462.92 Q1913.45 1462.73 1914.47 1462.73 Q1920.25 1462.73 1923.63 1465.9 Q1927.01 1469.08 1927.01 1474.49 Q1927.01 1480.07 1923.54 1483.17 Q1920.07 1486.25 1913.75 1486.25 Q1911.57 1486.25 1909.3 1485.88 Q1907.06 1485.51 1904.65 1484.77 L1904.65 1480.07 Q1906.73 1481.2 1908.96 1481.76 Q1911.18 1482.32 1913.66 1482.32 Q1917.66 1482.32 1920 1480.21 Q1922.34 1478.1 1922.34 1474.49 Q1922.34 1470.88 1920 1468.77 Q1917.66 1466.67 1913.66 1466.67 Q1911.78 1466.67 1909.91 1467.08 Q1908.05 1467.5 1906.11 1468.38 L1906.11 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M1943.19 1451.02 L1947.13 1451.02 L1935.09 1489.98 L1931.16 1489.98 L1943.19 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M1959.16 1451.02 L1963.1 1451.02 L1951.06 1489.98 L1947.13 1489.98 L1959.16 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M1982.34 1466.95 Q1985.69 1467.66 1987.57 1469.93 Q1989.47 1472.2 1989.47 1475.53 Q1989.47 1480.65 1985.95 1483.45 Q1982.43 1486.25 1975.95 1486.25 Q1973.77 1486.25 1971.46 1485.81 Q1969.16 1485.39 1966.71 1484.54 L1966.71 1480.02 Q1968.65 1481.16 1970.97 1481.74 Q1973.28 1482.32 1975.81 1482.32 Q1980.21 1482.32 1982.5 1480.58 Q1984.81 1478.84 1984.81 1475.53 Q1984.81 1472.48 1982.66 1470.77 Q1980.53 1469.03 1976.71 1469.03 L1972.68 1469.03 L1972.68 1465.19 L1976.9 1465.19 Q1980.34 1465.19 1982.17 1463.82 Q1984 1462.43 1984 1459.84 Q1984 1457.18 1982.1 1455.77 Q1980.23 1454.33 1976.71 1454.33 Q1974.79 1454.33 1972.59 1454.75 Q1970.39 1455.16 1967.75 1456.04 L1967.75 1451.88 Q1970.41 1451.14 1972.73 1450.77 Q1975.07 1450.39 1977.13 1450.39 Q1982.45 1450.39 1985.55 1452.83 Q1988.65 1455.23 1988.65 1459.35 Q1988.65 1462.22 1987.01 1464.21 Q1985.37 1466.18 1982.34 1466.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M1994.97 1459.65 L2019.81 1459.65 L2019.81 1463.91 L2016.55 1463.91 L2016.55 1479.84 Q2016.55 1481.51 2017.1 1482.25 Q2017.68 1482.96 2018.96 1482.96 Q2019.3 1482.96 2019.81 1482.92 Q2020.32 1482.85 2020.48 1482.83 L2020.48 1485.9 Q2019.67 1486.2 2018.82 1486.34 Q2017.96 1486.48 2017.1 1486.48 Q2014.33 1486.48 2013.26 1484.98 Q2012.2 1483.45 2012.2 1479.38 L2012.2 1463.91 L2002.64 1463.91 L2002.64 1485.58 L1998.28 1485.58 L1998.28 1463.91 L1994.97 1463.91 L1994.97 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M2240.85 1481.64 L2257.17 1481.64 L2257.17 1485.58 L2235.23 1485.58 L2235.23 1481.64 Q2237.89 1478.89 2242.47 1474.26 Q2247.08 1469.61 2248.26 1468.27 Q2250.5 1465.74 2251.38 1464.01 Q2252.29 1462.25 2252.29 1460.56 Q2252.29 1457.8 2250.34 1456.07 Q2248.42 1454.33 2245.32 1454.33 Q2243.12 1454.33 2240.67 1455.09 Q2238.24 1455.86 2235.46 1457.41 L2235.46 1452.69 Q2238.28 1451.55 2240.74 1450.97 Q2243.19 1450.39 2245.23 1450.39 Q2250.6 1450.39 2253.79 1453.08 Q2256.99 1455.77 2256.99 1460.26 Q2256.99 1462.39 2256.18 1464.31 Q2255.39 1466.2 2253.28 1468.8 Q2252.7 1469.47 2249.6 1472.69 Q2246.5 1475.88 2240.85 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M2273.95 1451.02 L2277.89 1451.02 L2265.85 1489.98 L2261.92 1489.98 L2273.95 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M2289.93 1451.02 L2293.86 1451.02 L2281.82 1489.98 L2277.89 1489.98 L2289.93 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M2299.74 1481.64 L2307.38 1481.64 L2307.38 1455.28 L2299.07 1456.95 L2299.07 1452.69 L2307.33 1451.02 L2312.01 1451.02 L2312.01 1481.64 L2319.65 1481.64 L2319.65 1485.58 L2299.74 1485.58 L2299.74 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M2325.74 1459.65 L2350.57 1459.65 L2350.57 1463.91 L2347.31 1463.91 L2347.31 1479.84 Q2347.31 1481.51 2347.86 1482.25 Q2348.44 1482.96 2349.72 1482.96 Q2350.06 1482.96 2350.57 1482.92 Q2351.08 1482.85 2351.24 1482.83 L2351.24 1485.9 Q2350.43 1486.2 2349.58 1486.34 Q2348.72 1486.48 2347.86 1486.48 Q2345.09 1486.48 2344.02 1484.98 Q2342.96 1483.45 2342.96 1479.38 L2342.96 1463.91 L2333.4 1463.91 L2333.4 1485.58 L2329.05 1485.58 L2329.05 1463.91 L2325.74 1463.91 L2325.74 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M1311.35 1545.45 L1291.08 1545.45 Q1291.55 1554.96 1293.75 1559 Q1296.49 1563.97 1301.23 1563.97 Q1306 1563.97 1308.65 1558.97 Q1310.97 1554.58 1311.35 1545.45 M1311.26 1540.03 Q1310.36 1531 1308.65 1527.81 Q1305.91 1522.78 1301.23 1522.78 Q1296.36 1522.78 1293.78 1527.75 Q1291.75 1531.76 1291.14 1540.03 L1311.26 1540.03 M1301.23 1518.01 Q1308.87 1518.01 1313.23 1524.76 Q1317.59 1531.47 1317.59 1543.38 Q1317.59 1555.25 1313.23 1562 Q1308.87 1568.78 1301.23 1568.78 Q1293.56 1568.78 1289.23 1562 Q1284.87 1555.25 1284.87 1543.38 Q1284.87 1531.47 1289.23 1524.76 Q1293.56 1518.01 1301.23 1518.01 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><polyline clip-path="url(#clip180)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="249.704,1423.18 249.704,47.2441 "/>
<polyline clip-path="url(#clip180)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="249.704,1256.34 268.602,1256.34 "/>
<polyline clip-path="url(#clip180)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="249.704,1014.52 268.602,1014.52 "/>
<polyline clip-path="url(#clip180)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="249.704,772.701 268.602,772.701 "/>
<polyline clip-path="url(#clip180)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="249.704,530.882 268.602,530.882 "/>
<polyline clip-path="url(#clip180)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="249.704,289.063 268.602,289.063 "/>
<path clip-path="url(#clip180)" d="M126.367 1242.14 Q122.755 1242.14 120.927 1245.7 Q119.121 1249.24 119.121 1256.37 Q119.121 1263.48 120.927 1267.05 Q122.755 1270.59 126.367 1270.59 Q130.001 1270.59 131.806 1267.05 Q133.635 1263.48 133.635 1256.37 Q133.635 1249.24 131.806 1245.7 Q130.001 1242.14 126.367 1242.14 M126.367 1238.43 Q132.177 1238.43 135.232 1243.04 Q138.311 1247.62 138.311 1256.37 Q138.311 1265.1 135.232 1269.71 Q132.177 1274.29 126.367 1274.29 Q120.556 1274.29 117.478 1269.71 Q114.422 1265.1 114.422 1256.37 Q114.422 1247.62 117.478 1243.04 Q120.556 1238.43 126.367 1238.43 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M146.529 1267.74 L151.413 1267.74 L151.413 1273.62 L146.529 1273.62 L146.529 1267.74 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M161.737 1272.9 L161.737 1268.64 Q163.496 1269.48 165.302 1269.92 Q167.107 1270.36 168.843 1270.36 Q173.473 1270.36 175.903 1267.25 Q178.357 1264.13 178.704 1257.79 Q177.362 1259.78 175.301 1260.84 Q173.241 1261.91 170.741 1261.91 Q165.556 1261.91 162.524 1258.78 Q159.515 1255.63 159.515 1250.19 Q159.515 1244.87 162.663 1241.65 Q165.811 1238.43 171.042 1238.43 Q177.038 1238.43 180.186 1243.04 Q183.357 1247.62 183.357 1256.37 Q183.357 1264.55 179.468 1269.43 Q175.602 1274.29 169.052 1274.29 Q167.292 1274.29 165.487 1273.94 Q163.681 1273.6 161.737 1272.9 M171.042 1258.25 Q174.19 1258.25 176.019 1256.1 Q177.871 1253.94 177.871 1250.19 Q177.871 1246.47 176.019 1244.31 Q174.19 1242.14 171.042 1242.14 Q167.894 1242.14 166.042 1244.31 Q164.214 1246.47 164.214 1250.19 Q164.214 1253.94 166.042 1256.1 Q167.894 1258.25 171.042 1258.25 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M201.76 1242.14 Q198.149 1242.14 196.32 1245.7 Q194.514 1249.24 194.514 1256.37 Q194.514 1263.48 196.32 1267.05 Q198.149 1270.59 201.76 1270.59 Q205.394 1270.59 207.199 1267.05 Q209.028 1263.48 209.028 1256.37 Q209.028 1249.24 207.199 1245.7 Q205.394 1242.14 201.76 1242.14 M201.76 1238.43 Q207.57 1238.43 210.625 1243.04 Q213.704 1247.62 213.704 1256.37 Q213.704 1265.1 210.625 1269.71 Q207.57 1274.29 201.76 1274.29 Q195.95 1274.29 192.871 1269.71 Q189.815 1265.1 189.815 1256.37 Q189.815 1247.62 192.871 1243.04 Q195.95 1238.43 201.76 1238.43 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M127.015 1000.32 Q123.404 1000.32 121.575 1003.88 Q119.769 1007.43 119.769 1014.56 Q119.769 1021.66 121.575 1025.23 Q123.404 1028.77 127.015 1028.77 Q130.649 1028.77 132.455 1025.23 Q134.283 1021.66 134.283 1014.56 Q134.283 1007.43 132.455 1003.88 Q130.649 1000.32 127.015 1000.32 M127.015 996.615 Q132.825 996.615 135.88 1001.22 Q138.959 1005.81 138.959 1014.56 Q138.959 1023.28 135.88 1027.89 Q132.825 1032.47 127.015 1032.47 Q121.205 1032.47 118.126 1027.89 Q115.07 1023.28 115.07 1014.56 Q115.07 1005.81 118.126 1001.22 Q121.205 996.615 127.015 996.615 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M147.177 1025.92 L152.061 1025.92 L152.061 1031.8 L147.177 1031.8 L147.177 1025.92 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M162.385 1031.08 L162.385 1026.82 Q164.144 1027.66 165.95 1028.1 Q167.755 1028.54 169.491 1028.54 Q174.121 1028.54 176.551 1025.43 Q179.005 1022.31 179.352 1015.97 Q178.01 1017.96 175.95 1019.02 Q173.889 1020.09 171.389 1020.09 Q166.204 1020.09 163.172 1016.96 Q160.163 1013.81 160.163 1008.37 Q160.163 1003.05 163.311 999.833 Q166.459 996.615 171.69 996.615 Q177.686 996.615 180.834 1001.22 Q184.005 1005.81 184.005 1014.56 Q184.005 1022.73 180.116 1027.61 Q176.251 1032.47 169.7 1032.47 Q167.94 1032.47 166.135 1032.12 Q164.329 1031.78 162.385 1031.08 M171.69 1016.43 Q174.839 1016.43 176.667 1014.28 Q178.519 1012.12 178.519 1008.37 Q178.519 1004.65 176.667 1002.49 Q174.839 1000.32 171.69 1000.32 Q168.542 1000.32 166.69 1002.49 Q164.862 1004.65 164.862 1008.37 Q164.862 1012.12 166.69 1014.28 Q168.542 1016.43 171.69 1016.43 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M206.574 1013.17 Q209.931 1013.88 211.806 1016.15 Q213.704 1018.42 213.704 1021.75 Q213.704 1026.87 210.186 1029.67 Q206.667 1032.47 200.186 1032.47 Q198.01 1032.47 195.695 1032.03 Q193.403 1031.62 190.95 1030.76 L190.95 1026.24 Q192.894 1027.38 195.209 1027.96 Q197.524 1028.54 200.047 1028.54 Q204.445 1028.54 206.737 1026.8 Q209.051 1025.06 209.051 1021.75 Q209.051 1018.7 206.899 1016.99 Q204.769 1015.25 200.95 1015.25 L196.922 1015.25 L196.922 1011.41 L201.135 1011.41 Q204.584 1011.41 206.412 1010.04 Q208.241 1008.65 208.241 1006.06 Q208.241 1003.4 206.343 1001.99 Q204.468 1000.55 200.95 1000.55 Q199.028 1000.55 196.829 1000.97 Q194.63 1001.38 191.991 1002.26 L191.991 998.097 Q194.653 997.356 196.968 996.986 Q199.306 996.615 201.366 996.615 Q206.69 996.615 209.792 999.046 Q212.894 1001.45 212.894 1005.57 Q212.894 1008.44 211.25 1010.43 Q209.607 1012.4 206.574 1013.17 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M126.205 758.5 Q122.593 758.5 120.765 762.065 Q118.959 765.606 118.959 772.736 Q118.959 779.842 120.765 783.407 Q122.593 786.949 126.205 786.949 Q129.839 786.949 131.644 783.407 Q133.473 779.842 133.473 772.736 Q133.473 765.606 131.644 762.065 Q129.839 758.5 126.205 758.5 M126.205 754.796 Q132.015 754.796 135.07 759.403 Q138.149 763.986 138.149 772.736 Q138.149 781.463 135.07 786.069 Q132.015 790.653 126.205 790.653 Q120.394 790.653 117.316 786.069 Q114.26 781.463 114.26 772.736 Q114.26 763.986 117.316 759.403 Q120.394 754.796 126.205 754.796 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M146.366 784.102 L151.251 784.102 L151.251 789.981 L146.366 789.981 L146.366 784.102 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M161.575 789.264 L161.575 785.004 Q163.334 785.838 165.14 786.278 Q166.945 786.717 168.681 786.717 Q173.311 786.717 175.741 783.616 Q178.195 780.491 178.542 774.148 Q177.2 776.139 175.139 777.204 Q173.079 778.268 170.579 778.268 Q165.394 778.268 162.362 775.143 Q159.353 771.995 159.353 766.556 Q159.353 761.231 162.501 758.014 Q165.649 754.796 170.88 754.796 Q176.876 754.796 180.024 759.403 Q183.195 763.986 183.195 772.736 Q183.195 780.907 179.306 785.792 Q175.44 790.653 168.889 790.653 Q167.13 790.653 165.325 790.305 Q163.519 789.958 161.575 789.264 M170.88 774.611 Q174.028 774.611 175.857 772.458 Q177.709 770.305 177.709 766.556 Q177.709 762.829 175.857 760.676 Q174.028 758.5 170.88 758.5 Q167.732 758.5 165.88 760.676 Q164.052 762.829 164.052 766.556 Q164.052 770.305 165.88 772.458 Q167.732 774.611 170.88 774.611 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M202.176 770.838 Q199.028 770.838 197.176 772.991 Q195.348 775.143 195.348 778.893 Q195.348 782.62 197.176 784.796 Q199.028 786.949 202.176 786.949 Q205.324 786.949 207.153 784.796 Q209.005 782.62 209.005 778.893 Q209.005 775.143 207.153 772.991 Q205.324 770.838 202.176 770.838 M211.459 756.185 L211.459 760.444 Q209.699 759.611 207.894 759.171 Q206.112 758.731 204.352 758.731 Q199.723 758.731 197.269 761.856 Q194.838 764.981 194.491 771.301 Q195.857 769.287 197.917 768.222 Q199.977 767.134 202.454 767.134 Q207.662 767.134 210.672 770.305 Q213.704 773.454 213.704 778.893 Q213.704 784.217 210.556 787.435 Q207.408 790.653 202.176 790.653 Q196.181 790.653 193.01 786.069 Q189.838 781.463 189.838 772.736 Q189.838 764.542 193.727 759.681 Q197.616 754.796 204.167 754.796 Q205.926 754.796 207.709 755.144 Q209.514 755.491 211.459 756.185 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M126.552 516.681 Q122.941 516.681 121.112 520.246 Q119.306 523.787 119.306 530.917 Q119.306 538.023 121.112 541.588 Q122.941 545.13 126.552 545.13 Q130.186 545.13 131.992 541.588 Q133.82 538.023 133.82 530.917 Q133.82 523.787 131.992 520.246 Q130.186 516.681 126.552 516.681 M126.552 512.977 Q132.362 512.977 135.417 517.584 Q138.496 522.167 138.496 530.917 Q138.496 539.644 135.417 544.25 Q132.362 548.834 126.552 548.834 Q120.742 548.834 117.663 544.25 Q114.607 539.644 114.607 530.917 Q114.607 522.167 117.663 517.584 Q120.742 512.977 126.552 512.977 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M146.714 542.283 L151.598 542.283 L151.598 548.162 L146.714 548.162 L146.714 542.283 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M161.922 547.445 L161.922 543.185 Q163.681 544.019 165.487 544.459 Q167.292 544.898 169.028 544.898 Q173.658 544.898 176.089 541.797 Q178.542 538.672 178.889 532.329 Q177.547 534.32 175.487 535.385 Q173.427 536.449 170.927 536.449 Q165.741 536.449 162.709 533.324 Q159.7 530.176 159.7 524.736 Q159.7 519.412 162.848 516.195 Q165.996 512.977 171.227 512.977 Q177.223 512.977 180.371 517.584 Q183.542 522.167 183.542 530.917 Q183.542 539.088 179.653 543.972 Q175.788 548.834 169.237 548.834 Q167.477 548.834 165.672 548.486 Q163.866 548.139 161.922 547.445 M171.227 532.792 Q174.376 532.792 176.204 530.639 Q178.056 528.486 178.056 524.736 Q178.056 521.01 176.204 518.857 Q174.376 516.681 171.227 516.681 Q168.079 516.681 166.227 518.857 Q164.399 521.01 164.399 524.736 Q164.399 528.486 166.227 530.639 Q168.079 532.792 171.227 532.792 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M192.084 547.445 L192.084 543.185 Q193.843 544.019 195.649 544.459 Q197.454 544.898 199.19 544.898 Q203.82 544.898 206.25 541.797 Q208.704 538.672 209.051 532.329 Q207.709 534.32 205.649 535.385 Q203.588 536.449 201.088 536.449 Q195.903 536.449 192.871 533.324 Q189.862 530.176 189.862 524.736 Q189.862 519.412 193.01 516.195 Q196.158 512.977 201.389 512.977 Q207.385 512.977 210.533 517.584 Q213.704 522.167 213.704 530.917 Q213.704 539.088 209.815 543.972 Q205.949 548.834 199.399 548.834 Q197.639 548.834 195.834 548.486 Q194.028 548.139 192.084 547.445 M201.389 532.792 Q204.537 532.792 206.366 530.639 Q208.218 528.486 208.218 524.736 Q208.218 521.01 206.366 518.857 Q204.537 516.681 201.389 516.681 Q198.241 516.681 196.389 518.857 Q194.561 521.01 194.561 524.736 Q194.561 528.486 196.389 530.639 Q198.241 532.792 201.389 532.792 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M118.774 302.408 L126.413 302.408 L126.413 276.042 L118.103 277.709 L118.103 273.45 L126.367 271.783 L131.042 271.783 L131.042 302.408 L138.681 302.408 L138.681 306.343 L118.774 306.343 L118.774 302.408 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M148.126 300.464 L153.01 300.464 L153.01 306.343 L148.126 306.343 L148.126 300.464 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M173.195 274.862 Q169.584 274.862 167.755 278.427 Q165.95 281.968 165.95 289.098 Q165.95 296.204 167.755 299.769 Q169.584 303.311 173.195 303.311 Q176.829 303.311 178.635 299.769 Q180.464 296.204 180.464 289.098 Q180.464 281.968 178.635 278.427 Q176.829 274.862 173.195 274.862 M173.195 271.158 Q179.005 271.158 182.061 275.765 Q185.139 280.348 185.139 289.098 Q185.139 297.825 182.061 302.431 Q179.005 307.014 173.195 307.014 Q167.385 307.014 164.306 302.431 Q161.251 297.825 161.251 289.098 Q161.251 280.348 164.306 275.765 Q167.385 271.158 173.195 271.158 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M197.385 302.408 L213.704 302.408 L213.704 306.343 L191.76 306.343 L191.76 302.408 Q194.422 299.653 199.005 295.024 Q203.612 290.371 204.792 289.028 Q207.037 286.505 207.917 284.769 Q208.82 283.01 208.82 281.32 Q208.82 278.566 206.875 276.829 Q204.954 275.093 201.852 275.093 Q199.653 275.093 197.2 275.857 Q194.769 276.621 191.991 278.172 L191.991 273.45 Q194.815 272.316 197.269 271.737 Q199.723 271.158 201.76 271.158 Q207.13 271.158 210.324 273.843 Q213.519 276.529 213.519 281.019 Q213.519 283.149 212.709 285.07 Q211.922 286.968 209.815 289.561 Q209.237 290.232 206.135 293.45 Q203.033 296.644 197.385 302.408 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M33.8307 724.772 Q33.2578 725.759 33.0032 726.937 Q32.7167 728.082 32.7167 729.483 Q32.7167 734.448 35.9632 737.122 Q39.1779 739.763 45.2253 739.763 L64.0042 739.763 L64.0042 745.652 L28.3562 745.652 L28.3562 739.763 L33.8944 739.763 Q30.6479 737.917 29.0883 734.957 Q27.4968 731.997 27.4968 727.764 Q27.4968 727.159 27.5923 726.427 Q27.656 725.695 27.8151 724.804 L33.8307 724.772 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><line clip-path="url(#clip182)" x1="2293.24" y1="450.276" x2="2293.24" y2="434.276" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="2293.24" y1="450.276" x2="2277.24" y2="450.276" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="2293.24" y1="450.276" x2="2293.24" y2="466.276" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="2293.24" y1="450.276" x2="2309.24" y2="450.276" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1301.23" y1="450.536" x2="1301.23" y2="434.536" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1301.23" y1="450.536" x2="1285.23" y2="450.536" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1301.23" y1="450.536" x2="1301.23" y2="466.536" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1301.23" y1="450.536" x2="1317.23" y2="450.536" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="2293.24" y1="758.408" x2="2293.24" y2="742.408" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="2293.24" y1="758.408" x2="2277.24" y2="758.408" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="2293.24" y1="758.408" x2="2293.24" y2="774.408" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="2293.24" y1="758.408" x2="2309.24" y2="758.408" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1301.23" y1="762.989" x2="1301.23" y2="746.989" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1301.23" y1="762.989" x2="1285.23" y2="762.989" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1301.23" y1="762.989" x2="1301.23" y2="778.989" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1301.23" y1="762.989" x2="1317.23" y2="762.989" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="519.251" y1="794.078" x2="519.251" y2="778.078" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="519.251" y1="794.078" x2="503.251" y2="794.078" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="519.251" y1="794.078" x2="519.251" y2="810.078" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="519.251" y1="794.078" x2="535.251" y2="794.078" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="2083.21" y1="794.078" x2="2083.21" y2="778.078" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="2083.21" y1="794.078" x2="2067.21" y2="794.078" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="2083.21" y1="794.078" x2="2083.21" y2="810.078" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="2083.21" y1="794.078" x2="2099.21" y2="794.078" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1091.2" y1="794.1" x2="1091.2" y2="778.1" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1091.2" y1="794.1" x2="1075.2" y2="794.1" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1091.2" y1="794.1" x2="1091.2" y2="810.1" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1091.2" y1="794.1" x2="1107.2" y2="794.1" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1511.26" y1="794.1" x2="1511.26" y2="778.1" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1511.26" y1="794.1" x2="1495.26" y2="794.1" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1511.26" y1="794.1" x2="1511.26" y2="810.1" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1511.26" y1="794.1" x2="1527.26" y2="794.1" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="592.75" y1="843.785" x2="592.75" y2="827.785" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="592.75" y1="843.785" x2="576.75" y2="843.785" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="592.75" y1="843.785" x2="592.75" y2="859.785" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="592.75" y1="843.785" x2="608.75" y2="843.785" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="2009.71" y1="843.785" x2="2009.71" y2="827.785" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="2009.71" y1="843.785" x2="1993.71" y2="843.785" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="2009.71" y1="843.785" x2="2009.71" y2="859.785" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="2009.71" y1="843.785" x2="2025.71" y2="843.785" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1584.76" y1="843.816" x2="1584.76" y2="827.816" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1584.76" y1="843.816" x2="1568.76" y2="843.816" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1584.76" y1="843.816" x2="1584.76" y2="859.816" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1584.76" y1="843.816" x2="1600.76" y2="843.816" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1017.7" y1="843.816" x2="1017.7" y2="827.816" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1017.7" y1="843.816" x2="1001.7" y2="843.816" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1017.7" y1="843.816" x2="1017.7" y2="859.816" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1017.7" y1="843.816" x2="1033.7" y2="843.816" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="810.834" y1="1083.03" x2="810.834" y2="1067.03" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="810.834" y1="1083.03" x2="794.834" y2="1083.03" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="810.834" y1="1083.03" x2="810.834" y2="1099.03" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="810.834" y1="1083.03" x2="826.834" y2="1083.03" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1791.63" y1="1083.03" x2="1791.63" y2="1067.03" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1791.63" y1="1083.03" x2="1775.63" y2="1083.03" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1791.63" y1="1083.03" x2="1791.63" y2="1099.03" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1791.63" y1="1083.03" x2="1807.63" y2="1083.03" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1802.84" y1="1083.04" x2="1802.84" y2="1067.04" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1802.84" y1="1083.04" x2="1786.84" y2="1083.04" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1802.84" y1="1083.04" x2="1802.84" y2="1099.04" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1802.84" y1="1083.04" x2="1818.84" y2="1083.04" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="799.621" y1="1083.04" x2="799.621" y2="1067.04" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="799.621" y1="1083.04" x2="783.621" y2="1083.04" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="799.621" y1="1083.04" x2="799.621" y2="1099.04" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="799.621" y1="1083.04" x2="815.621" y2="1083.04" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="371.229" y1="1369.31" x2="371.229" y2="1353.31" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="371.229" y1="1369.31" x2="355.229" y2="1369.31" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="371.229" y1="1369.31" x2="371.229" y2="1385.31" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="371.229" y1="1369.31" x2="387.229" y2="1369.31" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="2231.23" y1="1369.31" x2="2231.23" y2="1353.31" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="2231.23" y1="1369.31" x2="2215.23" y2="1369.31" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="2231.23" y1="1369.31" x2="2231.23" y2="1385.31" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="2231.23" y1="1369.31" x2="2247.23" y2="1369.31" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1363.23" y1="1369.35" x2="1363.23" y2="1353.35" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1363.23" y1="1369.35" x2="1347.23" y2="1369.35" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1363.23" y1="1369.35" x2="1363.23" y2="1385.35" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1363.23" y1="1369.35" x2="1379.23" y2="1369.35" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1239.23" y1="1369.35" x2="1239.23" y2="1353.35" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1239.23" y1="1369.35" x2="1223.23" y2="1369.35" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1239.23" y1="1369.35" x2="1239.23" y2="1385.35" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1239.23" y1="1369.35" x2="1255.23" y2="1369.35" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="2023.04" y1="1422.58" x2="2023.04" y2="1406.58" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="2023.04" y1="1422.58" x2="2007.04" y2="1422.58" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="2023.04" y1="1422.58" x2="2023.04" y2="1438.58" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="2023.04" y1="1422.58" x2="2039.04" y2="1422.58" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1031.04" y1="1423.18" x2="1031.04" y2="1407.18" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1031.04" y1="1423.18" x2="1015.04" y2="1423.18" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1031.04" y1="1423.18" x2="1031.04" y2="1439.18" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip182)" x1="1031.04" y1="1423.18" x2="1047.04" y2="1423.18" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<path clip-path="url(#clip180)" d="M1901.19 196.789 L2282.65 196.789 L2282.65 93.1086 L1901.19 93.1086  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<polyline clip-path="url(#clip180)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1901.19,196.789 2282.65,196.789 2282.65,93.1086 1901.19,93.1086 1901.19,196.789 "/>
<line clip-path="url(#clip180)" x1="1994.66" y1="144.949" x2="1994.66" y2="122.193" style="stroke:#009af9; stroke-width:4.55111; stroke-opacity:1"/>
<line clip-path="url(#clip180)" x1="1994.66" y1="144.949" x2="1971.91" y2="144.949" style="stroke:#009af9; stroke-width:4.55111; stroke-opacity:1"/>
<line clip-path="url(#clip180)" x1="1994.66" y1="144.949" x2="1994.66" y2="167.704" style="stroke:#009af9; stroke-width:4.55111; stroke-opacity:1"/>
<line clip-path="url(#clip180)" x1="1994.66" y1="144.949" x2="2017.42" y2="144.949" style="stroke:#009af9; stroke-width:4.55111; stroke-opacity:1"/>
<path clip-path="url(#clip180)" d="M2088.13 127.669 L2117.37 127.669 L2117.37 131.604 L2105.1 131.604 L2105.1 162.229 L2100.4 162.229 L2100.4 131.604 L2088.13 131.604 L2088.13 127.669 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M2129.75 140.284 Q2129.03 139.868 2128.18 139.682 Q2127.34 139.474 2126.32 139.474 Q2122.71 139.474 2120.77 141.835 Q2118.85 144.173 2118.85 148.571 L2118.85 162.229 L2114.57 162.229 L2114.57 136.303 L2118.85 136.303 L2118.85 140.331 Q2120.19 137.969 2122.34 136.835 Q2124.5 135.678 2127.57 135.678 Q2128.01 135.678 2128.55 135.747 Q2129.08 135.794 2129.73 135.909 L2129.75 140.284 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M2134.22 136.303 L2138.48 136.303 L2138.48 162.229 L2134.22 162.229 L2134.22 136.303 M2134.22 126.21 L2138.48 126.21 L2138.48 131.604 L2134.22 131.604 L2134.22 126.21 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M2144.33 136.303 L2148.85 136.303 L2156.95 158.062 L2165.05 136.303 L2169.57 136.303 L2159.84 162.229 L2154.06 162.229 L2144.33 136.303 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M2175.44 136.303 L2179.7 136.303 L2179.7 162.229 L2175.44 162.229 L2175.44 136.303 M2175.44 126.21 L2179.7 126.21 L2179.7 131.604 L2175.44 131.604 L2175.44 126.21 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M2200.4 149.196 Q2195.24 149.196 2193.25 150.377 Q2191.25 151.557 2191.25 154.405 Q2191.25 156.673 2192.74 158.016 Q2194.24 159.335 2196.81 159.335 Q2200.35 159.335 2202.48 156.835 Q2204.63 154.312 2204.63 150.145 L2204.63 149.196 L2200.4 149.196 M2208.89 147.437 L2208.89 162.229 L2204.63 162.229 L2204.63 158.293 Q2203.18 160.655 2201 161.789 Q2198.82 162.9 2195.68 162.9 Q2191.69 162.9 2189.33 160.678 Q2187 158.432 2187 154.682 Q2187 150.307 2189.91 148.085 Q2192.85 145.863 2198.66 145.863 L2204.63 145.863 L2204.63 145.446 Q2204.63 142.507 2202.69 140.909 Q2200.77 139.289 2197.27 139.289 Q2195.05 139.289 2192.94 139.821 Q2190.84 140.354 2188.89 141.419 L2188.89 137.483 Q2191.23 136.581 2193.43 136.141 Q2195.63 135.678 2197.71 135.678 Q2203.34 135.678 2206.12 138.594 Q2208.89 141.511 2208.89 147.437 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M2217.67 126.21 L2221.93 126.21 L2221.93 162.229 L2217.67 162.229 L2217.67 126.21 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M2241.07 126.257 Q2237.97 131.581 2236.46 136.789 Q2234.96 141.997 2234.96 147.344 Q2234.96 152.692 2236.46 157.946 Q2237.99 163.178 2241.07 168.479 L2237.37 168.479 Q2233.89 163.039 2232.16 157.784 Q2230.44 152.53 2230.44 147.344 Q2230.44 142.182 2232.16 136.951 Q2233.87 131.72 2237.37 126.257 L2241.07 126.257 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip180)" d="M2248.66 126.257 L2252.37 126.257 Q2255.84 131.72 2257.55 136.951 Q2259.29 142.182 2259.29 147.344 Q2259.29 152.53 2257.55 157.784 Q2255.84 163.039 2252.37 168.479 L2248.66 168.479 Q2251.74 163.178 2253.25 157.946 Q2254.77 152.692 2254.77 147.344 Q2254.77 141.997 2253.25 136.789 Q2251.74 131.581 2248.66 126.257 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /></svg>

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
[ Info: VUMPS init:	obj = +4.984019701133e-01	err = 8.5802e-02
[ Info: VUMPS   1:	obj = -1.587916149322e-01	err = 3.0275893671e-01	time = 0.04 sec
[ Info: VUMPS   2:	obj = -8.673455076862e-01	err = 1.2709823901e-01	time = 0.03 sec
[ Info: VUMPS   3:	obj = -8.851574658168e-01	err = 1.3169797608e-02	time = 0.03 sec
[ Info: VUMPS   4:	obj = -8.859091848496e-01	err = 6.4392637469e-03	time = 0.10 sec
[ Info: VUMPS   5:	obj = -8.861109088796e-01	err = 4.0495543537e-03	time = 0.03 sec
[ Info: VUMPS   6:	obj = -8.861799998365e-01	err = 3.2423747666e-03	time = 0.04 sec
[ Info: VUMPS   7:	obj = -8.862098845534e-01	err = 2.5878808961e-03	time = 0.04 sec
[ Info: VUMPS   8:	obj = -8.862231663400e-01	err = 2.2980643190e-03	time = 0.04 sec
[ Info: VUMPS   9:	obj = -8.862302410094e-01	err = 1.9030792239e-03	time = 0.04 sec
[ Info: VUMPS  10:	obj = -8.862337476758e-01	err = 1.5771139900e-03	time = 0.04 sec
[ Info: VUMPS  11:	obj = -8.862357477670e-01	err = 1.3386319635e-03	time = 0.05 sec
[ Info: VUMPS  12:	obj = -8.862368077969e-01	err = 1.1118556991e-03	time = 0.04 sec
[ Info: VUMPS  13:	obj = -8.862374143689e-01	err = 9.6368190823e-04	time = 0.05 sec
[ Info: VUMPS  14:	obj = -8.862377772951e-01	err = 8.9527621802e-04	time = 0.04 sec
[ Info: VUMPS  15:	obj = -8.862380175272e-01	err = 8.0363531590e-04	time = 0.05 sec
[ Info: VUMPS  16:	obj = -8.862381809910e-01	err = 8.1310379974e-04	time = 0.05 sec
[ Info: VUMPS  17:	obj = -8.862383289606e-01	err = 7.6372744918e-04	time = 0.04 sec
[ Info: VUMPS  18:	obj = -8.862384512228e-01	err = 7.7168418781e-04	time = 0.05 sec
[ Info: VUMPS  19:	obj = -8.862385939943e-01	err = 7.2169619957e-04	time = 0.05 sec
[ Info: VUMPS  20:	obj = -8.862387296086e-01	err = 6.9847750143e-04	time = 0.14 sec
[ Info: VUMPS  21:	obj = -8.862388823501e-01	err = 6.4519851323e-04	time = 0.08 sec
[ Info: VUMPS  22:	obj = -8.862390299430e-01	err = 5.9543430924e-04	time = 0.08 sec
[ Info: VUMPS  23:	obj = -8.862391796049e-01	err = 5.2965230331e-04	time = 0.08 sec
[ Info: VUMPS  24:	obj = -8.862393105628e-01	err = 4.6746973021e-04	time = 0.08 sec
[ Info: VUMPS  25:	obj = -8.862394256169e-01	err = 4.0012165330e-04	time = 0.08 sec
[ Info: VUMPS  26:	obj = -8.862395146213e-01	err = 3.4348612692e-04	time = 0.05 sec
[ Info: VUMPS  27:	obj = -8.862395831043e-01	err = 2.9133403018e-04	time = 0.05 sec
[ Info: VUMPS  28:	obj = -8.862396319943e-01	err = 2.5002069641e-04	time = 0.05 sec
[ Info: VUMPS  29:	obj = -8.862396671228e-01	err = 2.1555353820e-04	time = 0.05 sec
[ Info: VUMPS  30:	obj = -8.862396917938e-01	err = 1.8787657127e-04	time = 0.05 sec
[ Info: VUMPS  31:	obj = -8.862397096146e-01	err = 1.6535795978e-04	time = 0.05 sec
[ Info: VUMPS  32:	obj = -8.862397226287e-01	err = 1.4647959939e-04	time = 0.04 sec
[ Info: VUMPS  33:	obj = -8.862397325704e-01	err = 1.3058029797e-04	time = 0.04 sec
[ Info: VUMPS  34:	obj = -8.862397403210e-01	err = 1.1703644593e-04	time = 0.05 sec
[ Info: VUMPS  35:	obj = -8.862397466535e-01	err = 1.0479443543e-04	time = 0.05 sec
[ Info: VUMPS  36:	obj = -8.862397518883e-01	err = 9.4757697496e-05	time = 0.07 sec
[ Info: VUMPS  37:	obj = -8.862397563857e-01	err = 8.4793760071e-05	time = 0.05 sec
[ Info: VUMPS  38:	obj = -8.862397602492e-01	err = 7.7330476046e-05	time = 0.05 sec
[ Info: VUMPS  39:	obj = -8.862397636664e-01	err = 6.9007080884e-05	time = 0.05 sec
[ Info: VUMPS  40:	obj = -8.862397666688e-01	err = 6.3574783777e-05	time = 0.05 sec
[ Info: VUMPS  41:	obj = -8.862397693669e-01	err = 5.6541518218e-05	time = 0.05 sec
[ Info: VUMPS  42:	obj = -8.862397717706e-01	err = 5.2754788593e-05	time = 0.05 sec
[ Info: VUMPS  43:	obj = -8.862397739510e-01	err = 4.6775683210e-05	time = 0.05 sec
[ Info: VUMPS  44:	obj = -8.862397759130e-01	err = 4.4320904414e-05	time = 0.05 sec
[ Info: VUMPS  45:	obj = -8.862397777047e-01	err = 3.9206640444e-05	time = 0.05 sec
[ Info: VUMPS  46:	obj = -8.862397793305e-01	err = 3.7808257666e-05	time = 0.05 sec
[ Info: VUMPS  47:	obj = -8.862397808237e-01	err = 3.3398078106e-05	time = 0.05 sec
[ Info: VUMPS  48:	obj = -8.862397821887e-01	err = 3.2810027183e-05	time = 0.05 sec
[ Info: VUMPS  49:	obj = -8.862397834489e-01	err = 2.8968297396e-05	time = 0.05 sec
[ Info: VUMPS  50:	obj = -8.862397846088e-01	err = 2.8975455638e-05	time = 0.07 sec
[ Info: VUMPS  51:	obj = -8.862397856847e-01	err = 2.5588978527e-05	time = 0.05 sec
[ Info: VUMPS  52:	obj = -8.862397866809e-01	err = 2.6012564811e-05	time = 0.05 sec
[ Info: VUMPS  53:	obj = -8.862397876092e-01	err = 2.3457901855e-05	time = 0.05 sec
[ Info: VUMPS  54:	obj = -8.862397884732e-01	err = 2.3689215433e-05	time = 0.05 sec
[ Info: VUMPS  55:	obj = -8.862397892815e-01	err = 2.1824090291e-05	time = 0.05 sec
[ Info: VUMPS  56:	obj = -8.862397900373e-01	err = 2.1828940191e-05	time = 0.05 sec
[ Info: VUMPS  57:	obj = -8.862397907469e-01	err = 2.0446827272e-05	time = 0.05 sec
[ Info: VUMPS  58:	obj = -8.862397914131e-01	err = 2.0302723176e-05	time = 0.05 sec
[ Info: VUMPS  59:	obj = -8.862397920405e-01	err = 1.9254981490e-05	time = 0.05 sec
[ Info: VUMPS  60:	obj = -8.862397926316e-01	err = 1.9019089435e-05	time = 0.05 sec
[ Info: VUMPS  61:	obj = -8.862397931898e-01	err = 1.8202179915e-05	time = 0.05 sec
[ Info: VUMPS  62:	obj = -8.862397937172e-01	err = 1.7914246472e-05	time = 0.05 sec
[ Info: VUMPS  63:	obj = -8.862397942165e-01	err = 1.7257569224e-05	time = 0.05 sec
[ Info: VUMPS  64:	obj = -8.862397946895e-01	err = 1.6944227459e-05	time = 0.05 sec
[ Info: VUMPS  65:	obj = -8.862397951382e-01	err = 1.6400243806e-05	time = 0.07 sec
[ Info: VUMPS  66:	obj = -8.862397955643e-01	err = 1.6078941938e-05	time = 0.05 sec
[ Info: VUMPS  67:	obj = -8.862397959693e-01	err = 1.5615639391e-05	time = 0.05 sec
[ Info: VUMPS  68:	obj = -8.862397963546e-01	err = 1.5297227747e-05	time = 0.05 sec
[ Info: VUMPS  69:	obj = -8.862397967215e-01	err = 1.4893178345e-05	time = 0.05 sec
[ Info: VUMPS  70:	obj = -8.862397970712e-01	err = 1.4584085106e-05	time = 0.05 sec
[ Info: VUMPS  71:	obj = -8.862397974047e-01	err = 1.4224856670e-05	time = 0.05 sec
[ Info: VUMPS  72:	obj = -8.862397977230e-01	err = 1.3928611175e-05	time = 0.05 sec
[ Info: VUMPS  73:	obj = -8.862397980271e-01	err = 1.3604370840e-05	time = 0.05 sec
[ Info: VUMPS  74:	obj = -8.862397983177e-01	err = 1.3322648086e-05	time = 0.05 sec
[ Info: VUMPS  75:	obj = -8.862397985958e-01	err = 1.3026579806e-05	time = 0.05 sec
[ Info: VUMPS  76:	obj = -8.862397988619e-01	err = 1.2759903586e-05	time = 0.05 sec
[ Info: VUMPS  77:	obj = -8.862397991167e-01	err = 1.2487175232e-05	time = 0.05 sec
[ Info: VUMPS  78:	obj = -8.862397993609e-01	err = 1.2235379085e-05	time = 0.05 sec
[ Info: VUMPS  79:	obj = -8.862397995950e-01	err = 1.1982470825e-05	time = 0.05 sec
[ Info: VUMPS  80:	obj = -8.862397998196e-01	err = 1.1744996531e-05	time = 0.07 sec
[ Info: VUMPS  81:	obj = -8.862398000351e-01	err = 1.1509266452e-05	time = 0.05 sec
[ Info: VUMPS  82:	obj = -8.862398002421e-01	err = 1.1285351075e-05	time = 0.05 sec
[ Info: VUMPS  83:	obj = -8.862398004410e-01	err = 1.1064753788e-05	time = 0.05 sec
[ Info: VUMPS  84:	obj = -8.862398006321e-01	err = 1.0853545836e-05	time = 0.05 sec
[ Info: VUMPS  85:	obj = -8.862398008159e-01	err = 1.0646449681e-05	time = 0.05 sec
[ Info: VUMPS  86:	obj = -8.862398009928e-01	err = 1.0447078851e-05	time = 0.05 sec
[ Info: VUMPS  87:	obj = -8.862398011630e-01	err = 1.0252146279e-05	time = 0.05 sec
[ Info: VUMPS  88:	obj = -8.862398013268e-01	err = 1.0063763685e-05	time = 0.05 sec
[ Info: VUMPS  89:	obj = -8.862398014847e-01	err = 9.8798722021e-06	time = 0.05 sec
[ Info: VUMPS  90:	obj = -8.862398016367e-01	err = 9.7016721333e-06	time = 0.04 sec
[ Info: VUMPS  91:	obj = -8.862398017834e-01	err = 9.5278612850e-06	time = 0.04 sec
[ Info: VUMPS  92:	obj = -8.862398019247e-01	err = 9.3590916174e-06	time = 0.04 sec
[ Info: VUMPS  93:	obj = -8.862398020611e-01	err = 9.1945267654e-06	time = 0.04 sec
[ Info: VUMPS  94:	obj = -8.862398021927e-01	err = 9.0344925557e-06	time = 0.05 sec
[ Info: VUMPS  95:	obj = -8.862398023197e-01	err = 8.8784395519e-06	time = 0.07 sec
[ Info: VUMPS  96:	obj = -8.862398024423e-01	err = 8.7265026811e-06	time = 0.05 sec
[ Info: VUMPS  97:	obj = -8.862398025608e-01	err = 8.5783096813e-06	time = 0.05 sec
[ Info: VUMPS  98:	obj = -8.862398026752e-01	err = 8.4338863245e-06	time = 0.05 sec
[ Info: VUMPS  99:	obj = -8.862398027858e-01	err = 8.2929703487e-06	time = 0.05 sec
┌ Warning: VUMPS cancel 100:	obj = -8.862398028927e-01	err = 8.1555273601e-06	time = 5.09 sec
└ @ MPSKit ~/git/MPSKit.jl/src/algorithms/groundstate/vumps.jl:73

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
  <clipPath id="clip210">
    <rect x="0" y="0" width="2400" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip210)" d="M0 1600 L2400 1600 L2400 0 L0 0  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip211">
    <rect x="480" y="0" width="1681" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip210)" d="M187.803 1352.62 L2352.76 1352.62 L2352.76 123.472 L187.803 123.472  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip212">
    <rect x="187" y="123" width="2166" height="1230"/>
  </clipPath>
</defs>
<polyline clip-path="url(#clip212)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="249.075,1352.62 249.075,123.472 "/>
<polyline clip-path="url(#clip212)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="187.803,769.01 2352.76,769.01 "/>
<polyline clip-path="url(#clip212)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="187.803,170.643 2352.76,170.643 "/>
<polyline clip-path="url(#clip210)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="187.803,1352.62 2352.76,1352.62 "/>
<polyline clip-path="url(#clip210)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="249.075,1352.62 249.075,1371.52 "/>
<path clip-path="url(#clip210)" d="M115.831 1508.55 L136.504 1487.88 L139.286 1490.66 L130.611 1499.34 L152.266 1520.99 L148.943 1524.32 L127.288 1502.66 L118.613 1511.34 L115.831 1508.55 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M154.181 1488.04 Q153.379 1488.26 152.643 1488.73 Q151.906 1489.17 151.186 1489.89 Q148.632 1492.45 148.927 1495.49 Q149.222 1498.5 152.332 1501.61 L161.989 1511.27 L158.961 1514.3 L140.628 1495.97 L143.657 1492.94 L146.505 1495.79 Q145.784 1493.17 146.505 1490.84 Q147.208 1488.5 149.385 1486.33 Q149.696 1486.01 150.122 1485.69 Q150.531 1485.34 151.071 1484.97 L154.181 1488.04 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M154.525 1482.07 L157.537 1479.06 L175.869 1497.39 L172.857 1500.4 L154.525 1482.07 M147.388 1474.93 L150.4 1471.92 L154.214 1475.74 L151.202 1478.75 L147.388 1474.93 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M161.678 1474.92 L164.87 1471.73 L185.985 1481.38 L176.327 1460.27 L179.519 1457.08 L190.977 1482.28 L186.885 1486.37 L161.678 1474.92 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M183.677 1452.92 L186.688 1449.91 L205.021 1468.24 L202.009 1471.25 L183.677 1452.92 M176.54 1445.78 L179.552 1442.77 L183.366 1446.58 L180.354 1449.6 L176.54 1445.78 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M210.439 1444.39 Q206.788 1448.04 206.216 1450.28 Q205.643 1452.53 207.656 1454.54 Q209.26 1456.14 211.257 1456.04 Q213.254 1455.91 215.071 1454.1 Q217.575 1451.59 217.313 1448.32 Q217.051 1445.01 214.105 1442.07 L213.434 1441.4 L210.439 1444.39 M215.202 1437.14 L225.661 1447.6 L222.649 1450.61 L219.867 1447.83 Q220.505 1450.53 219.768 1452.87 Q219.015 1455.19 216.789 1457.42 Q213.974 1460.23 210.733 1460.33 Q207.492 1460.4 204.841 1457.75 Q201.747 1454.65 202.238 1451.02 Q202.745 1447.37 206.854 1443.26 L211.077 1439.04 L210.782 1438.74 Q208.703 1436.66 206.199 1436.91 Q203.695 1437.12 201.223 1439.59 Q199.652 1441.17 198.539 1443.03 Q197.426 1444.9 196.804 1447.03 L194.021 1444.24 Q195.036 1441.95 196.28 1440.09 Q197.508 1438.2 198.981 1436.73 Q202.958 1432.75 206.985 1432.85 Q211.011 1432.95 215.202 1437.14 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M206.396 1415.93 L209.407 1412.91 L234.876 1438.38 L231.864 1441.4 L206.396 1415.93 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M222.976 1399.41 Q224.548 1405.37 227.167 1410.12 Q229.786 1414.86 233.567 1418.64 Q237.348 1422.42 242.127 1425.08 Q246.907 1427.7 252.832 1429.27 L250.213 1431.89 Q243.911 1430.49 238.968 1428.01 Q234.041 1425.5 230.375 1421.84 Q226.725 1418.19 224.237 1413.27 Q221.749 1408.36 220.358 1402.03 L222.976 1399.41 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M228.345 1394.04 L230.964 1391.42 Q237.282 1392.83 242.193 1395.32 Q247.119 1397.79 250.77 1401.44 Q254.436 1405.11 256.924 1410.05 Q259.428 1414.98 260.82 1421.28 L258.201 1423.9 Q256.629 1417.97 253.994 1413.21 Q251.359 1408.41 247.578 1404.63 Q243.797 1400.85 239.034 1398.25 Q234.287 1395.63 228.345 1394.04 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M1186.48 1611.28 L1181.73 1599.09 L1171.96 1616.92 L1165.05 1616.92 L1178.87 1591.71 L1173.08 1576.72 Q1171.52 1572.71 1166.61 1572.71 L1165.05 1572.71 L1165.05 1567.68 L1167.28 1567.74 Q1175.49 1567.96 1177.56 1573.28 L1182.27 1585.47 L1192.05 1567.64 L1198.95 1567.64 L1185.14 1592.85 L1190.93 1607.84 Q1192.49 1611.85 1197.39 1611.85 L1198.95 1611.85 L1198.95 1616.88 L1196.72 1616.82 Q1188.51 1616.6 1186.48 1611.28 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M1228.46 1573.72 L1269.26 1573.72 L1269.26 1579.07 L1228.46 1579.07 L1228.46 1573.72 M1228.46 1586.71 L1269.26 1586.71 L1269.26 1592.12 L1228.46 1592.12 L1228.46 1586.71 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M1303.92 1555.8 L1329.16 1555.8 L1329.16 1561.22 L1309.81 1561.22 L1309.81 1572.86 Q1311.21 1572.39 1312.61 1572.16 Q1314.01 1571.91 1315.41 1571.91 Q1323.37 1571.91 1328.02 1576.27 Q1332.66 1580.63 1332.66 1588.08 Q1332.66 1595.75 1327.89 1600.01 Q1323.11 1604.25 1314.43 1604.25 Q1311.43 1604.25 1308.31 1603.74 Q1305.23 1603.23 1301.92 1602.21 L1301.92 1595.75 Q1304.78 1597.31 1307.84 1598.07 Q1310.89 1598.84 1314.3 1598.84 Q1319.8 1598.84 1323.02 1595.94 Q1326.23 1593.04 1326.23 1588.08 Q1326.23 1583.11 1323.02 1580.22 Q1319.8 1577.32 1314.3 1577.32 Q1311.72 1577.32 1309.14 1577.89 Q1306.6 1578.47 1303.92 1579.68 L1303.92 1555.8 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M1359.08 1560.04 Q1354.12 1560.04 1351.6 1564.94 Q1349.12 1569.81 1349.12 1579.61 Q1349.12 1589.38 1351.6 1594.29 Q1354.12 1599.16 1359.08 1599.16 Q1364.08 1599.16 1366.56 1594.29 Q1369.08 1589.38 1369.08 1579.61 Q1369.08 1569.81 1366.56 1564.94 Q1364.08 1560.04 1359.08 1560.04 M1359.08 1554.95 Q1367.07 1554.95 1371.27 1561.28 Q1375.5 1567.58 1375.5 1579.61 Q1375.5 1591.61 1371.27 1597.95 Q1367.07 1604.25 1359.08 1604.25 Q1351.09 1604.25 1346.86 1597.95 Q1342.66 1591.61 1342.66 1579.61 Q1342.66 1567.58 1346.86 1561.28 Q1351.09 1554.95 1359.08 1554.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><polyline clip-path="url(#clip210)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="187.803,1352.62 187.803,123.472 "/>
<polyline clip-path="url(#clip210)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="187.803,769.01 206.701,769.01 "/>
<polyline clip-path="url(#clip210)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="187.803,170.643 206.701,170.643 "/>
<path clip-path="url(#clip210)" d="M51.6634 788.802 L59.3023 788.802 L59.3023 762.437 L50.9921 764.103 L50.9921 759.844 L59.256 758.178 L63.9319 758.178 L63.9319 788.802 L71.5707 788.802 L71.5707 792.738 L51.6634 792.738 L51.6634 788.802 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M91.0151 761.256 Q87.404 761.256 85.5753 764.821 Q83.7697 768.363 83.7697 775.492 Q83.7697 782.599 85.5753 786.164 Q87.404 789.705 91.0151 789.705 Q94.6493 789.705 96.4548 786.164 Q98.2835 782.599 98.2835 775.492 Q98.2835 768.363 96.4548 764.821 Q94.6493 761.256 91.0151 761.256 M91.0151 757.553 Q96.8252 757.553 99.8808 762.159 Q102.959 766.742 102.959 775.492 Q102.959 784.219 99.8808 788.826 Q96.8252 793.409 91.0151 793.409 Q85.2049 793.409 82.1262 788.826 Q79.0707 784.219 79.0707 775.492 Q79.0707 766.742 82.1262 762.159 Q85.2049 757.553 91.0151 757.553 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M102.959 751.654 L127.071 751.654 L127.071 754.851 L102.959 754.851 L102.959 751.654 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M138.544 762.13 L151.803 762.13 L151.803 765.327 L133.973 765.327 L133.973 762.13 Q136.136 759.892 139.86 756.13 Q143.603 752.35 144.562 751.259 Q146.387 749.209 147.101 747.798 Q147.835 746.369 147.835 744.996 Q147.835 742.758 146.255 741.347 Q144.694 739.937 142.174 739.937 Q140.387 739.937 138.393 740.557 Q136.418 741.178 134.162 742.438 L134.162 738.601 Q136.456 737.68 138.45 737.21 Q140.443 736.739 142.098 736.739 Q146.462 736.739 149.057 738.921 Q151.653 741.103 151.653 744.751 Q151.653 746.482 150.994 748.043 Q150.355 749.585 148.644 751.692 Q148.173 752.237 145.653 754.851 Q143.133 757.447 138.544 762.13 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M81.0976 190.436 L88.7364 190.436 L88.7364 164.07 L80.4263 165.737 L80.4263 161.478 L88.6901 159.811 L93.366 159.811 L93.366 190.436 L101.005 190.436 L101.005 194.371 L81.0976 194.371 L81.0976 190.436 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M120.449 162.89 Q116.838 162.89 115.009 166.454 Q113.204 169.996 113.204 177.126 Q113.204 184.232 115.009 187.797 Q116.838 191.338 120.449 191.338 Q124.083 191.338 125.889 187.797 Q127.718 184.232 127.718 177.126 Q127.718 169.996 125.889 166.454 Q124.083 162.89 120.449 162.89 M120.449 159.186 Q126.259 159.186 129.315 163.792 Q132.394 168.376 132.394 177.126 Q132.394 185.852 129.315 190.459 Q126.259 195.042 120.449 195.042 Q114.639 195.042 111.56 190.459 Q108.505 185.852 108.505 177.126 Q108.505 168.376 111.56 163.792 Q114.639 159.186 120.449 159.186 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M142.098 141.382 Q139.164 141.382 137.679 144.278 Q136.212 147.156 136.212 152.949 Q136.212 158.723 137.679 161.619 Q139.164 164.497 142.098 164.497 Q145.051 164.497 146.518 161.619 Q148.004 158.723 148.004 152.949 Q148.004 147.156 146.518 144.278 Q145.051 141.382 142.098 141.382 M142.098 138.373 Q146.819 138.373 149.302 142.115 Q151.803 145.839 151.803 152.949 Q151.803 160.039 149.302 163.782 Q146.819 167.506 142.098 167.506 Q137.378 167.506 134.876 163.782 Q132.394 160.039 132.394 152.949 Q132.394 145.839 134.876 142.115 Q137.378 138.373 142.098 138.373 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M771.35 12.096 L809.59 12.096 L809.59 18.9825 L779.533 18.9825 L779.533 36.8875 L808.335 36.8875 L808.335 43.7741 L779.533 43.7741 L779.533 65.6895 L810.32 65.6895 L810.32 72.576 L771.35 72.576 L771.35 12.096 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M861.158 45.1919 L861.158 72.576 L853.705 72.576 L853.705 45.4349 Q853.705 38.994 851.193 35.7938 Q848.682 32.5936 843.659 32.5936 Q837.623 32.5936 834.139 36.4419 Q830.655 40.2903 830.655 46.9338 L830.655 72.576 L823.161 72.576 L823.161 27.2059 L830.655 27.2059 L830.655 34.2544 Q833.329 30.163 836.934 28.1376 Q840.58 26.1121 845.319 26.1121 Q853.138 26.1121 857.148 30.9732 Q861.158 35.7938 861.158 45.1919 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M883.398 14.324 L883.398 27.2059 L898.751 27.2059 L898.751 32.9987 L883.398 32.9987 L883.398 57.6282 Q883.398 63.1779 884.897 64.7578 Q886.436 66.3376 891.095 66.3376 L898.751 66.3376 L898.751 72.576 L891.095 72.576 Q882.466 72.576 879.185 69.3758 Q875.904 66.1351 875.904 57.6282 L875.904 32.9987 L870.435 32.9987 L870.435 27.2059 L875.904 27.2059 L875.904 14.324 L883.398 14.324 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M929.173 49.7694 Q920.14 49.7694 916.656 51.8354 Q913.172 53.9013 913.172 58.8839 Q913.172 62.8538 915.765 65.2034 Q918.398 67.5124 922.894 67.5124 Q929.092 67.5124 932.819 63.1374 Q936.586 58.7219 936.586 51.4303 L936.586 49.7694 L929.173 49.7694 M944.04 46.6907 L944.04 72.576 L936.586 72.576 L936.586 65.6895 Q934.034 69.8214 930.226 71.8063 Q926.419 73.7508 920.909 73.7508 Q913.942 73.7508 909.81 69.8619 Q905.718 65.9325 905.718 59.3701 Q905.718 51.7138 910.823 47.825 Q915.967 43.9361 926.135 43.9361 L936.586 43.9361 L936.586 43.2069 Q936.586 38.0623 933.184 35.2672 Q929.821 32.4315 923.704 32.4315 Q919.816 32.4315 916.129 33.3632 Q912.443 34.295 909.04 36.1584 L909.04 29.2718 Q913.132 27.692 916.98 26.9223 Q920.828 26.1121 924.474 26.1121 Q934.318 26.1121 939.179 31.2163 Q944.04 36.3204 944.04 46.6907 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M997.107 45.1919 L997.107 72.576 L989.653 72.576 L989.653 45.4349 Q989.653 38.994 987.142 35.7938 Q984.63 32.5936 979.607 32.5936 Q973.571 32.5936 970.087 36.4419 Q966.604 40.2903 966.604 46.9338 L966.604 72.576 L959.109 72.576 L959.109 27.2059 L966.604 27.2059 L966.604 34.2544 Q969.277 30.163 972.882 28.1376 Q976.528 26.1121 981.268 26.1121 Q989.086 26.1121 993.096 30.9732 Q997.107 35.7938 997.107 45.1919 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M1041.83 49.3643 Q1041.83 41.2625 1038.47 36.8065 Q1035.14 32.3505 1029.11 32.3505 Q1023.11 32.3505 1019.75 36.8065 Q1016.43 41.2625 1016.43 49.3643 Q1016.43 57.4256 1019.75 61.8816 Q1023.11 66.3376 1029.11 66.3376 Q1035.14 66.3376 1038.47 61.8816 Q1041.83 57.4256 1041.83 49.3643 M1049.28 66.9452 Q1049.28 78.5308 1044.14 84.1616 Q1038.99 89.8329 1028.38 89.8329 Q1024.45 89.8329 1020.97 89.2252 Q1017.48 88.6581 1014.2 87.4428 L1014.2 80.1917 Q1017.48 81.9741 1020.68 82.8248 Q1023.88 83.6755 1027.21 83.6755 Q1034.54 83.6755 1038.18 79.8271 Q1041.83 76.0193 1041.83 68.282 L1041.83 64.5957 Q1039.52 68.6061 1035.91 70.5911 Q1032.31 72.576 1027.29 72.576 Q1018.94 72.576 1013.84 66.2161 Q1008.73 59.8562 1008.73 49.3643 Q1008.73 38.832 1013.84 32.472 Q1018.94 26.1121 1027.29 26.1121 Q1032.31 26.1121 1035.91 28.0971 Q1039.52 30.082 1041.83 34.0924 L1041.83 27.2059 L1049.28 27.2059 L1049.28 66.9452 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M1064.64 9.54393 L1072.09 9.54393 L1072.09 72.576 L1064.64 72.576 L1064.64 9.54393 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M1126.49 48.0275 L1126.49 51.6733 L1092.22 51.6733 Q1092.71 59.3701 1096.84 63.421 Q1101.01 67.4314 1108.43 67.4314 Q1112.72 67.4314 1116.73 66.3781 Q1120.78 65.3249 1124.75 63.2184 L1124.75 70.267 Q1120.74 71.9684 1116.53 72.8596 Q1112.31 73.7508 1107.98 73.7508 Q1097.12 73.7508 1090.76 67.4314 Q1084.44 61.1119 1084.44 50.3365 Q1084.44 39.1965 1090.44 32.6746 Q1096.48 26.1121 1106.68 26.1121 Q1115.84 26.1121 1121.15 32.0264 Q1126.49 37.9003 1126.49 48.0275 M1119.04 45.84 Q1118.96 39.7232 1115.6 36.0774 Q1112.27 32.4315 1106.76 32.4315 Q1100.53 32.4315 1096.76 35.9558 Q1093.03 39.4801 1092.47 45.8805 L1119.04 45.84 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M1174.05 35.9153 Q1176.85 30.8922 1180.73 28.5022 Q1184.62 26.1121 1189.89 26.1121 Q1196.98 26.1121 1200.83 31.0947 Q1204.68 36.0368 1204.68 45.1919 L1204.68 72.576 L1197.18 72.576 L1197.18 45.4349 Q1197.18 38.913 1194.87 35.7533 Q1192.56 32.5936 1187.82 32.5936 Q1182.03 32.5936 1178.67 36.4419 Q1175.31 40.2903 1175.31 46.9338 L1175.31 72.576 L1167.81 72.576 L1167.81 45.4349 Q1167.81 38.8725 1165.5 35.7533 Q1163.19 32.5936 1158.37 32.5936 Q1152.66 32.5936 1149.3 36.4824 Q1145.94 40.3308 1145.94 46.9338 L1145.94 72.576 L1138.44 72.576 L1138.44 27.2059 L1145.94 27.2059 L1145.94 34.2544 Q1148.49 30.082 1152.05 28.0971 Q1155.62 26.1121 1160.52 26.1121 Q1165.46 26.1121 1168.91 28.6237 Q1172.39 31.1352 1174.05 35.9153 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M1258.35 48.0275 L1258.35 51.6733 L1224.08 51.6733 Q1224.57 59.3701 1228.7 63.421 Q1232.87 67.4314 1240.28 67.4314 Q1244.58 67.4314 1248.59 66.3781 Q1252.64 65.3249 1256.61 63.2184 L1256.61 70.267 Q1252.6 71.9684 1248.38 72.8596 Q1244.17 73.7508 1239.84 73.7508 Q1228.98 73.7508 1222.62 67.4314 Q1216.3 61.1119 1216.3 50.3365 Q1216.3 39.1965 1222.3 32.6746 Q1228.33 26.1121 1238.54 26.1121 Q1247.7 26.1121 1253 32.0264 Q1258.35 37.9003 1258.35 48.0275 M1250.9 45.84 Q1250.81 39.7232 1247.45 36.0774 Q1244.13 32.4315 1238.62 32.4315 Q1232.38 32.4315 1228.62 35.9558 Q1224.89 39.4801 1224.32 45.8805 L1250.9 45.84 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M1308.3 45.1919 L1308.3 72.576 L1300.84 72.576 L1300.84 45.4349 Q1300.84 38.994 1298.33 35.7938 Q1295.82 32.5936 1290.8 32.5936 Q1284.76 32.5936 1281.28 36.4419 Q1277.79 40.2903 1277.79 46.9338 L1277.79 72.576 L1270.3 72.576 L1270.3 27.2059 L1277.79 27.2059 L1277.79 34.2544 Q1280.47 30.163 1284.07 28.1376 Q1287.72 26.1121 1292.46 26.1121 Q1300.28 26.1121 1304.29 30.9732 Q1308.3 35.7938 1308.3 45.1919 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M1330.54 14.324 L1330.54 27.2059 L1345.89 27.2059 L1345.89 32.9987 L1330.54 32.9987 L1330.54 57.6282 Q1330.54 63.1779 1332.04 64.7578 Q1333.57 66.3376 1338.23 66.3376 L1345.89 66.3376 L1345.89 72.576 L1338.23 72.576 Q1329.61 72.576 1326.32 69.3758 Q1323.04 66.1351 1323.04 57.6282 L1323.04 32.9987 L1317.57 32.9987 L1317.57 27.2059 L1323.04 27.2059 L1323.04 14.324 L1330.54 14.324 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M1418.64 14.0809 L1418.64 22.0612 Q1413.99 19.8332 1409.85 18.7395 Q1405.72 17.6457 1401.87 17.6457 Q1395.19 17.6457 1391.54 20.2383 Q1387.94 22.8309 1387.94 27.611 Q1387.94 31.6214 1390.33 33.6873 Q1392.76 35.7128 1399.48 36.9686 L1404.43 37.9813 Q1413.58 39.7232 1417.91 44.1387 Q1422.29 48.5136 1422.29 55.8863 Q1422.29 64.6767 1416.38 69.2137 Q1410.5 73.7508 1399.12 73.7508 Q1394.82 73.7508 1389.96 72.7785 Q1385.14 71.8063 1379.96 69.9024 L1379.96 61.4765 Q1384.94 64.2716 1389.72 65.6895 Q1394.5 67.1073 1399.12 67.1073 Q1406.13 67.1073 1409.93 64.3527 Q1413.74 61.598 1413.74 56.4939 Q1413.74 52.0379 1410.99 49.5264 Q1408.27 47.0148 1402.04 45.759 L1397.05 44.7868 Q1387.9 42.9639 1383.81 39.075 Q1379.71 35.1862 1379.71 28.2591 Q1379.71 20.2383 1385.35 15.6203 Q1391.02 11.0023 1400.94 11.0023 Q1405.19 11.0023 1409.61 11.7719 Q1414.03 12.5416 1418.64 14.0809 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M1441.94 65.7705 L1441.94 89.8329 L1434.44 89.8329 L1434.44 27.2059 L1441.94 27.2059 L1441.94 34.0924 Q1444.29 30.0415 1447.85 28.0971 Q1451.46 26.1121 1456.44 26.1121 Q1464.7 26.1121 1469.85 32.6746 Q1475.03 39.2371 1475.03 49.9314 Q1475.03 60.6258 1469.85 67.1883 Q1464.7 73.7508 1456.44 73.7508 Q1451.46 73.7508 1447.85 71.8063 Q1444.29 69.8214 1441.94 65.7705 M1467.3 49.9314 Q1467.3 41.7081 1463.89 37.0496 Q1460.53 32.3505 1454.62 32.3505 Q1448.7 32.3505 1445.3 37.0496 Q1441.94 41.7081 1441.94 49.9314 Q1441.94 58.1548 1445.3 62.8538 Q1448.7 67.5124 1454.62 67.5124 Q1460.53 67.5124 1463.89 62.8538 Q1467.3 58.1548 1467.3 49.9314 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M1526.2 48.0275 L1526.2 51.6733 L1491.92 51.6733 Q1492.41 59.3701 1496.54 63.421 Q1500.72 67.4314 1508.13 67.4314 Q1512.42 67.4314 1516.43 66.3781 Q1520.48 65.3249 1524.45 63.2184 L1524.45 70.267 Q1520.44 71.9684 1516.23 72.8596 Q1512.02 73.7508 1507.68 73.7508 Q1496.83 73.7508 1490.47 67.4314 Q1484.15 61.1119 1484.15 50.3365 Q1484.15 39.1965 1490.14 32.6746 Q1496.18 26.1121 1506.39 26.1121 Q1515.54 26.1121 1520.85 32.0264 Q1526.2 37.9003 1526.2 48.0275 M1518.74 45.84 Q1518.66 39.7232 1515.3 36.0774 Q1511.98 32.4315 1506.47 32.4315 Q1500.23 32.4315 1496.46 35.9558 Q1492.73 39.4801 1492.17 45.8805 L1518.74 45.84 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M1571.08 28.9478 L1571.08 35.9153 Q1567.92 34.1734 1564.72 33.3227 Q1561.56 32.4315 1558.32 32.4315 Q1551.07 32.4315 1547.06 37.0496 Q1543.05 41.6271 1543.05 49.9314 Q1543.05 58.2358 1547.06 62.8538 Q1551.07 67.4314 1558.32 67.4314 Q1561.56 67.4314 1564.72 66.5807 Q1567.92 65.6895 1571.08 63.9476 L1571.08 70.8341 Q1567.96 72.2924 1564.6 73.0216 Q1561.28 73.7508 1557.51 73.7508 Q1547.26 73.7508 1541.22 67.3098 Q1535.19 60.8689 1535.19 49.9314 Q1535.19 38.832 1541.26 32.472 Q1547.38 26.1121 1558 26.1121 Q1561.44 26.1121 1564.72 26.8413 Q1568 27.5299 1571.08 28.9478 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M1591.41 14.324 L1591.41 27.2059 L1606.77 27.2059 L1606.77 32.9987 L1591.41 32.9987 L1591.41 57.6282 Q1591.41 63.1779 1592.91 64.7578 Q1594.45 66.3376 1599.11 66.3376 L1606.77 66.3376 L1606.77 72.576 L1599.11 72.576 Q1590.48 72.576 1587.2 69.3758 Q1583.92 66.1351 1583.92 57.6282 L1583.92 32.9987 L1578.45 32.9987 L1578.45 27.2059 L1583.92 27.2059 L1583.92 14.324 L1591.41 14.324 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M1642.86 34.1734 Q1641.61 33.4443 1640.11 33.1202 Q1638.65 32.7556 1636.87 32.7556 Q1630.55 32.7556 1627.14 36.8875 Q1623.78 40.9789 1623.78 48.6757 L1623.78 72.576 L1616.29 72.576 L1616.29 27.2059 L1623.78 27.2059 L1623.78 34.2544 Q1626.13 30.1225 1629.9 28.1376 Q1633.67 26.1121 1639.05 26.1121 Q1639.82 26.1121 1640.76 26.2337 Q1641.69 26.3147 1642.82 26.5172 L1642.86 34.1734 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M1649.91 54.671 L1649.91 27.2059 L1657.36 27.2059 L1657.36 54.3874 Q1657.36 60.8284 1659.88 64.0691 Q1662.39 67.2693 1667.41 67.2693 Q1673.45 67.2693 1676.93 63.421 Q1680.45 59.5726 1680.45 52.9291 L1680.45 27.2059 L1687.91 27.2059 L1687.91 72.576 L1680.45 72.576 L1680.45 65.6084 Q1677.74 69.7404 1674.13 71.7658 Q1670.57 73.7508 1665.83 73.7508 Q1658.01 73.7508 1653.96 68.8897 Q1649.91 64.0286 1649.91 54.671 M1668.67 26.1121 L1668.67 26.1121 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip210)" d="M1738.58 35.9153 Q1741.38 30.8922 1745.27 28.5022 Q1749.16 26.1121 1754.42 26.1121 Q1761.51 26.1121 1765.36 31.0947 Q1769.21 36.0368 1769.21 45.1919 L1769.21 72.576 L1761.72 72.576 L1761.72 45.4349 Q1761.72 38.913 1759.41 35.7533 Q1757.1 32.5936 1752.36 32.5936 Q1746.56 32.5936 1743.2 36.4419 Q1739.84 40.2903 1739.84 46.9338 L1739.84 72.576 L1732.35 72.576 L1732.35 45.4349 Q1732.35 38.8725 1730.04 35.7533 Q1727.73 32.5936 1722.91 32.5936 Q1717.2 32.5936 1713.83 36.4824 Q1710.47 40.3308 1710.47 46.9338 L1710.47 72.576 L1702.98 72.576 L1702.98 27.2059 L1710.47 27.2059 L1710.47 34.2544 Q1713.02 30.082 1716.59 28.0971 Q1720.15 26.1121 1725.05 26.1121 Q1730 26.1121 1733.44 28.6237 Q1736.92 31.1352 1738.58 35.9153 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><circle clip-path="url(#clip212)" cx="453.316" cy="195.062" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="486.662" cy="307.133" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="520.007" cy="323.77" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="553.353" cy="338.335" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="586.698" cy="499.025" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="620.043" cy="516.442" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="653.389" cy="530.653" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="686.734" cy="544.601" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="720.08" cy="661.078" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="753.425" cy="678.286" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="786.771" cy="693.498" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="820.116" cy="738.198" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="853.462" cy="742.721" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="886.807" cy="756.422" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="920.152" cy="762.41" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="953.498" cy="771.737" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="986.843" cy="778.894" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="1020.19" cy="783.14" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="1053.53" cy="793.206" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="1086.88" cy="805.829" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="1120.23" cy="909.068" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="1153.57" cy="927.714" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="1186.92" cy="942.981" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="1220.26" cy="971.679" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="1253.61" cy="998.31" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="1286.95" cy="1019.26" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="1320.3" cy="1036.23" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="1353.64" cy="1050.75" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="1386.99" cy="1056.67" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="1420.33" cy="1062.83" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="1453.68" cy="1075.61" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="1487.02" cy="1087.38" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="1520.37" cy="1091.09" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="1553.72" cy="1103.06" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="1587.06" cy="1105.16" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="1620.41" cy="1118.96" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="1653.75" cy="1120.09" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="1687.1" cy="1139.32" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="1720.44" cy="1154.99" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="1753.79" cy="1207.6" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="1787.13" cy="1227.38" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="1820.48" cy="1244.85" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="1853.82" cy="1256.7" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="1887.17" cy="1261.91" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="1920.52" cy="1274.04" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="1953.86" cy="1278.38" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="1987.21" cy="1292.67" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="2020.55" cy="1297.26" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="2053.9" cy="1309.12" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip212)" cx="2087.24" cy="1317.83" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
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
└ @ MPSKit ~/git/MPSKit.jl/src/states/infinitemps.jl:149

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
[ Info: VUMPS init:	obj = +2.876758033543e-02	err = 4.0644e-01
[ Info: VUMPS   1:	obj = -8.737906665472e-01	err = 1.2045339353e-01	time = 0.02 sec
[ Info: VUMPS   2:	obj = -8.857246450469e-01	err = 7.4507510196e-03	time = 0.02 sec
[ Info: VUMPS   3:	obj = -8.861168843924e-01	err = 3.8806594310e-03	time = 0.02 sec
[ Info: VUMPS   4:	obj = -8.862224229828e-01	err = 1.8866990312e-03	time = 0.02 sec
[ Info: VUMPS   5:	obj = -8.862604919935e-01	err = 9.5632297117e-04	time = 0.03 sec
[ Info: VUMPS   6:	obj = -8.862756622261e-01	err = 6.9689351881e-04	time = 0.03 sec
[ Info: VUMPS   7:	obj = -8.862821568604e-01	err = 5.4384343431e-04	time = 0.03 sec
[ Info: VUMPS   8:	obj = -8.862851144799e-01	err = 4.3498937171e-04	time = 0.03 sec
[ Info: VUMPS   9:	obj = -8.862865208255e-01	err = 3.4022284480e-04	time = 0.04 sec
[ Info: VUMPS  10:	obj = -8.862872092893e-01	err = 2.6155639792e-04	time = 0.04 sec
[ Info: VUMPS  11:	obj = -8.862875517233e-01	err = 1.9816403910e-04	time = 0.04 sec
[ Info: VUMPS  12:	obj = -8.862877233984e-01	err = 1.4833091166e-04	time = 0.04 sec
[ Info: VUMPS  13:	obj = -8.862878098491e-01	err = 1.1002989377e-04	time = 0.04 sec
[ Info: VUMPS  14:	obj = -8.862878534672e-01	err = 8.1031399491e-05	time = 0.04 sec
[ Info: VUMPS  15:	obj = -8.862878754996e-01	err = 5.9348893102e-05	time = 0.04 sec
[ Info: VUMPS  16:	obj = -8.862878866404e-01	err = 4.3288174232e-05	time = 0.05 sec
[ Info: VUMPS  17:	obj = -8.862878922801e-01	err = 3.1476890398e-05	time = 0.05 sec
[ Info: VUMPS  18:	obj = -8.862878951379e-01	err = 2.2833727293e-05	time = 0.04 sec
[ Info: VUMPS  19:	obj = -8.862878965878e-01	err = 1.6534345712e-05	time = 0.04 sec
[ Info: VUMPS  20:	obj = -8.862878973242e-01	err = 1.1956033801e-05	time = 0.04 sec
[ Info: VUMPS  21:	obj = -8.862878976987e-01	err = 8.6354344118e-06	time = 0.04 sec
[ Info: VUMPS  22:	obj = -8.862878978893e-01	err = 6.2316368830e-06	time = 0.14 sec
[ Info: VUMPS  23:	obj = -8.862878979864e-01	err = 4.4936852018e-06	time = 0.04 sec
[ Info: VUMPS  24:	obj = -8.862878980359e-01	err = 3.2380407694e-06	time = 0.04 sec
[ Info: VUMPS  25:	obj = -8.862878980612e-01	err = 2.3317908791e-06	time = 0.05 sec
[ Info: VUMPS  26:	obj = -8.862878980741e-01	err = 1.6782796290e-06	time = 0.04 sec
[ Info: VUMPS  27:	obj = -8.862878980807e-01	err = 1.2073259818e-06	time = 0.04 sec
[ Info: VUMPS  28:	obj = -8.862878980841e-01	err = 8.6818060956e-07	time = 0.04 sec
[ Info: VUMPS  29:	obj = -8.862878980858e-01	err = 6.2400887691e-07	time = 0.04 sec
[ Info: VUMPS  30:	obj = -8.862878980867e-01	err = 4.4833124713e-07	time = 0.04 sec
[ Info: VUMPS  31:	obj = -8.862878980871e-01	err = 3.2198399757e-07	time = 0.04 sec
[ Info: VUMPS  32:	obj = -8.862878980874e-01	err = 2.3117466228e-07	time = 0.04 sec
[ Info: VUMPS  33:	obj = -8.862878980875e-01	err = 1.6592578229e-07	time = 0.04 sec
[ Info: VUMPS  34:	obj = -8.862878980876e-01	err = 1.1905960048e-07	time = 0.04 sec
[ Info: VUMPS  35:	obj = -8.862878980876e-01	err = 8.5408410847e-08	time = 0.04 sec
[ Info: VUMPS  36:	obj = -8.862878980876e-01	err = 6.1253530320e-08	time = 0.04 sec
[ Info: VUMPS  37:	obj = -8.862878980876e-01	err = 4.3920166462e-08	time = 0.04 sec
[ Info: VUMPS  38:	obj = -8.862878980877e-01	err = 3.1485556009e-08	time = 0.04 sec
[ Info: VUMPS  39:	obj = -8.862878980877e-01	err = 2.2566832411e-08	time = 0.04 sec
[ Info: VUMPS  40:	obj = -8.862878980877e-01	err = 1.6171599796e-08	time = 0.04 sec
[ Info: VUMPS  41:	obj = -8.862878980877e-01	err = 1.1586823278e-08	time = 0.04 sec
[ Info: VUMPS  42:	obj = -8.862878980877e-01	err = 8.3006991909e-09	time = 0.04 sec
[ Info: VUMPS  43:	obj = -8.862878980877e-01	err = 5.9456786414e-09	time = 0.04 sec
[ Info: VUMPS  44:	obj = -8.862878980877e-01	err = 4.2582593636e-09	time = 0.10 sec
[ Info: VUMPS  45:	obj = -8.862878980877e-01	err = 3.0493882526e-09	time = 0.04 sec
[ Info: VUMPS  46:	obj = -8.862878980877e-01	err = 2.1834676831e-09	time = 0.04 sec
[ Info: VUMPS  47:	obj = -8.862878980877e-01	err = 1.5632717223e-09	time = 0.04 sec
[ Info: VUMPS  48:	obj = -8.862878980877e-01	err = 1.1191534669e-09	time = 0.04 sec
[ Info: VUMPS  49:	obj = -8.862878980877e-01	err = 8.0114053028e-10	time = 0.04 sec
[ Info: VUMPS  50:	obj = -8.862878980877e-01	err = 5.7344926629e-10	time = 0.04 sec
[ Info: VUMPS  51:	obj = -8.862878980877e-01	err = 4.1044557276e-10	time = 0.04 sec
[ Info: VUMPS  52:	obj = -8.862878980877e-01	err = 2.9375533030e-10	time = 0.04 sec
[ Info: VUMPS  53:	obj = -8.862878980877e-01	err = 2.1022890604e-10	time = 0.04 sec
[ Info: VUMPS  54:	obj = -8.862878980877e-01	err = 1.5044585215e-10	time = 0.04 sec
[ Info: VUMPS  55:	obj = -8.862878980877e-01	err = 1.0765471631e-10	time = 0.04 sec
[ Info: VUMPS  56:	obj = -8.862878980878e-01	err = 7.7031624881e-11	time = 0.04 sec
[ Info: VUMPS  57:	obj = -8.862878980878e-01	err = 5.5116540056e-11	time = 0.04 sec
[ Info: VUMPS  58:	obj = -8.862878980878e-01	err = 3.9433877868e-11	time = 0.04 sec
[ Info: VUMPS  59:	obj = -8.862878980878e-01	err = 2.8212244668e-11	time = 0.04 sec
[ Info: VUMPS  60:	obj = -8.862878980878e-01	err = 2.0184013518e-11	time = 0.03 sec
[ Info: VUMPS  61:	obj = -8.862878980878e-01	err = 1.4441829469e-11	time = 0.03 sec
[ Info: VUMPS  62:	obj = -8.862878980878e-01	err = 1.0334072641e-11	time = 0.03 sec
[ Info: VUMPS  63:	obj = -8.862878980878e-01	err = 7.3919424370e-12	time = 0.03 sec
[ Info: VUMPS  64:	obj = -8.862878980878e-01	err = 5.2875140115e-12	time = 0.03 sec
[ Info: VUMPS  65:	obj = -8.862878980878e-01	err = 3.7817587214e-12	time = 0.03 sec
[ Info: VUMPS  66:	obj = -8.862878980878e-01	err = 2.7051025623e-12	time = 0.03 sec
[ Info: VUMPS  67:	obj = -8.862878980878e-01	err = 1.9384032618e-12	time = 0.03 sec
[ Info: VUMPS  68:	obj = -8.862878980878e-01	err = 1.3868042929e-12	time = 0.06 sec
[ Info: VUMPS conv 69:	obj = -8.862878980878e-01	err = 9.8790427735e-13	time = 2.76 sec

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

