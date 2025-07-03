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
[ Info: VUMPS init:	obj = +2.499983871472e-01	err = 3.7422e-03
[ Info: VUMPS   1:	obj = -2.063525174233e-01	err = 3.5574923452e-01	time = 0.06 sec
[ Info: VUMPS   2:	obj = -1.966172141461e-01	err = 3.8079422097e-01	time = 0.05 sec
[ Info: VUMPS   3:	obj = -3.358833718314e-01	err = 3.2023421118e-01	time = 0.07 sec
[ Info: VUMPS   4:	obj = -3.653869064234e-01	err = 2.8638216456e-01	time = 0.09 sec
[ Info: VUMPS   5:	obj = -3.253694830622e-01	err = 3.5289659250e-01	time = 0.10 sec
[ Info: VUMPS   6:	obj = -2.547497274822e-01	err = 3.5808845978e-01	time = 0.11 sec
[ Info: VUMPS   7:	obj = -3.215592920047e-01	err = 3.4308865334e-01	time = 0.12 sec
[ Info: VUMPS   8:	obj = -4.049670254998e-01	err = 2.4203176136e-01	time = 0.14 sec
[ Info: VUMPS   9:	obj = +1.442123980648e-01	err = 3.6271688056e-01	time = 0.14 sec
[ Info: VUMPS  10:	obj = -2.053482032064e-01	err = 3.8117603562e-01	time = 0.09 sec
[ Info: VUMPS  11:	obj = -1.326260701174e-01	err = 3.7311278975e-01	time = 0.09 sec
[ Info: VUMPS  12:	obj = -2.805313827027e-01	err = 3.5820197253e-01	time = 0.10 sec
[ Info: VUMPS  13:	obj = -3.173664554397e-01	err = 3.2887538514e-01	time = 0.12 sec
[ Info: VUMPS  14:	obj = +1.250313380044e-02	err = 3.8170064477e-01	time = 0.09 sec
[ Info: VUMPS  15:	obj = -8.048404492680e-02	err = 3.8469702174e-01	time = 0.09 sec
[ Info: VUMPS  16:	obj = -3.147066201922e-01	err = 3.2805873509e-01	time = 0.06 sec
[ Info: VUMPS  17:	obj = -3.153071316546e-01	err = 3.4288551181e-01	time = 0.10 sec
[ Info: VUMPS  18:	obj = -1.627914055133e-01	err = 4.0364659659e-01	time = 0.08 sec
[ Info: VUMPS  19:	obj = -3.375871052071e-01	err = 3.2374169350e-01	time = 0.09 sec
[ Info: VUMPS  20:	obj = +1.012893312898e-02	err = 3.7436527185e-01	time = 0.07 sec
[ Info: VUMPS  21:	obj = -1.298922118265e-01	err = 3.7987803101e-01	time = 0.09 sec
[ Info: VUMPS  22:	obj = -1.361147329184e-01	err = 3.6661142078e-01	time = 0.09 sec
[ Info: VUMPS  23:	obj = -1.988181151574e-01	err = 3.6829432389e-01	time = 0.07 sec
[ Info: VUMPS  24:	obj = -1.325161253829e-01	err = 3.9986524249e-01	time = 0.09 sec
[ Info: VUMPS  25:	obj = -2.754147162595e-01	err = 3.5752301171e-01	time = 0.34 sec
[ Info: VUMPS  26:	obj = +2.284136624012e-02	err = 3.8624440884e-01	time = 0.08 sec
[ Info: VUMPS  27:	obj = -1.533319473927e-01	err = 4.0668352809e-01	time = 0.09 sec
[ Info: VUMPS  28:	obj = -2.549098405406e-01	err = 3.6520371840e-01	time = 0.06 sec
[ Info: VUMPS  29:	obj = -2.008055328636e-01	err = 3.6284552902e-01	time = 0.07 sec
[ Info: VUMPS  30:	obj = -2.214812189661e-01	err = 3.7744648980e-01	time = 0.09 sec
[ Info: VUMPS  31:	obj = -3.212861478190e-01	err = 3.4156373138e-01	time = 0.07 sec
[ Info: VUMPS  32:	obj = -2.621703492836e-01	err = 3.6018890906e-01	time = 0.07 sec
[ Info: VUMPS  33:	obj = -2.629031725595e-02	err = 4.1677899736e-01	time = 0.07 sec
[ Info: VUMPS  34:	obj = -3.166290164494e-02	err = 3.9602981058e-01	time = 0.05 sec
[ Info: VUMPS  35:	obj = -1.201745763373e-01	err = 3.9443392382e-01	time = 0.04 sec
[ Info: VUMPS  36:	obj = -1.276330954584e-01	err = 3.6305111576e-01	time = 0.03 sec
[ Info: VUMPS  37:	obj = +2.700512484266e-02	err = 3.9135429720e-01	time = 0.03 sec
[ Info: VUMPS  38:	obj = -6.349042905323e-02	err = 3.9780333777e-01	time = 0.04 sec
[ Info: VUMPS  39:	obj = -7.068393596597e-02	err = 3.7819722580e-01	time = 0.03 sec
[ Info: VUMPS  40:	obj = -6.152794450144e-02	err = 4.0354490342e-01	time = 0.03 sec
[ Info: VUMPS  41:	obj = +3.792648835753e-02	err = 3.6290173916e-01	time = 0.03 sec
[ Info: VUMPS  42:	obj = -4.335289598436e-02	err = 3.6448095409e-01	time = 0.02 sec
[ Info: VUMPS  43:	obj = -3.466739695596e-01	err = 3.2124355212e-01	time = 0.04 sec
[ Info: VUMPS  44:	obj = -2.251752651856e-01	err = 3.7247588308e-01	time = 0.05 sec
[ Info: VUMPS  45:	obj = -2.484151824761e-01	err = 3.6983660293e-01	time = 0.04 sec
[ Info: VUMPS  46:	obj = -2.254973308254e-01	err = 3.7218385889e-01	time = 0.04 sec
[ Info: VUMPS  47:	obj = -2.498573440863e-01	err = 3.5618674683e-01	time = 0.03 sec
[ Info: VUMPS  48:	obj = -2.769333338940e-01	err = 3.6205687877e-01	time = 0.03 sec
[ Info: VUMPS  49:	obj = -1.748887576251e-01	err = 3.8612945301e-01	time = 0.04 sec
[ Info: VUMPS  50:	obj = -2.599778950095e-01	err = 3.9065516189e-01	time = 0.04 sec
[ Info: VUMPS  51:	obj = -6.973297200834e-02	err = 4.0598095480e-01	time = 0.03 sec
[ Info: VUMPS  52:	obj = -1.508886724218e-01	err = 3.9414449451e-01	time = 0.04 sec
[ Info: VUMPS  53:	obj = -1.472371314173e-01	err = 3.9157489592e-01	time = 0.02 sec
[ Info: VUMPS  54:	obj = -2.733123904900e-01	err = 3.4841044726e-01	time = 0.02 sec
[ Info: VUMPS  55:	obj = -2.626559265073e-01	err = 3.5066197277e-01	time = 0.03 sec
[ Info: VUMPS  56:	obj = -1.755105074294e-01	err = 3.6566604380e-01	time = 0.03 sec
[ Info: VUMPS  57:	obj = -2.873757188880e-01	err = 3.4512689663e-01	time = 0.03 sec
[ Info: VUMPS  58:	obj = -1.882835011658e-01	err = 3.7810921433e-01	time = 0.03 sec
[ Info: VUMPS  59:	obj = -1.213265981733e-01	err = 4.3545043540e-01	time = 0.04 sec
[ Info: VUMPS  60:	obj = -7.057345301695e-02	err = 4.0587216976e-01	time = 0.03 sec
[ Info: VUMPS  61:	obj = +1.568730212024e-02	err = 4.2319714021e-01	time = 0.04 sec
[ Info: VUMPS  62:	obj = +6.300630281618e-02	err = 3.5947433289e-01	time = 0.04 sec
[ Info: VUMPS  63:	obj = -1.839724171536e-01	err = 3.7674966611e-01	time = 0.05 sec
[ Info: VUMPS  64:	obj = -2.890525430168e-01	err = 3.5226705042e-01	time = 0.04 sec
[ Info: VUMPS  65:	obj = -1.542410381207e-01	err = 3.9122053159e-01	time = 0.05 sec
[ Info: VUMPS  66:	obj = -2.211463957799e-01	err = 3.7347840312e-01	time = 0.04 sec
[ Info: VUMPS  67:	obj = -1.349717287503e-01	err = 3.8867734508e-01	time = 0.08 sec
[ Info: VUMPS  68:	obj = -7.869057213654e-02	err = 3.8707850708e-01	time = 0.03 sec
[ Info: VUMPS  69:	obj = -2.410192529204e-01	err = 3.6662424970e-01	time = 0.02 sec
[ Info: VUMPS  70:	obj = -1.166068691822e-01	err = 3.9576199157e-01	time = 0.02 sec
[ Info: VUMPS  71:	obj = -1.062577034280e-01	err = 3.8968393700e-01	time = 0.02 sec
[ Info: VUMPS  72:	obj = +3.179638144437e-02	err = 3.6789960681e-01	time = 0.03 sec
[ Info: VUMPS  73:	obj = +9.641215458614e-02	err = 3.7514164978e-01	time = 0.03 sec
[ Info: VUMPS  74:	obj = -2.105952009345e-01	err = 3.7296996469e-01	time = 0.02 sec
[ Info: VUMPS  75:	obj = +1.693674425858e-01	err = 3.3528625590e-01	time = 0.02 sec
[ Info: VUMPS  76:	obj = -3.296480578151e-02	err = 3.8665727637e-01	time = 0.03 sec
[ Info: VUMPS  77:	obj = -1.275013814321e-01	err = 3.6684408890e-01	time = 0.03 sec
[ Info: VUMPS  78:	obj = -1.565292795446e-01	err = 3.9656699980e-01	time = 0.03 sec
[ Info: VUMPS  79:	obj = -9.826992974175e-02	err = 3.6356780860e-01	time = 0.02 sec
[ Info: VUMPS  80:	obj = -1.736354442482e-01	err = 3.5616317706e-01	time = 0.03 sec
[ Info: VUMPS  81:	obj = -3.418407559702e-01	err = 3.2420327092e-01	time = 0.03 sec
[ Info: VUMPS  82:	obj = -3.665175156451e-01	err = 2.8072864814e-01	time = 0.04 sec
[ Info: VUMPS  83:	obj = -1.744061430031e-01	err = 3.7943381080e-01	time = 0.04 sec
[ Info: VUMPS  84:	obj = -1.128677462266e-01	err = 4.1536438216e-01	time = 0.02 sec
[ Info: VUMPS  85:	obj = -8.892311383360e-02	err = 3.8443540698e-01	time = 0.03 sec
[ Info: VUMPS  86:	obj = -2.258502320075e-01	err = 3.7776538738e-01	time = 0.04 sec
[ Info: VUMPS  87:	obj = -2.373409735876e-01	err = 3.7662496206e-01	time = 0.03 sec
[ Info: VUMPS  88:	obj = -2.470863380363e-01	err = 3.6935485705e-01	time = 0.03 sec
[ Info: VUMPS  89:	obj = -3.093689415115e-01	err = 3.2708444985e-01	time = 0.04 sec
[ Info: VUMPS  90:	obj = +1.289407011770e-01	err = 3.8394266857e-01	time = 0.04 sec
[ Info: VUMPS  91:	obj = +3.780537454921e-02	err = 4.2492876845e-01	time = 0.03 sec
[ Info: VUMPS  92:	obj = -9.032662936965e-02	err = 4.1898821311e-01	time = 0.04 sec
[ Info: VUMPS  93:	obj = -2.041819768796e-01	err = 3.7315012467e-01	time = 0.02 sec
[ Info: VUMPS  94:	obj = +8.092369079039e-02	err = 3.7230814357e-01	time = 0.02 sec
[ Info: VUMPS  95:	obj = -1.567374172192e-01	err = 3.8157377919e-01	time = 0.02 sec
[ Info: VUMPS  96:	obj = -2.075974963200e-01	err = 3.6677836372e-01	time = 0.02 sec
[ Info: VUMPS  97:	obj = -1.526835327801e-01	err = 3.7278356306e-01	time = 0.03 sec
[ Info: VUMPS  98:	obj = -1.565080739952e-02	err = 3.8513322468e-01	time = 0.03 sec
[ Info: VUMPS  99:	obj = -2.684144006852e-01	err = 3.6500026351e-01	time = 0.03 sec
[ Info: VUMPS 100:	obj = -3.474646511786e-01	err = 3.1319159593e-01	time = 0.03 sec
[ Info: VUMPS 101:	obj = -2.708687609986e-01	err = 3.6361020840e-01	time = 0.04 sec
[ Info: VUMPS 102:	obj = -1.778075197531e-01	err = 3.7426902911e-01	time = 0.02 sec
[ Info: VUMPS 103:	obj = -3.591902974838e-01	err = 3.0719520961e-01	time = 0.03 sec
[ Info: VUMPS 104:	obj = -2.817299190235e-01	err = 3.3614106777e-01	time = 0.04 sec
[ Info: VUMPS 105:	obj = -1.324546940242e-01	err = 3.8103325196e-01	time = 0.04 sec
[ Info: VUMPS 106:	obj = -6.683826747593e-02	err = 4.0044456382e-01	time = 0.04 sec
[ Info: VUMPS 107:	obj = -2.342398956185e-01	err = 3.6236841262e-01	time = 0.02 sec
[ Info: VUMPS 108:	obj = -2.754174236447e-01	err = 3.3311031646e-01	time = 0.07 sec
[ Info: VUMPS 109:	obj = -7.240746681273e-02	err = 3.6953978416e-01	time = 0.04 sec
[ Info: VUMPS 110:	obj = -2.440267290272e-01	err = 3.5255437143e-01	time = 0.02 sec
[ Info: VUMPS 111:	obj = -2.318278961426e-01	err = 3.6423008551e-01	time = 0.04 sec
[ Info: VUMPS 112:	obj = -3.530141020561e-01	err = 3.0712337800e-01	time = 0.03 sec
[ Info: VUMPS 113:	obj = -4.042911123239e-01	err = 2.2987711135e-01	time = 0.04 sec
[ Info: VUMPS 114:	obj = -2.729341186721e-01	err = 3.6896276601e-01	time = 0.05 sec
[ Info: VUMPS 115:	obj = -1.418158133996e-01	err = 3.9576409679e-01	time = 0.04 sec
[ Info: VUMPS 116:	obj = -3.132282137565e-01	err = 3.4520411856e-01	time = 0.03 sec
[ Info: VUMPS 117:	obj = -9.757982865642e-02	err = 4.2267275580e-01	time = 0.03 sec
[ Info: VUMPS 118:	obj = +1.891070077617e-01	err = 3.3881402978e-01	time = 0.03 sec
[ Info: VUMPS 119:	obj = -2.157951017110e-01	err = 3.9036991930e-01	time = 0.03 sec
[ Info: VUMPS 120:	obj = -3.180493885920e-01	err = 3.2920508615e-01	time = 0.03 sec
[ Info: VUMPS 121:	obj = -7.135253209545e-02	err = 3.6100125478e-01	time = 0.03 sec
[ Info: VUMPS 122:	obj = -1.103549362015e-01	err = 3.9321079733e-01	time = 0.04 sec
[ Info: VUMPS 123:	obj = -1.859194425659e-01	err = 3.6448486374e-01	time = 0.03 sec
[ Info: VUMPS 124:	obj = -2.996916762343e-01	err = 3.4275333772e-01	time = 0.03 sec
[ Info: VUMPS 125:	obj = -1.249591579931e-02	err = 3.6358026272e-01	time = 0.02 sec
[ Info: VUMPS 126:	obj = -2.067357178460e-01	err = 3.5869346655e-01	time = 0.02 sec
[ Info: VUMPS 127:	obj = -3.436173299557e-01	err = 3.1795241740e-01	time = 0.03 sec
[ Info: VUMPS 128:	obj = -4.189533941112e-01	err = 1.7870019255e-01	time = 0.04 sec
[ Info: VUMPS 129:	obj = -1.492677595785e-01	err = 3.8380050678e-01	time = 0.04 sec
[ Info: VUMPS 130:	obj = -1.726973588317e-01	err = 3.6915060325e-01	time = 0.02 sec
[ Info: VUMPS 131:	obj = +4.039456588716e-05	err = 3.8267870748e-01	time = 0.03 sec
[ Info: VUMPS 132:	obj = -3.665590978477e-01	err = 3.0236787574e-01	time = 0.03 sec
[ Info: VUMPS 133:	obj = +8.226007982810e-02	err = 3.6881198595e-01	time = 0.04 sec
[ Info: VUMPS 134:	obj = -1.206738476211e-01	err = 3.6821435749e-01	time = 0.03 sec
[ Info: VUMPS 135:	obj = +6.225836366499e-02	err = 3.8637132260e-01	time = 0.02 sec
[ Info: VUMPS 136:	obj = -1.193417142631e-01	err = 3.8838583360e-01	time = 0.03 sec
[ Info: VUMPS 137:	obj = -2.971752843016e-01	err = 3.5058567787e-01	time = 0.02 sec
[ Info: VUMPS 138:	obj = -6.389758510920e-02	err = 3.7521021944e-01	time = 0.02 sec
[ Info: VUMPS 139:	obj = -3.747140853811e-02	err = 3.6373044270e-01	time = 0.02 sec
[ Info: VUMPS 140:	obj = -3.282279082784e-02	err = 3.7558191255e-01	time = 0.03 sec
[ Info: VUMPS 141:	obj = -1.007298467336e-01	err = 3.8024280224e-01	time = 0.03 sec
[ Info: VUMPS 142:	obj = -4.388073210477e-02	err = 3.7628114626e-01	time = 0.03 sec
[ Info: VUMPS 143:	obj = +4.109629556433e-02	err = 3.8442037461e-01	time = 0.03 sec
[ Info: VUMPS 144:	obj = +3.422715479826e-02	err = 3.7395066049e-01	time = 0.03 sec
[ Info: VUMPS 145:	obj = -2.232166082163e-02	err = 3.8195417272e-01	time = 0.03 sec
[ Info: VUMPS 146:	obj = -6.014589617987e-02	err = 3.8441484264e-01	time = 0.04 sec
[ Info: VUMPS 147:	obj = -1.992962792325e-01	err = 3.7040002611e-01	time = 0.04 sec
[ Info: VUMPS 148:	obj = -3.527684444926e-01	err = 3.0948035074e-01	time = 0.06 sec
[ Info: VUMPS 149:	obj = -4.132860251154e-01	err = 1.9243576221e-01	time = 0.04 sec
[ Info: VUMPS 150:	obj = +7.199962328196e-02	err = 3.6551811014e-01	time = 0.03 sec
[ Info: VUMPS 151:	obj = -3.113863706166e-02	err = 3.7717331498e-01	time = 0.03 sec
[ Info: VUMPS 152:	obj = -2.719190923920e-01	err = 3.4665041796e-01	time = 0.03 sec
[ Info: VUMPS 153:	obj = -5.771399678069e-02	err = 3.7035063390e-01	time = 0.02 sec
[ Info: VUMPS 154:	obj = -2.994927348718e-01	err = 3.6281290077e-01	time = 0.04 sec
[ Info: VUMPS 155:	obj = -2.317915228074e-01	err = 3.7532539684e-01	time = 0.03 sec
[ Info: VUMPS 156:	obj = -2.217739990492e-01	err = 3.8738345359e-01	time = 0.04 sec
[ Info: VUMPS 157:	obj = +2.196813123218e-02	err = 3.9655815404e-01	time = 0.03 sec
[ Info: VUMPS 158:	obj = -1.818346624421e-02	err = 3.9168430014e-01	time = 0.03 sec
[ Info: VUMPS 159:	obj = -4.550509716910e-02	err = 3.7625735761e-01	time = 0.02 sec
[ Info: VUMPS 160:	obj = -6.265559102949e-02	err = 3.8805817037e-01	time = 0.03 sec
[ Info: VUMPS 161:	obj = +4.153515092261e-02	err = 3.7061251790e-01	time = 0.03 sec
[ Info: VUMPS 162:	obj = -1.377019803131e-01	err = 3.7305610151e-01	time = 0.03 sec
[ Info: VUMPS 163:	obj = -1.461964086393e-01	err = 3.7332157473e-01	time = 0.02 sec
[ Info: VUMPS 164:	obj = -2.861413716215e-01	err = 3.4599192940e-01	time = 0.02 sec
[ Info: VUMPS 165:	obj = -2.650343963380e-01	err = 3.6283026239e-01	time = 0.04 sec
[ Info: VUMPS 166:	obj = -1.255896792447e-01	err = 3.8858619834e-01	time = 0.03 sec
[ Info: VUMPS 167:	obj = -1.636247023869e-01	err = 3.9796032707e-01	time = 0.04 sec
[ Info: VUMPS 168:	obj = -2.299914291075e-01	err = 3.8214177886e-01	time = 0.03 sec
[ Info: VUMPS 169:	obj = -2.718528075489e-01	err = 3.3894651241e-01	time = 0.03 sec
[ Info: VUMPS 170:	obj = -4.078473349080e-02	err = 3.7303944187e-01	time = 0.02 sec
[ Info: VUMPS 171:	obj = -2.442812296306e-01	err = 3.7044694730e-01	time = 0.03 sec
[ Info: VUMPS 172:	obj = +9.302572373226e-02	err = 3.6720541630e-01	time = 0.04 sec
[ Info: VUMPS 173:	obj = -2.070094092430e-01	err = 3.7009534705e-01	time = 0.03 sec
[ Info: VUMPS 174:	obj = -1.027513365156e-01	err = 3.5924824820e-01	time = 0.02 sec
[ Info: VUMPS 175:	obj = -1.083522162526e-01	err = 4.2980987011e-01	time = 0.03 sec
[ Info: VUMPS 176:	obj = -2.396376689046e-01	err = 3.4716688878e-01	time = 0.03 sec
[ Info: VUMPS 177:	obj = -3.643716010929e-01	err = 2.9647631700e-01	time = 0.03 sec
[ Info: VUMPS 178:	obj = -6.903039239604e-02	err = 4.0829866932e-01	time = 0.03 sec
[ Info: VUMPS 179:	obj = -1.352468971136e-01	err = 3.6072505862e-01	time = 0.03 sec
[ Info: VUMPS 180:	obj = +5.776947297214e-02	err = 3.9041297403e-01	time = 0.02 sec
[ Info: VUMPS 181:	obj = -2.883078561741e-01	err = 3.4068046607e-01	time = 0.03 sec
[ Info: VUMPS 182:	obj = -2.684187217879e-01	err = 3.6022021727e-01	time = 0.03 sec
[ Info: VUMPS 183:	obj = -2.113457081006e-01	err = 3.7603452142e-01	time = 0.04 sec
[ Info: VUMPS 184:	obj = -2.828595996450e-01	err = 3.6065647566e-01	time = 0.03 sec
[ Info: VUMPS 185:	obj = -1.822129780012e-01	err = 3.5925178921e-01	time = 0.04 sec
[ Info: VUMPS 186:	obj = -2.353354313798e-01	err = 3.8655657878e-01	time = 0.03 sec
[ Info: VUMPS 187:	obj = -2.210146389915e-01	err = 3.7300491469e-01	time = 0.04 sec
[ Info: VUMPS 188:	obj = -2.984597537617e-01	err = 3.6703306166e-01	time = 0.07 sec
[ Info: VUMPS 189:	obj = -3.091748935398e-01	err = 3.2988666183e-01	time = 0.03 sec
[ Info: VUMPS 190:	obj = -3.976392432653e-01	err = 2.4255646424e-01	time = 0.04 sec
[ Info: VUMPS 191:	obj = -1.171842832213e-01	err = 3.8963103617e-01	time = 0.03 sec
[ Info: VUMPS 192:	obj = -2.386620031281e-01	err = 3.6185852723e-01	time = 0.02 sec
[ Info: VUMPS 193:	obj = -1.971553964802e-01	err = 3.8366047969e-01	time = 0.03 sec
[ Info: VUMPS 194:	obj = -2.086391373346e-01	err = 3.9959605566e-01	time = 0.04 sec
[ Info: VUMPS 195:	obj = -7.720592836186e-02	err = 3.8698566641e-01	time = 0.04 sec
[ Info: VUMPS 196:	obj = -2.173301511469e-01	err = 3.4742444743e-01	time = 0.03 sec
[ Info: VUMPS 197:	obj = -2.659400212546e-01	err = 3.5647314894e-01	time = 0.03 sec
[ Info: VUMPS 198:	obj = -1.172972093301e-01	err = 4.2374913198e-01	time = 0.04 sec
[ Info: VUMPS 199:	obj = -1.315856317921e-01	err = 4.0826850110e-01	time = 0.03 sec
┌ Warning: VUMPS cancel 200:	obj = -6.793162683723e-02	err = 3.7052797287e-01	time = 8.62 sec
└ @ MPSKit ~/git/MPSKit.jl/src/algorithms/groundstate/vumps.jl:73

````

As you can see, VUMPS struggles to converge.
On it's own, that is already quite curious.
Maybe we can do better using another algorithm, such as gradient descent.

````julia
groundstate, cache, delta = find_groundstate(state, H, GradientGrassmann(; maxiter=20));
````

````
[ Info: CG: initializing with f = 0.249998387147, ‖∇f‖ = 2.6463e-03
┌ Warning: resorting to η
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/cg.jl:207
┌ Warning: CG: not converged to requested tol after 20 iterations and time 2.90 s: f = -0.442690602388, ‖∇f‖ = 6.2203e-03
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
<polyline clip-path="url(#clip332)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="219.288,1334.38 2352.76,1334.38 "/>
<polyline clip-path="url(#clip332)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="219.288,966.629 2352.76,966.629 "/>
<polyline clip-path="url(#clip332)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="219.288,598.875 2352.76,598.875 "/>
<polyline clip-path="url(#clip332)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="219.288,231.121 2352.76,231.121 "/>
<polyline clip-path="url(#clip330)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="219.288,1423.18 2352.76,1423.18 "/>
<polyline clip-path="url(#clip330)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="279.669,1423.18 279.669,1404.28 "/>
<polyline clip-path="url(#clip330)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="615.12,1423.18 615.12,1404.28 "/>
<polyline clip-path="url(#clip330)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="950.571,1423.18 950.571,1404.28 "/>
<polyline clip-path="url(#clip330)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1286.02,1423.18 1286.02,1404.28 "/>
<polyline clip-path="url(#clip330)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1621.47,1423.18 1621.47,1404.28 "/>
<polyline clip-path="url(#clip330)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1956.92,1423.18 1956.92,1404.28 "/>
<polyline clip-path="url(#clip330)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="2292.37,1423.18 2292.37,1404.28 "/>
<path clip-path="url(#clip330)" d="M233.431 1454.1 Q229.819 1454.1 227.991 1457.66 Q226.185 1461.2 226.185 1468.33 Q226.185 1475.44 227.991 1479.01 Q229.819 1482.55 233.431 1482.55 Q237.065 1482.55 238.87 1479.01 Q240.699 1475.44 240.699 1468.33 Q240.699 1461.2 238.87 1457.66 Q237.065 1454.1 233.431 1454.1 M233.431 1450.39 Q239.241 1450.39 242.296 1455 Q245.375 1459.58 245.375 1468.33 Q245.375 1477.06 242.296 1481.67 Q239.241 1486.25 233.431 1486.25 Q227.62 1486.25 224.542 1481.67 Q221.486 1477.06 221.486 1468.33 Q221.486 1459.58 224.542 1455 Q227.62 1450.39 233.431 1450.39 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M260.56 1451.02 L264.495 1451.02 L252.458 1489.98 L248.523 1489.98 L260.56 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M276.532 1451.02 L280.467 1451.02 L268.43 1489.98 L264.495 1489.98 L276.532 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M286.347 1481.64 L293.986 1481.64 L293.986 1455.28 L285.676 1456.95 L285.676 1452.69 L293.94 1451.02 L298.615 1451.02 L298.615 1481.64 L306.254 1481.64 L306.254 1485.58 L286.347 1485.58 L286.347 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M312.342 1459.65 L337.18 1459.65 L337.18 1463.91 L333.916 1463.91 L333.916 1479.84 Q333.916 1481.51 334.472 1482.25 Q335.05 1482.96 336.324 1482.96 Q336.671 1482.96 337.18 1482.92 Q337.689 1482.85 337.851 1482.83 L337.851 1485.9 Q337.041 1486.2 336.185 1486.34 Q335.328 1486.48 334.472 1486.48 Q331.694 1486.48 330.629 1484.98 Q329.564 1483.45 329.564 1479.38 L329.564 1463.91 L320.004 1463.91 L320.004 1485.58 L315.652 1485.58 L315.652 1463.91 L312.342 1463.91 L312.342 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M558.65 1481.64 L566.289 1481.64 L566.289 1455.28 L557.979 1456.95 L557.979 1452.69 L566.243 1451.02 L570.919 1451.02 L570.919 1481.64 L578.557 1481.64 L578.557 1485.58 L558.65 1485.58 L558.65 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M594.969 1451.02 L598.905 1451.02 L586.868 1489.98 L582.932 1489.98 L594.969 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M610.942 1451.02 L614.877 1451.02 L602.84 1489.98 L598.905 1489.98 L610.942 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M634.113 1466.95 Q637.469 1467.66 639.344 1469.93 Q641.242 1472.2 641.242 1475.53 Q641.242 1480.65 637.724 1483.45 Q634.205 1486.25 627.724 1486.25 Q625.548 1486.25 623.233 1485.81 Q620.941 1485.39 618.488 1484.54 L618.488 1480.02 Q620.432 1481.16 622.747 1481.74 Q625.062 1482.32 627.585 1482.32 Q631.983 1482.32 634.275 1480.58 Q636.59 1478.84 636.59 1475.53 Q636.59 1472.48 634.437 1470.77 Q632.307 1469.03 628.488 1469.03 L624.46 1469.03 L624.46 1465.19 L628.673 1465.19 Q632.122 1465.19 633.951 1463.82 Q635.779 1462.43 635.779 1459.84 Q635.779 1457.18 633.881 1455.77 Q632.006 1454.33 628.488 1454.33 Q626.566 1454.33 624.367 1454.75 Q622.168 1455.16 619.529 1456.04 L619.529 1451.88 Q622.191 1451.14 624.506 1450.77 Q626.844 1450.39 628.904 1450.39 Q634.228 1450.39 637.33 1452.83 Q640.432 1455.23 640.432 1459.35 Q640.432 1462.22 638.789 1464.21 Q637.145 1466.18 634.113 1466.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M646.752 1459.65 L671.589 1459.65 L671.589 1463.91 L668.325 1463.91 L668.325 1479.84 Q668.325 1481.51 668.881 1482.25 Q669.46 1482.96 670.733 1482.96 Q671.08 1482.96 671.589 1482.92 Q672.099 1482.85 672.261 1482.83 L672.261 1485.9 Q671.45 1486.2 670.594 1486.34 Q669.738 1486.48 668.881 1486.48 Q666.103 1486.48 665.038 1484.98 Q663.974 1483.45 663.974 1479.38 L663.974 1463.91 L654.414 1463.91 L654.414 1485.58 L650.062 1485.58 L650.062 1463.91 L646.752 1463.91 L646.752 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M898.187 1481.64 L914.506 1481.64 L914.506 1485.58 L892.562 1485.58 L892.562 1481.64 Q895.224 1478.89 899.807 1474.26 Q904.414 1469.61 905.594 1468.27 Q907.839 1465.74 908.719 1464.01 Q909.622 1462.25 909.622 1460.56 Q909.622 1457.8 907.677 1456.07 Q905.756 1454.33 902.654 1454.33 Q900.455 1454.33 898.002 1455.09 Q895.571 1455.86 892.793 1457.41 L892.793 1452.69 Q895.617 1451.55 898.071 1450.97 Q900.525 1450.39 902.562 1450.39 Q907.932 1450.39 911.127 1453.08 Q914.321 1455.77 914.321 1460.26 Q914.321 1462.39 913.511 1464.31 Q912.724 1466.2 910.617 1468.8 Q910.039 1469.47 906.937 1472.69 Q903.835 1475.88 898.187 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M931.288 1451.02 L935.224 1451.02 L923.187 1489.98 L919.251 1489.98 L931.288 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M947.261 1451.02 L951.196 1451.02 L939.159 1489.98 L935.224 1489.98 L947.261 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M970.432 1466.95 Q973.788 1467.66 975.663 1469.93 Q977.561 1472.2 977.561 1475.53 Q977.561 1480.65 974.043 1483.45 Q970.524 1486.25 964.043 1486.25 Q961.867 1486.25 959.552 1485.81 Q957.261 1485.39 954.807 1484.54 L954.807 1480.02 Q956.751 1481.16 959.066 1481.74 Q961.381 1482.32 963.904 1482.32 Q968.302 1482.32 970.594 1480.58 Q972.909 1478.84 972.909 1475.53 Q972.909 1472.48 970.756 1470.77 Q968.626 1469.03 964.807 1469.03 L960.779 1469.03 L960.779 1465.19 L964.992 1465.19 Q968.441 1465.19 970.27 1463.82 Q972.098 1462.43 972.098 1459.84 Q972.098 1457.18 970.2 1455.77 Q968.325 1454.33 964.807 1454.33 Q962.885 1454.33 960.686 1454.75 Q958.487 1455.16 955.848 1456.04 L955.848 1451.88 Q958.511 1451.14 960.825 1450.77 Q963.163 1450.39 965.223 1450.39 Q970.547 1450.39 973.649 1452.83 Q976.751 1455.23 976.751 1459.35 Q976.751 1462.22 975.108 1464.21 Q973.464 1466.18 970.432 1466.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M983.071 1459.65 L1007.91 1459.65 L1007.91 1463.91 L1004.64 1463.91 L1004.64 1479.84 Q1004.64 1481.51 1005.2 1482.25 Q1005.78 1482.96 1007.05 1482.96 Q1007.4 1482.96 1007.91 1482.92 Q1008.42 1482.85 1008.58 1482.83 L1008.58 1485.9 Q1007.77 1486.2 1006.91 1486.34 Q1006.06 1486.48 1005.2 1486.48 Q1002.42 1486.48 1001.36 1484.98 Q1000.29 1483.45 1000.29 1479.38 L1000.29 1463.91 L990.733 1463.91 L990.733 1485.58 L986.381 1485.58 L986.381 1463.91 L983.071 1463.91 L983.071 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M1229.55 1481.64 L1237.19 1481.64 L1237.19 1455.28 L1228.88 1456.95 L1228.88 1452.69 L1237.14 1451.02 L1241.82 1451.02 L1241.82 1481.64 L1249.46 1481.64 L1249.46 1485.58 L1229.55 1485.58 L1229.55 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M1265.87 1451.02 L1269.81 1451.02 L1257.77 1489.98 L1253.83 1489.98 L1265.87 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M1281.84 1451.02 L1285.78 1451.02 L1273.74 1489.98 L1269.81 1489.98 L1281.84 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M1291.66 1481.64 L1299.3 1481.64 L1299.3 1455.28 L1290.99 1456.95 L1290.99 1452.69 L1299.25 1451.02 L1303.93 1451.02 L1303.93 1481.64 L1311.57 1481.64 L1311.57 1485.58 L1291.66 1485.58 L1291.66 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M1317.65 1459.65 L1342.49 1459.65 L1342.49 1463.91 L1339.23 1463.91 L1339.23 1479.84 Q1339.23 1481.51 1339.78 1482.25 Q1340.36 1482.96 1341.63 1482.96 Q1341.98 1482.96 1342.49 1482.92 Q1343 1482.85 1343.16 1482.83 L1343.16 1485.9 Q1342.35 1486.2 1341.5 1486.34 Q1340.64 1486.48 1339.78 1486.48 Q1337.01 1486.48 1335.94 1484.98 Q1334.88 1483.45 1334.88 1479.38 L1334.88 1463.91 L1325.32 1463.91 L1325.32 1485.58 L1320.96 1485.58 L1320.96 1463.91 L1317.65 1463.91 L1317.65 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M1578.49 1455.09 L1566.68 1473.54 L1578.49 1473.54 L1578.49 1455.09 M1577.26 1451.02 L1583.14 1451.02 L1583.14 1473.54 L1588.07 1473.54 L1588.07 1477.43 L1583.14 1477.43 L1583.14 1485.58 L1578.49 1485.58 L1578.49 1477.43 L1562.89 1477.43 L1562.89 1472.92 L1577.26 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M1602.77 1451.02 L1606.7 1451.02 L1594.67 1489.98 L1590.73 1489.98 L1602.77 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M1618.74 1451.02 L1622.68 1451.02 L1610.64 1489.98 L1606.7 1489.98 L1618.74 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M1641.91 1466.95 Q1645.27 1467.66 1647.14 1469.93 Q1649.04 1472.2 1649.04 1475.53 Q1649.04 1480.65 1645.52 1483.45 Q1642.01 1486.25 1635.52 1486.25 Q1633.35 1486.25 1631.03 1485.81 Q1628.74 1485.39 1626.29 1484.54 L1626.29 1480.02 Q1628.23 1481.16 1630.55 1481.74 Q1632.86 1482.32 1635.38 1482.32 Q1639.78 1482.32 1642.07 1480.58 Q1644.39 1478.84 1644.39 1475.53 Q1644.39 1472.48 1642.24 1470.77 Q1640.11 1469.03 1636.29 1469.03 L1632.26 1469.03 L1632.26 1465.19 L1636.47 1465.19 Q1639.92 1465.19 1641.75 1463.82 Q1643.58 1462.43 1643.58 1459.84 Q1643.58 1457.18 1641.68 1455.77 Q1639.81 1454.33 1636.29 1454.33 Q1634.37 1454.33 1632.17 1454.75 Q1629.97 1455.16 1627.33 1456.04 L1627.33 1451.88 Q1629.99 1451.14 1632.31 1450.77 Q1634.64 1450.39 1636.7 1450.39 Q1642.03 1450.39 1645.13 1452.83 Q1648.23 1455.23 1648.23 1459.35 Q1648.23 1462.22 1646.59 1464.21 Q1644.94 1466.18 1641.91 1466.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M1654.55 1459.65 L1679.39 1459.65 L1679.39 1463.91 L1676.13 1463.91 L1676.13 1479.84 Q1676.13 1481.51 1676.68 1482.25 Q1677.26 1482.96 1678.53 1482.96 Q1678.88 1482.96 1679.39 1482.92 Q1679.9 1482.85 1680.06 1482.83 L1680.06 1485.9 Q1679.25 1486.2 1678.39 1486.34 Q1677.54 1486.48 1676.68 1486.48 Q1673.9 1486.48 1672.84 1484.98 Q1671.77 1483.45 1671.77 1479.38 L1671.77 1463.91 L1662.21 1463.91 L1662.21 1485.58 L1657.86 1485.58 L1657.86 1463.91 L1654.55 1463.91 L1654.55 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M1900.47 1451.02 L1918.82 1451.02 L1918.82 1454.96 L1904.75 1454.96 L1904.75 1463.43 Q1905.77 1463.08 1906.79 1462.92 Q1907.8 1462.73 1908.82 1462.73 Q1914.61 1462.73 1917.99 1465.9 Q1921.37 1469.08 1921.37 1474.49 Q1921.37 1480.07 1917.9 1483.17 Q1914.42 1486.25 1908.1 1486.25 Q1905.93 1486.25 1903.66 1485.88 Q1901.41 1485.51 1899.01 1484.77 L1899.01 1480.07 Q1901.09 1481.2 1903.31 1481.76 Q1905.54 1482.32 1908.01 1482.32 Q1912.02 1482.32 1914.35 1480.21 Q1916.69 1478.1 1916.69 1474.49 Q1916.69 1470.88 1914.35 1468.77 Q1912.02 1466.67 1908.01 1466.67 Q1906.14 1466.67 1904.26 1467.08 Q1902.41 1467.5 1900.47 1468.38 L1900.47 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M1937.55 1451.02 L1941.48 1451.02 L1929.45 1489.98 L1925.51 1489.98 L1937.55 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M1953.52 1451.02 L1957.46 1451.02 L1945.42 1489.98 L1941.48 1489.98 L1953.52 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M1976.69 1466.95 Q1980.05 1467.66 1981.92 1469.93 Q1983.82 1472.2 1983.82 1475.53 Q1983.82 1480.65 1980.3 1483.45 Q1976.78 1486.25 1970.3 1486.25 Q1968.13 1486.25 1965.81 1485.81 Q1963.52 1485.39 1961.07 1484.54 L1961.07 1480.02 Q1963.01 1481.16 1965.33 1481.74 Q1967.64 1482.32 1970.16 1482.32 Q1974.56 1482.32 1976.85 1480.58 Q1979.17 1478.84 1979.17 1475.53 Q1979.17 1472.48 1977.02 1470.77 Q1974.89 1469.03 1971.07 1469.03 L1967.04 1469.03 L1967.04 1465.19 L1971.25 1465.19 Q1974.7 1465.19 1976.53 1463.82 Q1978.36 1462.43 1978.36 1459.84 Q1978.36 1457.18 1976.46 1455.77 Q1974.59 1454.33 1971.07 1454.33 Q1969.15 1454.33 1966.95 1454.75 Q1964.75 1455.16 1962.11 1456.04 L1962.11 1451.88 Q1964.77 1451.14 1967.09 1450.77 Q1969.42 1450.39 1971.48 1450.39 Q1976.81 1450.39 1979.91 1452.83 Q1983.01 1455.23 1983.01 1459.35 Q1983.01 1462.22 1981.37 1464.21 Q1979.72 1466.18 1976.69 1466.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M1989.33 1459.65 L2014.17 1459.65 L2014.17 1463.91 L2010.9 1463.91 L2010.9 1479.84 Q2010.9 1481.51 2011.46 1482.25 Q2012.04 1482.96 2013.31 1482.96 Q2013.66 1482.96 2014.17 1482.92 Q2014.68 1482.85 2014.84 1482.83 L2014.84 1485.9 Q2014.03 1486.2 2013.17 1486.34 Q2012.32 1486.48 2011.46 1486.48 Q2008.68 1486.48 2007.62 1484.98 Q2006.55 1483.45 2006.55 1479.38 L2006.55 1463.91 L1996.99 1463.91 L1996.99 1485.58 L1992.64 1485.58 L1992.64 1463.91 L1989.33 1463.91 L1989.33 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M2239.99 1481.64 L2256.31 1481.64 L2256.31 1485.58 L2234.37 1485.58 L2234.37 1481.64 Q2237.03 1478.89 2241.61 1474.26 Q2246.22 1469.61 2247.4 1468.27 Q2249.64 1465.74 2250.52 1464.01 Q2251.43 1462.25 2251.43 1460.56 Q2251.43 1457.8 2249.48 1456.07 Q2247.56 1454.33 2244.46 1454.33 Q2242.26 1454.33 2239.81 1455.09 Q2237.38 1455.86 2234.6 1457.41 L2234.6 1452.69 Q2237.42 1451.55 2239.88 1450.97 Q2242.33 1450.39 2244.37 1450.39 Q2249.74 1450.39 2252.93 1453.08 Q2256.12 1455.77 2256.12 1460.26 Q2256.12 1462.39 2255.31 1464.31 Q2254.53 1466.2 2252.42 1468.8 Q2251.84 1469.47 2248.74 1472.69 Q2245.64 1475.88 2239.99 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M2273.09 1451.02 L2277.03 1451.02 L2264.99 1489.98 L2261.06 1489.98 L2273.09 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M2289.06 1451.02 L2293 1451.02 L2280.96 1489.98 L2277.03 1489.98 L2289.06 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M2298.88 1481.64 L2306.52 1481.64 L2306.52 1455.28 L2298.21 1456.95 L2298.21 1452.69 L2306.47 1451.02 L2311.15 1451.02 L2311.15 1481.64 L2318.79 1481.64 L2318.79 1485.58 L2298.88 1485.58 L2298.88 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M2324.87 1459.65 L2349.71 1459.65 L2349.71 1463.91 L2346.45 1463.91 L2346.45 1479.84 Q2346.45 1481.51 2347 1482.25 Q2347.58 1482.96 2348.86 1482.96 Q2349.2 1482.96 2349.71 1482.92 Q2350.22 1482.85 2350.38 1482.83 L2350.38 1485.9 Q2349.57 1486.2 2348.72 1486.34 Q2347.86 1486.48 2347 1486.48 Q2344.23 1486.48 2343.16 1484.98 Q2342.1 1483.45 2342.1 1479.38 L2342.1 1463.91 L2332.54 1463.91 L2332.54 1485.58 L2328.18 1485.58 L2328.18 1463.91 L2324.87 1463.91 L2324.87 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M1296.14 1545.45 L1275.87 1545.45 Q1276.35 1554.96 1278.54 1559 Q1281.28 1563.97 1286.02 1563.97 Q1290.8 1563.97 1293.44 1558.97 Q1295.76 1554.58 1296.14 1545.45 M1296.05 1540.03 Q1295.16 1531 1293.44 1527.81 Q1290.7 1522.78 1286.02 1522.78 Q1281.15 1522.78 1278.57 1527.75 Q1276.54 1531.76 1275.93 1540.03 L1296.05 1540.03 M1286.02 1518.01 Q1293.66 1518.01 1298.02 1524.76 Q1302.38 1531.47 1302.38 1543.38 Q1302.38 1555.25 1298.02 1562 Q1293.66 1568.78 1286.02 1568.78 Q1278.35 1568.78 1274.02 1562 Q1269.66 1555.25 1269.66 1543.38 Q1269.66 1531.47 1274.02 1524.76 Q1278.35 1518.01 1286.02 1518.01 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><polyline clip-path="url(#clip330)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="219.288,1423.18 219.288,47.2441 "/>
<polyline clip-path="url(#clip330)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="219.288,1334.38 238.185,1334.38 "/>
<polyline clip-path="url(#clip330)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="219.288,966.629 238.185,966.629 "/>
<polyline clip-path="url(#clip330)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="219.288,598.875 238.185,598.875 "/>
<polyline clip-path="url(#clip330)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="219.288,231.121 238.185,231.121 "/>
<path clip-path="url(#clip330)" d="M127.015 1320.18 Q123.404 1320.18 121.575 1323.75 Q119.769 1327.29 119.769 1334.42 Q119.769 1341.52 121.575 1345.09 Q123.404 1348.63 127.015 1348.63 Q130.649 1348.63 132.455 1345.09 Q134.283 1341.52 134.283 1334.42 Q134.283 1327.29 132.455 1323.75 Q130.649 1320.18 127.015 1320.18 M127.015 1316.48 Q132.825 1316.48 135.88 1321.08 Q138.959 1325.67 138.959 1334.42 Q138.959 1343.14 135.88 1347.75 Q132.825 1352.33 127.015 1352.33 Q121.205 1352.33 118.126 1347.75 Q115.07 1343.14 115.07 1334.42 Q115.07 1325.67 118.126 1321.08 Q121.205 1316.48 127.015 1316.48 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M147.177 1345.78 L152.061 1345.78 L152.061 1351.66 L147.177 1351.66 L147.177 1345.78 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M161.065 1317.1 L183.288 1317.1 L183.288 1319.09 L170.741 1351.66 L165.857 1351.66 L177.663 1321.04 L161.065 1321.04 L161.065 1317.1 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M126.205 952.428 Q122.593 952.428 120.765 955.993 Q118.959 959.534 118.959 966.664 Q118.959 973.771 120.765 977.335 Q122.593 980.877 126.205 980.877 Q129.839 980.877 131.644 977.335 Q133.473 973.771 133.473 966.664 Q133.473 959.534 131.644 955.993 Q129.839 952.428 126.205 952.428 M126.205 948.724 Q132.015 948.724 135.07 953.331 Q138.149 957.914 138.149 966.664 Q138.149 975.391 135.07 979.997 Q132.015 984.581 126.205 984.581 Q120.394 984.581 117.316 979.997 Q114.26 975.391 114.26 966.664 Q114.26 957.914 117.316 953.331 Q120.394 948.724 126.205 948.724 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M146.366 978.03 L151.251 978.03 L151.251 983.909 L146.366 983.909 L146.366 978.03 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M171.436 967.497 Q168.102 967.497 166.181 969.28 Q164.283 971.062 164.283 974.187 Q164.283 977.312 166.181 979.095 Q168.102 980.877 171.436 980.877 Q174.769 980.877 176.69 979.095 Q178.612 977.289 178.612 974.187 Q178.612 971.062 176.69 969.28 Q174.792 967.497 171.436 967.497 M166.76 965.507 Q163.751 964.766 162.061 962.706 Q160.394 960.646 160.394 957.683 Q160.394 953.539 163.334 951.132 Q166.297 948.724 171.436 948.724 Q176.598 948.724 179.538 951.132 Q182.477 953.539 182.477 957.683 Q182.477 960.646 180.788 962.706 Q179.121 964.766 176.135 965.507 Q179.514 966.294 181.389 968.585 Q183.288 970.877 183.288 974.187 Q183.288 979.21 180.209 981.895 Q177.153 984.581 171.436 984.581 Q165.718 984.581 162.64 981.895 Q159.584 979.21 159.584 974.187 Q159.584 970.877 161.482 968.585 Q163.38 966.294 166.76 965.507 M165.047 958.122 Q165.047 960.808 166.714 962.312 Q168.403 963.817 171.436 963.817 Q174.445 963.817 176.135 962.312 Q177.848 960.808 177.848 958.122 Q177.848 955.437 176.135 953.933 Q174.445 952.428 171.436 952.428 Q168.403 952.428 166.714 953.933 Q165.047 955.437 165.047 958.122 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M126.297 584.674 Q122.686 584.674 120.857 588.239 Q119.052 591.78 119.052 598.91 Q119.052 606.016 120.857 609.581 Q122.686 613.123 126.297 613.123 Q129.931 613.123 131.737 609.581 Q133.566 606.016 133.566 598.91 Q133.566 591.78 131.737 588.239 Q129.931 584.674 126.297 584.674 M126.297 580.97 Q132.107 580.97 135.163 585.577 Q138.242 590.16 138.242 598.91 Q138.242 607.637 135.163 612.243 Q132.107 616.827 126.297 616.827 Q120.487 616.827 117.408 612.243 Q114.353 607.637 114.353 598.91 Q114.353 590.16 117.408 585.577 Q120.487 580.97 126.297 580.97 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M146.459 610.276 L151.343 610.276 L151.343 616.155 L146.459 616.155 L146.459 610.276 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M161.667 615.438 L161.667 611.178 Q163.427 612.012 165.232 612.452 Q167.038 612.891 168.774 612.891 Q173.403 612.891 175.834 609.79 Q178.288 606.665 178.635 600.322 Q177.292 602.313 175.232 603.378 Q173.172 604.442 170.672 604.442 Q165.487 604.442 162.454 601.317 Q159.445 598.169 159.445 592.729 Q159.445 587.405 162.593 584.188 Q165.741 580.97 170.973 580.97 Q176.968 580.97 180.116 585.577 Q183.288 590.16 183.288 598.91 Q183.288 607.081 179.399 611.965 Q175.533 616.827 168.982 616.827 Q167.223 616.827 165.417 616.479 Q163.612 616.132 161.667 615.438 M170.973 600.785 Q174.121 600.785 175.95 598.632 Q177.801 596.479 177.801 592.729 Q177.801 589.003 175.95 586.85 Q174.121 584.674 170.973 584.674 Q167.825 584.674 165.973 586.85 Q164.144 589.003 164.144 592.729 Q164.144 596.479 165.973 598.632 Q167.825 600.785 170.973 600.785 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M116.922 244.466 L124.561 244.466 L124.561 218.1 L116.251 219.767 L116.251 215.508 L124.515 213.841 L129.191 213.841 L129.191 244.466 L136.829 244.466 L136.829 248.401 L116.922 248.401 L116.922 244.466 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M146.274 242.522 L151.158 242.522 L151.158 248.401 L146.274 248.401 L146.274 242.522 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M171.343 216.92 Q167.732 216.92 165.903 220.485 Q164.098 224.026 164.098 231.156 Q164.098 238.262 165.903 241.827 Q167.732 245.369 171.343 245.369 Q174.977 245.369 176.783 241.827 Q178.612 238.262 178.612 231.156 Q178.612 224.026 176.783 220.485 Q174.977 216.92 171.343 216.92 M171.343 213.216 Q177.153 213.216 180.209 217.823 Q183.288 222.406 183.288 231.156 Q183.288 239.883 180.209 244.489 Q177.153 249.072 171.343 249.072 Q165.533 249.072 162.454 244.489 Q159.399 239.883 159.399 231.156 Q159.399 222.406 162.454 217.823 Q165.533 213.216 171.343 213.216 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip330)" d="M33.8307 724.772 Q33.2578 725.759 33.0032 726.937 Q32.7167 728.082 32.7167 729.483 Q32.7167 734.448 35.9632 737.122 Q39.1779 739.763 45.2253 739.763 L64.0042 739.763 L64.0042 745.652 L28.3562 745.652 L28.3562 739.763 L33.8944 739.763 Q30.6479 737.917 29.0883 734.957 Q27.4968 731.997 27.4968 727.764 Q27.4968 727.159 27.5923 726.427 Q27.656 725.695 27.8151 724.804 L33.8307 724.772 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><line clip-path="url(#clip332)" x1="279.669" y1="231.121" x2="279.669" y2="215.121" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="279.669" y1="231.121" x2="263.669" y2="231.121" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="279.669" y1="231.121" x2="279.669" y2="247.121" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="279.669" y1="231.121" x2="295.669" y2="231.121" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1286.02" y1="231.207" x2="1286.02" y2="215.207" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1286.02" y1="231.207" x2="1270.02" y2="231.207" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1286.02" y1="231.207" x2="1286.02" y2="247.207" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1286.02" y1="231.207" x2="1302.02" y2="231.207" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="786.542" y1="361.207" x2="786.542" y2="345.207" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="786.542" y1="361.207" x2="770.542" y2="361.207" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="786.542" y1="361.207" x2="786.542" y2="377.207" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="786.542" y1="361.207" x2="802.542" y2="361.207" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1785.5" y1="361.207" x2="1785.5" y2="345.207" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1785.5" y1="361.207" x2="1769.5" y2="361.207" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1785.5" y1="361.207" x2="1785.5" y2="377.207" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1785.5" y1="361.207" x2="1801.5" y2="361.207" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1792.89" y1="361.214" x2="1792.89" y2="345.214" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1792.89" y1="361.214" x2="1776.89" y2="361.214" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1792.89" y1="361.214" x2="1792.89" y2="377.214" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1792.89" y1="361.214" x2="1808.89" y2="361.214" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="779.149" y1="361.214" x2="779.149" y2="345.214" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="779.149" y1="361.214" x2="763.149" y2="361.214" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="779.149" y1="361.214" x2="779.149" y2="377.214" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="779.149" y1="361.214" x2="795.149" y2="361.214" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1286.02" y1="735.438" x2="1286.02" y2="719.438" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1286.02" y1="735.438" x2="1270.02" y2="735.438" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1286.02" y1="735.438" x2="1286.02" y2="751.438" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1286.02" y1="735.438" x2="1302.02" y2="735.438" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="2292.37" y1="735.449" x2="2292.37" y2="719.449" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="2292.37" y1="735.449" x2="2276.37" y2="735.449" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="2292.37" y1="735.449" x2="2292.37" y2="751.449" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="2292.37" y1="735.449" x2="2308.37" y2="735.449" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="286.223" y1="773.845" x2="286.223" y2="757.845" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="286.223" y1="773.845" x2="270.223" y2="773.845" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="286.223" y1="773.845" x2="286.223" y2="789.845" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="286.223" y1="773.845" x2="302.223" y2="773.845" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="2285.82" y1="773.845" x2="2285.82" y2="757.845" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="2285.82" y1="773.845" x2="2269.82" y2="773.845" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="2285.82" y1="773.845" x2="2285.82" y2="789.845" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="2285.82" y1="773.845" x2="2301.82" y2="773.845" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1292.58" y1="773.883" x2="1292.58" y2="757.883" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1292.58" y1="773.883" x2="1276.58" y2="773.883" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1292.58" y1="773.883" x2="1292.58" y2="789.883" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1292.58" y1="773.883" x2="1308.58" y2="773.883" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1279.47" y1="773.883" x2="1279.47" y2="757.883" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1279.47" y1="773.883" x2="1263.47" y2="773.883" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1279.47" y1="773.883" x2="1279.47" y2="789.883" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1279.47" y1="773.883" x2="1295.47" y2="773.883" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="777.631" y1="919.594" x2="777.631" y2="903.594" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="777.631" y1="919.594" x2="761.631" y2="919.594" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="777.631" y1="919.594" x2="777.631" y2="935.594" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="777.631" y1="919.594" x2="793.631" y2="919.594" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1794.41" y1="919.594" x2="1794.41" y2="903.594" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1794.41" y1="919.594" x2="1778.41" y2="919.594" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1794.41" y1="919.594" x2="1794.41" y2="935.594" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1794.41" y1="919.594" x2="1810.41" y2="919.594" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1783.98" y1="919.597" x2="1783.98" y2="903.597" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1783.98" y1="919.597" x2="1767.98" y2="919.597" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1783.98" y1="919.597" x2="1783.98" y2="935.597" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1783.98" y1="919.597" x2="1799.98" y2="919.597" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="788.065" y1="919.597" x2="788.065" y2="903.597" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="788.065" y1="919.597" x2="772.065" y2="919.597" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="788.065" y1="919.597" x2="788.065" y2="935.597" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="788.065" y1="919.597" x2="804.065" y2="919.597" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="2292.37" y1="1253.54" x2="2292.37" y2="1237.54" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="2292.37" y1="1253.54" x2="2276.37" y2="1253.54" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="2292.37" y1="1253.54" x2="2292.37" y2="1269.54" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="2292.37" y1="1253.54" x2="2308.37" y2="1253.54" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1286.02" y1="1254.13" x2="1286.02" y2="1238.13" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1286.02" y1="1254.13" x2="1270.02" y2="1254.13" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1286.02" y1="1254.13" x2="1286.02" y2="1270.13" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1286.02" y1="1254.13" x2="1302.02" y2="1254.13" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="781.197" y1="1423.15" x2="781.197" y2="1407.15" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="781.197" y1="1423.15" x2="765.197" y2="1423.15" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="781.197" y1="1423.15" x2="781.197" y2="1439.15" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="781.197" y1="1423.15" x2="797.197" y2="1423.15" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1790.85" y1="1423.15" x2="1790.85" y2="1407.15" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1790.85" y1="1423.15" x2="1774.85" y2="1423.15" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1790.85" y1="1423.15" x2="1790.85" y2="1439.15" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1790.85" y1="1423.15" x2="1806.85" y2="1423.15" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1787.55" y1="1423.18" x2="1787.55" y2="1407.18" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1787.55" y1="1423.18" x2="1771.55" y2="1423.18" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1787.55" y1="1423.18" x2="1787.55" y2="1439.18" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip332)" x1="1787.55" y1="1423.18" x2="1803.55" y2="1423.18" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
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
[ Info: VUMPS init:	obj = +4.983697867393e-01	err = 8.5072e-02
[ Info: VUMPS   1:	obj = -4.656735732107e-01	err = 3.3872930221e-01	time = 0.04 sec
[ Info: VUMPS   2:	obj = -8.769788304070e-01	err = 7.9811165363e-02	time = 0.03 sec
[ Info: VUMPS   3:	obj = -8.853566275004e-01	err = 1.0455423289e-02	time = 0.03 sec
[ Info: VUMPS   4:	obj = -8.859597619854e-01	err = 6.0168902769e-03	time = 0.03 sec
[ Info: VUMPS   5:	obj = -8.861267456470e-01	err = 3.7663986578e-03	time = 0.03 sec
[ Info: VUMPS   6:	obj = -8.861864570526e-01	err = 3.0381633175e-03	time = 0.04 sec
[ Info: VUMPS   7:	obj = -8.862125855082e-01	err = 2.3840295258e-03	time = 0.04 sec
[ Info: VUMPS   8:	obj = -8.862243873372e-01	err = 2.0736456024e-03	time = 0.04 sec
[ Info: VUMPS   9:	obj = -8.862300502699e-01	err = 1.9870596118e-03	time = 0.04 sec
[ Info: VUMPS  10:	obj = -8.862329197985e-01	err = 1.8666223968e-03	time = 0.05 sec
[ Info: VUMPS  11:	obj = -8.862341428689e-01	err = 2.0696080482e-03	time = 0.05 sec
[ Info: VUMPS  12:	obj = -8.862348049952e-01	err = 2.0229813738e-03	time = 0.05 sec
[ Info: VUMPS  13:	obj = -8.862347555253e-01	err = 2.4774235828e-03	time = 0.18 sec
[ Info: VUMPS  14:	obj = -8.862348122984e-01	err = 2.4506304845e-03	time = 0.05 sec
[ Info: VUMPS  15:	obj = -8.862330320053e-01	err = 3.5774989233e-03	time = 0.06 sec
[ Info: VUMPS  16:	obj = -8.862342168001e-01	err = 2.9845997178e-03	time = 0.05 sec
[ Info: VUMPS  17:	obj = -8.862333196761e-01	err = 3.7051254673e-03	time = 0.06 sec
[ Info: VUMPS  18:	obj = -8.862340313637e-01	err = 3.1202819885e-03	time = 0.05 sec
[ Info: VUMPS  19:	obj = -8.862295195897e-01	err = 5.2479795114e-03	time = 0.05 sec
[ Info: VUMPS  20:	obj = -8.862333159428e-01	err = 3.6007668774e-03	time = 0.05 sec
[ Info: VUMPS  21:	obj = -8.862321388578e-01	err = 4.4839503187e-03	time = 0.05 sec
[ Info: VUMPS  22:	obj = -8.862349140966e-01	err = 2.8839607924e-03	time = 0.06 sec
[ Info: VUMPS  23:	obj = -8.862323449676e-01	err = 4.5677411356e-03	time = 0.06 sec
[ Info: VUMPS  24:	obj = -8.862367535160e-01	err = 2.4066577810e-03	time = 0.06 sec
[ Info: VUMPS  25:	obj = -8.862353255959e-01	err = 3.4176669644e-03	time = 0.06 sec
[ Info: VUMPS  26:	obj = -8.862386648847e-01	err = 1.4624365726e-03	time = 0.06 sec
[ Info: VUMPS  27:	obj = -8.862390323760e-01	err = 1.2082689449e-03	time = 0.06 sec
[ Info: VUMPS  28:	obj = -8.862394595644e-01	err = 6.6519013426e-04	time = 0.06 sec
[ Info: VUMPS  29:	obj = -8.862395834445e-01	err = 4.6995439024e-04	time = 0.07 sec
[ Info: VUMPS  30:	obj = -8.862396632124e-01	err = 2.8271906335e-04	time = 0.06 sec
[ Info: VUMPS  31:	obj = -8.862396983145e-01	err = 2.0049311295e-04	time = 0.06 sec
[ Info: VUMPS  32:	obj = -8.862397207128e-01	err = 1.3241298064e-04	time = 0.07 sec
[ Info: VUMPS  33:	obj = -8.862397342037e-01	err = 1.0446821448e-04	time = 0.07 sec
[ Info: VUMPS  34:	obj = -8.862397439877e-01	err = 7.6090936129e-05	time = 0.07 sec
[ Info: VUMPS  35:	obj = -8.862397511011e-01	err = 6.8318270257e-05	time = 0.07 sec
[ Info: VUMPS  36:	obj = -8.862397567165e-01	err = 5.3661161669e-05	time = 0.12 sec
[ Info: VUMPS  37:	obj = -8.862397612140e-01	err = 5.2528534120e-05	time = 0.05 sec
[ Info: VUMPS  38:	obj = -8.862397649485e-01	err = 4.5610862829e-05	time = 0.05 sec
[ Info: VUMPS  39:	obj = -8.862397680976e-01	err = 4.3850287626e-05	time = 0.05 sec
[ Info: VUMPS  40:	obj = -8.862397708045e-01	err = 3.9808479516e-05	time = 0.05 sec
[ Info: VUMPS  41:	obj = -8.862397731620e-01	err = 3.8133019920e-05	time = 0.05 sec
[ Info: VUMPS  42:	obj = -8.862397752411e-01	err = 3.5403649092e-05	time = 0.05 sec
[ Info: VUMPS  43:	obj = -8.862397770939e-01	err = 3.3962615119e-05	time = 0.05 sec
[ Info: VUMPS  44:	obj = -8.862397787598e-01	err = 3.1965886282e-05	time = 0.05 sec
[ Info: VUMPS  45:	obj = -8.862397802694e-01	err = 3.0742159895e-05	time = 0.05 sec
[ Info: VUMPS  46:	obj = -8.862397816467e-01	err = 2.9206077633e-05	time = 0.05 sec
[ Info: VUMPS  47:	obj = -8.862397829104e-01	err = 2.8156580019e-05	time = 0.05 sec
[ Info: VUMPS  48:	obj = -8.862397840755e-01	err = 2.6929849527e-05	time = 0.05 sec
[ Info: VUMPS  49:	obj = -8.862397851542e-01	err = 2.6018468370e-05	time = 0.05 sec
[ Info: VUMPS  50:	obj = -8.862397861566e-01	err = 2.5007719190e-05	time = 0.04 sec
[ Info: VUMPS  51:	obj = -8.862397870909e-01	err = 2.4206500528e-05	time = 0.04 sec
[ Info: VUMPS  52:	obj = -8.862397879640e-01	err = 2.3352551444e-05	time = 0.05 sec
[ Info: VUMPS  53:	obj = -8.862397887818e-01	err = 2.2640472962e-05	time = 0.05 sec
[ Info: VUMPS  54:	obj = -8.862397895495e-01	err = 2.1903812868e-05	time = 0.05 sec
[ Info: VUMPS  55:	obj = -8.862397902713e-01	err = 2.1265562178e-05	time = 0.05 sec
[ Info: VUMPS  56:	obj = -8.862397909510e-01	err = 2.0619412782e-05	time = 0.05 sec
[ Info: VUMPS  57:	obj = -8.862397915922e-01	err = 2.0043203197e-05	time = 0.08 sec
[ Info: VUMPS  58:	obj = -8.862397921977e-01	err = 1.9468679866e-05	time = 0.05 sec
[ Info: VUMPS  59:	obj = -8.862397927702e-01	err = 1.8945450963e-05	time = 0.05 sec
[ Info: VUMPS  60:	obj = -8.862397933121e-01	err = 1.8429260498e-05	time = 0.05 sec
[ Info: VUMPS  61:	obj = -8.862397938257e-01	err = 1.7951673450e-05	time = 0.05 sec
[ Info: VUMPS  62:	obj = -8.862397943128e-01	err = 1.7483983273e-05	time = 0.05 sec
[ Info: VUMPS  63:	obj = -8.862397947753e-01	err = 1.7046410368e-05	time = 0.05 sec
[ Info: VUMPS  64:	obj = -8.862397952148e-01	err = 1.6619867025e-05	time = 0.05 sec
[ Info: VUMPS  65:	obj = -8.862397956328e-01	err = 1.6217300082e-05	time = 0.05 sec
[ Info: VUMPS  66:	obj = -8.862397960307e-01	err = 1.5826070723e-05	time = 0.05 sec
[ Info: VUMPS  67:	obj = -8.862397964096e-01	err = 1.5454492741e-05	time = 0.05 sec
[ Info: VUMPS  68:	obj = -8.862397967709e-01	err = 1.5094031636e-05	time = 0.05 sec
[ Info: VUMPS  69:	obj = -8.862397971156e-01	err = 1.4750023746e-05	time = 0.05 sec
[ Info: VUMPS  70:	obj = -8.862397974446e-01	err = 1.4416625729e-05	time = 0.05 sec
[ Info: VUMPS  71:	obj = -8.862397977589e-01	err = 1.4097258273e-05	time = 0.05 sec
[ Info: VUMPS  72:	obj = -8.862397980593e-01	err = 1.3787858525e-05	time = 0.05 sec
[ Info: VUMPS  73:	obj = -8.862397983466e-01	err = 1.3490605980e-05	time = 0.05 sec
[ Info: VUMPS  74:	obj = -8.862397986216e-01	err = 1.3202629073e-05	time = 0.05 sec
[ Info: VUMPS  75:	obj = -8.862397988850e-01	err = 1.2925362961e-05	time = 0.05 sec
[ Info: VUMPS  76:	obj = -8.862397991373e-01	err = 1.2656609570e-05	time = 0.05 sec
[ Info: VUMPS  77:	obj = -8.862397993792e-01	err = 1.2397289179e-05	time = 0.05 sec
[ Info: VUMPS  78:	obj = -8.862397996112e-01	err = 1.2145887647e-05	time = 0.05 sec
[ Info: VUMPS  79:	obj = -8.862397998339e-01	err = 1.1902902411e-05	time = 0.07 sec
[ Info: VUMPS  80:	obj = -8.862398000477e-01	err = 1.1667209784e-05	time = 0.05 sec
[ Info: VUMPS  81:	obj = -8.862398002530e-01	err = 1.1439084229e-05	time = 0.05 sec
[ Info: VUMPS  82:	obj = -8.862398004504e-01	err = 1.1217670254e-05	time = 0.05 sec
[ Info: VUMPS  83:	obj = -8.862398006402e-01	err = 1.1003098257e-05	time = 0.05 sec
[ Info: VUMPS  84:	obj = -8.862398008227e-01	err = 1.0794704732e-05	time = 0.05 sec
[ Info: VUMPS  85:	obj = -8.862398009983e-01	err = 1.0592526990e-05	time = 0.05 sec
[ Info: VUMPS  86:	obj = -8.862398011674e-01	err = 1.0396041209e-05	time = 0.05 sec
[ Info: VUMPS  87:	obj = -8.862398013303e-01	err = 1.0205225393e-05	time = 0.05 sec
[ Info: VUMPS  88:	obj = -8.862398014872e-01	err = 1.0019658714e-05	time = 0.05 sec
[ Info: VUMPS  89:	obj = -8.862398016384e-01	err = 9.8392824719e-06	time = 0.05 sec
[ Info: VUMPS  90:	obj = -8.862398017842e-01	err = 9.6637533355e-06	time = 0.05 sec
[ Info: VUMPS  91:	obj = -8.862398019248e-01	err = 9.4929904995e-06	time = 0.05 sec
[ Info: VUMPS  92:	obj = -8.862398020605e-01	err = 9.3267096532e-06	time = 0.05 sec
[ Info: VUMPS  93:	obj = -8.862398021914e-01	err = 9.1648179316e-06	time = 0.05 sec
[ Info: VUMPS  94:	obj = -8.862398023178e-01	err = 9.0070766407e-06	time = 0.05 sec
[ Info: VUMPS  95:	obj = -8.862398024399e-01	err = 8.8533874515e-06	time = 0.05 sec
[ Info: VUMPS  96:	obj = -8.862398025579e-01	err = 8.7035474415e-06	time = 0.05 sec
[ Info: VUMPS  97:	obj = -8.862398026718e-01	err = 8.5574571340e-06	time = 0.05 sec
[ Info: VUMPS  98:	obj = -8.862398027820e-01	err = 8.4149420539e-06	time = 0.05 sec
[ Info: VUMPS  99:	obj = -8.862398028885e-01	err = 8.2759043511e-06	time = 0.05 sec
┌ Warning: VUMPS cancel 100:	obj = -8.862398029915e-01	err = 8.1401924815e-06	time = 5.27 sec
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
<polyline clip-path="url(#clip362)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="189.496,1325.87 2352.76,1325.87 "/>
<polyline clip-path="url(#clip362)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="189.496,1036.96 2352.76,1036.96 "/>
<polyline clip-path="url(#clip362)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="189.496,748.046 2352.76,748.046 "/>
<polyline clip-path="url(#clip362)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="189.496,459.132 2352.76,459.132 "/>
<polyline clip-path="url(#clip362)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="189.496,170.218 2352.76,170.218 "/>
<polyline clip-path="url(#clip360)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="189.496,1352.62 2352.76,1352.62 "/>
<polyline clip-path="url(#clip360)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="250.72,1352.62 250.72,1371.52 "/>
<path clip-path="url(#clip360)" d="M117.476 1508.55 L138.148 1487.88 L140.931 1490.66 L132.256 1499.34 L153.911 1520.99 L150.588 1524.32 L128.933 1502.66 L120.258 1511.34 L117.476 1508.55 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M155.826 1488.04 Q155.024 1488.26 154.287 1488.73 Q153.551 1489.17 152.831 1489.89 Q150.277 1492.45 150.572 1495.49 Q150.867 1498.5 153.976 1501.61 L163.634 1511.27 L160.606 1514.3 L142.273 1495.97 L145.301 1492.94 L148.149 1495.79 Q147.429 1493.17 148.149 1490.84 Q148.853 1488.5 151.03 1486.33 Q151.341 1486.01 151.767 1485.69 Q152.176 1485.34 152.716 1484.97 L155.826 1488.04 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M156.17 1482.07 L159.182 1479.06 L177.514 1497.39 L174.502 1500.4 L156.17 1482.07 M149.033 1474.93 L152.045 1471.92 L155.859 1475.74 L152.847 1478.75 L149.033 1474.93 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M163.323 1474.92 L166.514 1471.73 L187.629 1481.38 L177.972 1460.27 L181.164 1457.08 L192.622 1482.28 L188.53 1486.37 L163.323 1474.92 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M185.321 1452.92 L188.333 1449.91 L206.665 1468.24 L203.654 1471.25 L185.321 1452.92 M178.185 1445.78 L181.197 1442.77 L185.01 1446.58 L181.999 1449.6 L178.185 1445.78 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M212.083 1444.39 Q208.433 1448.04 207.86 1450.28 Q207.287 1452.53 209.301 1454.54 Q210.905 1456.14 212.902 1456.04 Q214.899 1455.91 216.715 1454.1 Q219.22 1451.59 218.958 1448.32 Q218.696 1445.01 215.75 1442.07 L215.079 1441.4 L212.083 1444.39 M216.846 1437.14 L227.306 1447.6 L224.294 1450.61 L221.511 1447.83 Q222.15 1450.53 221.413 1452.87 Q220.66 1455.19 218.434 1457.42 Q215.619 1460.23 212.378 1460.33 Q209.137 1460.4 206.485 1457.75 Q203.392 1454.65 203.883 1451.02 Q204.39 1447.37 208.499 1443.26 L212.722 1439.04 L212.427 1438.74 Q210.348 1436.66 207.844 1436.91 Q205.34 1437.12 202.868 1439.59 Q201.297 1441.17 200.184 1443.03 Q199.071 1444.9 198.449 1447.03 L195.666 1444.24 Q196.681 1441.95 197.925 1440.09 Q199.152 1438.2 200.626 1436.73 Q204.603 1432.75 208.63 1432.85 Q212.656 1432.95 216.846 1437.14 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M208.04 1415.93 L211.052 1412.91 L236.521 1438.38 L233.509 1441.4 L208.04 1415.93 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M224.621 1399.41 Q226.193 1405.37 228.812 1410.12 Q231.43 1414.86 235.211 1418.64 Q238.992 1422.42 243.772 1425.08 Q248.551 1427.7 254.477 1429.27 L251.858 1431.89 Q245.556 1430.49 240.613 1428.01 Q235.686 1425.5 232.02 1421.84 Q228.37 1418.19 225.882 1413.27 Q223.394 1408.36 222.002 1402.03 L224.621 1399.41 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M229.99 1394.04 L232.609 1391.42 Q238.927 1392.83 243.837 1395.32 Q248.764 1397.79 252.414 1401.44 Q256.081 1405.11 258.569 1410.05 Q261.073 1414.98 262.464 1421.28 L259.845 1423.9 Q258.274 1417.97 255.639 1413.21 Q253.004 1408.41 249.223 1404.63 Q245.442 1400.85 240.678 1398.25 Q235.932 1395.63 229.99 1394.04 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1187.32 1611.28 L1182.58 1599.09 L1172.81 1616.92 L1165.9 1616.92 L1179.71 1591.71 L1173.92 1576.72 Q1172.36 1572.71 1167.46 1572.71 L1165.9 1572.71 L1165.9 1567.68 L1168.13 1567.74 Q1176.34 1567.96 1178.41 1573.28 L1183.12 1585.47 L1192.89 1567.64 L1199.8 1567.64 L1185.98 1592.85 L1191.78 1607.84 Q1193.34 1611.85 1198.24 1611.85 L1199.8 1611.85 L1199.8 1616.88 L1197.57 1616.82 Q1189.36 1616.6 1187.32 1611.28 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1229.3 1573.72 L1270.11 1573.72 L1270.11 1579.07 L1229.3 1579.07 L1229.3 1573.72 M1229.3 1586.71 L1270.11 1586.71 L1270.11 1592.12 L1229.3 1592.12 L1229.3 1586.71 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1304.77 1555.8 L1330.01 1555.8 L1330.01 1561.22 L1310.66 1561.22 L1310.66 1572.86 Q1312.06 1572.39 1313.46 1572.16 Q1314.86 1571.91 1316.26 1571.91 Q1324.22 1571.91 1328.86 1576.27 Q1333.51 1580.63 1333.51 1588.08 Q1333.51 1595.75 1328.74 1600.01 Q1323.96 1604.25 1315.27 1604.25 Q1312.28 1604.25 1309.16 1603.74 Q1306.07 1603.23 1302.76 1602.21 L1302.76 1595.75 Q1305.63 1597.31 1308.68 1598.07 Q1311.74 1598.84 1315.14 1598.84 Q1320.65 1598.84 1323.87 1595.94 Q1327.08 1593.04 1327.08 1588.08 Q1327.08 1583.11 1323.87 1580.22 Q1320.65 1577.32 1315.14 1577.32 Q1312.57 1577.32 1309.99 1577.89 Q1307.44 1578.47 1304.77 1579.68 L1304.77 1555.8 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1359.93 1560.04 Q1354.96 1560.04 1352.45 1564.94 Q1349.97 1569.81 1349.97 1579.61 Q1349.97 1589.38 1352.45 1594.29 Q1354.96 1599.16 1359.93 1599.16 Q1364.92 1599.16 1367.41 1594.29 Q1369.92 1589.38 1369.92 1579.61 Q1369.92 1569.81 1367.41 1564.94 Q1364.92 1560.04 1359.93 1560.04 M1359.93 1554.95 Q1367.92 1554.95 1372.12 1561.28 Q1376.35 1567.58 1376.35 1579.61 Q1376.35 1591.61 1372.12 1597.95 Q1367.92 1604.25 1359.93 1604.25 Q1351.94 1604.25 1347.71 1597.95 Q1343.5 1591.61 1343.5 1579.61 Q1343.5 1567.58 1347.71 1561.28 Q1351.94 1554.95 1359.93 1554.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><polyline clip-path="url(#clip360)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="189.496,1352.62 189.496,123.472 "/>
<polyline clip-path="url(#clip360)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="189.496,1325.87 208.394,1325.87 "/>
<polyline clip-path="url(#clip360)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="189.496,1036.96 208.394,1036.96 "/>
<polyline clip-path="url(#clip360)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="189.496,748.046 208.394,748.046 "/>
<polyline clip-path="url(#clip360)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="189.496,459.132 208.394,459.132 "/>
<polyline clip-path="url(#clip360)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="189.496,170.218 208.394,170.218 "/>
<path clip-path="url(#clip360)" d="M51.6634 1345.67 L59.3023 1345.67 L59.3023 1319.3 L50.9921 1320.97 L50.9921 1316.71 L59.256 1315.04 L63.9319 1315.04 L63.9319 1345.67 L71.5707 1345.67 L71.5707 1349.6 L51.6634 1349.6 L51.6634 1345.67 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M91.0151 1318.12 Q87.404 1318.12 85.5753 1321.68 Q83.7697 1325.23 83.7697 1332.36 Q83.7697 1339.46 85.5753 1343.03 Q87.404 1346.57 91.0151 1346.57 Q94.6493 1346.57 96.4548 1343.03 Q98.2835 1339.46 98.2835 1332.36 Q98.2835 1325.23 96.4548 1321.68 Q94.6493 1318.12 91.0151 1318.12 M91.0151 1314.42 Q96.8252 1314.42 99.8808 1319.02 Q102.959 1323.61 102.959 1332.36 Q102.959 1341.08 99.8808 1345.69 Q96.8252 1350.27 91.0151 1350.27 Q85.2049 1350.27 82.1262 1345.69 Q79.0707 1341.08 79.0707 1332.36 Q79.0707 1323.61 82.1262 1319.02 Q85.2049 1314.42 91.0151 1314.42 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M102.959 1308.52 L127.071 1308.52 L127.071 1311.71 L102.959 1311.71 L102.959 1308.52 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M145.71 1297.42 L136.118 1312.41 L145.71 1312.41 L145.71 1297.42 M144.713 1294.11 L149.49 1294.11 L149.49 1312.41 L153.496 1312.41 L153.496 1315.57 L149.49 1315.57 L149.49 1322.19 L145.71 1322.19 L145.71 1315.57 L133.033 1315.57 L133.033 1311.9 L144.713 1294.11 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M52.585 1056.75 L60.2238 1056.75 L60.2238 1030.39 L51.9137 1032.05 L51.9137 1027.79 L60.1776 1026.13 L64.8535 1026.13 L64.8535 1056.75 L72.4923 1056.75 L72.4923 1060.69 L52.585 1060.69 L52.585 1056.75 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M91.9366 1029.21 Q88.3255 1029.21 86.4969 1032.77 Q84.6913 1036.31 84.6913 1043.44 Q84.6913 1050.55 86.4969 1054.11 Q88.3255 1057.65 91.9366 1057.65 Q95.5709 1057.65 97.3764 1054.11 Q99.2051 1050.55 99.2051 1043.44 Q99.2051 1036.31 97.3764 1032.77 Q95.5709 1029.21 91.9366 1029.21 M91.9366 1025.5 Q97.7468 1025.5 100.802 1030.11 Q103.881 1034.69 103.881 1043.44 Q103.881 1052.17 100.802 1056.78 Q97.7468 1061.36 91.9366 1061.36 Q86.1265 1061.36 83.0478 1056.78 Q79.9923 1052.17 79.9923 1043.44 Q79.9923 1034.69 83.0478 1030.11 Q86.1265 1025.5 91.9366 1025.5 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M103.881 1019.6 L127.993 1019.6 L127.993 1022.8 L103.881 1022.8 L103.881 1019.6 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M147.703 1018.14 Q150.43 1018.72 151.954 1020.56 Q153.496 1022.41 153.496 1025.11 Q153.496 1029.27 150.637 1031.55 Q147.778 1033.82 142.512 1033.82 Q140.744 1033.82 138.863 1033.47 Q137.002 1033.13 135.008 1032.43 L135.008 1028.76 Q136.588 1029.68 138.469 1030.15 Q140.349 1030.63 142.399 1030.63 Q145.973 1030.63 147.835 1029.21 Q149.716 1027.8 149.716 1025.11 Q149.716 1022.63 147.966 1021.24 Q146.236 1019.83 143.133 1019.83 L139.86 1019.83 L139.86 1016.71 L143.283 1016.71 Q146.086 1016.71 147.571 1015.6 Q149.057 1014.47 149.057 1012.36 Q149.057 1010.2 147.515 1009.05 Q145.992 1007.89 143.133 1007.89 Q141.572 1007.89 139.785 1008.23 Q137.998 1008.56 135.854 1009.28 L135.854 1005.89 Q138.017 1005.29 139.898 1004.99 Q141.797 1004.69 143.471 1004.69 Q147.797 1004.69 150.317 1006.66 Q152.838 1008.62 152.838 1011.97 Q152.838 1014.3 151.502 1015.92 Q150.167 1017.52 147.703 1018.14 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M53.3561 767.838 L60.995 767.838 L60.995 741.473 L52.6848 743.139 L52.6848 738.88 L60.9487 737.214 L65.6246 737.214 L65.6246 767.838 L73.2634 767.838 L73.2634 771.774 L53.3561 771.774 L53.3561 767.838 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M92.7078 740.292 Q89.0967 740.292 87.268 743.857 Q85.4624 747.399 85.4624 754.528 Q85.4624 761.635 87.268 765.199 Q89.0967 768.741 92.7078 768.741 Q96.342 768.741 98.1475 765.199 Q99.9762 761.635 99.9762 754.528 Q99.9762 747.399 98.1475 743.857 Q96.342 740.292 92.7078 740.292 M92.7078 736.589 Q98.5179 736.589 101.573 741.195 Q104.652 745.778 104.652 754.528 Q104.652 763.255 101.573 767.862 Q98.5179 772.445 92.7078 772.445 Q86.8976 772.445 83.8189 767.862 Q80.7634 763.255 80.7634 754.528 Q80.7634 745.778 83.8189 741.195 Q86.8976 736.589 92.7078 736.589 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M104.652 730.69 L128.764 730.69 L128.764 733.887 L104.652 733.887 L104.652 730.69 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M140.236 741.166 L153.496 741.166 L153.496 744.363 L135.666 744.363 L135.666 741.166 Q137.829 738.928 141.553 735.166 Q145.296 731.386 146.255 730.295 Q148.079 728.245 148.794 726.834 Q149.527 725.405 149.527 724.032 Q149.527 721.794 147.948 720.383 Q146.387 718.973 143.866 718.973 Q142.08 718.973 140.086 719.593 Q138.111 720.214 135.854 721.474 L135.854 717.637 Q138.149 716.716 140.142 716.246 Q142.136 715.775 143.791 715.775 Q148.155 715.775 150.75 717.957 Q153.345 720.139 153.345 723.787 Q153.345 725.518 152.687 727.079 Q152.048 728.621 150.336 730.727 Q149.866 731.273 147.346 733.887 Q144.826 736.483 140.236 741.166 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M53.0552 478.924 L60.694 478.924 L60.694 452.559 L52.3839 454.226 L52.3839 449.966 L60.6477 448.3 L65.3236 448.3 L65.3236 478.924 L72.9625 478.924 L72.9625 482.86 L53.0552 482.86 L53.0552 478.924 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M92.4068 451.378 Q88.7957 451.378 86.967 454.943 Q85.1615 458.485 85.1615 465.614 Q85.1615 472.721 86.967 476.286 Q88.7957 479.827 92.4068 479.827 Q96.0411 479.827 97.8466 476.286 Q99.6753 472.721 99.6753 465.614 Q99.6753 458.485 97.8466 454.943 Q96.0411 451.378 92.4068 451.378 M92.4068 447.675 Q98.217 447.675 101.273 452.281 Q104.351 456.864 104.351 465.614 Q104.351 474.341 101.273 478.948 Q98.217 483.531 92.4068 483.531 Q86.5967 483.531 83.518 478.948 Q80.4625 474.341 80.4625 465.614 Q80.4625 456.864 83.518 452.281 Q86.5967 447.675 92.4068 447.675 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M104.351 441.776 L128.463 441.776 L128.463 444.973 L104.351 444.973 L104.351 441.776 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M137.321 452.252 L143.528 452.252 L143.528 430.83 L136.776 432.184 L136.776 428.723 L143.49 427.369 L147.289 427.369 L147.289 452.252 L153.496 452.252 L153.496 455.449 L137.321 455.449 L137.321 452.252 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M82.7903 190.011 L90.4291 190.011 L90.4291 163.645 L82.119 165.312 L82.119 161.052 L90.3828 159.386 L95.0587 159.386 L95.0587 190.011 L102.698 190.011 L102.698 193.946 L82.7903 193.946 L82.7903 190.011 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M122.142 162.464 Q118.531 162.464 116.702 166.029 Q114.897 169.571 114.897 176.701 Q114.897 183.807 116.702 187.372 Q118.531 190.913 122.142 190.913 Q125.776 190.913 127.582 187.372 Q129.41 183.807 129.41 176.701 Q129.41 169.571 127.582 166.029 Q125.776 162.464 122.142 162.464 M122.142 158.761 Q127.952 158.761 131.008 163.367 Q134.086 167.951 134.086 176.701 Q134.086 185.427 131.008 190.034 Q127.952 194.617 122.142 194.617 Q116.332 194.617 113.253 190.034 Q110.198 185.427 110.198 176.701 Q110.198 167.951 113.253 163.367 Q116.332 158.761 122.142 158.761 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M143.791 140.957 Q140.857 140.957 139.371 143.853 Q137.904 146.731 137.904 152.524 Q137.904 158.298 139.371 161.194 Q140.857 164.072 143.791 164.072 Q146.744 164.072 148.211 161.194 Q149.697 158.298 149.697 152.524 Q149.697 146.731 148.211 143.853 Q146.744 140.957 143.791 140.957 M143.791 137.948 Q148.512 137.948 150.994 141.69 Q153.496 145.414 153.496 152.524 Q153.496 159.614 150.994 163.357 Q148.512 167.081 143.791 167.081 Q139.07 167.081 136.569 163.357 Q134.086 159.614 134.086 152.524 Q134.086 145.414 136.569 141.69 Q139.07 137.948 143.791 137.948 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M772.196 12.096 L810.437 12.096 L810.437 18.9825 L780.379 18.9825 L780.379 36.8875 L809.181 36.8875 L809.181 43.7741 L780.379 43.7741 L780.379 65.6895 L811.166 65.6895 L811.166 72.576 L772.196 72.576 L772.196 12.096 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M862.005 45.1919 L862.005 72.576 L854.551 72.576 L854.551 45.4349 Q854.551 38.994 852.04 35.7938 Q849.528 32.5936 844.505 32.5936 Q838.469 32.5936 834.985 36.4419 Q831.502 40.2903 831.502 46.9338 L831.502 72.576 L824.007 72.576 L824.007 27.2059 L831.502 27.2059 L831.502 34.2544 Q834.175 30.163 837.78 28.1376 Q841.426 26.1121 846.166 26.1121 Q853.984 26.1121 857.994 30.9732 Q862.005 35.7938 862.005 45.1919 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M884.244 14.324 L884.244 27.2059 L899.597 27.2059 L899.597 32.9987 L884.244 32.9987 L884.244 57.6282 Q884.244 63.1779 885.743 64.7578 Q887.282 66.3376 891.941 66.3376 L899.597 66.3376 L899.597 72.576 L891.941 72.576 Q883.313 72.576 880.031 69.3758 Q876.75 66.1351 876.75 57.6282 L876.75 32.9987 L871.281 32.9987 L871.281 27.2059 L876.75 27.2059 L876.75 14.324 L884.244 14.324 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M930.02 49.7694 Q920.986 49.7694 917.502 51.8354 Q914.018 53.9013 914.018 58.8839 Q914.018 62.8538 916.611 65.2034 Q919.244 67.5124 923.741 67.5124 Q929.939 67.5124 933.665 63.1374 Q937.433 58.7219 937.433 51.4303 L937.433 49.7694 L930.02 49.7694 M944.886 46.6907 L944.886 72.576 L937.433 72.576 L937.433 65.6895 Q934.881 69.8214 931.073 71.8063 Q927.265 73.7508 921.756 73.7508 Q914.788 73.7508 910.656 69.8619 Q906.565 65.9325 906.565 59.3701 Q906.565 51.7138 911.669 47.825 Q916.814 43.9361 926.981 43.9361 L937.433 43.9361 L937.433 43.2069 Q937.433 38.0623 934.03 35.2672 Q930.668 32.4315 924.551 32.4315 Q920.662 32.4315 916.976 33.3632 Q913.289 34.295 909.887 36.1584 L909.887 29.2718 Q913.978 27.692 917.826 26.9223 Q921.675 26.1121 925.32 26.1121 Q935.164 26.1121 940.025 31.2163 Q944.886 36.3204 944.886 46.6907 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M997.953 45.1919 L997.953 72.576 L990.5 72.576 L990.5 45.4349 Q990.5 38.994 987.988 35.7938 Q985.476 32.5936 980.453 32.5936 Q974.417 32.5936 970.934 36.4419 Q967.45 40.2903 967.45 46.9338 L967.45 72.576 L959.956 72.576 L959.956 27.2059 L967.45 27.2059 L967.45 34.2544 Q970.123 30.163 973.729 28.1376 Q977.375 26.1121 982.114 26.1121 Q989.932 26.1121 993.943 30.9732 Q997.953 35.7938 997.953 45.1919 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1042.68 49.3643 Q1042.68 41.2625 1039.31 36.8065 Q1035.99 32.3505 1029.96 32.3505 Q1023.96 32.3505 1020.6 36.8065 Q1017.28 41.2625 1017.28 49.3643 Q1017.28 57.4256 1020.6 61.8816 Q1023.96 66.3376 1029.96 66.3376 Q1035.99 66.3376 1039.31 61.8816 Q1042.68 57.4256 1042.68 49.3643 M1050.13 66.9452 Q1050.13 78.5308 1044.98 84.1616 Q1039.84 89.8329 1029.23 89.8329 Q1025.3 89.8329 1021.81 89.2252 Q1018.33 88.6581 1015.05 87.4428 L1015.05 80.1917 Q1018.33 81.9741 1021.53 82.8248 Q1024.73 83.6755 1028.05 83.6755 Q1035.38 83.6755 1039.03 79.8271 Q1042.68 76.0193 1042.68 68.282 L1042.68 64.5957 Q1040.37 68.6061 1036.76 70.5911 Q1033.16 72.576 1028.13 72.576 Q1019.79 72.576 1014.68 66.2161 Q1009.58 59.8562 1009.58 49.3643 Q1009.58 38.832 1014.68 32.472 Q1019.79 26.1121 1028.13 26.1121 Q1033.16 26.1121 1036.76 28.0971 Q1040.37 30.082 1042.68 34.0924 L1042.68 27.2059 L1050.13 27.2059 L1050.13 66.9452 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1065.48 9.54393 L1072.94 9.54393 L1072.94 72.576 L1065.48 72.576 L1065.48 9.54393 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1127.34 48.0275 L1127.34 51.6733 L1093.07 51.6733 Q1093.55 59.3701 1097.69 63.421 Q1101.86 67.4314 1109.27 67.4314 Q1113.57 67.4314 1117.58 66.3781 Q1121.63 65.3249 1125.6 63.2184 L1125.6 70.267 Q1121.59 71.9684 1117.37 72.8596 Q1113.16 73.7508 1108.83 73.7508 Q1097.97 73.7508 1091.61 67.4314 Q1085.29 61.1119 1085.29 50.3365 Q1085.29 39.1965 1091.29 32.6746 Q1097.32 26.1121 1107.53 26.1121 Q1116.69 26.1121 1121.99 32.0264 Q1127.34 37.9003 1127.34 48.0275 M1119.89 45.84 Q1119.8 39.7232 1116.44 36.0774 Q1113.12 32.4315 1107.61 32.4315 Q1101.37 32.4315 1097.61 35.9558 Q1093.88 39.4801 1093.31 45.8805 L1119.89 45.84 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1174.9 35.9153 Q1177.69 30.8922 1181.58 28.5022 Q1185.47 26.1121 1190.74 26.1121 Q1197.82 26.1121 1201.67 31.0947 Q1205.52 36.0368 1205.52 45.1919 L1205.52 72.576 L1198.03 72.576 L1198.03 45.4349 Q1198.03 38.913 1195.72 35.7533 Q1193.41 32.5936 1188.67 32.5936 Q1182.88 32.5936 1179.51 36.4419 Q1176.15 40.2903 1176.15 46.9338 L1176.15 72.576 L1168.66 72.576 L1168.66 45.4349 Q1168.66 38.8725 1166.35 35.7533 Q1164.04 32.5936 1159.22 32.5936 Q1153.51 32.5936 1150.15 36.4824 Q1146.78 40.3308 1146.78 46.9338 L1146.78 72.576 L1139.29 72.576 L1139.29 27.2059 L1146.78 27.2059 L1146.78 34.2544 Q1149.34 30.082 1152.9 28.0971 Q1156.47 26.1121 1161.37 26.1121 Q1166.31 26.1121 1169.75 28.6237 Q1173.24 31.1352 1174.9 35.9153 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1259.2 48.0275 L1259.2 51.6733 L1224.93 51.6733 Q1225.41 59.3701 1229.54 63.421 Q1233.72 67.4314 1241.13 67.4314 Q1245.42 67.4314 1249.43 66.3781 Q1253.48 65.3249 1257.45 63.2184 L1257.45 70.267 Q1253.44 71.9684 1249.23 72.8596 Q1245.02 73.7508 1240.68 73.7508 Q1229.83 73.7508 1223.47 67.4314 Q1217.15 61.1119 1217.15 50.3365 Q1217.15 39.1965 1223.14 32.6746 Q1229.18 26.1121 1239.39 26.1121 Q1248.54 26.1121 1253.85 32.0264 Q1259.2 37.9003 1259.2 48.0275 M1251.74 45.84 Q1251.66 39.7232 1248.3 36.0774 Q1244.98 32.4315 1239.47 32.4315 Q1233.23 32.4315 1229.46 35.9558 Q1225.74 39.4801 1225.17 45.8805 L1251.74 45.84 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1309.14 45.1919 L1309.14 72.576 L1301.69 72.576 L1301.69 45.4349 Q1301.69 38.994 1299.18 35.7938 Q1296.67 32.5936 1291.64 32.5936 Q1285.61 32.5936 1282.12 36.4419 Q1278.64 40.2903 1278.64 46.9338 L1278.64 72.576 L1271.15 72.576 L1271.15 27.2059 L1278.64 27.2059 L1278.64 34.2544 Q1281.31 30.163 1284.92 28.1376 Q1288.57 26.1121 1293.3 26.1121 Q1301.12 26.1121 1305.13 30.9732 Q1309.14 35.7938 1309.14 45.1919 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1331.38 14.324 L1331.38 27.2059 L1346.74 27.2059 L1346.74 32.9987 L1331.38 32.9987 L1331.38 57.6282 Q1331.38 63.1779 1332.88 64.7578 Q1334.42 66.3376 1339.08 66.3376 L1346.74 66.3376 L1346.74 72.576 L1339.08 72.576 Q1330.45 72.576 1327.17 69.3758 Q1323.89 66.1351 1323.89 57.6282 L1323.89 32.9987 L1318.42 32.9987 L1318.42 27.2059 L1323.89 27.2059 L1323.89 14.324 L1331.38 14.324 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1419.49 14.0809 L1419.49 22.0612 Q1414.83 19.8332 1410.7 18.7395 Q1406.57 17.6457 1402.72 17.6457 Q1396.04 17.6457 1392.39 20.2383 Q1388.78 22.8309 1388.78 27.611 Q1388.78 31.6214 1391.17 33.6873 Q1393.61 35.7128 1400.33 36.9686 L1405.27 37.9813 Q1414.43 39.7232 1418.76 44.1387 Q1423.14 48.5136 1423.14 55.8863 Q1423.14 64.6767 1417.22 69.2137 Q1411.35 73.7508 1399.96 73.7508 Q1395.67 73.7508 1390.81 72.7785 Q1385.99 71.8063 1380.8 69.9024 L1380.8 61.4765 Q1385.79 64.2716 1390.57 65.6895 Q1395.35 67.1073 1399.96 67.1073 Q1406.97 67.1073 1410.78 64.3527 Q1414.59 61.598 1414.59 56.4939 Q1414.59 52.0379 1411.83 49.5264 Q1409.12 47.0148 1402.88 45.759 L1397.9 44.7868 Q1388.74 42.9639 1384.65 39.075 Q1380.56 35.1862 1380.56 28.2591 Q1380.56 20.2383 1386.19 15.6203 Q1391.86 11.0023 1401.79 11.0023 Q1406.04 11.0023 1410.46 11.7719 Q1414.87 12.5416 1419.49 14.0809 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1442.78 65.7705 L1442.78 89.8329 L1435.29 89.8329 L1435.29 27.2059 L1442.78 27.2059 L1442.78 34.0924 Q1445.13 30.0415 1448.7 28.0971 Q1452.3 26.1121 1457.29 26.1121 Q1465.55 26.1121 1470.69 32.6746 Q1475.88 39.2371 1475.88 49.9314 Q1475.88 60.6258 1470.69 67.1883 Q1465.55 73.7508 1457.29 73.7508 Q1452.3 73.7508 1448.7 71.8063 Q1445.13 69.8214 1442.78 65.7705 M1468.14 49.9314 Q1468.14 41.7081 1464.74 37.0496 Q1461.38 32.3505 1455.46 32.3505 Q1449.55 32.3505 1446.15 37.0496 Q1442.78 41.7081 1442.78 49.9314 Q1442.78 58.1548 1446.15 62.8538 Q1449.55 67.5124 1455.46 67.5124 Q1461.38 67.5124 1464.74 62.8538 Q1468.14 58.1548 1468.14 49.9314 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1527.04 48.0275 L1527.04 51.6733 L1492.77 51.6733 Q1493.26 59.3701 1497.39 63.421 Q1501.56 67.4314 1508.97 67.4314 Q1513.27 67.4314 1517.28 66.3781 Q1521.33 65.3249 1525.3 63.2184 L1525.3 70.267 Q1521.29 71.9684 1517.08 72.8596 Q1512.86 73.7508 1508.53 73.7508 Q1497.67 73.7508 1491.31 67.4314 Q1484.99 61.1119 1484.99 50.3365 Q1484.99 39.1965 1490.99 32.6746 Q1497.02 26.1121 1507.23 26.1121 Q1516.39 26.1121 1521.69 32.0264 Q1527.04 37.9003 1527.04 48.0275 M1519.59 45.84 Q1519.51 39.7232 1516.14 36.0774 Q1512.82 32.4315 1507.31 32.4315 Q1501.08 32.4315 1497.31 35.9558 Q1493.58 39.4801 1493.01 45.8805 L1519.59 45.84 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1571.93 28.9478 L1571.93 35.9153 Q1568.77 34.1734 1565.57 33.3227 Q1562.41 32.4315 1559.17 32.4315 Q1551.91 32.4315 1547.9 37.0496 Q1543.89 41.6271 1543.89 49.9314 Q1543.89 58.2358 1547.9 62.8538 Q1551.91 67.4314 1559.17 67.4314 Q1562.41 67.4314 1565.57 66.5807 Q1568.77 65.6895 1571.93 63.9476 L1571.93 70.8341 Q1568.81 72.2924 1565.44 73.0216 Q1562.12 73.7508 1558.36 73.7508 Q1548.11 73.7508 1542.07 67.3098 Q1536.03 60.8689 1536.03 49.9314 Q1536.03 38.832 1542.11 32.472 Q1548.23 26.1121 1558.84 26.1121 Q1562.28 26.1121 1565.57 26.8413 Q1568.85 27.5299 1571.93 28.9478 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1592.26 14.324 L1592.26 27.2059 L1607.61 27.2059 L1607.61 32.9987 L1592.26 32.9987 L1592.26 57.6282 Q1592.26 63.1779 1593.76 64.7578 Q1595.3 66.3376 1599.96 66.3376 L1607.61 66.3376 L1607.61 72.576 L1599.96 72.576 Q1591.33 72.576 1588.05 69.3758 Q1584.77 66.1351 1584.77 57.6282 L1584.77 32.9987 L1579.3 32.9987 L1579.3 27.2059 L1584.77 27.2059 L1584.77 14.324 L1592.26 14.324 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1643.71 34.1734 Q1642.45 33.4443 1640.95 33.1202 Q1639.49 32.7556 1637.71 32.7556 Q1631.39 32.7556 1627.99 36.8875 Q1624.63 40.9789 1624.63 48.6757 L1624.63 72.576 L1617.13 72.576 L1617.13 27.2059 L1624.63 27.2059 L1624.63 34.2544 Q1626.98 30.1225 1630.74 28.1376 Q1634.51 26.1121 1639.9 26.1121 Q1640.67 26.1121 1641.6 26.2337 Q1642.53 26.3147 1643.67 26.5172 L1643.71 34.1734 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1650.76 54.671 L1650.76 27.2059 L1658.21 27.2059 L1658.21 54.3874 Q1658.21 60.8284 1660.72 64.0691 Q1663.23 67.2693 1668.26 67.2693 Q1674.29 67.2693 1677.78 63.421 Q1681.3 59.5726 1681.3 52.9291 L1681.3 27.2059 L1688.75 27.2059 L1688.75 72.576 L1681.3 72.576 L1681.3 65.6084 Q1678.59 69.7404 1674.98 71.7658 Q1671.42 73.7508 1666.68 73.7508 Q1658.86 73.7508 1654.81 68.8897 Q1650.76 64.0286 1650.76 54.671 M1669.51 26.1121 L1669.51 26.1121 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip360)" d="M1739.43 35.9153 Q1742.23 30.8922 1746.11 28.5022 Q1750 26.1121 1755.27 26.1121 Q1762.36 26.1121 1766.21 31.0947 Q1770.06 36.0368 1770.06 45.1919 L1770.06 72.576 L1762.56 72.576 L1762.56 45.4349 Q1762.56 38.913 1760.25 35.7533 Q1757.94 32.5936 1753.2 32.5936 Q1747.41 32.5936 1744.05 36.4419 Q1740.69 40.2903 1740.69 46.9338 L1740.69 72.576 L1733.19 72.576 L1733.19 45.4349 Q1733.19 38.8725 1730.88 35.7533 Q1728.57 32.5936 1723.75 32.5936 Q1718.04 32.5936 1714.68 36.4824 Q1711.32 40.3308 1711.32 46.9338 L1711.32 72.576 L1703.82 72.576 L1703.82 27.2059 L1711.32 27.2059 L1711.32 34.2544 Q1713.87 30.082 1717.43 28.0971 Q1721 26.1121 1725.9 26.1121 Q1730.84 26.1121 1734.29 28.6237 Q1737.77 31.1352 1739.43 35.9153 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><circle clip-path="url(#clip362)" cx="454.801" cy="195.575" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="488.121" cy="269.52" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="521.44" cy="337.631" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="554.759" cy="383.074" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="588.079" cy="446.673" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="621.398" cy="513.726" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="654.718" cy="534.039" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="688.037" cy="570.787" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="721.356" cy="590.556" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="754.676" cy="640.279" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="787.995" cy="665.809" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="821.314" cy="683.697" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="854.634" cy="737.851" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="887.953" cy="740.507" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="921.273" cy="752.389" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="954.592" cy="771.064" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="987.911" cy="806.237" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1021.23" cy="813.098" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1054.55" cy="813.693" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1087.87" cy="855.412" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1121.19" cy="856.608" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1154.51" cy="888.723" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1187.83" cy="915.743" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1221.15" cy="937.48" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1254.47" cy="947.163" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1287.79" cy="985.469" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1321.1" cy="1000.85" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1354.42" cy="1014.64" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1387.74" cy="1020.73" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1421.06" cy="1039.07" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1454.38" cy="1055.6" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1487.7" cy="1062.82" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1521.02" cy="1077.15" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1554.34" cy="1092.2" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1587.66" cy="1106.13" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1620.98" cy="1115.85" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1654.3" cy="1126.36" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1687.62" cy="1128.88" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1720.94" cy="1131.42" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1754.26" cy="1154.63" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1787.58" cy="1163.79" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1820.9" cy="1164.77" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1854.21" cy="1206.62" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1887.53" cy="1209.76" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1920.85" cy="1219.11" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1954.17" cy="1224.07" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="1987.49" cy="1238.11" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="2020.81" cy="1265.79" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip362)" cx="2054.13" cy="1293.74" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
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
[ Info: VUMPS init:	obj = +2.385218766018e-02	err = 4.1022e-01
[ Info: VUMPS   1:	obj = -8.767148354354e-01	err = 1.1240525621e-01	time = 0.02 sec
[ Info: VUMPS   2:	obj = -8.857670521712e-01	err = 7.1097024400e-03	time = 0.02 sec
[ Info: VUMPS   3:	obj = -8.861235142686e-01	err = 3.6997098906e-03	time = 0.02 sec
[ Info: VUMPS   4:	obj = -8.862242715172e-01	err = 1.9276860974e-03	time = 0.02 sec
[ Info: VUMPS   5:	obj = -8.862605893411e-01	err = 1.0771519045e-03	time = 0.02 sec
[ Info: VUMPS   6:	obj = -8.862751630730e-01	err = 7.2395911133e-04	time = 0.02 sec
[ Info: VUMPS   7:	obj = -8.862816545693e-01	err = 6.0213934261e-04	time = 0.03 sec
[ Info: VUMPS   8:	obj = -8.862846937789e-01	err = 5.3704262252e-04	time = 0.03 sec
[ Info: VUMPS   9:	obj = -8.862862391179e-01	err = 4.2756371588e-04	time = 0.03 sec
[ Info: VUMPS  10:	obj = -8.862870493397e-01	err = 3.2013184158e-04	time = 0.03 sec
[ Info: VUMPS  11:	obj = -8.862874612254e-01	err = 2.4852297759e-04	time = 0.03 sec
[ Info: VUMPS  12:	obj = -8.862876706904e-01	err = 1.9645117688e-04	time = 0.03 sec
[ Info: VUMPS  13:	obj = -8.862877789677e-01	err = 1.5341375579e-04	time = 0.03 sec
[ Info: VUMPS  14:	obj = -8.862878354822e-01	err = 1.1786988555e-04	time = 0.03 sec
[ Info: VUMPS  15:	obj = -8.862878651060e-01	err = 8.9469567537e-05	time = 0.03 sec
[ Info: VUMPS  16:	obj = -8.862878806736e-01	err = 6.7354109300e-05	time = 0.03 sec
[ Info: VUMPS  17:	obj = -8.862878888740e-01	err = 5.0403433985e-05	time = 0.03 sec
[ Info: VUMPS  18:	obj = -8.862878932036e-01	err = 3.7546601684e-05	time = 0.03 sec
[ Info: VUMPS  19:	obj = -8.862878954945e-01	err = 2.7870366278e-05	time = 0.03 sec
[ Info: VUMPS  20:	obj = -8.862878967088e-01	err = 2.0628099471e-05	time = 0.03 sec
[ Info: VUMPS  21:	obj = -8.862878973536e-01	err = 1.5233566626e-05	time = 0.03 sec
[ Info: VUMPS  22:	obj = -8.862878976964e-01	err = 1.1230235875e-05	time = 0.03 sec
[ Info: VUMPS  23:	obj = -8.862878978789e-01	err = 8.2677802059e-06	time = 0.04 sec
[ Info: VUMPS  24:	obj = -8.862878979762e-01	err = 6.0803840271e-06	time = 0.04 sec
[ Info: VUMPS  25:	obj = -8.862878980281e-01	err = 4.4683850443e-06	time = 0.03 sec
[ Info: VUMPS  26:	obj = -8.862878980558e-01	err = 3.2813077496e-06	time = 0.03 sec
[ Info: VUMPS  27:	obj = -8.862878980706e-01	err = 2.4083290994e-06	time = 0.03 sec
[ Info: VUMPS  28:	obj = -8.862878980785e-01	err = 1.7667392001e-06	time = 0.03 sec
[ Info: VUMPS  29:	obj = -8.862878980828e-01	err = 1.2956799322e-06	time = 0.03 sec
[ Info: VUMPS  30:	obj = -8.862878980850e-01	err = 9.4994822120e-07	time = 0.12 sec
[ Info: VUMPS  31:	obj = -8.862878980862e-01	err = 6.9629729807e-07	time = 0.03 sec
[ Info: VUMPS  32:	obj = -8.862878980869e-01	err = 5.1026702719e-07	time = 0.03 sec
[ Info: VUMPS  33:	obj = -8.862878980872e-01	err = 3.7387002510e-07	time = 0.03 sec
[ Info: VUMPS  34:	obj = -8.862878980874e-01	err = 2.7388892106e-07	time = 0.03 sec
[ Info: VUMPS  35:	obj = -8.862878980875e-01	err = 2.0061693329e-07	time = 0.03 sec
[ Info: VUMPS  36:	obj = -8.862878980876e-01	err = 1.4692892393e-07	time = 0.03 sec
[ Info: VUMPS  37:	obj = -8.862878980876e-01	err = 1.0759691882e-07	time = 0.03 sec
[ Info: VUMPS  38:	obj = -8.862878980876e-01	err = 7.8786274215e-08	time = 0.03 sec
[ Info: VUMPS  39:	obj = -8.862878980877e-01	err = 5.7685171951e-08	time = 0.03 sec
[ Info: VUMPS  40:	obj = -8.862878980877e-01	err = 4.2232300444e-08	time = 0.03 sec
[ Info: VUMPS  41:	obj = -8.862878980877e-01	err = 3.0916895851e-08	time = 0.03 sec
[ Info: VUMPS  42:	obj = -8.862878980877e-01	err = 2.2631886378e-08	time = 0.03 sec
[ Info: VUMPS  43:	obj = -8.862878980877e-01	err = 1.6566176024e-08	time = 0.03 sec
[ Info: VUMPS  44:	obj = -8.862878980877e-01	err = 1.2125589768e-08	time = 0.03 sec
[ Info: VUMPS  45:	obj = -8.862878980877e-01	err = 8.8749256987e-09	time = 0.03 sec
[ Info: VUMPS  46:	obj = -8.862878980877e-01	err = 6.4954639093e-09	time = 0.03 sec
[ Info: VUMPS  47:	obj = -8.862878980877e-01	err = 4.7537965115e-09	time = 0.03 sec
[ Info: VUMPS  48:	obj = -8.862878980877e-01	err = 3.4790265634e-09	time = 0.03 sec
[ Info: VUMPS  49:	obj = -8.862878980877e-01	err = 2.5459749625e-09	time = 0.03 sec
[ Info: VUMPS  50:	obj = -8.862878980877e-01	err = 1.8631464725e-09	time = 0.03 sec
[ Info: VUMPS  51:	obj = -8.862878980877e-01	err = 1.3634323517e-09	time = 0.03 sec
[ Info: VUMPS  52:	obj = -8.862878980877e-01	err = 9.9772830752e-10	time = 0.03 sec
[ Info: VUMPS  53:	obj = -8.862878980878e-01	err = 7.3010148214e-10	time = 0.03 sec
[ Info: VUMPS  54:	obj = -8.862878980878e-01	err = 5.3425508212e-10	time = 0.03 sec
[ Info: VUMPS  55:	obj = -8.862878980878e-01	err = 3.9093719198e-10	time = 0.03 sec
[ Info: VUMPS  56:	obj = -8.862878980878e-01	err = 2.8606551842e-10	time = 0.03 sec
[ Info: VUMPS  57:	obj = -8.862878980878e-01	err = 2.0932271119e-10	time = 0.03 sec
[ Info: VUMPS  58:	obj = -8.862878980878e-01	err = 1.5316595582e-10	time = 0.03 sec
[ Info: VUMPS  59:	obj = -8.862878980878e-01	err = 1.1207198643e-10	time = 0.03 sec
[ Info: VUMPS  60:	obj = -8.862878980878e-01	err = 8.2001568087e-11	time = 0.03 sec
[ Info: VUMPS  61:	obj = -8.862878980878e-01	err = 6.0001311312e-11	time = 0.03 sec
[ Info: VUMPS  62:	obj = -8.862878980878e-01	err = 4.3902609432e-11	time = 0.03 sec
[ Info: VUMPS  63:	obj = -8.862878980878e-01	err = 3.2123333550e-11	time = 0.03 sec
[ Info: VUMPS  64:	obj = -8.862878980878e-01	err = 2.3507103007e-11	time = 0.03 sec
[ Info: VUMPS  65:	obj = -8.862878980878e-01	err = 1.7194123408e-11	time = 0.10 sec
[ Info: VUMPS  66:	obj = -8.862878980878e-01	err = 1.2580874479e-11	time = 0.03 sec
[ Info: VUMPS  67:	obj = -8.862878980878e-01	err = 9.2031981520e-12	time = 0.02 sec
[ Info: VUMPS  68:	obj = -8.862878980878e-01	err = 6.7348895902e-12	time = 0.03 sec
[ Info: VUMPS  69:	obj = -8.862878980878e-01	err = 4.9269673018e-12	time = 0.02 sec
[ Info: VUMPS  70:	obj = -8.862878980878e-01	err = 3.6031006275e-12	time = 0.02 sec
[ Info: VUMPS  71:	obj = -8.862878980879e-01	err = 2.6363286470e-12	time = 0.02 sec
[ Info: VUMPS  72:	obj = -8.862878980879e-01	err = 1.9254763358e-12	time = 0.02 sec
[ Info: VUMPS  73:	obj = -8.862878980879e-01	err = 1.4046463278e-12	time = 0.02 sec
[ Info: VUMPS  74:	obj = -8.862878980879e-01	err = 1.0273656801e-12	time = 0.02 sec
[ Info: VUMPS conv 75:	obj = -8.862878980879e-01	err = 7.5593842818e-13	time = 2.36 sec

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

