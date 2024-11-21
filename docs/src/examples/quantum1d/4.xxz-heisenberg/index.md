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
H = heisenberg_XXX(; spin=1 // 2);
````

We then need an intial state, which we shall later optimize. In this example we work directly in the thermodynamic limit.

````julia
random_data = TensorMap(rand, ComplexF64, ℂ^20 * ℂ^2, ℂ^20);
state = InfiniteMPS([random_data]);
````

The groundstate can then be found by calling `find_groundstate`.

````julia
groundstate, cache, delta = find_groundstate(state, H, VUMPS());
````

````
[ Info: VUMPS init:	obj = +2.499971249561e-01	err = 4.8102e-03
[ Info: VUMPS   1:	obj = -1.047871149491e-01	err = 3.6332884520e-01	time = 0.08 sec
[ Info: VUMPS   2:	obj = -2.424105097429e-01	err = 3.6011566451e-01	time = 0.02 sec
┌ Warning: ignoring imaginary component 1.475763903743127e-6 from total weight 2.0662736930482595: operator might not be hermitian?
│   α = 0.8739699649778873 + 1.475763903743127e-6im
│   β₁ = 1.345735500203679
│   β₂ = 1.3017908581601034
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.1833380872891541e-6 from total weight 2.0507172924982444: operator might not be hermitian?
│   α = 0.7372884749735164 + 1.1833380872891541e-6im
│   β₁ = 1.3017908581601034
│   β₂ = 1.4025646794468833
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.2416644342636007e-6 from total weight 2.031220380535854: operator might not be hermitian?
│   α = 0.7386571067649788 + 1.2416644342636007e-6im
│   β₁ = 1.4025646794468833
│   β₂ = 1.2700607201611787
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.3004780743588323e-6 from total weight 1.9679978250135914: operator might not be hermitian?
│   α = 0.7393243960693548 + 1.3004780743588323e-6im
│   β₁ = 1.2700607201611787
│   β₂ = 1.3089540265940922
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.6458907476889217e-6 from total weight 2.0349614523247865: operator might not be hermitian?
│   α = 0.7249377970347587 + 1.6458907476889217e-6im
│   β₁ = 1.3089540265940922
│   β₂ = 1.3791927563392283
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.368995952526035e-6 from total weight 2.054876147824496: operator might not be hermitian?
│   α = 0.7217714242687149 + 1.368995952526035e-6im
│   β₁ = 1.3791927563392283
│   β₂ = 1.3414131857361027
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.2494864182130416e-6 from total weight 2.0096836188378915: operator might not be hermitian?
│   α = 0.807454611112027 + 1.2494864182130416e-6im
│   β₁ = 1.2894881464660273
│   β₂ = 1.313036792683791
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.1639985493788058e-6 from total weight 2.0933444093691493: operator might not be hermitian?
│   α = 0.8461206485955095 + 1.1639985493788058e-6im
│   β₁ = 1.313036792683791
│   β₂ = 1.3935942900696372
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.141568236568552e-6 from total weight 2.0729883136407588: operator might not be hermitian?
│   α = 0.5431648307956423 + 1.141568236568552e-6im
│   β₁ = 1.3935942900696372
│   β₂ = 1.435321382047205
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.6521519737508483e-6 from total weight 2.060744558293663: operator might not be hermitian?
│   α = 0.5404024011086792 + 1.6521519737508483e-6im
│   β₁ = 1.435321382047205
│   β₂ = 1.376403251103502
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.6923960462905674e-6 from total weight 1.9964796296703375: operator might not be hermitian?
│   α = 0.6219891554004395 + 1.6923960462905674e-6im
│   β₁ = 1.376403251103502
│   β₂ = 1.3055935403492547
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.5709160999938243e-6 from total weight 2.0472728905311404: operator might not be hermitian?
│   α = 0.8795272048710723 + 1.5709160999938243e-6im
│   β₁ = 1.3055935403492547
│   β₂ = 1.3088864318921027
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.3793371174981167e-6 from total weight 2.008276861479189: operator might not be hermitian?
│   α = 0.7608179355505824 + 1.3793371174981167e-6im
│   β₁ = 1.3088864318921027
│   β₂ = 1.3195257972863235
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.6662444934072845e-6 from total weight 1.9904775101744567: operator might not be hermitian?
│   α = 0.6960250799777928 + 1.6662444934072845e-6im
│   β₁ = 1.3195257972863235
│   β₂ = 1.3177258731789887
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.8182383051540407e-6 from total weight 1.9900059862237078: operator might not be hermitian?
│   α = 0.7703642977709004 + 1.8182383051540407e-6im
│   β₁ = 1.3177258731789887
│   β₂ = 1.2768168220529936
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.3705999932014934e-6 from total weight 1.9641339239991362: operator might not be hermitian?
│   α = 0.6373205365232443 + 1.3705999932014934e-6im
│   β₁ = 1.2768168220529936
│   β₂ = 1.3495863840638673
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.2716514099669375e-6 from total weight 2.021948581163632: operator might not be hermitian?
│   α = 0.6621740011024483 + 1.2716514099669375e-6im
│   β₁ = 1.3495863840638673
│   β₂ = 1.3521901674991565
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.748967636133747e-6 from total weight 2.007126065968527: operator might not be hermitian?
│   α = 0.7855947486826597 + 1.748967636133747e-6im
│   β₁ = 1.3521901674991565
│   β₂ = 1.2581644115329562
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.8063431283139864e-6 from total weight 1.9589927438694472: operator might not be hermitian?
│   α = 0.8791231530779282 + 1.8063431283139864e-6im
│   β₁ = 1.2581644115329562
│   β₂ = 1.2172992096457365
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.5885285506861413e-6 from total weight 1.9829497440587416: operator might not be hermitian?
│   α = 0.9043769872180605 + 1.5885285506861413e-6im
│   β₁ = 1.2172992096457365
│   β₂ = 1.2776441549376552
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.5188320461479582e-6 from total weight 1.9588640484991768: operator might not be hermitian?
│   α = 0.6899280363007593 + 1.5188320461479582e-6im
│   β₁ = 1.2776441549376552
│   β₂ = 1.3148281555321597
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.8580505688672172e-6 from total weight 2.008682608503421: operator might not be hermitian?
│   α = 0.6913519715595543 + 1.8580505688672172e-6im
│   β₁ = 1.3148281555321597
│   β₂ = 1.3520596120516548
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.956068479325515e-6 from total weight 1.9935567596530768: operator might not be hermitian?
│   α = 0.6475058176538288 + 1.956068479325515e-6im
│   β₁ = 1.3520596120516548
│   β₂ = 1.314130729995241
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.9744130875937183e-6 from total weight 2.0021703956141623: operator might not be hermitian?
│   α = 0.782921176999384 + 1.9744130875937183e-6im
│   β₁ = 1.314130729995241
│   β₂ = 1.2918131243171154
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.6117145099628014e-6 from total weight 2.0198779920499415: operator might not be hermitian?
│   α = 0.7898358256992374 + 1.6117145099628014e-6im
│   β₁ = 1.2918131243171154
│   β₂ = 1.3368939086738196
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.3593759494220231e-6 from total weight 2.019513392870969: operator might not be hermitian?
│   α = 0.7955165842907529 + 1.3593759494220231e-6im
│   β₁ = 1.3368939086738196
│   β₂ = 1.2877509017867448
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.0936758011396291e-6 from total weight 1.962446160174074: operator might not be hermitian?
│   α = 0.7681454973098479 + 1.0936758011396291e-6im
│   β₁ = 1.0473109475340552
│   β₂ = 1.4711516596594305
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.4140393918386784e-6 from total weight 2.3305892057232325: operator might not be hermitian?
│   α = 1.2725888691538776 + 1.4140393918386784e-6im
│   β₁ = 1.4711516596594305
│   β₂ = 1.2836963855283718
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.2363655292799668e-6 from total weight 1.9215446561532274: operator might not be hermitian?
│   α = 0.7043198883856858 + 1.2363655292799668e-6im
│   β₁ = 1.2836963855283718
│   β₂ = 1.2443435820525077
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.3621234929601933e-6 from total weight 2.031241267058665: operator might not be hermitian?
│   α = 0.8491909978696255 + 1.3621234929601933e-6im
│   β₁ = 1.2443435820525077
│   β₂ = 1.362506801429688
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.5702297864253478e-6 from total weight 2.0046033526873535: operator might not be hermitian?
│   α = 0.6551477286897311 + 1.5702297864253478e-6im
│   β₁ = 1.362506801429688
│   β₂ = 1.3163552982585813
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.5972965097966335e-6 from total weight 1.9982889168855587: operator might not be hermitian?
│   α = 0.6762387934135368 + 1.5972965097966335e-6im
│   β₁ = 1.3163552982585813
│   β₂ = 1.3427838315880456
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.5781264739826717e-6 from total weight 2.0091768575110542: operator might not be hermitian?
│   α = 0.797943851228216 + 1.5781264739826717e-6im
│   β₁ = 1.3427838315880456
│   β₂ = 1.2637281498282438
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.500389786990644e-6 from total weight 1.9365460830408492: operator might not be hermitian?
│   α = 0.8370715044470002 + 1.500389786990644e-6im
│   β₁ = 1.2637281498282438
│   β₂ = 1.2052025520687897
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.6848371347591429e-6 from total weight 1.9701367571309558: operator might not be hermitian?
│   α = 0.9381977580271958 + 1.6848371347591429e-6im
│   β₁ = 1.2052025520687897
│   β₂ = 1.2444720234361477
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.301224902904019e-6 from total weight 2.032783165297502: operator might not be hermitian?
│   α = 1.0018865864506536 + 1.301224902904019e-6im
│   β₁ = 1.3560745448480025
│   β₂ = 1.135558230044051
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.4331223396387566e-6 from total weight 1.891875931072235: operator might not be hermitian?
│   α = 0.9245962509023617 + 1.4331223396387566e-6im
│   β₁ = 1.135558230044051
│   β₂ = 1.1978413156862229
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.2020958610227395e-6 from total weight 1.8535112515440146: operator might not be hermitian?
│   α = 0.7599300178221852 + 1.2020958610227395e-6im
│   β₁ = 1.1978413156862229
│   β₂ = 1.1929738094554658
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.2029225218155096e-6 from total weight 1.8361265730664569: operator might not be hermitian?
│   α = 0.8126705814608668 + 1.2029225218155096e-6im
│   β₁ = 1.1929738094554658
│   β₂ = 1.1347866796454273
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.72781939736244e-6 from total weight 1.8379544196724218: operator might not be hermitian?
│   α = 0.7746976579831134 + 1.72781939736244e-6im
│   β₁ = 1.1347866796454273
│   β₂ = 1.220728954029185
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.3164896480177823e-6 from total weight 1.8641899207308146: operator might not be hermitian?
│   α = 0.753715042628617 + 1.3164896480177823e-6im
│   β₁ = 1.220728954029185
│   β₂ = 1.1903522654503504
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.1951999317263984e-6 from total weight 1.8262076303378614: operator might not be hermitian?
│   α = 0.7790016654288074 + 1.1951999317263984e-6im
│   β₁ = 1.1903522654503504
│   β₂ = 1.1450992090204066
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.329043586510703e-6 from total weight 1.8656619514160093: operator might not be hermitian?
│   α = 0.8930858144233594 + 1.329043586510703e-6im
│   β₁ = 1.1450992090204066
│   β₂ = 1.1712557562446508
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.137416697809951e-6 from total weight 1.9324138401886977: operator might not be hermitian?
│   α = 0.8059601405207465 + 1.137416697809951e-6im
│   β₁ = 1.1712557562446508
│   β₂ = 1.3087442282994357
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.2041512158535161e-6 from total weight 1.8635788832563664: operator might not be hermitian?
│   α = 0.624787036120966 + 1.2041512158535161e-6im
│   β₁ = 1.3087442282994357
│   β₂ = 1.170365737069267
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.6380061677104263e-6 from total weight 1.79911763127121: operator might not be hermitian?
│   α = 0.8545636636861133 + 1.6380061677104263e-6im
│   β₁ = 1.170365737069267
│   β₂ = 1.0662031876476101
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.2228514773388825e-6 from total weight 1.8124365450754356: operator might not be hermitian?
│   α = 0.9939764795428676 + 1.2228514773388825e-6im
│   β₁ = 1.0662031876476101
│   β₂ = 1.0771015507783512
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.3577038761106164e-6 from total weight 1.7818490802101545: operator might not be hermitian?
│   α = 0.8828101046503334 + 1.3577038761106164e-6im
│   β₁ = 1.0771015507783512
│   β₂ = 1.1115235998763446
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.573225774296333e-6 from total weight 1.7952965109074976: operator might not be hermitian?
│   α = 0.8037556413672008 + 1.573225774296333e-6im
│   β₁ = 1.1115235998763446
│   β₂ = 1.1582666868914475
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.9993628590166297e-6 from total weight 1.7966684663477028: operator might not be hermitian?
│   α = 0.7833344237000951 + 1.9993628590166297e-6im
│   β₁ = 1.1582666868914475
│   β₂ = 1.1281945934315403
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.4402545049987947e-6 from total weight 1.7911396609775052: operator might not be hermitian?
│   α = 0.8380019083518554 + 1.4402545049987947e-6im
│   β₁ = 1.1281945934315403
│   β₂ = 1.110455332768963
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.2109032088564176e-6 from total weight 1.7813199833273619: operator might not be hermitian?
│   α = 0.7777249070445168 + 1.2109032088564176e-6im
│   β₁ = 1.110455332768963
│   β₂ = 1.1554799028488103
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.36909457887531e-6 from total weight 1.7554473671236752: operator might not be hermitian?
│   α = 0.7782114591736479 + 1.36909457887531e-6im
│   β₁ = 1.1554799028488103
│   β₂ = 1.0681051341805705
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.6953973636414654e-6 from total weight 1.8478627263512848: operator might not be hermitian?
│   α = 1.0786054983876303 + 1.6953973636414654e-6im
│   β₁ = 1.0681051341805705
│   β₂ = 1.0537353826367444
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.757283796120085e-6 from total weight 1.8111473736314618: operator might not be hermitian?
│   α = 0.9110031771776171 + 1.757283796120085e-6im
│   β₁ = 1.0537353826367444
│   β₂ = 1.1575706300527562
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.6318890057095276e-6 from total weight 1.8203681556361564: operator might not be hermitian?
│   α = 0.9124447934701678 + 1.6318890057095276e-6im
│   β₁ = 1.1575706300527562
│   β₂ = 1.0682766296048822
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.8390788291272608e-6 from total weight 1.742392537575849: operator might not be hermitian?
│   α = 0.8617339094018543 + 1.8390788291272608e-6im
│   β₁ = 1.0682766296048822
│   β₂ = 1.0733738710363945
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.6564830520246487e-6 from total weight 1.7426059463027106: operator might not be hermitian?
│   α = 0.7671804483853782 + 1.6564830520246487e-6im
│   β₁ = 1.0733738710363945
│   β₂ = 1.138410372703261
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.852045748698336e-6 from total weight 1.8175974775984134: operator might not be hermitian?
│   α = 0.7528916040098729 + 1.852045748698336e-6im
│   β₁ = 1.138410372703261
│   β₂ = 1.2003485520888286
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.6617520472352737e-6 from total weight 1.8926943344910268: operator might not be hermitian?
│   α = 0.8539520758169454 + 1.6617520472352737e-6im
│   β₁ = 1.2003485520888286
│   β₂ = 1.1883690712560016
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.3522051637501753e-6 from total weight 1.8705455442967294: operator might not be hermitian?
│   α = 0.9282228149268801 + 1.3522051637501753e-6im
│   β₁ = 1.1883690712560016
│   β₂ = 1.1068522889789256
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.0122348380516705e-6 from total weight 1.7670985003089827: operator might not be hermitian?
│   α = 0.930628231856587 + 1.0122348380516705e-6im
│   β₁ = 1.1068522889789256
│   β₂ = 1.0156014052012858
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.3856631444933258e-6 from total weight 1.825912721996031: operator might not be hermitian?
│   α = 1.0815656907671525 + 1.3856631444933258e-6im
│   β₁ = 1.0156014052012858
│   β₂ = 1.0642963453162757
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
[ Info: VUMPS   3:	obj = -2.945462988948e-01	err = 3.4609424308e-01	time = 0.03 sec
┌ Warning: ignoring imaginary component -1.392483040988518e-6 from total weight 1.8955122145817562: operator might not be hermitian?
│   α = 1.101941371687926 - 1.392483040988518e-6im
│   β₁ = 1.0798020846906913
│   β₂ = 1.1012353185794865
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0753622344539249e-6 from total weight 1.8648990216541668: operator might not be hermitian?
│   α = 0.8383791538157135 - 1.0753622344539249e-6im
│   β₁ = 1.1012353185794865
│   β₂ = 1.2498998073949357
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.2549910538232206e-6 from total weight 2.1121408373175985: operator might not be hermitian?
│   α = 1.3409655729255554 - 1.2549910538232206e-6im
│   β₁ = 1.2136080492246442
│   β₂ = 1.0909196816212923
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.4022760645906324e-6 from total weight 1.7304296250907174: operator might not be hermitian?
│   α = 0.8127621709548318 - 1.4022760645906324e-6im
│   β₁ = 1.0909196816212923
│   β₂ = 1.0694384456833976
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.4313764805165552e-6 from total weight 1.8794569157231034: operator might not be hermitian?
│   α = 1.0784242798463932 - 1.4313764805165552e-6im
│   β₁ = 1.0694384456833976
│   β₂ = 1.1070956515086754
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1180232882476715e-6 from total weight 1.8844800254598146: operator might not be hermitian?
│   α = 1.216318455706141 - 1.1180232882476715e-6im
│   β₁ = 1.0649907248461274
│   β₂ = 0.9683125201380608
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
[ Info: VUMPS   4:	obj = -2.319707042439e-01	err = 3.6024027466e-01	time = 0.02 sec
┌ Warning: ignoring imaginary component -1.0204260968149637e-6 from total weight 1.7960622649204319: operator might not be hermitian?
│   α = 0.8465484658404078 - 1.0204260968149637e-6im
│   β₁ = 1.0893153937710733
│   β₂ = 1.1500379677848442
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -9.980297221391307e-7 from total weight 1.724221091947597: operator might not be hermitian?
│   α = 0.6903523472065521 - 9.980297221391307e-7im
│   β₁ = 1.1477237120574055
│   β₂ = 1.0858555573387934
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
[ Info: VUMPS   5:	obj = -3.341745290133e-01	err = 3.3118494023e-01	time = 0.08 sec
[ Info: VUMPS   6:	obj = -3.494924855347e-01	err = 2.8628386188e-01	time = 0.03 sec
┌ Warning: ignoring imaginary component -1.2137926916363064e-6 from total weight 1.8486757461171281: operator might not be hermitian?
│   α = 0.771117205144365 - 1.2137926916363064e-6im
│   β₁ = 1.1911622391957852
│   β₂ = 1.1849526531151673
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.8052469317100112e-6 from total weight 1.8549819756331998: operator might not be hermitian?
│   α = 0.8368904023840432 - 1.8052469317100112e-6im
│   β₁ = 1.1849526531151673
│   β₂ = 1.1560535429613934
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.0673666736130414e-6 from total weight 1.879996592383449: operator might not be hermitian?
│   α = 0.882809938116432 - 2.0673666736130414e-6im
│   β₁ = 1.1560535429613934
│   β₂ = 1.1910390448420975
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.194385859464673e-6 from total weight 1.7942145377417382: operator might not be hermitian?
│   α = 0.8003805303951113 - 2.194385859464673e-6im
│   β₁ = 1.1910390448420975
│   β₂ = 1.0770435495675412
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.3709987418962797e-6 from total weight 1.8718062529591089: operator might not be hermitian?
│   α = 0.956423888603269 - 2.3709987418962797e-6im
│   β₁ = 1.0770435495675412
│   β₂ = 1.19536152951948
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.3673814470905746e-6 from total weight 1.8555379367315137: operator might not be hermitian?
│   α = 0.8933821221410543 - 2.3673814470905746e-6im
│   β₁ = 1.19536152951948
│   β₂ = 1.102724005464574
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.334346772072532e-6 from total weight 1.888065121940592: operator might not be hermitian?
│   α = 0.9721742902891686 - 2.334346772072532e-6im
│   β₁ = 1.102724005464574
│   β₂ = 1.1847644583443624
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.177633131844603e-6 from total weight 1.9390501291320332: operator might not be hermitian?
│   α = 1.014147105401859 - 2.177633131844603e-6im
│   β₁ = 1.1847644583443624
│   β₂ = 1.152282183378374
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.9774983512077515e-6 from total weight 1.8305152178308537: operator might not be hermitian?
│   α = 0.7177922188043688 - 1.9774983512077515e-6im
│   β₁ = 1.152282183378374
│   β₂ = 1.2279275480251655
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.9651366731853956e-6 from total weight 1.8439158098506916: operator might not be hermitian?
│   α = 0.6857331180630781 - 1.9651366731853956e-6im
│   β₁ = 1.2279275480251655
│   β₂ = 1.1924720296115392
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.0190197140557142e-6 from total weight 1.8592053567515292: operator might not be hermitian?
│   α = 0.7838374650404871 - 2.0190197140557142e-6im
│   β₁ = 1.1924720296115392
│   β₂ = 1.1917440352536046
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.105103661853916e-6 from total weight 1.8407455207629144: operator might not be hermitian?
│   α = 0.840354593043139 - 2.105103661853916e-6im
│   β₁ = 1.1917440352536046
│   β₂ = 1.1233407250665566
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.06494826259819e-6 from total weight 1.841267305393983: operator might not be hermitian?
│   α = 0.8882808636031334 - 2.06494826259819e-6im
│   β₁ = 1.1233407250665566
│   β₂ = 1.15729339956295
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.2230019045606925e-6 from total weight 1.8228821941757196: operator might not be hermitian?
│   α = 0.7940779018084841 - 2.2230019045606925e-6im
│   β₁ = 1.15729339956295
│   β₂ = 1.1631903399811268
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.145042114523512e-6 from total weight 1.8171564444253046: operator might not be hermitian?
│   α = 0.8544521983500817 - 2.145042114523512e-6im
│   β₁ = 1.1631903399811268
│   β₂ = 1.104063955222304
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.6467638147572156e-6 from total weight 1.8097942535063836: operator might not be hermitian?
│   α = 0.8187040301882722 - 1.6467638147572156e-6im
│   β₁ = 1.104063955222304
│   β₂ = 1.177336712140745
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.5244830939625462e-6 from total weight 1.8186387999784603: operator might not be hermitian?
│   α = 0.6481570111134087 - 1.5244830939625462e-6im
│   β₁ = 1.177336712140745
│   β₂ = 1.2252419516058546
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.9022438640223151e-6 from total weight 1.845672069956571: operator might not be hermitian?
│   α = 0.8341584370695838 - 1.9022438640223151e-6im
│   β₁ = 1.2252419516058546
│   β₂ = 1.0997578150233172
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.097853804325367e-6 from total weight 1.859100484948138: operator might not be hermitian?
│   α = 0.9829316071898353 - 2.097853804325367e-6im
│   β₁ = 1.0997578150233172
│   β₂ = 1.1316504835912666
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.218477325925325e-6 from total weight 1.824760097059155: operator might not be hermitian?
│   α = 0.7981966528552535 - 2.218477325925325e-6im
│   β₁ = 1.1316504835912666
│   β₂ = 1.1882755144211035
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.107775001676873e-6 from total weight 1.7948760632162533: operator might not be hermitian?
│   α = 0.715662975666142 - 2.107775001676873e-6im
│   β₁ = 1.1882755144211035
│   β₂ = 1.1390381422016462
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.8943371334406364e-6 from total weight 1.7786107483977287: operator might not be hermitian?
│   α = 0.7939679645380324 - 1.8943371334406364e-6im
│   β₁ = 1.1390381422016462
│   β₂ = 1.111603875582258
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.940765476975831e-6 from total weight 1.8050001473989414: operator might not be hermitian?
│   α = 0.8874112102188971 - 1.940765476975831e-6im
│   β₁ = 1.111603875582258
│   β₂ = 1.111244212527008
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.199444159377284e-6 from total weight 1.759110189883705: operator might not be hermitian?
│   α = 0.8304008173000483 - 2.199444159377284e-6im
│   β₁ = 1.111244212527008
│   β₂ = 1.081683614972737
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.404082553034842e-6 from total weight 1.7701574894902945: operator might not be hermitian?
│   α = 0.8176565476152748 - 2.404082553034842e-6im
│   β₁ = 1.081683614972737
│   β₂ = 1.1379173365557727
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.1608698665687154e-6 from total weight 1.777911385657947: operator might not be hermitian?
│   α = 0.702221919442864 - 2.1608698665687154e-6im
│   β₁ = 1.1379173365557727
│   β₂ = 1.1717497199774738
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.0444276374407405e-6 from total weight 1.8530092878500277: operator might not be hermitian?
│   α = 0.9004041535488609 - 2.0444276374407405e-6im
│   β₁ = 1.1717497199774738
│   β₂ = 1.1179974842811402
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.2029817202846858e-6 from total weight 1.8209273687022072: operator might not be hermitian?
│   α = 0.8967101105620928 - 2.2029817202846858e-6im
│   β₁ = 0.8788405586960297
│   β₂ = 1.3188353695933623
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.216417890362702e-6 from total weight 2.1106092940068626: operator might not be hermitian?
│   α = 1.2231888468396814 - 2.216417890362702e-6im
│   β₁ = 1.3188353695933623
│   β₂ = 1.104153025997658
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.3737135441750934e-6 from total weight 1.7108761064821731: operator might not be hermitian?
│   α = 0.7689084483760471 - 2.3737135441750934e-6im
│   β₁ = 1.104153025997658
│   β₂ = 1.056751127239579
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.2950472934567256e-6 from total weight 1.805845919611404: operator might not be hermitian?
│   α = 0.9011490916083735 - 2.2950472934567256e-6im
│   β₁ = 1.056751127239579
│   β₂ = 1.1542473110833626
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.9165190242613137e-6 from total weight 1.7806954185448687: operator might not be hermitian?
│   α = 0.6766133971213509 - 1.9165190242613137e-6im
│   β₁ = 1.1542473110833626
│   β₂ = 1.175067499897663
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.7371752475472957e-6 from total weight 1.7997772548877546: operator might not be hermitian?
│   α = 0.682777908206393 - 1.7371752475472957e-6im
│   β₁ = 1.175067499897663
│   β₂ = 1.1799274833471092
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.9827521738555176e-6 from total weight 1.826349907461843: operator might not be hermitian?
│   α = 0.8580873616013737 - 1.9827521738555176e-6im
│   β₁ = 1.1799274833471092
│   β₂ = 1.0986406138424443
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.91092151146776e-6 from total weight 1.7435181739830286: operator might not be hermitian?
│   α = 0.7394903063302251 - 1.91092151146776e-6im
│   β₁ = 1.0986406138424443
│   β₂ = 1.134018743877274
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.9595898916001386e-6 from total weight 1.8251022615462704: operator might not be hermitian?
│   α = 0.9164809096157032 - 1.9595898916001386e-6im
│   β₁ = 1.134018743877274
│   β₂ = 1.0977533857576816
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.05986185874453e-6 from total weight 1.8186394706547202: operator might not be hermitian?
│   α = 0.9455626165039587 - 2.05986185874453e-6im
│   β₁ = 1.0977533857576816
│   β₂ = 1.0992262581228545
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.9751055681822144e-6 from total weight 1.8210053523773804: operator might not be hermitian?
│   α = 0.9358713510025474 - 1.9751055681822144e-6im
│   β₁ = 1.0992262581228545
│   β₂ = 1.1099130331737983
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.982306293563628e-6 from total weight 1.7917057681901003: operator might not be hermitian?
│   α = 0.9121882545888488 - 1.982306293563628e-6im
│   β₁ = 1.1099130331737983
│   β₂ = 1.0706144061905936
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.196445485691187e-6 from total weight 1.7744014664512398: operator might not be hermitian?
│   α = 0.874804800955041 - 2.196445485691187e-6im
│   β₁ = 0.5978282520080367
│   β₂ = 1.4233125115260221
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.180456329423097e-6 from total weight 2.1061953707531647: operator might not be hermitian?
│   α = 1.1540327503490424 - 2.180456329423097e-6im
│   β₁ = 1.4233125115260221
│   β₂ = 1.0384839167905384
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.151285085509602e-6 from total weight 1.804623532007284: operator might not be hermitian?
│   α = 0.8516682302055673 - 2.151285085509602e-6im
│   β₁ = 1.0384839167905384
│   β₂ = 1.2053540859414364
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.9497181986960488e-6 from total weight 1.8272748413277866: operator might not be hermitian?
│   α = 0.7717001680448697 - 1.9497181986960488e-6im
│   β₁ = 1.2053540859414364
│   β₂ = 1.1360166037032604
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.9811992536446915e-6 from total weight 1.8330648060187416: operator might not be hermitian?
│   α = 0.9324764918655245 - 1.9811992536446915e-6im
│   β₁ = 1.1360166037032604
│   β₂ = 1.0954818352164835
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.444192831056429e-6 from total weight 1.680031417900311: operator might not be hermitian?
│   α = 0.8412545867799889 - 1.444192831056429e-6im
│   β₁ = 1.0396167334930557
│   β₂ = 1.0168546271679657
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.012070060116583e-6 from total weight 1.6996488398921563: operator might not be hermitian?
│   α = 0.9563600153628701 - 2.012070060116583e-6im
│   β₁ = 1.0168546271679657
│   β₂ = 0.969633109565124
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.2284985421137937e-6 from total weight 1.625611124450328: operator might not be hermitian?
│   α = 0.9076143355875635 - 2.2284985421137937e-6im
│   β₁ = 0.969633109565124
│   β₂ = 0.9373683260078226
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.3787217767051727e-6 from total weight 1.6320977987504197: operator might not be hermitian?
│   α = 0.9151559473878687 - 2.3787217767051727e-6im
│   β₁ = 0.9373683260078226
│   β₂ = 0.9734338385520672
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.4459831018432132e-6 from total weight 1.7171504373726327: operator might not be hermitian?
│   α = 1.0490080777046167 - 2.4459831018432132e-6im
│   β₁ = 0.9734338385520672
│   β₂ = 0.949006975440799
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.502042695789325e-6 from total weight 1.7461038387840222: operator might not be hermitian?
│   α = 1.063200920663453 - 2.502042695789325e-6im
│   β₁ = 0.949006975440799
│   β₂ = 1.0088945329791073
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.2991596320565827e-6 from total weight 1.7649566358660225: operator might not be hermitian?
│   α = 1.0323010887631134 - 2.2991596320565827e-6im
│   β₁ = 1.0088945329791073
│   β₂ = 1.0156565413295877
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.3680517285112196e-6 from total weight 1.6911743091363975: operator might not be hermitian?
│   α = 0.822688208588965 - 2.3680517285112196e-6im
│   β₁ = 1.0156565413295877
│   β₂ = 1.0731712097239952
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.3240639867236346e-6 from total weight 1.7008433430515881: operator might not be hermitian?
│   α = 0.9047109560948917 - 2.3240639867236346e-6im
│   β₁ = 1.0731712097239952
│   β₂ = 0.96055698328569
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.4438641293214602e-6 from total weight 1.6863390242825544: operator might not be hermitian?
│   α = 1.0018650093118424 - 2.4438641293214602e-6im
│   β₁ = 0.96055698328569
│   β₂ = 0.9577766387785523
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.2852982025176743e-6 from total weight 1.6664847188204572: operator might not be hermitian?
│   α = 0.9481338905918142 - 2.2852982025176743e-6im
│   β₁ = 0.9577766387785523
│   β₂ = 0.980243517590483
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.6503217592163675e-6 from total weight 1.6570209299072616: operator might not be hermitian?
│   α = 0.9553022495142013 - 2.6503217592163675e-6im
│   β₁ = 0.980243517590483
│   β₂ = 0.9339371608618243
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.4977568402820283e-6 from total weight 1.6146874716863424: operator might not be hermitian?
│   α = 0.9937470670638898 - 2.4977568402820283e-6im
│   β₁ = 0.9339371608618243
│   β₂ = 0.8645483083540594
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.7911041787048443e-6 from total weight 1.6526314248172924: operator might not be hermitian?
│   α = 0.9942222833655865 - 1.7911041787048443e-6im
│   β₁ = 0.8645483083540594
│   β₂ = 0.997631645484384
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0910619358400764e-6 from total weight 1.628256128074027: operator might not be hermitian?
│   α = 0.7468594858110634 - 1.0910619358400764e-6im
│   β₁ = 0.997631645484384
│   β₂ = 1.047926537020409
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.4086345025538481e-6 from total weight 1.7208828389475097: operator might not be hermitian?
│   α = 1.0230379305064004 - 1.4086345025538481e-6im
│   β₁ = 1.047926537020409
│   β₂ = 0.9037041059637005
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.5801647672217256e-6 from total weight 1.6935875767437443: operator might not be hermitian?
│   α = 1.0573393121341177 - 1.5801647672217256e-6im
│   β₁ = 0.9037041059637005
│   β₂ = 0.966225309117056
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.8503500605684953e-6 from total weight 1.687885653739837: operator might not be hermitian?
│   α = 0.9077058420558016 - 1.8503500605684953e-6im
│   β₁ = 0.966225309117056
│   β₂ = 1.0447184962547331
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.7245869534127795e-6 from total weight 1.630010231135931: operator might not be hermitian?
│   α = 0.8428816315538215 - 1.7245869534127795e-6im
│   β₁ = 1.0447184962547331
│   β₂ = 0.9246876079937777
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.669866318011401e-6 from total weight 1.5672706436085126: operator might not be hermitian?
│   α = 0.7835317567308676 - 1.669866318011401e-6im
│   β₁ = 0.9246876079937777
│   β₂ = 0.9936639694238888
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.7442226515947817e-6 from total weight 1.6505309294564066: operator might not be hermitian?
│   α = 0.8815928759787162 - 1.7442226515947817e-6im
│   β₁ = 0.9936639694238888
│   β₂ = 0.9796316991510248
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.5551504819709405e-6 from total weight 1.6065777130754422: operator might not be hermitian?
│   α = 0.7930473224308218 - 1.5551504819709405e-6im
│   β₁ = 0.9796316991510248
│   β₂ = 0.9962377359606892
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.5768281077382929e-6 from total weight 1.6432036667533771: operator might not be hermitian?
│   α = 0.8513104308677694 - 1.5768281077382929e-6im
│   β₁ = 0.9962377359606892
│   β₂ = 0.9914127365395857
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.7341376412037535e-6 from total weight 1.6510721319879114: operator might not be hermitian?
│   α = 0.9480572889512263 - 1.7341376412037535e-6im
│   β₁ = 0.9914127365395857
│   β₂ = 0.9188728680931008
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.0380127565208794e-6 from total weight 1.6657878517427898: operator might not be hermitian?
│   α = 1.0551087967110504 - 2.0380127565208794e-6im
│   β₁ = 0.9188728680931008
│   β₂ = 0.9040283438006927
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.9689745632016017e-6 from total weight 1.6316000751641972: operator might not be hermitian?
│   α = 1.0068062496864232 - 1.9689745632016017e-6im
│   β₁ = 0.9040283438006927
│   β₂ = 0.9116977209959855
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.7641602776535814e-6 from total weight 1.643619185973048: operator might not be hermitian?
│   α = 0.9832497995595976 - 1.7641602776535814e-6im
│   β₁ = 0.9116977209959855
│   β₂ = 0.9505320224444467
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.8717770270057499e-6 from total weight 1.6186044448934445: operator might not be hermitian?
│   α = 0.933889625841253 - 1.8717770270057499e-6im
│   β₁ = 0.7782860350350426
│   β₂ = 1.0686446385215385
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.9087480197105267e-6 from total weight 1.872223188302107: operator might not be hermitian?
│   α = 1.1855968939468708 - 1.9087480197105267e-6im
│   β₁ = 1.0686446385215385
│   β₂ = 0.9785593024621279
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.9402894978638036e-6 from total weight 1.565662888890479: operator might not be hermitian?
│   α = 0.8073816698233107 - 1.9402894978638036e-6im
│   β₁ = 0.9785593024621279
│   β₂ = 0.9175275540512514
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.911761391519551e-6 from total weight 1.69861571249875: operator might not be hermitian?
│   α = 0.9952744384587691 - 1.911761391519551e-6im
│   β₁ = 0.9175275540512514
│   β₂ = 1.0260932308768491
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.8291039981729873e-6 from total weight 1.6498451539412835: operator might not be hermitian?
│   α = 0.858745751697278 - 1.8291039981729873e-6im
│   β₁ = 1.0260932308768491
│   β₂ = 0.9652344002732076
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.1171286886112473e-6 from total weight 1.6049927885630448: operator might not be hermitian?
│   α = 0.9039923380658194 - 2.1171286886112473e-6im
│   β₁ = 0.9652344002732076
│   β₂ = 0.9094626196729643
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.7631975034633485e-6 from total weight 1.6635044587627164: operator might not be hermitian?
│   α = 1.0268033716558147 - 1.7631975034633485e-6im
│   β₁ = 0.9094626196729643
│   β₂ = 0.9411693066043798
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.7538638426399522e-6 from total weight 1.598357955260205: operator might not be hermitian?
│   α = 0.8395676881149241 - 1.7538638426399522e-6im
│   β₁ = 0.9411693066043798
│   β₂ = 0.9818729991804948
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.466572661972844e-6 from total weight 1.6275090983582583: operator might not be hermitian?
│   α = 0.8477297716903176 - 1.466572661972844e-6im
│   β₁ = 0.9818729991804948
│   β₂ = 0.9828863173871848
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.2983254894863294e-6 from total weight 1.6025938022231123: operator might not be hermitian?
│   α = 0.8238345469560299 - 1.2983254894863294e-6im
│   β₁ = 0.9828863173871848
│   β₂ = 0.961008855972232
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.753803488737521e-6 from total weight 1.6853450238186074: operator might not be hermitian?
│   α = 1.0058728121272003 - 1.753803488737521e-6im
│   β₁ = 0.961008855972232
│   β₂ = 0.9513515196147673
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.8713013819338098e-6 from total weight 1.6457558464254889: operator might not be hermitian?
│   α = 0.953253090989849 - 1.8713013819338098e-6im
│   β₁ = 0.9513515196147673
│   β₂ = 0.945912858927897
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.1940684603594107e-6 from total weight 1.6685947310681075: operator might not be hermitian?
│   α = 1.0035877974418752 - 2.1940684603594107e-6im
│   β₁ = 0.5519727146275405
│   β₂ = 1.2134026667499564
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.0196296084545873e-6 from total weight 1.9239962959657293: operator might not be hermitian?
│   α = 1.2417813732350762 - 2.0196296084545873e-6im
│   β₁ = 1.2134026667499564
│   β₂ = 0.8290927187572397
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.8201529821404439e-6 from total weight 1.6421902342159629: operator might not be hermitian?
│   α = 1.1015110960833447 - 1.8201529821404439e-6im
│   β₁ = 0.8290927187572397
│   β₂ = 0.8922260555823076
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.8864060040545833e-6 from total weight 1.6289668855724284: operator might not be hermitian?
│   α = 1.054375019166633 - 1.8864060040545833e-6im
│   β₁ = 0.8922260555823076
│   β₂ = 0.8635734473600828
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
[ Info: VUMPS   7:	obj = -2.477828349789e-01	err = 3.6861869835e-01	time = 0.04 sec
┌ Warning: ignoring imaginary component 9.797819969445398e-7 from total weight 1.7826507126092328: operator might not be hermitian?
│   α = 0.37266497932262266 + 9.797819969445398e-7im
│   β₁ = 1.2529923849189937
│   β₂ = 1.2120125658122227
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.0083546393710799e-6 from total weight 1.5121046107716045: operator might not be hermitian?
│   α = 0.42712182311279506 + 1.0083546393710799e-6im
│   β₁ = 1.0452302250562036
│   β₂ = 1.0057440423713626
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 8.626484624303654e-7 from total weight 1.5025688882814423: operator might not be hermitian?
│   α = 0.3906593447423297 + 8.626484624303654e-7im
│   β₁ = 0.8502359245275253
│   β₂ = 1.175668921524637
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
[ Info: VUMPS   8:	obj = +3.021742828810e-02	err = 3.8513237752e-01	time = 0.02 sec
[ Info: VUMPS   9:	obj = -8.745011232470e-02	err = 3.9063780450e-01	time = 0.02 sec
[ Info: VUMPS  10:	obj = -1.499513490072e-01	err = 3.7840293472e-01	time = 0.06 sec
[ Info: VUMPS  11:	obj = -2.958681759723e-01	err = 3.3123568953e-01	time = 0.03 sec
[ Info: VUMPS  12:	obj = -7.976662855182e-03	err = 3.9656031279e-01	time = 0.03 sec
[ Info: VUMPS  13:	obj = -5.967400377513e-02	err = 3.9708070211e-01	time = 0.03 sec
[ Info: VUMPS  14:	obj = -1.005858342981e-01	err = 3.6761593768e-01	time = 0.02 sec
┌ Warning: ignoring imaginary component 1.1228371103039203e-6 from total weight 1.925618999181602: operator might not be hermitian?
│   α = 0.3165350873163584 + 1.1228371103039203e-6im
│   β₁ = 1.326599888107445
│   β₂ = 1.359392071986163
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.0111171559745724e-6 from total weight 1.7301725734241629: operator might not be hermitian?
│   α = 0.3705094109208439 + 1.0111171559745724e-6im
│   β₁ = 1.2507643011136502
│   β₂ = 1.136577570299071
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.0452915227276982e-6 from total weight 1.7398003898725583: operator might not be hermitian?
│   α = 0.32520104482655454 + 1.0452915227276982e-6im
│   β₁ = 1.136577570299071
│   β₂ = 1.276456463705829
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.043670650770212e-6 from total weight 1.741533774893118: operator might not be hermitian?
│   α = 0.230415239863528 + 1.043670650770212e-6im
│   β₁ = 1.225976649377648
│   β₂ = 1.2152489298541975
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 9.49492928604273e-7 from total weight 1.7203306219892969: operator might not be hermitian?
│   α = 0.28756839381882887 + 9.49492928604273e-7im
│   β₁ = 1.2152489298541975
│   β₂ = 1.1832209879468272
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.0232852264632775e-6 from total weight 1.686947657369024: operator might not be hermitian?
│   α = 0.22818200575305508 + 1.0232852264632775e-6im
│   β₁ = 1.16015510353531
│   β₂ = 1.2032312773084308
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 9.778034893506704e-7 from total weight 1.7441005741153381: operator might not be hermitian?
│   α = 0.3359468737393356 + 9.778034893506704e-7im
│   β₁ = 1.2032312773084308
│   β₂ = 1.2170706651464347
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 9.625125309067895e-7 from total weight 1.6930493324860827: operator might not be hermitian?
│   α = 0.15689474930621 + 9.625125309067895e-7im
│   β₁ = 1.2170706651464347
│   β₂ = 1.1664214829600834
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.0255041846228552e-6 from total weight 1.6163568378358877: operator might not be hermitian?
│   α = 0.15946995305765732 + 1.0255041846228552e-6im
│   β₁ = 1.1664214829600834
│   β₂ = 1.1075376677019915
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 9.450003205895008e-7 from total weight 1.63310722976173: operator might not be hermitian?
│   α = 0.4164024184737056 + 9.450003205895008e-7im
│   β₁ = 1.1188595855688177
│   β₂ = 1.1143614662977006
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.080879533445732e-6 from total weight 1.614688674276658: operator might not be hermitian?
│   α = 0.30429877958921225 + 1.080879533445732e-6im
│   β₁ = 1.1143614662977006
│   β₂ = 1.1281933743855694
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.0988263242361906e-6 from total weight 1.6115083028404351: operator might not be hermitian?
│   α = 0.2897351912699093 + 1.0988263242361906e-6im
│   β₁ = 1.1281933743855694
│   β₂ = 1.113639187104487
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.003436960373555e-6 from total weight 1.6172105156629197: operator might not be hermitian?
│   α = 0.14272259326333872 + 1.003436960373555e-6im
│   β₁ = 1.113639187104487
│   β₂ = 1.1639621446967956
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.0252943553334948e-6 from total weight 1.6211799504508382: operator might not be hermitian?
│   α = 0.1247595589372499 + 1.0252943553334948e-6im
│   β₁ = 1.1829896553891046
│   β₂ = 1.10145129689827
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 9.687233263298844e-7 from total weight 1.5808042072341244: operator might not be hermitian?
│   α = 0.1804895992486852 + 9.687233263298844e-7im
│   β₁ = 1.10145129689827
│   β₂ = 1.1194509755823783
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 9.67778768440515e-7 from total weight 1.564685677695013: operator might not be hermitian?
│   α = 0.27799810530818225 + 9.67778768440515e-7im
│   β₁ = 0.9536269829322506
│   β₂ = 1.2089474351069032
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
[ Info: VUMPS  15:	obj = -1.268018919710e-01	err = 3.9770372558e-01	time = 0.03 sec
[ Info: VUMPS  16:	obj = -2.089366409470e-01	err = 3.5604649696e-01	time = 0.06 sec
┌ Warning: ignoring imaginary component -1.4642964467764807e-6 from total weight 2.394006887776447: operator might not be hermitian?
│   α = 0.8242382100564384 - 1.4642964467764807e-6im
│   β₁ = 1.655282987088156
│   β₂ = 1.520506029076635
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.3983965608810323e-6 from total weight 2.298232529409139: operator might not be hermitian?
│   α = 0.9982576460523188 - 1.3983965608810323e-6im
│   β₁ = 1.520506029076635
│   β₂ = 1.4047832028010792
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.2414856767097027e-6 from total weight 2.2348174847568463: operator might not be hermitian?
│   α = 0.980492672115643 - 1.2414856767097027e-6im
│   β₁ = 1.4047832028010792
│   β₂ = 1.4351402242390265
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.4617492379419977e-6 from total weight 2.20291267013881: operator might not be hermitian?
│   α = 0.9485389110548547 - 1.4617492379419977e-6im
│   β₁ = 1.4765975180930613
│   β₂ = 1.3314495619557836
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.3979009563739458e-6 from total weight 2.1101472430666828: operator might not be hermitian?
│   α = 1.0471134224194303 - 1.3979009563739458e-6im
│   β₁ = 1.3314495619557836
│   β₂ = 1.2583786918002022
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.2539214374338373e-6 from total weight 2.036435547475786: operator might not be hermitian?
│   α = 0.9129612687076675 - 1.2539214374338373e-6im
│   β₁ = 1.2583786918002022
│   β₂ = 1.3153153724047817
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
[ Info: VUMPS  17:	obj = -2.048644413129e-01	err = 3.6617686541e-01	time = 0.03 sec
[ Info: VUMPS  18:	obj = -2.799693370510e-01	err = 3.3255889569e-01	time = 0.03 sec
[ Info: VUMPS  19:	obj = -1.753528829672e-01	err = 3.8615523295e-01	time = 0.03 sec
[ Info: VUMPS  20:	obj = +6.627601983914e-02	err = 3.7292664656e-01	time = 0.06 sec
[ Info: VUMPS  21:	obj = -2.320790738208e-01	err = 3.6756828615e-01	time = 0.03 sec
[ Info: VUMPS  22:	obj = +9.275292947159e-03	err = 4.0192510807e-01	time = 0.02 sec
[ Info: VUMPS  23:	obj = -7.298797837172e-02	err = 3.7988627737e-01	time = 0.02 sec
[ Info: VUMPS  24:	obj = -1.854126547483e-01	err = 3.6419418325e-01	time = 0.05 sec
[ Info: VUMPS  25:	obj = -3.671777760261e-01	err = 3.0812532655e-01	time = 0.03 sec
[ Info: VUMPS  26:	obj = -6.058668310156e-02	err = 3.8116081944e-01	time = 0.03 sec
[ Info: VUMPS  27:	obj = -2.015499701872e-03	err = 4.0317162622e-01	time = 0.02 sec
[ Info: VUMPS  28:	obj = -1.933169651612e-01	err = 3.8584662029e-01	time = 0.05 sec
┌ Warning: ignoring imaginary component -1.1661211199304189e-6 from total weight 1.9830955524856506: operator might not be hermitian?
│   α = 0.7062017103994722 - 1.1661211199304189e-6im
│   β₁ = 1.2975553722381348
│   β₂ = 1.322987970652728
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1145563612105502e-6 from total weight 1.9381249814965889: operator might not be hermitian?
│   α = 0.5691588295585481 - 1.1145563612105502e-6im
│   β₁ = 1.322987970652728
│   β₂ = 1.2969539313883243
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.2189929755176618e-6 from total weight 2.056544867261239: operator might not be hermitian?
│   α = 0.5470064500394781 - 1.2189929755176618e-6im
│   β₁ = 1.4301041013320408
│   β₂ = 1.3729395449274322
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1440933743427778e-6 from total weight 2.0445152640029147: operator might not be hermitian?
│   α = 0.6236570359401358 - 1.1440933743427778e-6im
│   β₁ = 1.350665796986038
│   β₂ = 1.4024252105242667
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.248874490177021e-6 from total weight 2.031575324892375: operator might not be hermitian?
│   α = 0.5290086048223288 - 1.248874490177021e-6im
│   β₁ = 1.4024252105242667
│   β₂ = 1.371368559366846
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1024766432207733e-6 from total weight 2.0089343008618497: operator might not be hermitian?
│   α = 0.7138216358572168 - 1.1024766432207733e-6im
│   β₁ = 1.371368559366846
│   β₂ = 1.282818760285451
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1510270212095053e-6 from total weight 1.9537611011393712: operator might not be hermitian?
│   α = 0.6076833782060458 - 1.1510270212095053e-6im
│   β₁ = 1.282818760285451
│   β₂ = 1.342489992676217
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.224997782992704e-6 from total weight 1.9645654639590098: operator might not be hermitian?
│   α = 0.5951035704552179 - 1.224997782992704e-6im
│   β₁ = 1.342489992676217
│   β₂ = 1.305024835845909
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1876444488932714e-6 from total weight 2.0169775219024793: operator might not be hermitian?
│   α = 0.7688953944030071 - 1.1876444488932714e-6im
│   β₁ = 1.305024835845909
│   β₂ = 1.331881516558308
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1840660663932356e-6 from total weight 1.984838719137105: operator might not be hermitian?
│   α = 0.785780036375005 - 1.1840660663932356e-6im
│   β₁ = 1.331881516558308
│   β₂ = 1.2442773409771872
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.077349919163459e-6 from total weight 1.8902174931287379: operator might not be hermitian?
│   α = 0.7502244171904665 - 1.077349919163459e-6im
│   β₁ = 1.2442773409771872
│   β₂ = 1.2090737752141907
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0988026773565973e-6 from total weight 1.9119512889411112: operator might not be hermitian?
│   α = 0.6200929725096753 - 1.0988026773565973e-6im
│   β₁ = 1.2354323373196032
│   β₂ = 1.3208896156119578
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.2560645931206094e-6 from total weight 1.7909478939100432: operator might not be hermitian?
│   α = 0.7333732110449686 - 1.2560645931206094e-6im
│   β₁ = 1.1135982474665187
│   β₂ = 1.1956408471026458
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1138326366866638e-6 from total weight 1.7427923996019516: operator might not be hermitian?
│   α = 0.6189073977995818 - 1.1138326366866638e-6im
│   β₁ = 1.1956408471026458
│   β₂ = 1.1066715618455663
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -9.82501486145211e-7 from total weight 1.737355447292098: operator might not be hermitian?
│   α = 0.7353320014976531 - 9.82501486145211e-7im
│   β₁ = 1.0893509501287795
│   β₂ = 1.136224143935316
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.2054549464787095e-6 from total weight 1.765964675748259: operator might not be hermitian?
│   α = 0.5948599677205233 - 1.2054549464787095e-6im
│   β₁ = 1.136224143935316
│   β₂ = 1.2139882822874353
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1910613334550207e-6 from total weight 1.7457281622540435: operator might not be hermitian?
│   α = 0.5876265067570648 - 1.1910613334550207e-6im
│   β₁ = 1.2139882822874353
│   β₂ = 1.1083746458263801
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1145402977093266e-6 from total weight 1.8228236011903869: operator might not be hermitian?
│   α = 0.7583728039666247 - 1.1145402977093266e-6im
│   β₁ = 1.1812863090759282
│   β₂ = 1.1628066164453221
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0244123553188045e-6 from total weight 1.827416176813848: operator might not be hermitian?
│   α = 0.7209367429139465 - 1.0244123553188045e-6im
│   β₁ = 1.1628066164453221
│   β₂ = 1.2114375216029523
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0686049562343147e-6 from total weight 1.8521257122682948: operator might not be hermitian?
│   α = 0.9046242484510791 - 1.0686049562343147e-6im
│   β₁ = 1.1441752673029573
│   β₂ = 1.1414410106749817
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.02064543803463e-6 from total weight 1.8303026002338043: operator might not be hermitian?
│   α = 0.8018410455211272 - 1.02064543803463e-6im
│   β₁ = 1.1414410106749817
│   β₂ = 1.1849772003243124
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1415911429371942e-6 from total weight 1.8503026810057916: operator might not be hermitian?
│   α = 0.7878644256718402 - 1.1415911429371942e-6im
│   β₁ = 1.1849772003243124
│   β₂ = 1.1826743815641958
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1470443615248238e-6 from total weight 1.8849501176514116: operator might not be hermitian?
│   α = 0.8619538770355701 - 1.1470443615248238e-6im
│   β₁ = 1.1826743815641958
│   β₂ = 1.1880041107200983
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0568060722793843e-6 from total weight 1.7864879735856112: operator might not be hermitian?
│   α = 0.7350531632675616 - 1.0568060722793843e-6im
│   β₁ = 1.1880041107200983
│   β₂ = 1.1135000493252825
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -9.602859024383792e-7 from total weight 1.7025702297398062: operator might not be hermitian?
│   α = 0.6791999518194767 - 9.602859024383792e-7im
│   β₁ = 1.1135000493252825
│   β₂ = 1.0943264836402107
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0558620769408888e-6 from total weight 1.7596527593571747: operator might not be hermitian?
│   α = 0.7769001371869034 - 1.0558620769408888e-6im
│   β₁ = 1.1619115114008967
│   β₂ = 1.0690021749394427
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0354194948727913e-6 from total weight 1.7038410836176745: operator might not be hermitian?
│   α = 0.7165984827525741 - 1.0354194948727913e-6im
│   β₁ = 1.0690021749394427
│   β₂ = 1.116599929569149
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.091324247760525e-6 from total weight 1.7342773510387077: operator might not be hermitian?
│   α = 0.8026570431688571 - 1.091324247760525e-6im
│   β₁ = 1.0720399440829314
│   β₂ = 1.1019028812317075
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1757481594952546e-6 from total weight 1.7331994241999757: operator might not be hermitian?
│   α = 0.866234625976551 - 1.1757481594952546e-6im
│   β₁ = 1.1019028812317075
│   β₂ = 1.0195233480103687
└ @ KrylovKit ~/.julia/packages/KrylovKit/xccMN/src/factorizations/lanczos.jl:170
[ Info: VUMPS  29:	obj = -1.409357861027e-01	err = 3.5767477119e-01	time = 0.03 sec
[ Info: VUMPS  30:	obj = -2.045532833861e-01	err = 3.6013137514e-01	time = 0.02 sec
[ Info: VUMPS  31:	obj = -3.677340068350e-01	err = 2.9848315584e-01	time = 0.02 sec
[ Info: VUMPS  32:	obj = -3.332844888666e-01	err = 3.3334001697e-01	time = 0.06 sec
[ Info: VUMPS  33:	obj = -2.919507908127e-01	err = 3.5475313994e-01	time = 0.03 sec
[ Info: VUMPS  34:	obj = -2.808449276253e-02	err = 3.6950599342e-01	time = 0.08 sec
[ Info: VUMPS  35:	obj = -9.830142151296e-02	err = 3.7023131749e-01	time = 0.06 sec
[ Info: VUMPS  36:	obj = -2.504433745603e-01	err = 3.5111294760e-01	time = 0.02 sec
[ Info: VUMPS  37:	obj = -3.194192131126e-01	err = 3.2385721415e-01	time = 0.02 sec
[ Info: VUMPS  38:	obj = +8.884748290821e-02	err = 3.9871163038e-01	time = 0.07 sec
[ Info: VUMPS  39:	obj = -1.989518543281e-01	err = 3.6796439117e-01	time = 0.02 sec
[ Info: VUMPS  40:	obj = -2.568596388181e-01	err = 3.5396377788e-01	time = 0.02 sec
[ Info: VUMPS  41:	obj = -1.742517027774e-01	err = 3.7560253512e-01	time = 0.03 sec
[ Info: VUMPS  42:	obj = -2.181842216501e-01	err = 3.7550074763e-01	time = 0.02 sec
[ Info: VUMPS  43:	obj = -6.227109008840e-02	err = 4.1909295201e-01	time = 0.05 sec
[ Info: VUMPS  44:	obj = -7.668274105361e-02	err = 4.0844645003e-01	time = 0.03 sec
[ Info: VUMPS  45:	obj = -1.835375816162e-01	err = 3.4820396366e-01	time = 0.02 sec
[ Info: VUMPS  46:	obj = -6.357716783443e-02	err = 3.7708810363e-01	time = 0.02 sec
[ Info: VUMPS  47:	obj = +1.388559492933e-04 -5.781399664562e-15im	err = 3.6961508021e-01	time = 0.05 sec
[ Info: VUMPS  48:	obj = -1.404104902042e-01	err = 3.9016059615e-01	time = 0.02 sec
[ Info: VUMPS  49:	obj = -2.106708660955e-01	err = 3.6760368335e-01	time = 0.02 sec
[ Info: VUMPS  50:	obj = +1.735796736157e-01	err = 3.5521706372e-01	time = 0.02 sec
[ Info: VUMPS  51:	obj = -5.367170688984e-02	err = 3.9109124649e-01	time = 0.06 sec
[ Info: VUMPS  52:	obj = -2.298653965104e-01	err = 3.3926679477e-01	time = 0.02 sec
[ Info: VUMPS  53:	obj = -3.165325039869e-01	err = 3.3133971575e-01	time = 0.02 sec
[ Info: VUMPS  54:	obj = -2.950197697207e-01	err = 3.4876181076e-01	time = 0.03 sec
[ Info: VUMPS  55:	obj = -2.374027247452e-01	err = 3.7519899465e-01	time = 0.06 sec
[ Info: VUMPS  56:	obj = -1.619819474039e-02	err = 3.7718729795e-01	time = 0.02 sec
[ Info: VUMPS  57:	obj = -1.705343092043e-01	err = 3.8558807493e-01	time = 0.02 sec
[ Info: VUMPS  58:	obj = -3.730847939935e-01	err = 2.8129746259e-01	time = 0.02 sec
[ Info: VUMPS  59:	obj = -2.151394001240e-01	err = 4.1030105789e-01	time = 0.06 sec
[ Info: VUMPS  60:	obj = -1.561041094602e-01	err = 3.9671982762e-01	time = 0.02 sec
[ Info: VUMPS  61:	obj = -9.599837934985e-02	err = 3.8528434214e-01	time = 0.02 sec
[ Info: VUMPS  62:	obj = -2.825562036813e-01	err = 3.3900898933e-01	time = 0.02 sec
[ Info: VUMPS  63:	obj = -1.332286499835e-02	err = 3.8572490466e-01	time = 0.06 sec
[ Info: VUMPS  64:	obj = -1.912868202818e-02	err = 3.9649310912e-01	time = 0.02 sec
[ Info: VUMPS  65:	obj = -3.005244814032e-02	err = 4.1809852519e-01	time = 0.02 sec
[ Info: VUMPS  66:	obj = -1.895637479899e-01	err = 3.5881009323e-01	time = 0.05 sec
[ Info: VUMPS  67:	obj = -2.664858728631e-01	err = 3.6123422433e-01	time = 0.03 sec
[ Info: VUMPS  68:	obj = -2.080458300379e-01	err = 3.8597810940e-01	time = 0.03 sec
[ Info: VUMPS  69:	obj = -3.878141128139e-01	err = 2.6683063405e-01	time = 0.02 sec
[ Info: VUMPS  70:	obj = -4.330198106833e-01	err = 1.1995217484e-01	time = 0.08 sec
[ Info: VUMPS  71:	obj = -7.298278720525e-02	err = 3.8622717637e-01	time = 0.03 sec
[ Info: VUMPS  72:	obj = -2.945720162791e-01	err = 3.5422369206e-01	time = 0.02 sec
[ Info: VUMPS  73:	obj = -3.416537322445e-01	err = 3.1228535420e-01	time = 0.02 sec
[ Info: VUMPS  74:	obj = -3.331595948832e-01	err = 3.4179032729e-01	time = 0.06 sec
[ Info: VUMPS  75:	obj = -1.222292014205e-01	err = 4.0978910412e-01	time = 0.03 sec
[ Info: VUMPS  76:	obj = -6.382262951270e-02	err = 3.8950144913e-01	time = 0.04 sec
[ Info: VUMPS  77:	obj = -2.249182124905e-01	err = 3.6705315029e-01	time = 0.06 sec
[ Info: VUMPS  78:	obj = -3.169211687090e-01	err = 3.1190417293e-01	time = 0.03 sec
[ Info: VUMPS  79:	obj = -3.795013187369e-01	err = 2.8575855105e-01	time = 0.04 sec
[ Info: VUMPS  80:	obj = -3.541737574468e-01	err = 3.1274076174e-01	time = 0.03 sec
[ Info: VUMPS  81:	obj = -1.482341913801e-01	err = 3.9486023268e-01	time = 0.06 sec
[ Info: VUMPS  82:	obj = -2.431823642066e-01	err = 3.5485354611e-01	time = 0.03 sec
[ Info: VUMPS  83:	obj = -1.426154410251e-01	err = 3.9431885582e-01	time = 0.02 sec
[ Info: VUMPS  84:	obj = -1.879354265828e-01	err = 3.7761650164e-01	time = 0.06 sec
[ Info: VUMPS  85:	obj = -3.883275844777e-02	err = 3.3484189710e-01	time = 0.03 sec
[ Info: VUMPS  86:	obj = -2.541785323711e-01	err = 3.6013211771e-01	time = 0.02 sec
[ Info: VUMPS  87:	obj = -3.516595702805e-01	err = 3.0264434875e-01	time = 0.03 sec
[ Info: VUMPS  88:	obj = -2.234563948608e-01	err = 3.6699482034e-01	time = 0.05 sec
[ Info: VUMPS  89:	obj = +2.492258044917e-02	err = 3.8778077237e-01	time = 0.02 sec
[ Info: VUMPS  90:	obj = -3.286041797336e-01	err = 3.2736069863e-01	time = 0.02 sec
[ Info: VUMPS  91:	obj = -2.158862857429e-01	err = 3.7995863438e-01	time = 0.03 sec
[ Info: VUMPS  92:	obj = -2.675801195049e-01	err = 3.6561496937e-01	time = 0.06 sec
[ Info: VUMPS  93:	obj = +5.624309743865e-03	err = 4.2867100082e-01	time = 0.03 sec
[ Info: VUMPS  94:	obj = -1.759420429917e-01	err = 3.9349409438e-01	time = 0.02 sec
[ Info: VUMPS  95:	obj = -3.361200941070e-01	err = 2.9968812876e-01	time = 0.02 sec
[ Info: VUMPS  96:	obj = -2.210577853944e-03	err = 3.9551380217e-01	time = 0.06 sec
[ Info: VUMPS  97:	obj = -2.235056018794e-02	err = 3.9877664313e-01	time = 0.02 sec
[ Info: VUMPS  98:	obj = +1.266964324347e-01	err = 3.5087282451e-01	time = 0.02 sec
[ Info: VUMPS  99:	obj = -1.649834446522e-01	err = 3.7313907507e-01	time = 0.02 sec
[ Info: VUMPS 100:	obj = -2.810724087197e-01	err = 3.4803141841e-01	time = 0.06 sec
[ Info: VUMPS 101:	obj = -3.934095054094e-01	err = 2.5788371653e-01	time = 0.03 sec
[ Info: VUMPS 102:	obj = -2.769141978863e-02	err = 4.2113724750e-01	time = 0.03 sec
[ Info: VUMPS 103:	obj = -1.428941704956e-01	err = 3.7175159384e-01	time = 0.05 sec
[ Info: VUMPS 104:	obj = -2.825079318885e-01	err = 3.4365263618e-01	time = 0.02 sec
[ Info: VUMPS 105:	obj = -3.895828699840e-01	err = 2.5772008058e-01	time = 0.03 sec
[ Info: VUMPS 106:	obj = -3.789769542971e-01	err = 2.7647068815e-01	time = 0.03 sec
[ Info: VUMPS 107:	obj = -4.131370544119e-01	err = 2.0771775492e-01	time = 0.06 sec
[ Info: VUMPS 108:	obj = +1.891808047741e-02	err = 4.0211631708e-01	time = 0.03 sec
[ Info: VUMPS 109:	obj = -3.126203477249e-01	err = 3.3581501662e-01	time = 0.03 sec
[ Info: VUMPS 110:	obj = -3.891256643994e-01	err = 2.6587912557e-01	time = 0.07 sec
[ Info: VUMPS 111:	obj = -3.350222584524e-01	err = 3.2878659423e-01	time = 0.03 sec
[ Info: VUMPS 112:	obj = -3.383500923964e-01	err = 3.0810425001e-01	time = 0.03 sec
[ Info: VUMPS 113:	obj = -2.921012433328e-01	err = 3.5940858624e-01	time = 0.03 sec
[ Info: VUMPS 114:	obj = -3.304195736343e-01	err = 3.2278577971e-01	time = 0.06 sec
[ Info: VUMPS 115:	obj = -1.435645156425e-01	err = 4.0512614823e-01	time = 0.03 sec
[ Info: VUMPS 116:	obj = -1.679237781365e-01	err = 3.6250880408e-01	time = 0.02 sec
[ Info: VUMPS 117:	obj = -1.408768635481e-01	err = 3.6326347756e-01	time = 0.05 sec
[ Info: VUMPS 118:	obj = -1.939462004720e-02	err = 4.0498337302e-01	time = 0.03 sec
[ Info: VUMPS 119:	obj = -2.352201460179e-01	err = 3.6574114214e-01	time = 0.02 sec
[ Info: VUMPS 120:	obj = -2.295422994168e-02	err = 3.6188803296e-01	time = 0.03 sec
[ Info: VUMPS 121:	obj = -6.989486583808e-02	err = 3.7006193288e-01	time = 0.05 sec
[ Info: VUMPS 122:	obj = -3.493531583983e-02	err = 3.8804888357e-01	time = 0.03 sec
[ Info: VUMPS 123:	obj = +7.790731640126e-04	err = 3.8138863176e-01	time = 0.02 sec
[ Info: VUMPS 124:	obj = -1.919530149789e-01	err = 3.6070095280e-01	time = 0.03 sec
[ Info: VUMPS 125:	obj = -3.218424809114e-01	err = 3.3338267984e-01	time = 0.05 sec
[ Info: VUMPS 126:	obj = -3.667333159463e-01	err = 2.8821000587e-01	time = 0.02 sec
[ Info: VUMPS 127:	obj = -2.130648461950e-01	err = 3.7595362973e-01	time = 0.03 sec
[ Info: VUMPS 128:	obj = -1.030371749297e-01	err = 4.2417993413e-01	time = 0.03 sec
[ Info: VUMPS 129:	obj = -1.830830998421e-01	err = 3.7685385391e-01	time = 0.05 sec
[ Info: VUMPS 130:	obj = -7.071567794803e-02	err = 3.9744980880e-01	time = 0.02 sec
[ Info: VUMPS 131:	obj = -4.769518696950e-02	err = 3.8112218583e-01	time = 0.02 sec
[ Info: VUMPS 132:	obj = -2.991402416948e-01	err = 3.3370573847e-01	time = 0.02 sec
[ Info: VUMPS 133:	obj = -2.746313335444e-01	err = 3.5097484163e-01	time = 0.06 sec
[ Info: VUMPS 134:	obj = -3.276842746897e-01	err = 3.1708681372e-01	time = 0.02 sec
[ Info: VUMPS 135:	obj = -2.049761563157e-01	err = 3.7370273426e-01	time = 0.02 sec
[ Info: VUMPS 136:	obj = -3.833983237486e-01	err = 2.6114682914e-01	time = 0.02 sec
[ Info: VUMPS 137:	obj = -6.177005685578e-02	err = 3.8019469489e-01	time = 0.06 sec
[ Info: VUMPS 138:	obj = -1.316570790104e-01	err = 3.8729233886e-01	time = 0.02 sec
[ Info: VUMPS 139:	obj = -2.366225337929e-01	err = 3.6759309875e-01	time = 0.03 sec
[ Info: VUMPS 140:	obj = +8.913339361766e-02	err = 3.7167547836e-01	time = 0.08 sec
[ Info: VUMPS 141:	obj = -7.481699300835e-02	err = 3.8011320937e-01	time = 0.04 sec
[ Info: VUMPS 142:	obj = -2.554895657661e-01	err = 3.5626764832e-01	time = 0.03 sec
[ Info: VUMPS 143:	obj = -2.752729453871e-01	err = 3.5827930411e-01	time = 0.03 sec
[ Info: VUMPS 144:	obj = -1.078710456452e-01	err = 3.7452629595e-01	time = 0.06 sec
[ Info: VUMPS 145:	obj = -1.834468230261e-01	err = 3.9202973168e-01	time = 0.03 sec
[ Info: VUMPS 146:	obj = -2.880934140139e-01	err = 3.4460619232e-01	time = 0.04 sec
[ Info: VUMPS 147:	obj = -2.585829024830e-01	err = 3.4906000653e-01	time = 0.03 sec
[ Info: VUMPS 148:	obj = -7.446540858731e-02	err = 4.2725017960e-01	time = 0.06 sec
[ Info: VUMPS 149:	obj = +1.223133246756e-02	err = 4.2025804351e-01	time = 0.02 sec
[ Info: VUMPS 150:	obj = -7.807677131111e-02	err = 4.0384260646e-01	time = 0.02 sec
[ Info: VUMPS 151:	obj = -1.510082807256e-01	err = 4.1367387477e-01	time = 0.03 sec
[ Info: VUMPS 152:	obj = -1.077844357575e-01	err = 3.8783544296e-01	time = 0.05 sec
[ Info: VUMPS 153:	obj = -2.687661583404e-01	err = 3.5954953023e-01	time = 0.02 sec
[ Info: VUMPS 154:	obj = -2.216118903419e-01	err = 3.6631386313e-01	time = 0.03 sec
[ Info: VUMPS 155:	obj = -3.346454865703e-03	err = 3.7228195751e-01	time = 0.03 sec
[ Info: VUMPS 156:	obj = -2.182777019563e-01	err = 3.7120325813e-01	time = 0.06 sec
[ Info: VUMPS 157:	obj = -2.260481514080e-01	err = 3.7084200121e-01	time = 0.03 sec
[ Info: VUMPS 158:	obj = -3.753639631248e-01	err = 2.8129702848e-01	time = 0.02 sec
[ Info: VUMPS 159:	obj = +1.493818093509e-01	err = 3.5990583300e-01	time = 0.06 sec
[ Info: VUMPS 160:	obj = -7.335279560891e-02	err = 3.8634405938e-01	time = 0.02 sec
[ Info: VUMPS 161:	obj = -1.712788073268e-01	err = 3.8098418215e-01	time = 0.03 sec
[ Info: VUMPS 162:	obj = -2.775045952826e-02	err = 3.6432182146e-01	time = 0.03 sec
[ Info: VUMPS 163:	obj = -2.974540518908e-01	err = 3.2120804570e-01	time = 0.06 sec
[ Info: VUMPS 164:	obj = -4.054076797989e-01	err = 2.3514693235e-01	time = 0.04 sec
[ Info: VUMPS 165:	obj = -7.519182934781e-02	err = 4.0110456979e-01	time = 0.03 sec
[ Info: VUMPS 166:	obj = -1.784723524117e-01	err = 3.8591296831e-01	time = 0.02 sec
[ Info: VUMPS 167:	obj = -1.116507302906e-01	err = 3.7195004686e-01	time = 0.06 sec
[ Info: VUMPS 168:	obj = -3.122365679686e-01	err = 3.3350953764e-01	time = 0.02 sec
[ Info: VUMPS 169:	obj = -3.560972462991e-01	err = 2.8786350505e-01	time = 0.03 sec
[ Info: VUMPS 170:	obj = -2.497036138206e-01	err = 3.7652961352e-01	time = 0.03 sec
[ Info: VUMPS 171:	obj = -1.204935017155e-01	err = 3.7477170313e-01	time = 0.06 sec
[ Info: VUMPS 172:	obj = -3.140736291711e-01	err = 3.3651541095e-01	time = 0.02 sec
[ Info: VUMPS 173:	obj = -3.777537348031e-01	err = 2.8281048666e-01	time = 0.03 sec
[ Info: VUMPS 174:	obj = -4.081318507282e-01	err = 2.2729687790e-01	time = 0.06 sec
[ Info: VUMPS 175:	obj = +8.272685287566e-03	err = 4.2559224619e-01	time = 0.03 sec
[ Info: VUMPS 176:	obj = -1.393268322253e-01	err = 3.9579900440e-01	time = 0.03 sec
[ Info: VUMPS 177:	obj = -9.734245038937e-02	err = 3.5396586975e-01	time = 0.03 sec
[ Info: VUMPS 178:	obj = -2.821477914715e-01	err = 3.5794786333e-01	time = 0.06 sec
[ Info: VUMPS 179:	obj = -4.040896409723e-01	err = 2.2336565394e-01	time = 0.03 sec
[ Info: VUMPS 180:	obj = +8.386756242174e-03	err = 4.0133165398e-01	time = 0.02 sec
[ Info: VUMPS 181:	obj = -1.390628324690e-01	err = 3.5859390916e-01	time = 0.05 sec
[ Info: VUMPS 182:	obj = -1.281482713515e-01	err = 3.6674107529e-01	time = 0.02 sec
[ Info: VUMPS 183:	obj = -9.466660283788e-02	err = 4.0504716786e-01	time = 0.03 sec
[ Info: VUMPS 184:	obj = -1.755981307732e-01	err = 3.8378790855e-01	time = 0.02 sec
[ Info: VUMPS 185:	obj = -9.036868885930e-02	err = 3.6141032024e-01	time = 0.05 sec
[ Info: VUMPS 186:	obj = -3.703625904624e-01	err = 2.8654729147e-01	time = 0.02 sec
[ Info: VUMPS 187:	obj = -3.162732937988e-01	err = 3.4485685016e-01	time = 0.03 sec
[ Info: VUMPS 188:	obj = -1.112394896513e-01	err = 3.9807795128e-01	time = 0.03 sec
[ Info: VUMPS 189:	obj = +1.824853010192e-03	err = 3.9447516200e-01	time = 0.06 sec
[ Info: VUMPS 190:	obj = -2.140867094456e-01	err = 3.7898558231e-01	time = 0.02 sec
[ Info: VUMPS 191:	obj = +4.894440406361e-02	err = 3.9055437558e-01	time = 0.02 sec
[ Info: VUMPS 192:	obj = -3.751396808486e-02	err = 3.8799740635e-01	time = 0.02 sec
[ Info: VUMPS 193:	obj = -1.682962360358e-01	err = 3.7336809188e-01	time = 0.05 sec
[ Info: VUMPS 194:	obj = -1.748904771170e-01	err = 3.3383969111e-01	time = 0.02 sec
[ Info: VUMPS 195:	obj = -2.023258032592e-01	err = 3.5779592817e-01	time = 0.02 sec
[ Info: VUMPS 196:	obj = -2.562842975568e-01	err = 3.5451811621e-01	time = 0.03 sec
[ Info: VUMPS 197:	obj = -1.198772391755e-01	err = 3.8884771754e-01	time = 0.06 sec
[ Info: VUMPS 198:	obj = -1.194696203501e-01	err = 3.6306124333e-01	time = 0.03 sec
[ Info: VUMPS 199:	obj = -3.165670058206e-01	err = 3.4580209634e-01	time = 0.03 sec
┌ Warning: VUMPS cancel 200:	obj = -3.605645351808e-01	err = 3.2889262642e-01	time = 6.93 sec
└ @ MPSKit ~/Projects/MPSKit.jl/src/algorithms/groundstate/vumps.jl:67

````

As you can see, VUMPS struggles to converge.
On it's own, that is already quite curious.
Maybe we can do better using another algorithm, such as gradient descent.

````julia
groundstate, cache, delta = find_groundstate(state, H, GradientGrassmann(; maxiter=20));
````

````
[ Info: CG: initializing with f = 0.249997124956, ‖∇f‖ = 3.4016e-03
[ Info: CG: iter    1: f = -0.025384109228, ‖∇f‖ = 6.9269e-01, α = 8.72e+05, β = 0.00e+00, nfg = 18
[ Info: CG: iter    2: f = -0.034734780784, ‖∇f‖ = 6.7957e-01, α = 3.41e-02, β = 2.69e+04, nfg = 19
[ Info: CG: iter    3: f = -0.035137291778, ‖∇f‖ = 6.6715e-01, α = 5.70e-03, β = 4.46e-01, nfg = 3
┌ Warning: resorting to η
└ @ OptimKit ~/.julia/packages/OptimKit/xpmbV/src/cg.jl:139
[ Info: CG: iter    4: f = -0.230959138308, ‖∇f‖ = 6.9544e-01, α = 1.02e+00, β = -2.32e-04, nfg = 5
[ Info: CG: iter    5: f = -0.344516007182, ‖∇f‖ = 5.3414e-01, α = 7.93e-01, β = 3.06e-01, nfg = 3
[ Info: CG: iter    6: f = -0.394577435495, ‖∇f‖ = 4.0969e-01, α = 7.63e-01, β = 1.75e-01, nfg = 2
[ Info: CG: iter    7: f = -0.423605152054, ‖∇f‖ = 2.3260e-01, α = 5.92e-01, β = 1.59e-01, nfg = 2
[ Info: CG: iter    8: f = -0.433877622086, ‖∇f‖ = 1.0355e-01, α = 4.39e-01, β = 1.19e-01, nfg = 2
[ Info: CG: iter    9: f = -0.436274421836, ‖∇f‖ = 7.4845e-02, α = 3.14e-01, β = 2.45e-01, nfg = 2
[ Info: CG: iter   10: f = -0.437275612812, ‖∇f‖ = 6.7115e-02, α = 1.56e-01, β = 6.92e-01, nfg = 2
[ Info: CG: iter   11: f = -0.438430268284, ‖∇f‖ = 6.7977e-02, α = 3.09e-01, β = 4.96e-01, nfg = 2
[ Info: CG: iter   12: f = -0.439719125881, ‖∇f‖ = 5.3438e-02, α = 4.32e-01, β = 3.51e-01, nfg = 2
[ Info: CG: iter   13: f = -0.440296278919, ‖∇f‖ = 4.4785e-02, α = 2.16e-01, β = 5.24e-01, nfg = 2
[ Info: CG: iter   14: f = -0.440794890531, ‖∇f‖ = 3.8641e-02, α = 2.22e-01, β = 6.23e-01, nfg = 2
[ Info: CG: iter   15: f = -0.441206155655, ‖∇f‖ = 3.7357e-02, α = 3.41e-01, β = 3.86e-01, nfg = 2
[ Info: CG: iter   16: f = -0.441529728743, ‖∇f‖ = 3.0799e-02, α = 2.78e-01, β = 4.26e-01, nfg = 2
[ Info: CG: iter   17: f = -0.441770475509, ‖∇f‖ = 2.7481e-02, α = 2.52e-01, β = 5.75e-01, nfg = 2
[ Info: CG: iter   18: f = -0.441987084171, ‖∇f‖ = 2.5934e-02, α = 3.26e-01, β = 4.60e-01, nfg = 2
[ Info: CG: iter   19: f = -0.442146991122, ‖∇f‖ = 2.3586e-02, α = 2.69e-01, β = 5.43e-01, nfg = 2
┌ Warning: CG: not converged to requested tol: f = -0.442291713427, ‖∇f‖ = 2.0461e-02
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
  <clipPath id="clip650">
    <rect x="0" y="0" width="2400" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip650)" d="M0 1600 L2400 1600 L2400 0 L0 0  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip651">
    <rect x="480" y="0" width="1681" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip650)" d="M219.288 1423.18 L2352.76 1423.18 L2352.76 47.2441 L219.288 47.2441  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip652">
    <rect x="219" y="47" width="2134" height="1377"/>
  </clipPath>
</defs>
<polyline clip-path="url(#clip652)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="279.669,1423.18 279.669,47.2441 "/>
<polyline clip-path="url(#clip652)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="615.12,1423.18 615.12,47.2441 "/>
<polyline clip-path="url(#clip652)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="950.571,1423.18 950.571,47.2441 "/>
<polyline clip-path="url(#clip652)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1286.02,1423.18 1286.02,47.2441 "/>
<polyline clip-path="url(#clip652)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1621.47,1423.18 1621.47,47.2441 "/>
<polyline clip-path="url(#clip652)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1956.92,1423.18 1956.92,47.2441 "/>
<polyline clip-path="url(#clip652)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="2292.37,1423.18 2292.37,47.2441 "/>
<polyline clip-path="url(#clip652)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="219.288,1398.58 2352.76,1398.58 "/>
<polyline clip-path="url(#clip652)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="219.288,1012.49 2352.76,1012.49 "/>
<polyline clip-path="url(#clip652)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="219.288,626.389 2352.76,626.389 "/>
<polyline clip-path="url(#clip652)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="219.288,240.292 2352.76,240.292 "/>
<polyline clip-path="url(#clip650)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="219.288,1423.18 2352.76,1423.18 "/>
<polyline clip-path="url(#clip650)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="279.669,1423.18 279.669,1404.28 "/>
<polyline clip-path="url(#clip650)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="615.12,1423.18 615.12,1404.28 "/>
<polyline clip-path="url(#clip650)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="950.571,1423.18 950.571,1404.28 "/>
<polyline clip-path="url(#clip650)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1286.02,1423.18 1286.02,1404.28 "/>
<polyline clip-path="url(#clip650)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1621.47,1423.18 1621.47,1404.28 "/>
<polyline clip-path="url(#clip650)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1956.92,1423.18 1956.92,1404.28 "/>
<polyline clip-path="url(#clip650)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="2292.37,1423.18 2292.37,1404.28 "/>
<path clip-path="url(#clip650)" d="M233.431 1454.1 Q229.819 1454.1 227.991 1457.66 Q226.185 1461.2 226.185 1468.33 Q226.185 1475.44 227.991 1479.01 Q229.819 1482.55 233.431 1482.55 Q237.065 1482.55 238.87 1479.01 Q240.699 1475.44 240.699 1468.33 Q240.699 1461.2 238.87 1457.66 Q237.065 1454.1 233.431 1454.1 M233.431 1450.39 Q239.241 1450.39 242.296 1455 Q245.375 1459.58 245.375 1468.33 Q245.375 1477.06 242.296 1481.67 Q239.241 1486.25 233.431 1486.25 Q227.62 1486.25 224.542 1481.67 Q221.486 1477.06 221.486 1468.33 Q221.486 1459.58 224.542 1455 Q227.62 1450.39 233.431 1450.39 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M260.56 1451.02 L264.495 1451.02 L252.458 1489.98 L248.523 1489.98 L260.56 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M276.532 1451.02 L280.467 1451.02 L268.43 1489.98 L264.495 1489.98 L276.532 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M286.347 1481.64 L293.986 1481.64 L293.986 1455.28 L285.676 1456.95 L285.676 1452.69 L293.94 1451.02 L298.615 1451.02 L298.615 1481.64 L306.254 1481.64 L306.254 1485.58 L286.347 1485.58 L286.347 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M312.342 1459.65 L337.18 1459.65 L337.18 1463.91 L333.916 1463.91 L333.916 1479.84 Q333.916 1481.51 334.472 1482.25 Q335.05 1482.96 336.324 1482.96 Q336.671 1482.96 337.18 1482.92 Q337.689 1482.85 337.851 1482.83 L337.851 1485.9 Q337.041 1486.2 336.185 1486.34 Q335.328 1486.48 334.472 1486.48 Q331.694 1486.48 330.629 1484.98 Q329.564 1483.45 329.564 1479.38 L329.564 1463.91 L320.004 1463.91 L320.004 1485.58 L315.652 1485.58 L315.652 1463.91 L312.342 1463.91 L312.342 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M558.65 1481.64 L566.289 1481.64 L566.289 1455.28 L557.979 1456.95 L557.979 1452.69 L566.243 1451.02 L570.919 1451.02 L570.919 1481.64 L578.557 1481.64 L578.557 1485.58 L558.65 1485.58 L558.65 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M594.969 1451.02 L598.905 1451.02 L586.868 1489.98 L582.932 1489.98 L594.969 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M610.942 1451.02 L614.877 1451.02 L602.84 1489.98 L598.905 1489.98 L610.942 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M634.113 1466.95 Q637.469 1467.66 639.344 1469.93 Q641.242 1472.2 641.242 1475.53 Q641.242 1480.65 637.724 1483.45 Q634.205 1486.25 627.724 1486.25 Q625.548 1486.25 623.233 1485.81 Q620.941 1485.39 618.488 1484.54 L618.488 1480.02 Q620.432 1481.16 622.747 1481.74 Q625.062 1482.32 627.585 1482.32 Q631.983 1482.32 634.275 1480.58 Q636.59 1478.84 636.59 1475.53 Q636.59 1472.48 634.437 1470.77 Q632.307 1469.03 628.488 1469.03 L624.46 1469.03 L624.46 1465.19 L628.673 1465.19 Q632.122 1465.19 633.951 1463.82 Q635.779 1462.43 635.779 1459.84 Q635.779 1457.18 633.881 1455.77 Q632.006 1454.33 628.488 1454.33 Q626.566 1454.33 624.367 1454.75 Q622.168 1455.16 619.529 1456.04 L619.529 1451.88 Q622.191 1451.14 624.506 1450.77 Q626.844 1450.39 628.904 1450.39 Q634.228 1450.39 637.33 1452.83 Q640.432 1455.23 640.432 1459.35 Q640.432 1462.22 638.789 1464.21 Q637.145 1466.18 634.113 1466.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M646.752 1459.65 L671.589 1459.65 L671.589 1463.91 L668.325 1463.91 L668.325 1479.84 Q668.325 1481.51 668.881 1482.25 Q669.46 1482.96 670.733 1482.96 Q671.08 1482.96 671.589 1482.92 Q672.099 1482.85 672.261 1482.83 L672.261 1485.9 Q671.45 1486.2 670.594 1486.34 Q669.738 1486.48 668.881 1486.48 Q666.103 1486.48 665.038 1484.98 Q663.974 1483.45 663.974 1479.38 L663.974 1463.91 L654.414 1463.91 L654.414 1485.58 L650.062 1485.58 L650.062 1463.91 L646.752 1463.91 L646.752 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M898.187 1481.64 L914.506 1481.64 L914.506 1485.58 L892.562 1485.58 L892.562 1481.64 Q895.224 1478.89 899.807 1474.26 Q904.414 1469.61 905.594 1468.27 Q907.839 1465.74 908.719 1464.01 Q909.622 1462.25 909.622 1460.56 Q909.622 1457.8 907.677 1456.07 Q905.756 1454.33 902.654 1454.33 Q900.455 1454.33 898.002 1455.09 Q895.571 1455.86 892.793 1457.41 L892.793 1452.69 Q895.617 1451.55 898.071 1450.97 Q900.525 1450.39 902.562 1450.39 Q907.932 1450.39 911.127 1453.08 Q914.321 1455.77 914.321 1460.26 Q914.321 1462.39 913.511 1464.31 Q912.724 1466.2 910.617 1468.8 Q910.039 1469.47 906.937 1472.69 Q903.835 1475.88 898.187 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M931.288 1451.02 L935.224 1451.02 L923.187 1489.98 L919.251 1489.98 L931.288 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M947.261 1451.02 L951.196 1451.02 L939.159 1489.98 L935.224 1489.98 L947.261 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M970.432 1466.95 Q973.788 1467.66 975.663 1469.93 Q977.561 1472.2 977.561 1475.53 Q977.561 1480.65 974.043 1483.45 Q970.524 1486.25 964.043 1486.25 Q961.867 1486.25 959.552 1485.81 Q957.261 1485.39 954.807 1484.54 L954.807 1480.02 Q956.751 1481.16 959.066 1481.74 Q961.381 1482.32 963.904 1482.32 Q968.302 1482.32 970.594 1480.58 Q972.909 1478.84 972.909 1475.53 Q972.909 1472.48 970.756 1470.77 Q968.626 1469.03 964.807 1469.03 L960.779 1469.03 L960.779 1465.19 L964.992 1465.19 Q968.441 1465.19 970.27 1463.82 Q972.098 1462.43 972.098 1459.84 Q972.098 1457.18 970.2 1455.77 Q968.325 1454.33 964.807 1454.33 Q962.885 1454.33 960.686 1454.75 Q958.487 1455.16 955.848 1456.04 L955.848 1451.88 Q958.511 1451.14 960.825 1450.77 Q963.163 1450.39 965.223 1450.39 Q970.547 1450.39 973.649 1452.83 Q976.751 1455.23 976.751 1459.35 Q976.751 1462.22 975.108 1464.21 Q973.464 1466.18 970.432 1466.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M983.071 1459.65 L1007.91 1459.65 L1007.91 1463.91 L1004.64 1463.91 L1004.64 1479.84 Q1004.64 1481.51 1005.2 1482.25 Q1005.78 1482.96 1007.05 1482.96 Q1007.4 1482.96 1007.91 1482.92 Q1008.42 1482.85 1008.58 1482.83 L1008.58 1485.9 Q1007.77 1486.2 1006.91 1486.34 Q1006.06 1486.48 1005.2 1486.48 Q1002.42 1486.48 1001.36 1484.98 Q1000.29 1483.45 1000.29 1479.38 L1000.29 1463.91 L990.733 1463.91 L990.733 1485.58 L986.381 1485.58 L986.381 1463.91 L983.071 1463.91 L983.071 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M1229.55 1481.64 L1237.19 1481.64 L1237.19 1455.28 L1228.88 1456.95 L1228.88 1452.69 L1237.14 1451.02 L1241.82 1451.02 L1241.82 1481.64 L1249.46 1481.64 L1249.46 1485.58 L1229.55 1485.58 L1229.55 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M1265.87 1451.02 L1269.81 1451.02 L1257.77 1489.98 L1253.83 1489.98 L1265.87 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M1281.84 1451.02 L1285.78 1451.02 L1273.74 1489.98 L1269.81 1489.98 L1281.84 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M1291.66 1481.64 L1299.3 1481.64 L1299.3 1455.28 L1290.99 1456.95 L1290.99 1452.69 L1299.25 1451.02 L1303.93 1451.02 L1303.93 1481.64 L1311.57 1481.64 L1311.57 1485.58 L1291.66 1485.58 L1291.66 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M1317.65 1459.65 L1342.49 1459.65 L1342.49 1463.91 L1339.23 1463.91 L1339.23 1479.84 Q1339.23 1481.51 1339.78 1482.25 Q1340.36 1482.96 1341.63 1482.96 Q1341.98 1482.96 1342.49 1482.92 Q1343 1482.85 1343.16 1482.83 L1343.16 1485.9 Q1342.35 1486.2 1341.5 1486.34 Q1340.64 1486.48 1339.78 1486.48 Q1337.01 1486.48 1335.94 1484.98 Q1334.88 1483.45 1334.88 1479.38 L1334.88 1463.91 L1325.32 1463.91 L1325.32 1485.58 L1320.96 1485.58 L1320.96 1463.91 L1317.65 1463.91 L1317.65 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M1578.49 1455.09 L1566.68 1473.54 L1578.49 1473.54 L1578.49 1455.09 M1577.26 1451.02 L1583.14 1451.02 L1583.14 1473.54 L1588.07 1473.54 L1588.07 1477.43 L1583.14 1477.43 L1583.14 1485.58 L1578.49 1485.58 L1578.49 1477.43 L1562.89 1477.43 L1562.89 1472.92 L1577.26 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M1602.77 1451.02 L1606.7 1451.02 L1594.67 1489.98 L1590.73 1489.98 L1602.77 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M1618.74 1451.02 L1622.68 1451.02 L1610.64 1489.98 L1606.7 1489.98 L1618.74 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M1641.91 1466.95 Q1645.27 1467.66 1647.14 1469.93 Q1649.04 1472.2 1649.04 1475.53 Q1649.04 1480.65 1645.52 1483.45 Q1642.01 1486.25 1635.52 1486.25 Q1633.35 1486.25 1631.03 1485.81 Q1628.74 1485.39 1626.29 1484.54 L1626.29 1480.02 Q1628.23 1481.16 1630.55 1481.74 Q1632.86 1482.32 1635.38 1482.32 Q1639.78 1482.32 1642.07 1480.58 Q1644.39 1478.84 1644.39 1475.53 Q1644.39 1472.48 1642.24 1470.77 Q1640.11 1469.03 1636.29 1469.03 L1632.26 1469.03 L1632.26 1465.19 L1636.47 1465.19 Q1639.92 1465.19 1641.75 1463.82 Q1643.58 1462.43 1643.58 1459.84 Q1643.58 1457.18 1641.68 1455.77 Q1639.81 1454.33 1636.29 1454.33 Q1634.37 1454.33 1632.17 1454.75 Q1629.97 1455.16 1627.33 1456.04 L1627.33 1451.88 Q1629.99 1451.14 1632.31 1450.77 Q1634.64 1450.39 1636.7 1450.39 Q1642.03 1450.39 1645.13 1452.83 Q1648.23 1455.23 1648.23 1459.35 Q1648.23 1462.22 1646.59 1464.21 Q1644.94 1466.18 1641.91 1466.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M1654.55 1459.65 L1679.39 1459.65 L1679.39 1463.91 L1676.13 1463.91 L1676.13 1479.84 Q1676.13 1481.51 1676.68 1482.25 Q1677.26 1482.96 1678.53 1482.96 Q1678.88 1482.96 1679.39 1482.92 Q1679.9 1482.85 1680.06 1482.83 L1680.06 1485.9 Q1679.25 1486.2 1678.39 1486.34 Q1677.54 1486.48 1676.68 1486.48 Q1673.9 1486.48 1672.84 1484.98 Q1671.77 1483.45 1671.77 1479.38 L1671.77 1463.91 L1662.21 1463.91 L1662.21 1485.58 L1657.86 1485.58 L1657.86 1463.91 L1654.55 1463.91 L1654.55 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M1900.47 1451.02 L1918.82 1451.02 L1918.82 1454.96 L1904.75 1454.96 L1904.75 1463.43 Q1905.77 1463.08 1906.79 1462.92 Q1907.8 1462.73 1908.82 1462.73 Q1914.61 1462.73 1917.99 1465.9 Q1921.37 1469.08 1921.37 1474.49 Q1921.37 1480.07 1917.9 1483.17 Q1914.42 1486.25 1908.1 1486.25 Q1905.93 1486.25 1903.66 1485.88 Q1901.41 1485.51 1899.01 1484.77 L1899.01 1480.07 Q1901.09 1481.2 1903.31 1481.76 Q1905.54 1482.32 1908.01 1482.32 Q1912.02 1482.32 1914.35 1480.21 Q1916.69 1478.1 1916.69 1474.49 Q1916.69 1470.88 1914.35 1468.77 Q1912.02 1466.67 1908.01 1466.67 Q1906.14 1466.67 1904.26 1467.08 Q1902.41 1467.5 1900.47 1468.38 L1900.47 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M1937.55 1451.02 L1941.48 1451.02 L1929.45 1489.98 L1925.51 1489.98 L1937.55 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M1953.52 1451.02 L1957.46 1451.02 L1945.42 1489.98 L1941.48 1489.98 L1953.52 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M1976.69 1466.95 Q1980.05 1467.66 1981.92 1469.93 Q1983.82 1472.2 1983.82 1475.53 Q1983.82 1480.65 1980.3 1483.45 Q1976.78 1486.25 1970.3 1486.25 Q1968.13 1486.25 1965.81 1485.81 Q1963.52 1485.39 1961.07 1484.54 L1961.07 1480.02 Q1963.01 1481.16 1965.33 1481.74 Q1967.64 1482.32 1970.16 1482.32 Q1974.56 1482.32 1976.85 1480.58 Q1979.17 1478.84 1979.17 1475.53 Q1979.17 1472.48 1977.02 1470.77 Q1974.89 1469.03 1971.07 1469.03 L1967.04 1469.03 L1967.04 1465.19 L1971.25 1465.19 Q1974.7 1465.19 1976.53 1463.82 Q1978.36 1462.43 1978.36 1459.84 Q1978.36 1457.18 1976.46 1455.77 Q1974.59 1454.33 1971.07 1454.33 Q1969.15 1454.33 1966.95 1454.75 Q1964.75 1455.16 1962.11 1456.04 L1962.11 1451.88 Q1964.77 1451.14 1967.09 1450.77 Q1969.42 1450.39 1971.48 1450.39 Q1976.81 1450.39 1979.91 1452.83 Q1983.01 1455.23 1983.01 1459.35 Q1983.01 1462.22 1981.37 1464.21 Q1979.72 1466.18 1976.69 1466.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M1989.33 1459.65 L2014.17 1459.65 L2014.17 1463.91 L2010.9 1463.91 L2010.9 1479.84 Q2010.9 1481.51 2011.46 1482.25 Q2012.04 1482.96 2013.31 1482.96 Q2013.66 1482.96 2014.17 1482.92 Q2014.68 1482.85 2014.84 1482.83 L2014.84 1485.9 Q2014.03 1486.2 2013.17 1486.34 Q2012.32 1486.48 2011.46 1486.48 Q2008.68 1486.48 2007.62 1484.98 Q2006.55 1483.45 2006.55 1479.38 L2006.55 1463.91 L1996.99 1463.91 L1996.99 1485.58 L1992.64 1485.58 L1992.64 1463.91 L1989.33 1463.91 L1989.33 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M2239.99 1481.64 L2256.31 1481.64 L2256.31 1485.58 L2234.37 1485.58 L2234.37 1481.64 Q2237.03 1478.89 2241.61 1474.26 Q2246.22 1469.61 2247.4 1468.27 Q2249.64 1465.74 2250.52 1464.01 Q2251.43 1462.25 2251.43 1460.56 Q2251.43 1457.8 2249.48 1456.07 Q2247.56 1454.33 2244.46 1454.33 Q2242.26 1454.33 2239.81 1455.09 Q2237.38 1455.86 2234.6 1457.41 L2234.6 1452.69 Q2237.42 1451.55 2239.88 1450.97 Q2242.33 1450.39 2244.37 1450.39 Q2249.74 1450.39 2252.93 1453.08 Q2256.12 1455.77 2256.12 1460.26 Q2256.12 1462.39 2255.31 1464.31 Q2254.53 1466.2 2252.42 1468.8 Q2251.84 1469.47 2248.74 1472.69 Q2245.64 1475.88 2239.99 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M2273.09 1451.02 L2277.03 1451.02 L2264.99 1489.98 L2261.06 1489.98 L2273.09 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M2289.06 1451.02 L2293 1451.02 L2280.96 1489.98 L2277.03 1489.98 L2289.06 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M2298.88 1481.64 L2306.52 1481.64 L2306.52 1455.28 L2298.21 1456.95 L2298.21 1452.69 L2306.47 1451.02 L2311.15 1451.02 L2311.15 1481.64 L2318.79 1481.64 L2318.79 1485.58 L2298.88 1485.58 L2298.88 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M2324.87 1459.65 L2349.71 1459.65 L2349.71 1463.91 L2346.45 1463.91 L2346.45 1479.84 Q2346.45 1481.51 2347 1482.25 Q2347.58 1482.96 2348.86 1482.96 Q2349.2 1482.96 2349.71 1482.92 Q2350.22 1482.85 2350.38 1482.83 L2350.38 1485.9 Q2349.57 1486.2 2348.72 1486.34 Q2347.86 1486.48 2347 1486.48 Q2344.23 1486.48 2343.16 1484.98 Q2342.1 1483.45 2342.1 1479.38 L2342.1 1463.91 L2332.54 1463.91 L2332.54 1485.58 L2328.18 1485.58 L2328.18 1463.91 L2324.87 1463.91 L2324.87 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M1296.14 1545.45 L1275.87 1545.45 Q1276.35 1554.96 1278.54 1559 Q1281.28 1563.97 1286.02 1563.97 Q1290.8 1563.97 1293.44 1558.97 Q1295.76 1554.58 1296.14 1545.45 M1296.05 1540.03 Q1295.16 1531 1293.44 1527.81 Q1290.7 1522.78 1286.02 1522.78 Q1281.15 1522.78 1278.57 1527.75 Q1276.54 1531.76 1275.93 1540.03 L1296.05 1540.03 M1286.02 1518.01 Q1293.66 1518.01 1298.02 1524.76 Q1302.38 1531.47 1302.38 1543.38 Q1302.38 1555.25 1298.02 1562 Q1293.66 1568.78 1286.02 1568.78 Q1278.35 1568.78 1274.02 1562 Q1269.66 1555.25 1269.66 1543.38 Q1269.66 1531.47 1274.02 1524.76 Q1278.35 1518.01 1286.02 1518.01 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><polyline clip-path="url(#clip650)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="219.288,1423.18 219.288,47.2441 "/>
<polyline clip-path="url(#clip650)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="219.288,1398.58 238.185,1398.58 "/>
<polyline clip-path="url(#clip650)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="219.288,1012.49 238.185,1012.49 "/>
<polyline clip-path="url(#clip650)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="219.288,626.389 238.185,626.389 "/>
<polyline clip-path="url(#clip650)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="219.288,240.292 238.185,240.292 "/>
<path clip-path="url(#clip650)" d="M127.015 1384.38 Q123.404 1384.38 121.575 1387.95 Q119.769 1391.49 119.769 1398.62 Q119.769 1405.72 121.575 1409.29 Q123.404 1412.83 127.015 1412.83 Q130.649 1412.83 132.455 1409.29 Q134.283 1405.72 134.283 1398.62 Q134.283 1391.49 132.455 1387.95 Q130.649 1384.38 127.015 1384.38 M127.015 1380.68 Q132.825 1380.68 135.88 1385.28 Q138.959 1389.87 138.959 1398.62 Q138.959 1407.34 135.88 1411.95 Q132.825 1416.53 127.015 1416.53 Q121.205 1416.53 118.126 1411.95 Q115.07 1407.34 115.07 1398.62 Q115.07 1389.87 118.126 1385.28 Q121.205 1380.68 127.015 1380.68 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M147.177 1409.98 L152.061 1409.98 L152.061 1415.86 L147.177 1415.86 L147.177 1409.98 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M161.065 1381.3 L183.288 1381.3 L183.288 1383.29 L170.741 1415.86 L165.857 1415.86 L177.663 1385.24 L161.065 1385.24 L161.065 1381.3 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M126.205 998.284 Q122.593 998.284 120.765 1001.85 Q118.959 1005.39 118.959 1012.52 Q118.959 1019.63 120.765 1023.19 Q122.593 1026.73 126.205 1026.73 Q129.839 1026.73 131.644 1023.19 Q133.473 1019.63 133.473 1012.52 Q133.473 1005.39 131.644 1001.85 Q129.839 998.284 126.205 998.284 M126.205 994.581 Q132.015 994.581 135.07 999.187 Q138.149 1003.77 138.149 1012.52 Q138.149 1021.25 135.07 1025.85 Q132.015 1030.44 126.205 1030.44 Q120.394 1030.44 117.316 1025.85 Q114.26 1021.25 114.26 1012.52 Q114.26 1003.77 117.316 999.187 Q120.394 994.581 126.205 994.581 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M146.366 1023.89 L151.251 1023.89 L151.251 1029.77 L146.366 1029.77 L146.366 1023.89 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M171.436 1013.35 Q168.102 1013.35 166.181 1015.14 Q164.283 1016.92 164.283 1020.04 Q164.283 1023.17 166.181 1024.95 Q168.102 1026.73 171.436 1026.73 Q174.769 1026.73 176.69 1024.95 Q178.612 1023.15 178.612 1020.04 Q178.612 1016.92 176.69 1015.14 Q174.792 1013.35 171.436 1013.35 M166.76 1011.36 Q163.751 1010.62 162.061 1008.56 Q160.394 1006.5 160.394 1003.54 Q160.394 999.395 163.334 996.988 Q166.297 994.581 171.436 994.581 Q176.598 994.581 179.538 996.988 Q182.477 999.395 182.477 1003.54 Q182.477 1006.5 180.788 1008.56 Q179.121 1010.62 176.135 1011.36 Q179.514 1012.15 181.389 1014.44 Q183.288 1016.73 183.288 1020.04 Q183.288 1025.07 180.209 1027.75 Q177.153 1030.44 171.436 1030.44 Q165.718 1030.44 162.64 1027.75 Q159.584 1025.07 159.584 1020.04 Q159.584 1016.73 161.482 1014.44 Q163.38 1012.15 166.76 1011.36 M165.047 1003.98 Q165.047 1006.66 166.714 1008.17 Q168.403 1009.67 171.436 1009.67 Q174.445 1009.67 176.135 1008.17 Q177.848 1006.66 177.848 1003.98 Q177.848 1001.29 176.135 999.789 Q174.445 998.284 171.436 998.284 Q168.403 998.284 166.714 999.789 Q165.047 1001.29 165.047 1003.98 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M126.297 612.188 Q122.686 612.188 120.857 615.752 Q119.052 619.294 119.052 626.424 Q119.052 633.53 120.857 637.095 Q122.686 640.637 126.297 640.637 Q129.931 640.637 131.737 637.095 Q133.566 633.53 133.566 626.424 Q133.566 619.294 131.737 615.752 Q129.931 612.188 126.297 612.188 M126.297 608.484 Q132.107 608.484 135.163 613.09 Q138.242 617.674 138.242 626.424 Q138.242 635.15 135.163 639.757 Q132.107 644.34 126.297 644.34 Q120.487 644.34 117.408 639.757 Q114.353 635.15 114.353 626.424 Q114.353 617.674 117.408 613.09 Q120.487 608.484 126.297 608.484 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M146.459 637.789 L151.343 637.789 L151.343 643.669 L146.459 643.669 L146.459 637.789 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M161.667 642.951 L161.667 638.692 Q163.427 639.525 165.232 639.965 Q167.038 640.405 168.774 640.405 Q173.403 640.405 175.834 637.303 Q178.288 634.178 178.635 627.836 Q177.292 629.826 175.232 630.891 Q173.172 631.956 170.672 631.956 Q165.487 631.956 162.454 628.831 Q159.445 625.683 159.445 620.243 Q159.445 614.919 162.593 611.702 Q165.741 608.484 170.973 608.484 Q176.968 608.484 180.116 613.09 Q183.288 617.674 183.288 626.424 Q183.288 634.595 179.399 639.479 Q175.533 644.34 168.982 644.34 Q167.223 644.34 165.417 643.993 Q163.612 643.646 161.667 642.951 M170.973 628.299 Q174.121 628.299 175.95 626.146 Q177.801 623.993 177.801 620.243 Q177.801 616.516 175.95 614.364 Q174.121 612.188 170.973 612.188 Q167.825 612.188 165.973 614.364 Q164.144 616.516 164.144 620.243 Q164.144 623.993 165.973 626.146 Q167.825 628.299 170.973 628.299 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M116.922 253.637 L124.561 253.637 L124.561 227.272 L116.251 228.938 L116.251 224.679 L124.515 223.012 L129.191 223.012 L129.191 253.637 L136.829 253.637 L136.829 257.572 L116.922 257.572 L116.922 253.637 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M146.274 251.693 L151.158 251.693 L151.158 257.572 L146.274 257.572 L146.274 251.693 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M171.343 226.091 Q167.732 226.091 165.903 229.656 Q164.098 233.198 164.098 240.327 Q164.098 247.434 165.903 250.998 Q167.732 254.54 171.343 254.54 Q174.977 254.54 176.783 250.998 Q178.612 247.434 178.612 240.327 Q178.612 233.198 176.783 229.656 Q174.977 226.091 171.343 226.091 M171.343 222.387 Q177.153 222.387 180.209 226.994 Q183.288 231.577 183.288 240.327 Q183.288 249.054 180.209 253.66 Q177.153 258.244 171.343 258.244 Q165.533 258.244 162.454 253.66 Q159.399 249.054 159.399 240.327 Q159.399 231.577 162.454 226.994 Q165.533 222.387 171.343 222.387 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M33.8307 724.772 Q33.2578 725.759 33.0032 726.937 Q32.7167 728.082 32.7167 729.483 Q32.7167 734.448 35.9632 737.122 Q39.1779 739.763 45.2253 739.763 L64.0042 739.763 L64.0042 745.652 L28.3562 745.652 L28.3562 739.763 L33.8944 739.763 Q30.6479 737.917 29.0883 734.957 Q27.4968 731.997 27.4968 727.764 Q27.4968 727.159 27.5923 726.427 Q27.656 725.695 27.8151 724.804 L33.8307 724.772 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><line clip-path="url(#clip652)" x1="2292.37" y1="240.292" x2="2292.37" y2="224.292" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="2292.37" y1="240.292" x2="2276.37" y2="240.292" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="2292.37" y1="240.292" x2="2292.37" y2="256.292" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="2292.37" y1="240.292" x2="2308.37" y2="240.292" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1286.02" y1="240.38" x2="1286.02" y2="224.38" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1286.02" y1="240.38" x2="1270.02" y2="240.38" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1286.02" y1="240.38" x2="1286.02" y2="256.38" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1286.02" y1="240.38" x2="1302.02" y2="240.38" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="792.219" y1="386.435" x2="792.219" y2="370.435" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="792.219" y1="386.435" x2="776.219" y2="386.435" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="792.219" y1="386.435" x2="792.219" y2="402.435" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="792.219" y1="386.435" x2="808.219" y2="386.435" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1779.82" y1="386.435" x2="1779.82" y2="370.435" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1779.82" y1="386.435" x2="1763.82" y2="386.435" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1779.82" y1="386.435" x2="1779.82" y2="402.435" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1779.82" y1="386.435" x2="1795.82" y2="386.435" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="773.472" y1="386.45" x2="773.472" y2="370.45" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="773.472" y1="386.45" x2="757.472" y2="386.45" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="773.472" y1="386.45" x2="773.472" y2="402.45" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="773.472" y1="386.45" x2="789.472" y2="386.45" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1798.57" y1="386.45" x2="1798.57" y2="370.45" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1798.57" y1="386.45" x2="1782.57" y2="386.45" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1798.57" y1="386.45" x2="1798.57" y2="402.45" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1798.57" y1="386.45" x2="1814.57" y2="386.45" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="2292.37" y1="751.782" x2="2292.37" y2="735.782" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="2292.37" y1="751.782" x2="2276.37" y2="751.782" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="2292.37" y1="751.782" x2="2292.37" y2="767.782" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="2292.37" y1="751.782" x2="2308.37" y2="751.782" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1286.02" y1="751.783" x2="1286.02" y2="735.783" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1286.02" y1="751.783" x2="1270.02" y2="751.783" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1286.02" y1="751.783" x2="1286.02" y2="767.783" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1286.02" y1="751.783" x2="1302.02" y2="751.783" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1303.49" y1="852.693" x2="1303.49" y2="836.693" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1303.49" y1="852.693" x2="1287.49" y2="852.693" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1303.49" y1="852.693" x2="1303.49" y2="868.693" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1303.49" y1="852.693" x2="1319.49" y2="852.693" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1268.55" y1="852.693" x2="1268.55" y2="836.693" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1268.55" y1="852.693" x2="1252.55" y2="852.693" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1268.55" y1="852.693" x2="1268.55" y2="868.693" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1268.55" y1="852.693" x2="1284.55" y2="852.693" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="297.141" y1="852.711" x2="297.141" y2="836.711" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="297.141" y1="852.711" x2="281.141" y2="852.711" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="297.141" y1="852.711" x2="297.141" y2="868.711" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="297.141" y1="852.711" x2="313.141" y2="852.711" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="2274.9" y1="852.711" x2="2274.9" y2="836.711" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="2274.9" y1="852.711" x2="2258.9" y2="852.711" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="2274.9" y1="852.711" x2="2274.9" y2="868.711" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="2274.9" y1="852.711" x2="2290.9" y2="852.711" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="792.309" y1="978.924" x2="792.309" y2="962.924" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="792.309" y1="978.924" x2="776.309" y2="978.924" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="792.309" y1="978.924" x2="792.309" y2="994.924" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="792.309" y1="978.924" x2="808.309" y2="978.924" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1779.73" y1="978.924" x2="1779.73" y2="962.924" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1779.73" y1="978.924" x2="1763.73" y2="978.924" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1779.73" y1="978.924" x2="1779.73" y2="994.924" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1779.73" y1="978.924" x2="1795.73" y2="978.924" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="773.385" y1="978.924" x2="773.385" y2="962.924" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="773.385" y1="978.924" x2="757.385" y2="978.924" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="773.385" y1="978.924" x2="773.385" y2="994.924" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="773.385" y1="978.924" x2="789.385" y2="978.924" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1798.66" y1="978.924" x2="1798.66" y2="962.924" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1798.66" y1="978.924" x2="1782.66" y2="978.924" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1798.66" y1="978.924" x2="1798.66" y2="994.924" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1798.66" y1="978.924" x2="1814.66" y2="978.924" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="279.669" y1="1343.45" x2="279.669" y2="1327.45" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="279.669" y1="1343.45" x2="263.669" y2="1343.45" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="279.669" y1="1343.45" x2="279.669" y2="1359.45" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="279.669" y1="1343.45" x2="295.669" y2="1343.45" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1286.02" y1="1343.9" x2="1286.02" y2="1327.9" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1286.02" y1="1343.9" x2="1270.02" y2="1343.9" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1286.02" y1="1343.9" x2="1286.02" y2="1359.9" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1286.02" y1="1343.9" x2="1302.02" y2="1343.9" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="279.669" y1="1415.6" x2="279.669" y2="1399.6" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="279.669" y1="1415.6" x2="263.669" y2="1415.6" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="279.669" y1="1415.6" x2="279.669" y2="1431.6" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="279.669" y1="1415.6" x2="295.669" y2="1415.6" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1286.02" y1="1415.76" x2="1286.02" y2="1399.76" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1286.02" y1="1415.76" x2="1270.02" y2="1415.76" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1286.02" y1="1415.76" x2="1286.02" y2="1431.76" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1286.02" y1="1415.76" x2="1302.02" y2="1415.76" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="774.965" y1="1423.16" x2="774.965" y2="1407.16" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="774.965" y1="1423.16" x2="758.965" y2="1423.16" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="774.965" y1="1423.16" x2="774.965" y2="1439.16" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="774.965" y1="1423.16" x2="790.965" y2="1423.16" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1797.08" y1="1423.16" x2="1797.08" y2="1407.16" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1797.08" y1="1423.16" x2="1781.08" y2="1423.16" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1797.08" y1="1423.16" x2="1797.08" y2="1439.16" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1797.08" y1="1423.16" x2="1813.08" y2="1423.16" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1781.32" y1="1423.18" x2="1781.32" y2="1407.18" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1781.32" y1="1423.18" x2="1765.32" y2="1423.18" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1781.32" y1="1423.18" x2="1781.32" y2="1439.18" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="1781.32" y1="1423.18" x2="1797.32" y2="1423.18" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="790.727" y1="1423.18" x2="790.727" y2="1407.18" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="790.727" y1="1423.18" x2="774.727" y2="1423.18" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="790.727" y1="1423.18" x2="790.727" y2="1439.18" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip652)" x1="790.727" y1="1423.18" x2="806.727" y2="1423.18" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<path clip-path="url(#clip650)" d="M290.403 196.789 L674.907 196.789 L674.907 93.1086 L290.403 93.1086  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<polyline clip-path="url(#clip650)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="290.403,196.789 674.907,196.789 674.907,93.1086 290.403,93.1086 290.403,196.789 "/>
<line clip-path="url(#clip650)" x1="385.224" y1="144.949" x2="385.224" y2="122.193" style="stroke:#009af9; stroke-width:4.55111; stroke-opacity:1"/>
<line clip-path="url(#clip650)" x1="385.224" y1="144.949" x2="362.468" y2="144.949" style="stroke:#009af9; stroke-width:4.55111; stroke-opacity:1"/>
<line clip-path="url(#clip650)" x1="385.224" y1="144.949" x2="385.224" y2="167.704" style="stroke:#009af9; stroke-width:4.55111; stroke-opacity:1"/>
<line clip-path="url(#clip650)" x1="385.224" y1="144.949" x2="407.98" y2="144.949" style="stroke:#009af9; stroke-width:4.55111; stroke-opacity:1"/>
<path clip-path="url(#clip650)" d="M480.045 127.669 L509.281 127.669 L509.281 131.604 L497.012 131.604 L497.012 162.229 L492.313 162.229 L492.313 131.604 L480.045 131.604 L480.045 127.669 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M521.665 140.284 Q520.947 139.868 520.091 139.682 Q519.258 139.474 518.239 139.474 Q514.628 139.474 512.684 141.835 Q510.762 144.173 510.762 148.571 L510.762 162.229 L506.48 162.229 L506.48 136.303 L510.762 136.303 L510.762 140.331 Q512.105 137.969 514.258 136.835 Q516.41 135.678 519.489 135.678 Q519.929 135.678 520.461 135.747 Q520.994 135.794 521.642 135.909 L521.665 140.284 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M526.133 136.303 L530.392 136.303 L530.392 162.229 L526.133 162.229 L526.133 136.303 M526.133 126.21 L530.392 126.21 L530.392 131.604 L526.133 131.604 L526.133 126.21 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M536.248 136.303 L540.762 136.303 L548.864 158.062 L556.966 136.303 L561.48 136.303 L551.757 162.229 L545.97 162.229 L536.248 136.303 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M567.359 136.303 L571.618 136.303 L571.618 162.229 L567.359 162.229 L567.359 136.303 M567.359 126.21 L571.618 126.21 L571.618 131.604 L567.359 131.604 L567.359 126.21 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M592.313 149.196 Q587.151 149.196 585.16 150.377 Q583.169 151.557 583.169 154.405 Q583.169 156.673 584.651 158.016 Q586.155 159.335 588.725 159.335 Q592.266 159.335 594.396 156.835 Q596.549 154.312 596.549 150.145 L596.549 149.196 L592.313 149.196 M600.808 147.437 L600.808 162.229 L596.549 162.229 L596.549 158.293 Q595.09 160.655 592.915 161.789 Q590.739 162.9 587.591 162.9 Q583.609 162.9 581.248 160.678 Q578.91 158.432 578.91 154.682 Q578.91 150.307 581.827 148.085 Q584.766 145.863 590.577 145.863 L596.549 145.863 L596.549 145.446 Q596.549 142.507 594.604 140.909 Q592.683 139.289 589.188 139.289 Q586.966 139.289 584.859 139.821 Q582.753 140.354 580.808 141.419 L580.808 137.483 Q583.146 136.581 585.345 136.141 Q587.544 135.678 589.628 135.678 Q595.253 135.678 598.03 138.594 Q600.808 141.511 600.808 147.437 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M609.581 126.21 L613.84 126.21 L613.84 162.229 L609.581 162.229 L609.581 126.21 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M632.984 126.257 Q629.882 131.581 628.377 136.789 Q626.873 141.997 626.873 147.344 Q626.873 152.692 628.377 157.946 Q629.905 163.178 632.984 168.479 L629.28 168.479 Q625.808 163.039 624.072 157.784 Q622.359 152.53 622.359 147.344 Q622.359 142.182 624.072 136.951 Q625.785 131.72 629.28 126.257 L632.984 126.257 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M640.576 126.257 L644.28 126.257 Q647.752 131.72 649.465 136.951 Q651.201 142.182 651.201 147.344 Q651.201 152.53 649.465 157.784 Q647.752 163.039 644.28 168.479 L640.576 168.479 Q643.655 163.178 645.16 157.946 Q646.687 152.692 646.687 147.344 Q646.687 141.997 645.16 136.789 Q643.655 131.581 640.576 126.257 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /></svg>

```

We can clearly see multiple eigenvalues close to the unit circle.
Our state is close to being non-injective, and represents the sum of multiple injective states.
This is numerically very problematic, but also indicates that we used an incorrect ansatz to approximate the groundstate.
We should retry with a larger unit cell.

## Success

Let's initialize a different initial state, this time with a 2-site unit cell:

````julia
A = TensorMap(rand, ComplexF64, ℂ^20 * ℂ^2, ℂ^20);
B = TensorMap(rand, ComplexF64, ℂ^20 * ℂ^2, ℂ^20);
state = InfiniteMPS([A, B]);
````

In MPSKit, we require that the periodicity of the hamiltonian equals that of the state it is applied to.
This is not a big obstacle, you can simply repeat the original hamiltonian.
Alternatively, the hamiltonian can be constructed directly on a two-site unitcell by making use of MPSKitModels.jl's `@mpoham`.

````julia
# H2 = repeat(H, 2); -- copies the one-site version
H2 = heisenberg_XXX(ComplexF64, Trivial, InfiniteChain(2); spin=1 // 2)
groundstate, cache, delta = find_groundstate(state, H2,
                                             VUMPS(; maxiter=100, tol=1e-12));
````

````
[ Info: VUMPS init:	obj = +4.994034626541e-01	err = 5.1147e-02
[ Info: VUMPS   1:	obj = -6.262869203043e-02	err = 3.8580046779e-01	time = 0.11 sec
[ Info: VUMPS   2:	obj = -8.607141221574e-01	err = 1.2356188469e-01	time = 0.02 sec
[ Info: VUMPS   3:	obj = -8.851326788898e-01	err = 1.3843868842e-02	time = 0.02 sec
[ Info: VUMPS   4:	obj = -8.859081586783e-01	err = 6.1401047820e-03	time = 0.03 sec
[ Info: VUMPS   5:	obj = -8.861065721524e-01	err = 4.0177146467e-03	time = 0.03 sec
[ Info: VUMPS   6:	obj = -8.861785252458e-01	err = 2.7819071955e-03	time = 0.05 sec
[ Info: VUMPS   7:	obj = -8.862090474504e-01	err = 2.0054538265e-03	time = 0.09 sec
[ Info: VUMPS   8:	obj = -8.862231227796e-01	err = 1.4866506538e-03	time = 0.03 sec
[ Info: VUMPS   9:	obj = -8.862299174133e-01	err = 1.0781153418e-03	time = 0.03 sec
[ Info: VUMPS  10:	obj = -8.862332526284e-01	err = 8.0176561905e-04	time = 0.04 sec
[ Info: VUMPS  11:	obj = -8.862349227193e-01	err = 5.9244791851e-04	time = 0.07 sec
[ Info: VUMPS  12:	obj = -8.862357635094e-01	err = 4.4166081769e-04	time = 0.04 sec
[ Info: VUMPS  13:	obj = -8.862361924426e-01	err = 3.4628140735e-04	time = 0.04 sec
[ Info: VUMPS  14:	obj = -8.862364148218e-01	err = 2.7392973609e-04	time = 0.07 sec
[ Info: VUMPS  15:	obj = -8.862365333782e-01	err = 2.3548153669e-04	time = 0.04 sec
[ Info: VUMPS  16:	obj = -8.862366003882e-01	err = 2.0924520967e-04	time = 0.07 sec
[ Info: VUMPS  17:	obj = -8.862366415980e-01	err = 1.9635595936e-04	time = 0.04 sec
[ Info: VUMPS  18:	obj = -8.862366705874e-01	err = 1.9049012489e-04	time = 0.07 sec
[ Info: VUMPS  19:	obj = -8.862366940299e-01	err = 1.8899241029e-04	time = 0.04 sec
[ Info: VUMPS  20:	obj = -8.862367155996e-01	err = 1.9097122153e-04	time = 0.07 sec
[ Info: VUMPS  21:	obj = -8.862367373638e-01	err = 1.9481944475e-04	time = 0.05 sec
[ Info: VUMPS  22:	obj = -8.862367604209e-01	err = 2.0067663970e-04	time = 0.09 sec
[ Info: VUMPS  23:	obj = -8.862367858969e-01	err = 2.0818860599e-04	time = 0.04 sec
[ Info: VUMPS  24:	obj = -8.862368141807e-01	err = 2.1736488170e-04	time = 0.07 sec
[ Info: VUMPS  25:	obj = -8.862368464497e-01	err = 2.2859478788e-04	time = 0.04 sec
[ Info: VUMPS  26:	obj = -8.862368828456e-01	err = 2.4175658730e-04	time = 0.07 sec
[ Info: VUMPS  27:	obj = -8.862369249947e-01	err = 2.5762627945e-04	time = 0.04 sec
[ Info: VUMPS  28:	obj = -8.862369728770e-01	err = 2.7619128372e-04	time = 0.07 sec
[ Info: VUMPS  29:	obj = -8.862370289739e-01	err = 2.9841300959e-04	time = 0.05 sec
[ Info: VUMPS  30:	obj = -8.862370930650e-01	err = 3.2477737551e-04	time = 0.07 sec
[ Info: VUMPS  31:	obj = -8.862371692951e-01	err = 3.5590713379e-04	time = 0.04 sec
[ Info: VUMPS  32:	obj = -8.862372570904e-01	err = 3.9353059847e-04	time = 0.07 sec
[ Info: VUMPS  33:	obj = -8.862373640071e-01	err = 4.3603005725e-04	time = 0.04 sec
[ Info: VUMPS  34:	obj = -8.862374883321e-01	err = 4.8815487884e-04	time = 0.07 sec
[ Info: VUMPS  35:	obj = -8.862376445282e-01	err = 5.3937202139e-04	time = 0.04 sec
[ Info: VUMPS  36:	obj = -8.862378254047e-01	err = 6.0151343771e-04	time = 0.04 sec
[ Info: VUMPS  37:	obj = -8.862380560924e-01	err = 6.3791369135e-04	time = 0.07 sec
[ Info: VUMPS  38:	obj = -8.862383068811e-01	err = 6.7996919433e-04	time = 0.04 sec
[ Info: VUMPS  39:	obj = -8.862386058811e-01	err = 6.4634261288e-04	time = 0.08 sec
[ Info: VUMPS  40:	obj = -8.862388802875e-01	err = 6.2080152793e-04	time = 0.04 sec
[ Info: VUMPS  41:	obj = -8.862391460561e-01	err = 5.0675637451e-04	time = 0.08 sec
[ Info: VUMPS  42:	obj = -8.862393416168e-01	err = 4.2985263716e-04	time = 0.04 sec
[ Info: VUMPS  43:	obj = -8.862394885298e-01	err = 3.1911525906e-04	time = 0.08 sec
[ Info: VUMPS  44:	obj = -8.862395827629e-01	err = 2.5276803245e-04	time = 0.04 sec
[ Info: VUMPS  45:	obj = -8.862396443908e-01	err = 1.8512521569e-04	time = 0.08 sec
[ Info: VUMPS  46:	obj = -8.862396829362e-01	err = 1.4700821652e-04	time = 0.04 sec
[ Info: VUMPS  47:	obj = -8.862397077973e-01	err = 1.1022783530e-04	time = 0.07 sec
[ Info: VUMPS  48:	obj = -8.862397240917e-01	err = 9.2219279764e-05	time = 0.04 sec
[ Info: VUMPS  49:	obj = -8.862397352818e-01	err = 7.1892346813e-05	time = 0.07 sec
[ Info: VUMPS  50:	obj = -8.862397433166e-01	err = 6.5073723507e-05	time = 0.04 sec
[ Info: VUMPS  51:	obj = -8.862397493970e-01	err = 5.5282518065e-05	time = 0.08 sec
[ Info: VUMPS  52:	obj = -8.862397542161e-01	err = 5.1456531474e-05	time = 0.04 sec
[ Info: VUMPS  53:	obj = -8.862397581953e-01	err = 4.6714202684e-05	time = 0.08 sec
[ Info: VUMPS  54:	obj = -8.862397615871e-01	err = 4.3942466103e-05	time = 0.04 sec
[ Info: VUMPS  55:	obj = -8.862397645493e-01	err = 4.1177713870e-05	time = 0.07 sec
[ Info: VUMPS  56:	obj = -8.862397671820e-01	err = 3.9114982243e-05	time = 0.04 sec
[ Info: VUMPS  57:	obj = -8.862397695517e-01	err = 3.7186097920e-05	time = 0.07 sec
[ Info: VUMPS  58:	obj = -8.862397717040e-01	err = 3.5553409941e-05	time = 0.04 sec
[ Info: VUMPS  59:	obj = -8.862397736718e-01	err = 3.4038601553e-05	time = 0.08 sec
[ Info: VUMPS  60:	obj = -8.862397754798e-01	err = 3.2679861075e-05	time = 0.04 sec
[ Info: VUMPS  61:	obj = -8.862397771471e-01	err = 3.1412139117e-05	time = 0.07 sec
[ Info: VUMPS  62:	obj = -8.862397786894e-01	err = 3.0244528404e-05	time = 0.04 sec
[ Info: VUMPS  63:	obj = -8.862397801195e-01	err = 2.9148581648e-05	time = 0.07 sec
[ Info: VUMPS  64:	obj = -8.862397814483e-01	err = 2.8125831750e-05	time = 0.04 sec
[ Info: VUMPS  65:	obj = -8.862397826855e-01	err = 2.7161607890e-05	time = 0.07 sec
[ Info: VUMPS  66:	obj = -8.862397838391e-01	err = 2.6254981486e-05	time = 0.04 sec
[ Info: VUMPS  67:	obj = -8.862397849165e-01	err = 2.5397377333e-05	time = 0.07 sec
[ Info: VUMPS  68:	obj = -8.862397859242e-01	err = 2.4587152172e-05	time = 0.04 sec
[ Info: VUMPS  69:	obj = -8.862397868680e-01	err = 2.3818718802e-05	time = 0.07 sec
[ Info: VUMPS  70:	obj = -8.862397877531e-01	err = 2.3090202212e-05	time = 0.04 sec
[ Info: VUMPS  71:	obj = -8.862397885843e-01	err = 2.2397708419e-05	time = 0.07 sec
[ Info: VUMPS  72:	obj = -8.862397893656e-01	err = 2.1739356071e-05	time = 0.04 sec
[ Info: VUMPS  73:	obj = -8.862397901011e-01	err = 2.1112310644e-05	time = 0.07 sec
[ Info: VUMPS  74:	obj = -8.862397907941e-01	err = 2.0514795921e-05	time = 0.04 sec
[ Info: VUMPS  75:	obj = -8.862397914478e-01	err = 1.9944644746e-05	time = 0.07 sec
[ Info: VUMPS  76:	obj = -8.862397920651e-01	err = 1.9400238943e-05	time = 0.04 sec
[ Info: VUMPS  77:	obj = -8.862397926487e-01	err = 1.8879869617e-05	time = 0.07 sec
[ Info: VUMPS  78:	obj = -8.862397932009e-01	err = 1.8382107216e-05	time = 0.04 sec
[ Info: VUMPS  79:	obj = -8.862397937239e-01	err = 1.7905544693e-05	time = 0.07 sec
[ Info: VUMPS  80:	obj = -8.862397942198e-01	err = 1.7448935744e-05	time = 0.04 sec
[ Info: VUMPS  81:	obj = -8.862397946902e-01	err = 1.7011113669e-05	time = 0.07 sec
[ Info: VUMPS  82:	obj = -8.862397951371e-01	err = 1.6590972981e-05	time = 0.04 sec
[ Info: VUMPS  83:	obj = -8.862397955618e-01	err = 1.6187537766e-05	time = 0.07 sec
[ Info: VUMPS  84:	obj = -8.862397959658e-01	err = 1.5799851792e-05	time = 0.04 sec
[ Info: VUMPS  85:	obj = -8.862397963505e-01	err = 1.5427072391e-05	time = 0.07 sec
[ Info: VUMPS  86:	obj = -8.862397967170e-01	err = 1.5068369727e-05	time = 0.06 sec
[ Info: VUMPS  87:	obj = -8.862397970664e-01	err = 1.4723011390e-05	time = 0.09 sec
[ Info: VUMPS  88:	obj = -8.862397973998e-01	err = 1.4390277758e-05	time = 0.04 sec
[ Info: VUMPS  89:	obj = -8.862397977181e-01	err = 1.4069534998e-05	time = 0.07 sec
[ Info: VUMPS  90:	obj = -8.862397980223e-01	err = 1.3760148626e-05	time = 0.04 sec
[ Info: VUMPS  91:	obj = -8.862397983131e-01	err = 1.3461564491e-05	time = 0.07 sec
[ Info: VUMPS  92:	obj = -8.862397985912e-01	err = 1.3173227645e-05	time = 0.04 sec
[ Info: VUMPS  93:	obj = -8.862397988575e-01	err = 1.2894656269e-05	time = 0.07 sec
[ Info: VUMPS  94:	obj = -8.862397991125e-01	err = 1.2625358983e-05	time = 0.04 sec
[ Info: VUMPS  95:	obj = -8.862397993569e-01	err = 1.2364911249e-05	time = 0.07 sec
[ Info: VUMPS  96:	obj = -8.862397995913e-01	err = 1.2112879975e-05	time = 0.04 sec
[ Info: VUMPS  97:	obj = -8.862397998161e-01	err = 1.1868886466e-05	time = 0.07 sec
[ Info: VUMPS  98:	obj = -8.862398000318e-01	err = 1.1632550359e-05	time = 0.04 sec
[ Info: VUMPS  99:	obj = -8.862398002390e-01	err = 1.1403537234e-05	time = 0.07 sec
┌ Warning: VUMPS cancel 100:	obj = -8.862398004381e-01	err = 1.1181508752e-05	time = 5.61 sec
└ @ MPSKit ~/Projects/MPSKit.jl/src/algorithms/groundstate/vumps.jl:67

````

We get convergence, but it takes an enormous amount of iterations.
The reason behind this becomes more obvious at higher bond dimensions:

````julia
groundstate, cache, delta = find_groundstate(state, H2,
                                             IDMRG2(; trscheme=truncdim(50), maxiter=20,
                                                    tol=1e-12))
entanglementplot(groundstate)
````

```@raw html
<?xml version="1.0" encoding="utf-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="600" height="400" viewBox="0 0 2400 1600">
<defs>
  <clipPath id="clip690">
    <rect x="0" y="0" width="2400" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip690)" d="M0 1600 L2400 1600 L2400 0 L0 0  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip691">
    <rect x="480" y="0" width="1681" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip690)" d="M189.496 1352.62 L2352.76 1352.62 L2352.76 123.472 L189.496 123.472  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip692">
    <rect x="189" y="123" width="2164" height="1230"/>
  </clipPath>
</defs>
<polyline clip-path="url(#clip692)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="250.72,1352.62 250.72,123.472 "/>
<polyline clip-path="url(#clip692)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="189.496,1348.61 2352.76,1348.61 "/>
<polyline clip-path="url(#clip692)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="189.496,1054.07 2352.76,1054.07 "/>
<polyline clip-path="url(#clip692)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="189.496,759.533 2352.76,759.533 "/>
<polyline clip-path="url(#clip692)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="189.496,464.992 2352.76,464.992 "/>
<polyline clip-path="url(#clip692)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="189.496,170.451 2352.76,170.451 "/>
<polyline clip-path="url(#clip690)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="189.496,1352.62 2352.76,1352.62 "/>
<polyline clip-path="url(#clip690)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="250.72,1352.62 250.72,1371.52 "/>
<path clip-path="url(#clip690)" d="M117.476 1508.55 L138.148 1487.88 L140.931 1490.66 L132.256 1499.34 L153.911 1520.99 L150.588 1524.32 L128.933 1502.66 L120.258 1511.34 L117.476 1508.55 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M155.826 1488.04 Q155.024 1488.26 154.287 1488.73 Q153.551 1489.17 152.831 1489.89 Q150.277 1492.45 150.572 1495.49 Q150.867 1498.5 153.976 1501.61 L163.634 1511.27 L160.606 1514.3 L142.273 1495.97 L145.301 1492.94 L148.149 1495.79 Q147.429 1493.17 148.149 1490.84 Q148.853 1488.5 151.03 1486.33 Q151.341 1486.01 151.767 1485.69 Q152.176 1485.34 152.716 1484.97 L155.826 1488.04 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M156.17 1482.07 L159.182 1479.06 L177.514 1497.39 L174.502 1500.4 L156.17 1482.07 M149.033 1474.93 L152.045 1471.92 L155.859 1475.74 L152.847 1478.75 L149.033 1474.93 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M163.323 1474.92 L166.514 1471.73 L187.629 1481.38 L177.972 1460.27 L181.164 1457.08 L192.622 1482.28 L188.53 1486.37 L163.323 1474.92 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M185.321 1452.92 L188.333 1449.91 L206.665 1468.24 L203.654 1471.25 L185.321 1452.92 M178.185 1445.78 L181.197 1442.77 L185.01 1446.58 L181.999 1449.6 L178.185 1445.78 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M212.083 1444.39 Q208.433 1448.04 207.86 1450.28 Q207.287 1452.53 209.301 1454.54 Q210.905 1456.14 212.902 1456.04 Q214.899 1455.91 216.715 1454.1 Q219.22 1451.59 218.958 1448.32 Q218.696 1445.01 215.75 1442.07 L215.079 1441.4 L212.083 1444.39 M216.846 1437.14 L227.306 1447.6 L224.294 1450.61 L221.511 1447.83 Q222.15 1450.53 221.413 1452.87 Q220.66 1455.19 218.434 1457.42 Q215.619 1460.23 212.378 1460.33 Q209.137 1460.4 206.485 1457.75 Q203.392 1454.65 203.883 1451.02 Q204.39 1447.37 208.499 1443.26 L212.722 1439.04 L212.427 1438.74 Q210.348 1436.66 207.844 1436.91 Q205.34 1437.12 202.868 1439.59 Q201.297 1441.17 200.184 1443.03 Q199.071 1444.9 198.449 1447.03 L195.666 1444.24 Q196.681 1441.95 197.925 1440.09 Q199.152 1438.2 200.626 1436.73 Q204.603 1432.75 208.63 1432.85 Q212.656 1432.95 216.846 1437.14 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M208.04 1415.93 L211.052 1412.91 L236.521 1438.38 L233.509 1441.4 L208.04 1415.93 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M224.621 1399.41 Q226.193 1405.37 228.812 1410.12 Q231.43 1414.86 235.211 1418.64 Q238.992 1422.42 243.772 1425.08 Q248.551 1427.7 254.477 1429.27 L251.858 1431.89 Q245.556 1430.49 240.613 1428.01 Q235.686 1425.5 232.02 1421.84 Q228.37 1418.19 225.882 1413.27 Q223.394 1408.36 222.002 1402.03 L224.621 1399.41 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M229.99 1394.04 L232.609 1391.42 Q238.927 1392.83 243.837 1395.32 Q248.764 1397.79 252.414 1401.44 Q256.081 1405.11 258.569 1410.05 Q261.073 1414.98 262.464 1421.28 L259.845 1423.9 Q258.274 1417.97 255.639 1413.21 Q253.004 1408.41 249.223 1404.63 Q245.442 1400.85 240.678 1398.25 Q235.932 1395.63 229.99 1394.04 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M1187.32 1611.28 L1182.58 1599.09 L1172.81 1616.92 L1165.9 1616.92 L1179.71 1591.71 L1173.92 1576.72 Q1172.36 1572.71 1167.46 1572.71 L1165.9 1572.71 L1165.9 1567.68 L1168.13 1567.74 Q1176.34 1567.96 1178.41 1573.28 L1183.12 1585.47 L1192.89 1567.64 L1199.8 1567.64 L1185.98 1592.85 L1191.78 1607.84 Q1193.34 1611.85 1198.24 1611.85 L1199.8 1611.85 L1199.8 1616.88 L1197.57 1616.82 Q1189.36 1616.6 1187.32 1611.28 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M1229.3 1573.72 L1270.11 1573.72 L1270.11 1579.07 L1229.3 1579.07 L1229.3 1573.72 M1229.3 1586.71 L1270.11 1586.71 L1270.11 1592.12 L1229.3 1592.12 L1229.3 1586.71 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M1304.77 1555.8 L1330.01 1555.8 L1330.01 1561.22 L1310.66 1561.22 L1310.66 1572.86 Q1312.06 1572.39 1313.46 1572.16 Q1314.86 1571.91 1316.26 1571.91 Q1324.22 1571.91 1328.86 1576.27 Q1333.51 1580.63 1333.51 1588.08 Q1333.51 1595.75 1328.74 1600.01 Q1323.96 1604.25 1315.27 1604.25 Q1312.28 1604.25 1309.16 1603.74 Q1306.07 1603.23 1302.76 1602.21 L1302.76 1595.75 Q1305.63 1597.31 1308.68 1598.07 Q1311.74 1598.84 1315.14 1598.84 Q1320.65 1598.84 1323.87 1595.94 Q1327.08 1593.04 1327.08 1588.08 Q1327.08 1583.11 1323.87 1580.22 Q1320.65 1577.32 1315.14 1577.32 Q1312.57 1577.32 1309.99 1577.89 Q1307.44 1578.47 1304.77 1579.68 L1304.77 1555.8 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M1359.93 1560.04 Q1354.96 1560.04 1352.45 1564.94 Q1349.97 1569.81 1349.97 1579.61 Q1349.97 1589.38 1352.45 1594.29 Q1354.96 1599.16 1359.93 1599.16 Q1364.92 1599.16 1367.41 1594.29 Q1369.92 1589.38 1369.92 1579.61 Q1369.92 1569.81 1367.41 1564.94 Q1364.92 1560.04 1359.93 1560.04 M1359.93 1554.95 Q1367.92 1554.95 1372.12 1561.28 Q1376.35 1567.58 1376.35 1579.61 Q1376.35 1591.61 1372.12 1597.95 Q1367.92 1604.25 1359.93 1604.25 Q1351.94 1604.25 1347.71 1597.95 Q1343.5 1591.61 1343.5 1579.61 Q1343.5 1567.58 1347.71 1561.28 Q1351.94 1554.95 1359.93 1554.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><polyline clip-path="url(#clip690)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="189.496,1352.62 189.496,123.472 "/>
<polyline clip-path="url(#clip690)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="189.496,1348.61 208.394,1348.61 "/>
<polyline clip-path="url(#clip690)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="189.496,1054.07 208.394,1054.07 "/>
<polyline clip-path="url(#clip690)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="189.496,759.533 208.394,759.533 "/>
<polyline clip-path="url(#clip690)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="189.496,464.992 208.394,464.992 "/>
<polyline clip-path="url(#clip690)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="189.496,170.451 208.394,170.451 "/>
<path clip-path="url(#clip690)" d="M51.6634 1368.41 L59.3023 1368.41 L59.3023 1342.04 L50.9921 1343.71 L50.9921 1339.45 L59.256 1337.78 L63.9319 1337.78 L63.9319 1368.41 L71.5707 1368.41 L71.5707 1372.34 L51.6634 1372.34 L51.6634 1368.41 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M91.0151 1340.86 Q87.404 1340.86 85.5753 1344.43 Q83.7697 1347.97 83.7697 1355.1 Q83.7697 1362.2 85.5753 1365.77 Q87.404 1369.31 91.0151 1369.31 Q94.6493 1369.31 96.4548 1365.77 Q98.2835 1362.2 98.2835 1355.1 Q98.2835 1347.97 96.4548 1344.43 Q94.6493 1340.86 91.0151 1340.86 M91.0151 1337.16 Q96.8252 1337.16 99.8808 1341.76 Q102.959 1346.35 102.959 1355.1 Q102.959 1363.82 99.8808 1368.43 Q96.8252 1373.01 91.0151 1373.01 Q85.2049 1373.01 82.1262 1368.43 Q79.0707 1363.82 79.0707 1355.1 Q79.0707 1346.35 82.1262 1341.76 Q85.2049 1337.16 91.0151 1337.16 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M102.959 1331.26 L127.071 1331.26 L127.071 1334.46 L102.959 1334.46 L102.959 1331.26 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M145.71 1320.16 L136.118 1335.15 L145.71 1335.15 L145.71 1320.16 M144.713 1316.85 L149.49 1316.85 L149.49 1335.15 L153.496 1335.15 L153.496 1338.31 L149.49 1338.31 L149.49 1344.93 L145.71 1344.93 L145.71 1338.31 L133.033 1338.31 L133.033 1334.64 L144.713 1316.85 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M52.585 1073.87 L60.2238 1073.87 L60.2238 1047.5 L51.9137 1049.17 L51.9137 1044.91 L60.1776 1043.24 L64.8535 1043.24 L64.8535 1073.87 L72.4923 1073.87 L72.4923 1077.8 L52.585 1077.8 L52.585 1073.87 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M91.9366 1046.32 Q88.3255 1046.32 86.4969 1049.88 Q84.6913 1053.43 84.6913 1060.56 Q84.6913 1067.66 86.4969 1071.23 Q88.3255 1074.77 91.9366 1074.77 Q95.5709 1074.77 97.3764 1071.23 Q99.2051 1067.66 99.2051 1060.56 Q99.2051 1053.43 97.3764 1049.88 Q95.5709 1046.32 91.9366 1046.32 M91.9366 1042.62 Q97.7468 1042.62 100.802 1047.22 Q103.881 1051.81 103.881 1060.56 Q103.881 1069.28 100.802 1073.89 Q97.7468 1078.47 91.9366 1078.47 Q86.1265 1078.47 83.0478 1073.89 Q79.9923 1069.28 79.9923 1060.56 Q79.9923 1051.81 83.0478 1047.22 Q86.1265 1042.62 91.9366 1042.62 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M103.881 1036.72 L127.993 1036.72 L127.993 1039.91 L103.881 1039.91 L103.881 1036.72 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M147.703 1035.25 Q150.43 1035.83 151.954 1037.68 Q153.496 1039.52 153.496 1042.23 Q153.496 1046.38 150.637 1048.66 Q147.778 1050.94 142.512 1050.94 Q140.744 1050.94 138.863 1050.58 Q137.002 1050.24 135.008 1049.54 L135.008 1045.88 Q136.588 1046.8 138.469 1047.27 Q140.349 1047.74 142.399 1047.74 Q145.973 1047.74 147.835 1046.33 Q149.716 1044.92 149.716 1042.23 Q149.716 1039.75 147.966 1038.35 Q146.236 1036.94 143.133 1036.94 L139.86 1036.94 L139.86 1033.82 L143.283 1033.82 Q146.086 1033.82 147.571 1032.71 Q149.057 1031.58 149.057 1029.48 Q149.057 1027.31 147.515 1026.17 Q145.992 1025 143.133 1025 Q141.572 1025 139.785 1025.34 Q137.998 1025.68 135.854 1026.39 L135.854 1023.01 Q138.017 1022.4 139.898 1022.1 Q141.797 1021.8 143.471 1021.8 Q147.797 1021.8 150.317 1023.78 Q152.838 1025.73 152.838 1029.08 Q152.838 1031.41 151.502 1033.03 Q150.167 1034.63 147.703 1035.25 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M53.3561 779.325 L60.995 779.325 L60.995 752.96 L52.6848 754.626 L52.6848 750.367 L60.9487 748.7 L65.6246 748.7 L65.6246 779.325 L73.2634 779.325 L73.2634 783.26 L53.3561 783.26 L53.3561 779.325 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M92.7078 751.779 Q89.0967 751.779 87.268 755.344 Q85.4624 758.886 85.4624 766.015 Q85.4624 773.122 87.268 776.686 Q89.0967 780.228 92.7078 780.228 Q96.342 780.228 98.1475 776.686 Q99.9762 773.122 99.9762 766.015 Q99.9762 758.886 98.1475 755.344 Q96.342 751.779 92.7078 751.779 M92.7078 748.075 Q98.5179 748.075 101.573 752.682 Q104.652 757.265 104.652 766.015 Q104.652 774.742 101.573 779.348 Q98.5179 783.932 92.7078 783.932 Q86.8976 783.932 83.8189 779.348 Q80.7634 774.742 80.7634 766.015 Q80.7634 757.265 83.8189 752.682 Q86.8976 748.075 92.7078 748.075 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M104.652 742.177 L128.764 742.177 L128.764 745.374 L104.652 745.374 L104.652 742.177 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M140.236 752.653 L153.496 752.653 L153.496 755.85 L135.666 755.85 L135.666 752.653 Q137.829 750.415 141.553 746.653 Q145.296 742.873 146.255 741.782 Q148.079 739.732 148.794 738.321 Q149.527 736.892 149.527 735.519 Q149.527 733.281 147.948 731.87 Q146.387 730.46 143.866 730.46 Q142.08 730.46 140.086 731.08 Q138.111 731.701 135.854 732.961 L135.854 729.124 Q138.149 728.203 140.142 727.732 Q142.136 727.262 143.791 727.262 Q148.155 727.262 150.75 729.444 Q153.345 731.626 153.345 735.274 Q153.345 737.005 152.687 738.566 Q152.048 740.108 150.336 742.214 Q149.866 742.76 147.346 745.374 Q144.826 747.97 140.236 752.653 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M53.0552 484.784 L60.694 484.784 L60.694 458.419 L52.3839 460.085 L52.3839 455.826 L60.6477 454.16 L65.3236 454.16 L65.3236 484.784 L72.9625 484.784 L72.9625 488.72 L53.0552 488.72 L53.0552 484.784 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M92.4068 457.238 Q88.7957 457.238 86.967 460.803 Q85.1615 464.345 85.1615 471.474 Q85.1615 478.581 86.967 482.146 Q88.7957 485.687 92.4068 485.687 Q96.0411 485.687 97.8466 482.146 Q99.6753 478.581 99.6753 471.474 Q99.6753 464.345 97.8466 460.803 Q96.0411 457.238 92.4068 457.238 M92.4068 453.535 Q98.217 453.535 101.273 458.141 Q104.351 462.724 104.351 471.474 Q104.351 480.201 101.273 484.808 Q98.217 489.391 92.4068 489.391 Q86.5967 489.391 83.518 484.808 Q80.4625 480.201 80.4625 471.474 Q80.4625 462.724 83.518 458.141 Q86.5967 453.535 92.4068 453.535 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M104.351 447.636 L128.463 447.636 L128.463 450.833 L104.351 450.833 L104.351 447.636 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M137.321 458.112 L143.528 458.112 L143.528 436.69 L136.776 438.044 L136.776 434.583 L143.49 433.229 L147.289 433.229 L147.289 458.112 L153.496 458.112 L153.496 461.309 L137.321 461.309 L137.321 458.112 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M82.7903 190.244 L90.4291 190.244 L90.4291 163.878 L82.119 165.545 L82.119 161.285 L90.3828 159.619 L95.0587 159.619 L95.0587 190.244 L102.698 190.244 L102.698 194.179 L82.7903 194.179 L82.7903 190.244 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M122.142 162.697 Q118.531 162.697 116.702 166.262 Q114.897 169.804 114.897 176.933 Q114.897 184.04 116.702 187.605 Q118.531 191.146 122.142 191.146 Q125.776 191.146 127.582 187.605 Q129.41 184.04 129.41 176.933 Q129.41 169.804 127.582 166.262 Q125.776 162.697 122.142 162.697 M122.142 158.994 Q127.952 158.994 131.008 163.6 Q134.086 168.183 134.086 176.933 Q134.086 185.66 131.008 190.267 Q127.952 194.85 122.142 194.85 Q116.332 194.85 113.253 190.267 Q110.198 185.66 110.198 176.933 Q110.198 168.183 113.253 163.6 Q116.332 158.994 122.142 158.994 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M143.791 141.19 Q140.857 141.19 139.371 144.086 Q137.904 146.964 137.904 152.757 Q137.904 158.531 139.371 161.427 Q140.857 164.305 143.791 164.305 Q146.744 164.305 148.211 161.427 Q149.697 158.531 149.697 152.757 Q149.697 146.964 148.211 144.086 Q146.744 141.19 143.791 141.19 M143.791 138.181 Q148.512 138.181 150.994 141.923 Q153.496 145.647 153.496 152.757 Q153.496 159.847 150.994 163.59 Q148.512 167.314 143.791 167.314 Q139.07 167.314 136.569 163.59 Q134.086 159.847 134.086 152.757 Q134.086 145.647 136.569 141.923 Q139.07 138.181 143.791 138.181 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M772.196 12.096 L810.437 12.096 L810.437 18.9825 L780.379 18.9825 L780.379 36.8875 L809.181 36.8875 L809.181 43.7741 L780.379 43.7741 L780.379 65.6895 L811.166 65.6895 L811.166 72.576 L772.196 72.576 L772.196 12.096 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M862.005 45.1919 L862.005 72.576 L854.551 72.576 L854.551 45.4349 Q854.551 38.994 852.04 35.7938 Q849.528 32.5936 844.505 32.5936 Q838.469 32.5936 834.985 36.4419 Q831.502 40.2903 831.502 46.9338 L831.502 72.576 L824.007 72.576 L824.007 27.2059 L831.502 27.2059 L831.502 34.2544 Q834.175 30.163 837.78 28.1376 Q841.426 26.1121 846.166 26.1121 Q853.984 26.1121 857.994 30.9732 Q862.005 35.7938 862.005 45.1919 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M884.244 14.324 L884.244 27.2059 L899.597 27.2059 L899.597 32.9987 L884.244 32.9987 L884.244 57.6282 Q884.244 63.1779 885.743 64.7578 Q887.282 66.3376 891.941 66.3376 L899.597 66.3376 L899.597 72.576 L891.941 72.576 Q883.313 72.576 880.031 69.3758 Q876.75 66.1351 876.75 57.6282 L876.75 32.9987 L871.281 32.9987 L871.281 27.2059 L876.75 27.2059 L876.75 14.324 L884.244 14.324 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M930.02 49.7694 Q920.986 49.7694 917.502 51.8354 Q914.018 53.9013 914.018 58.8839 Q914.018 62.8538 916.611 65.2034 Q919.244 67.5124 923.741 67.5124 Q929.939 67.5124 933.665 63.1374 Q937.433 58.7219 937.433 51.4303 L937.433 49.7694 L930.02 49.7694 M944.886 46.6907 L944.886 72.576 L937.433 72.576 L937.433 65.6895 Q934.881 69.8214 931.073 71.8063 Q927.265 73.7508 921.756 73.7508 Q914.788 73.7508 910.656 69.8619 Q906.565 65.9325 906.565 59.3701 Q906.565 51.7138 911.669 47.825 Q916.814 43.9361 926.981 43.9361 L937.433 43.9361 L937.433 43.2069 Q937.433 38.0623 934.03 35.2672 Q930.668 32.4315 924.551 32.4315 Q920.662 32.4315 916.976 33.3632 Q913.289 34.295 909.887 36.1584 L909.887 29.2718 Q913.978 27.692 917.826 26.9223 Q921.675 26.1121 925.32 26.1121 Q935.164 26.1121 940.025 31.2163 Q944.886 36.3204 944.886 46.6907 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M997.953 45.1919 L997.953 72.576 L990.5 72.576 L990.5 45.4349 Q990.5 38.994 987.988 35.7938 Q985.476 32.5936 980.453 32.5936 Q974.417 32.5936 970.934 36.4419 Q967.45 40.2903 967.45 46.9338 L967.45 72.576 L959.956 72.576 L959.956 27.2059 L967.45 27.2059 L967.45 34.2544 Q970.123 30.163 973.729 28.1376 Q977.375 26.1121 982.114 26.1121 Q989.932 26.1121 993.943 30.9732 Q997.953 35.7938 997.953 45.1919 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M1042.68 49.3643 Q1042.68 41.2625 1039.31 36.8065 Q1035.99 32.3505 1029.96 32.3505 Q1023.96 32.3505 1020.6 36.8065 Q1017.28 41.2625 1017.28 49.3643 Q1017.28 57.4256 1020.6 61.8816 Q1023.96 66.3376 1029.96 66.3376 Q1035.99 66.3376 1039.31 61.8816 Q1042.68 57.4256 1042.68 49.3643 M1050.13 66.9452 Q1050.13 78.5308 1044.98 84.1616 Q1039.84 89.8329 1029.23 89.8329 Q1025.3 89.8329 1021.81 89.2252 Q1018.33 88.6581 1015.05 87.4428 L1015.05 80.1917 Q1018.33 81.9741 1021.53 82.8248 Q1024.73 83.6755 1028.05 83.6755 Q1035.38 83.6755 1039.03 79.8271 Q1042.68 76.0193 1042.68 68.282 L1042.68 64.5957 Q1040.37 68.6061 1036.76 70.5911 Q1033.16 72.576 1028.13 72.576 Q1019.79 72.576 1014.68 66.2161 Q1009.58 59.8562 1009.58 49.3643 Q1009.58 38.832 1014.68 32.472 Q1019.79 26.1121 1028.13 26.1121 Q1033.16 26.1121 1036.76 28.0971 Q1040.37 30.082 1042.68 34.0924 L1042.68 27.2059 L1050.13 27.2059 L1050.13 66.9452 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M1065.48 9.54393 L1072.94 9.54393 L1072.94 72.576 L1065.48 72.576 L1065.48 9.54393 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M1127.34 48.0275 L1127.34 51.6733 L1093.07 51.6733 Q1093.55 59.3701 1097.69 63.421 Q1101.86 67.4314 1109.27 67.4314 Q1113.57 67.4314 1117.58 66.3781 Q1121.63 65.3249 1125.6 63.2184 L1125.6 70.267 Q1121.59 71.9684 1117.37 72.8596 Q1113.16 73.7508 1108.83 73.7508 Q1097.97 73.7508 1091.61 67.4314 Q1085.29 61.1119 1085.29 50.3365 Q1085.29 39.1965 1091.29 32.6746 Q1097.32 26.1121 1107.53 26.1121 Q1116.69 26.1121 1121.99 32.0264 Q1127.34 37.9003 1127.34 48.0275 M1119.89 45.84 Q1119.8 39.7232 1116.44 36.0774 Q1113.12 32.4315 1107.61 32.4315 Q1101.37 32.4315 1097.61 35.9558 Q1093.88 39.4801 1093.31 45.8805 L1119.89 45.84 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M1174.9 35.9153 Q1177.69 30.8922 1181.58 28.5022 Q1185.47 26.1121 1190.74 26.1121 Q1197.82 26.1121 1201.67 31.0947 Q1205.52 36.0368 1205.52 45.1919 L1205.52 72.576 L1198.03 72.576 L1198.03 45.4349 Q1198.03 38.913 1195.72 35.7533 Q1193.41 32.5936 1188.67 32.5936 Q1182.88 32.5936 1179.51 36.4419 Q1176.15 40.2903 1176.15 46.9338 L1176.15 72.576 L1168.66 72.576 L1168.66 45.4349 Q1168.66 38.8725 1166.35 35.7533 Q1164.04 32.5936 1159.22 32.5936 Q1153.51 32.5936 1150.15 36.4824 Q1146.78 40.3308 1146.78 46.9338 L1146.78 72.576 L1139.29 72.576 L1139.29 27.2059 L1146.78 27.2059 L1146.78 34.2544 Q1149.34 30.082 1152.9 28.0971 Q1156.47 26.1121 1161.37 26.1121 Q1166.31 26.1121 1169.75 28.6237 Q1173.24 31.1352 1174.9 35.9153 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M1259.2 48.0275 L1259.2 51.6733 L1224.93 51.6733 Q1225.41 59.3701 1229.54 63.421 Q1233.72 67.4314 1241.13 67.4314 Q1245.42 67.4314 1249.43 66.3781 Q1253.48 65.3249 1257.45 63.2184 L1257.45 70.267 Q1253.44 71.9684 1249.23 72.8596 Q1245.02 73.7508 1240.68 73.7508 Q1229.83 73.7508 1223.47 67.4314 Q1217.15 61.1119 1217.15 50.3365 Q1217.15 39.1965 1223.14 32.6746 Q1229.18 26.1121 1239.39 26.1121 Q1248.54 26.1121 1253.85 32.0264 Q1259.2 37.9003 1259.2 48.0275 M1251.74 45.84 Q1251.66 39.7232 1248.3 36.0774 Q1244.98 32.4315 1239.47 32.4315 Q1233.23 32.4315 1229.46 35.9558 Q1225.74 39.4801 1225.17 45.8805 L1251.74 45.84 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M1309.14 45.1919 L1309.14 72.576 L1301.69 72.576 L1301.69 45.4349 Q1301.69 38.994 1299.18 35.7938 Q1296.67 32.5936 1291.64 32.5936 Q1285.61 32.5936 1282.12 36.4419 Q1278.64 40.2903 1278.64 46.9338 L1278.64 72.576 L1271.15 72.576 L1271.15 27.2059 L1278.64 27.2059 L1278.64 34.2544 Q1281.31 30.163 1284.92 28.1376 Q1288.57 26.1121 1293.3 26.1121 Q1301.12 26.1121 1305.13 30.9732 Q1309.14 35.7938 1309.14 45.1919 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M1331.38 14.324 L1331.38 27.2059 L1346.74 27.2059 L1346.74 32.9987 L1331.38 32.9987 L1331.38 57.6282 Q1331.38 63.1779 1332.88 64.7578 Q1334.42 66.3376 1339.08 66.3376 L1346.74 66.3376 L1346.74 72.576 L1339.08 72.576 Q1330.45 72.576 1327.17 69.3758 Q1323.89 66.1351 1323.89 57.6282 L1323.89 32.9987 L1318.42 32.9987 L1318.42 27.2059 L1323.89 27.2059 L1323.89 14.324 L1331.38 14.324 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M1419.49 14.0809 L1419.49 22.0612 Q1414.83 19.8332 1410.7 18.7395 Q1406.57 17.6457 1402.72 17.6457 Q1396.04 17.6457 1392.39 20.2383 Q1388.78 22.8309 1388.78 27.611 Q1388.78 31.6214 1391.17 33.6873 Q1393.61 35.7128 1400.33 36.9686 L1405.27 37.9813 Q1414.43 39.7232 1418.76 44.1387 Q1423.14 48.5136 1423.14 55.8863 Q1423.14 64.6767 1417.22 69.2137 Q1411.35 73.7508 1399.96 73.7508 Q1395.67 73.7508 1390.81 72.7785 Q1385.99 71.8063 1380.8 69.9024 L1380.8 61.4765 Q1385.79 64.2716 1390.57 65.6895 Q1395.35 67.1073 1399.96 67.1073 Q1406.97 67.1073 1410.78 64.3527 Q1414.59 61.598 1414.59 56.4939 Q1414.59 52.0379 1411.83 49.5264 Q1409.12 47.0148 1402.88 45.759 L1397.9 44.7868 Q1388.74 42.9639 1384.65 39.075 Q1380.56 35.1862 1380.56 28.2591 Q1380.56 20.2383 1386.19 15.6203 Q1391.86 11.0023 1401.79 11.0023 Q1406.04 11.0023 1410.46 11.7719 Q1414.87 12.5416 1419.49 14.0809 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M1442.78 65.7705 L1442.78 89.8329 L1435.29 89.8329 L1435.29 27.2059 L1442.78 27.2059 L1442.78 34.0924 Q1445.13 30.0415 1448.7 28.0971 Q1452.3 26.1121 1457.29 26.1121 Q1465.55 26.1121 1470.69 32.6746 Q1475.88 39.2371 1475.88 49.9314 Q1475.88 60.6258 1470.69 67.1883 Q1465.55 73.7508 1457.29 73.7508 Q1452.3 73.7508 1448.7 71.8063 Q1445.13 69.8214 1442.78 65.7705 M1468.14 49.9314 Q1468.14 41.7081 1464.74 37.0496 Q1461.38 32.3505 1455.46 32.3505 Q1449.55 32.3505 1446.15 37.0496 Q1442.78 41.7081 1442.78 49.9314 Q1442.78 58.1548 1446.15 62.8538 Q1449.55 67.5124 1455.46 67.5124 Q1461.38 67.5124 1464.74 62.8538 Q1468.14 58.1548 1468.14 49.9314 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M1527.04 48.0275 L1527.04 51.6733 L1492.77 51.6733 Q1493.26 59.3701 1497.39 63.421 Q1501.56 67.4314 1508.97 67.4314 Q1513.27 67.4314 1517.28 66.3781 Q1521.33 65.3249 1525.3 63.2184 L1525.3 70.267 Q1521.29 71.9684 1517.08 72.8596 Q1512.86 73.7508 1508.53 73.7508 Q1497.67 73.7508 1491.31 67.4314 Q1484.99 61.1119 1484.99 50.3365 Q1484.99 39.1965 1490.99 32.6746 Q1497.02 26.1121 1507.23 26.1121 Q1516.39 26.1121 1521.69 32.0264 Q1527.04 37.9003 1527.04 48.0275 M1519.59 45.84 Q1519.51 39.7232 1516.14 36.0774 Q1512.82 32.4315 1507.31 32.4315 Q1501.08 32.4315 1497.31 35.9558 Q1493.58 39.4801 1493.01 45.8805 L1519.59 45.84 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M1571.93 28.9478 L1571.93 35.9153 Q1568.77 34.1734 1565.57 33.3227 Q1562.41 32.4315 1559.17 32.4315 Q1551.91 32.4315 1547.9 37.0496 Q1543.89 41.6271 1543.89 49.9314 Q1543.89 58.2358 1547.9 62.8538 Q1551.91 67.4314 1559.17 67.4314 Q1562.41 67.4314 1565.57 66.5807 Q1568.77 65.6895 1571.93 63.9476 L1571.93 70.8341 Q1568.81 72.2924 1565.44 73.0216 Q1562.12 73.7508 1558.36 73.7508 Q1548.11 73.7508 1542.07 67.3098 Q1536.03 60.8689 1536.03 49.9314 Q1536.03 38.832 1542.11 32.472 Q1548.23 26.1121 1558.84 26.1121 Q1562.28 26.1121 1565.57 26.8413 Q1568.85 27.5299 1571.93 28.9478 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M1592.26 14.324 L1592.26 27.2059 L1607.61 27.2059 L1607.61 32.9987 L1592.26 32.9987 L1592.26 57.6282 Q1592.26 63.1779 1593.76 64.7578 Q1595.3 66.3376 1599.96 66.3376 L1607.61 66.3376 L1607.61 72.576 L1599.96 72.576 Q1591.33 72.576 1588.05 69.3758 Q1584.77 66.1351 1584.77 57.6282 L1584.77 32.9987 L1579.3 32.9987 L1579.3 27.2059 L1584.77 27.2059 L1584.77 14.324 L1592.26 14.324 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M1643.71 34.1734 Q1642.45 33.4443 1640.95 33.1202 Q1639.49 32.7556 1637.71 32.7556 Q1631.39 32.7556 1627.99 36.8875 Q1624.63 40.9789 1624.63 48.6757 L1624.63 72.576 L1617.13 72.576 L1617.13 27.2059 L1624.63 27.2059 L1624.63 34.2544 Q1626.98 30.1225 1630.74 28.1376 Q1634.51 26.1121 1639.9 26.1121 Q1640.67 26.1121 1641.6 26.2337 Q1642.53 26.3147 1643.67 26.5172 L1643.71 34.1734 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M1650.76 54.671 L1650.76 27.2059 L1658.21 27.2059 L1658.21 54.3874 Q1658.21 60.8284 1660.72 64.0691 Q1663.23 67.2693 1668.26 67.2693 Q1674.29 67.2693 1677.78 63.421 Q1681.3 59.5726 1681.3 52.9291 L1681.3 27.2059 L1688.75 27.2059 L1688.75 72.576 L1681.3 72.576 L1681.3 65.6084 Q1678.59 69.7404 1674.98 71.7658 Q1671.42 73.7508 1666.68 73.7508 Q1658.86 73.7508 1654.81 68.8897 Q1650.76 64.0286 1650.76 54.671 M1669.51 26.1121 L1669.51 26.1121 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip690)" d="M1739.43 35.9153 Q1742.23 30.8922 1746.11 28.5022 Q1750 26.1121 1755.27 26.1121 Q1762.36 26.1121 1766.21 31.0947 Q1770.06 36.0368 1770.06 45.1919 L1770.06 72.576 L1762.56 72.576 L1762.56 45.4349 Q1762.56 38.913 1760.25 35.7533 Q1757.94 32.5936 1753.2 32.5936 Q1747.41 32.5936 1744.05 36.4419 Q1740.69 40.2903 1740.69 46.9338 L1740.69 72.576 L1733.19 72.576 L1733.19 45.4349 Q1733.19 38.8725 1730.88 35.7533 Q1728.57 32.5936 1723.75 32.5936 Q1718.04 32.5936 1714.68 36.4824 Q1711.32 40.3308 1711.32 46.9338 L1711.32 72.576 L1703.82 72.576 L1703.82 27.2059 L1711.32 27.2059 L1711.32 34.2544 Q1713.87 30.082 1717.43 28.0971 Q1721 26.1121 1725.9 26.1121 Q1730.84 26.1121 1734.29 28.6237 Q1737.77 31.1352 1739.43 35.9153 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><circle clip-path="url(#clip692)" cx="454.801" cy="194.56" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="488.121" cy="297.794" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="521.44" cy="323.578" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="554.759" cy="344.755" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="588.079" cy="485.802" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="621.398" cy="513.898" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="654.718" cy="534.136" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="688.037" cy="538.189" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="721.356" cy="644.414" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="754.676" cy="672.681" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="787.995" cy="695.352" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="821.314" cy="717.383" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="854.634" cy="718.446" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="887.953" cy="751.575" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="921.273" cy="751.706" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="954.592" cy="772.678" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="987.911" cy="773.255" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="1021.23" cy="777.05" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="1054.55" cy="796.962" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="1087.87" cy="813.961" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="1121.19" cy="885.452" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="1154.51" cy="918.748" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="1187.83" cy="942.657" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="1221.15" cy="960.123" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="1254.47" cy="963.391" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="1287.79" cy="1005.07" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="1321.1" cy="1018.93" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="1354.42" cy="1031.69" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="1387.74" cy="1050.75" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="1421.06" cy="1063.66" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="1454.38" cy="1064.08" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="1487.7" cy="1069.65" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="1521.02" cy="1086.99" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="1554.34" cy="1088.01" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="1587.66" cy="1093.1" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="1620.98" cy="1096.26" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="1654.3" cy="1111.63" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="1687.62" cy="1127.08" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="1720.94" cy="1153.21" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="1754.26" cy="1170.22" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="1787.58" cy="1210.14" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="1820.9" cy="1212.99" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="1854.21" cy="1238.08" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="1887.53" cy="1259.73" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="1920.85" cy="1262.12" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="1954.17" cy="1263.56" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="1987.49" cy="1277.72" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="2020.81" cy="1291.8" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="2054.13" cy="1299.65" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip692)" cx="2087.45" cy="1317.83" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
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
[ Info: VUMPS init:	obj = +4.207308597914e-02	err = 3.8417e-01
[ Info: VUMPS   1:	obj = -8.770246689397e-01	err = 8.9387480236e-02	time = 0.09 sec
[ Info: VUMPS   2:	obj = -8.854224662459e-01	err = 1.2035494481e-02	time = 0.08 sec
[ Info: VUMPS   3:	obj = -8.860742922774e-01	err = 4.4294789545e-03	time = 0.08 sec
[ Info: VUMPS   4:	obj = -8.862108801823e-01	err = 1.9061849051e-03	time = 0.09 sec
[ Info: VUMPS   5:	obj = -8.862567332668e-01	err = 1.1244944387e-03	time = 0.11 sec
[ Info: VUMPS   6:	obj = -8.862739286226e-01	err = 8.4129140580e-04	time = 0.12 sec
[ Info: VUMPS   7:	obj = -8.862812942419e-01	err = 6.3937860306e-04	time = 0.23 sec
[ Info: VUMPS   8:	obj = -8.862846581258e-01	err = 5.2763554024e-04	time = 0.12 sec
[ Info: VUMPS   9:	obj = -8.862862830189e-01	err = 4.2102561366e-04	time = 0.12 sec
[ Info: VUMPS  10:	obj = -8.862870857559e-01	err = 3.3035934875e-04	time = 0.13 sec
[ Info: VUMPS  11:	obj = -8.862874886043e-01	err = 2.5432646892e-04	time = 0.13 sec
[ Info: VUMPS  12:	obj = -8.862876919660e-01	err = 1.9202079806e-04	time = 0.21 sec
[ Info: VUMPS  13:	obj = -8.862877946054e-01	err = 1.4279990039e-04	time = 0.19 sec
[ Info: VUMPS  14:	obj = -8.862878462682e-01	err = 1.0491846055e-04	time = 0.14 sec
[ Info: VUMPS  15:	obj = -8.862878721892e-01	err = 7.6390575113e-05	time = 0.20 sec
[ Info: VUMPS  16:	obj = -8.862878851596e-01	err = 5.5250778998e-05	time = 0.14 sec
[ Info: VUMPS  17:	obj = -8.862878916381e-01	err = 3.9767417638e-05	time = 0.15 sec
[ Info: VUMPS  18:	obj = -8.862878948705e-01	err = 2.8521655196e-05	time = 0.20 sec
[ Info: VUMPS  19:	obj = -8.862878964826e-01	err = 2.0403457752e-05	time = 0.15 sec
[ Info: VUMPS  20:	obj = -8.862878972865e-01	err = 1.4568719674e-05	time = 0.15 sec
[ Info: VUMPS  21:	obj = -8.862878976876e-01	err = 1.0388173825e-05	time = 0.19 sec
[ Info: VUMPS  22:	obj = -8.862878978877e-01	err = 7.3993990730e-06	time = 0.15 sec
[ Info: VUMPS  23:	obj = -8.862878979876e-01	err = 5.2663271673e-06	time = 0.20 sec
[ Info: VUMPS  24:	obj = -8.862878980375e-01	err = 3.7457858907e-06	time = 0.15 sec
[ Info: VUMPS  25:	obj = -8.862878980625e-01	err = 2.6630244108e-06	time = 0.19 sec
[ Info: VUMPS  26:	obj = -8.862878980749e-01	err = 1.8924316652e-06	time = 0.15 sec
[ Info: VUMPS  27:	obj = -8.862878980812e-01	err = 1.3443459346e-06	time = 0.19 sec
[ Info: VUMPS  28:	obj = -8.862878980843e-01	err = 9.5471634701e-07	time = 0.15 sec
[ Info: VUMPS  29:	obj = -8.862878980859e-01	err = 6.7781844497e-07	time = 0.19 sec
[ Info: VUMPS  30:	obj = -8.862878980866e-01	err = 4.8112158398e-07	time = 0.15 sec
[ Info: VUMPS  31:	obj = -8.862878980870e-01	err = 3.4143706519e-07	time = 0.19 sec
[ Info: VUMPS  32:	obj = -8.862878980872e-01	err = 2.4224840418e-07	time = 0.15 sec
[ Info: VUMPS  33:	obj = -8.862878980873e-01	err = 1.7186139672e-07	time = 0.18 sec
[ Info: VUMPS  34:	obj = -8.862878980874e-01	err = 1.2190374782e-07	time = 0.14 sec
[ Info: VUMPS  35:	obj = -8.862878980874e-01	err = 8.6464933933e-08	time = 0.17 sec
[ Info: VUMPS  36:	obj = -8.862878980874e-01	err = 6.1286598356e-08	time = 0.12 sec
[ Info: VUMPS  37:	obj = -8.862878980874e-01	err = 4.3458741190e-08	time = 0.17 sec
[ Info: VUMPS  38:	obj = -8.862878980874e-01	err = 3.0804670453e-08	time = 0.12 sec
[ Info: VUMPS  39:	obj = -8.862878980874e-01	err = 2.1850363058e-08	time = 0.16 sec
[ Info: VUMPS  40:	obj = -8.862878980874e-01	err = 1.5482006174e-08	time = 0.11 sec
[ Info: VUMPS  41:	obj = -8.862878980874e-01	err = 1.0955627446e-08	time = 0.11 sec
[ Info: VUMPS  42:	obj = -8.862878980874e-01	err = 7.7609982125e-09	time = 0.16 sec
[ Info: VUMPS  43:	obj = -8.862878980874e-01	err = 5.4975161355e-09	time = 0.10 sec
[ Info: VUMPS  44:	obj = -8.862878980874e-01	err = 3.8884417026e-09	time = 0.14 sec
[ Info: VUMPS  45:	obj = -8.862878980874e-01	err = 2.7526126601e-09	time = 0.10 sec
[ Info: VUMPS  46:	obj = -8.862878980874e-01	err = 1.9592691434e-09	time = 0.10 sec
[ Info: VUMPS  47:	obj = -8.862878980874e-01	err = 1.3970406578e-09	time = 0.14 sec
[ Info: VUMPS  48:	obj = -8.862878980874e-01	err = 9.8889708684e-10	time = 0.09 sec
[ Info: VUMPS  49:	obj = -8.862878980874e-01	err = 6.9459808321e-10	time = 0.09 sec
[ Info: VUMPS  50:	obj = -8.862878980874e-01	err = 5.0367864464e-10	time = 0.14 sec
[ Info: VUMPS  51:	obj = -8.862878980874e-01	err = 3.6463958695e-10	time = 0.09 sec
[ Info: VUMPS  52:	obj = -8.862878980874e-01	err = 2.6828514578e-10	time = 0.08 sec
[ Info: VUMPS  53:	obj = -8.862878980874e-01	err = 1.9666569024e-10	time = 0.12 sec
[ Info: VUMPS  54:	obj = -8.862878980874e-01	err = 1.7318181994e-10	time = 0.07 sec
[ Info: VUMPS  55:	obj = -8.862878980874e-01	err = 1.1881164874e-10	time = 0.07 sec
[ Info: VUMPS  56:	obj = -8.862878980874e-01	err = 1.5646264432e-10	time = 0.07 sec
[ Info: VUMPS  57:	obj = -8.862878980874e-01	err = 9.6907766503e-11	time = 0.10 sec
[ Info: VUMPS  58:	obj = -8.862878980874e-01	err = 7.7365993612e-11	time = 0.06 sec
[ Info: VUMPS  59:	obj = -8.862878980874e-01	err = 1.2792482228e-10	time = 0.05 sec
[ Info: VUMPS  60:	obj = -8.862878980874e-01	err = 4.5325752050e-11	time = 0.05 sec
[ Info: VUMPS  61:	obj = -8.862878980874e-01	err = 4.5779636260e-11	time = 0.09 sec
[ Info: VUMPS  62:	obj = -8.862878980874e-01	err = 4.5269457493e-11	time = 0.04 sec
[ Info: VUMPS  63:	obj = -8.862878980874e-01	err = 7.0408420460e-11	time = 0.06 sec
[ Info: VUMPS  64:	obj = -8.862878980874e-01	err = 1.0027819938e-10	time = 0.07 sec
[ Info: VUMPS  65:	obj = -8.862878980874e-01	err = 1.8123837296e-10	time = 0.05 sec
[ Info: VUMPS  66:	obj = -8.862878980874e-01	err = 1.1028483287e-10	time = 0.11 sec
[ Info: VUMPS  67:	obj = -8.862878980874e-01	err = 1.9542191849e-10	time = 0.07 sec
[ Info: VUMPS  68:	obj = -8.862878980874e-01	err = 1.3266968631e-10	time = 0.09 sec
[ Info: VUMPS  69:	obj = -8.862878980874e-01	err = 1.5648232544e-10	time = 0.10 sec
[ Info: VUMPS  70:	obj = -8.862878980874e-01	err = 1.6333828323e-10	time = 0.07 sec
[ Info: VUMPS  71:	obj = -8.862878980874e-01	err = 1.3636380966e-10	time = 0.07 sec
[ Info: VUMPS  72:	obj = -8.862878980874e-01	err = 1.8799364680e-10	time = 0.07 sec
[ Info: VUMPS  73:	obj = -8.862878980874e-01	err = 1.4472792724e-10	time = 0.11 sec
[ Info: VUMPS  74:	obj = -8.862878980874e-01	err = 1.1889880597e-10	time = 0.08 sec
[ Info: VUMPS  75:	obj = -8.862878980874e-01	err = 7.8177559574e-11	time = 0.07 sec
[ Info: VUMPS  76:	obj = -8.862878980874e-01	err = 1.2096744970e-10	time = 0.09 sec
[ Info: VUMPS  77:	obj = -8.862878980874e-01	err = 4.4508066303e-11	time = 0.05 sec
[ Info: VUMPS  78:	obj = -8.862878980874e-01	err = 4.5031266470e-11	time = 0.04 sec
[ Info: VUMPS  79:	obj = -8.862878980874e-01	err = 4.5991825775e-11	time = 0.04 sec
[ Info: VUMPS  80:	obj = -8.862878980874e-01	err = 4.4021673623e-11	time = 0.04 sec
[ Info: VUMPS  81:	obj = -8.862878980874e-01	err = 4.7747796467e-11	time = 0.09 sec
[ Info: VUMPS  82:	obj = -8.862878980874e-01	err = 1.1679442420e-10	time = 0.06 sec
[ Info: VUMPS  83:	obj = -8.862878980874e-01	err = 7.6452116242e-11	time = 0.05 sec
[ Info: VUMPS  84:	obj = -8.862878980874e-01	err = 7.6436900611e-11	time = 0.04 sec
[ Info: VUMPS  85:	obj = -8.862878980874e-01	err = 7.7080848736e-11	time = 0.04 sec
[ Info: VUMPS  86:	obj = -8.862878980874e-01	err = 1.3910926209e-10	time = 0.09 sec
[ Info: VUMPS  87:	obj = -8.862878980874e-01	err = 8.5011084551e-11	time = 0.05 sec
[ Info: VUMPS  88:	obj = -8.862878980874e-01	err = 1.2377629438e-10	time = 0.06 sec
[ Info: VUMPS  89:	obj = -8.862878980874e-01	err = 1.1509132562e-10	time = 0.05 sec
[ Info: VUMPS  90:	obj = -8.862878980874e-01	err = 3.6280763476e-11	time = 0.10 sec
[ Info: VUMPS  91:	obj = -8.862878980874e-01	err = 3.6759581497e-11	time = 0.04 sec
[ Info: VUMPS  92:	obj = -8.862878980874e-01	err = 4.1936246167e-11	time = 0.04 sec
[ Info: VUMPS  93:	obj = -8.862878980874e-01	err = 3.6586232570e-11	time = 0.04 sec
[ Info: VUMPS  94:	obj = -8.862878980874e-01	err = 3.6323263078e-11	time = 0.04 sec
[ Info: VUMPS  95:	obj = -8.862878980874e-01	err = 3.9734901103e-11	time = 0.04 sec
[ Info: VUMPS  96:	obj = -8.862878980874e-01	err = 7.4695673156e-11	time = 0.10 sec
[ Info: VUMPS  97:	obj = -8.862878980874e-01	err = 7.4601743849e-11	time = 0.04 sec
[ Info: VUMPS  98:	obj = -8.862878980874e-01	err = 7.3737980701e-11	time = 0.04 sec
[ Info: VUMPS  99:	obj = -8.862878980874e-01	err = 7.3617435190e-11	time = 0.04 sec
[ Info: VUMPS 100:	obj = -8.862878980874e-01	err = 1.4117716258e-10	time = 0.09 sec
[ Info: VUMPS 101:	obj = -8.862878980874e-01	err = 1.0140172092e-10	time = 0.07 sec
[ Info: VUMPS 102:	obj = -8.862878980874e-01	err = 9.0099910339e-11	time = 0.05 sec
[ Info: VUMPS 103:	obj = -8.862878980874e-01	err = 9.5223763299e-11	time = 0.05 sec
[ Info: VUMPS 104:	obj = -8.862878980874e-01	err = 9.8372998640e-11	time = 0.05 sec
[ Info: VUMPS 105:	obj = -8.862878980874e-01	err = 7.9944348222e-11	time = 0.09 sec
[ Info: VUMPS 106:	obj = -8.862878980874e-01	err = 1.1796835106e-10	time = 0.05 sec
[ Info: VUMPS 107:	obj = -8.862878980874e-01	err = 2.5584999237e-11	time = 0.05 sec
[ Info: VUMPS 108:	obj = -8.862878980874e-01	err = 2.5755393013e-11	time = 0.04 sec
[ Info: VUMPS 109:	obj = -8.862878980874e-01	err = 2.5654408608e-11	time = 0.09 sec
[ Info: VUMPS 110:	obj = -8.862878980874e-01	err = 2.8969243219e-11	time = 0.04 sec
[ Info: VUMPS 111:	obj = -8.862878980874e-01	err = 2.5648857275e-11	time = 0.04 sec
[ Info: VUMPS 112:	obj = -8.862878980874e-01	err = 2.6085432445e-11	time = 0.04 sec
[ Info: VUMPS 113:	obj = -8.862878980874e-01	err = 2.5711795068e-11	time = 0.04 sec
[ Info: VUMPS 114:	obj = -8.862878980874e-01	err = 3.1841855751e-11	time = 0.09 sec
[ Info: VUMPS 115:	obj = -8.862878980874e-01	err = 2.6088486457e-11	time = 0.04 sec
[ Info: VUMPS 116:	obj = -8.862878980874e-01	err = 2.5759528213e-11	time = 0.04 sec
[ Info: VUMPS 117:	obj = -8.862878980874e-01	err = 2.5550720189e-11	time = 0.04 sec
[ Info: VUMPS 118:	obj = -8.862878980874e-01	err = 2.5802019915e-11	time = 0.04 sec
[ Info: VUMPS 119:	obj = -8.862878980874e-01	err = 2.5672788136e-11	time = 0.04 sec
[ Info: VUMPS 120:	obj = -8.862878980874e-01	err = 2.6669410249e-11	time = 0.08 sec
[ Info: VUMPS 121:	obj = -8.862878980874e-01	err = 2.5760097393e-11	time = 0.04 sec
[ Info: VUMPS 122:	obj = -8.862878980874e-01	err = 3.1549960796e-11	time = 0.04 sec
[ Info: VUMPS 123:	obj = -8.862878980874e-01	err = 3.1008997814e-11	time = 0.04 sec
[ Info: VUMPS 124:	obj = -8.862878980874e-01	err = 3.1957370730e-11	time = 0.08 sec
[ Info: VUMPS 125:	obj = -8.862878980874e-01	err = 1.0149285465e-10	time = 0.07 sec
[ Info: VUMPS 126:	obj = -8.862878980874e-01	err = 1.0055628069e-10	time = 0.06 sec
[ Info: VUMPS 127:	obj = -8.862878980874e-01	err = 2.5279001435e-10	time = 0.05 sec
[ Info: VUMPS 128:	obj = -8.862878980874e-01	err = 1.0462278313e-10	time = 0.13 sec
[ Info: VUMPS 129:	obj = -8.862878980874e-01	err = 6.5624092038e-11	time = 0.05 sec
[ Info: VUMPS 130:	obj = -8.862878980874e-01	err = 1.3309340506e-10	time = 0.06 sec
[ Info: VUMPS 131:	obj = -8.862878980874e-01	err = 7.2784603802e-11	time = 0.05 sec
[ Info: VUMPS 132:	obj = -8.862878980874e-01	err = 7.2178376983e-11	time = 0.09 sec
[ Info: VUMPS 133:	obj = -8.862878980874e-01	err = 9.2518414406e-11	time = 0.05 sec
[ Info: VUMPS 134:	obj = -8.862878980874e-01	err = 1.4506025355e-10	time = 0.05 sec
[ Info: VUMPS 135:	obj = -8.862878980874e-01	err = 3.7219458775e-10	time = 0.06 sec
[ Info: VUMPS 136:	obj = -8.862878980874e-01	err = 1.3286196929e-10	time = 0.14 sec
[ Info: VUMPS 137:	obj = -8.862878980874e-01	err = 7.4787469080e-11	time = 0.06 sec
[ Info: VUMPS 138:	obj = -8.862878980874e-01	err = 1.4426681861e-10	time = 0.05 sec
[ Info: VUMPS 139:	obj = -8.862878980874e-01	err = 6.7558225447e-11	time = 0.05 sec
[ Info: VUMPS 140:	obj = -8.862878980874e-01	err = 7.3084667829e-11	time = 0.04 sec
[ Info: VUMPS 141:	obj = -8.862878980874e-01	err = 8.0793055079e-11	time = 0.11 sec
[ Info: VUMPS 142:	obj = -8.862878980874e-01	err = 8.0831600216e-11	time = 0.04 sec
[ Info: VUMPS 143:	obj = -8.862878980874e-01	err = 1.7136182241e-10	time = 0.05 sec
[ Info: VUMPS 144:	obj = -8.862878980874e-01	err = 9.1343736608e-11	time = 0.06 sec
[ Info: VUMPS 145:	obj = -8.862878980874e-01	err = 8.0089047570e-11	time = 0.09 sec
[ Info: VUMPS 146:	obj = -8.862878980874e-01	err = 1.2911200272e-10	time = 0.05 sec
[ Info: VUMPS 147:	obj = -8.862878980874e-01	err = 3.2779317163e-11	time = 0.05 sec
[ Info: VUMPS 148:	obj = -8.862878980874e-01	err = 3.1270641488e-11	time = 0.04 sec
[ Info: VUMPS 149:	obj = -8.862878980874e-01	err = 3.1527595715e-11	time = 0.04 sec
[ Info: VUMPS 150:	obj = -8.862878980874e-01	err = 3.1023821946e-11	time = 0.08 sec
[ Info: VUMPS 151:	obj = -8.862878980874e-01	err = 3.0958513933e-11	time = 0.04 sec
[ Info: VUMPS 152:	obj = -8.862878980874e-01	err = 3.1072862070e-11	time = 0.04 sec
[ Info: VUMPS 153:	obj = -8.862878980874e-01	err = 3.5383015513e-11	time = 0.04 sec
[ Info: VUMPS 154:	obj = -8.862878980874e-01	err = 9.8524530302e-11	time = 0.07 sec
[ Info: VUMPS 155:	obj = -8.862878980874e-01	err = 2.5977122435e-10	time = 0.10 sec
[ Info: VUMPS 156:	obj = -8.862878980874e-01	err = 1.0280150178e-10	time = 0.08 sec
[ Info: VUMPS 157:	obj = -8.862878980874e-01	err = 8.5396875631e-11	time = 0.06 sec
[ Info: VUMPS 158:	obj = -8.862878980874e-01	err = 1.4028825209e-10	time = 0.05 sec
[ Info: VUMPS 159:	obj = -8.862878980874e-01	err = 6.5407958603e-11	time = 0.09 sec
[ Info: VUMPS 160:	obj = -8.862878980874e-01	err = 8.6364408976e-11	time = 0.05 sec
[ Info: VUMPS 161:	obj = -8.862878980874e-01	err = 1.2150351995e-10	time = 0.05 sec
[ Info: VUMPS 162:	obj = -8.862878980874e-01	err = 2.4842761534e-11	time = 0.05 sec
[ Info: VUMPS 163:	obj = -8.862878980874e-01	err = 2.5748890253e-11	time = 0.09 sec
[ Info: VUMPS 164:	obj = -8.862878980874e-01	err = 3.7022391153e-11	time = 0.04 sec
[ Info: VUMPS 165:	obj = -8.862878980874e-01	err = 2.5404069966e-11	time = 0.04 sec
[ Info: VUMPS 166:	obj = -8.862878980874e-01	err = 2.4702363948e-11	time = 0.04 sec
[ Info: VUMPS 167:	obj = -8.862878980874e-01	err = 2.4860774600e-11	time = 0.04 sec
[ Info: VUMPS 168:	obj = -8.862878980874e-01	err = 3.2324495346e-11	time = 0.04 sec
[ Info: VUMPS 169:	obj = -8.862878980874e-01	err = 2.4719353428e-11	time = 0.09 sec
[ Info: VUMPS 170:	obj = -8.862878980874e-01	err = 2.5745806649e-11	time = 0.04 sec
[ Info: VUMPS 171:	obj = -8.862878980874e-01	err = 3.2930196760e-11	time = 0.04 sec
[ Info: VUMPS 172:	obj = -8.862878980874e-01	err = 2.5085034713e-11	time = 0.04 sec
[ Info: VUMPS 173:	obj = -8.862878980874e-01	err = 2.5520036513e-11	time = 0.04 sec
[ Info: VUMPS 174:	obj = -8.862878980874e-01	err = 2.5135503465e-11	time = 0.08 sec
[ Info: VUMPS 175:	obj = -8.862878980874e-01	err = 2.8503905090e-11	time = 0.04 sec
[ Info: VUMPS 176:	obj = -8.862878980874e-01	err = 2.5538540554e-11	time = 0.04 sec
[ Info: VUMPS 177:	obj = -8.862878980874e-01	err = 2.4691024801e-11	time = 0.04 sec
[ Info: VUMPS 178:	obj = -8.862878980874e-01	err = 2.5248726569e-11	time = 0.08 sec
[ Info: VUMPS 179:	obj = -8.862878980874e-01	err = 2.4760967560e-11	time = 0.04 sec
[ Info: VUMPS 180:	obj = -8.862878980874e-01	err = 2.4969607389e-11	time = 0.04 sec
[ Info: VUMPS 181:	obj = -8.862878980874e-01	err = 2.5195199024e-11	time = 0.04 sec
[ Info: VUMPS 182:	obj = -8.862878980874e-01	err = 2.4838502094e-11	time = 0.04 sec
[ Info: VUMPS 183:	obj = -8.862878980874e-01	err = 2.6113063075e-11	time = 0.08 sec
[ Info: VUMPS 184:	obj = -8.862878980874e-01	err = 2.9908520201e-11	time = 0.04 sec
[ Info: VUMPS 185:	obj = -8.862878980874e-01	err = 1.0991210900e-09	time = 0.05 sec
[ Info: VUMPS 186:	obj = -8.862878980874e-01	err = 1.3487615412e-10	time = 0.10 sec
[ Info: VUMPS 187:	obj = -8.862878980874e-01	err = 9.3069733146e-11	time = 0.11 sec
[ Info: VUMPS 188:	obj = -8.862878980874e-01	err = 5.9219170076e-11	time = 0.05 sec
[ Info: VUMPS 189:	obj = -8.862878980874e-01	err = 1.2416000800e-10	time = 0.05 sec
[ Info: VUMPS 190:	obj = -8.862878980874e-01	err = 1.3881531446e-10	time = 0.07 sec
[ Info: VUMPS 191:	obj = -8.862878980874e-01	err = 1.2937777179e-10	time = 0.12 sec
[ Info: VUMPS 192:	obj = -8.862878980874e-01	err = 1.7091864452e-10	time = 0.07 sec
[ Info: VUMPS 193:	obj = -8.862878980874e-01	err = 1.3809278028e-10	time = 0.07 sec
[ Info: VUMPS 194:	obj = -8.862878980874e-01	err = 1.3718743400e-10	time = 0.07 sec
[ Info: VUMPS 195:	obj = -8.862878980874e-01	err = 1.2999331067e-10	time = 0.12 sec
[ Info: VUMPS 196:	obj = -8.862878980874e-01	err = 1.0286658958e-10	time = 0.06 sec
[ Info: VUMPS 197:	obj = -8.862878980874e-01	err = 8.5927628448e-11	time = 0.06 sec
[ Info: VUMPS 198:	obj = -8.862878980874e-01	err = 9.8964650709e-11	time = 0.11 sec
[ Info: VUMPS 199:	obj = -8.862878980874e-01	err = 1.1996588094e-10	time = 0.05 sec
[ Info: VUMPS 200:	obj = -8.862878980874e-01	err = 5.6341554935e-11	time = 0.04 sec
[ Info: VUMPS 201:	obj = -8.862878980874e-01	err = 7.9000713006e-10	time = 0.06 sec
[ Info: VUMPS 202:	obj = -8.862878980874e-01	err = 1.5435808912e-10	time = 0.14 sec
[ Info: VUMPS 203:	obj = -8.862878980874e-01	err = 9.5848753583e-11	time = 0.08 sec
[ Info: VUMPS 204:	obj = -8.862878980874e-01	err = 1.0637816317e-10	time = 0.07 sec
[ Info: VUMPS 205:	obj = -8.862878980874e-01	err = 8.1143149644e-11	time = 0.06 sec
[ Info: VUMPS 206:	obj = -8.862878980874e-01	err = 1.0842849551e-10	time = 0.09 sec
[ Info: VUMPS 207:	obj = -8.862878980874e-01	err = 5.5594973835e-11	time = 0.05 sec
[ Info: VUMPS 208:	obj = -8.862878980874e-01	err = 5.5599308569e-11	time = 0.04 sec
[ Info: VUMPS 209:	obj = -8.862878980874e-01	err = 4.5664641092e-10	time = 0.05 sec
[ Info: VUMPS 210:	obj = -8.862878980874e-01	err = 1.0815766206e-10	time = 0.13 sec
[ Info: VUMPS 211:	obj = -8.862878980874e-01	err = 9.2896882921e-11	time = 0.07 sec
[ Info: VUMPS 212:	obj = -8.862878980874e-01	err = 8.3630387677e-11	time = 0.05 sec
[ Info: VUMPS 213:	obj = -8.862878980874e-01	err = 8.9177443058e-11	time = 0.07 sec
[ Info: VUMPS 214:	obj = -8.862878980874e-01	err = 8.4590201305e-11	time = 0.11 sec
[ Info: VUMPS 215:	obj = -8.862878980874e-01	err = 1.2973719725e-10	time = 0.07 sec
[ Info: VUMPS 216:	obj = -8.862878980874e-01	err = 8.7532935985e-11	time = 0.07 sec
[ Info: VUMPS 217:	obj = -8.862878980874e-01	err = 1.0094432065e-10	time = 0.05 sec
[ Info: VUMPS 218:	obj = -8.862878980874e-01	err = 7.2901490544e-11	time = 0.10 sec
[ Info: VUMPS 219:	obj = -8.862878980874e-01	err = 7.3030080658e-11	time = 0.04 sec
[ Info: VUMPS 220:	obj = -8.862878980874e-01	err = 7.2814253578e-11	time = 0.04 sec
[ Info: VUMPS 221:	obj = -8.862878980874e-01	err = 7.2721737007e-11	time = 0.04 sec
[ Info: VUMPS 222:	obj = -8.862878980874e-01	err = 7.2575908766e-11	time = 0.04 sec
[ Info: VUMPS 223:	obj = -8.862878980874e-01	err = 1.0886060787e-10	time = 0.09 sec
[ Info: VUMPS 224:	obj = -8.862878980874e-01	err = 9.8891539544e-11	time = 0.05 sec
[ Info: VUMPS 225:	obj = -8.862878980874e-01	err = 1.0482184140e-10	time = 0.05 sec
[ Info: VUMPS 226:	obj = -8.862878980874e-01	err = 1.9012146089e-11	time = 0.04 sec
[ Info: VUMPS 227:	obj = -8.862878980874e-01	err = 1.9322391656e-11	time = 0.04 sec
[ Info: VUMPS 228:	obj = -8.862878980874e-01	err = 3.0618708554e-11	time = 0.08 sec
[ Info: VUMPS 229:	obj = -8.862878980874e-01	err = 7.3510645058e-11	time = 0.06 sec
[ Info: VUMPS 230:	obj = -8.862878980874e-01	err = 3.8581370951e-10	time = 0.05 sec
[ Info: VUMPS 231:	obj = -8.862878980874e-01	err = 1.3190179067e-10	time = 0.09 sec
[ Info: VUMPS 232:	obj = -8.862878980874e-01	err = 9.8345519722e-11	time = 0.11 sec
[ Info: VUMPS 233:	obj = -8.862878980874e-01	err = 9.8692106436e-11	time = 0.06 sec
[ Info: VUMPS 234:	obj = -8.862878980874e-01	err = 5.0517584685e-11	time = 0.05 sec
[ Info: VUMPS 235:	obj = -8.862878980874e-01	err = 5.2479684150e-11	time = 0.04 sec
[ Info: VUMPS 236:	obj = -8.862878980874e-01	err = 1.0842073163e-10	time = 0.11 sec
[ Info: VUMPS 237:	obj = -8.862878980874e-01	err = 9.0981410086e-11	time = 0.06 sec
[ Info: VUMPS 238:	obj = -8.862878980874e-01	err = 1.2392458729e-10	time = 0.05 sec
[ Info: VUMPS 239:	obj = -8.862878980874e-01	err = 1.4216588231e-10	time = 0.06 sec
[ Info: VUMPS 240:	obj = -8.862878980874e-01	err = 1.0202004549e-10	time = 0.12 sec
[ Info: VUMPS 241:	obj = -8.862878980874e-01	err = 9.2663307948e-11	time = 0.07 sec
[ Info: VUMPS 242:	obj = -8.862878980874e-01	err = 7.1213927824e-11	time = 0.06 sec
[ Info: VUMPS 243:	obj = -8.862878980874e-01	err = 7.0990626515e-11	time = 0.06 sec
[ Info: VUMPS 244:	obj = -8.862878980874e-01	err = 7.1367091900e-11	time = 0.04 sec
[ Info: VUMPS 245:	obj = -8.862878980874e-01	err = 1.8669503143e-10	time = 0.09 sec
[ Info: VUMPS 246:	obj = -8.862878980874e-01	err = 1.5282765127e-10	time = 0.08 sec
[ Info: VUMPS 247:	obj = -8.862878980874e-01	err = 9.6415117857e-11	time = 0.07 sec
[ Info: VUMPS 248:	obj = -8.862878980874e-01	err = 1.4058659748e-10	time = 0.11 sec
[ Info: VUMPS 249:	obj = -8.862878980874e-01	err = 9.7187257969e-11	time = 0.07 sec
[ Info: VUMPS 250:	obj = -8.862878980874e-01	err = 9.7759239537e-11	time = 0.05 sec
[ Info: VUMPS 251:	obj = -8.862878980874e-01	err = 7.9875821567e-11	time = 0.07 sec
[ Info: VUMPS 252:	obj = -8.862878980874e-01	err = 8.0093896508e-11	time = 0.08 sec
[ Info: VUMPS 253:	obj = -8.862878980874e-01	err = 8.0057483179e-11	time = 0.04 sec
[ Info: VUMPS 254:	obj = -8.862878980874e-01	err = 8.0011136205e-11	time = 0.04 sec
[ Info: VUMPS 255:	obj = -8.862878980874e-01	err = 1.7883745718e-10	time = 0.05 sec
[ Info: VUMPS 256:	obj = -8.862878980874e-01	err = 2.8363520443e-10	time = 0.06 sec
[ Info: VUMPS 257:	obj = -8.862878980874e-01	err = 1.4712790190e-10	time = 0.12 sec
[ Info: VUMPS 258:	obj = -8.862878980874e-01	err = 8.0262189893e-11	time = 0.05 sec
[ Info: VUMPS 259:	obj = -8.862878980874e-01	err = 1.3607892547e-10	time = 0.05 sec
[ Info: VUMPS 260:	obj = -8.862878980874e-01	err = 3.5585465386e-11	time = 0.09 sec
[ Info: VUMPS 261:	obj = -8.862878980874e-01	err = 3.5886789540e-11	time = 0.04 sec
[ Info: VUMPS 262:	obj = -8.862878980874e-01	err = 3.5821881811e-11	time = 0.04 sec
[ Info: VUMPS 263:	obj = -8.862878980874e-01	err = 3.5295520311e-11	time = 0.04 sec
[ Info: VUMPS 264:	obj = -8.862878980874e-01	err = 3.5725163978e-11	time = 0.04 sec
[ Info: VUMPS 265:	obj = -8.862878980874e-01	err = 3.5743028152e-11	time = 0.09 sec
[ Info: VUMPS 266:	obj = -8.862878980874e-01	err = 3.5399343199e-11	time = 0.04 sec
[ Info: VUMPS 267:	obj = -8.862878980874e-01	err = 3.5477332519e-11	time = 0.04 sec
[ Info: VUMPS 268:	obj = -8.862878980874e-01	err = 3.5528256597e-11	time = 0.04 sec
[ Info: VUMPS 269:	obj = -8.862878980874e-01	err = 3.5192330992e-11	time = 0.04 sec
[ Info: VUMPS 270:	obj = -8.862878980874e-01	err = 3.5801828668e-11	time = 0.04 sec
[ Info: VUMPS 271:	obj = -8.862878980874e-01	err = 3.4963884022e-11	time = 0.08 sec
[ Info: VUMPS 272:	obj = -8.862878980874e-01	err = 3.5992081275e-11	time = 0.04 sec
[ Info: VUMPS 273:	obj = -8.862878980874e-01	err = 3.5473635188e-11	time = 0.04 sec
[ Info: VUMPS 274:	obj = -8.862878980874e-01	err = 3.5704579427e-11	time = 0.04 sec
[ Info: VUMPS 275:	obj = -8.862878980874e-01	err = 3.5442668182e-11	time = 0.04 sec
[ Info: VUMPS 276:	obj = -8.862878980874e-01	err = 3.5482076714e-11	time = 0.08 sec
[ Info: VUMPS 277:	obj = -8.862878980874e-01	err = 3.9356745202e-11	time = 0.04 sec
[ Info: VUMPS 278:	obj = -8.862878980874e-01	err = 4.0210942413e-11	time = 0.04 sec
[ Info: VUMPS 279:	obj = -8.862878980874e-01	err = 3.5167580179e-11	time = 0.04 sec
[ Info: VUMPS 280:	obj = -8.862878980874e-01	err = 3.6493383635e-11	time = 0.04 sec
[ Info: VUMPS 281:	obj = -8.862878980874e-01	err = 3.5805399155e-11	time = 0.08 sec
[ Info: VUMPS 282:	obj = -8.862878980874e-01	err = 3.5166101177e-11	time = 0.04 sec
[ Info: VUMPS 283:	obj = -8.862878980874e-01	err = 3.7246847607e-11	time = 0.04 sec
[ Info: VUMPS 284:	obj = -8.862878980874e-01	err = 4.0568537077e-11	time = 0.04 sec
[ Info: VUMPS 285:	obj = -8.862878980874e-01	err = 3.5491580364e-11	time = 0.04 sec
[ Info: VUMPS 286:	obj = -8.862878980874e-01	err = 3.5099655639e-11	time = 0.08 sec
[ Info: VUMPS 287:	obj = -8.862878980874e-01	err = 3.5165898259e-11	time = 0.04 sec
[ Info: VUMPS 288:	obj = -8.862878980874e-01	err = 3.5582900108e-11	time = 0.04 sec
[ Info: VUMPS 289:	obj = -8.862878980874e-01	err = 3.5267676837e-11	time = 0.04 sec
[ Info: VUMPS 290:	obj = -8.862878980874e-01	err = 3.5547016549e-11	time = 0.08 sec
[ Info: VUMPS 291:	obj = -8.862878980874e-01	err = 3.5194773385e-11	time = 0.04 sec
[ Info: VUMPS 292:	obj = -8.862878980874e-01	err = 3.8861800688e-11	time = 0.04 sec
[ Info: VUMPS 293:	obj = -8.862878980874e-01	err = 1.7694618399e-10	time = 0.06 sec
[ Info: VUMPS 294:	obj = -8.862878980874e-01	err = 9.6879691257e-11	time = 0.08 sec
[ Info: VUMPS 295:	obj = -8.862878980874e-01	err = 1.9734638828e-10	time = 0.09 sec
[ Info: VUMPS 296:	obj = -8.862878980874e-01	err = 9.5735210562e-11	time = 0.07 sec
[ Info: VUMPS 297:	obj = -8.862878980874e-01	err = 9.8202831809e-11	time = 0.06 sec
[ Info: VUMPS 298:	obj = -8.862878980874e-01	err = 1.0619064963e-10	time = 0.05 sec
[ Info: VUMPS 299:	obj = -8.862878980874e-01	err = 6.3326732099e-11	time = 0.09 sec
[ Info: VUMPS 300:	obj = -8.862878980874e-01	err = 6.3186774093e-11	time = 0.04 sec
[ Info: VUMPS 301:	obj = -8.862878980874e-01	err = 6.3238257167e-11	time = 0.04 sec
[ Info: VUMPS 302:	obj = -8.862878980874e-01	err = 7.8177257008e-11	time = 0.06 sec
[ Info: VUMPS 303:	obj = -8.862878980874e-01	err = 7.8085760822e-11	time = 0.08 sec
[ Info: VUMPS 304:	obj = -8.862878980874e-01	err = 7.3527317393e-11	time = 0.04 sec
[ Info: VUMPS 305:	obj = -8.862878980874e-01	err = 9.9416857299e-11	time = 0.04 sec
[ Info: VUMPS 306:	obj = -8.862878980874e-01	err = 5.4494619202e-11	time = 0.05 sec
[ Info: VUMPS 307:	obj = -8.862878980874e-01	err = 5.7062728379e-11	time = 0.04 sec
[ Info: VUMPS 308:	obj = -8.862878980874e-01	err = 5.4603016721e-11	time = 0.08 sec
[ Info: VUMPS 309:	obj = -8.862878980874e-01	err = 5.4567688376e-11	time = 0.04 sec
[ Info: VUMPS 310:	obj = -8.862878980874e-01	err = 5.7938510837e-11	time = 0.04 sec
[ Info: VUMPS 311:	obj = -8.862878980874e-01	err = 2.0622949240e-10	time = 0.06 sec
[ Info: VUMPS 312:	obj = -8.862878980874e-01	err = 1.3121811134e-10	time = 0.08 sec
[ Info: VUMPS 313:	obj = -8.862878980874e-01	err = 9.1235627109e-11	time = 0.10 sec
[ Info: VUMPS 314:	obj = -8.862878980874e-01	err = 8.4669915126e-11	time = 0.06 sec
[ Info: VUMPS 315:	obj = -8.862878980874e-01	err = 2.3252823264e-10	time = 0.05 sec
[ Info: VUMPS 316:	obj = -8.862878980874e-01	err = 1.1096003071e-10	time = 0.13 sec
[ Info: VUMPS 317:	obj = -8.862878980874e-01	err = 1.0920664731e-10	time = 0.06 sec
[ Info: VUMPS 318:	obj = -8.862878980874e-01	err = 8.8285347864e-11	time = 0.06 sec
[ Info: VUMPS 319:	obj = -8.862878980874e-01	err = 2.0864332560e-10	time = 0.06 sec
[ Info: VUMPS 320:	obj = -8.862878980874e-01	err = 1.3027532605e-10	time = 0.13 sec
[ Info: VUMPS 321:	obj = -8.862878980874e-01	err = 9.1835678484e-11	time = 0.06 sec
[ Info: VUMPS 322:	obj = -8.862878980874e-01	err = 1.0506692631e-10	time = 0.07 sec
[ Info: VUMPS 323:	obj = -8.862878980874e-01	err = 6.5764134932e-11	time = 0.05 sec
[ Info: VUMPS 324:	obj = -8.862878980874e-01	err = 8.5615212507e-11	time = 0.10 sec
[ Info: VUMPS 325:	obj = -8.862878980874e-01	err = 1.5884913185e-10	time = 0.05 sec
[ Info: VUMPS 326:	obj = -8.862878980874e-01	err = 7.5397480867e-11	time = 0.05 sec
[ Info: VUMPS 327:	obj = -8.862878980874e-01	err = 1.3672488066e-10	time = 0.06 sec
[ Info: VUMPS 328:	obj = -8.862878980874e-01	err = 1.2905960387e-10	time = 0.11 sec
[ Info: VUMPS 329:	obj = -8.862878980874e-01	err = 1.1137546210e-10	time = 0.07 sec
[ Info: VUMPS 330:	obj = -8.862878980874e-01	err = 7.6064748012e-11	time = 0.05 sec
[ Info: VUMPS 331:	obj = -8.862878980874e-01	err = 8.2230706651e-11	time = 0.06 sec
[ Info: VUMPS 332:	obj = -8.862878980874e-01	err = 1.5547070838e-10	time = 0.09 sec
[ Info: VUMPS 333:	obj = -8.862878980874e-01	err = 8.4530602671e-11	time = 0.05 sec
[ Info: VUMPS 334:	obj = -8.862878980874e-01	err = 2.7703069729e-10	time = 0.05 sec
[ Info: VUMPS 335:	obj = -8.862878980874e-01	err = 1.2843108098e-10	time = 0.07 sec
[ Info: VUMPS 336:	obj = -8.862878980874e-01	err = 7.7139997032e-11	time = 0.09 sec
[ Info: VUMPS 337:	obj = -8.862878980874e-01	err = 1.5230480609e-10	time = 0.05 sec
[ Info: VUMPS 338:	obj = -8.862878980874e-01	err = 1.4349332630e-10	time = 0.05 sec
[ Info: VUMPS 339:	obj = -8.862878980874e-01	err = 1.8791131229e-10	time = 0.07 sec
[ Info: VUMPS 340:	obj = -8.862878980874e-01	err = 5.6705763103e-11	time = 0.11 sec
[ Info: VUMPS 341:	obj = -8.862878980874e-01	err = 5.6628206369e-11	time = 0.04 sec
[ Info: VUMPS 342:	obj = -8.862878980874e-01	err = 5.7700869354e-11	time = 0.04 sec
[ Info: VUMPS 343:	obj = -8.862878980874e-01	err = 9.5014822831e-11	time = 0.07 sec
[ Info: VUMPS 344:	obj = -8.862878980874e-01	err = 9.9584310546e-11	time = 0.09 sec
[ Info: VUMPS 345:	obj = -8.862878980874e-01	err = 1.0944426167e-10	time = 0.04 sec
[ Info: VUMPS 346:	obj = -8.862878980874e-01	err = 4.1610580123e-11	time = 0.04 sec
[ Info: VUMPS 347:	obj = -8.862878980874e-01	err = 2.9414073597e-11	time = 0.04 sec
[ Info: VUMPS 348:	obj = -8.862878980874e-01	err = 2.9723554637e-11	time = 0.04 sec
[ Info: VUMPS 349:	obj = -8.862878980874e-01	err = 2.9370382842e-11	time = 0.08 sec
[ Info: VUMPS 350:	obj = -8.862878980874e-01	err = 2.9069190284e-11	time = 0.04 sec
[ Info: VUMPS 351:	obj = -8.862878980874e-01	err = 2.9200689531e-11	time = 0.04 sec
[ Info: VUMPS 352:	obj = -8.862878980874e-01	err = 1.0829490368e-10	time = 0.06 sec
[ Info: VUMPS 353:	obj = -8.862878980874e-01	err = 1.0018375647e-10	time = 0.11 sec
[ Info: VUMPS 354:	obj = -8.862878980874e-01	err = 1.5362554440e-10	time = 0.06 sec
[ Info: VUMPS 355:	obj = -8.862878980874e-01	err = 9.4013179693e-11	time = 0.07 sec
[ Info: VUMPS 356:	obj = -8.862878980874e-01	err = 8.5844015408e-11	time = 0.06 sec
[ Info: VUMPS 357:	obj = -8.862878980874e-01	err = 1.1938734236e-10	time = 0.09 sec
[ Info: VUMPS 358:	obj = -8.862878980874e-01	err = 5.4442781639e-11	time = 0.04 sec
[ Info: VUMPS 359:	obj = -8.862878980874e-01	err = 5.6747362650e-11	time = 0.04 sec
[ Info: VUMPS 360:	obj = -8.862878980874e-01	err = 1.1602629725e-10	time = 0.07 sec
[ Info: VUMPS 361:	obj = -8.862878980874e-01	err = 1.2693714656e-10	time = 0.06 sec
[ Info: VUMPS 362:	obj = -8.862878980874e-01	err = 1.3334525033e-10	time = 0.11 sec
[ Info: VUMPS 363:	obj = -8.862878980874e-01	err = 9.4883693020e-11	time = 0.07 sec
[ Info: VUMPS 364:	obj = -8.862878980874e-01	err = 8.9430760837e-11	time = 0.07 sec
[ Info: VUMPS 365:	obj = -8.862878980874e-01	err = 1.0615371329e-10	time = 0.11 sec
[ Info: VUMPS 366:	obj = -8.862878980874e-01	err = 1.1358908029e-10	time = 0.04 sec
[ Info: VUMPS 367:	obj = -8.862878980874e-01	err = 1.2828241179e-10	time = 0.05 sec
[ Info: VUMPS 368:	obj = -8.862878980874e-01	err = 7.5500082303e-11	time = 0.05 sec
[ Info: VUMPS 369:	obj = -8.862878980874e-01	err = 1.7133532560e-10	time = 0.05 sec
[ Info: VUMPS 370:	obj = -8.862878980874e-01	err = 6.2722200745e-11	time = 0.09 sec
[ Info: VUMPS 371:	obj = -8.862878980874e-01	err = 6.2778165324e-11	time = 0.04 sec
[ Info: VUMPS 372:	obj = -8.862878980874e-01	err = 6.4224600150e-10	time = 0.06 sec
[ Info: VUMPS 373:	obj = -8.862878980874e-01	err = 1.4567170198e-10	time = 0.09 sec
[ Info: VUMPS 374:	obj = -8.862878980874e-01	err = 1.0380674404e-10	time = 0.11 sec
[ Info: VUMPS 375:	obj = -8.862878980874e-01	err = 1.0822040195e-10	time = 0.07 sec
[ Info: VUMPS 376:	obj = -8.862878980874e-01	err = 1.0520173008e-10	time = 0.06 sec
[ Info: VUMPS 377:	obj = -8.862878980874e-01	err = 6.7365662418e-11	time = 0.10 sec
[ Info: VUMPS 378:	obj = -8.862878980874e-01	err = 6.7760750536e-11	time = 0.04 sec
[ Info: VUMPS 379:	obj = -8.862878980874e-01	err = 6.7722699479e-11	time = 0.04 sec
[ Info: VUMPS 380:	obj = -8.862878980874e-01	err = 6.7250908434e-11	time = 0.04 sec
[ Info: VUMPS 381:	obj = -8.862878980874e-01	err = 1.4694592798e-10	time = 0.06 sec
[ Info: VUMPS 382:	obj = -8.862878980874e-01	err = 8.9730375422e-11	time = 0.11 sec
[ Info: VUMPS 383:	obj = -8.862878980874e-01	err = 1.6221755374e-10	time = 0.06 sec
[ Info: VUMPS 384:	obj = -8.862878980874e-01	err = 1.2999407274e-10	time = 0.08 sec
[ Info: VUMPS 385:	obj = -8.862878980874e-01	err = 8.5732018525e-11	time = 0.06 sec
[ Info: VUMPS 386:	obj = -8.862878980874e-01	err = 1.6277251600e-10	time = 0.09 sec
[ Info: VUMPS 387:	obj = -8.862878980874e-01	err = 7.1550742938e-11	time = 0.04 sec
[ Info: VUMPS 388:	obj = -8.862878980874e-01	err = 7.4376591685e-11	time = 0.06 sec
[ Info: VUMPS 389:	obj = -8.862878980874e-01	err = 7.4713904887e-11	time = 0.04 sec
[ Info: VUMPS 390:	obj = -8.862878980874e-01	err = 7.4436898750e-11	time = 0.08 sec
[ Info: VUMPS 391:	obj = -8.862878980874e-01	err = 7.5772899650e-11	time = 0.04 sec
[ Info: VUMPS 392:	obj = -8.862878980874e-01	err = 7.4943197420e-11	time = 0.04 sec
[ Info: VUMPS 393:	obj = -8.862878980874e-01	err = 7.4400646683e-11	time = 0.04 sec
[ Info: VUMPS 394:	obj = -8.862878980874e-01	err = 7.5613615370e-11	time = 0.04 sec
[ Info: VUMPS 395:	obj = -8.862878980874e-01	err = 1.0859523215e-10	time = 0.11 sec
[ Info: VUMPS 396:	obj = -8.862878980874e-01	err = 9.0566692582e-11	time = 0.05 sec
[ Info: VUMPS 397:	obj = -8.862878980874e-01	err = 4.3667065720e-11	time = 0.05 sec
[ Info: VUMPS 398:	obj = -8.862878980874e-01	err = 4.3513888787e-11	time = 0.04 sec
[ Info: VUMPS 399:	obj = -8.862878980874e-01	err = 4.4283893567e-11	time = 0.04 sec
┌ Warning: VUMPS cancel 400:	obj = -8.862878980874e-01	err = 4.3759074890e-11	time = 28.78 sec
└ @ MPSKit ~/Projects/MPSKit.jl/src/algorithms/groundstate/vumps.jl:67

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

