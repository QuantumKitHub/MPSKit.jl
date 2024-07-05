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
[ Info: VUMPS init:	obj = +2.499974103828e-01	err = 4.7612e-03
[ Info: VUMPS   1:	obj = -1.619489094120e-01	err = 3.8111216818e-01	time = 0.09 sec
[ Info: VUMPS   2:	obj = -1.182665508511e-01	err = 4.0364154268e-01	time = 0.03 sec
┌ Warning: ignoring imaginary component 1.232882527538344e-6 from total weight 1.9914657594448284: operator might not be hermitian?
│   α = 0.25049121255958223 + 1.232882527538344e-6im
│   β₁ = 1.3726963690475595
│   β₂ = 1.4208781446252607
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.8392420257293263e-6 from total weight 1.9696900832784099: operator might not be hermitian?
│   α = 0.13613443595890026 + 1.8392420257293263e-6im
│   β₁ = 1.4208781446252607
│   β₂ = 1.3572957443513758
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.0830449709096218e-6 from total weight 1.913217098985634: operator might not be hermitian?
│   α = 0.30490943522699615 + 1.0830449709096218e-6im
│   β₁ = 1.3572957443513758
│   β₂ = 1.3134603787419823
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.3349880196560776e-6 from total weight 1.8091684366416438: operator might not be hermitian?
│   α = 0.15356513905659896 - 1.3349880196560776e-6im
│   β₁ = 1.2453864803058963
│   β₂ = 1.3032730699574846
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1612816465528253e-6 from total weight 1.8643524554082331: operator might not be hermitian?
│   α = 0.06307341568508065 - 1.1612816465528253e-6im
│   β₁ = 1.3146485309683946
│   β₂ = 1.3204283631616742
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.5596982755562358e-6 from total weight 1.8141643749302523: operator might not be hermitian?
│   α = 0.08072855112812488 - 1.5596982755562358e-6im
│   β₁ = 1.3204283631616742
│   β₂ = 1.2414282975889408
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.417172544438891e-6 from total weight 1.828785206932206: operator might not be hermitian?
│   α = 0.1285519752447895 - 1.417172544438891e-6im
│   β₁ = 1.2414282975889408
│   β₂ = 1.3367069629122716
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.289301417652794e-6 from total weight 1.885408518558631: operator might not be hermitian?
│   α = 0.08120850910419936 - 1.289301417652794e-6im
│   β₁ = 1.3367069629122716
│   β₂ = 1.3271717881279397
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.6249186336921018e-6 from total weight 1.8407680177572983: operator might not be hermitian?
│   α = 0.1289941860896659 - 1.6249186336921018e-6im
│   β₁ = 1.3271717881279397
│   β₂ = 1.2690163276915885
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0035504485646218e-6 from total weight 1.7396456869769024: operator might not be hermitian?
│   α = 0.19480800880290103 - 1.0035504485646218e-6im
│   β₁ = 1.2043364223883024
│   β₂ = 1.2401575454882836
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.451556656376779e-6 from total weight 1.7493717573021048: operator might not be hermitian?
│   α = 0.05705041450756472 - 1.451556656376779e-6im
│   β₁ = 1.2401575454882836
│   β₂ = 1.232499922035348
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.40004089359963e-6 from total weight 1.7309638146775101: operator might not be hermitian?
│   α = 0.15157604727739213 - 1.40004089359963e-6im
│   β₁ = 1.232499922035348
│   β₂ = 1.2059039645824123
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.15482454825927e-6 from total weight 1.918824053662481: operator might not be hermitian?
│   α = 0.5324714992933328 - 1.15482454825927e-6im
│   β₁ = 1.370524157251734
│   β₂ = 1.2328922847281172
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.5003680336328484e-6 from total weight 1.7004217179907795: operator might not be hermitian?
│   α = 0.08096111237620557 - 1.5003680336328484e-6im
│   β₁ = 1.2328922847281172
│   β₂ = 1.1682704873243839
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.8850326343952672e-6 from total weight 1.7755134739648675: operator might not be hermitian?
│   α = 0.21745063050632352 + 1.8850326343952672e-6im
│   β₁ = 1.2496335036506379
│   β₂ = 1.2424087194130087
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component 1.6318123482791241e-6 from total weight 1.6984384594060646: operator might not be hermitian?
│   α = 0.242203135278063 + 1.6318123482791241e-6im
│   β₁ = 1.2424087194130087
│   β₂ = 1.1324537145397988
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.3524781333182643e-6 from total weight 1.5660948460042454: operator might not be hermitian?
│   α = 0.20666413466810546 - 2.3524781333182643e-6im
│   β₁ = 1.0954146707412835
│   β₂ = 1.1000044096466957
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.5085042947292635e-6 from total weight 1.544753413057564: operator might not be hermitian?
│   α = 0.18456051333561468 - 2.5085042947292635e-6im
│   β₁ = 1.1000044096466957
│   β₂ = 1.0687332795519278
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.3454185131416283e-6 from total weight 1.5301327186155422: operator might not be hermitian?
│   α = 0.20034798781882637 - 2.3454185131416283e-6im
│   β₁ = 1.0687332795519278
│   β₂ = 1.0765574752549987
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.179886915059214e-6 from total weight 1.5137298893013147: operator might not be hermitian?
│   α = 0.19578629989152452 - 1.179886915059214e-6im
│   β₁ = 1.0765574752549987
│   β₂ = 1.0459779658339594
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -8.970966421210924e-7 from total weight 1.5259294767522584: operator might not be hermitian?
│   α = 0.15682426449605152 - 8.970966421210924e-7im
│   β₁ = 1.0459779658339594
│   β₂ = 1.0999077293463033
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.153581410850285e-6 from total weight 1.5895749085881306: operator might not be hermitian?
│   α = 0.17753069938549312 - 1.153581410850285e-6im
│   β₁ = 1.0999077293463033
│   β₂ = 1.1337699183306944
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -9.015247489363443e-7 from total weight 1.564145191103578: operator might not be hermitian?
│   α = 0.1101277597941326 - 9.015247489363443e-7im
│   β₁ = 1.1337699183306944
│   β₂ = 1.0719084978032278
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.5952526306768633e-6 from total weight 1.5301971849916518: operator might not be hermitian?
│   α = 0.22488518188133266 - 1.5952526306768633e-6im
│   β₁ = 1.0719084978032278
│   β₂ = 1.0686169810840824
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.5466909599401446e-6 from total weight 1.5491771112150903: operator might not be hermitian?
│   α = 0.3489354106585746 - 1.5466909599401446e-6im
│   β₁ = 1.0686169810840824
│   β₂ = 1.0659510067717062
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -9.445115051708031e-7 from total weight 1.5680129741020914: operator might not be hermitian?
│   α = 0.17418265114627196 - 9.445115051708031e-7im
│   β₁ = 1.1169232430393412
│   β₂ = 1.0866496952328888
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.5724137275956174e-6 from total weight 1.5312701489526515: operator might not be hermitian?
│   α = 0.16742280266428738 - 1.5724137275956174e-6im
│   β₁ = 1.0866496952328888
│   β₂ = 1.0658096988061743
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.166518523731258e-6 from total weight 1.5187736631710762: operator might not be hermitian?
│   α = 0.1398329444117618 - 1.166518523731258e-6im
│   β₁ = 1.0658096988061743
│   β₂ = 1.0729258471714427
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
[ Info: VUMPS   3:	obj = -1.493203861348e-01	err = 3.6296059201e-01	time = 0.03 sec
[ Info: VUMPS   4:	obj = -7.520251536244e-03	err = 3.9887360051e-01	time = 0.03 sec
┌ Warning: ignoring imaginary component 1.099900984910876e-6 from total weight 1.552718086186024: operator might not be hermitian?
│   α = -0.4505116196691295 + 1.099900984910876e-6im
│   β₁ = 1.0412366100283246
│   β₂ = 1.0600938909586262
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
[ Info: VUMPS   5:	obj = -2.352760184979e-01	err = 3.6013144030e-01	time = 0.02 sec
[ Info: VUMPS   6:	obj = -5.963750087299e-02	err = 3.9393533227e-01	time = 0.03 sec
┌ Warning: ignoring imaginary component -1.176743669343422e-6 from total weight 1.918106632926099: operator might not be hermitian?
│   α = 0.0783735962187843 - 1.176743669343422e-6im
│   β₁ = 1.396721847865214
│   β₂ = 1.3123104489355517
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0609971544759539e-6 from total weight 1.9196120278732995: operator might not be hermitian?
│   α = -0.021605437831401036 - 1.0609971544759539e-6im
│   β₁ = 1.3123104489355517
│   β₂ = 1.400815772407421
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0518092979673599e-6 from total weight 1.7924480070709155: operator might not be hermitian?
│   α = 0.04549748143559522 - 1.0518092979673599e-6im
│   β₁ = 1.2767035144224377
│   β₂ = 1.2573098160341405
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1231990716814577e-6 from total weight 1.6627984335729582: operator might not be hermitian?
│   α = 0.12980004518430788 - 1.1231990716814577e-6im
│   β₁ = 1.1464302734823937
│   β₂ = 1.197392252774611
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0130783653500286e-6 from total weight 1.6889379781104494: operator might not be hermitian?
│   α = 0.20140031806791586 - 1.0130783653500286e-6im
│   β₁ = 1.197392252774611
│   β₂ = 1.1739681421487853
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -9.379141249210576e-7 from total weight 1.5376281653096207: operator might not be hermitian?
│   α = 0.14438442595398784 - 9.379141249210576e-7im
│   β₁ = 1.1580977518851725
│   β₂ = 1.0011309152019015
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -8.680933230274801e-7 from total weight 1.5537236737888407: operator might not be hermitian?
│   α = 0.10991283382656072 - 8.680933230274801e-7im
│   β₁ = 1.0657973418412914
│   β₂ = 1.1251899615511112
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -9.723500291637655e-7 from total weight 1.5771422136428674: operator might not be hermitian?
│   α = 0.13579571306157712 - 9.723500291637655e-7im
│   β₁ = 1.1166899055108623
│   β₂ = 1.1054142849165727
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -9.882280200112364e-7 from total weight 1.5180688894033822: operator might not be hermitian?
│   α = 0.09617375331040615 - 9.882280200112364e-7im
│   β₁ = 1.1054142849165727
│   β₂ = 1.036022693211888
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.2660510717039247e-6 from total weight 1.5255263117407745: operator might not be hermitian?
│   α = 0.13347806049420893 - 1.2660510717039247e-6im
│   β₁ = 1.1218762152784267
│   β₂ = 1.0250892121035442
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.4643177038906052e-6 from total weight 1.4187342638925065: operator might not be hermitian?
│   α = 0.18563671478887794 - 1.4643177038906052e-6im
│   β₁ = 1.0250892121035442
│   β₂ = 0.9630877576274115
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -9.62525376541068e-7 from total weight 1.3877247027879926: operator might not be hermitian?
│   α = 0.170733761111782 - 9.62525376541068e-7im
│   β₁ = 0.9630877576274115
│   β₂ = 0.9844246058748866
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1086373975024988e-6 from total weight 1.4385164585733765: operator might not be hermitian?
│   α = 0.19199302226379647 - 1.1086373975024988e-6im
│   β₁ = 0.9844246058748866
│   β₂ = 1.0312014722329228
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
[ Info: VUMPS   7:	obj = +4.285328274012e-02	err = 3.6201107616e-01	time = 0.03 sec
┌ Warning: ignoring imaginary component 1.1099933890403035e-6 from total weight 1.9792036477866468: operator might not be hermitian?
│   α = -0.7379921644658364 + 1.1099933890403035e-6im
│   β₁ = 1.2986523358230453
│   β₂ = 1.2985055853784824
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
[ Info: VUMPS   8:	obj = -1.708578079666e-01	err = 3.8515380883e-01	time = 0.03 sec
[ Info: VUMPS   9:	obj = -1.934207351169e-01	err = 3.7546998968e-01	time = 0.03 sec
[ Info: VUMPS  10:	obj = -2.569456433937e-01	err = 3.6560214073e-01	time = 0.03 sec
[ Info: VUMPS  11:	obj = -1.937666328823e-01	err = 4.0884590460e-01	time = 0.05 sec
┌ Warning: ignoring imaginary component -1.0504283959426874e-6 from total weight 1.8402833625820303: operator might not be hermitian?
│   α = 0.5324141657601801 - 1.0504283959426874e-6im
│   β₁ = 1.2352702512807454
│   β₂ = 1.2559002416568827
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.088935510642694e-6 from total weight 1.8602705570479414: operator might not be hermitian?
│   α = 0.47903104132176993 - 1.088935510642694e-6im
│   β₁ = 1.2559002416568827
│   β₂ = 1.2860211467447324
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0003895707412502e-6 from total weight 1.6188876040636193: operator might not be hermitian?
│   α = 0.5767932246498841 - 1.0003895707412502e-6im
│   β₁ = 1.0414996723115992
│   β₂ = 1.0969890989260855
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -9.857157344875245e-7 from total weight 1.6680151278695234: operator might not be hermitian?
│   α = 0.5264739458256111 - 9.857157344875245e-7im
│   β₁ = 1.0969890989260855
│   β₂ = 1.1409270651556818
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
[ Info: VUMPS  12:	obj = -1.670459895402e-01	err = 3.7929964992e-01	time = 0.04 sec
[ Info: VUMPS  13:	obj = -2.026670357121e-01	err = 3.7248798904e-01	time = 0.05 sec
[ Info: VUMPS  14:	obj = +1.315709746715e-01	err = 3.6396753361e-01	time = 0.04 sec
[ Info: VUMPS  15:	obj = -1.737796153998e-01	err = 3.6226202233e-01	time = 0.03 sec
[ Info: VUMPS  16:	obj = -3.914396312413e-01	err = 2.5762054887e-01	time = 0.04 sec
[ Info: VUMPS  17:	obj = +1.208065704588e-01	err = 3.5537355326e-01	time = 0.03 sec
[ Info: VUMPS  18:	obj = -1.246491545662e-01	err = 3.7428714333e-01	time = 0.02 sec
[ Info: VUMPS  19:	obj = -1.617135365501e-01	err = 3.8121272586e-01	time = 0.03 sec
┌ Warning: ignoring imaginary component -1.041242800836964e-6 from total weight 1.8878543444675502: operator might not be hermitian?
│   α = 0.24524795174378963 - 1.041242800836964e-6im
│   β₁ = 1.3188603653095323
│   β₂ = 1.3283278228302693
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0352301777257628e-6 from total weight 1.8447819372771954: operator might not be hermitian?
│   α = 0.29785804984406666 - 1.0352301777257628e-6im
│   β₁ = 1.3044579500429994
│   β₂ = 1.2699962341738795
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1274336656243167e-6 from total weight 1.8643349432068144: operator might not be hermitian?
│   α = 0.26211325930649076 - 1.1274336656243167e-6im
│   β₁ = 1.3154591302446648
│   β₂ = 1.2948392550476535
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0171041124086566e-6 from total weight 1.8385424526104095: operator might not be hermitian?
│   α = 0.3313614251455128 - 1.0171041124086566e-6im
│   β₁ = 1.2948392550476535
│   β₂ = 1.2624695083695439
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0145686801365378e-6 from total weight 1.6609386863915652: operator might not be hermitian?
│   α = 0.3399359484890182 - 1.0145686801365378e-6im
│   β₁ = 1.1697513903269574
│   β₂ = 1.129089259405177
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0450914485968243e-6 from total weight 1.682026293949647: operator might not be hermitian?
│   α = 0.37087157882485045 - 1.0450914485968243e-6im
│   β₁ = 1.129089259405177
│   β₂ = 1.1903042341572698
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -9.77252247251828e-7 from total weight 1.6307158080421404: operator might not be hermitian?
│   α = 0.30514175950114303 - 9.77252247251828e-7im
│   β₁ = 1.170971147142692
│   β₂ = 1.093137285872845
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -9.588569277281175e-7 from total weight 1.5967583693052758: operator might not be hermitian?
│   α = 0.4226488627462033 - 9.588569277281175e-7im
│   β₁ = 1.093137285872845
│   β₂ = 1.0844612040084307
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -9.498961915273274e-7 from total weight 1.5966428260209404: operator might not be hermitian?
│   α = 0.31406158682358515 - 9.498961915273274e-7im
│   β₁ = 1.0844612040084307
│   β₂ = 1.1289718909546287
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0087007574714063e-6 from total weight 1.6253906087657857: operator might not be hermitian?
│   α = 0.3345859512478334 - 1.0087007574714063e-6im
│   β₁ = 1.1289718909546287
│   β₂ = 1.1204326582730915
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -9.978973924268386e-7 from total weight 1.6313955926302608: operator might not be hermitian?
│   α = 0.4474495516150536 - 9.978973924268386e-7im
│   β₁ = 1.0999799038430782
│   β₂ = 1.1186083718412372
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0079336151200269e-6 from total weight 1.67215352955067: operator might not be hermitian?
│   α = 0.312017547799734 - 1.0079336151200269e-6im
│   β₁ = 1.1253351445900006
│   β₂ = 1.1968138069906242
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -9.20250589822938e-7 from total weight 1.6321224428598387: operator might not be hermitian?
│   α = 0.3570833150407308 - 9.20250589822938e-7im
│   β₁ = 1.1659628844081862
│   β₂ = 1.0848252056382104
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -8.891854269034438e-7 from total weight 1.6083595644935376: operator might not be hermitian?
│   α = 0.4565536948967533 - 8.891854269034438e-7im
│   β₁ = 1.062492024298748
│   β₂ = 1.1178058465917602
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -9.623679356196568e-7 from total weight 1.5934469936613749: operator might not be hermitian?
│   α = 0.3469746904762515 - 9.623679356196568e-7im
│   β₁ = 1.1178058465917602
│   β₂ = 1.0812918084872996
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -9.1030968346964e-7 from total weight 1.581381778373773: operator might not be hermitian?
│   α = 0.3535414584487759 - 9.1030968346964e-7im
│   β₁ = 1.129565920198637
│   β₂ = 1.0487409585095295
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -8.763328321077096e-7 from total weight 1.5501726728818914: operator might not be hermitian?
│   α = 0.45956101925780013 - 8.763328321077096e-7im
│   β₁ = 1.0487409585095295
│   β₂ = 1.044979132457804
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -8.833465855047871e-7 from total weight 1.5559369862983596: operator might not be hermitian?
│   α = 0.5001231946818558 - 8.833465855047871e-7im
│   β₁ = 1.044979132457804
│   β₂ = 1.0386699707796463
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -8.554302556584847e-7 from total weight 1.5370431882570572: operator might not be hermitian?
│   α = 0.46749317820375846 - 8.554302556584847e-7im
│   β₁ = 1.0386699707796463
│   β₂ = 1.0320448549846297
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -9.88558524744193e-7 from total weight 1.5286768622038271: operator might not be hermitian?
│   α = 0.5720411114281577 - 9.88558524744193e-7im
│   β₁ = 0.8498777407927486
│   β₂ = 1.1346055444855774
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -8.424114983484471e-7 from total weight 1.4710741589707619: operator might not be hermitian?
│   α = 0.3078083239511698 - 8.424114983484471e-7im
│   β₁ = 1.0049864061890914
│   β₂ = 1.029230557393401
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -9.443609546022635e-7 from total weight 1.574485981114362: operator might not be hermitian?
│   α = 0.4466223362445244 - 9.443609546022635e-7im
│   β₁ = 1.029230557393401
│   β₂ = 1.1046352580014436
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0307124263374962e-6 from total weight 1.5620138044688787: operator might not be hermitian?
│   α = 0.38039651012084363 - 1.0307124263374962e-6im
│   β₁ = 1.1046352580014436
│   β₂ = 1.0368059448220026
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
[ Info: VUMPS  20:	obj = -3.585234798191e-01	err = 2.9715384480e-01	time = 0.03 sec
[ Info: VUMPS  21:	obj = -3.120924943693e-01	err = 3.4539465791e-01	time = 0.03 sec
┌ Warning: ignoring imaginary component -1.101222477394026e-6 from total weight 1.9131196119976634: operator might not be hermitian?
│   α = 0.7136486975992984 - 1.101222477394026e-6im
│   β₁ = 1.2660334860270381
│   β₂ = 1.2441428368486989
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0797377457440194e-6 from total weight 1.9473290871548898: operator might not be hermitian?
│   α = 0.8370940070708139 - 1.0797377457440194e-6im
│   β₁ = 1.2441428368486989
│   β₂ = 1.2423658070482022
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1093458114272425e-6 from total weight 1.9377361994168134: operator might not be hermitian?
│   α = 0.6001201014097328 - 1.1093458114272425e-6im
│   β₁ = 1.2423658070482022
│   β₂ = 1.3605898147092412
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.158222881363774e-6 from total weight 2.048460541734995: operator might not be hermitian?
│   α = 0.7464363437066065 - 1.158222881363774e-6im
│   β₁ = 1.3605898147092412
│   β₂ = 1.3370933893888164
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1498694335951992e-6 from total weight 2.0094403493801196: operator might not be hermitian?
│   α = 0.9629348644353627 - 1.1498694335951992e-6im
│   β₁ = 1.3370933893888164
│   β₂ = 1.1501253117044037
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1337398630566209e-6 from total weight 1.9506150403467022: operator might not be hermitian?
│   α = 0.8164721214339367 - 1.1337398630566209e-6im
│   β₁ = 1.1501253117044037
│   β₂ = 1.3473990047211317
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.101966695791004e-6 from total weight 1.929959391373378: operator might not be hermitian?
│   α = 0.6582940676182938 - 1.101966695791004e-6im
│   β₁ = 1.2482279939252525
│   β₂ = 1.3165557519789983
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0940951352973044e-6 from total weight 1.963168395071584: operator might not be hermitian?
│   α = 0.814866900127403 - 1.0940951352973044e-6im
│   β₁ = 1.3165557519789983
│   β₂ = 1.2069395322113334
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1378380324576776e-6 from total weight 1.9351476008024464: operator might not be hermitian?
│   α = 0.8471641801606176 - 1.1378380324576776e-6im
│   β₁ = 1.2069395322113334
│   β₂ = 1.253158431455674
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0654026989347087e-6 from total weight 1.880722856816104: operator might not be hermitian?
│   α = 0.7241790762545013 - 1.0654026989347087e-6im
│   β₁ = 1.253158431455674
│   β₂ = 1.2009484066087621
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1162494796936387e-6 from total weight 1.9063231994678866: operator might not be hermitian?
│   α = 0.8720535139516081 - 1.1162494796936387e-6im
│   β₁ = 1.2009484066087621
│   β₂ = 1.1963752481127155
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1902125000226582e-6 from total weight 1.8748641135008903: operator might not be hermitian?
│   α = 0.8055412360382196 - 1.1902125000226582e-6im
│   β₁ = 1.1963752481127155
│   β₂ = 1.1978752133829795
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0841214859360487e-6 from total weight 1.8981428651365628: operator might not be hermitian?
│   α = 0.8135546415885357 - 1.0841214859360487e-6im
│   β₁ = 1.1978752133829795
│   β₂ = 1.2272612414559885
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.137700440641265e-6 from total weight 1.9390242214140843: operator might not be hermitian?
│   α = 0.7186830060683032 - 1.137700440641265e-6im
│   β₁ = 1.2272612414559885
│   β₂ = 1.3180058851301593
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.157718810059112e-6 from total weight 1.9480246088532358: operator might not be hermitian?
│   α = 0.6778961778107517 - 1.157718810059112e-6im
│   β₁ = 1.3180058851301593
│   β₂ = 1.2641665774605424
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1765322366977249e-6 from total weight 2.00275903149103: operator might not be hermitian?
│   α = 0.8416337896623912 - 1.1765322366977249e-6im
│   β₁ = 1.2641665774605424
│   β₂ = 1.305595330394424
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.2027726534122318e-6 from total weight 1.9444398761248778: operator might not be hermitian?
│   α = 0.8138147830995662 - 1.2027726534122318e-6im
│   β₁ = 1.2660269736586829
│   β₂ = 1.2311489075819897
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.2050119907701384e-6 from total weight 1.9173639342920723: operator might not be hermitian?
│   α = 0.8412593948765817 - 1.2050119907701384e-6im
│   β₁ = 1.2311489075819897
│   β₂ = 1.2053379005134133
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1237154133049952e-6 from total weight 1.858065549589884: operator might not be hermitian?
│   α = 0.7127616925842981 - 1.1237154133049952e-6im
│   β₁ = 1.2053379005134133
│   β₂ = 1.2212857576103058
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0578611750605704e-6 from total weight 1.8430342285871433: operator might not be hermitian?
│   α = 0.5901892385132163 - 1.0578611750605704e-6im
│   β₁ = 1.2212857576103058
│   β₂ = 1.2477631701345007
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1640514067205732e-6 from total weight 1.8795569802053598: operator might not be hermitian?
│   α = 0.6449467131008518 - 1.1640514067205732e-6im
│   β₁ = 1.2477631701345007
│   β₂ = 1.248945655484515
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.2872909221240542e-6 from total weight 1.9420488977458514: operator might not be hermitian?
│   α = 0.964724061840767 - 1.2872909221240542e-6im
│   β₁ = 1.248945655484515
│   β₂ = 1.1318110069203366
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.251536440805695e-6 from total weight 1.8346592558570003: operator might not be hermitian?
│   α = 0.8659320524649449 - 1.251536440805695e-6im
│   β₁ = 0.8814763377039885
│   β₂ = 1.3561473856782706
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.2494786537451535e-6 from total weight 2.185013956688451: operator might not be hermitian?
│   α = 1.2293936691157223 - 1.2494786537451535e-6im
│   β₁ = 1.3561473856782706
│   β₂ = 1.193206380127903
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0433588381823095e-6 from total weight 1.814963244016936: operator might not be hermitian?
│   α = 0.7522114642910884 - 1.0433588381823095e-6im
│   β₁ = 1.193206380127903
│   β₂ = 1.1421593691523593
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.139217794356162e-6 from total weight 1.8612629247385752: operator might not be hermitian?
│   α = 0.821228848948878 - 1.139217794356162e-6im
│   β₁ = 1.1421593691523593
│   β₂ = 1.2187513397394416
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1087975631965744e-6 from total weight 1.896227193132863: operator might not be hermitian?
│   α = 0.8991678077171779 - 1.1087975631965744e-6im
│   β₁ = 1.2187513397394416
│   β₂ = 1.1409732658672187
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.174809510080116e-6 from total weight 1.9088015545409995: operator might not be hermitian?
│   α = 0.8914717660485393 - 1.174809510080116e-6im
│   β₁ = 1.1409732658672187
│   β₂ = 1.2437770988128656
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.2381195570107528e-6 from total weight 1.9520801898475855: operator might not be hermitian?
│   α = 0.8614585119019178 - 1.2381195570107528e-6im
│   β₁ = 1.2437770988128656
│   β₂ = 1.2335010451289605
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.2026763513330735e-6 from total weight 1.90755838099703: operator might not be hermitian?
│   α = 0.7016999126767994 - 1.2026763513330735e-6im
│   β₁ = 1.2335010451289605
│   β₂ = 1.2747044289268388
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.2071538935155834e-6 from total weight 1.9339188108541947: operator might not be hermitian?
│   α = 0.7949219841909692 - 1.2071538935155834e-6im
│   β₁ = 1.2747044289268388
│   β₂ = 1.2178955722468325
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1606930974371321e-6 from total weight 1.9370658765510747: operator might not be hermitian?
│   α = 0.954874437495209 - 1.1606930974371321e-6im
│   β₁ = 1.2178955722468325
│   β₂ = 1.164976134440961
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1268602565549582e-6 from total weight 1.9552759499022778: operator might not be hermitian?
│   α = 0.9492389130337321 - 1.1268602565549582e-6im
│   β₁ = 1.2467519043765316
│   β₂ = 1.1694696298668963
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.215592051885539e-6 from total weight 1.9488322122427535: operator might not be hermitian?
│   α = 1.0153114208108338 - 1.215592051885539e-6im
│   β₁ = 0.6714846852867788
│   β₂ = 1.5219060508684197
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.080734107393455e-6 from total weight 1.8464847663385096: operator might not be hermitian?
│   α = 1.1037814510079955 - 1.080734107393455e-6im
│   β₁ = 1.0312175766294405
│   β₂ = 1.061961774443977
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0426542516421555e-6 from total weight 1.7225192253666461: operator might not be hermitian?
│   α = 0.9480463615620647 - 1.0426542516421555e-6im
│   β₁ = 1.061961774443977
│   β₂ = 0.9698029530298176
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1328282575884252e-6 from total weight 1.7697990256796783: operator might not be hermitian?
│   α = 0.9658622062727309 - 1.1328282575884252e-6im
│   β₁ = 0.9698029530298176
│   β₂ = 1.1219541087246274
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0988690484076136e-6 from total weight 1.828016257969221: operator might not be hermitian?
│   α = 0.786906278336358 - 1.0988690484076136e-6im
│   β₁ = 1.1219541087246274
│   β₂ = 1.2098102853048218
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.142796129886825e-6 from total weight 1.8111805841289295: operator might not be hermitian?
│   α = 0.8638002477368484 - 1.142796129886825e-6im
│   β₁ = 1.2098102853048218
│   β₂ = 1.034689960280242
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.2517390058849887e-6 from total weight 1.830580564744666: operator might not be hermitian?
│   α = 1.036150575189292 - 1.2517390058849887e-6im
│   β₁ = 1.034689960280242
│   β₂ = 1.0985599099045549
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0559897735233637e-6 from total weight 1.8024078749168828: operator might not be hermitian?
│   α = 0.7896107548575517 - 1.0559897735233637e-6im
│   β₁ = 1.0985599099045549
│   β₂ = 1.1909471557231506
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0776481447613215e-6 from total weight 1.8447630176465906: operator might not be hermitian?
│   α = 0.9113608822751792 - 1.0776481447613215e-6im
│   β₁ = 1.1909471557231506
│   β₂ = 1.0743448263052873
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.238077302180042e-6 from total weight 1.8340076359601551: operator might not be hermitian?
│   α = 0.9175707442976321 - 1.238077302180042e-6im
│   β₁ = 1.0743448263052873
│   β₂ = 1.1693721102193158
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1686657275609275e-6 from total weight 1.7304630009406143: operator might not be hermitian?
│   α = 0.7354440288129548 - 1.1686657275609275e-6im
│   β₁ = 1.1693721102193158
│   β₂ = 1.042205903815376
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.2205737424525465e-6 from total weight 1.723630094975153: operator might not be hermitian?
│   α = 0.9056843148222427 - 1.2205737424525465e-6im
│   β₁ = 1.042205903815376
│   β₂ = 1.031718702088852
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.298260888585312e-6 from total weight 1.7345462402147205: operator might not be hermitian?
│   α = 0.9727698926351831 - 1.298260888585312e-6im
│   β₁ = 1.031718702088852
│   β₂ = 0.9989624193051336
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.2475635252917616e-6 from total weight 1.6978560223990682: operator might not be hermitian?
│   α = 0.9109309235783065 - 1.2475635252917616e-6im
│   β₁ = 0.9989624193051336
│   β₂ = 1.027129013357152
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0884506282324291e-6 from total weight 1.6753320770282918: operator might not be hermitian?
│   α = 0.7441182182543868 - 1.0884506282324291e-6im
│   β₁ = 1.027129013357152
│   β₂ = 1.094546314918024
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1834045275764538e-6 from total weight 1.7015965086066662: operator might not be hermitian?
│   α = 0.6866167659032616 - 1.1834045275764538e-6im
│   β₁ = 1.094546314918024
│   β₂ = 1.107229181055537
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0706567381644355e-6 from total weight 1.763274261244367: operator might not be hermitian?
│   α = 0.8172553309406643 - 1.0706567381644355e-6im
│   β₁ = 1.107229181055537
│   β₂ = 1.1023943872470834
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1632508603934277e-6 from total weight 1.7221080521650705: operator might not be hermitian?
│   α = 0.7673869852185097 - 1.1632508603934277e-6im
│   β₁ = 1.1023943872470834
│   β₂ = 1.0777290815477676
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1637710088698017e-6 from total weight 1.818889017221316: operator might not be hermitian?
│   α = 1.0293670834582107 - 1.1637710088698017e-6im
│   β₁ = 1.0777290815477676
│   β₂ = 1.0427179346524702
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.040114223001612e-6 from total weight 1.8324550004613034: operator might not be hermitian?
│   α = 1.0263900372755859 - 1.040114223001612e-6im
│   β₁ = 1.0427179346524702
│   β₂ = 1.103247084224591
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -9.449414066532713e-7 from total weight 1.695246006634505: operator might not be hermitian?
│   α = 0.824717824769762 - 9.449414066532713e-7im
│   β₁ = 1.0938490967740706
│   β₂ = 0.998595857192951
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0747455515061621e-6 from total weight 1.722904239881022: operator might not be hermitian?
│   α = 0.9353834131012809 - 1.0747455515061621e-6im
│   β₁ = 0.998595857192951
│   β₂ = 1.0470258852057677
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.223270898086798e-6 from total weight 1.708305544302349: operator might not be hermitian?
│   α = 0.7933798680425387 - 1.223270898086798e-6im
│   β₁ = 1.0470258852057677
│   β₂ = 1.092059070465756
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.104362944544246e-6 from total weight 1.7859815831529486: operator might not be hermitian?
│   α = 0.9186213194288729 - 1.104362944544246e-6im
│   β₁ = 1.092059070465756
│   β₂ = 1.0739050579378602
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0767294944188932e-6 from total weight 1.7517929358227784: operator might not be hermitian?
│   α = 0.9119793947642264 - 1.0767294944188932e-6im
│   β₁ = 1.0739050579378602
│   β₂ = 1.0410571550392351
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.2038780349871332e-6 from total weight 1.7380900252716378: operator might not be hermitian?
│   α = 1.0340693653604522 - 1.2038780349871332e-6im
│   β₁ = 1.0410571550392351
│   β₂ = 0.9315886879476152
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.288718792540576e-6 from total weight 1.7687358205143435: operator might not be hermitian?
│   α = 1.1554699297722864 - 1.288718792540576e-6im
│   β₁ = 0.9315886879476152
│   β₂ = 0.9620073599765004
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.2196683122703278e-6 from total weight 1.7308801621854317: operator might not be hermitian?
│   α = 1.0622505270312332 - 1.2196683122703278e-6im
│   β₁ = 0.7539467763585763
│   β₂ = 1.1397956887468725
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0853791424913356e-6 from total weight 1.6726926339879533: operator might not be hermitian?
│   α = 0.8492362545918903 - 1.0853791424913356e-6im
│   β₁ = 1.0284647484926124
│   β₂ = 1.0094348383086176
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0729252470451783e-6 from total weight 1.7382624570595517: operator might not be hermitian?
│   α = 0.9193105456968573 - 1.0729252470451783e-6im
│   β₁ = 1.0094348383086176
│   β₂ = 1.0758558441542996
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1420121823618734e-6 from total weight 1.7502520683145022: operator might not be hermitian?
│   α = 0.9100055220804504 - 1.1420121823618734e-6im
│   β₁ = 1.0758558441542996
│   β₂ = 1.03817457829597
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.127696722245941e-6 from total weight 1.7299611891318882: operator might not be hermitian?
│   α = 0.8541163993667705 - 1.127696722245941e-6im
│   β₁ = 1.03817457829597
│   β₂ = 1.0887811704902288
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1769815078735424e-6 from total weight 1.7594363401548583: operator might not be hermitian?
│   α = 0.9492140654268791 - 1.1769815078735424e-6im
│   β₁ = 1.0887811704902288
│   β₂ = 1.0045717773448748
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1666054236525447e-6 from total weight 1.6736956846097075: operator might not be hermitian?
│   α = 0.9196172145004238 - 1.1666054236525447e-6im
│   β₁ = 1.0045717773448748
│   β₂ = 0.9728293620344927
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
[ Info: VUMPS  22:	obj = -3.599926637146e-01	err = 3.1198099426e-01	time = 0.04 sec
[ Info: VUMPS  23:	obj = -2.006675577417e-01	err = 3.8157333729e-01	time = 0.03 sec
[ Info: VUMPS  24:	obj = -6.387493077780e-02	err = 3.9989829765e-01	time = 0.03 sec
[ Info: VUMPS  25:	obj = -8.834566879714e-02	err = 3.7005632685e-01	time = 0.07 sec
[ Info: VUMPS  26:	obj = -2.975734118485e-01	err = 3.4878293297e-01	time = 0.02 sec
[ Info: VUMPS  27:	obj = -9.464425124186e-02	err = 3.5775080203e-01	time = 0.02 sec
[ Info: VUMPS  28:	obj = -2.792871530928e-01	err = 3.5210943205e-01	time = 0.02 sec
[ Info: VUMPS  29:	obj = -2.977067791771e-01	err = 3.4012052565e-01	time = 0.02 sec
[ Info: VUMPS  30:	obj = -1.953218137614e-01	err = 4.1754928294e-01	time = 0.02 sec
[ Info: VUMPS  31:	obj = -3.156920626111e-01	err = 3.3106032255e-01	time = 0.02 sec
[ Info: VUMPS  32:	obj = -2.764432600413e-01	err = 3.3477318298e-01	time = 0.02 sec
[ Info: VUMPS  33:	obj = -1.793989121500e-01	err = 3.7982002408e-01	time = 0.02 sec
[ Info: VUMPS  34:	obj = -1.823377028659e-01	err = 4.0147712268e-01	time = 0.02 sec
[ Info: VUMPS  35:	obj = -7.808398436717e-02	err = 3.5721592731e-01	time = 0.03 sec
[ Info: VUMPS  36:	obj = +1.550446898034e-02	err = 3.7023278218e-01	time = 0.02 sec
[ Info: VUMPS  37:	obj = -6.567548208365e-02	err = 3.9599026194e-01	time = 0.02 sec
[ Info: VUMPS  38:	obj = -7.134198967732e-02	err = 3.9828655621e-01	time = 0.02 sec
[ Info: VUMPS  39:	obj = +1.245450991997e-01	err = 3.6040147377e-01	time = 0.02 sec
[ Info: VUMPS  40:	obj = -2.272805702370e-01	err = 3.7688254284e-01	time = 0.02 sec
[ Info: VUMPS  41:	obj = -2.711012098569e-01	err = 3.5608653488e-01	time = 0.02 sec
[ Info: VUMPS  42:	obj = -3.251122775684e-01	err = 3.1561344370e-01	time = 0.02 sec
[ Info: VUMPS  43:	obj = -3.314209855315e-01	err = 3.3621472798e-01	time = 0.02 sec
[ Info: VUMPS  44:	obj = -3.742005924137e-01	err = 2.9033969399e-01	time = 0.02 sec
[ Info: VUMPS  45:	obj = -3.065837675909e-01	err = 3.3609748111e-01	time = 0.03 sec
[ Info: VUMPS  46:	obj = -3.497171858968e-01	err = 2.9115179008e-01	time = 0.02 sec
[ Info: VUMPS  47:	obj = -9.910732501111e-02	err = 3.6751035382e-01	time = 0.03 sec
[ Info: VUMPS  48:	obj = -2.644925372437e-01	err = 3.5562378905e-01	time = 0.02 sec
[ Info: VUMPS  49:	obj = -2.513896040191e-01	err = 3.5870302716e-01	time = 0.02 sec
[ Info: VUMPS  50:	obj = -2.529548935577e-02	err = 3.8290361105e-01	time = 0.03 sec
[ Info: VUMPS  51:	obj = +1.088249666345e-02	err = 3.6750640959e-01	time = 0.05 sec
┌ Warning: ignoring imaginary component -1.0906260352510147e-6 from total weight 1.8497205242237273: operator might not be hermitian?
│   α = -0.5443504782080575 - 1.0906260352510147e-6im
│   β₁ = 1.2463426992339768
│   β₂ = 1.253705886830773
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.0323387949501966e-6 from total weight 1.8248219407307344: operator might not be hermitian?
│   α = -0.3488515008443419 - 1.0323387949501966e-6im
│   β₁ = 1.3036579841910374
│   β₂ = 1.2283133175150507
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -9.800749042193635e-7 from total weight 1.7392639240954022: operator might not be hermitian?
│   α = -0.15134665486679957 - 9.800749042193635e-7im
│   β₁ = 1.29843941540924
│   β₂ = 1.1472524884397144
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -9.218140629447397e-7 from total weight 1.6660225128067838: operator might not be hermitian?
│   α = -0.5406890303579995 - 9.218140629447397e-7im
│   β₁ = 1.1472524884397144
│   β₂ = 1.0803231523010095
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -9.389119079057651e-7 from total weight 1.6560892268929581: operator might not be hermitian?
│   α = -0.36913134770841677 - 9.389119079057651e-7im
│   β₁ = 1.0803231523010095
│   β₂ = 1.1996980712543517
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
[ Info: VUMPS  52:	obj = -1.835786047074e-01	err = 3.7058718254e-01	time = 0.02 sec
[ Info: VUMPS  53:	obj = -2.872446357917e-01	err = 3.4652966395e-01	time = 0.02 sec
[ Info: VUMPS  54:	obj = -2.501331960767e-01	err = 3.6074360899e-01	time = 0.02 sec
[ Info: VUMPS  55:	obj = -3.045540307815e-01	err = 3.6027238193e-01	time = 0.02 sec
[ Info: VUMPS  56:	obj = -3.786209442746e-01	err = 2.7551150472e-01	time = 0.02 sec
[ Info: VUMPS  57:	obj = -1.885356465151e-01	err = 3.5617449268e-01	time = 0.02 sec
[ Info: VUMPS  58:	obj = -8.242178567440e-02	err = 4.2485729708e-01	time = 0.02 sec
[ Info: VUMPS  59:	obj = +6.854256210998e-02	err = 3.8280553717e-01	time = 0.02 sec
[ Info: VUMPS  60:	obj = -1.334345522323e-01	err = 3.7199065525e-01	time = 0.02 sec
[ Info: VUMPS  61:	obj = -1.468728192806e-01	err = 3.7782674888e-01	time = 0.02 sec
[ Info: VUMPS  62:	obj = -3.248275691559e-01	err = 3.1910607395e-01	time = 0.02 sec
[ Info: VUMPS  63:	obj = -9.756232234052e-02	err = 3.6802847968e-01	time = 0.02 sec
[ Info: VUMPS  64:	obj = -3.541788730283e-02	err = 3.6654318560e-01	time = 0.02 sec
[ Info: VUMPS  65:	obj = -2.884534040748e-02	err = 3.8836376222e-01	time = 0.03 sec
[ Info: VUMPS  66:	obj = -2.543638757662e-02	err = 3.7289138253e-01	time = 0.03 sec
[ Info: VUMPS  67:	obj = +3.761430149410e-02	err = 3.9523804597e-01	time = 0.02 sec
[ Info: VUMPS  68:	obj = -8.502353960949e-02	err = 4.2605791567e-01	time = 0.02 sec
[ Info: VUMPS  69:	obj = +6.764248834569e-02	err = 3.3807684183e-01	time = 0.02 sec
[ Info: VUMPS  70:	obj = +3.620382339073e-04	err = 4.0191660644e-01	time = 0.02 sec
[ Info: VUMPS  71:	obj = -1.419203348175e-01	err = 3.9688127648e-01	time = 0.02 sec
[ Info: VUMPS  72:	obj = +2.227275498360e-02	err = 3.7840095594e-01	time = 0.02 sec
[ Info: VUMPS  73:	obj = -1.463126078271e-01	err = 4.1112786067e-01	time = 0.02 sec
[ Info: VUMPS  74:	obj = -2.645357516128e-02	err = 3.7830090977e-01	time = 0.03 sec
[ Info: VUMPS  75:	obj = -9.873560177855e-02	err = 4.0908071214e-01	time = 0.03 sec
[ Info: VUMPS  76:	obj = -1.080109565552e-01	err = 3.8462901759e-01	time = 0.02 sec
[ Info: VUMPS  77:	obj = -2.177950900439e-01	err = 3.4865826566e-01	time = 0.02 sec
[ Info: VUMPS  78:	obj = -2.145295046946e-01	err = 3.8179742518e-01	time = 0.04 sec
[ Info: VUMPS  79:	obj = -3.033777396122e-01	err = 3.3881849617e-01	time = 0.02 sec
[ Info: VUMPS  80:	obj = -1.018654078294e-01	err = 3.5728148275e-01	time = 0.03 sec
[ Info: VUMPS  81:	obj = -2.776735220330e-01	err = 3.3584583232e-01	time = 0.02 sec
[ Info: VUMPS  82:	obj = -1.158347284088e-01	err = 3.6885043774e-01	time = 0.02 sec
[ Info: VUMPS  83:	obj = -2.625401602298e-01	err = 3.7176511141e-01	time = 0.02 sec
[ Info: VUMPS  84:	obj = -1.074949158082e-01	err = 4.0177799241e-01	time = 0.03 sec
[ Info: VUMPS  85:	obj = -1.127766228428e-01	err = 4.0149881595e-01	time = 0.02 sec
[ Info: VUMPS  86:	obj = -2.808293730657e-01	err = 3.6168038744e-01	time = 0.02 sec
[ Info: VUMPS  87:	obj = -3.046853439863e-01	err = 3.3655658880e-01	time = 0.02 sec
[ Info: VUMPS  88:	obj = -3.318734618762e-01	err = 3.1566007228e-01	time = 0.02 sec
[ Info: VUMPS  89:	obj = -3.535869257430e-01	err = 3.0873844043e-01	time = 0.02 sec
[ Info: VUMPS  90:	obj = -3.747958764617e-01	err = 2.8288563034e-01	time = 0.02 sec
[ Info: VUMPS  91:	obj = +5.635703695733e-02	err = 3.7877695869e-01	time = 0.02 sec
[ Info: VUMPS  92:	obj = -2.546798027174e-01	err = 3.7343940301e-01	time = 0.02 sec
[ Info: VUMPS  93:	obj = -3.208056853929e-01	err = 3.3146175420e-01	time = 0.02 sec
[ Info: VUMPS  94:	obj = -1.076573438228e-01	err = 3.8291173087e-01	time = 0.02 sec
[ Info: VUMPS  95:	obj = +5.846672351870e-03	err = 3.8592643456e-01	time = 0.02 sec
[ Info: VUMPS  96:	obj = +3.595683066276e-02	err = 3.7085826416e-01	time = 0.02 sec
[ Info: VUMPS  97:	obj = -3.938910516171e-02	err = 3.5611514050e-01	time = 0.02 sec
[ Info: VUMPS  98:	obj = -2.612602243834e-01	err = 3.5599606870e-01	time = 0.03 sec
[ Info: VUMPS  99:	obj = -3.593251292915e-01	err = 3.0075205445e-01	time = 0.03 sec
[ Info: VUMPS 100:	obj = -2.840490132623e-01	err = 3.5596476789e-01	time = 0.04 sec
[ Info: VUMPS 101:	obj = -3.149203096197e-01	err = 3.4612410353e-01	time = 0.03 sec
[ Info: VUMPS 102:	obj = -1.771004727151e-01	err = 3.7690482781e-01	time = 0.03 sec
[ Info: VUMPS 103:	obj = -1.261057797685e-01	err = 3.2951815491e-01	time = 0.03 sec
[ Info: VUMPS 104:	obj = -1.686543661324e-01	err = 3.7094564016e-01	time = 0.04 sec
[ Info: VUMPS 105:	obj = -3.423027109471e-01	err = 3.2742213114e-01	time = 0.02 sec
[ Info: VUMPS 106:	obj = -1.878214833070e-01	err = 3.9431530329e-01	time = 0.02 sec
[ Info: VUMPS 107:	obj = -3.351840994830e-01	err = 3.2292784491e-01	time = 0.02 sec
[ Info: VUMPS 108:	obj = +3.684045487139e-02	err = 4.0052347039e-01	time = 0.02 sec
[ Info: VUMPS 109:	obj = -1.413589558971e-01	err = 4.0048426658e-01	time = 0.02 sec
[ Info: VUMPS 110:	obj = -1.296478941821e-01	err = 3.8353753071e-01	time = 0.02 sec
[ Info: VUMPS 111:	obj = -2.903612348667e-01	err = 3.4702087325e-01	time = 0.02 sec
[ Info: VUMPS 112:	obj = -2.889212283943e-01	err = 3.4180326230e-01	time = 0.02 sec
[ Info: VUMPS 113:	obj = -3.522496143264e-01	err = 3.1360801538e-01	time = 0.02 sec
[ Info: VUMPS 114:	obj = -2.150329065915e-01	err = 3.6902468730e-01	time = 0.03 sec
[ Info: VUMPS 115:	obj = -2.703393524794e-01	err = 3.5095708774e-01	time = 0.02 sec
[ Info: VUMPS 116:	obj = -3.750896311434e-01	err = 2.7607472783e-01	time = 0.02 sec
[ Info: VUMPS 117:	obj = -4.374578698028e-01	err = 8.7068984647e-02	time = 0.02 sec
[ Info: VUMPS 118:	obj = -2.068327025250e-01	err = 3.9598606710e-01	time = 0.03 sec
[ Info: VUMPS 119:	obj = +2.035115814900e-02	err = 3.8850017121e-01	time = 0.03 sec
[ Info: VUMPS 120:	obj = -9.325059606146e-02	err = 3.9888549509e-01	time = 0.02 sec
[ Info: VUMPS 121:	obj = -3.316950708480e-01	err = 3.2060866594e-01	time = 0.02 sec
[ Info: VUMPS 122:	obj = -3.655769344173e-01	err = 3.0447070338e-01	time = 0.02 sec
[ Info: VUMPS 123:	obj = -2.884718529955e-01	err = 3.4662993339e-01	time = 0.03 sec
[ Info: VUMPS 124:	obj = +1.006818501243e-01	err = 2.5844667684e-01	time = 0.03 sec
[ Info: VUMPS 125:	obj = -5.352193069862e-02	err = 3.9024450676e-01	time = 0.02 sec
[ Info: VUMPS 126:	obj = -2.373506245892e-01	err = 3.6512643964e-01	time = 0.02 sec
[ Info: VUMPS 127:	obj = -2.159468619452e-01	err = 3.7636947224e-01	time = 0.02 sec
[ Info: VUMPS 128:	obj = -2.882613511865e-01	err = 3.5104424909e-01	time = 0.06 sec
[ Info: VUMPS 129:	obj = -5.404755897444e-02	err = 4.0673697459e-01	time = 0.02 sec
[ Info: VUMPS 130:	obj = -1.110570289732e-02	err = 3.7300276680e-01	time = 0.02 sec
[ Info: VUMPS 131:	obj = -3.055714454115e-01	err = 3.2593658317e-01	time = 0.02 sec
[ Info: VUMPS 132:	obj = -4.138799853856e-01	err = 2.1596634660e-01	time = 0.03 sec
[ Info: VUMPS 133:	obj = -4.103726278341e-01	err = 2.2971553475e-01	time = 0.03 sec
[ Info: VUMPS 134:	obj = -2.062283378317e-01	err = 3.9155621302e-01	time = 0.10 sec
[ Info: VUMPS 135:	obj = -1.408095517660e-01	err = 3.7133013921e-01	time = 0.03 sec
[ Info: VUMPS 136:	obj = -2.733008995045e-01	err = 3.4869660555e-01	time = 0.03 sec
[ Info: VUMPS 137:	obj = -2.413146142661e-01	err = 3.6032029037e-01	time = 0.04 sec
[ Info: VUMPS 138:	obj = -2.345521401692e-01	err = 3.9156410210e-01	time = 0.04 sec
[ Info: VUMPS 139:	obj = -1.812659212104e-01	err = 3.7512375931e-01	time = 0.02 sec
[ Info: VUMPS 140:	obj = +5.504445469520e-02	err = 4.0833291366e-01	time = 0.02 sec
[ Info: VUMPS 141:	obj = +7.819335042351e-02	err = 3.6246986033e-01	time = 0.02 sec
[ Info: VUMPS 142:	obj = -8.837833565961e-02	err = 4.0255886671e-01	time = 0.02 sec
[ Info: VUMPS 143:	obj = -1.439796807770e-01	err = 3.7798874866e-01	time = 0.02 sec
[ Info: VUMPS 144:	obj = -9.845791091371e-02	err = 3.7883789962e-01	time = 0.03 sec
[ Info: VUMPS 145:	obj = -2.002438639637e-01	err = 3.6794432629e-01	time = 0.03 sec
[ Info: VUMPS 146:	obj = -2.925115290248e-01	err = 3.4496990483e-01	time = 0.02 sec
[ Info: VUMPS 147:	obj = -2.567134823031e-01	err = 3.7682994795e-01	time = 0.03 sec
[ Info: VUMPS 148:	obj = -5.450623211295e-02	err = 4.0354166337e-01	time = 0.03 sec
[ Info: VUMPS 149:	obj = -2.289380848022e-01	err = 3.8659563460e-01	time = 0.03 sec
[ Info: VUMPS 150:	obj = -5.474894900934e-02	err = 3.7384998711e-01	time = 0.03 sec
[ Info: VUMPS 151:	obj = -9.637261902165e-02	err = 3.9586432122e-01	time = 0.03 sec
[ Info: VUMPS 152:	obj = -2.454647681151e-01	err = 3.8403024140e-01	time = 0.03 sec
[ Info: VUMPS 153:	obj = -2.533922365600e-01	err = 3.7678126243e-01	time = 0.05 sec
[ Info: VUMPS 154:	obj = -2.827263113773e-01	err = 3.5212474060e-01	time = 0.02 sec
[ Info: VUMPS 155:	obj = -1.560655750127e-01	err = 3.9202883901e-01	time = 0.03 sec
[ Info: VUMPS 156:	obj = -3.816792402053e-02	err = 3.8546044810e-01	time = 0.02 sec
[ Info: VUMPS 157:	obj = -1.990621578324e-01	err = 3.6387099519e-01	time = 0.02 sec
[ Info: VUMPS 158:	obj = -2.456905842106e-01	err = 3.6718246410e-01	time = 0.02 sec
[ Info: VUMPS 159:	obj = -1.569727394566e-01	err = 3.8622969012e-01	time = 0.02 sec
[ Info: VUMPS 160:	obj = -3.806952311736e-01	err = 2.7009992090e-01	time = 0.03 sec
[ Info: VUMPS 161:	obj = -4.023014127533e-01	err = 2.4044971021e-01	time = 0.03 sec
[ Info: VUMPS 162:	obj = -8.174864155685e-02	err = 3.9868997751e-01	time = 0.02 sec
[ Info: VUMPS 163:	obj = -1.083133089400e-01	err = 3.9577054120e-01	time = 0.02 sec
[ Info: VUMPS 164:	obj = -4.117310247478e-02	err = 3.8288605079e-01	time = 0.02 sec
[ Info: VUMPS 165:	obj = +5.658238277328e-02	err = 3.4545767577e-01	time = 0.02 sec
[ Info: VUMPS 166:	obj = -1.562684251828e-01	err = 3.7668846749e-01	time = 0.02 sec
[ Info: VUMPS 167:	obj = -3.579967475307e-01	err = 2.9316630928e-01	time = 0.02 sec
[ Info: VUMPS 168:	obj = -3.566777325589e-01	err = 3.1340244360e-01	time = 0.03 sec
[ Info: VUMPS 169:	obj = -3.060527240289e-01	err = 3.5098667027e-01	time = 0.02 sec
[ Info: VUMPS 170:	obj = -3.374604547737e-01	err = 3.1443790743e-01	time = 0.03 sec
[ Info: VUMPS 171:	obj = -4.177203257232e-01	err = 1.9939123710e-01	time = 0.03 sec
[ Info: VUMPS 172:	obj = +5.125705660122e-03	err = 3.8405215483e-01	time = 0.02 sec
[ Info: VUMPS 173:	obj = -5.400499230366e-02	err = 4.1385947701e-01	time = 0.02 sec
[ Info: VUMPS 174:	obj = -1.621295573254e-01	err = 3.6125259721e-01	time = 0.02 sec
[ Info: VUMPS 175:	obj = -3.121116840774e-01	err = 3.3512185832e-01	time = 0.02 sec
[ Info: VUMPS 176:	obj = -3.849842725262e-01	err = 2.6782930861e-01	time = 0.02 sec
[ Info: VUMPS 177:	obj = +3.110563155614e-02	err = 3.7713634288e-01	time = 0.03 sec
[ Info: VUMPS 178:	obj = +3.669039323676e-02	err = 3.8251904836e-01	time = 0.04 sec
[ Info: VUMPS 179:	obj = -3.840454731001e-02	err = 3.7474626597e-01	time = 0.02 sec
[ Info: VUMPS 180:	obj = -2.524487813554e-01	err = 3.7049523698e-01	time = 0.02 sec
[ Info: VUMPS 181:	obj = -3.788502123588e-01	err = 2.8387660361e-01	time = 0.02 sec
[ Info: VUMPS 182:	obj = -4.082044238062e-01	err = 2.2067247971e-01	time = 0.03 sec
[ Info: VUMPS 183:	obj = -4.055634848980e-01	err = 2.3176961720e-01	time = 0.04 sec
[ Info: VUMPS 184:	obj = -1.976537578552e-01	err = 3.4371096572e-01	time = 0.04 sec
[ Info: VUMPS 185:	obj = -1.764473240992e-01	err = 3.6115673867e-01	time = 0.02 sec
[ Info: VUMPS 186:	obj = -3.885951111015e-01	err = 2.5363772934e-01	time = 0.02 sec
[ Info: VUMPS 187:	obj = -4.005815988144e-01	err = 2.3602724468e-01	time = 0.02 sec
[ Info: VUMPS 188:	obj = -1.118521194665e-01	err = 3.8155191888e-01	time = 0.03 sec
[ Info: VUMPS 189:	obj = -2.674536940230e-01	err = 3.2599437468e-01	time = 0.02 sec
[ Info: VUMPS 190:	obj = -2.881123357546e-01	err = 3.4671988495e-01	time = 0.02 sec
[ Info: VUMPS 191:	obj = -1.784544486614e-01	err = 3.7274285855e-01	time = 0.02 sec
[ Info: VUMPS 192:	obj = -1.619856307685e-01	err = 3.7835778084e-01	time = 0.02 sec
[ Info: VUMPS 193:	obj = -1.663182199638e-01	err = 3.6917729523e-01	time = 0.02 sec
[ Info: VUMPS 194:	obj = -1.794534817670e-02	err = 3.8707259520e-01	time = 0.02 sec
[ Info: VUMPS 195:	obj = +1.703845824273e-02	err = 3.7640930619e-01	time = 0.02 sec
[ Info: VUMPS 196:	obj = -1.947209060524e-01	err = 3.5645896038e-01	time = 0.02 sec
[ Info: VUMPS 197:	obj = -3.340877624423e-01	err = 3.2661456636e-01	time = 0.02 sec
[ Info: VUMPS 198:	obj = -3.589457517989e-01	err = 3.1047756734e-01	time = 0.02 sec
[ Info: VUMPS 199:	obj = -2.290122658012e-01	err = 3.6664407689e-01	time = 0.02 sec
┌ Warning: VUMPS cancel 200:	obj = -3.484120636346e-01	err = 3.0496744526e-01	time = 5.17 sec
└ @ MPSKit ~/Projects/Julia/MPSKit.jl/src/algorithms/groundstate/vumps.jl:67

````

As you can see, VUMPS struggles to converge.
On it's own, that is already quite curious.
Maybe we can do better using another algorithm, such as gradient descent.

````julia
groundstate, cache, delta = find_groundstate(state, H, GradientGrassmann(; maxiter=20));
````

````
[ Info: CG: initializing with f = 0.249997410383, ‖∇f‖ = 3.3669e-03
[ Info: CG: iter    1: f = -0.044704853193, ‖∇f‖ = 5.9152e-01, α = 7.29e+03, β = 0.00e+00, nfg = 5
[ Info: CG: iter    2: f = -0.058658885672, ‖∇f‖ = 6.1600e-01, α = 1.63e+00, β = 6.59e+03, nfg = 8
[ Info: CG: iter    3: f = -0.060627038309, ‖∇f‖ = 6.4371e-01, α = 1.17e-03, β = 1.95e+01, nfg = 12
[ Info: CG: iter    4: f = -0.181007718316, ‖∇f‖ = 6.8201e-01, α = 9.81e-01, β = 1.78e-03, nfg = 6
[ Info: CG: iter    5: f = -0.311679137866, ‖∇f‖ = 4.9353e-01, α = 7.38e-01, β = 3.68e-01, nfg = 3
[ Info: CG: iter    6: f = -0.369436221250, ‖∇f‖ = 3.9007e-01, α = 7.04e-01, β = 2.28e-01, nfg = 2
[ Info: CG: iter    7: f = -0.408163517301, ‖∇f‖ = 2.1812e-01, α = 5.24e-01, β = 2.14e-01, nfg = 2
[ Info: CG: iter    8: f = -0.417643389806, ‖∇f‖ = 1.6464e-01, α = 3.03e-01, β = 3.14e-01, nfg = 2
[ Info: CG: iter    9: f = -0.421957811162, ‖∇f‖ = 1.4205e-01, α = 1.89e-01, β = 4.76e-01, nfg = 2
[ Info: CG: iter   10: f = -0.425880699478, ‖∇f‖ = 1.3122e-01, α = 2.18e-01, β = 4.93e-01, nfg = 2
[ Info: CG: iter   11: f = -0.429137411499, ‖∇f‖ = 1.2451e-01, α = 2.10e-01, β = 4.95e-01, nfg = 2
[ Info: CG: iter   12: f = -0.431913122251, ‖∇f‖ = 1.1664e-01, α = 1.78e-01, β = 6.26e-01, nfg = 2
[ Info: CG: iter   13: f = -0.435009198881, ‖∇f‖ = 1.0914e-01, α = 2.57e-01, β = 5.20e-01, nfg = 2
[ Info: CG: iter   14: f = -0.437794275447, ‖∇f‖ = 8.3054e-02, α = 3.18e-01, β = 3.59e-01, nfg = 2
[ Info: CG: iter   15: f = -0.439321637163, ‖∇f‖ = 6.5852e-02, α = 2.84e-01, β = 3.80e-01, nfg = 2
[ Info: CG: iter   16: f = -0.440230609086, ‖∇f‖ = 5.3475e-02, α = 2.54e-01, β = 4.32e-01, nfg = 2
[ Info: CG: iter   17: f = -0.440915387974, ‖∇f‖ = 4.3318e-02, α = 3.21e-01, β = 3.60e-01, nfg = 2
[ Info: CG: iter   18: f = -0.441349404831, ‖∇f‖ = 3.5587e-02, α = 2.76e-01, β = 3.95e-01, nfg = 2
[ Info: CG: iter   19: f = -0.441562044343, ‖∇f‖ = 3.2287e-02, α = 1.49e-01, β = 6.79e-01, nfg = 2
┌ Warning: CG: not converged to requested tol: f = -0.441808873450, ‖∇f‖ = 2.9902e-02
└ @ OptimKit ~/Projects/Julia/OptimKit.jl/src/cg.jl:103

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
  <clipPath id="clip670">
    <rect x="0" y="0" width="2400" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip670)" d="M0 1600 L2400 1600 L2400 0 L0 0  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip671">
    <rect x="480" y="0" width="1681" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip670)" d="M279.704 1423.18 L2352.76 1423.18 L2352.76 47.2441 L279.704 47.2441  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip672">
    <rect x="279" y="47" width="2074" height="1377"/>
  </clipPath>
</defs>
<polyline clip-path="url(#clip672)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="338.375,1423.18 338.375,47.2441 "/>
<polyline clip-path="url(#clip672)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="664.327,1423.18 664.327,47.2441 "/>
<polyline clip-path="url(#clip672)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="990.278,1423.18 990.278,47.2441 "/>
<polyline clip-path="url(#clip672)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1316.23,1423.18 1316.23,47.2441 "/>
<polyline clip-path="url(#clip672)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1642.18,1423.18 1642.18,47.2441 "/>
<polyline clip-path="url(#clip672)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1968.13,1423.18 1968.13,47.2441 "/>
<polyline clip-path="url(#clip672)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="2294.08,1423.18 2294.08,47.2441 "/>
<polyline clip-path="url(#clip670)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="279.704,1423.18 2352.76,1423.18 "/>
<polyline clip-path="url(#clip670)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="338.375,1423.18 338.375,1404.28 "/>
<polyline clip-path="url(#clip670)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="664.327,1423.18 664.327,1404.28 "/>
<polyline clip-path="url(#clip670)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="990.278,1423.18 990.278,1404.28 "/>
<polyline clip-path="url(#clip670)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1316.23,1423.18 1316.23,1404.28 "/>
<polyline clip-path="url(#clip670)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1642.18,1423.18 1642.18,1404.28 "/>
<polyline clip-path="url(#clip670)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1968.13,1423.18 1968.13,1404.28 "/>
<polyline clip-path="url(#clip670)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="2294.08,1423.18 2294.08,1404.28 "/>
<path clip-path="url(#clip670)" d="M292.137 1454.1 Q288.526 1454.1 286.697 1457.66 Q284.892 1461.2 284.892 1468.33 Q284.892 1475.44 286.697 1479.01 Q288.526 1482.55 292.137 1482.55 Q295.771 1482.55 297.577 1479.01 Q299.405 1475.44 299.405 1468.33 Q299.405 1461.2 297.577 1457.66 Q295.771 1454.1 292.137 1454.1 M292.137 1450.39 Q297.947 1450.39 301.003 1455 Q304.081 1459.58 304.081 1468.33 Q304.081 1477.06 301.003 1481.67 Q297.947 1486.25 292.137 1486.25 Q286.327 1486.25 283.248 1481.67 Q280.193 1477.06 280.193 1468.33 Q280.193 1459.58 283.248 1455 Q286.327 1450.39 292.137 1450.39 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M319.266 1451.02 L323.202 1451.02 L311.165 1489.98 L307.23 1489.98 L319.266 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M335.239 1451.02 L339.174 1451.02 L327.137 1489.98 L323.202 1489.98 L335.239 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M345.053 1481.64 L352.692 1481.64 L352.692 1455.28 L344.382 1456.95 L344.382 1452.69 L352.646 1451.02 L357.322 1451.02 L357.322 1481.64 L364.961 1481.64 L364.961 1485.58 L345.053 1485.58 L345.053 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M371.049 1459.65 L395.886 1459.65 L395.886 1463.91 L392.623 1463.91 L392.623 1479.84 Q392.623 1481.51 393.178 1482.25 Q393.757 1482.96 395.03 1482.96 Q395.377 1482.96 395.886 1482.92 Q396.396 1482.85 396.558 1482.83 L396.558 1485.9 Q395.748 1486.2 394.891 1486.34 Q394.035 1486.48 393.178 1486.48 Q390.4 1486.48 389.336 1484.98 Q388.271 1483.45 388.271 1479.38 L388.271 1463.91 L378.711 1463.91 L378.711 1485.58 L374.359 1485.58 L374.359 1463.91 L371.049 1463.91 L371.049 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M607.857 1481.64 L615.496 1481.64 L615.496 1455.28 L607.186 1456.95 L607.186 1452.69 L615.45 1451.02 L620.126 1451.02 L620.126 1481.64 L627.764 1481.64 L627.764 1485.58 L607.857 1485.58 L607.857 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M644.176 1451.02 L648.112 1451.02 L636.075 1489.98 L632.139 1489.98 L644.176 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M660.149 1451.02 L664.084 1451.02 L652.047 1489.98 L648.112 1489.98 L660.149 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M683.32 1466.95 Q686.676 1467.66 688.551 1469.93 Q690.449 1472.2 690.449 1475.53 Q690.449 1480.65 686.931 1483.45 Q683.412 1486.25 676.931 1486.25 Q674.755 1486.25 672.44 1485.81 Q670.148 1485.39 667.695 1484.54 L667.695 1480.02 Q669.639 1481.16 671.954 1481.74 Q674.269 1482.32 676.792 1482.32 Q681.19 1482.32 683.482 1480.58 Q685.797 1478.84 685.797 1475.53 Q685.797 1472.48 683.644 1470.77 Q681.514 1469.03 677.695 1469.03 L673.667 1469.03 L673.667 1465.19 L677.88 1465.19 Q681.329 1465.19 683.158 1463.82 Q684.986 1462.43 684.986 1459.84 Q684.986 1457.18 683.088 1455.77 Q681.213 1454.33 677.695 1454.33 Q675.773 1454.33 673.574 1454.75 Q671.375 1455.16 668.736 1456.04 L668.736 1451.88 Q671.398 1451.14 673.713 1450.77 Q676.051 1450.39 678.111 1450.39 Q683.435 1450.39 686.537 1452.83 Q689.639 1455.23 689.639 1459.35 Q689.639 1462.22 687.996 1464.21 Q686.352 1466.18 683.32 1466.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M695.959 1459.65 L720.796 1459.65 L720.796 1463.91 L717.532 1463.91 L717.532 1479.84 Q717.532 1481.51 718.088 1482.25 Q718.667 1482.96 719.94 1482.96 Q720.287 1482.96 720.796 1482.92 Q721.306 1482.85 721.468 1482.83 L721.468 1485.9 Q720.657 1486.2 719.801 1486.34 Q718.945 1486.48 718.088 1486.48 Q715.31 1486.48 714.245 1484.98 Q713.181 1483.45 713.181 1479.38 L713.181 1463.91 L703.621 1463.91 L703.621 1485.58 L699.269 1485.58 L699.269 1463.91 L695.959 1463.91 L695.959 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M937.894 1481.64 L954.214 1481.64 L954.214 1485.58 L932.269 1485.58 L932.269 1481.64 Q934.931 1478.89 939.515 1474.26 Q944.121 1469.61 945.302 1468.27 Q947.547 1465.74 948.427 1464.01 Q949.329 1462.25 949.329 1460.56 Q949.329 1457.8 947.385 1456.07 Q945.464 1454.33 942.362 1454.33 Q940.163 1454.33 937.709 1455.09 Q935.279 1455.86 932.501 1457.41 L932.501 1452.69 Q935.325 1451.55 937.779 1450.97 Q940.232 1450.39 942.269 1450.39 Q947.64 1450.39 950.834 1453.08 Q954.029 1455.77 954.029 1460.26 Q954.029 1462.39 953.218 1464.31 Q952.431 1466.2 950.325 1468.8 Q949.746 1469.47 946.644 1472.69 Q943.542 1475.88 937.894 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M970.996 1451.02 L974.931 1451.02 L962.894 1489.98 L958.959 1489.98 L970.996 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M986.968 1451.02 L990.903 1451.02 L978.866 1489.98 L974.931 1489.98 L986.968 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M1010.14 1466.95 Q1013.5 1467.66 1015.37 1469.93 Q1017.27 1472.2 1017.27 1475.53 Q1017.27 1480.65 1013.75 1483.45 Q1010.23 1486.25 1003.75 1486.25 Q1001.57 1486.25 999.26 1485.81 Q996.968 1485.39 994.514 1484.54 L994.514 1480.02 Q996.459 1481.16 998.774 1481.74 Q1001.09 1482.32 1003.61 1482.32 Q1008.01 1482.32 1010.3 1480.58 Q1012.62 1478.84 1012.62 1475.53 Q1012.62 1472.48 1010.46 1470.77 Q1008.33 1469.03 1004.51 1469.03 L1000.49 1469.03 L1000.49 1465.19 L1004.7 1465.19 Q1008.15 1465.19 1009.98 1463.82 Q1011.81 1462.43 1011.81 1459.84 Q1011.81 1457.18 1009.91 1455.77 Q1008.03 1454.33 1004.51 1454.33 Q1002.59 1454.33 1000.39 1454.75 Q998.195 1455.16 995.556 1456.04 L995.556 1451.88 Q998.218 1451.14 1000.53 1450.77 Q1002.87 1450.39 1004.93 1450.39 Q1010.26 1450.39 1013.36 1452.83 Q1016.46 1455.23 1016.46 1459.35 Q1016.46 1462.22 1014.82 1464.21 Q1013.17 1466.18 1010.14 1466.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M1022.78 1459.65 L1047.62 1459.65 L1047.62 1463.91 L1044.35 1463.91 L1044.35 1479.84 Q1044.35 1481.51 1044.91 1482.25 Q1045.49 1482.96 1046.76 1482.96 Q1047.11 1482.96 1047.62 1482.92 Q1048.13 1482.85 1048.29 1482.83 L1048.29 1485.9 Q1047.48 1486.2 1046.62 1486.34 Q1045.76 1486.48 1044.91 1486.48 Q1042.13 1486.48 1041.07 1484.98 Q1040 1483.45 1040 1479.38 L1040 1463.91 L1030.44 1463.91 L1030.44 1485.58 L1026.09 1485.58 L1026.09 1463.91 L1022.78 1463.91 L1022.78 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M1259.76 1481.64 L1267.4 1481.64 L1267.4 1455.28 L1259.09 1456.95 L1259.09 1452.69 L1267.35 1451.02 L1272.03 1451.02 L1272.03 1481.64 L1279.67 1481.64 L1279.67 1485.58 L1259.76 1485.58 L1259.76 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M1296.08 1451.02 L1300.01 1451.02 L1287.98 1489.98 L1284.04 1489.98 L1296.08 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M1312.05 1451.02 L1315.99 1451.02 L1303.95 1489.98 L1300.01 1489.98 L1312.05 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M1321.87 1481.64 L1329.51 1481.64 L1329.51 1455.28 L1321.2 1456.95 L1321.2 1452.69 L1329.46 1451.02 L1334.13 1451.02 L1334.13 1481.64 L1341.77 1481.64 L1341.77 1485.58 L1321.87 1485.58 L1321.87 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M1347.86 1459.65 L1372.7 1459.65 L1372.7 1463.91 L1369.44 1463.91 L1369.44 1479.84 Q1369.44 1481.51 1369.99 1482.25 Q1370.57 1482.96 1371.84 1482.96 Q1372.19 1482.96 1372.7 1482.92 Q1373.21 1482.85 1373.37 1482.83 L1373.37 1485.9 Q1372.56 1486.2 1371.7 1486.34 Q1370.85 1486.48 1369.99 1486.48 Q1367.21 1486.48 1366.15 1484.98 Q1365.08 1483.45 1365.08 1479.38 L1365.08 1463.91 L1355.52 1463.91 L1355.52 1485.58 L1351.17 1485.58 L1351.17 1463.91 L1347.86 1463.91 L1347.86 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M1599.2 1455.09 L1587.39 1473.54 L1599.2 1473.54 L1599.2 1455.09 M1597.97 1451.02 L1603.85 1451.02 L1603.85 1473.54 L1608.78 1473.54 L1608.78 1477.43 L1603.85 1477.43 L1603.85 1485.58 L1599.2 1485.58 L1599.2 1477.43 L1583.59 1477.43 L1583.59 1472.92 L1597.97 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M1623.48 1451.02 L1627.41 1451.02 L1615.38 1489.98 L1611.44 1489.98 L1623.48 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M1639.45 1451.02 L1643.39 1451.02 L1631.35 1489.98 L1627.41 1489.98 L1639.45 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M1662.62 1466.95 Q1665.98 1467.66 1667.85 1469.93 Q1669.75 1472.2 1669.75 1475.53 Q1669.75 1480.65 1666.23 1483.45 Q1662.71 1486.25 1656.23 1486.25 Q1654.06 1486.25 1651.74 1485.81 Q1649.45 1485.39 1647 1484.54 L1647 1480.02 Q1648.94 1481.16 1651.26 1481.74 Q1653.57 1482.32 1656.09 1482.32 Q1660.49 1482.32 1662.78 1480.58 Q1665.1 1478.84 1665.1 1475.53 Q1665.1 1472.48 1662.95 1470.77 Q1660.82 1469.03 1657 1469.03 L1652.97 1469.03 L1652.97 1465.19 L1657.18 1465.19 Q1660.63 1465.19 1662.46 1463.82 Q1664.29 1462.43 1664.29 1459.84 Q1664.29 1457.18 1662.39 1455.77 Q1660.51 1454.33 1657 1454.33 Q1655.07 1454.33 1652.88 1454.75 Q1650.68 1455.16 1648.04 1456.04 L1648.04 1451.88 Q1650.7 1451.14 1653.01 1450.77 Q1655.35 1450.39 1657.41 1450.39 Q1662.74 1450.39 1665.84 1452.83 Q1668.94 1455.23 1668.94 1459.35 Q1668.94 1462.22 1667.3 1464.21 Q1665.65 1466.18 1662.62 1466.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M1675.26 1459.65 L1700.1 1459.65 L1700.1 1463.91 L1696.83 1463.91 L1696.83 1479.84 Q1696.83 1481.51 1697.39 1482.25 Q1697.97 1482.96 1699.24 1482.96 Q1699.59 1482.96 1700.1 1482.92 Q1700.61 1482.85 1700.77 1482.83 L1700.77 1485.9 Q1699.96 1486.2 1699.1 1486.34 Q1698.25 1486.48 1697.39 1486.48 Q1694.61 1486.48 1693.55 1484.98 Q1692.48 1483.45 1692.48 1479.38 L1692.48 1463.91 L1682.92 1463.91 L1682.92 1485.58 L1678.57 1485.58 L1678.57 1463.91 L1675.26 1463.91 L1675.26 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M1911.68 1451.02 L1930.03 1451.02 L1930.03 1454.96 L1915.96 1454.96 L1915.96 1463.43 Q1916.98 1463.08 1917.99 1462.92 Q1919.01 1462.73 1920.03 1462.73 Q1925.82 1462.73 1929.2 1465.9 Q1932.58 1469.08 1932.58 1474.49 Q1932.58 1480.07 1929.11 1483.17 Q1925.63 1486.25 1919.31 1486.25 Q1917.14 1486.25 1914.87 1485.88 Q1912.62 1485.51 1910.22 1484.77 L1910.22 1480.07 Q1912.3 1481.2 1914.52 1481.76 Q1916.74 1482.32 1919.22 1482.32 Q1923.23 1482.32 1925.56 1480.21 Q1927.9 1478.1 1927.9 1474.49 Q1927.9 1470.88 1925.56 1468.77 Q1923.23 1466.67 1919.22 1466.67 Q1917.35 1466.67 1915.47 1467.08 Q1913.62 1467.5 1911.68 1468.38 L1911.68 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M1948.76 1451.02 L1952.69 1451.02 L1940.66 1489.98 L1936.72 1489.98 L1948.76 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M1964.73 1451.02 L1968.67 1451.02 L1956.63 1489.98 L1952.69 1489.98 L1964.73 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M1987.9 1466.95 Q1991.26 1467.66 1993.13 1469.93 Q1995.03 1472.2 1995.03 1475.53 Q1995.03 1480.65 1991.51 1483.45 Q1987.99 1486.25 1981.51 1486.25 Q1979.34 1486.25 1977.02 1485.81 Q1974.73 1485.39 1972.28 1484.54 L1972.28 1480.02 Q1974.22 1481.16 1976.54 1481.74 Q1978.85 1482.32 1981.37 1482.32 Q1985.77 1482.32 1988.06 1480.58 Q1990.38 1478.84 1990.38 1475.53 Q1990.38 1472.48 1988.23 1470.77 Q1986.1 1469.03 1982.28 1469.03 L1978.25 1469.03 L1978.25 1465.19 L1982.46 1465.19 Q1985.91 1465.19 1987.74 1463.82 Q1989.57 1462.43 1989.57 1459.84 Q1989.57 1457.18 1987.67 1455.77 Q1985.79 1454.33 1982.28 1454.33 Q1980.36 1454.33 1978.16 1454.75 Q1975.96 1455.16 1973.32 1456.04 L1973.32 1451.88 Q1975.98 1451.14 1978.3 1450.77 Q1980.63 1450.39 1982.69 1450.39 Q1988.02 1450.39 1991.12 1452.83 Q1994.22 1455.23 1994.22 1459.35 Q1994.22 1462.22 1992.58 1464.21 Q1990.93 1466.18 1987.9 1466.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M2000.54 1459.65 L2025.38 1459.65 L2025.38 1463.91 L2022.11 1463.91 L2022.11 1479.84 Q2022.11 1481.51 2022.67 1482.25 Q2023.25 1482.96 2024.52 1482.96 Q2024.87 1482.96 2025.38 1482.92 Q2025.89 1482.85 2026.05 1482.83 L2026.05 1485.9 Q2025.24 1486.2 2024.38 1486.34 Q2023.53 1486.48 2022.67 1486.48 Q2019.89 1486.48 2018.83 1484.98 Q2017.76 1483.45 2017.76 1479.38 L2017.76 1463.91 L2008.2 1463.91 L2008.2 1485.58 L2003.85 1485.58 L2003.85 1463.91 L2000.54 1463.91 L2000.54 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M2241.7 1481.64 L2258.02 1481.64 L2258.02 1485.58 L2236.08 1485.58 L2236.08 1481.64 Q2238.74 1478.89 2243.32 1474.26 Q2247.93 1469.61 2249.11 1468.27 Q2251.35 1465.74 2252.23 1464.01 Q2253.14 1462.25 2253.14 1460.56 Q2253.14 1457.8 2251.19 1456.07 Q2249.27 1454.33 2246.17 1454.33 Q2243.97 1454.33 2241.52 1455.09 Q2239.08 1455.86 2236.31 1457.41 L2236.31 1452.69 Q2239.13 1451.55 2241.58 1450.97 Q2244.04 1450.39 2246.08 1450.39 Q2251.45 1450.39 2254.64 1453.08 Q2257.83 1455.77 2257.83 1460.26 Q2257.83 1462.39 2257.02 1464.31 Q2256.24 1466.2 2254.13 1468.8 Q2253.55 1469.47 2250.45 1472.69 Q2247.35 1475.88 2241.7 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M2274.8 1451.02 L2278.74 1451.02 L2266.7 1489.98 L2262.77 1489.98 L2274.8 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M2290.77 1451.02 L2294.71 1451.02 L2282.67 1489.98 L2278.74 1489.98 L2290.77 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M2300.59 1481.64 L2308.23 1481.64 L2308.23 1455.28 L2299.92 1456.95 L2299.92 1452.69 L2308.18 1451.02 L2312.86 1451.02 L2312.86 1481.64 L2320.5 1481.64 L2320.5 1485.58 L2300.59 1485.58 L2300.59 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M2326.58 1459.65 L2351.42 1459.65 L2351.42 1463.91 L2348.16 1463.91 L2348.16 1479.84 Q2348.16 1481.51 2348.71 1482.25 Q2349.29 1482.96 2350.57 1482.96 Q2350.91 1482.96 2351.42 1482.92 Q2351.93 1482.85 2352.09 1482.83 L2352.09 1485.9 Q2351.28 1486.2 2350.43 1486.34 Q2349.57 1486.48 2348.71 1486.48 Q2345.94 1486.48 2344.87 1484.98 Q2343.81 1483.45 2343.81 1479.38 L2343.81 1463.91 L2334.25 1463.91 L2334.25 1485.58 L2329.89 1485.58 L2329.89 1463.91 L2326.58 1463.91 L2326.58 1459.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M1326.35 1545.45 L1306.08 1545.45 Q1306.55 1554.96 1308.75 1559 Q1311.49 1563.97 1316.23 1563.97 Q1321 1563.97 1323.65 1558.97 Q1325.97 1554.58 1326.35 1545.45 M1326.26 1540.03 Q1325.36 1531 1323.65 1527.81 Q1320.91 1522.78 1316.23 1522.78 Q1311.36 1522.78 1308.78 1527.75 Q1306.75 1531.76 1306.14 1540.03 L1326.26 1540.03 M1316.23 1518.01 Q1323.87 1518.01 1328.23 1524.76 Q1332.59 1531.47 1332.59 1543.38 Q1332.59 1555.25 1328.23 1562 Q1323.87 1568.78 1316.23 1568.78 Q1308.56 1568.78 1304.23 1562 Q1299.87 1555.25 1299.87 1543.38 Q1299.87 1531.47 1304.23 1524.76 Q1308.56 1518.01 1316.23 1518.01 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><polyline clip-path="url(#clip672)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="279.704,1299.17 2352.76,1299.17 "/>
<polyline clip-path="url(#clip672)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="279.704,1048.78 2352.76,1048.78 "/>
<polyline clip-path="url(#clip672)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="279.704,798.399 2352.76,798.399 "/>
<polyline clip-path="url(#clip672)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="279.704,548.014 2352.76,548.014 "/>
<polyline clip-path="url(#clip672)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="279.704,297.629 2352.76,297.629 "/>
<polyline clip-path="url(#clip670)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="279.704,1423.18 279.704,47.2441 "/>
<polyline clip-path="url(#clip670)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="279.704,1299.17 298.602,1299.17 "/>
<polyline clip-path="url(#clip670)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="279.704,1048.78 298.602,1048.78 "/>
<polyline clip-path="url(#clip670)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="279.704,798.399 298.602,798.399 "/>
<polyline clip-path="url(#clip670)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="279.704,548.014 298.602,548.014 "/>
<polyline clip-path="url(#clip670)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="279.704,297.629 298.602,297.629 "/>
<path clip-path="url(#clip670)" d="M127.2 1284.97 Q123.589 1284.97 121.76 1288.53 Q119.955 1292.07 119.955 1299.2 Q119.955 1306.31 121.76 1309.87 Q123.589 1313.42 127.2 1313.42 Q130.834 1313.42 132.64 1309.87 Q134.468 1306.31 134.468 1299.2 Q134.468 1292.07 132.64 1288.53 Q130.834 1284.97 127.2 1284.97 M127.2 1281.26 Q133.01 1281.26 136.066 1285.87 Q139.144 1290.45 139.144 1299.2 Q139.144 1307.93 136.066 1312.54 Q133.01 1317.12 127.2 1317.12 Q121.39 1317.12 118.311 1312.54 Q115.256 1307.93 115.256 1299.2 Q115.256 1290.45 118.311 1285.87 Q121.39 1281.26 127.2 1281.26 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M147.362 1310.57 L152.246 1310.57 L152.246 1316.45 L147.362 1316.45 L147.362 1310.57 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M162.57 1315.73 L162.57 1311.47 Q164.329 1312.31 166.135 1312.75 Q167.94 1313.18 169.677 1313.18 Q174.306 1313.18 176.737 1310.08 Q179.19 1306.96 179.538 1300.62 Q178.195 1302.61 176.135 1303.67 Q174.075 1304.74 171.575 1304.74 Q166.39 1304.74 163.357 1301.61 Q160.348 1298.46 160.348 1293.02 Q160.348 1287.7 163.496 1284.48 Q166.644 1281.26 171.876 1281.26 Q177.871 1281.26 181.019 1285.87 Q184.19 1290.45 184.19 1299.2 Q184.19 1307.37 180.301 1312.26 Q176.436 1317.12 169.885 1317.12 Q168.126 1317.12 166.32 1316.77 Q164.515 1316.43 162.57 1315.73 M171.876 1301.08 Q175.024 1301.08 176.852 1298.93 Q178.704 1296.77 178.704 1293.02 Q178.704 1289.3 176.852 1287.14 Q175.024 1284.97 171.876 1284.97 Q168.727 1284.97 166.876 1287.14 Q165.047 1289.3 165.047 1293.02 Q165.047 1296.77 166.876 1298.93 Q168.727 1301.08 171.876 1301.08 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M196.621 1312.51 L212.94 1312.51 L212.94 1316.45 L190.996 1316.45 L190.996 1312.51 Q193.658 1309.76 198.241 1305.13 Q202.848 1300.48 204.028 1299.13 Q206.274 1296.61 207.153 1294.87 Q208.056 1293.12 208.056 1291.43 Q208.056 1288.67 206.112 1286.94 Q204.19 1285.2 201.088 1285.2 Q198.889 1285.2 196.436 1285.96 Q194.005 1286.73 191.227 1288.28 L191.227 1283.56 Q194.051 1282.42 196.505 1281.84 Q198.959 1281.26 200.996 1281.26 Q206.366 1281.26 209.561 1283.95 Q212.755 1286.63 212.755 1291.12 Q212.755 1293.25 211.945 1295.18 Q211.158 1297.07 209.051 1299.67 Q208.473 1300.34 205.371 1303.56 Q202.269 1306.75 196.621 1312.51 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M222.801 1281.89 L241.158 1281.89 L241.158 1285.82 L227.084 1285.82 L227.084 1294.3 Q228.102 1293.95 229.121 1293.79 Q230.139 1293.6 231.158 1293.6 Q236.945 1293.6 240.324 1296.77 Q243.704 1299.94 243.704 1305.36 Q243.704 1310.94 240.232 1314.04 Q236.76 1317.12 230.44 1317.12 Q228.264 1317.12 225.996 1316.75 Q223.75 1316.38 221.343 1315.64 L221.343 1310.94 Q223.426 1312.07 225.648 1312.63 Q227.871 1313.18 230.347 1313.18 Q234.352 1313.18 236.69 1311.08 Q239.028 1308.97 239.028 1305.36 Q239.028 1301.75 236.69 1299.64 Q234.352 1297.54 230.347 1297.54 Q228.473 1297.54 226.598 1297.95 Q224.746 1298.37 222.801 1299.25 L222.801 1281.89 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M126.205 1034.58 Q122.593 1034.58 120.765 1038.15 Q118.959 1041.69 118.959 1048.82 Q118.959 1055.93 120.765 1059.49 Q122.593 1063.03 126.205 1063.03 Q129.839 1063.03 131.644 1059.49 Q133.473 1055.93 133.473 1048.82 Q133.473 1041.69 131.644 1038.15 Q129.839 1034.58 126.205 1034.58 M126.205 1030.88 Q132.015 1030.88 135.07 1035.49 Q138.149 1040.07 138.149 1048.82 Q138.149 1057.55 135.07 1062.15 Q132.015 1066.74 126.205 1066.74 Q120.394 1066.74 117.316 1062.15 Q114.26 1057.55 114.26 1048.82 Q114.26 1040.07 117.316 1035.49 Q120.394 1030.88 126.205 1030.88 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M146.366 1060.18 L151.251 1060.18 L151.251 1066.06 L146.366 1066.06 L146.366 1060.18 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M161.575 1065.35 L161.575 1061.09 Q163.334 1061.92 165.14 1062.36 Q166.945 1062.8 168.681 1062.8 Q173.311 1062.8 175.741 1059.7 Q178.195 1056.57 178.542 1050.23 Q177.2 1052.22 175.139 1053.29 Q173.079 1054.35 170.579 1054.35 Q165.394 1054.35 162.362 1051.23 Q159.353 1048.08 159.353 1042.64 Q159.353 1037.31 162.501 1034.1 Q165.649 1030.88 170.88 1030.88 Q176.876 1030.88 180.024 1035.49 Q183.195 1040.07 183.195 1048.82 Q183.195 1056.99 179.306 1061.87 Q175.44 1066.74 168.889 1066.74 Q167.13 1066.74 165.325 1066.39 Q163.519 1066.04 161.575 1065.35 M170.88 1050.69 Q174.028 1050.69 175.857 1048.54 Q177.709 1046.39 177.709 1042.64 Q177.709 1038.91 175.857 1036.76 Q174.028 1034.58 170.88 1034.58 Q167.732 1034.58 165.88 1036.76 Q164.052 1038.91 164.052 1042.64 Q164.052 1046.39 165.88 1048.54 Q167.732 1050.69 170.88 1050.69 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M191.644 1031.5 L210 1031.5 L210 1035.44 L195.926 1035.44 L195.926 1043.91 Q196.945 1043.56 197.963 1043.4 Q198.982 1043.22 200 1043.22 Q205.787 1043.22 209.167 1046.39 Q212.547 1049.56 212.547 1054.98 Q212.547 1060.55 209.074 1063.66 Q205.602 1066.74 199.283 1066.74 Q197.107 1066.74 194.838 1066.36 Q192.593 1065.99 190.186 1065.25 L190.186 1060.55 Q192.269 1061.69 194.491 1062.24 Q196.713 1062.8 199.19 1062.8 Q203.195 1062.8 205.533 1060.69 Q207.871 1058.59 207.871 1054.98 Q207.871 1051.36 205.533 1049.26 Q203.195 1047.15 199.19 1047.15 Q197.315 1047.15 195.44 1047.57 Q193.588 1047.99 191.644 1048.86 L191.644 1031.5 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M231.76 1034.58 Q228.148 1034.58 226.32 1038.15 Q224.514 1041.69 224.514 1048.82 Q224.514 1055.93 226.32 1059.49 Q228.148 1063.03 231.76 1063.03 Q235.394 1063.03 237.199 1059.49 Q239.028 1055.93 239.028 1048.82 Q239.028 1041.69 237.199 1038.15 Q235.394 1034.58 231.76 1034.58 M231.76 1030.88 Q237.57 1030.88 240.625 1035.49 Q243.704 1040.07 243.704 1048.82 Q243.704 1057.55 240.625 1062.15 Q237.57 1066.74 231.76 1066.74 Q225.949 1066.74 222.871 1062.15 Q219.815 1057.55 219.815 1048.82 Q219.815 1040.07 222.871 1035.49 Q225.949 1030.88 231.76 1030.88 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M127.2 784.198 Q123.589 784.198 121.76 787.762 Q119.955 791.304 119.955 798.434 Q119.955 805.54 121.76 809.105 Q123.589 812.647 127.2 812.647 Q130.834 812.647 132.64 809.105 Q134.468 805.54 134.468 798.434 Q134.468 791.304 132.64 787.762 Q130.834 784.198 127.2 784.198 M127.2 780.494 Q133.01 780.494 136.066 785.1 Q139.144 789.684 139.144 798.434 Q139.144 807.16 136.066 811.767 Q133.01 816.35 127.2 816.35 Q121.39 816.35 118.311 811.767 Q115.256 807.16 115.256 798.434 Q115.256 789.684 118.311 785.1 Q121.39 780.494 127.2 780.494 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M147.362 809.799 L152.246 809.799 L152.246 815.679 L147.362 815.679 L147.362 809.799 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M162.57 814.961 L162.57 810.702 Q164.329 811.535 166.135 811.975 Q167.94 812.415 169.677 812.415 Q174.306 812.415 176.737 809.313 Q179.19 806.188 179.538 799.846 Q178.195 801.836 176.135 802.901 Q174.075 803.966 171.575 803.966 Q166.39 803.966 163.357 800.841 Q160.348 797.693 160.348 792.253 Q160.348 786.929 163.496 783.711 Q166.644 780.494 171.876 780.494 Q177.871 780.494 181.019 785.1 Q184.19 789.684 184.19 798.434 Q184.19 806.605 180.301 811.489 Q176.436 816.35 169.885 816.35 Q168.126 816.35 166.32 816.003 Q164.515 815.656 162.57 814.961 M171.876 800.309 Q175.024 800.309 176.852 798.156 Q178.704 796.003 178.704 792.253 Q178.704 788.526 176.852 786.374 Q175.024 784.198 171.876 784.198 Q168.727 784.198 166.876 786.374 Q165.047 788.526 165.047 792.253 Q165.047 796.003 166.876 798.156 Q168.727 800.309 171.876 800.309 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M191.413 781.119 L213.635 781.119 L213.635 783.11 L201.088 815.679 L196.204 815.679 L208.01 785.054 L191.413 785.054 L191.413 781.119 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M222.801 781.119 L241.158 781.119 L241.158 785.054 L227.084 785.054 L227.084 793.526 Q228.102 793.179 229.121 793.017 Q230.139 792.832 231.158 792.832 Q236.945 792.832 240.324 796.003 Q243.704 799.174 243.704 804.591 Q243.704 810.17 240.232 813.272 Q236.76 816.35 230.44 816.35 Q228.264 816.35 225.996 815.98 Q223.75 815.609 221.343 814.869 L221.343 810.17 Q223.426 811.304 225.648 811.859 Q227.871 812.415 230.347 812.415 Q234.352 812.415 236.69 810.309 Q239.028 808.202 239.028 804.591 Q239.028 800.98 236.69 798.873 Q234.352 796.767 230.347 796.767 Q228.473 796.767 226.598 797.184 Q224.746 797.6 222.801 798.48 L222.801 781.119 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M117.015 561.359 L124.654 561.359 L124.654 534.993 L116.343 536.66 L116.343 532.401 L124.607 530.734 L129.283 530.734 L129.283 561.359 L136.922 561.359 L136.922 565.294 L117.015 565.294 L117.015 561.359 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M146.366 559.414 L151.251 559.414 L151.251 565.294 L146.366 565.294 L146.366 559.414 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M171.436 533.813 Q167.825 533.813 165.996 537.377 Q164.19 540.919 164.19 548.049 Q164.19 555.155 165.996 558.72 Q167.825 562.262 171.436 562.262 Q175.07 562.262 176.876 558.72 Q178.704 555.155 178.704 548.049 Q178.704 540.919 176.876 537.377 Q175.07 533.813 171.436 533.813 M171.436 530.109 Q177.246 530.109 180.301 534.715 Q183.38 539.299 183.38 548.049 Q183.38 556.776 180.301 561.382 Q177.246 565.965 171.436 565.965 Q165.626 565.965 162.547 561.382 Q159.491 556.776 159.491 548.049 Q159.491 539.299 162.547 534.715 Q165.626 530.109 171.436 530.109 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M201.598 533.813 Q197.987 533.813 196.158 537.377 Q194.352 540.919 194.352 548.049 Q194.352 555.155 196.158 558.72 Q197.987 562.262 201.598 562.262 Q205.232 562.262 207.037 558.72 Q208.866 555.155 208.866 548.049 Q208.866 540.919 207.037 537.377 Q205.232 533.813 201.598 533.813 M201.598 530.109 Q207.408 530.109 210.463 534.715 Q213.542 539.299 213.542 548.049 Q213.542 556.776 210.463 561.382 Q207.408 565.965 201.598 565.965 Q195.787 565.965 192.709 561.382 Q189.653 556.776 189.653 548.049 Q189.653 539.299 192.709 534.715 Q195.787 530.109 201.598 530.109 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M231.76 533.813 Q228.148 533.813 226.32 537.377 Q224.514 540.919 224.514 548.049 Q224.514 555.155 226.32 558.72 Q228.148 562.262 231.76 562.262 Q235.394 562.262 237.199 558.72 Q239.028 555.155 239.028 548.049 Q239.028 540.919 237.199 537.377 Q235.394 533.813 231.76 533.813 M231.76 530.109 Q237.57 530.109 240.625 534.715 Q243.704 539.299 243.704 548.049 Q243.704 556.776 240.625 561.382 Q237.57 565.965 231.76 565.965 Q225.949 565.965 222.871 561.382 Q219.815 556.776 219.815 548.049 Q219.815 539.299 222.871 534.715 Q225.949 530.109 231.76 530.109 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M118.01 310.974 L125.649 310.974 L125.649 284.608 L117.339 286.275 L117.339 282.016 L125.603 280.349 L130.279 280.349 L130.279 310.974 L137.917 310.974 L137.917 314.909 L118.01 314.909 L118.01 310.974 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M147.362 309.029 L152.246 309.029 L152.246 314.909 L147.362 314.909 L147.362 309.029 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M172.431 283.428 Q168.82 283.428 166.991 286.993 Q165.186 290.534 165.186 297.664 Q165.186 304.77 166.991 308.335 Q168.82 311.877 172.431 311.877 Q176.065 311.877 177.871 308.335 Q179.7 304.77 179.7 297.664 Q179.7 290.534 177.871 286.993 Q176.065 283.428 172.431 283.428 M172.431 279.724 Q178.241 279.724 181.297 284.33 Q184.376 288.914 184.376 297.664 Q184.376 306.391 181.297 310.997 Q178.241 315.58 172.431 315.58 Q166.621 315.58 163.542 310.997 Q160.487 306.391 160.487 297.664 Q160.487 288.914 163.542 284.33 Q166.621 279.724 172.431 279.724 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M196.621 310.974 L212.94 310.974 L212.94 314.909 L190.996 314.909 L190.996 310.974 Q193.658 308.219 198.241 303.59 Q202.848 298.937 204.028 297.594 Q206.274 295.071 207.153 293.335 Q208.056 291.576 208.056 289.886 Q208.056 287.131 206.112 285.395 Q204.19 283.659 201.088 283.659 Q198.889 283.659 196.436 284.423 Q194.005 285.187 191.227 286.738 L191.227 282.016 Q194.051 280.881 196.505 280.303 Q198.959 279.724 200.996 279.724 Q206.366 279.724 209.561 282.409 Q212.755 285.094 212.755 289.585 Q212.755 291.715 211.945 293.636 Q211.158 295.534 209.051 298.127 Q208.473 298.798 205.371 302.016 Q202.269 305.21 196.621 310.974 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M222.801 280.349 L241.158 280.349 L241.158 284.284 L227.084 284.284 L227.084 292.756 Q228.102 292.409 229.121 292.247 Q230.139 292.062 231.158 292.062 Q236.945 292.062 240.324 295.233 Q243.704 298.404 243.704 303.821 Q243.704 309.4 240.232 312.502 Q236.76 315.58 230.44 315.58 Q228.264 315.58 225.996 315.21 Q223.75 314.84 221.343 314.099 L221.343 309.4 Q223.426 310.534 225.648 311.09 Q227.871 311.645 230.347 311.645 Q234.352 311.645 236.69 309.539 Q239.028 307.432 239.028 303.821 Q239.028 300.21 236.69 298.104 Q234.352 295.997 230.347 295.997 Q228.473 295.997 226.598 296.414 Q224.746 296.83 222.801 297.71 L222.801 280.349 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M33.8307 724.772 Q33.2578 725.759 33.0032 726.937 Q32.7167 728.082 32.7167 729.483 Q32.7167 734.448 35.9632 737.122 Q39.1779 739.763 45.2253 739.763 L64.0042 739.763 L64.0042 745.652 L28.3562 745.652 L28.3562 739.763 L33.8944 739.763 Q30.6479 737.917 29.0883 734.957 Q27.4968 731.997 27.4968 727.764 Q27.4968 727.159 27.5923 726.427 Q27.656 725.695 27.8151 724.804 L33.8307 724.772 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><line clip-path="url(#clip672)" x1="338.375" y1="548.014" x2="338.375" y2="532.014" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="338.375" y1="548.014" x2="322.375" y2="548.014" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="338.375" y1="548.014" x2="338.375" y2="564.014" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="338.375" y1="548.014" x2="354.375" y2="548.014" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1316.23" y1="548.565" x2="1316.23" y2="532.565" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1316.23" y1="548.565" x2="1300.23" y2="548.565" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1316.23" y1="548.565" x2="1316.23" y2="564.565" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1316.23" y1="548.565" x2="1332.23" y2="548.565" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="338.375" y1="612.706" x2="338.375" y2="596.706" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="338.375" y1="612.706" x2="322.375" y2="612.706" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="338.375" y1="612.706" x2="338.375" y2="628.706" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="338.375" y1="612.706" x2="354.375" y2="612.706" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1316.23" y1="612.809" x2="1316.23" y2="596.809" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1316.23" y1="612.809" x2="1300.23" y2="612.809" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1316.23" y1="612.809" x2="1316.23" y2="628.809" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1316.23" y1="612.809" x2="1332.23" y2="612.809" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="440.384" y1="729.975" x2="440.384" y2="713.975" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="440.384" y1="729.975" x2="424.384" y2="729.975" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="440.384" y1="729.975" x2="440.384" y2="745.975" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="440.384" y1="729.975" x2="456.384" y2="729.975" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="2192.08" y1="729.975" x2="2192.08" y2="713.975" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="2192.08" y1="729.975" x2="2176.08" y2="729.975" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="2192.08" y1="729.975" x2="2192.08" y2="745.975" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="2192.08" y1="729.975" x2="2208.08" y2="729.975" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1418.24" y1="729.989" x2="1418.24" y2="713.989" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1418.24" y1="729.989" x2="1402.24" y2="729.989" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1418.24" y1="729.989" x2="1418.24" y2="745.989" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1418.24" y1="729.989" x2="1434.24" y2="729.989" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1214.22" y1="729.989" x2="1214.22" y2="713.989" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1214.22" y1="729.989" x2="1198.22" y2="729.989" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1214.22" y1="729.989" x2="1214.22" y2="745.989" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1214.22" y1="729.989" x2="1230.22" y2="729.989" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1706.89" y1="793.857" x2="1706.89" y2="777.857" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1706.89" y1="793.857" x2="1690.89" y2="793.857" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1706.89" y1="793.857" x2="1706.89" y2="809.857" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1706.89" y1="793.857" x2="1722.89" y2="793.857" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="925.57" y1="793.857" x2="925.57" y2="777.857" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="925.57" y1="793.857" x2="909.57" y2="793.857" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="925.57" y1="793.857" x2="925.57" y2="809.857" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="925.57" y1="793.857" x2="941.57" y2="793.857" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1903.42" y1="793.87" x2="1903.42" y2="777.87" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1903.42" y1="793.87" x2="1887.42" y2="793.87" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1903.42" y1="793.87" x2="1903.42" y2="809.87" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1903.42" y1="793.87" x2="1919.42" y2="793.87" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="729.036" y1="793.87" x2="729.036" y2="777.87" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="729.036" y1="793.87" x2="713.036" y2="793.87" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="729.036" y1="793.87" x2="729.036" y2="809.87" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="729.036" y1="793.87" x2="745.036" y2="793.87" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="834.739" y1="1123.57" x2="834.739" y2="1107.57" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="834.739" y1="1123.57" x2="818.739" y2="1123.57" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="834.739" y1="1123.57" x2="834.739" y2="1139.57" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="834.739" y1="1123.57" x2="850.739" y2="1123.57" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1797.72" y1="1123.57" x2="1797.72" y2="1107.57" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1797.72" y1="1123.57" x2="1781.72" y2="1123.57" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1797.72" y1="1123.57" x2="1797.72" y2="1139.57" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1797.72" y1="1123.57" x2="1813.72" y2="1123.57" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="819.865" y1="1123.59" x2="819.865" y2="1107.59" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="819.865" y1="1123.59" x2="803.865" y2="1123.59" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="819.865" y1="1123.59" x2="819.865" y2="1139.59" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="819.865" y1="1123.59" x2="835.865" y2="1123.59" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1812.59" y1="1123.59" x2="1812.59" y2="1107.59" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1812.59" y1="1123.59" x2="1796.59" y2="1123.59" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1812.59" y1="1123.59" x2="1812.59" y2="1139.59" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1812.59" y1="1123.59" x2="1828.59" y2="1123.59" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1801.69" y1="1423.06" x2="1801.69" y2="1407.06" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1801.69" y1="1423.06" x2="1785.69" y2="1423.06" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1801.69" y1="1423.06" x2="1801.69" y2="1439.06" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1801.69" y1="1423.06" x2="1817.69" y2="1423.06" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="830.768" y1="1423.06" x2="830.768" y2="1407.06" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="830.768" y1="1423.06" x2="814.768" y2="1423.06" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="830.768" y1="1423.06" x2="830.768" y2="1439.06" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="830.768" y1="1423.06" x2="846.768" y2="1423.06" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="823.836" y1="1423.18" x2="823.836" y2="1407.18" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="823.836" y1="1423.18" x2="807.836" y2="1423.18" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="823.836" y1="1423.18" x2="823.836" y2="1439.18" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="823.836" y1="1423.18" x2="839.836" y2="1423.18" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1808.62" y1="1423.18" x2="1808.62" y2="1407.18" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1808.62" y1="1423.18" x2="1792.62" y2="1423.18" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1808.62" y1="1423.18" x2="1808.62" y2="1439.18" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<line clip-path="url(#clip672)" x1="1808.62" y1="1423.18" x2="1824.62" y2="1423.18" style="stroke:#009af9; stroke-width:3.2; stroke-opacity:1"/>
<path clip-path="url(#clip670)" d="M1905.19 196.789 L2283.65 196.789 L2283.65 93.1086 L1905.19 93.1086  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<polyline clip-path="url(#clip670)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1905.19,196.789 2283.65,196.789 2283.65,93.1086 1905.19,93.1086 1905.19,196.789 "/>
<line clip-path="url(#clip670)" x1="1997.33" y1="144.949" x2="1997.33" y2="122.193" style="stroke:#009af9; stroke-width:4.55111; stroke-opacity:1"/>
<line clip-path="url(#clip670)" x1="1997.33" y1="144.949" x2="1974.57" y2="144.949" style="stroke:#009af9; stroke-width:4.55111; stroke-opacity:1"/>
<line clip-path="url(#clip670)" x1="1997.33" y1="144.949" x2="1997.33" y2="167.704" style="stroke:#009af9; stroke-width:4.55111; stroke-opacity:1"/>
<line clip-path="url(#clip670)" x1="1997.33" y1="144.949" x2="2020.08" y2="144.949" style="stroke:#009af9; stroke-width:4.55111; stroke-opacity:1"/>
<path clip-path="url(#clip670)" d="M2089.46 127.669 L2118.7 127.669 L2118.7 131.604 L2106.43 131.604 L2106.43 162.229 L2101.73 162.229 L2101.73 131.604 L2089.46 131.604 L2089.46 127.669 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M2131.08 140.284 Q2130.37 139.868 2129.51 139.682 Q2128.68 139.474 2127.66 139.474 Q2124.05 139.474 2122.1 141.835 Q2120.18 144.173 2120.18 148.571 L2120.18 162.229 L2115.9 162.229 L2115.9 136.303 L2120.18 136.303 L2120.18 140.331 Q2121.52 137.969 2123.68 136.835 Q2125.83 135.678 2128.91 135.678 Q2129.35 135.678 2129.88 135.747 Q2130.41 135.794 2131.06 135.909 L2131.08 140.284 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M2135.55 136.303 L2139.81 136.303 L2139.81 162.229 L2135.55 162.229 L2135.55 136.303 M2135.55 126.21 L2139.81 126.21 L2139.81 131.604 L2135.55 131.604 L2135.55 126.21 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M2145.67 136.303 L2150.18 136.303 L2158.28 158.062 L2166.38 136.303 L2170.9 136.303 L2161.18 162.229 L2155.39 162.229 L2145.67 136.303 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M2176.78 136.303 L2181.04 136.303 L2181.04 162.229 L2176.78 162.229 L2176.78 136.303 M2176.78 126.21 L2181.04 126.21 L2181.04 131.604 L2176.78 131.604 L2176.78 126.21 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M2201.73 149.196 Q2196.57 149.196 2194.58 150.377 Q2192.59 151.557 2192.59 154.405 Q2192.59 156.673 2194.07 158.016 Q2195.57 159.335 2198.14 159.335 Q2201.69 159.335 2203.82 156.835 Q2205.97 154.312 2205.97 150.145 L2205.97 149.196 L2201.73 149.196 M2210.23 147.437 L2210.23 162.229 L2205.97 162.229 L2205.97 158.293 Q2204.51 160.655 2202.33 161.789 Q2200.16 162.9 2197.01 162.9 Q2193.03 162.9 2190.67 160.678 Q2188.33 158.432 2188.33 154.682 Q2188.33 150.307 2191.25 148.085 Q2194.19 145.863 2200 145.863 L2205.97 145.863 L2205.97 145.446 Q2205.97 142.507 2204.02 140.909 Q2202.1 139.289 2198.61 139.289 Q2196.38 139.289 2194.28 139.821 Q2192.17 140.354 2190.23 141.419 L2190.23 137.483 Q2192.57 136.581 2194.76 136.141 Q2196.96 135.678 2199.05 135.678 Q2204.67 135.678 2207.45 138.594 Q2210.23 141.511 2210.23 147.437 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M2219 126.21 L2223.26 126.21 L2223.26 162.229 L2219 162.229 L2219 126.21 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M2242.4 126.257 Q2239.3 131.581 2237.8 136.789 Q2236.29 141.997 2236.29 147.344 Q2236.29 152.692 2237.8 157.946 Q2239.32 163.178 2242.4 168.479 L2238.7 168.479 Q2235.23 163.039 2233.49 157.784 Q2231.78 152.53 2231.78 147.344 Q2231.78 142.182 2233.49 136.951 Q2235.2 131.72 2238.7 126.257 L2242.4 126.257 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip670)" d="M2250 126.257 L2253.7 126.257 Q2257.17 131.72 2258.88 136.951 Q2260.62 142.182 2260.62 147.344 Q2260.62 152.53 2258.88 157.784 Q2257.17 163.039 2253.7 168.479 L2250 168.479 Q2253.07 163.178 2254.58 157.946 Q2256.11 152.692 2256.11 147.344 Q2256.11 141.997 2254.58 136.789 Q2253.07 131.581 2250 126.257 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /></svg>

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
[ Info: VUMPS init:	obj = +4.989263659981e-01	err = 7.2657e-02
[ Info: VUMPS   1:	obj = -3.327977953900e-01	err = 3.6721157182e-01	time = 0.04 sec
[ Info: VUMPS   2:	obj = -8.613582314697e-01	err = 1.3817395701e-01	time = 0.03 sec
[ Info: VUMPS   3:	obj = -8.854094286633e-01	err = 1.2636996941e-02	time = 0.03 sec
[ Info: VUMPS   4:	obj = -8.860034868896e-01	err = 4.8894227850e-03	time = 0.04 sec
[ Info: VUMPS   5:	obj = -8.861408641612e-01	err = 3.6101081699e-03	time = 0.03 sec
[ Info: VUMPS   6:	obj = -8.861956019962e-01	err = 2.3873340370e-03	time = 0.03 sec
[ Info: VUMPS   7:	obj = -8.862171468896e-01	err = 2.3946876976e-03	time = 0.08 sec
[ Info: VUMPS   8:	obj = -8.862275876134e-01	err = 2.0620267318e-03	time = 0.03 sec
[ Info: VUMPS   9:	obj = -8.862319848598e-01	err = 2.3656536837e-03	time = 0.03 sec
[ Info: VUMPS  10:	obj = -8.862342347151e-01	err = 2.3171962616e-03	time = 0.03 sec
[ Info: VUMPS  11:	obj = -8.862351816575e-01	err = 2.5035157089e-03	time = 0.03 sec
[ Info: VUMPS  12:	obj = -8.862358518698e-01	err = 2.3743584569e-03	time = 0.03 sec
[ Info: VUMPS  13:	obj = -8.862358277380e-01	err = 2.6661781012e-03	time = 0.03 sec
[ Info: VUMPS  14:	obj = -8.862361321088e-01	err = 2.5349398073e-03	time = 0.03 sec
[ Info: VUMPS  15:	obj = -8.862362398790e-01	err = 2.5961328460e-03	time = 0.03 sec
[ Info: VUMPS  16:	obj = -8.862364361640e-01	err = 2.4630430503e-03	time = 0.03 sec
[ Info: VUMPS  17:	obj = -8.862363204823e-01	err = 2.6657013311e-03	time = 0.03 sec
[ Info: VUMPS  18:	obj = -8.862366543525e-01	err = 2.4033859380e-03	time = 0.03 sec
[ Info: VUMPS  19:	obj = -8.862368027756e-01	err = 2.4751545251e-03	time = 0.03 sec
[ Info: VUMPS  20:	obj = -8.862370949666e-01	err = 2.2173189980e-03	time = 0.05 sec
[ Info: VUMPS  21:	obj = -8.862372454594e-01	err = 2.3525966932e-03	time = 0.05 sec
[ Info: VUMPS  22:	obj = -8.862376422218e-01	err = 1.9729740148e-03	time = 0.04 sec
[ Info: VUMPS  23:	obj = -8.862378512527e-01	err = 2.1069315226e-03	time = 0.07 sec
[ Info: VUMPS  24:	obj = -8.862381943532e-01	err = 1.7728171465e-03	time = 0.04 sec
[ Info: VUMPS  25:	obj = -8.862384261527e-01	err = 1.8605694862e-03	time = 0.03 sec
[ Info: VUMPS  26:	obj = -8.862386821954e-01	err = 1.6005785804e-03	time = 0.04 sec
[ Info: VUMPS  27:	obj = -8.862388952173e-01	err = 1.6219782649e-03	time = 0.03 sec
[ Info: VUMPS  28:	obj = -8.862390777872e-01	err = 1.4098536096e-03	time = 0.05 sec
[ Info: VUMPS  29:	obj = -8.862392504199e-01	err = 1.3742836059e-03	time = 0.05 sec
[ Info: VUMPS  30:	obj = -8.862393918124e-01	err = 1.1981526208e-03	time = 0.05 sec
[ Info: VUMPS  31:	obj = -8.862395189366e-01	err = 1.1227495232e-03	time = 0.05 sec
[ Info: VUMPS  32:	obj = -8.862396317750e-01	err = 9.6790961802e-04	time = 0.05 sec
[ Info: VUMPS  33:	obj = -8.862397273358e-01	err = 8.7235460197e-04	time = 0.04 sec
[ Info: VUMPS  34:	obj = -8.862398137238e-01	err = 7.3864358738e-04	time = 0.03 sec
[ Info: VUMPS  35:	obj = -8.862398845201e-01	err = 6.4279120671e-04	time = 0.03 sec
[ Info: VUMPS  36:	obj = -8.862399481195e-01	err = 5.3518713271e-04	time = 0.04 sec
[ Info: VUMPS  37:	obj = -8.862400017558e-01	err = 4.5437031592e-04	time = 0.04 sec
[ Info: VUMPS  38:	obj = -8.862400510754e-01	err = 3.7529927192e-04	time = 0.04 sec
[ Info: VUMPS  39:	obj = -8.862400961622e-01	err = 3.1709478672e-04	time = 0.04 sec
[ Info: VUMPS  40:	obj = -8.862401404447e-01	err = 2.6516037727e-04	time = 0.05 sec
[ Info: VUMPS  41:	obj = -8.862401848458e-01	err = 2.3115404187e-04	time = 0.04 sec
[ Info: VUMPS  42:	obj = -8.862402315158e-01	err = 2.0263944361e-04	time = 0.04 sec
[ Info: VUMPS  43:	obj = -8.862402813257e-01	err = 1.8985412367e-04	time = 0.04 sec
[ Info: VUMPS  44:	obj = -8.862403357012e-01	err = 1.7867891600e-04	time = 0.04 sec
[ Info: VUMPS  45:	obj = -8.862403952507e-01	err = 1.7958231642e-04	time = 0.04 sec
[ Info: VUMPS  46:	obj = -8.862404608464e-01	err = 1.7788607349e-04	time = 0.06 sec
[ Info: VUMPS  47:	obj = -8.862405326781e-01	err = 1.8395392822e-04	time = 0.06 sec
[ Info: VUMPS  48:	obj = -8.862406109377e-01	err = 1.8586675040e-04	time = 0.06 sec
[ Info: VUMPS  49:	obj = -8.862406951285e-01	err = 1.9086368515e-04	time = 0.06 sec
[ Info: VUMPS  50:	obj = -8.862407845139e-01	err = 1.9239669989e-04	time = 0.06 sec
[ Info: VUMPS  51:	obj = -8.862408776839e-01	err = 1.9330600812e-04	time = 0.04 sec
[ Info: VUMPS  52:	obj = -8.862409729223e-01	err = 1.9221903809e-04	time = 0.04 sec
[ Info: VUMPS  53:	obj = -8.862410680644e-01	err = 1.8904918253e-04	time = 0.06 sec
[ Info: VUMPS  54:	obj = -8.862411608804e-01	err = 1.8372886328e-04	time = 0.07 sec
[ Info: VUMPS  55:	obj = -8.862412491530e-01	err = 1.7627491752e-04	time = 0.04 sec
[ Info: VUMPS  56:	obj = -8.862413310135e-01	err = 1.6700719117e-04	time = 0.06 sec
[ Info: VUMPS  57:	obj = -8.862414050343e-01	err = 1.5632121418e-04	time = 0.06 sec
[ Info: VUMPS  58:	obj = -8.862414703832e-01	err = 1.4463100169e-04	time = 0.06 sec
[ Info: VUMPS  59:	obj = -8.862415267819e-01	err = 1.3240541746e-04	time = 0.04 sec
[ Info: VUMPS  60:	obj = -8.862415744605e-01	err = 1.2003094122e-04	time = 0.04 sec
[ Info: VUMPS  61:	obj = -8.862416140187e-01	err = 1.0787541315e-04	time = 0.04 sec
[ Info: VUMPS  62:	obj = -8.862416463016e-01	err = 9.6199569445e-05	time = 0.04 sec
[ Info: VUMPS  63:	obj = -8.862416722680e-01	err = 8.5213478564e-05	time = 0.04 sec
[ Info: VUMPS  64:	obj = -8.862416928959e-01	err = 7.5037966909e-05	time = 0.04 sec
[ Info: VUMPS  65:	obj = -8.862417091097e-01	err = 6.5748007888e-05	time = 0.04 sec
[ Info: VUMPS  66:	obj = -8.862417217414e-01	err = 5.7359888977e-05	time = 0.04 sec
[ Info: VUMPS  67:	obj = -8.862417315096e-01	err = 4.9862390372e-05	time = 0.08 sec
[ Info: VUMPS  68:	obj = -8.862417390182e-01	err = 4.3212839861e-05	time = 0.04 sec
[ Info: VUMPS  69:	obj = -8.862417447618e-01	err = 3.7357188222e-05	time = 0.04 sec
[ Info: VUMPS  70:	obj = -8.862417491385e-01	err = 3.2228700378e-05	time = 0.04 sec
[ Info: VUMPS  71:	obj = -8.862417524639e-01	err = 2.7759484498e-05	time = 0.04 sec
[ Info: VUMPS  72:	obj = -8.862417549849e-01	err = 2.3879497160e-05	time = 0.04 sec
[ Info: VUMPS  73:	obj = -8.862417568933e-01	err = 2.0522797005e-05	time = 0.04 sec
[ Info: VUMPS  74:	obj = -8.862417583366e-01	err = 1.7626192855e-05	time = 0.04 sec
[ Info: VUMPS  75:	obj = -8.862417594276e-01	err = 1.5132458948e-05	time = 0.04 sec
[ Info: VUMPS  76:	obj = -8.862417602523e-01	err = 1.2989135774e-05	time = 0.06 sec
[ Info: VUMPS  77:	obj = -8.862417608759e-01	err = 1.1149763921e-05	time = 0.04 sec
[ Info: VUMPS  78:	obj = -8.862417613476e-01	err = 9.5726959771e-06	time = 0.04 sec
[ Info: VUMPS  79:	obj = -8.862417617047e-01	err = 8.2216567005e-06	time = 0.05 sec
[ Info: VUMPS  80:	obj = -8.862417619753e-01	err = 7.0647166216e-06	time = 0.05 sec
[ Info: VUMPS  81:	obj = -8.862417621806e-01	err = 6.0742789761e-06	time = 0.04 sec
[ Info: VUMPS  82:	obj = -8.862417623365e-01	err = 5.2264003503e-06	time = 0.04 sec
[ Info: VUMPS  83:	obj = -8.862417624550e-01	err = 4.5004618841e-06	time = 0.04 sec
[ Info: VUMPS  84:	obj = -8.862417625453e-01	err = 3.8786795482e-06	time = 0.04 sec
[ Info: VUMPS  85:	obj = -8.862417626141e-01	err = 3.3458775562e-06	time = 0.04 sec
[ Info: VUMPS  86:	obj = -8.862417626666e-01	err = 2.8890331408e-06	time = 0.06 sec
[ Info: VUMPS  87:	obj = -8.862417627068e-01	err = 2.4970540348e-06	time = 0.04 sec
[ Info: VUMPS  88:	obj = -8.862417627375e-01	err = 2.1604343906e-06	time = 0.06 sec
[ Info: VUMPS  89:	obj = -8.862417627610e-01	err = 1.8711402154e-06	time = 0.05 sec
[ Info: VUMPS  90:	obj = -8.862417627790e-01	err = 1.6222006325e-06	time = 0.04 sec
[ Info: VUMPS  91:	obj = -8.862417627929e-01	err = 1.4077789981e-06	time = 0.04 sec
[ Info: VUMPS  92:	obj = -8.862417628035e-01	err = 1.2229350105e-06	time = 0.04 sec
[ Info: VUMPS  93:	obj = -8.862417628117e-01	err = 1.0633924828e-06	time = 0.04 sec
[ Info: VUMPS  94:	obj = -8.862417628180e-01	err = 9.2553335577e-07	time = 0.06 sec
[ Info: VUMPS  95:	obj = -8.862417628229e-01	err = 8.0725584351e-07	time = 0.04 sec
[ Info: VUMPS  96:	obj = -8.862417628266e-01	err = 7.0473150657e-07	time = 0.04 sec
[ Info: VUMPS  97:	obj = -8.862417628295e-01	err = 6.1568056415e-07	time = 0.04 sec
[ Info: VUMPS  98:	obj = -8.862417628318e-01	err = 5.3828819844e-07	time = 0.04 sec
[ Info: VUMPS  99:	obj = -8.862417628335e-01	err = 4.7093780362e-07	time = 0.05 sec
┌ Warning: VUMPS cancel 100:	obj = -8.862417628348e-01	err = 4.1227746817e-07	time = 4.34 sec
└ @ MPSKit ~/Projects/Julia/MPSKit.jl/src/algorithms/groundstate/vumps.jl:67

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
  <clipPath id="clip700">
    <rect x="0" y="0" width="2400" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip700)" d="M0 1600 L2400 1600 L2400 0 L0 0  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip701">
    <rect x="480" y="0" width="1681" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip700)" d="M189.496 1352.62 L2352.76 1352.62 L2352.76 123.472 L189.496 123.472  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip702">
    <rect x="189" y="123" width="2164" height="1230"/>
  </clipPath>
</defs>
<polyline clip-path="url(#clip702)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="250.72,1352.62 250.72,123.472 "/>
<polyline clip-path="url(#clip700)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="189.496,1352.62 2352.76,1352.62 "/>
<polyline clip-path="url(#clip700)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="250.72,1352.62 250.72,1371.52 "/>
<path clip-path="url(#clip700)" d="M117.476 1508.55 L138.148 1487.88 L140.931 1490.66 L132.256 1499.34 L153.911 1520.99 L150.588 1524.32 L128.933 1502.66 L120.258 1511.34 L117.476 1508.55 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M155.826 1488.04 Q155.024 1488.26 154.287 1488.73 Q153.551 1489.17 152.831 1489.89 Q150.277 1492.45 150.572 1495.49 Q150.867 1498.5 153.976 1501.61 L163.634 1511.27 L160.606 1514.3 L142.273 1495.97 L145.301 1492.94 L148.149 1495.79 Q147.429 1493.17 148.149 1490.84 Q148.853 1488.5 151.03 1486.33 Q151.341 1486.01 151.767 1485.69 Q152.176 1485.34 152.716 1484.97 L155.826 1488.04 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M156.17 1482.07 L159.182 1479.06 L177.514 1497.39 L174.502 1500.4 L156.17 1482.07 M149.033 1474.93 L152.045 1471.92 L155.859 1475.74 L152.847 1478.75 L149.033 1474.93 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M163.323 1474.92 L166.514 1471.73 L187.629 1481.38 L177.972 1460.27 L181.164 1457.08 L192.622 1482.28 L188.53 1486.37 L163.323 1474.92 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M185.321 1452.92 L188.333 1449.91 L206.665 1468.24 L203.654 1471.25 L185.321 1452.92 M178.185 1445.78 L181.197 1442.77 L185.01 1446.58 L181.999 1449.6 L178.185 1445.78 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M212.083 1444.39 Q208.433 1448.04 207.86 1450.28 Q207.287 1452.53 209.301 1454.54 Q210.905 1456.14 212.902 1456.04 Q214.899 1455.91 216.715 1454.1 Q219.22 1451.59 218.958 1448.32 Q218.696 1445.01 215.75 1442.07 L215.079 1441.4 L212.083 1444.39 M216.846 1437.14 L227.306 1447.6 L224.294 1450.61 L221.511 1447.83 Q222.15 1450.53 221.413 1452.87 Q220.66 1455.19 218.434 1457.42 Q215.619 1460.23 212.378 1460.33 Q209.137 1460.4 206.485 1457.75 Q203.392 1454.65 203.883 1451.02 Q204.39 1447.37 208.499 1443.26 L212.722 1439.04 L212.427 1438.74 Q210.348 1436.66 207.844 1436.91 Q205.34 1437.12 202.868 1439.59 Q201.297 1441.17 200.184 1443.03 Q199.071 1444.9 198.449 1447.03 L195.666 1444.24 Q196.681 1441.95 197.925 1440.09 Q199.152 1438.2 200.626 1436.73 Q204.603 1432.75 208.63 1432.85 Q212.656 1432.95 216.846 1437.14 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M208.04 1415.93 L211.052 1412.91 L236.521 1438.38 L233.509 1441.4 L208.04 1415.93 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M224.621 1399.41 Q226.193 1405.37 228.812 1410.12 Q231.43 1414.86 235.211 1418.64 Q238.992 1422.42 243.772 1425.08 Q248.551 1427.7 254.477 1429.27 L251.858 1431.89 Q245.556 1430.49 240.613 1428.01 Q235.686 1425.5 232.02 1421.84 Q228.37 1418.19 225.882 1413.27 Q223.394 1408.36 222.002 1402.03 L224.621 1399.41 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M229.99 1394.04 L232.609 1391.42 Q238.927 1392.83 243.837 1395.32 Q248.764 1397.79 252.414 1401.44 Q256.081 1405.11 258.569 1410.05 Q261.073 1414.98 262.464 1421.28 L259.845 1423.9 Q258.274 1417.97 255.639 1413.21 Q253.004 1408.41 249.223 1404.63 Q245.442 1400.85 240.678 1398.25 Q235.932 1395.63 229.99 1394.04 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M1187.32 1611.28 L1182.58 1599.09 L1172.81 1616.92 L1165.9 1616.92 L1179.71 1591.71 L1173.92 1576.72 Q1172.36 1572.71 1167.46 1572.71 L1165.9 1572.71 L1165.9 1567.68 L1168.13 1567.74 Q1176.34 1567.96 1178.41 1573.28 L1183.12 1585.47 L1192.89 1567.64 L1199.8 1567.64 L1185.98 1592.85 L1191.78 1607.84 Q1193.34 1611.85 1198.24 1611.85 L1199.8 1611.85 L1199.8 1616.88 L1197.57 1616.82 Q1189.36 1616.6 1187.32 1611.28 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M1229.3 1573.72 L1270.11 1573.72 L1270.11 1579.07 L1229.3 1579.07 L1229.3 1573.72 M1229.3 1586.71 L1270.11 1586.71 L1270.11 1592.12 L1229.3 1592.12 L1229.3 1586.71 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M1304.77 1555.8 L1330.01 1555.8 L1330.01 1561.22 L1310.66 1561.22 L1310.66 1572.86 Q1312.06 1572.39 1313.46 1572.16 Q1314.86 1571.91 1316.26 1571.91 Q1324.22 1571.91 1328.86 1576.27 Q1333.51 1580.63 1333.51 1588.08 Q1333.51 1595.75 1328.74 1600.01 Q1323.96 1604.25 1315.27 1604.25 Q1312.28 1604.25 1309.16 1603.74 Q1306.07 1603.23 1302.76 1602.21 L1302.76 1595.75 Q1305.63 1597.31 1308.68 1598.07 Q1311.74 1598.84 1315.14 1598.84 Q1320.65 1598.84 1323.87 1595.94 Q1327.08 1593.04 1327.08 1588.08 Q1327.08 1583.11 1323.87 1580.22 Q1320.65 1577.32 1315.14 1577.32 Q1312.57 1577.32 1309.99 1577.89 Q1307.44 1578.47 1304.77 1579.68 L1304.77 1555.8 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M1359.93 1560.04 Q1354.96 1560.04 1352.45 1564.94 Q1349.97 1569.81 1349.97 1579.61 Q1349.97 1589.38 1352.45 1594.29 Q1354.96 1599.16 1359.93 1599.16 Q1364.92 1599.16 1367.41 1594.29 Q1369.92 1589.38 1369.92 1579.61 Q1369.92 1569.81 1367.41 1564.94 Q1364.92 1560.04 1359.93 1560.04 M1359.93 1554.95 Q1367.92 1554.95 1372.12 1561.28 Q1376.35 1567.58 1376.35 1579.61 Q1376.35 1591.61 1372.12 1597.95 Q1367.92 1604.25 1359.93 1604.25 Q1351.94 1604.25 1347.71 1597.95 Q1343.5 1591.61 1343.5 1579.61 Q1343.5 1567.58 1347.71 1561.28 Q1351.94 1554.95 1359.93 1554.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><polyline clip-path="url(#clip702)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="189.496,1341.81 2352.76,1341.81 "/>
<polyline clip-path="url(#clip702)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="189.496,1048.95 2352.76,1048.95 "/>
<polyline clip-path="url(#clip702)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="189.496,756.095 2352.76,756.095 "/>
<polyline clip-path="url(#clip702)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="189.496,463.238 2352.76,463.238 "/>
<polyline clip-path="url(#clip702)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="189.496,170.381 2352.76,170.381 "/>
<polyline clip-path="url(#clip700)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="189.496,1352.62 189.496,123.472 "/>
<polyline clip-path="url(#clip700)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="189.496,1341.81 208.394,1341.81 "/>
<polyline clip-path="url(#clip700)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="189.496,1048.95 208.394,1048.95 "/>
<polyline clip-path="url(#clip700)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="189.496,756.095 208.394,756.095 "/>
<polyline clip-path="url(#clip700)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="189.496,463.238 208.394,463.238 "/>
<polyline clip-path="url(#clip700)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="189.496,170.381 208.394,170.381 "/>
<path clip-path="url(#clip700)" d="M51.6634 1361.6 L59.3023 1361.6 L59.3023 1335.24 L50.9921 1336.9 L50.9921 1332.64 L59.256 1330.98 L63.9319 1330.98 L63.9319 1361.6 L71.5707 1361.6 L71.5707 1365.54 L51.6634 1365.54 L51.6634 1361.6 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M91.0151 1334.06 Q87.404 1334.06 85.5753 1337.62 Q83.7697 1341.16 83.7697 1348.29 Q83.7697 1355.4 85.5753 1358.96 Q87.404 1362.5 91.0151 1362.5 Q94.6493 1362.5 96.4548 1358.96 Q98.2835 1355.4 98.2835 1348.29 Q98.2835 1341.16 96.4548 1337.62 Q94.6493 1334.06 91.0151 1334.06 M91.0151 1330.35 Q96.8252 1330.35 99.8808 1334.96 Q102.959 1339.54 102.959 1348.29 Q102.959 1357.02 99.8808 1361.63 Q96.8252 1366.21 91.0151 1366.21 Q85.2049 1366.21 82.1262 1361.63 Q79.0707 1357.02 79.0707 1348.29 Q79.0707 1339.54 82.1262 1334.96 Q85.2049 1330.35 91.0151 1330.35 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M102.959 1324.45 L127.071 1324.45 L127.071 1327.65 L102.959 1327.65 L102.959 1324.45 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M145.71 1313.36 L136.118 1328.35 L145.71 1328.35 L145.71 1313.36 M144.713 1310.05 L149.49 1310.05 L149.49 1328.35 L153.496 1328.35 L153.496 1331.51 L149.49 1331.51 L149.49 1338.13 L145.71 1338.13 L145.71 1331.51 L133.033 1331.51 L133.033 1327.84 L144.713 1310.05 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M52.585 1068.74 L60.2238 1068.74 L60.2238 1042.38 L51.9137 1044.05 L51.9137 1039.79 L60.1776 1038.12 L64.8535 1038.12 L64.8535 1068.74 L72.4923 1068.74 L72.4923 1072.68 L52.585 1072.68 L52.585 1068.74 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M91.9366 1041.2 Q88.3255 1041.2 86.4969 1044.76 Q84.6913 1048.31 84.6913 1055.43 Q84.6913 1062.54 86.4969 1066.11 Q88.3255 1069.65 91.9366 1069.65 Q95.5709 1069.65 97.3764 1066.11 Q99.2051 1062.54 99.2051 1055.43 Q99.2051 1048.31 97.3764 1044.76 Q95.5709 1041.2 91.9366 1041.2 M91.9366 1037.5 Q97.7468 1037.5 100.802 1042.1 Q103.881 1046.68 103.881 1055.43 Q103.881 1064.16 100.802 1068.77 Q97.7468 1073.35 91.9366 1073.35 Q86.1265 1073.35 83.0478 1068.77 Q79.9923 1064.16 79.9923 1055.43 Q79.9923 1046.68 83.0478 1042.1 Q86.1265 1037.5 91.9366 1037.5 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M103.881 1031.6 L127.993 1031.6 L127.993 1034.79 L103.881 1034.79 L103.881 1031.6 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M147.703 1030.13 Q150.43 1030.71 151.954 1032.56 Q153.496 1034.4 153.496 1037.11 Q153.496 1041.26 150.637 1043.54 Q147.778 1045.82 142.512 1045.82 Q140.744 1045.82 138.863 1045.46 Q137.002 1045.12 135.008 1044.42 L135.008 1040.76 Q136.588 1041.68 138.469 1042.15 Q140.349 1042.62 142.399 1042.62 Q145.973 1042.62 147.835 1041.21 Q149.716 1039.8 149.716 1037.11 Q149.716 1034.62 147.966 1033.23 Q146.236 1031.82 143.133 1031.82 L139.86 1031.82 L139.86 1028.7 L143.283 1028.7 Q146.086 1028.7 147.571 1027.59 Q149.057 1026.46 149.057 1024.36 Q149.057 1022.19 147.515 1021.05 Q145.992 1019.88 143.133 1019.88 Q141.572 1019.88 139.785 1020.22 Q137.998 1020.56 135.854 1021.27 L135.854 1017.89 Q138.017 1017.28 139.898 1016.98 Q141.797 1016.68 143.471 1016.68 Q147.797 1016.68 150.317 1018.66 Q152.838 1020.61 152.838 1023.96 Q152.838 1026.29 151.502 1027.91 Q150.167 1029.51 147.703 1030.13 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M53.3561 775.888 L60.995 775.888 L60.995 749.522 L52.6848 751.189 L52.6848 746.93 L60.9487 745.263 L65.6246 745.263 L65.6246 775.888 L73.2634 775.888 L73.2634 779.823 L53.3561 779.823 L53.3561 775.888 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M92.7078 748.342 Q89.0967 748.342 87.268 751.907 Q85.4624 755.448 85.4624 762.578 Q85.4624 769.684 87.268 773.249 Q89.0967 776.791 92.7078 776.791 Q96.342 776.791 98.1475 773.249 Q99.9762 769.684 99.9762 762.578 Q99.9762 755.448 98.1475 751.907 Q96.342 748.342 92.7078 748.342 M92.7078 744.638 Q98.5179 744.638 101.573 749.245 Q104.652 753.828 104.652 762.578 Q104.652 771.305 101.573 775.911 Q98.5179 780.494 92.7078 780.494 Q86.8976 780.494 83.8189 775.911 Q80.7634 771.305 80.7634 762.578 Q80.7634 753.828 83.8189 749.245 Q86.8976 744.638 92.7078 744.638 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M104.652 738.739 L128.764 738.739 L128.764 741.937 L104.652 741.937 L104.652 738.739 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M140.236 749.215 L153.496 749.215 L153.496 752.413 L135.666 752.413 L135.666 749.215 Q137.829 746.977 141.553 743.216 Q145.296 739.435 146.255 738.344 Q148.079 736.294 148.794 734.884 Q149.527 733.454 149.527 732.081 Q149.527 729.843 147.948 728.433 Q146.387 727.022 143.866 727.022 Q142.08 727.022 140.086 727.643 Q138.111 728.263 135.854 729.524 L135.854 725.687 Q138.149 724.765 140.142 724.295 Q142.136 723.825 143.791 723.825 Q148.155 723.825 150.75 726.007 Q153.345 728.188 153.345 731.837 Q153.345 733.567 152.687 735.128 Q152.048 736.671 150.336 738.777 Q149.866 739.322 147.346 741.937 Q144.826 744.532 140.236 749.215 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M53.0552 483.031 L60.694 483.031 L60.694 456.665 L52.3839 458.332 L52.3839 454.073 L60.6477 452.406 L65.3236 452.406 L65.3236 483.031 L72.9625 483.031 L72.9625 486.966 L53.0552 486.966 L53.0552 483.031 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M92.4068 455.485 Q88.7957 455.485 86.967 459.05 Q85.1615 462.591 85.1615 469.721 Q85.1615 476.827 86.967 480.392 Q88.7957 483.934 92.4068 483.934 Q96.0411 483.934 97.8466 480.392 Q99.6753 476.827 99.6753 469.721 Q99.6753 462.591 97.8466 459.05 Q96.0411 455.485 92.4068 455.485 M92.4068 451.781 Q98.217 451.781 101.273 456.387 Q104.351 460.971 104.351 469.721 Q104.351 478.448 101.273 483.054 Q98.217 487.637 92.4068 487.637 Q86.5967 487.637 83.518 483.054 Q80.4625 478.448 80.4625 469.721 Q80.4625 460.971 83.518 456.387 Q86.5967 451.781 92.4068 451.781 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M104.351 445.882 L128.463 445.882 L128.463 449.08 L104.351 449.08 L104.351 445.882 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M137.321 456.358 L143.528 456.358 L143.528 434.936 L136.776 436.29 L136.776 432.83 L143.49 431.476 L147.289 431.476 L147.289 456.358 L153.496 456.358 L153.496 459.556 L137.321 459.556 L137.321 456.358 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M82.7903 190.174 L90.4291 190.174 L90.4291 163.808 L82.119 165.475 L82.119 161.216 L90.3828 159.549 L95.0587 159.549 L95.0587 190.174 L102.698 190.174 L102.698 194.109 L82.7903 194.109 L82.7903 190.174 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M122.142 162.628 Q118.531 162.628 116.702 166.193 Q114.897 169.734 114.897 176.864 Q114.897 183.97 116.702 187.535 Q118.531 191.077 122.142 191.077 Q125.776 191.077 127.582 187.535 Q129.41 183.97 129.41 176.864 Q129.41 169.734 127.582 166.193 Q125.776 162.628 122.142 162.628 M122.142 158.924 Q127.952 158.924 131.008 163.53 Q134.086 168.114 134.086 176.864 Q134.086 185.591 131.008 190.197 Q127.952 194.78 122.142 194.78 Q116.332 194.78 113.253 190.197 Q110.198 185.591 110.198 176.864 Q110.198 168.114 113.253 163.53 Q116.332 158.924 122.142 158.924 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M143.791 141.12 Q140.857 141.12 139.371 144.016 Q137.904 146.894 137.904 152.687 Q137.904 158.461 139.371 161.357 Q140.857 164.235 143.791 164.235 Q146.744 164.235 148.211 161.357 Q149.697 158.461 149.697 152.687 Q149.697 146.894 148.211 144.016 Q146.744 141.12 143.791 141.12 M143.791 138.111 Q148.512 138.111 150.994 141.854 Q153.496 145.577 153.496 152.687 Q153.496 159.777 150.994 163.52 Q148.512 167.244 143.791 167.244 Q139.07 167.244 136.569 163.52 Q134.086 159.777 134.086 152.687 Q134.086 145.577 136.569 141.854 Q139.07 138.111 143.791 138.111 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M772.196 12.096 L810.437 12.096 L810.437 18.9825 L780.379 18.9825 L780.379 36.8875 L809.181 36.8875 L809.181 43.7741 L780.379 43.7741 L780.379 65.6895 L811.166 65.6895 L811.166 72.576 L772.196 72.576 L772.196 12.096 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M862.005 45.1919 L862.005 72.576 L854.551 72.576 L854.551 45.4349 Q854.551 38.994 852.04 35.7938 Q849.528 32.5936 844.505 32.5936 Q838.469 32.5936 834.985 36.4419 Q831.502 40.2903 831.502 46.9338 L831.502 72.576 L824.007 72.576 L824.007 27.2059 L831.502 27.2059 L831.502 34.2544 Q834.175 30.163 837.78 28.1376 Q841.426 26.1121 846.166 26.1121 Q853.984 26.1121 857.994 30.9732 Q862.005 35.7938 862.005 45.1919 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M884.244 14.324 L884.244 27.2059 L899.597 27.2059 L899.597 32.9987 L884.244 32.9987 L884.244 57.6282 Q884.244 63.1779 885.743 64.7578 Q887.282 66.3376 891.941 66.3376 L899.597 66.3376 L899.597 72.576 L891.941 72.576 Q883.313 72.576 880.031 69.3758 Q876.75 66.1351 876.75 57.6282 L876.75 32.9987 L871.281 32.9987 L871.281 27.2059 L876.75 27.2059 L876.75 14.324 L884.244 14.324 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M930.02 49.7694 Q920.986 49.7694 917.502 51.8354 Q914.018 53.9013 914.018 58.8839 Q914.018 62.8538 916.611 65.2034 Q919.244 67.5124 923.741 67.5124 Q929.939 67.5124 933.665 63.1374 Q937.433 58.7219 937.433 51.4303 L937.433 49.7694 L930.02 49.7694 M944.886 46.6907 L944.886 72.576 L937.433 72.576 L937.433 65.6895 Q934.881 69.8214 931.073 71.8063 Q927.265 73.7508 921.756 73.7508 Q914.788 73.7508 910.656 69.8619 Q906.565 65.9325 906.565 59.3701 Q906.565 51.7138 911.669 47.825 Q916.814 43.9361 926.981 43.9361 L937.433 43.9361 L937.433 43.2069 Q937.433 38.0623 934.03 35.2672 Q930.668 32.4315 924.551 32.4315 Q920.662 32.4315 916.976 33.3632 Q913.289 34.295 909.887 36.1584 L909.887 29.2718 Q913.978 27.692 917.826 26.9223 Q921.675 26.1121 925.32 26.1121 Q935.164 26.1121 940.025 31.2163 Q944.886 36.3204 944.886 46.6907 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M997.953 45.1919 L997.953 72.576 L990.5 72.576 L990.5 45.4349 Q990.5 38.994 987.988 35.7938 Q985.476 32.5936 980.453 32.5936 Q974.417 32.5936 970.934 36.4419 Q967.45 40.2903 967.45 46.9338 L967.45 72.576 L959.956 72.576 L959.956 27.2059 L967.45 27.2059 L967.45 34.2544 Q970.123 30.163 973.729 28.1376 Q977.375 26.1121 982.114 26.1121 Q989.932 26.1121 993.943 30.9732 Q997.953 35.7938 997.953 45.1919 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M1042.68 49.3643 Q1042.68 41.2625 1039.31 36.8065 Q1035.99 32.3505 1029.96 32.3505 Q1023.96 32.3505 1020.6 36.8065 Q1017.28 41.2625 1017.28 49.3643 Q1017.28 57.4256 1020.6 61.8816 Q1023.96 66.3376 1029.96 66.3376 Q1035.99 66.3376 1039.31 61.8816 Q1042.68 57.4256 1042.68 49.3643 M1050.13 66.9452 Q1050.13 78.5308 1044.98 84.1616 Q1039.84 89.8329 1029.23 89.8329 Q1025.3 89.8329 1021.81 89.2252 Q1018.33 88.6581 1015.05 87.4428 L1015.05 80.1917 Q1018.33 81.9741 1021.53 82.8248 Q1024.73 83.6755 1028.05 83.6755 Q1035.38 83.6755 1039.03 79.8271 Q1042.68 76.0193 1042.68 68.282 L1042.68 64.5957 Q1040.37 68.6061 1036.76 70.5911 Q1033.16 72.576 1028.13 72.576 Q1019.79 72.576 1014.68 66.2161 Q1009.58 59.8562 1009.58 49.3643 Q1009.58 38.832 1014.68 32.472 Q1019.79 26.1121 1028.13 26.1121 Q1033.16 26.1121 1036.76 28.0971 Q1040.37 30.082 1042.68 34.0924 L1042.68 27.2059 L1050.13 27.2059 L1050.13 66.9452 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M1065.48 9.54393 L1072.94 9.54393 L1072.94 72.576 L1065.48 72.576 L1065.48 9.54393 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M1127.34 48.0275 L1127.34 51.6733 L1093.07 51.6733 Q1093.55 59.3701 1097.69 63.421 Q1101.86 67.4314 1109.27 67.4314 Q1113.57 67.4314 1117.58 66.3781 Q1121.63 65.3249 1125.6 63.2184 L1125.6 70.267 Q1121.59 71.9684 1117.37 72.8596 Q1113.16 73.7508 1108.83 73.7508 Q1097.97 73.7508 1091.61 67.4314 Q1085.29 61.1119 1085.29 50.3365 Q1085.29 39.1965 1091.29 32.6746 Q1097.32 26.1121 1107.53 26.1121 Q1116.69 26.1121 1121.99 32.0264 Q1127.34 37.9003 1127.34 48.0275 M1119.89 45.84 Q1119.8 39.7232 1116.44 36.0774 Q1113.12 32.4315 1107.61 32.4315 Q1101.37 32.4315 1097.61 35.9558 Q1093.88 39.4801 1093.31 45.8805 L1119.89 45.84 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M1174.9 35.9153 Q1177.69 30.8922 1181.58 28.5022 Q1185.47 26.1121 1190.74 26.1121 Q1197.82 26.1121 1201.67 31.0947 Q1205.52 36.0368 1205.52 45.1919 L1205.52 72.576 L1198.03 72.576 L1198.03 45.4349 Q1198.03 38.913 1195.72 35.7533 Q1193.41 32.5936 1188.67 32.5936 Q1182.88 32.5936 1179.51 36.4419 Q1176.15 40.2903 1176.15 46.9338 L1176.15 72.576 L1168.66 72.576 L1168.66 45.4349 Q1168.66 38.8725 1166.35 35.7533 Q1164.04 32.5936 1159.22 32.5936 Q1153.51 32.5936 1150.15 36.4824 Q1146.78 40.3308 1146.78 46.9338 L1146.78 72.576 L1139.29 72.576 L1139.29 27.2059 L1146.78 27.2059 L1146.78 34.2544 Q1149.34 30.082 1152.9 28.0971 Q1156.47 26.1121 1161.37 26.1121 Q1166.31 26.1121 1169.75 28.6237 Q1173.24 31.1352 1174.9 35.9153 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M1259.2 48.0275 L1259.2 51.6733 L1224.93 51.6733 Q1225.41 59.3701 1229.54 63.421 Q1233.72 67.4314 1241.13 67.4314 Q1245.42 67.4314 1249.43 66.3781 Q1253.48 65.3249 1257.45 63.2184 L1257.45 70.267 Q1253.44 71.9684 1249.23 72.8596 Q1245.02 73.7508 1240.68 73.7508 Q1229.83 73.7508 1223.47 67.4314 Q1217.15 61.1119 1217.15 50.3365 Q1217.15 39.1965 1223.14 32.6746 Q1229.18 26.1121 1239.39 26.1121 Q1248.54 26.1121 1253.85 32.0264 Q1259.2 37.9003 1259.2 48.0275 M1251.74 45.84 Q1251.66 39.7232 1248.3 36.0774 Q1244.98 32.4315 1239.47 32.4315 Q1233.23 32.4315 1229.46 35.9558 Q1225.74 39.4801 1225.17 45.8805 L1251.74 45.84 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M1309.14 45.1919 L1309.14 72.576 L1301.69 72.576 L1301.69 45.4349 Q1301.69 38.994 1299.18 35.7938 Q1296.67 32.5936 1291.64 32.5936 Q1285.61 32.5936 1282.12 36.4419 Q1278.64 40.2903 1278.64 46.9338 L1278.64 72.576 L1271.15 72.576 L1271.15 27.2059 L1278.64 27.2059 L1278.64 34.2544 Q1281.31 30.163 1284.92 28.1376 Q1288.57 26.1121 1293.3 26.1121 Q1301.12 26.1121 1305.13 30.9732 Q1309.14 35.7938 1309.14 45.1919 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M1331.38 14.324 L1331.38 27.2059 L1346.74 27.2059 L1346.74 32.9987 L1331.38 32.9987 L1331.38 57.6282 Q1331.38 63.1779 1332.88 64.7578 Q1334.42 66.3376 1339.08 66.3376 L1346.74 66.3376 L1346.74 72.576 L1339.08 72.576 Q1330.45 72.576 1327.17 69.3758 Q1323.89 66.1351 1323.89 57.6282 L1323.89 32.9987 L1318.42 32.9987 L1318.42 27.2059 L1323.89 27.2059 L1323.89 14.324 L1331.38 14.324 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M1419.49 14.0809 L1419.49 22.0612 Q1414.83 19.8332 1410.7 18.7395 Q1406.57 17.6457 1402.72 17.6457 Q1396.04 17.6457 1392.39 20.2383 Q1388.78 22.8309 1388.78 27.611 Q1388.78 31.6214 1391.17 33.6873 Q1393.61 35.7128 1400.33 36.9686 L1405.27 37.9813 Q1414.43 39.7232 1418.76 44.1387 Q1423.14 48.5136 1423.14 55.8863 Q1423.14 64.6767 1417.22 69.2137 Q1411.35 73.7508 1399.96 73.7508 Q1395.67 73.7508 1390.81 72.7785 Q1385.99 71.8063 1380.8 69.9024 L1380.8 61.4765 Q1385.79 64.2716 1390.57 65.6895 Q1395.35 67.1073 1399.96 67.1073 Q1406.97 67.1073 1410.78 64.3527 Q1414.59 61.598 1414.59 56.4939 Q1414.59 52.0379 1411.83 49.5264 Q1409.12 47.0148 1402.88 45.759 L1397.9 44.7868 Q1388.74 42.9639 1384.65 39.075 Q1380.56 35.1862 1380.56 28.2591 Q1380.56 20.2383 1386.19 15.6203 Q1391.86 11.0023 1401.79 11.0023 Q1406.04 11.0023 1410.46 11.7719 Q1414.87 12.5416 1419.49 14.0809 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M1442.78 65.7705 L1442.78 89.8329 L1435.29 89.8329 L1435.29 27.2059 L1442.78 27.2059 L1442.78 34.0924 Q1445.13 30.0415 1448.7 28.0971 Q1452.3 26.1121 1457.29 26.1121 Q1465.55 26.1121 1470.69 32.6746 Q1475.88 39.2371 1475.88 49.9314 Q1475.88 60.6258 1470.69 67.1883 Q1465.55 73.7508 1457.29 73.7508 Q1452.3 73.7508 1448.7 71.8063 Q1445.13 69.8214 1442.78 65.7705 M1468.14 49.9314 Q1468.14 41.7081 1464.74 37.0496 Q1461.38 32.3505 1455.46 32.3505 Q1449.55 32.3505 1446.15 37.0496 Q1442.78 41.7081 1442.78 49.9314 Q1442.78 58.1548 1446.15 62.8538 Q1449.55 67.5124 1455.46 67.5124 Q1461.38 67.5124 1464.74 62.8538 Q1468.14 58.1548 1468.14 49.9314 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M1527.04 48.0275 L1527.04 51.6733 L1492.77 51.6733 Q1493.26 59.3701 1497.39 63.421 Q1501.56 67.4314 1508.97 67.4314 Q1513.27 67.4314 1517.28 66.3781 Q1521.33 65.3249 1525.3 63.2184 L1525.3 70.267 Q1521.29 71.9684 1517.08 72.8596 Q1512.86 73.7508 1508.53 73.7508 Q1497.67 73.7508 1491.31 67.4314 Q1484.99 61.1119 1484.99 50.3365 Q1484.99 39.1965 1490.99 32.6746 Q1497.02 26.1121 1507.23 26.1121 Q1516.39 26.1121 1521.69 32.0264 Q1527.04 37.9003 1527.04 48.0275 M1519.59 45.84 Q1519.51 39.7232 1516.14 36.0774 Q1512.82 32.4315 1507.31 32.4315 Q1501.08 32.4315 1497.31 35.9558 Q1493.58 39.4801 1493.01 45.8805 L1519.59 45.84 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M1571.93 28.9478 L1571.93 35.9153 Q1568.77 34.1734 1565.57 33.3227 Q1562.41 32.4315 1559.17 32.4315 Q1551.91 32.4315 1547.9 37.0496 Q1543.89 41.6271 1543.89 49.9314 Q1543.89 58.2358 1547.9 62.8538 Q1551.91 67.4314 1559.17 67.4314 Q1562.41 67.4314 1565.57 66.5807 Q1568.77 65.6895 1571.93 63.9476 L1571.93 70.8341 Q1568.81 72.2924 1565.44 73.0216 Q1562.12 73.7508 1558.36 73.7508 Q1548.11 73.7508 1542.07 67.3098 Q1536.03 60.8689 1536.03 49.9314 Q1536.03 38.832 1542.11 32.472 Q1548.23 26.1121 1558.84 26.1121 Q1562.28 26.1121 1565.57 26.8413 Q1568.85 27.5299 1571.93 28.9478 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M1592.26 14.324 L1592.26 27.2059 L1607.61 27.2059 L1607.61 32.9987 L1592.26 32.9987 L1592.26 57.6282 Q1592.26 63.1779 1593.76 64.7578 Q1595.3 66.3376 1599.96 66.3376 L1607.61 66.3376 L1607.61 72.576 L1599.96 72.576 Q1591.33 72.576 1588.05 69.3758 Q1584.77 66.1351 1584.77 57.6282 L1584.77 32.9987 L1579.3 32.9987 L1579.3 27.2059 L1584.77 27.2059 L1584.77 14.324 L1592.26 14.324 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M1643.71 34.1734 Q1642.45 33.4443 1640.95 33.1202 Q1639.49 32.7556 1637.71 32.7556 Q1631.39 32.7556 1627.99 36.8875 Q1624.63 40.9789 1624.63 48.6757 L1624.63 72.576 L1617.13 72.576 L1617.13 27.2059 L1624.63 27.2059 L1624.63 34.2544 Q1626.98 30.1225 1630.74 28.1376 Q1634.51 26.1121 1639.9 26.1121 Q1640.67 26.1121 1641.6 26.2337 Q1642.53 26.3147 1643.67 26.5172 L1643.71 34.1734 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M1650.76 54.671 L1650.76 27.2059 L1658.21 27.2059 L1658.21 54.3874 Q1658.21 60.8284 1660.72 64.0691 Q1663.23 67.2693 1668.26 67.2693 Q1674.29 67.2693 1677.78 63.421 Q1681.3 59.5726 1681.3 52.9291 L1681.3 27.2059 L1688.75 27.2059 L1688.75 72.576 L1681.3 72.576 L1681.3 65.6084 Q1678.59 69.7404 1674.98 71.7658 Q1671.42 73.7508 1666.68 73.7508 Q1658.86 73.7508 1654.81 68.8897 Q1650.76 64.0286 1650.76 54.671 M1669.51 26.1121 L1669.51 26.1121 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip700)" d="M1739.43 35.9153 Q1742.23 30.8922 1746.11 28.5022 Q1750 26.1121 1755.27 26.1121 Q1762.36 26.1121 1766.21 31.0947 Q1770.06 36.0368 1770.06 45.1919 L1770.06 72.576 L1762.56 72.576 L1762.56 45.4349 Q1762.56 38.913 1760.25 35.7533 Q1757.94 32.5936 1753.2 32.5936 Q1747.41 32.5936 1744.05 36.4419 Q1740.69 40.2903 1740.69 46.9338 L1740.69 72.576 L1733.19 72.576 L1733.19 45.4349 Q1733.19 38.8725 1730.88 35.7533 Q1728.57 32.5936 1723.75 32.5936 Q1718.04 32.5936 1714.68 36.4824 Q1711.32 40.3308 1711.32 46.9338 L1711.32 72.576 L1703.82 72.576 L1703.82 27.2059 L1711.32 27.2059 L1711.32 34.2544 Q1713.87 30.082 1717.43 28.0971 Q1721 26.1121 1725.9 26.1121 Q1730.84 26.1121 1734.29 28.6237 Q1737.77 31.1352 1739.43 35.9153 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><circle clip-path="url(#clip702)" cx="454.801" cy="194.275" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="488.121" cy="301.402" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="521.44" cy="321.065" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="554.759" cy="337.908" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="588.079" cy="489.375" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="621.398" cy="510.013" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="654.718" cy="526.312" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="688.037" cy="536.665" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="721.356" cy="647.829" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="754.676" cy="668.475" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="787.995" cy="686.068" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="821.314" cy="723.71" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="854.634" cy="725.028" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="887.953" cy="745.765" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="921.273" cy="749.199" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="954.592" cy="762.836" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="987.911" cy="768.487" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="1021.23" cy="771.008" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="1054.55" cy="784.86" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="1087.87" cy="799.087" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="1121.19" cy="890.166" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="1154.51" cy="913.151" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="1187.83" cy="930.784" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="1221.15" cy="955.598" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="1254.47" cy="973.128" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="1287.79" cy="1001.16" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="1321.1" cy="1021.12" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="1354.42" cy="1034.73" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="1387.74" cy="1037.36" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="1421.06" cy="1050.7" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="1454.38" cy="1058.8" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="1487.7" cy="1065.19" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="1521.02" cy="1075.47" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="1554.34" cy="1084.74" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="1587.66" cy="1087.32" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="1620.98" cy="1098.5" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="1654.3" cy="1102.27" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="1687.62" cy="1121.17" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="1720.94" cy="1139.5" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="1754.26" cy="1180.64" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="1787.58" cy="1206.24" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="1820.9" cy="1226.47" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="1854.21" cy="1229" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="1887.53" cy="1244.38" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="1920.85" cy="1257.41" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="1954.17" cy="1260.35" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="1987.49" cy="1264.86" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="2020.81" cy="1278.08" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="2054.13" cy="1295.82" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip702)" cx="2087.45" cy="1317.83" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
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
└ @ MPSKit ~/Projects/Julia/MPSKit.jl/src/states/infinitemps.jl:149

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
[ Info: VUMPS init:	obj = +2.529209601096e-02	err = 3.8506e-01
[ Info: VUMPS   1:	obj = -8.700668137574e-01	err = 1.2760693513e-01	time = 0.09 sec
┌ Warning: ignoring imaginary component -9.671389572637824e-7 from total weight 1.071431546174058: operator might not be hermitian?
│   α = 0.5076925045006396 - 9.671389572637824e-7im
│   β₁ = 0.38537381617811456
│   β₂ = 0.8612205877788699
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1081642497353856e-6 from total weight 1.9191569889382531: operator might not be hermitian?
│   α = 1.2769155360134903 - 1.1081642497353856e-6im
│   β₁ = 0.8612205877788699
│   β₂ = 1.1449669695072051
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.1190854093148994e-6 from total weight 3.730590304906558: operator might not be hermitian?
│   α = 2.794800756698454 - 2.1190854093148994e-6im
│   β₁ = 1.664432597674601
│   β₂ = 1.8264875803613458
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.3849446146892266e-6 from total weight 4.097529275374596: operator might not be hermitian?
│   α = 3.177829412143231 - 2.3849446146892266e-6im
│   β₁ = 1.7562777461027703
│   β₂ = 1.8991142325852197
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -9.864978109262819e-7 from total weight 1.1361682375672237: operator might not be hermitian?
│   α = 0.6796442852991254 - 9.864978109262819e-7im
│   β₁ = 0.36574221428184467
│   β₂ = 0.8337832705254361
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.2560459516609994e-6 from total weight 2.0188159922504125: operator might not be hermitian?
│   α = 1.475196807523427 - 1.2560459516609994e-6im
│   β₁ = 0.8337832705254361
│   β₂ = 1.09736860144121
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.287798814792349e-6 from total weight 3.131391735738219: operator might not be hermitian?
│   α = 2.3816372787502273 - 2.287798814792349e-6im
│   β₁ = 1.398562278073576
│   β₂ = 1.4756156781022765
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.627322084186091e-6 from total weight 3.4851002789188974: operator might not be hermitian?
│   α = 2.7576707367226887 - 2.627322084186091e-6im
│   β₁ = 1.4756156781022765
│   β₂ = 1.537444123367018
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.7586626591682636e-6 from total weight 3.988678975499865: operator might not be hermitian?
│   α = 3.369986079025038 - 2.7586626591682636e-6im
│   β₁ = 1.537444123367018
│   β₂ = 1.479533495494006
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.576279468168019e-6 from total weight 4.408138403225203: operator might not be hermitian?
│   α = 3.756862474916753 - 2.576279468168019e-6im
│   β₁ = 1.479533495494006
│   β₂ = 1.7687987907774767
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.0838745975609863e-6 from total weight 3.6721874341141185: operator might not be hermitian?
│   α = 2.9242549058641267 - 2.0838745975609863e-6im
│   β₁ = 1.4510488349306505
│   β₂ = 1.6817107585543003
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -3.2011468193795557e-6 from total weight 3.928995034765234: operator might not be hermitian?
│   α = 3.2218982183839864 - 3.2011468193795557e-6im
│   β₁ = 1.6817107585543003
│   β₂ = 1.4927232758071491
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.6483490946991716e-6 from total weight 3.8342135775491344: operator might not be hermitian?
│   α = 3.147392732501323 - 2.6483490946991716e-6im
│   β₁ = 1.4927232758071491
│   β₂ = 1.6021516680753263
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -8.008456815874523e-7 from total weight 1.068648760572627: operator might not be hermitian?
│   α = 0.49256503829854587 - 8.008456815874523e-7im
│   β₁ = 0.3861387753335071
│   β₂ = 0.8661909158509234
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.1445368569925613e-6 from total weight 1.9110645467027934: operator might not be hermitian?
│   α = 1.2679109876128705 - 1.1445368569925613e-6im
│   β₁ = 0.8661909158509234
│   β₂ = 1.1376654721186439
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -1.8545564228642994e-6 from total weight 3.293944443070549: operator might not be hermitian?
│   α = 2.4420621357903936 - 1.8545564228642994e-6im
│   β₁ = 1.4944926110182144
│   β₂ = 1.6287708109438626
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.3486831544836106e-6 from total weight 3.712645223079187: operator might not be hermitian?
│   α = 2.775675517357043 - 2.3486831544836106e-6im
│   β₁ = 1.6287708109438626
│   β₂ = 1.851071478957302
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.5715279290162985e-6 from total weight 4.138742304007041: operator might not be hermitian?
│   α = 3.244964908413238 - 2.5715279290162985e-6im
│   β₁ = 1.851071478957302
│   β₂ = 1.7812706088431347
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.704545960600588e-6 from total weight 4.638773087532732: operator might not be hermitian?
│   α = 3.888714950195714 - 2.704545960600588e-6im
│   β₁ = 1.7812706088431347
│   β₂ = 1.7953235952904407
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -2.3470771421838996e-6 from total weight 4.105392145218653: operator might not be hermitian?
│   α = 3.336496533127281 - 2.3470771421838996e-6im
│   β₁ = 1.7117357192414597
│   β₂ = 1.670926802083221
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
┌ Warning: ignoring imaginary component -8.845749970285409e-7 from total weight 1.5036342874312476: operator might not be hermitian?
│   α = 1.170551214174138 - 8.845749970285409e-7im
│   β₁ = 0.38496445782055083
│   β₂ = 0.8617008132458002
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
[ Info: VUMPS   2:	obj = -8.856910840126e-01	err = 7.8578408649e-03	time = 0.08 sec
┌ Warning: ignoring imaginary component 2.7018785862182825e-7 from total weight 0.44194776924492263: operator might not be hermitian?
│   α = -0.07978677254444026 + 2.7018785862182825e-7im
│   β₁ = 0.02857112167935202
│   β₂ = 0.43374600018146076
└ @ KrylovKit ~/.julia/packages/KrylovKit/Y0zG7/src/factorizations/lanczos.jl:170
[ Info: VUMPS   3:	obj = -8.861118311158e-01	err = 4.0509339869e-03	time = 0.05 sec
[ Info: VUMPS   4:	obj = -8.862216799517e-01	err = 2.1307990266e-03	time = 0.06 sec
[ Info: VUMPS   5:	obj = -8.862597578596e-01	err = 1.2064832500e-03	time = 0.07 sec
[ Info: VUMPS   6:	obj = -8.862749304464e-01	err = 8.5365973329e-04	time = 0.09 sec
[ Info: VUMPS   7:	obj = -8.862815509334e-01	err = 6.3063221768e-04	time = 0.09 sec
[ Info: VUMPS   8:	obj = -8.862847899699e-01	err = 4.7867144673e-04	time = 0.10 sec
[ Info: VUMPS   9:	obj = -8.862863646071e-01	err = 3.8612691378e-04	time = 0.13 sec
[ Info: VUMPS  10:	obj = -8.862871410116e-01	err = 3.0291821377e-04	time = 0.11 sec
[ Info: VUMPS  11:	obj = -8.862875242967e-01	err = 2.3195751791e-04	time = 0.13 sec
[ Info: VUMPS  12:	obj = -8.862877135929e-01	err = 1.7414560667e-04	time = 0.16 sec
[ Info: VUMPS  13:	obj = -8.862878071048e-01	err = 1.2883734924e-04	time = 0.21 sec
[ Info: VUMPS  14:	obj = -8.862878532530e-01	err = 9.4253044026e-05	time = 0.10 sec
[ Info: VUMPS  15:	obj = -8.862878759981e-01	err = 6.8384197821e-05	time = 0.10 sec
[ Info: VUMPS  16:	obj = -8.862878872034e-01	err = 4.9321469248e-05	time = 0.12 sec
[ Info: VUMPS  17:	obj = -8.862878927221e-01	err = 3.5421334288e-05	time = 0.16 sec
[ Info: VUMPS  18:	obj = -8.862878954405e-01	err = 2.5359982562e-05	time = 0.14 sec
[ Info: VUMPS  19:	obj = -8.862878967805e-01	err = 1.8115338221e-05	time = 0.17 sec
[ Info: VUMPS  20:	obj = -8.862878974415e-01	err = 1.2918474214e-05	time = 0.16 sec
[ Info: VUMPS  21:	obj = -8.862878977679e-01	err = 9.2003371054e-06	time = 0.17 sec
[ Info: VUMPS  22:	obj = -8.862878979292e-01	err = 6.5462638559e-06	time = 0.26 sec
[ Info: VUMPS  23:	obj = -8.862878980090e-01	err = 4.6539903266e-06	time = 0.11 sec
[ Info: VUMPS  24:	obj = -8.862878980486e-01	err = 3.3063133163e-06	time = 0.14 sec
[ Info: VUMPS  25:	obj = -8.862878980681e-01	err = 2.3475142715e-06	time = 0.12 sec
[ Info: VUMPS  26:	obj = -8.862878980779e-01	err = 1.6659191513e-06	time = 0.16 sec
[ Info: VUMPS  27:	obj = -8.862878980827e-01	err = 1.1816853804e-06	time = 0.13 sec
[ Info: VUMPS  28:	obj = -8.862878980851e-01	err = 8.3791026298e-07	time = 0.14 sec
[ Info: VUMPS  29:	obj = -8.862878980863e-01	err = 5.9388578145e-07	time = 0.13 sec
[ Info: VUMPS  30:	obj = -8.862878980869e-01	err = 4.2078193645e-07	time = 0.13 sec
[ Info: VUMPS  31:	obj = -8.862878980871e-01	err = 2.9801452450e-07	time = 0.20 sec
[ Info: VUMPS  32:	obj = -8.862878980873e-01	err = 2.1104589939e-07	time = 0.14 sec
[ Info: VUMPS  33:	obj = -8.862878980874e-01	err = 1.4939769241e-07	time = 0.10 sec
[ Info: VUMPS  34:	obj = -8.862878980874e-01	err = 1.0573206754e-07	time = 0.11 sec
[ Info: VUMPS  35:	obj = -8.862878980874e-01	err = 7.4808317980e-08	time = 0.12 sec
[ Info: VUMPS  36:	obj = -8.862878980874e-01	err = 5.2917998892e-08	time = 0.10 sec
[ Info: VUMPS  37:	obj = -8.862878980874e-01	err = 3.7429304204e-08	time = 0.12 sec
[ Info: VUMPS  38:	obj = -8.862878980874e-01	err = 2.6464141713e-08	time = 0.13 sec
[ Info: VUMPS  39:	obj = -8.862878980874e-01	err = 1.8734153006e-08	time = 0.11 sec
[ Info: VUMPS  40:	obj = -8.862878980874e-01	err = 1.3233756785e-08	time = 0.10 sec
[ Info: VUMPS  41:	obj = -8.862878980874e-01	err = 9.3267037924e-09	time = 0.09 sec
[ Info: VUMPS  42:	obj = -8.862878980874e-01	err = 6.6095283192e-09	time = 0.18 sec
[ Info: VUMPS  43:	obj = -8.862878980874e-01	err = 4.6673057032e-09	time = 0.08 sec
[ Info: VUMPS  44:	obj = -8.862878980874e-01	err = 3.2968234716e-09	time = 0.07 sec
[ Info: VUMPS  45:	obj = -8.862878980874e-01	err = 2.3386800405e-09	time = 0.07 sec
[ Info: VUMPS  46:	obj = -8.862878980874e-01	err = 1.6622890895e-09	time = 0.07 sec
[ Info: VUMPS  47:	obj = -8.862878980874e-01	err = 1.1730161705e-09	time = 0.09 sec
[ Info: VUMPS  48:	obj = -8.862878980874e-01	err = 8.2342105674e-10	time = 0.10 sec
[ Info: VUMPS  49:	obj = -8.862878980874e-01	err = 5.8767379287e-10	time = 0.10 sec
[ Info: VUMPS  50:	obj = -8.862878980874e-01	err = 4.3377628724e-10	time = 0.08 sec
[ Info: VUMPS  51:	obj = -8.862878980874e-01	err = 2.9148815706e-10	time = 0.06 sec
[ Info: VUMPS  52:	obj = -8.862878980874e-01	err = 2.3530568367e-10	time = 0.08 sec
[ Info: VUMPS  53:	obj = -8.862878980874e-01	err = 1.9738411426e-10	time = 0.08 sec
[ Info: VUMPS  54:	obj = -8.862878980874e-01	err = 1.6903098032e-10	time = 0.06 sec
[ Info: VUMPS  55:	obj = -8.862878980874e-01	err = 1.0479313324e-10	time = 0.07 sec
[ Info: VUMPS  56:	obj = -8.862878980874e-01	err = 1.1793027010e-10	time = 0.05 sec
[ Info: VUMPS  57:	obj = -8.862878980874e-01	err = 8.7877060091e-11	time = 0.05 sec
[ Info: VUMPS  58:	obj = -8.862878980874e-01	err = 7.7838852527e-11	time = 0.05 sec
[ Info: VUMPS  59:	obj = -8.862878980874e-01	err = 7.7808315661e-11	time = 0.04 sec
[ Info: VUMPS  60:	obj = -8.862878980874e-01	err = 5.9890246741e-11	time = 0.05 sec
[ Info: VUMPS  61:	obj = -8.862878980874e-01	err = 5.9599058037e-11	time = 0.03 sec
[ Info: VUMPS  62:	obj = -8.862878980874e-01	err = 8.2169329602e-11	time = 0.06 sec
[ Info: VUMPS  63:	obj = -8.862878980874e-01	err = 1.9584055882e-10	time = 0.18 sec
[ Info: VUMPS  64:	obj = -8.862878980874e-01	err = 1.3190704587e-10	time = 0.08 sec
[ Info: VUMPS  65:	obj = -8.862878980874e-01	err = 9.4587810726e-11	time = 0.05 sec
[ Info: VUMPS  66:	obj = -8.862878980874e-01	err = 7.6642342387e-11	time = 0.06 sec
[ Info: VUMPS  67:	obj = -8.862878980874e-01	err = 7.6453977892e-11	time = 0.03 sec
[ Info: VUMPS  68:	obj = -8.862878980874e-01	err = 1.6135931629e-10	time = 0.03 sec
[ Info: VUMPS  69:	obj = -8.862878980874e-01	err = 4.8613631788e-11	time = 0.04 sec
[ Info: VUMPS  70:	obj = -8.862878980874e-01	err = 4.8715755949e-11	time = 0.03 sec
[ Info: VUMPS  71:	obj = -8.862878980874e-01	err = 4.9417123546e-11	time = 0.03 sec
[ Info: VUMPS  72:	obj = -8.862878980874e-01	err = 4.8238297283e-11	time = 0.03 sec
[ Info: VUMPS  73:	obj = -8.862878980874e-01	err = 4.9499679981e-11	time = 0.03 sec
[ Info: VUMPS  74:	obj = -8.862878980874e-01	err = 5.0219935378e-11	time = 0.03 sec
[ Info: VUMPS  75:	obj = -8.862878980874e-01	err = 4.8350451541e-11	time = 0.03 sec
[ Info: VUMPS  76:	obj = -8.862878980874e-01	err = 4.8479938582e-11	time = 0.03 sec
[ Info: VUMPS  77:	obj = -8.862878980874e-01	err = 4.8492803216e-11	time = 0.03 sec
[ Info: VUMPS  78:	obj = -8.862878980874e-01	err = 4.8444269610e-11	time = 0.03 sec
[ Info: VUMPS  79:	obj = -8.862878980874e-01	err = 5.2381270902e-11	time = 0.03 sec
[ Info: VUMPS  80:	obj = -8.862878980874e-01	err = 4.8563550538e-11	time = 0.03 sec
[ Info: VUMPS  81:	obj = -8.862878980874e-01	err = 5.0827468520e-11	time = 0.03 sec
[ Info: VUMPS  82:	obj = -8.862878980874e-01	err = 1.1400600324e-10	time = 0.06 sec
[ Info: VUMPS  83:	obj = -8.862878980874e-01	err = 1.9067343860e-10	time = 0.05 sec
[ Info: VUMPS  84:	obj = -8.862878980874e-01	err = 1.6689382966e-10	time = 0.05 sec
[ Info: VUMPS  85:	obj = -8.862878980874e-01	err = 9.3634468049e-11	time = 0.07 sec
[ Info: VUMPS  86:	obj = -8.862878980874e-01	err = 5.4612311876e-11	time = 0.04 sec
[ Info: VUMPS  87:	obj = -8.862878980874e-01	err = 5.9920739880e-11	time = 0.03 sec
[ Info: VUMPS  88:	obj = -8.862878980874e-01	err = 5.5300426287e-11	time = 0.03 sec
[ Info: VUMPS  89:	obj = -8.862878980874e-01	err = 5.4697291743e-11	time = 0.04 sec
[ Info: VUMPS  90:	obj = -8.862878980874e-01	err = 9.0893602359e-11	time = 0.06 sec
[ Info: VUMPS  91:	obj = -8.862878980874e-01	err = 9.1263591308e-11	time = 0.07 sec
[ Info: VUMPS  92:	obj = -8.862878980874e-01	err = 8.4998690637e-11	time = 0.06 sec
[ Info: VUMPS  93:	obj = -8.862878980874e-01	err = 1.1553836703e-10	time = 0.07 sec
[ Info: VUMPS  94:	obj = -8.862878980874e-01	err = 9.6747893284e-11	time = 0.07 sec
[ Info: VUMPS  95:	obj = -8.862878980874e-01	err = 1.1376915191e-10	time = 0.05 sec
[ Info: VUMPS  96:	obj = -8.862878980874e-01	err = 8.4905437150e-11	time = 0.05 sec
[ Info: VUMPS  97:	obj = -8.862878980874e-01	err = 1.3602624799e-10	time = 0.05 sec
[ Info: VUMPS  98:	obj = -8.862878980874e-01	err = 3.8537544337e-11	time = 0.05 sec
[ Info: VUMPS  99:	obj = -8.862878980874e-01	err = 3.8926546453e-11	time = 0.03 sec
[ Info: VUMPS 100:	obj = -8.862878980874e-01	err = 3.9274493760e-11	time = 0.03 sec
[ Info: VUMPS 101:	obj = -8.862878980874e-01	err = 3.8635555369e-11	time = 0.03 sec
[ Info: VUMPS 102:	obj = -8.862878980874e-01	err = 3.8807575779e-11	time = 0.03 sec
[ Info: VUMPS 103:	obj = -8.862878980874e-01	err = 3.8568495332e-11	time = 0.03 sec
[ Info: VUMPS 104:	obj = -8.862878980874e-01	err = 3.8509242746e-11	time = 0.03 sec
[ Info: VUMPS 105:	obj = -8.862878980874e-01	err = 3.8135791403e-11	time = 0.03 sec
[ Info: VUMPS 106:	obj = -8.862878980874e-01	err = 3.8362355119e-11	time = 0.03 sec
[ Info: VUMPS 107:	obj = -8.862878980874e-01	err = 3.9744912662e-11	time = 0.03 sec
[ Info: VUMPS 108:	obj = -8.862878980874e-01	err = 3.8554727985e-11	time = 0.03 sec
[ Info: VUMPS 109:	obj = -8.862878980874e-01	err = 3.8695253586e-11	time = 0.03 sec
[ Info: VUMPS 110:	obj = -8.862878980874e-01	err = 3.8520588629e-11	time = 0.03 sec
[ Info: VUMPS 111:	obj = -8.862878980874e-01	err = 3.8541288460e-11	time = 0.03 sec
[ Info: VUMPS 112:	obj = -8.862878980874e-01	err = 3.8249357381e-11	time = 0.03 sec
[ Info: VUMPS 113:	obj = -8.862878980874e-01	err = 3.8284823677e-11	time = 0.03 sec
[ Info: VUMPS 114:	obj = -8.862878980874e-01	err = 4.0318445321e-11	time = 0.03 sec
[ Info: VUMPS 115:	obj = -8.862878980874e-01	err = 3.8438479352e-11	time = 0.22 sec
[ Info: VUMPS 116:	obj = -8.862878980874e-01	err = 3.8957255842e-11	time = 0.03 sec
[ Info: VUMPS 117:	obj = -8.862878980874e-01	err = 3.8351906108e-11	time = 0.03 sec
[ Info: VUMPS 118:	obj = -8.862878980874e-01	err = 8.1169575478e-11	time = 0.05 sec
[ Info: VUMPS 119:	obj = -8.862878980874e-01	err = 1.4558819323e-10	time = 0.03 sec
[ Info: VUMPS 120:	obj = -8.862878980874e-01	err = 7.2695848580e-11	time = 0.03 sec
[ Info: VUMPS 121:	obj = -8.862878980874e-01	err = 6.3820080934e-11	time = 0.04 sec
[ Info: VUMPS 122:	obj = -8.862878980874e-01	err = 6.4468448875e-11	time = 0.03 sec
[ Info: VUMPS 123:	obj = -8.862878980874e-01	err = 3.5386680828e-10	time = 0.05 sec
[ Info: VUMPS 124:	obj = -8.862878980874e-01	err = 1.0528805822e-10	time = 0.09 sec
[ Info: VUMPS 125:	obj = -8.862878980874e-01	err = 1.0096381963e-10	time = 0.07 sec
[ Info: VUMPS 126:	obj = -8.862878980874e-01	err = 8.0559599886e-11	time = 0.04 sec
[ Info: VUMPS 127:	obj = -8.862878980874e-01	err = 1.0983328944e-10	time = 0.05 sec
[ Info: VUMPS 128:	obj = -8.862878980874e-01	err = 9.6899927046e-11	time = 0.06 sec
[ Info: VUMPS 129:	obj = -8.862878980874e-01	err = 9.1928629389e-11	time = 0.04 sec
[ Info: VUMPS 130:	obj = -8.862878980874e-01	err = 4.9719194648e-11	time = 0.04 sec
[ Info: VUMPS 131:	obj = -8.862878980874e-01	err = 4.2945823596e-11	time = 0.03 sec
[ Info: VUMPS 132:	obj = -8.862878980874e-01	err = 5.4051247432e-10	time = 0.06 sec
[ Info: VUMPS 133:	obj = -8.862878980874e-01	err = 1.4364385018e-10	time = 0.07 sec
[ Info: VUMPS 134:	obj = -8.862878980874e-01	err = 8.8389549749e-11	time = 0.04 sec
[ Info: VUMPS 135:	obj = -8.862878980874e-01	err = 8.7770612654e-11	time = 0.05 sec
[ Info: VUMPS 136:	obj = -8.862878980874e-01	err = 1.0755372350e-10	time = 0.07 sec
[ Info: VUMPS 137:	obj = -8.862878980874e-01	err = 9.1157795643e-11	time = 0.07 sec
[ Info: VUMPS 138:	obj = -8.862878980874e-01	err = 1.2395993089e-10	time = 0.04 sec
[ Info: VUMPS 139:	obj = -8.862878980874e-01	err = 5.8947156696e-11	time = 0.03 sec
[ Info: VUMPS 140:	obj = -8.862878980874e-01	err = 8.3233578587e-10	time = 0.05 sec
[ Info: VUMPS 141:	obj = -8.862878980874e-01	err = 1.1778486865e-10	time = 0.09 sec
[ Info: VUMPS 142:	obj = -8.862878980874e-01	err = 7.4548842297e-11	time = 0.04 sec
[ Info: VUMPS 143:	obj = -8.862878980874e-01	err = 7.7244247108e-11	time = 0.04 sec
[ Info: VUMPS 144:	obj = -8.862878980874e-01	err = 7.7255861317e-11	time = 0.03 sec
[ Info: VUMPS 145:	obj = -8.862878980874e-01	err = 7.7791688138e-11	time = 0.03 sec
[ Info: VUMPS 146:	obj = -8.862878980874e-01	err = 1.8573962248e-10	time = 0.04 sec
[ Info: VUMPS 147:	obj = -8.862878980874e-01	err = 7.3035341919e-11	time = 0.04 sec
[ Info: VUMPS 148:	obj = -8.862878980874e-01	err = 1.0634890052e-10	time = 0.04 sec
[ Info: VUMPS 149:	obj = -8.862878980874e-01	err = 4.7521976517e-11	time = 0.04 sec
[ Info: VUMPS 150:	obj = -8.862878980874e-01	err = 4.8276089374e-11	time = 0.03 sec
[ Info: VUMPS 151:	obj = -8.862878980874e-01	err = 4.7024688428e-11	time = 0.17 sec
[ Info: VUMPS 152:	obj = -8.862878980874e-01	err = 4.3748964822e-10	time = 0.04 sec
[ Info: VUMPS 153:	obj = -8.862878980874e-01	err = 9.2704143948e-11	time = 0.06 sec
[ Info: VUMPS 154:	obj = -8.862878980874e-01	err = 1.3725345226e-10	time = 0.05 sec
[ Info: VUMPS 155:	obj = -8.862878980874e-01	err = 7.8221525663e-11	time = 0.05 sec
[ Info: VUMPS 156:	obj = -8.862878980874e-01	err = 1.5274244678e-10	time = 0.03 sec
[ Info: VUMPS 157:	obj = -8.862878980874e-01	err = 5.2021209796e-11	time = 0.04 sec
[ Info: VUMPS 158:	obj = -8.862878980874e-01	err = 6.4744543971e-11	time = 0.03 sec
[ Info: VUMPS 159:	obj = -8.862878980874e-01	err = 9.3681465642e-11	time = 0.06 sec
[ Info: VUMPS 160:	obj = -8.862878980874e-01	err = 7.0751354204e-11	time = 0.04 sec
[ Info: VUMPS 161:	obj = -8.862878980874e-01	err = 1.1351374411e-10	time = 0.04 sec
[ Info: VUMPS 162:	obj = -8.862878980874e-01	err = 8.1558260001e-11	time = 0.05 sec
[ Info: VUMPS 163:	obj = -8.862878980874e-01	err = 1.1864028950e-10	time = 0.03 sec
[ Info: VUMPS 164:	obj = -8.862878980874e-01	err = 4.4746986769e-11	time = 0.03 sec
[ Info: VUMPS 165:	obj = -8.862878980874e-01	err = 4.3438831097e-11	time = 0.03 sec
[ Info: VUMPS 166:	obj = -8.862878980874e-01	err = 4.1460085149e-11	time = 0.03 sec
[ Info: VUMPS 167:	obj = -8.862878980874e-01	err = 4.4618208376e-11	time = 0.03 sec
[ Info: VUMPS 168:	obj = -8.862878980874e-01	err = 4.2337696098e-11	time = 0.03 sec
[ Info: VUMPS 169:	obj = -8.862878980874e-01	err = 4.1978118014e-11	time = 0.03 sec
[ Info: VUMPS 170:	obj = -8.862878980874e-01	err = 4.2214168317e-11	time = 0.03 sec
[ Info: VUMPS 171:	obj = -8.862878980874e-01	err = 4.1785244916e-11	time = 0.03 sec
[ Info: VUMPS 172:	obj = -8.862878980874e-01	err = 4.2141386221e-11	time = 0.04 sec
[ Info: VUMPS 173:	obj = -8.862878980874e-01	err = 4.1732296021e-11	time = 0.03 sec
[ Info: VUMPS 174:	obj = -8.862878980874e-01	err = 4.2256117546e-11	time = 0.04 sec
[ Info: VUMPS 175:	obj = -8.862878980874e-01	err = 4.1618346731e-11	time = 0.04 sec
[ Info: VUMPS 176:	obj = -8.862878980874e-01	err = 4.2266344901e-11	time = 0.04 sec
[ Info: VUMPS 177:	obj = -8.862878980874e-01	err = 9.8734315013e-11	time = 0.07 sec
[ Info: VUMPS 178:	obj = -8.862878980874e-01	err = 7.4589677259e-11	time = 0.07 sec
[ Info: VUMPS 179:	obj = -8.862878980874e-01	err = 7.9375086697e-11	time = 0.03 sec
[ Info: VUMPS 180:	obj = -8.862878980874e-01	err = 1.7247929023e-10	time = 0.07 sec
[ Info: VUMPS 181:	obj = -8.862878980874e-01	err = 8.6039641730e-11	time = 0.08 sec
[ Info: VUMPS 182:	obj = -8.862878980874e-01	err = 1.1202139192e-10	time = 0.05 sec
[ Info: VUMPS 183:	obj = -8.862878980874e-01	err = 8.0254748319e-11	time = 0.04 sec
[ Info: VUMPS 184:	obj = -8.862878980874e-01	err = 8.0049698591e-11	time = 0.03 sec
[ Info: VUMPS 185:	obj = -8.862878980874e-01	err = 8.0748438983e-11	time = 0.03 sec
[ Info: VUMPS 186:	obj = -8.862878980874e-01	err = 1.7758301363e-10	time = 0.04 sec
[ Info: VUMPS 187:	obj = -8.862878980874e-01	err = 7.9041173850e-11	time = 0.05 sec
[ Info: VUMPS 188:	obj = -8.862878980874e-01	err = 4.2753988414e-10	time = 0.06 sec
[ Info: VUMPS 189:	obj = -8.862878980874e-01	err = 1.4598945632e-10	time = 0.10 sec
[ Info: VUMPS 190:	obj = -8.862878980874e-01	err = 1.7166605554e-10	time = 0.07 sec
[ Info: VUMPS 191:	obj = -8.862878980874e-01	err = 1.2776212768e-10	time = 0.23 sec
[ Info: VUMPS 192:	obj = -8.862878980874e-01	err = 8.7435361404e-11	time = 0.06 sec
[ Info: VUMPS 193:	obj = -8.862878980874e-01	err = 1.6452219360e-10	time = 0.04 sec
[ Info: VUMPS 194:	obj = -8.862878980874e-01	err = 4.9424927706e-11	time = 0.04 sec
[ Info: VUMPS 195:	obj = -8.862878980874e-01	err = 1.6118096258e-10	time = 0.04 sec
[ Info: VUMPS 196:	obj = -8.862878980874e-01	err = 1.3946342996e-10	time = 0.05 sec
[ Info: VUMPS 197:	obj = -8.862878980874e-01	err = 1.1171007390e-10	time = 0.05 sec
[ Info: VUMPS 198:	obj = -8.862878980874e-01	err = 1.2122617840e-10	time = 0.05 sec
[ Info: VUMPS 199:	obj = -8.862878980874e-01	err = 9.6244952949e-11	time = 0.05 sec
[ Info: VUMPS 200:	obj = -8.862878980874e-01	err = 8.0799742424e-11	time = 0.06 sec
[ Info: VUMPS 201:	obj = -8.862878980874e-01	err = 5.1118445166e-11	time = 0.04 sec
[ Info: VUMPS 202:	obj = -8.862878980874e-01	err = 5.1326558160e-11	time = 0.03 sec
[ Info: VUMPS 203:	obj = -8.862878980874e-01	err = 5.0993858963e-11	time = 0.03 sec
[ Info: VUMPS 204:	obj = -8.862878980874e-01	err = 5.1037654449e-11	time = 0.03 sec
[ Info: VUMPS 205:	obj = -8.862878980874e-01	err = 5.3455271604e-11	time = 0.03 sec
[ Info: VUMPS 206:	obj = -8.862878980874e-01	err = 3.8366163414e-10	time = 0.06 sec
[ Info: VUMPS 207:	obj = -8.862878980874e-01	err = 1.4898391762e-10	time = 0.10 sec
[ Info: VUMPS 208:	obj = -8.862878980874e-01	err = 1.0419203578e-10	time = 0.06 sec
[ Info: VUMPS 209:	obj = -8.862878980874e-01	err = 1.3088789757e-10	time = 0.06 sec
[ Info: VUMPS 210:	obj = -8.862878980874e-01	err = 7.8891392613e-11	time = 0.07 sec
[ Info: VUMPS 211:	obj = -8.862878980874e-01	err = 1.0527138595e-10	time = 0.06 sec
[ Info: VUMPS 212:	obj = -8.862878980874e-01	err = 1.6479041684e-10	time = 0.05 sec
[ Info: VUMPS 213:	obj = -8.862878980874e-01	err = 8.2540083159e-11	time = 0.06 sec
[ Info: VUMPS 214:	obj = -8.862878980874e-01	err = 9.7164188376e-11	time = 0.04 sec
[ Info: VUMPS 215:	obj = -8.862878980874e-01	err = 1.3187308825e-10	time = 0.05 sec
[ Info: VUMPS 216:	obj = -8.862878980874e-01	err = 1.3425752903e-10	time = 0.06 sec
[ Info: VUMPS 217:	obj = -8.862878980874e-01	err = 9.7020802238e-11	time = 0.05 sec
[ Info: VUMPS 218:	obj = -8.862878980874e-01	err = 1.3853989912e-10	time = 0.05 sec
[ Info: VUMPS 219:	obj = -8.862878980874e-01	err = 7.8371664837e-11	time = 0.05 sec
[ Info: VUMPS 220:	obj = -8.862878980874e-01	err = 9.2483636476e-11	time = 0.18 sec
[ Info: VUMPS 221:	obj = -8.862878980874e-01	err = 1.6661668775e-10	time = 0.03 sec
[ Info: VUMPS 222:	obj = -8.862878980874e-01	err = 5.5364490079e-11	time = 0.03 sec
[ Info: VUMPS 223:	obj = -8.862878980874e-01	err = 5.5793836675e-11	time = 0.02 sec
[ Info: VUMPS 224:	obj = -8.862878980874e-01	err = 5.9912710624e-11	time = 0.03 sec
[ Info: VUMPS 225:	obj = -8.862878980874e-01	err = 5.5433095161e-11	time = 0.03 sec
[ Info: VUMPS 226:	obj = -8.862878980874e-01	err = 5.5259932758e-11	time = 0.03 sec
[ Info: VUMPS 227:	obj = -8.862878980874e-01	err = 5.5334515585e-11	time = 0.03 sec
[ Info: VUMPS 228:	obj = -8.862878980874e-01	err = 5.5367265928e-11	time = 0.03 sec
[ Info: VUMPS 229:	obj = -8.862878980874e-01	err = 5.5669228209e-11	time = 0.03 sec
[ Info: VUMPS 230:	obj = -8.862878980874e-01	err = 1.2870203660e-10	time = 0.06 sec
[ Info: VUMPS 231:	obj = -8.862878980874e-01	err = 9.5902953804e-11	time = 0.05 sec
[ Info: VUMPS 232:	obj = -8.862878980874e-01	err = 8.2657443795e-11	time = 0.04 sec
[ Info: VUMPS 233:	obj = -8.862878980874e-01	err = 1.1202671025e-10	time = 0.03 sec
[ Info: VUMPS 234:	obj = -8.862878980874e-01	err = 9.9921378101e-11	time = 0.03 sec
[ Info: VUMPS 235:	obj = -8.862878980874e-01	err = 5.2966240423e-11	time = 0.03 sec
[ Info: VUMPS 236:	obj = -8.862878980874e-01	err = 5.3006366140e-11	time = 0.03 sec
[ Info: VUMPS 237:	obj = -8.862878980874e-01	err = 5.4969988490e-11	time = 0.03 sec
[ Info: VUMPS 238:	obj = -8.862878980874e-01	err = 5.3036937147e-11	time = 0.03 sec
[ Info: VUMPS 239:	obj = -8.862878980874e-01	err = 5.2974854628e-11	time = 0.03 sec
[ Info: VUMPS 240:	obj = -8.862878980874e-01	err = 5.3066412887e-11	time = 0.03 sec
[ Info: VUMPS 241:	obj = -8.862878980874e-01	err = 5.2897434286e-11	time = 0.03 sec
[ Info: VUMPS 242:	obj = -8.862878980874e-01	err = 5.2813844342e-11	time = 0.03 sec
[ Info: VUMPS 243:	obj = -8.862878980874e-01	err = 5.2817999691e-11	time = 0.03 sec
[ Info: VUMPS 244:	obj = -8.862878980874e-01	err = 5.4379547470e-11	time = 0.03 sec
[ Info: VUMPS 245:	obj = -8.862878980874e-01	err = 5.2951047463e-11	time = 0.03 sec
[ Info: VUMPS 246:	obj = -8.862878980874e-01	err = 5.2946313572e-11	time = 0.03 sec
[ Info: VUMPS 247:	obj = -8.862878980874e-01	err = 5.2826255725e-11	time = 0.03 sec
[ Info: VUMPS 248:	obj = -8.862878980874e-01	err = 5.2832477486e-11	time = 0.03 sec
[ Info: VUMPS 249:	obj = -8.862878980874e-01	err = 5.2853933154e-11	time = 0.03 sec
[ Info: VUMPS 250:	obj = -8.862878980874e-01	err = 5.2941505326e-11	time = 0.03 sec
[ Info: VUMPS 251:	obj = -8.862878980874e-01	err = 5.3204273831e-11	time = 0.03 sec
[ Info: VUMPS 252:	obj = -8.862878980874e-01	err = 7.1352315398e-11	time = 0.05 sec
[ Info: VUMPS 253:	obj = -8.862878980874e-01	err = 7.2069779014e-11	time = 0.03 sec
[ Info: VUMPS 254:	obj = -8.862878980874e-01	err = 7.1435783262e-11	time = 0.03 sec
[ Info: VUMPS 255:	obj = -8.862878980874e-01	err = 2.6540427189e-10	time = 0.05 sec
[ Info: VUMPS 256:	obj = -8.862878980874e-01	err = 1.2459100212e-10	time = 0.08 sec
[ Info: VUMPS 257:	obj = -8.862878980874e-01	err = 7.9535019000e-11	time = 0.05 sec
[ Info: VUMPS 258:	obj = -8.862878980874e-01	err = 9.5195176352e-11	time = 0.05 sec
[ Info: VUMPS 259:	obj = -8.862878980874e-01	err = 9.2004952435e-11	time = 0.05 sec
[ Info: VUMPS 260:	obj = -8.862878980874e-01	err = 9.2983561462e-11	time = 0.04 sec
[ Info: VUMPS 261:	obj = -8.862878980874e-01	err = 8.3117224581e-11	time = 0.05 sec
[ Info: VUMPS 262:	obj = -8.862878980874e-01	err = 1.5819106604e-10	time = 0.04 sec
[ Info: VUMPS 263:	obj = -8.862878980874e-01	err = 8.4087291620e-11	time = 0.04 sec
[ Info: VUMPS 264:	obj = -8.862878980874e-01	err = 1.0037148070e-10	time = 0.04 sec
[ Info: VUMPS 265:	obj = -8.862878980874e-01	err = 1.0422725516e-10	time = 0.06 sec
[ Info: VUMPS 266:	obj = -8.862878980874e-01	err = 1.1122581785e-10	time = 0.04 sec
[ Info: VUMPS 267:	obj = -8.862878980874e-01	err = 5.5430916499e-11	time = 0.04 sec
[ Info: VUMPS 268:	obj = -8.862878980874e-01	err = 5.8977721798e-11	time = 0.03 sec
[ Info: VUMPS 269:	obj = -8.862878980874e-01	err = 5.5407226961e-11	time = 0.03 sec
[ Info: VUMPS 270:	obj = -8.862878980874e-01	err = 5.5421359078e-11	time = 0.21 sec
[ Info: VUMPS 271:	obj = -8.862878980874e-01	err = 5.5461650973e-11	time = 0.03 sec
[ Info: VUMPS 272:	obj = -8.862878980874e-01	err = 5.5516068411e-11	time = 0.03 sec
[ Info: VUMPS 273:	obj = -8.862878980874e-01	err = 5.5301390961e-11	time = 0.03 sec
[ Info: VUMPS 274:	obj = -8.862878980874e-01	err = 5.5628679726e-11	time = 0.03 sec
[ Info: VUMPS 275:	obj = -8.862878980874e-01	err = 5.7142294401e-11	time = 0.03 sec
[ Info: VUMPS 276:	obj = -8.862878980874e-01	err = 9.1958955056e-11	time = 0.05 sec
[ Info: VUMPS 277:	obj = -8.862878980874e-01	err = 1.6568749384e-10	time = 0.04 sec
[ Info: VUMPS 278:	obj = -8.862878980874e-01	err = 6.6761715141e-11	time = 0.04 sec
[ Info: VUMPS 279:	obj = -8.862878980874e-01	err = 7.0094294787e-11	time = 0.03 sec
[ Info: VUMPS 280:	obj = -8.862878980874e-01	err = 6.5066113456e-11	time = 0.03 sec
[ Info: VUMPS 281:	obj = -8.862878980874e-01	err = 6.5187520477e-11	time = 0.03 sec
[ Info: VUMPS 282:	obj = -8.862878980874e-01	err = 6.5033651663e-11	time = 0.03 sec
[ Info: VUMPS 283:	obj = -8.862878980874e-01	err = 6.4922953860e-11	time = 0.03 sec
[ Info: VUMPS 284:	obj = -8.862878980874e-01	err = 6.7560788576e-11	time = 0.03 sec
[ Info: VUMPS 285:	obj = -8.862878980874e-01	err = 6.5202786842e-11	time = 0.03 sec
[ Info: VUMPS 286:	obj = -8.862878980874e-01	err = 6.4740890521e-11	time = 0.03 sec
[ Info: VUMPS 287:	obj = -8.862878980874e-01	err = 6.4684719121e-11	time = 0.03 sec
[ Info: VUMPS 288:	obj = -8.862878980874e-01	err = 6.8395714074e-11	time = 0.03 sec
[ Info: VUMPS 289:	obj = -8.862878980874e-01	err = 6.5175379613e-11	time = 0.03 sec
[ Info: VUMPS 290:	obj = -8.862878980874e-01	err = 6.5151302835e-11	time = 0.03 sec
[ Info: VUMPS 291:	obj = -8.862878980874e-01	err = 8.9918749464e-10	time = 0.04 sec
[ Info: VUMPS 292:	obj = -8.862878980874e-01	err = 1.3012580604e-10	time = 0.07 sec
[ Info: VUMPS 293:	obj = -8.862878980874e-01	err = 1.3272650327e-10	time = 0.06 sec
[ Info: VUMPS 294:	obj = -8.862878980874e-01	err = 1.1097892363e-10	time = 0.05 sec
[ Info: VUMPS 295:	obj = -8.862878980874e-01	err = 1.1846768600e-10	time = 0.05 sec
[ Info: VUMPS 296:	obj = -8.862878980874e-01	err = 9.8873851293e-11	time = 0.06 sec
[ Info: VUMPS 297:	obj = -8.862878980874e-01	err = 8.6754908504e-11	time = 0.05 sec
[ Info: VUMPS 298:	obj = -8.862878980874e-01	err = 1.8386713998e-10	time = 0.04 sec
[ Info: VUMPS 299:	obj = -8.862878980874e-01	err = 4.9214682549e-11	time = 0.04 sec
[ Info: VUMPS 300:	obj = -8.862878980874e-01	err = 5.1220666282e-11	time = 0.03 sec
[ Info: VUMPS 301:	obj = -8.862878980874e-01	err = 4.9294368720e-11	time = 0.03 sec
[ Info: VUMPS 302:	obj = -8.862878980874e-01	err = 4.9394454081e-11	time = 0.04 sec
[ Info: VUMPS 303:	obj = -8.862878980874e-01	err = 5.0050931411e-11	time = 0.04 sec
[ Info: VUMPS 304:	obj = -8.862878980874e-01	err = 4.9485290287e-11	time = 0.04 sec
[ Info: VUMPS 305:	obj = -8.862878980874e-01	err = 4.9485117322e-11	time = 0.05 sec
[ Info: VUMPS 306:	obj = -8.862878980874e-01	err = 4.9400562584e-11	time = 0.04 sec
[ Info: VUMPS 307:	obj = -8.862878980874e-01	err = 8.0370473935e-11	time = 0.08 sec
[ Info: VUMPS 308:	obj = -8.862878980874e-01	err = 1.0500194811e-10	time = 0.08 sec
[ Info: VUMPS 309:	obj = -8.862878980874e-01	err = 7.8444943070e-11	time = 0.07 sec
[ Info: VUMPS 310:	obj = -8.862878980874e-01	err = 8.9319276364e-11	time = 0.06 sec
[ Info: VUMPS 311:	obj = -8.862878980874e-01	err = 5.9033763479e-11	time = 0.05 sec
[ Info: VUMPS 312:	obj = -8.862878980874e-01	err = 8.0174422677e-11	time = 0.07 sec
[ Info: VUMPS 313:	obj = -8.862878980874e-01	err = 8.0105274577e-11	time = 0.04 sec
[ Info: VUMPS 314:	obj = -8.862878980874e-01	err = 7.9879071897e-11	time = 0.04 sec
[ Info: VUMPS 315:	obj = -8.862878980874e-01	err = 6.7914930086e-11	time = 0.06 sec
[ Info: VUMPS 316:	obj = -8.862878980874e-01	err = 6.7644805100e-11	time = 0.04 sec
[ Info: VUMPS 317:	obj = -8.862878980874e-01	err = 6.7801565679e-11	time = 0.04 sec
[ Info: VUMPS 318:	obj = -8.862878980874e-01	err = 6.8325197783e-11	time = 0.21 sec
[ Info: VUMPS 319:	obj = -8.862878980874e-01	err = 6.7826120749e-11	time = 0.03 sec
[ Info: VUMPS 320:	obj = -8.862878980874e-01	err = 7.2621205313e-11	time = 0.03 sec
[ Info: VUMPS 321:	obj = -8.862878980874e-01	err = 3.8210726894e-10	time = 0.05 sec
[ Info: VUMPS 322:	obj = -8.862878980874e-01	err = 1.0235740670e-10	time = 0.09 sec
[ Info: VUMPS 323:	obj = -8.862878980874e-01	err = 9.6428191884e-11	time = 0.07 sec
[ Info: VUMPS 324:	obj = -8.862878980874e-01	err = 1.0034202926e-10	time = 0.07 sec
[ Info: VUMPS 325:	obj = -8.862878980874e-01	err = 7.0797901542e-11	time = 0.05 sec
[ Info: VUMPS 326:	obj = -8.862878980874e-01	err = 2.3373034775e-10	time = 0.06 sec
[ Info: VUMPS 327:	obj = -8.862878980874e-01	err = 9.8970620406e-11	time = 0.10 sec
[ Info: VUMPS 328:	obj = -8.862878980874e-01	err = 1.3577761530e-10	time = 0.08 sec
[ Info: VUMPS 329:	obj = -8.862878980874e-01	err = 1.0567516615e-10	time = 0.08 sec
[ Info: VUMPS 330:	obj = -8.862878980874e-01	err = 7.2032452782e-11	time = 0.07 sec
[ Info: VUMPS 331:	obj = -8.862878980874e-01	err = 5.2541282385e-11	time = 0.05 sec
[ Info: VUMPS 332:	obj = -8.862878980874e-01	err = 5.2821022372e-11	time = 0.04 sec
[ Info: VUMPS 333:	obj = -8.862878980874e-01	err = 5.4626716043e-11	time = 0.04 sec
[ Info: VUMPS 334:	obj = -8.862878980874e-01	err = 2.7830125061e-10	time = 0.05 sec
[ Info: VUMPS 335:	obj = -8.862878980874e-01	err = 9.9239900298e-11	time = 0.08 sec
[ Info: VUMPS 336:	obj = -8.862878980874e-01	err = 1.0349176036e-10	time = 0.06 sec
[ Info: VUMPS 337:	obj = -8.862878980874e-01	err = 1.0452650046e-10	time = 0.06 sec
[ Info: VUMPS 338:	obj = -8.862878980874e-01	err = 1.4555221521e-10	time = 0.07 sec
[ Info: VUMPS 339:	obj = -8.862878980874e-01	err = 1.2059414831e-10	time = 0.07 sec
[ Info: VUMPS 340:	obj = -8.862878980874e-01	err = 1.4914061305e-10	time = 0.07 sec
[ Info: VUMPS 341:	obj = -8.862878980874e-01	err = 1.2389694233e-10	time = 0.07 sec
[ Info: VUMPS 342:	obj = -8.862878980874e-01	err = 1.1053522260e-10	time = 0.07 sec
[ Info: VUMPS 343:	obj = -8.862878980874e-01	err = 1.0533349208e-10	time = 0.06 sec
[ Info: VUMPS 344:	obj = -8.862878980874e-01	err = 5.5087727545e-11	time = 0.05 sec
[ Info: VUMPS 345:	obj = -8.862878980874e-01	err = 4.7044124461e-10	time = 0.07 sec
[ Info: VUMPS 346:	obj = -8.862878980874e-01	err = 7.7993212022e-11	time = 0.11 sec
[ Info: VUMPS 347:	obj = -8.862878980874e-01	err = 9.9188349166e-11	time = 0.19 sec
[ Info: VUMPS 348:	obj = -8.862878980874e-01	err = 9.0546548017e-11	time = 0.06 sec
[ Info: VUMPS 349:	obj = -8.862878980874e-01	err = 1.1229111874e-10	time = 0.04 sec
[ Info: VUMPS 350:	obj = -8.862878980874e-01	err = 9.4673378845e-11	time = 0.06 sec
[ Info: VUMPS 351:	obj = -8.862878980874e-01	err = 1.0171542217e-10	time = 0.04 sec
[ Info: VUMPS 352:	obj = -8.862878980874e-01	err = 1.4319624203e-10	time = 0.04 sec
[ Info: VUMPS 353:	obj = -8.862878980874e-01	err = 4.4804644943e-11	time = 0.04 sec
[ Info: VUMPS 354:	obj = -8.862878980874e-01	err = 4.5253009904e-11	time = 0.03 sec
[ Info: VUMPS 355:	obj = -8.862878980874e-01	err = 4.8526941357e-11	time = 0.03 sec
[ Info: VUMPS 356:	obj = -8.862878980874e-01	err = 4.5267178026e-11	time = 0.03 sec
[ Info: VUMPS 357:	obj = -8.862878980874e-01	err = 5.3269395139e-11	time = 0.03 sec
[ Info: VUMPS 358:	obj = -8.862878980874e-01	err = 6.9373720406e-11	time = 0.06 sec
[ Info: VUMPS 359:	obj = -8.862878980874e-01	err = 1.6373642528e-10	time = 0.05 sec
[ Info: VUMPS 360:	obj = -8.862878980874e-01	err = 4.8545728886e-11	time = 0.04 sec
[ Info: VUMPS 361:	obj = -8.862878980874e-01	err = 4.8166216362e-11	time = 0.03 sec
[ Info: VUMPS 362:	obj = -8.862878980874e-01	err = 4.8091531818e-11	time = 0.03 sec
[ Info: VUMPS 363:	obj = -8.862878980874e-01	err = 8.5244170558e-11	time = 0.06 sec
[ Info: VUMPS 364:	obj = -8.862878980874e-01	err = 1.0011641995e-10	time = 0.05 sec
[ Info: VUMPS 365:	obj = -8.862878980874e-01	err = 1.0253292457e-10	time = 0.05 sec
[ Info: VUMPS 366:	obj = -8.862878980874e-01	err = 1.1336554420e-10	time = 0.05 sec
[ Info: VUMPS 367:	obj = -8.862878980874e-01	err = 7.8522322860e-11	time = 0.05 sec
[ Info: VUMPS 368:	obj = -8.862878980874e-01	err = 2.1253946297e-10	time = 0.04 sec
[ Info: VUMPS 369:	obj = -8.862878980874e-01	err = 6.1130038172e-11	time = 0.04 sec
[ Info: VUMPS 370:	obj = -8.862878980874e-01	err = 6.1383942312e-11	time = 0.03 sec
[ Info: VUMPS 371:	obj = -8.862878980874e-01	err = 6.1510474358e-11	time = 0.03 sec
[ Info: VUMPS 372:	obj = -8.862878980874e-01	err = 6.1809499146e-11	time = 0.03 sec
[ Info: VUMPS 373:	obj = -8.862878980874e-01	err = 6.1698467469e-11	time = 0.03 sec
[ Info: VUMPS 374:	obj = -8.862878980874e-01	err = 6.2385550089e-11	time = 0.03 sec
[ Info: VUMPS 375:	obj = -8.862878980874e-01	err = 2.3629545490e-10	time = 0.05 sec
[ Info: VUMPS 376:	obj = -8.862878980874e-01	err = 1.0499986243e-10	time = 0.08 sec
[ Info: VUMPS 377:	obj = -8.862878980874e-01	err = 7.2603816424e-11	time = 0.06 sec
[ Info: VUMPS 378:	obj = -8.862878980874e-01	err = 7.2618849446e-11	time = 0.04 sec
[ Info: VUMPS 379:	obj = -8.862878980874e-01	err = 7.7260293051e-11	time = 0.03 sec
[ Info: VUMPS 380:	obj = -8.862878980874e-01	err = 1.2108909054e-10	time = 0.07 sec
[ Info: VUMPS 381:	obj = -8.862878980874e-01	err = 1.0656096009e-10	time = 0.08 sec
[ Info: VUMPS 382:	obj = -8.862878980874e-01	err = 7.6401106319e-11	time = 0.05 sec
[ Info: VUMPS 383:	obj = -8.862878980874e-01	err = 1.3364758610e-10	time = 0.05 sec
[ Info: VUMPS 384:	obj = -8.862878980874e-01	err = 6.9742993842e-11	time = 0.05 sec
[ Info: VUMPS 385:	obj = -8.862878980874e-01	err = 1.7345659259e-10	time = 0.05 sec
[ Info: VUMPS 386:	obj = -8.862878980874e-01	err = 7.7533324291e-11	time = 0.04 sec
[ Info: VUMPS 387:	obj = -8.862878980874e-01	err = 1.4396981605e-10	time = 0.06 sec
[ Info: VUMPS 388:	obj = -8.862878980874e-01	err = 1.5404124820e-10	time = 0.05 sec
[ Info: VUMPS 389:	obj = -8.862878980874e-01	err = 8.9440098189e-11	time = 0.07 sec
[ Info: VUMPS 390:	obj = -8.862878980874e-01	err = 5.7181420413e-11	time = 0.04 sec
[ Info: VUMPS 391:	obj = -8.862878980874e-01	err = 5.9548328290e-10	time = 0.21 sec
[ Info: VUMPS 392:	obj = -8.862878980874e-01	err = 1.5282754945e-10	time = 0.07 sec
[ Info: VUMPS 393:	obj = -8.862878980874e-01	err = 1.5131412340e-10	time = 0.05 sec
[ Info: VUMPS 394:	obj = -8.862878980874e-01	err = 1.4049801953e-10	time = 0.04 sec
[ Info: VUMPS 395:	obj = -8.862878980874e-01	err = 6.3217689116e-11	time = 0.04 sec
[ Info: VUMPS 396:	obj = -8.862878980874e-01	err = 6.3150855078e-11	time = 0.03 sec
[ Info: VUMPS 397:	obj = -8.862878980874e-01	err = 6.3381605325e-11	time = 0.03 sec
[ Info: VUMPS 398:	obj = -8.862878980874e-01	err = 6.3282895468e-11	time = 0.03 sec
[ Info: VUMPS 399:	obj = -8.862878980874e-01	err = 6.3281276721e-11	time = 0.03 sec
┌ Warning: VUMPS cancel 400:	obj = -8.862878980874e-01	err = 6.3702904878e-11	time = 23.07 sec
└ @ MPSKit ~/Projects/Julia/MPSKit.jl/src/algorithms/groundstate/vumps.jl:67

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

