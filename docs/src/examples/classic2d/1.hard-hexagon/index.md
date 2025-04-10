```@meta
EditURL = "../../../../../examples/classic2d/1.hard-hexagon/main.jl"
```

[![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QuantumKitHub/MPSKit.jl/gh-pages?filepath=dev/examples/classic2d/1.hard-hexagon/main.ipynb)
[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](https://nbviewer.jupyter.org/github/QuantumKitHub/MPSKit.jl/blob/gh-pages/dev/examples/classic2d/1.hard-hexagon/main.ipynb)
[![](https://img.shields.io/badge/download-project-orange)](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/QuantumKitHub/MPSKit.jl/examples/tree/gh-pages/dev/examples/classic2d/1.hard-hexagon)

# [The Hard Hexagon model](@id demo_hardhexagon)

![logo](hexagon.svg)

Tensor networks are a natural way to do statistical mechanics on a lattice.
As an example of this we will extract the central charge of the hard hexagon model.
This model is known to have central charge 0.8, and has very peculiar non-local (anyonic) symmetries.
Because TensorKit supports anyonic symmetries, so does MPSKit.
To follow the tutorial you need the following packages.

````julia
using MPSKit, MPSKitModels, TensorKit, Plots, Polynomials
````

````
Precompiling Plots...
   4016.9 ms  ✓ Latexify → SparseArraysExt
   3805.8 ms  ✓ Wayland_jll
   4676.6 ms  ✓ FFMPEG_jll
   3098.7 ms  ✓ xkbcommon_jll
   3503.2 ms  ✓ FFMPEG
   9148.5 ms  ✓ OpenSSL
   3644.7 ms  ✓ Vulkan_Loader_jll
   3906.8 ms  ✓ libdecor_jll
  14001.3 ms  ✓ PlotThemes
   3774.9 ms  ✓ GLFW_jll
   4336.4 ms  ✓ Qt6Base_jll
  18159.7 ms  ✓ RecipesPipeline
   4039.3 ms  ✓ Qt6ShaderTools_jll
   4439.9 ms  ✓ GR_jll
   2870.3 ms  ✓ Qt6Declarative_jll
   1550.2 ms  ✓ Qt6Wayland_jll
  31118.7 ms  ✓ HTTP
   4397.9 ms  ✓ GR
 101636.2 ms  ✓ Plots
   4345.4 ms  ✓ Plots → UnitfulExt
  20 dependencies successfully precompiled in 153 seconds. 167 already precompiled.

````

The [hard hexagon model](https://en.wikipedia.org/wiki/Hard_hexagon_model) is a 2-dimensional lattice model of a gas, where particles are allowed to be on the vertices of a triangular lattice, but no two particles may be adjacent.
This can be encoded in a transfer matrix with a local MPO tensor using anyonic symmetries, and the resulting MPO has been implemented in MPSKitModels.

In order to use these anyonic symmetries, we need to generalise the notion of the bond dimension and define how it interacts with the symmetry. Thus, we implement away of converting integers to symmetric spaces of the given dimension, which provides a crude guess for how the final MPS would distribute its Schmidt spectrum.

````julia
mpo = hard_hexagon()
P = physicalspace(mpo, 1)
function virtual_space(D::Integer)
    _D = round(Int, D / sum(dim, values(FibonacciAnyon)))
    return Vect[FibonacciAnyon](sector => _D for sector in (:I, :τ))
end

@assert isapprox(dim(virtual_space(100)), 100; atol=3)
````

## The leading boundary

One way to study statistical mechanics in infinite systems with tensor networks is by approximating the dominant eigenvector of the transfer matrix by an MPS.
This dominant eigenvector contains a lot of hidden information.
For example, the free energy can be extracted by computing the expectation value of the mpo.
Additionally, we can compute the entanglement entropy as well as the correlation length of the state:

````julia
D = 10
V = virtual_space(D)
ψ₀ = InfiniteMPS([P], [V])
ψ, envs, = leading_boundary(ψ₀, mpo,
                            VUMPS(; verbosity=0,
                                  alg_eigsolve=MPSKit.Defaults.alg_eigsolve(;
                                                                            ishermitian=false))) # use non-hermitian eigensolver
F = real(expectation_value(ψ, mpo))
S = real(first(entropy(ψ)))
ξ = correlation_length(ψ)
println("F = $F\tS = $S\tξ = $ξ")
````

````
F = 0.8839037051703845	S = 1.28078296220416	ξ = 13.849682582836742

````

## The scaling hypothesis

The dominant eigenvector is of course only an approximation. The finite bond dimension enforces a finite correlation length, which effectively introduces a length scale in the system. This can be exploited to formulate a scaling hypothesis [pollmann2009](@cite), which in turn allows to extract the central charge.

First we need to know the entropy and correlation length at a bunch of different bond dimensions. Our approach will be to re-use the previous approximated dominant eigenvector, and then expanding its bond dimension and re-running VUMPS.
According to the scaling hypothesis we should have ``S \propto \frac{c}{6} log(ξ)``. Therefore we should find ``c`` using

````julia
function scaling_simulations(ψ₀, mpo, Ds; verbosity=0, tol=1e-6,
                             alg_eigsolve=MPSKit.Defaults.alg_eigsolve(; ishermitian=false))
    entropies = similar(Ds, Float64)
    correlations = similar(Ds, Float64)
    alg = VUMPS(; verbosity, tol, alg_eigsolve)

    ψ, envs, = leading_boundary(ψ₀, mpo, alg)
    entropies[1] = real(entropy(ψ)[1])
    correlations[1] = correlation_length(ψ)

    for (i, d) in enumerate(diff(Ds))
        ψ, envs = changebonds(ψ, mpo, OptimalExpand(; trscheme=truncdim(d)), envs)
        ψ, envs, = leading_boundary(ψ, mpo, alg, envs)
        entropies[i + 1] = real(entropy(ψ)[1])
        correlations[i + 1] = correlation_length(ψ)
    end
    return entropies, correlations
end

bond_dimensions = 10:5:25
ψ₀ = InfiniteMPS([P], [virtual_space(bond_dimensions[1])])
Ss, ξs = scaling_simulations(ψ₀, mpo, bond_dimensions)

f = fit(log.(ξs), 6 * Ss, 1)
c = f.coeffs[2]
````

````
0.8025415402364622
````

````julia
p = plot(; xlabel="logarithmic correlation length", ylabel="entanglement entropy")
p = plot(log.(ξs), Ss; seriestype=:scatter, label=nothing)
plot!(p, ξ -> f(ξ) / 6; label="fit")
````

```@raw html
<?xml version="1.0" encoding="utf-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="600" height="400" viewBox="0 0 2400 1600">
<defs>
  <clipPath id="clip290">
    <rect x="0" y="0" width="2400" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip290)" d="M0 1600 L2400 1600 L2400 0 L0 0  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip291">
    <rect x="480" y="0" width="1681" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip290)" d="M184.191 1486.45 L2352.76 1486.45 L2352.76 47.2441 L184.191 47.2441  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip292">
    <rect x="184" y="47" width="2170" height="1440"/>
  </clipPath>
</defs>
<polyline clip-path="url(#clip292)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="401.769,1486.45 401.769,47.2441 "/>
<polyline clip-path="url(#clip292)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="722.519,1486.45 722.519,47.2441 "/>
<polyline clip-path="url(#clip292)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1043.27,1486.45 1043.27,47.2441 "/>
<polyline clip-path="url(#clip292)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1364.02,1486.45 1364.02,47.2441 "/>
<polyline clip-path="url(#clip292)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1684.77,1486.45 1684.77,47.2441 "/>
<polyline clip-path="url(#clip292)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="2005.52,1486.45 2005.52,47.2441 "/>
<polyline clip-path="url(#clip292)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="2326.27,1486.45 2326.27,47.2441 "/>
<polyline clip-path="url(#clip292)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="184.191,1324.01 2352.76,1324.01 "/>
<polyline clip-path="url(#clip292)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="184.191,1007.4 2352.76,1007.4 "/>
<polyline clip-path="url(#clip292)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="184.191,690.779 2352.76,690.779 "/>
<polyline clip-path="url(#clip292)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="184.191,374.162 2352.76,374.162 "/>
<polyline clip-path="url(#clip292)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="184.191,57.5442 2352.76,57.5442 "/>
<polyline clip-path="url(#clip290)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="184.191,1486.45 2352.76,1486.45 "/>
<polyline clip-path="url(#clip290)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="401.769,1486.45 401.769,1467.55 "/>
<polyline clip-path="url(#clip290)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="722.519,1486.45 722.519,1467.55 "/>
<polyline clip-path="url(#clip290)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1043.27,1486.45 1043.27,1467.55 "/>
<polyline clip-path="url(#clip290)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1364.02,1486.45 1364.02,1467.55 "/>
<polyline clip-path="url(#clip290)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1684.77,1486.45 1684.77,1467.55 "/>
<polyline clip-path="url(#clip290)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="2005.52,1486.45 2005.52,1467.55 "/>
<polyline clip-path="url(#clip290)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="2326.27,1486.45 2326.27,1467.55 "/>
<path clip-path="url(#clip290)" d="M358.424 1544.91 L374.744 1544.91 L374.744 1548.85 L352.799 1548.85 L352.799 1544.91 Q355.461 1542.16 360.045 1537.53 Q364.651 1532.88 365.832 1531.53 Q368.077 1529.01 368.957 1527.27 Q369.859 1525.51 369.859 1523.82 Q369.859 1521.07 367.915 1519.33 Q365.994 1517.6 362.892 1517.6 Q360.693 1517.6 358.239 1518.36 Q355.808 1519.13 353.031 1520.68 L353.031 1515.95 Q355.855 1514.82 358.308 1514.24 Q360.762 1513.66 362.799 1513.66 Q368.17 1513.66 371.364 1516.35 Q374.558 1519.03 374.558 1523.52 Q374.558 1525.65 373.748 1527.57 Q372.961 1529.47 370.855 1532.07 Q370.276 1532.74 367.174 1535.95 Q364.072 1539.15 358.424 1544.91 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M384.558 1542.97 L389.443 1542.97 L389.443 1548.85 L384.558 1548.85 L384.558 1542.97 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M398.447 1514.29 L420.669 1514.29 L420.669 1516.28 L408.123 1548.85 L403.239 1548.85 L415.044 1518.22 L398.447 1518.22 L398.447 1514.29 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M429.836 1514.29 L448.192 1514.29 L448.192 1518.22 L434.118 1518.22 L434.118 1526.7 Q435.137 1526.35 436.155 1526.19 Q437.174 1526 438.192 1526 Q443.979 1526 447.359 1529.17 Q450.739 1532.34 450.739 1537.76 Q450.739 1543.34 447.266 1546.44 Q443.794 1549.52 437.475 1549.52 Q435.299 1549.52 433.03 1549.15 Q430.785 1548.78 428.378 1548.04 L428.378 1543.34 Q430.461 1544.47 432.683 1545.03 Q434.905 1545.58 437.382 1545.58 Q441.387 1545.58 443.725 1543.48 Q446.063 1541.37 446.063 1537.76 Q446.063 1534.15 443.725 1532.04 Q441.387 1529.94 437.382 1529.94 Q435.507 1529.94 433.632 1530.35 Q431.78 1530.77 429.836 1531.65 L429.836 1514.29 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M688.746 1530.21 Q692.102 1530.93 693.977 1533.2 Q695.875 1535.47 695.875 1538.8 Q695.875 1543.92 692.357 1546.72 Q688.838 1549.52 682.357 1549.52 Q680.181 1549.52 677.866 1549.08 Q675.574 1548.66 673.121 1547.81 L673.121 1543.29 Q675.065 1544.43 677.38 1545.01 Q679.695 1545.58 682.218 1545.58 Q686.616 1545.58 688.908 1543.85 Q691.222 1542.11 691.222 1538.8 Q691.222 1535.75 689.07 1534.03 Q686.94 1532.3 683.121 1532.3 L679.093 1532.3 L679.093 1528.45 L683.306 1528.45 Q686.755 1528.45 688.584 1527.09 Q690.412 1525.7 690.412 1523.11 Q690.412 1520.45 688.514 1519.03 Q686.639 1517.6 683.121 1517.6 Q681.199 1517.6 679 1518.01 Q676.801 1518.43 674.162 1519.31 L674.162 1515.14 Q676.824 1514.4 679.139 1514.03 Q681.477 1513.66 683.537 1513.66 Q688.861 1513.66 691.963 1516.09 Q695.065 1518.5 695.065 1522.62 Q695.065 1525.49 693.422 1527.48 Q691.778 1529.45 688.746 1530.21 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M704.741 1542.97 L709.625 1542.97 L709.625 1548.85 L704.741 1548.85 L704.741 1542.97 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M729.81 1517.37 Q726.199 1517.37 724.37 1520.93 Q722.565 1524.47 722.565 1531.6 Q722.565 1538.71 724.37 1542.27 Q726.199 1545.82 729.81 1545.82 Q733.444 1545.82 735.25 1542.27 Q737.079 1538.71 737.079 1531.6 Q737.079 1524.47 735.25 1520.93 Q733.444 1517.37 729.81 1517.37 M729.81 1513.66 Q735.62 1513.66 738.676 1518.27 Q741.755 1522.85 741.755 1531.6 Q741.755 1540.33 738.676 1544.94 Q735.62 1549.52 729.81 1549.52 Q724 1549.52 720.921 1544.94 Q717.866 1540.33 717.866 1531.6 Q717.866 1522.85 720.921 1518.27 Q724 1513.66 729.81 1513.66 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M759.972 1517.37 Q756.361 1517.37 754.532 1520.93 Q752.727 1524.47 752.727 1531.6 Q752.727 1538.71 754.532 1542.27 Q756.361 1545.82 759.972 1545.82 Q763.606 1545.82 765.412 1542.27 Q767.241 1538.71 767.241 1531.6 Q767.241 1524.47 765.412 1520.93 Q763.606 1517.37 759.972 1517.37 M759.972 1513.66 Q765.782 1513.66 768.838 1518.27 Q771.916 1522.85 771.916 1531.6 Q771.916 1540.33 768.838 1544.94 Q765.782 1549.52 759.972 1549.52 Q754.162 1549.52 751.083 1544.94 Q748.028 1540.33 748.028 1531.6 Q748.028 1522.85 751.083 1518.27 Q754.162 1513.66 759.972 1513.66 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M1009.99 1530.21 Q1013.35 1530.93 1015.22 1533.2 Q1017.12 1535.47 1017.12 1538.8 Q1017.12 1543.92 1013.6 1546.72 Q1010.09 1549.52 1003.6 1549.52 Q1001.43 1549.52 999.113 1549.08 Q996.822 1548.66 994.368 1547.81 L994.368 1543.29 Q996.313 1544.43 998.627 1545.01 Q1000.94 1545.58 1003.47 1545.58 Q1007.86 1545.58 1010.16 1543.85 Q1012.47 1542.11 1012.47 1538.8 Q1012.47 1535.75 1010.32 1534.03 Q1008.19 1532.3 1004.37 1532.3 L1000.34 1532.3 L1000.34 1528.45 L1004.55 1528.45 Q1008 1528.45 1009.83 1527.09 Q1011.66 1525.7 1011.66 1523.11 Q1011.66 1520.45 1009.76 1519.03 Q1007.89 1517.6 1004.37 1517.6 Q1002.45 1517.6 1000.25 1518.01 Q998.049 1518.43 995.41 1519.31 L995.41 1515.14 Q998.072 1514.4 1000.39 1514.03 Q1002.72 1513.66 1004.78 1513.66 Q1010.11 1513.66 1013.21 1516.09 Q1016.31 1518.5 1016.31 1522.62 Q1016.31 1525.49 1014.67 1527.48 Q1013.03 1529.45 1009.99 1530.21 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M1025.99 1542.97 L1030.87 1542.97 L1030.87 1548.85 L1025.99 1548.85 L1025.99 1542.97 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M1045.09 1544.91 L1061.4 1544.91 L1061.4 1548.85 L1039.46 1548.85 L1039.46 1544.91 Q1042.12 1542.16 1046.71 1537.53 Q1051.31 1532.88 1052.49 1531.53 Q1054.74 1529.01 1055.62 1527.27 Q1056.52 1525.51 1056.52 1523.82 Q1056.52 1521.07 1054.58 1519.33 Q1052.65 1517.6 1049.55 1517.6 Q1047.35 1517.6 1044.9 1518.36 Q1042.47 1519.13 1039.69 1520.68 L1039.69 1515.95 Q1042.52 1514.82 1044.97 1514.24 Q1047.42 1513.66 1049.46 1513.66 Q1054.83 1513.66 1058.03 1516.35 Q1061.22 1519.03 1061.22 1523.52 Q1061.22 1525.65 1060.41 1527.57 Q1059.62 1529.47 1057.52 1532.07 Q1056.94 1532.74 1053.84 1535.95 Q1050.73 1539.15 1045.09 1544.91 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M1071.27 1514.29 L1089.62 1514.29 L1089.62 1518.22 L1075.55 1518.22 L1075.55 1526.7 Q1076.57 1526.35 1077.59 1526.19 Q1078.6 1526 1079.62 1526 Q1085.41 1526 1088.79 1529.17 Q1092.17 1532.34 1092.17 1537.76 Q1092.17 1543.34 1088.7 1546.44 Q1085.22 1549.52 1078.9 1549.52 Q1076.73 1549.52 1074.46 1549.15 Q1072.21 1548.78 1069.81 1548.04 L1069.81 1543.34 Q1071.89 1544.47 1074.11 1545.03 Q1076.34 1545.58 1078.81 1545.58 Q1082.82 1545.58 1085.15 1543.48 Q1087.49 1541.37 1087.49 1537.76 Q1087.49 1534.15 1085.15 1532.04 Q1082.82 1529.94 1078.81 1529.94 Q1076.94 1529.94 1075.06 1530.35 Q1073.21 1530.77 1071.27 1531.65 L1071.27 1514.29 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M1330.25 1530.21 Q1333.6 1530.93 1335.48 1533.2 Q1337.37 1535.47 1337.37 1538.8 Q1337.37 1543.92 1333.86 1546.72 Q1330.34 1549.52 1323.86 1549.52 Q1321.68 1549.52 1319.37 1549.08 Q1317.07 1548.66 1314.62 1547.81 L1314.62 1543.29 Q1316.56 1544.43 1318.88 1545.01 Q1321.19 1545.58 1323.72 1545.58 Q1328.12 1545.58 1330.41 1543.85 Q1332.72 1542.11 1332.72 1538.8 Q1332.72 1535.75 1330.57 1534.03 Q1328.44 1532.3 1324.62 1532.3 L1320.59 1532.3 L1320.59 1528.45 L1324.81 1528.45 Q1328.25 1528.45 1330.08 1527.09 Q1331.91 1525.7 1331.91 1523.11 Q1331.91 1520.45 1330.01 1519.03 Q1328.14 1517.6 1324.62 1517.6 Q1322.7 1517.6 1320.5 1518.01 Q1318.3 1518.43 1315.66 1519.31 L1315.66 1515.14 Q1318.32 1514.4 1320.64 1514.03 Q1322.98 1513.66 1325.04 1513.66 Q1330.36 1513.66 1333.46 1516.09 Q1336.56 1518.5 1336.56 1522.62 Q1336.56 1525.49 1334.92 1527.48 Q1333.28 1529.45 1330.25 1530.21 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M1346.24 1542.97 L1351.12 1542.97 L1351.12 1548.85 L1346.24 1548.85 L1346.24 1542.97 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M1361.36 1514.29 L1379.71 1514.29 L1379.71 1518.22 L1365.64 1518.22 L1365.64 1526.7 Q1366.66 1526.35 1367.68 1526.19 Q1368.69 1526 1369.71 1526 Q1375.5 1526 1378.88 1529.17 Q1382.26 1532.34 1382.26 1537.76 Q1382.26 1543.34 1378.79 1546.44 Q1375.31 1549.52 1368.99 1549.52 Q1366.82 1549.52 1364.55 1549.15 Q1362.31 1548.78 1359.9 1548.04 L1359.9 1543.34 Q1361.98 1544.47 1364.2 1545.03 Q1366.43 1545.58 1368.9 1545.58 Q1372.91 1545.58 1375.24 1543.48 Q1377.58 1541.37 1377.58 1537.76 Q1377.58 1534.15 1375.24 1532.04 Q1372.91 1529.94 1368.9 1529.94 Q1367.03 1529.94 1365.15 1530.35 Q1363.3 1530.77 1361.36 1531.65 L1361.36 1514.29 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M1401.47 1517.37 Q1397.86 1517.37 1396.03 1520.93 Q1394.23 1524.47 1394.23 1531.6 Q1394.23 1538.71 1396.03 1542.27 Q1397.86 1545.82 1401.47 1545.82 Q1405.11 1545.82 1406.91 1542.27 Q1408.74 1538.71 1408.74 1531.6 Q1408.74 1524.47 1406.91 1520.93 Q1405.11 1517.37 1401.47 1517.37 M1401.47 1513.66 Q1407.28 1513.66 1410.34 1518.27 Q1413.42 1522.85 1413.42 1531.6 Q1413.42 1540.33 1410.34 1544.94 Q1407.28 1549.52 1401.47 1549.52 Q1395.66 1549.52 1392.58 1544.94 Q1389.53 1540.33 1389.53 1531.6 Q1389.53 1522.85 1392.58 1518.27 Q1395.66 1513.66 1401.47 1513.66 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M1651.49 1530.21 Q1654.85 1530.93 1656.72 1533.2 Q1658.62 1535.47 1658.62 1538.8 Q1658.62 1543.92 1655.1 1546.72 Q1651.59 1549.52 1645.1 1549.52 Q1642.93 1549.52 1640.61 1549.08 Q1638.32 1548.66 1635.87 1547.81 L1635.87 1543.29 Q1637.81 1544.43 1640.13 1545.01 Q1642.44 1545.58 1644.96 1545.58 Q1649.36 1545.58 1651.65 1543.85 Q1653.97 1542.11 1653.97 1538.8 Q1653.97 1535.75 1651.82 1534.03 Q1649.69 1532.3 1645.87 1532.3 L1641.84 1532.3 L1641.84 1528.45 L1646.05 1528.45 Q1649.5 1528.45 1651.33 1527.09 Q1653.16 1525.7 1653.16 1523.11 Q1653.16 1520.45 1651.26 1519.03 Q1649.39 1517.6 1645.87 1517.6 Q1643.95 1517.6 1641.75 1518.01 Q1639.55 1518.43 1636.91 1519.31 L1636.91 1515.14 Q1639.57 1514.4 1641.89 1514.03 Q1644.22 1513.66 1646.28 1513.66 Q1651.61 1513.66 1654.71 1516.09 Q1657.81 1518.5 1657.81 1522.62 Q1657.81 1525.49 1656.17 1527.48 Q1654.52 1529.45 1651.49 1530.21 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M1667.49 1542.97 L1672.37 1542.97 L1672.37 1548.85 L1667.49 1548.85 L1667.49 1542.97 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M1681.38 1514.29 L1703.6 1514.29 L1703.6 1516.28 L1691.05 1548.85 L1686.17 1548.85 L1697.97 1518.22 L1681.38 1518.22 L1681.38 1514.29 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M1712.77 1514.29 L1731.12 1514.29 L1731.12 1518.22 L1717.05 1518.22 L1717.05 1526.7 Q1718.07 1526.35 1719.08 1526.19 Q1720.1 1526 1721.12 1526 Q1726.91 1526 1730.29 1529.17 Q1733.67 1532.34 1733.67 1537.76 Q1733.67 1543.34 1730.2 1546.44 Q1726.72 1549.52 1720.4 1549.52 Q1718.23 1549.52 1715.96 1549.15 Q1713.71 1548.78 1711.31 1548.04 L1711.31 1543.34 Q1713.39 1544.47 1715.61 1545.03 Q1717.83 1545.58 1720.31 1545.58 Q1724.32 1545.58 1726.65 1543.48 Q1728.99 1541.37 1728.99 1537.76 Q1728.99 1534.15 1726.65 1532.04 Q1724.32 1529.94 1720.31 1529.94 Q1718.44 1529.94 1716.56 1530.35 Q1714.71 1530.77 1712.77 1531.65 L1712.77 1514.29 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M1971.07 1518.36 L1959.27 1536.81 L1971.07 1536.81 L1971.07 1518.36 M1969.85 1514.29 L1975.73 1514.29 L1975.73 1536.81 L1980.66 1536.81 L1980.66 1540.7 L1975.73 1540.7 L1975.73 1548.85 L1971.07 1548.85 L1971.07 1540.7 L1955.47 1540.7 L1955.47 1536.19 L1969.85 1514.29 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M1988.39 1542.97 L1993.27 1542.97 L1993.27 1548.85 L1988.39 1548.85 L1988.39 1542.97 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M2013.46 1517.37 Q2009.85 1517.37 2008.02 1520.93 Q2006.21 1524.47 2006.21 1531.6 Q2006.21 1538.71 2008.02 1542.27 Q2009.85 1545.82 2013.46 1545.82 Q2017.09 1545.82 2018.9 1542.27 Q2020.73 1538.71 2020.73 1531.6 Q2020.73 1524.47 2018.9 1520.93 Q2017.09 1517.37 2013.46 1517.37 M2013.46 1513.66 Q2019.27 1513.66 2022.32 1518.27 Q2025.4 1522.85 2025.4 1531.6 Q2025.4 1540.33 2022.32 1544.94 Q2019.27 1549.52 2013.46 1549.52 Q2007.65 1549.52 2004.57 1544.94 Q2001.51 1540.33 2001.51 1531.6 Q2001.51 1522.85 2004.57 1518.27 Q2007.65 1513.66 2013.46 1513.66 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M2043.62 1517.37 Q2040.01 1517.37 2038.18 1520.93 Q2036.37 1524.47 2036.37 1531.6 Q2036.37 1538.71 2038.18 1542.27 Q2040.01 1545.82 2043.62 1545.82 Q2047.25 1545.82 2049.06 1542.27 Q2050.89 1538.71 2050.89 1531.6 Q2050.89 1524.47 2049.06 1520.93 Q2047.25 1517.37 2043.62 1517.37 M2043.62 1513.66 Q2049.43 1513.66 2052.48 1518.27 Q2055.56 1522.85 2055.56 1531.6 Q2055.56 1540.33 2052.48 1544.94 Q2049.43 1549.52 2043.62 1549.52 Q2037.81 1549.52 2034.73 1544.94 Q2031.67 1540.33 2031.67 1531.6 Q2031.67 1522.85 2034.73 1518.27 Q2037.81 1513.66 2043.62 1513.66 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M2292.32 1518.36 L2280.52 1536.81 L2292.32 1536.81 L2292.32 1518.36 M2291.09 1514.29 L2296.97 1514.29 L2296.97 1536.81 L2301.9 1536.81 L2301.9 1540.7 L2296.97 1540.7 L2296.97 1548.85 L2292.32 1548.85 L2292.32 1540.7 L2276.72 1540.7 L2276.72 1536.19 L2291.09 1514.29 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M2309.64 1542.97 L2314.52 1542.97 L2314.52 1548.85 L2309.64 1548.85 L2309.64 1542.97 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M2328.73 1544.91 L2345.05 1544.91 L2345.05 1548.85 L2323.11 1548.85 L2323.11 1544.91 Q2325.77 1542.16 2330.35 1537.53 Q2334.96 1532.88 2336.14 1531.53 Q2338.39 1529.01 2339.26 1527.27 Q2340.17 1525.51 2340.17 1523.82 Q2340.17 1521.07 2338.22 1519.33 Q2336.3 1517.6 2333.2 1517.6 Q2331 1517.6 2328.55 1518.36 Q2326.12 1519.13 2323.34 1520.68 L2323.34 1515.95 Q2326.16 1514.82 2328.62 1514.24 Q2331.07 1513.66 2333.11 1513.66 Q2338.48 1513.66 2341.67 1516.35 Q2344.87 1519.03 2344.87 1523.52 Q2344.87 1525.65 2344.06 1527.57 Q2343.27 1529.47 2341.16 1532.07 Q2340.58 1532.74 2337.48 1535.95 Q2334.38 1539.15 2328.73 1544.91 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M2354.91 1514.29 L2373.27 1514.29 L2373.27 1518.22 L2359.2 1518.22 L2359.2 1526.7 Q2360.21 1526.35 2361.23 1526.19 Q2362.25 1526 2363.27 1526 Q2369.06 1526 2372.44 1529.17 Q2375.82 1532.34 2375.82 1537.76 Q2375.82 1543.34 2372.34 1546.44 Q2368.87 1549.52 2362.55 1549.52 Q2360.38 1549.52 2358.11 1549.15 Q2355.86 1548.78 2353.45 1548.04 L2353.45 1543.34 Q2355.54 1544.47 2357.76 1545.03 Q2359.98 1545.58 2362.46 1545.58 Q2366.46 1545.58 2368.8 1543.48 Q2371.14 1541.37 2371.14 1537.76 Q2371.14 1534.15 2368.8 1532.04 Q2366.46 1529.94 2362.46 1529.94 Q2360.58 1529.94 2358.71 1530.35 Q2356.86 1530.77 2354.91 1531.65 L2354.91 1514.29 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><polyline clip-path="url(#clip290)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="184.191,1486.45 184.191,47.2441 "/>
<polyline clip-path="url(#clip290)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="184.191,1324.01 203.088,1324.01 "/>
<polyline clip-path="url(#clip290)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="184.191,1007.4 203.088,1007.4 "/>
<polyline clip-path="url(#clip290)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="184.191,690.779 203.088,690.779 "/>
<polyline clip-path="url(#clip290)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="184.191,374.162 203.088,374.162 "/>
<polyline clip-path="url(#clip290)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="184.191,57.5442 203.088,57.5442 "/>
<path clip-path="url(#clip290)" d="M51.6634 1337.36 L59.3023 1337.36 L59.3023 1310.99 L50.9921 1312.66 L50.9921 1308.4 L59.256 1306.73 L63.9319 1306.73 L63.9319 1337.36 L71.5707 1337.36 L71.5707 1341.29 L51.6634 1341.29 L51.6634 1337.36 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M81.0151 1335.41 L85.8993 1335.41 L85.8993 1341.29 L81.0151 1341.29 L81.0151 1335.41 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M110.251 1322.66 Q113.608 1323.38 115.483 1325.65 Q117.381 1327.91 117.381 1331.25 Q117.381 1336.36 113.862 1339.16 Q110.344 1341.97 103.862 1341.97 Q101.686 1341.97 99.3715 1341.53 Q97.0798 1341.11 94.6262 1340.25 L94.6262 1335.74 Q96.5706 1336.87 98.8854 1337.45 Q101.2 1338.03 103.723 1338.03 Q108.121 1338.03 110.413 1336.29 Q112.728 1334.56 112.728 1331.25 Q112.728 1328.19 110.575 1326.48 Q108.446 1324.74 104.626 1324.74 L100.598 1324.74 L100.598 1320.9 L104.811 1320.9 Q108.26 1320.9 110.089 1319.53 Q111.918 1318.15 111.918 1315.55 Q111.918 1312.89 110.02 1311.48 Q108.145 1310.04 104.626 1310.04 Q102.705 1310.04 100.506 1310.46 Q98.3067 1310.88 95.6678 1311.76 L95.6678 1307.59 Q98.3298 1306.85 100.645 1306.48 Q102.983 1306.11 105.043 1306.11 Q110.367 1306.11 113.469 1308.54 Q116.57 1310.95 116.57 1315.07 Q116.57 1317.94 114.927 1319.93 Q113.283 1321.9 110.251 1322.66 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M136.246 1309.81 Q132.635 1309.81 130.807 1313.38 Q129.001 1316.92 129.001 1324.05 Q129.001 1331.15 130.807 1334.72 Q132.635 1338.26 136.246 1338.26 Q139.881 1338.26 141.686 1334.72 Q143.515 1331.15 143.515 1324.05 Q143.515 1316.92 141.686 1313.38 Q139.881 1309.81 136.246 1309.81 M136.246 1306.11 Q142.056 1306.11 145.112 1310.72 Q148.191 1315.3 148.191 1324.05 Q148.191 1332.78 145.112 1337.38 Q142.056 1341.97 136.246 1341.97 Q130.436 1341.97 127.357 1337.38 Q124.302 1332.78 124.302 1324.05 Q124.302 1315.3 127.357 1310.72 Q130.436 1306.11 136.246 1306.11 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M52.6588 1020.74 L60.2976 1020.74 L60.2976 994.376 L51.9875 996.042 L51.9875 991.783 L60.2513 990.116 L64.9272 990.116 L64.9272 1020.74 L72.5661 1020.74 L72.5661 1024.68 L52.6588 1024.68 L52.6588 1020.74 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M82.0105 1018.8 L86.8947 1018.8 L86.8947 1024.68 L82.0105 1024.68 L82.0105 1018.8 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M111.246 1006.04 Q114.603 1006.76 116.478 1009.03 Q118.376 1011.3 118.376 1014.63 Q118.376 1019.75 114.858 1022.55 Q111.339 1025.35 104.858 1025.35 Q102.682 1025.35 100.367 1024.91 Q98.0752 1024.49 95.6215 1023.63 L95.6215 1019.12 Q97.566 1020.26 99.8808 1020.83 Q102.196 1021.41 104.719 1021.41 Q109.117 1021.41 111.408 1019.68 Q113.723 1017.94 113.723 1014.63 Q113.723 1011.57 111.571 1009.86 Q109.441 1008.13 105.621 1008.13 L101.594 1008.13 L101.594 1004.28 L105.807 1004.28 Q109.256 1004.28 111.084 1002.92 Q112.913 1001.53 112.913 998.936 Q112.913 996.274 111.015 994.862 Q109.14 993.427 105.621 993.427 Q103.7 993.427 101.501 993.843 Q99.3021 994.26 96.6632 995.139 L96.6632 990.973 Q99.3252 990.232 101.64 989.862 Q103.978 989.491 106.038 989.491 Q111.362 989.491 114.464 991.922 Q117.566 994.329 117.566 998.45 Q117.566 1001.32 115.922 1003.31 Q114.279 1005.28 111.246 1006.04 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M127.288 990.116 L145.644 990.116 L145.644 994.052 L131.57 994.052 L131.57 1002.52 Q132.589 1002.18 133.607 1002.01 Q134.626 1001.83 135.644 1001.83 Q141.431 1001.83 144.811 1005 Q148.191 1008.17 148.191 1013.59 Q148.191 1019.17 144.718 1022.27 Q141.246 1025.35 134.927 1025.35 Q132.751 1025.35 130.482 1024.98 Q128.237 1024.61 125.83 1023.87 L125.83 1019.17 Q127.913 1020.3 130.135 1020.86 Q132.357 1021.41 134.834 1021.41 Q138.839 1021.41 141.177 1019.31 Q143.515 1017.2 143.515 1013.59 Q143.515 1009.98 141.177 1007.87 Q138.839 1005.76 134.834 1005.76 Q132.959 1005.76 131.084 1006.18 Q129.232 1006.6 127.288 1007.48 L127.288 990.116 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M51.6634 704.124 L59.3023 704.124 L59.3023 677.758 L50.9921 679.425 L50.9921 675.166 L59.256 673.499 L63.9319 673.499 L63.9319 704.124 L71.5707 704.124 L71.5707 708.059 L51.6634 708.059 L51.6634 704.124 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M81.0151 702.179 L85.8993 702.179 L85.8993 708.059 L81.0151 708.059 L81.0151 702.179 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M108.932 677.573 L97.1261 696.022 L108.932 696.022 L108.932 677.573 M107.705 673.499 L113.584 673.499 L113.584 696.022 L118.515 696.022 L118.515 699.911 L113.584 699.911 L113.584 708.059 L108.932 708.059 L108.932 699.911 L93.3299 699.911 L93.3299 695.397 L107.705 673.499 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M136.246 676.578 Q132.635 676.578 130.807 680.142 Q129.001 683.684 129.001 690.814 Q129.001 697.92 130.807 701.485 Q132.635 705.027 136.246 705.027 Q139.881 705.027 141.686 701.485 Q143.515 697.92 143.515 690.814 Q143.515 683.684 141.686 680.142 Q139.881 676.578 136.246 676.578 M136.246 672.874 Q142.056 672.874 145.112 677.48 Q148.191 682.064 148.191 690.814 Q148.191 699.541 145.112 704.147 Q142.056 708.73 136.246 708.73 Q130.436 708.73 127.357 704.147 Q124.302 699.541 124.302 690.814 Q124.302 682.064 127.357 677.48 Q130.436 672.874 136.246 672.874 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M52.6588 387.506 L60.2976 387.506 L60.2976 361.141 L51.9875 362.808 L51.9875 358.548 L60.2513 356.882 L64.9272 356.882 L64.9272 387.506 L72.5661 387.506 L72.5661 391.442 L52.6588 391.442 L52.6588 387.506 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M82.0105 385.562 L86.8947 385.562 L86.8947 391.442 L82.0105 391.442 L82.0105 385.562 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M109.927 360.956 L98.1215 379.405 L109.927 379.405 L109.927 360.956 M108.7 356.882 L114.58 356.882 L114.58 379.405 L119.51 379.405 L119.51 383.294 L114.58 383.294 L114.58 391.442 L109.927 391.442 L109.927 383.294 L94.3252 383.294 L94.3252 378.78 L108.7 356.882 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M127.288 356.882 L145.644 356.882 L145.644 360.817 L131.57 360.817 L131.57 369.289 Q132.589 368.942 133.607 368.78 Q134.626 368.595 135.644 368.595 Q141.431 368.595 144.811 371.766 Q148.191 374.937 148.191 380.354 Q148.191 385.932 144.718 389.034 Q141.246 392.113 134.927 392.113 Q132.751 392.113 130.482 391.743 Q128.237 391.372 125.83 390.631 L125.83 385.932 Q127.913 387.067 130.135 387.622 Q132.357 388.178 134.834 388.178 Q138.839 388.178 141.177 386.071 Q143.515 383.965 143.515 380.354 Q143.515 376.743 141.177 374.636 Q138.839 372.53 134.834 372.53 Q132.959 372.53 131.084 372.946 Q129.232 373.363 127.288 374.243 L127.288 356.882 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M51.6634 70.8891 L59.3023 70.8891 L59.3023 44.5235 L50.9921 46.1901 L50.9921 41.9309 L59.256 40.2642 L63.9319 40.2642 L63.9319 70.8891 L71.5707 70.8891 L71.5707 74.8242 L51.6634 74.8242 L51.6634 70.8891 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M81.0151 68.9447 L85.8993 68.9447 L85.8993 74.8242 L81.0151 74.8242 L81.0151 68.9447 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M96.1308 40.2642 L114.487 40.2642 L114.487 44.1994 L100.413 44.1994 L100.413 52.6716 Q101.432 52.3244 102.45 52.1623 Q103.469 51.9771 104.487 51.9771 Q110.274 51.9771 113.654 55.1484 Q117.033 58.3197 117.033 63.7363 Q117.033 69.315 113.561 72.4169 Q110.089 75.4955 103.77 75.4955 Q101.594 75.4955 99.3252 75.1252 Q97.0798 74.7548 94.6724 74.0141 L94.6724 69.315 Q96.7558 70.4493 98.978 71.0048 Q101.2 71.5604 103.677 71.5604 Q107.682 71.5604 110.02 69.4539 Q112.358 67.3474 112.358 63.7363 Q112.358 60.1253 110.02 58.0188 Q107.682 55.9123 103.677 55.9123 Q101.802 55.9123 99.927 56.329 Q98.0752 56.7456 96.1308 57.6253 L96.1308 40.2642 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M136.246 43.3429 Q132.635 43.3429 130.807 46.9077 Q129.001 50.4494 129.001 57.579 Q129.001 64.6854 130.807 68.2502 Q132.635 71.7919 136.246 71.7919 Q139.881 71.7919 141.686 68.2502 Q143.515 64.6854 143.515 57.579 Q143.515 50.4494 141.686 46.9077 Q139.881 43.3429 136.246 43.3429 M136.246 39.6393 Q142.056 39.6393 145.112 44.2457 Q148.191 48.829 148.191 57.579 Q148.191 66.3058 145.112 70.9122 Q142.056 75.4955 136.246 75.4955 Q130.436 75.4955 127.357 70.9122 Q124.302 66.3058 124.302 57.579 Q124.302 48.829 127.357 44.2457 Q130.436 39.6393 136.246 39.6393 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><circle clip-path="url(#clip292)" cx="245.565" cy="1445.72" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip292)" cx="1413.3" cy="665.359" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip292)" cx="1820.54" cy="421.306" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip292)" cx="2291.38" cy="87.9763" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<polyline clip-path="url(#clip292)" style="stroke:#e26f46; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="245.565,1445.4 266.686,1431.45 390.386,1349.79 447.931,1311.8 525.063,1260.88 588.798,1218.81 648.066,1179.68 722.466,1130.56 789.204,1086.5 865.595,1036.07 922.697,998.376 995.731,950.161 1062.39,906.155 1138.26,856.071 1198.4,816.368 1273.11,767.042 1330.33,729.267 1411.01,676.008 1471.29,636.209 1539.85,590.949 1607.75,546.126 1682.45,496.812 1739.51,459.145 1811.51,411.612 1878.1,367.649 1946.78,322.309 2016.39,276.352 2089.08,228.364 2146.35,190.561 2276.75,104.473 2291.38,94.814 "/>
<path clip-path="url(#clip290)" d="M256.476 198.898 L519.559 198.898 L519.559 95.2176 L256.476 95.2176  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<polyline clip-path="url(#clip290)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="256.476,198.898 519.559,198.898 519.559,95.2176 256.476,95.2176 256.476,198.898 "/>
<polyline clip-path="url(#clip290)" style="stroke:#e26f46; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="280.571,147.058 425.142,147.058 "/>
<path clip-path="url(#clip290)" d="M465.742 128.319 L465.742 131.861 L461.668 131.861 Q459.376 131.861 458.474 132.787 Q457.594 133.713 457.594 136.12 L457.594 138.412 L464.608 138.412 L464.608 141.722 L457.594 141.722 L457.594 164.338 L453.312 164.338 L453.312 141.722 L449.238 141.722 L449.238 138.412 L453.312 138.412 L453.312 136.606 Q453.312 132.278 455.325 130.31 Q457.339 128.319 461.714 128.319 L465.742 128.319 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M469.307 138.412 L473.566 138.412 L473.566 164.338 L469.307 164.338 L469.307 138.412 M469.307 128.319 L473.566 128.319 L473.566 133.713 L469.307 133.713 L469.307 128.319 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip290)" d="M486.691 131.051 L486.691 138.412 L495.464 138.412 L495.464 141.722 L486.691 141.722 L486.691 155.796 Q486.691 158.967 487.548 159.87 Q488.427 160.773 491.089 160.773 L495.464 160.773 L495.464 164.338 L491.089 164.338 Q486.159 164.338 484.284 162.509 Q482.409 160.657 482.409 155.796 L482.409 141.722 L479.284 141.722 L479.284 138.412 L482.409 138.412 L482.409 131.051 L486.691 131.051 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /></svg>

```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

