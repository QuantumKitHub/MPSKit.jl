```@meta
EditURL = "../../../../../examples/quantum1d/1.ising-cft/main.jl"
```

[![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/maartenvd/MPSKit.jl/gh-pages?filepath=dev/examples/quantum1d/1.ising-cft/main.ipynb)
[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](https://nbviewer.jupyter.org/github/maartenvd/MPSKit.jl/blob/gh-pages/dev/examples/quantum1d/1.ising-cft/main.ipynb)
[![](https://img.shields.io/badge/download-project-orange)](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/maartenvd/MPSKit.jl/examples/tree/gh-pages/dev/examples/quantum1d/1.ising-cft)

# The Ising CFT spectrum

This tutorial is meant to show the finite size CFT spectrum for the quantum Ising model. We
do this by first employing an exact diagonalization technique, and then extending the
analysis to larger system sizes through the use of MPS techniques.

````julia
using MPSKit, MPSKitModels, TensorKit, Plots, KrylovKit
using LinearAlgebra: eigen, diagm, Hermitian
````

The hamiltonian is defined on a finite lattice with periodic boundary conditions,
which can be implemented as follows:

````julia
L = 12
H = periodic_boundary_conditions(transverse_field_ising(), L);
````

## Exact diagonalisation

In MPSKit, there is support for exact diagonalisation by leveraging the fact that applying
the hamiltonian to an untruncated MPS will result in an effective hamiltonian on the center
site which implements the action of the entire hamiltonian. Thus, optimizing the middle
tensor is equivalent to optimixing a state in the entire Hilbert space, as all other tensors
are just unitary matrices that mix the basis.

````julia
energies, states = exact_diagonalization(H; num=18, alg=Lanczos(; krylovdim=200));
plot(
    real.(energies);
    seriestype=:scatter,
    legend=false,
    ylabel="energy",
    xlabel="#eigenvalue",
)
````

```@raw html
<?xml version="1.0" encoding="utf-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="600" height="400" viewBox="0 0 2400 1600">
<defs>
  <clipPath id="clip570">
    <rect x="0" y="0" width="2400" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip570)" d="M0 1600 L2400 1600 L2400 0 L0 0  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip571">
    <rect x="480" y="0" width="1681" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip570)" d="M242.621 1423.18 L2352.76 1423.18 L2352.76 47.2441 L242.621 47.2441  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip572">
    <rect x="242" y="47" width="2111" height="1377"/>
  </clipPath>
</defs>
<polyline clip-path="url(#clip572)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="597.259,1423.18 597.259,47.2441 "/>
<polyline clip-path="url(#clip572)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="965.906,1423.18 965.906,47.2441 "/>
<polyline clip-path="url(#clip572)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1334.55,1423.18 1334.55,47.2441 "/>
<polyline clip-path="url(#clip572)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1703.2,1423.18 1703.2,47.2441 "/>
<polyline clip-path="url(#clip572)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="2071.85,1423.18 2071.85,47.2441 "/>
<polyline clip-path="url(#clip570)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="242.621,1423.18 2352.76,1423.18 "/>
<polyline clip-path="url(#clip570)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="597.259,1423.18 597.259,1404.28 "/>
<polyline clip-path="url(#clip570)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="965.906,1423.18 965.906,1404.28 "/>
<polyline clip-path="url(#clip570)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1334.55,1423.18 1334.55,1404.28 "/>
<polyline clip-path="url(#clip570)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1703.2,1423.18 1703.2,1404.28 "/>
<polyline clip-path="url(#clip570)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="2071.85,1423.18 2071.85,1404.28 "/>
<path clip-path="url(#clip570)" d="M587.537 1451.02 L605.893 1451.02 L605.893 1454.96 L591.819 1454.96 L591.819 1463.43 Q592.838 1463.08 593.856 1462.92 Q594.875 1462.73 595.893 1462.73 Q601.68 1462.73 605.06 1465.9 Q608.44 1469.08 608.44 1474.49 Q608.44 1480.07 604.967 1483.17 Q601.495 1486.25 595.176 1486.25 Q593 1486.25 590.731 1485.88 Q588.486 1485.51 586.079 1484.77 L586.079 1480.07 Q588.162 1481.2 590.384 1481.76 Q592.606 1482.32 595.083 1482.32 Q599.088 1482.32 601.426 1480.21 Q603.764 1478.1 603.764 1474.49 Q603.764 1470.88 601.426 1468.77 Q599.088 1466.67 595.083 1466.67 Q593.208 1466.67 591.333 1467.08 Q589.481 1467.5 587.537 1468.38 L587.537 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M940.594 1481.64 L948.233 1481.64 L948.233 1455.28 L939.922 1456.95 L939.922 1452.69 L948.186 1451.02 L952.862 1451.02 L952.862 1481.64 L960.501 1481.64 L960.501 1485.58 L940.594 1485.58 L940.594 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M979.945 1454.1 Q976.334 1454.1 974.506 1457.66 Q972.7 1461.2 972.7 1468.33 Q972.7 1475.44 974.506 1479.01 Q976.334 1482.55 979.945 1482.55 Q983.58 1482.55 985.385 1479.01 Q987.214 1475.44 987.214 1468.33 Q987.214 1461.2 985.385 1457.66 Q983.58 1454.1 979.945 1454.1 M979.945 1450.39 Q985.756 1450.39 988.811 1455 Q991.89 1459.58 991.89 1468.33 Q991.89 1477.06 988.811 1481.67 Q985.756 1486.25 979.945 1486.25 Q974.135 1486.25 971.057 1481.67 Q968.001 1477.06 968.001 1468.33 Q968.001 1459.58 971.057 1455 Q974.135 1450.39 979.945 1450.39 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M1309.74 1481.64 L1317.38 1481.64 L1317.38 1455.28 L1309.07 1456.95 L1309.07 1452.69 L1317.33 1451.02 L1322.01 1451.02 L1322.01 1481.64 L1329.65 1481.64 L1329.65 1485.58 L1309.74 1485.58 L1309.74 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M1339.14 1451.02 L1357.49 1451.02 L1357.49 1454.96 L1343.42 1454.96 L1343.42 1463.43 Q1344.44 1463.08 1345.46 1462.92 Q1346.47 1462.73 1347.49 1462.73 Q1353.28 1462.73 1356.66 1465.9 Q1360.04 1469.08 1360.04 1474.49 Q1360.04 1480.07 1356.57 1483.17 Q1353.09 1486.25 1346.78 1486.25 Q1344.6 1486.25 1342.33 1485.88 Q1340.09 1485.51 1337.68 1484.77 L1337.68 1480.07 Q1339.76 1481.2 1341.98 1481.76 Q1344.21 1482.32 1346.68 1482.32 Q1350.69 1482.32 1353.03 1480.21 Q1355.36 1478.1 1355.36 1474.49 Q1355.36 1470.88 1353.03 1468.77 Q1350.69 1466.67 1346.68 1466.67 Q1344.81 1466.67 1342.93 1467.08 Q1341.08 1467.5 1339.14 1468.38 L1339.14 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M1681.97 1481.64 L1698.29 1481.64 L1698.29 1485.58 L1676.35 1485.58 L1676.35 1481.64 Q1679.01 1478.89 1683.59 1474.26 Q1688.2 1469.61 1689.38 1468.27 Q1691.63 1465.74 1692.51 1464.01 Q1693.41 1462.25 1693.41 1460.56 Q1693.41 1457.8 1691.46 1456.07 Q1689.54 1454.33 1686.44 1454.33 Q1684.24 1454.33 1681.79 1455.09 Q1679.36 1455.86 1676.58 1457.41 L1676.58 1452.69 Q1679.4 1451.55 1681.86 1450.97 Q1684.31 1450.39 1686.35 1450.39 Q1691.72 1450.39 1694.91 1453.08 Q1698.11 1455.77 1698.11 1460.26 Q1698.11 1462.39 1697.3 1464.31 Q1696.51 1466.2 1694.4 1468.8 Q1693.83 1469.47 1690.72 1472.69 Q1687.62 1475.88 1681.97 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M1718.11 1454.1 Q1714.5 1454.1 1712.67 1457.66 Q1710.86 1461.2 1710.86 1468.33 Q1710.86 1475.44 1712.67 1479.01 Q1714.5 1482.55 1718.11 1482.55 Q1721.74 1482.55 1723.55 1479.01 Q1725.38 1475.44 1725.38 1468.33 Q1725.38 1461.2 1723.55 1457.66 Q1721.74 1454.1 1718.11 1454.1 M1718.11 1450.39 Q1723.92 1450.39 1726.97 1455 Q1730.05 1459.58 1730.05 1468.33 Q1730.05 1477.06 1726.97 1481.67 Q1723.92 1486.25 1718.11 1486.25 Q1712.3 1486.25 1709.22 1481.67 Q1706.16 1477.06 1706.16 1468.33 Q1706.16 1459.58 1709.22 1455 Q1712.3 1450.39 1718.11 1450.39 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M2051.12 1481.64 L2067.44 1481.64 L2067.44 1485.58 L2045.49 1485.58 L2045.49 1481.64 Q2048.15 1478.89 2052.74 1474.26 Q2057.34 1469.61 2058.53 1468.27 Q2060.77 1465.74 2061.65 1464.01 Q2062.55 1462.25 2062.55 1460.56 Q2062.55 1457.8 2060.61 1456.07 Q2058.69 1454.33 2055.59 1454.33 Q2053.39 1454.33 2050.93 1455.09 Q2048.5 1455.86 2045.72 1457.41 L2045.72 1452.69 Q2048.55 1451.55 2051 1450.97 Q2053.46 1450.39 2055.49 1450.39 Q2060.86 1450.39 2064.06 1453.08 Q2067.25 1455.77 2067.25 1460.26 Q2067.25 1462.39 2066.44 1464.31 Q2065.65 1466.2 2063.55 1468.8 Q2062.97 1469.47 2059.87 1472.69 Q2056.77 1475.88 2051.12 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M2077.3 1451.02 L2095.65 1451.02 L2095.65 1454.96 L2081.58 1454.96 L2081.58 1463.43 Q2082.6 1463.08 2083.62 1462.92 Q2084.64 1462.73 2085.65 1462.73 Q2091.44 1462.73 2094.82 1465.9 Q2098.2 1469.08 2098.2 1474.49 Q2098.2 1480.07 2094.73 1483.17 Q2091.26 1486.25 2084.94 1486.25 Q2082.76 1486.25 2080.49 1485.88 Q2078.25 1485.51 2075.84 1484.77 L2075.84 1480.07 Q2077.92 1481.2 2080.15 1481.76 Q2082.37 1482.32 2084.84 1482.32 Q2088.85 1482.32 2091.19 1480.21 Q2093.53 1478.1 2093.53 1474.49 Q2093.53 1470.88 2091.19 1468.77 Q2088.85 1466.67 2084.84 1466.67 Q2082.97 1466.67 2081.09 1467.08 Q2079.24 1467.5 2077.3 1468.38 L2077.3 1451.02 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M1123.4 1539.37 L1114.13 1539.37 L1111.46 1550 L1120.79 1550 L1123.4 1539.37 M1118.62 1521.26 L1115.31 1534.46 L1124.6 1534.46 L1127.95 1521.26 L1133.04 1521.26 L1129.76 1534.46 L1139.69 1534.46 L1139.69 1539.37 L1128.52 1539.37 L1125.91 1550 L1136.03 1550 L1136.03 1554.87 L1124.67 1554.87 L1121.36 1568.04 L1116.27 1568.04 L1119.54 1554.87 L1110.22 1554.87 L1106.94 1568.04 L1101.82 1568.04 L1105.13 1554.87 L1095.1 1554.87 L1095.1 1550 L1106.3 1550 L1108.98 1539.37 L1098.73 1539.37 L1098.73 1534.46 L1110.22 1534.46 L1113.46 1521.26 L1118.62 1521.26 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M1181.32 1548.76 L1181.32 1551.62 L1154.4 1551.62 Q1154.78 1557.67 1158.02 1560.85 Q1161.3 1564 1167.13 1564 Q1170.5 1564 1173.65 1563.17 Q1176.84 1562.35 1179.95 1560.69 L1179.95 1566.23 Q1176.8 1567.57 1173.49 1568.27 Q1170.18 1568.97 1166.78 1568.97 Q1158.25 1568.97 1153.25 1564 Q1148.29 1559.04 1148.29 1550.57 Q1148.29 1541.82 1153 1536.69 Q1157.74 1531.54 1165.76 1531.54 Q1172.95 1531.54 1177.12 1536.18 Q1181.32 1540.8 1181.32 1548.76 M1175.47 1547.04 Q1175.4 1542.23 1172.76 1539.37 Q1170.15 1536.5 1165.82 1536.5 Q1160.92 1536.5 1157.96 1539.27 Q1155.03 1542.04 1154.59 1547.07 L1175.47 1547.04 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M1190.94 1532.4 L1196.79 1532.4 L1196.79 1568.04 L1190.94 1568.04 L1190.94 1532.4 M1190.94 1518.52 L1196.79 1518.52 L1196.79 1525.93 L1190.94 1525.93 L1190.94 1518.52 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M1232.5 1549.81 Q1232.5 1543.44 1229.86 1539.94 Q1227.25 1536.44 1222.51 1536.44 Q1217.8 1536.44 1215.16 1539.94 Q1212.55 1543.44 1212.55 1549.81 Q1212.55 1556.14 1215.16 1559.64 Q1217.8 1563.14 1222.51 1563.14 Q1227.25 1563.14 1229.86 1559.64 Q1232.5 1556.14 1232.5 1549.81 M1238.36 1563.62 Q1238.36 1572.72 1234.32 1577.15 Q1230.28 1581.6 1221.94 1581.6 Q1218.85 1581.6 1216.11 1581.13 Q1213.37 1580.68 1210.8 1579.72 L1210.8 1574.03 Q1213.37 1575.43 1215.89 1576.1 Q1218.4 1576.76 1221.01 1576.76 Q1226.77 1576.76 1229.64 1573.74 Q1232.5 1570.75 1232.5 1564.67 L1232.5 1561.77 Q1230.69 1564.92 1227.86 1566.48 Q1225.02 1568.04 1221.08 1568.04 Q1214.52 1568.04 1210.51 1563.05 Q1206.5 1558.05 1206.5 1549.81 Q1206.5 1541.53 1210.51 1536.53 Q1214.52 1531.54 1221.08 1531.54 Q1225.02 1531.54 1227.86 1533.1 Q1230.69 1534.66 1232.5 1537.81 L1232.5 1532.4 L1238.36 1532.4 L1238.36 1563.62 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M1280.91 1548.76 L1280.91 1551.62 L1253.99 1551.62 Q1254.37 1557.67 1257.62 1560.85 Q1260.89 1564 1266.72 1564 Q1270.09 1564 1273.24 1563.17 Q1276.43 1562.35 1279.55 1560.69 L1279.55 1566.23 Q1276.4 1567.57 1273.08 1568.27 Q1269.77 1568.97 1266.37 1568.97 Q1257.84 1568.97 1252.84 1564 Q1247.88 1559.04 1247.88 1550.57 Q1247.88 1541.82 1252.59 1536.69 Q1257.33 1531.54 1265.35 1531.54 Q1272.54 1531.54 1276.71 1536.18 Q1280.91 1540.8 1280.91 1548.76 M1275.06 1547.04 Q1274.99 1542.23 1272.35 1539.37 Q1269.74 1536.5 1265.41 1536.5 Q1260.51 1536.5 1257.55 1539.27 Q1254.62 1542.04 1254.18 1547.07 L1275.06 1547.04 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M1320.16 1546.53 L1320.16 1568.04 L1314.3 1568.04 L1314.3 1546.72 Q1314.3 1541.66 1312.33 1539.14 Q1310.36 1536.63 1306.41 1536.63 Q1301.67 1536.63 1298.93 1539.65 Q1296.19 1542.68 1296.19 1547.9 L1296.19 1568.04 L1290.3 1568.04 L1290.3 1532.4 L1296.19 1532.4 L1296.19 1537.93 Q1298.29 1534.72 1301.13 1533.13 Q1303.99 1531.54 1307.71 1531.54 Q1313.86 1531.54 1317.01 1535.36 Q1320.16 1539.14 1320.16 1546.53 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M1327.64 1532.4 L1333.85 1532.4 L1344.99 1562.31 L1356.13 1532.4 L1362.33 1532.4 L1348.96 1568.04 L1341.01 1568.04 L1327.64 1532.4 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M1386.62 1550.12 Q1379.52 1550.12 1376.78 1551.75 Q1374.04 1553.37 1374.04 1557.29 Q1374.04 1560.4 1376.08 1562.25 Q1378.15 1564.07 1381.68 1564.07 Q1386.55 1564.07 1389.48 1560.63 Q1392.44 1557.16 1392.44 1551.43 L1392.44 1550.12 L1386.62 1550.12 M1398.3 1547.71 L1398.3 1568.04 L1392.44 1568.04 L1392.44 1562.63 Q1390.44 1565.88 1387.44 1567.44 Q1384.45 1568.97 1380.12 1568.97 Q1374.65 1568.97 1371.4 1565.91 Q1368.19 1562.82 1368.19 1557.67 Q1368.19 1551.65 1372.2 1548.6 Q1376.24 1545.54 1384.23 1545.54 L1392.44 1545.54 L1392.44 1544.97 Q1392.44 1540.93 1389.77 1538.73 Q1387.13 1536.5 1382.32 1536.5 Q1379.26 1536.5 1376.37 1537.23 Q1373.47 1537.97 1370.8 1539.43 L1370.8 1534.02 Q1374.01 1532.78 1377.04 1532.17 Q1380.06 1531.54 1382.93 1531.54 Q1390.66 1531.54 1394.48 1535.55 Q1398.3 1539.56 1398.3 1547.71 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M1410.36 1518.52 L1416.22 1518.52 L1416.22 1568.04 L1410.36 1568.04 L1410.36 1518.52 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M1427.87 1553.98 L1427.87 1532.4 L1433.72 1532.4 L1433.72 1553.75 Q1433.72 1558.81 1435.7 1561.36 Q1437.67 1563.87 1441.62 1563.87 Q1446.36 1563.87 1449.1 1560.85 Q1451.87 1557.83 1451.87 1552.61 L1451.87 1532.4 L1457.72 1532.4 L1457.72 1568.04 L1451.87 1568.04 L1451.87 1562.57 Q1449.73 1565.82 1446.9 1567.41 Q1444.1 1568.97 1440.38 1568.97 Q1434.23 1568.97 1431.05 1565.15 Q1427.87 1561.33 1427.87 1553.98 M1442.6 1531.54 L1442.6 1531.54 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M1500.28 1548.76 L1500.28 1551.62 L1473.35 1551.62 Q1473.73 1557.67 1476.98 1560.85 Q1480.26 1564 1486.08 1564 Q1489.46 1564 1492.61 1563.17 Q1495.79 1562.35 1498.91 1560.69 L1498.91 1566.23 Q1495.76 1567.57 1492.45 1568.27 Q1489.14 1568.97 1485.73 1568.97 Q1477.2 1568.97 1472.2 1564 Q1467.24 1559.04 1467.24 1550.57 Q1467.24 1541.82 1471.95 1536.69 Q1476.69 1531.54 1484.71 1531.54 Q1491.91 1531.54 1496.08 1536.18 Q1500.28 1540.8 1500.28 1548.76 M1494.42 1547.04 Q1494.36 1542.23 1491.72 1539.37 Q1489.11 1536.5 1484.78 1536.5 Q1479.87 1536.5 1476.91 1539.27 Q1473.99 1542.04 1473.54 1547.07 L1494.42 1547.04 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><polyline clip-path="url(#clip572)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="242.621,1270.93 2352.76,1270.93 "/>
<polyline clip-path="url(#clip572)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="242.621,919.676 2352.76,919.676 "/>
<polyline clip-path="url(#clip572)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="242.621,568.426 2352.76,568.426 "/>
<polyline clip-path="url(#clip572)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="242.621,217.175 2352.76,217.175 "/>
<polyline clip-path="url(#clip570)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="242.621,1423.18 242.621,47.2441 "/>
<polyline clip-path="url(#clip570)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="242.621,1270.93 261.518,1270.93 "/>
<polyline clip-path="url(#clip570)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="242.621,919.676 261.518,919.676 "/>
<polyline clip-path="url(#clip570)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="242.621,568.426 261.518,568.426 "/>
<polyline clip-path="url(#clip570)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="242.621,217.175 261.518,217.175 "/>
<path clip-path="url(#clip570)" d="M115.742 1271.38 L145.417 1271.38 L145.417 1275.31 L115.742 1275.31 L115.742 1271.38 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M156.32 1284.27 L163.959 1284.27 L163.959 1257.91 L155.649 1259.57 L155.649 1255.31 L163.913 1253.65 L168.589 1253.65 L168.589 1284.27 L176.227 1284.27 L176.227 1288.21 L156.32 1288.21 L156.32 1284.27 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M185.718 1253.65 L204.074 1253.65 L204.074 1257.58 L190 1257.58 L190 1266.05 Q191.019 1265.71 192.038 1265.54 Q193.056 1265.36 194.075 1265.36 Q199.862 1265.36 203.241 1268.53 Q206.621 1271.7 206.621 1277.12 Q206.621 1282.7 203.149 1285.8 Q199.676 1288.88 193.357 1288.88 Q191.181 1288.88 188.913 1288.51 Q186.667 1288.14 184.26 1287.4 L184.26 1282.7 Q186.343 1283.83 188.565 1284.39 Q190.788 1284.94 193.264 1284.94 Q197.269 1284.94 199.607 1282.84 Q201.945 1280.73 201.945 1277.12 Q201.945 1273.51 199.607 1271.4 Q197.269 1269.29 193.264 1269.29 Q191.389 1269.29 189.514 1269.71 Q187.663 1270.13 185.718 1271.01 L185.718 1253.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M114.26 920.127 L143.936 920.127 L143.936 924.063 L114.26 924.063 L114.26 920.127 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M154.839 933.021 L162.477 933.021 L162.477 906.655 L154.167 908.322 L154.167 904.063 L162.431 902.396 L167.107 902.396 L167.107 933.021 L174.746 933.021 L174.746 936.956 L154.839 936.956 L154.839 933.021 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M197.037 906.47 L185.232 924.919 L197.037 924.919 L197.037 906.47 M195.811 902.396 L201.69 902.396 L201.69 924.919 L206.621 924.919 L206.621 928.808 L201.69 928.808 L201.69 936.956 L197.037 936.956 L197.037 928.808 L181.436 928.808 L181.436 924.294 L195.811 902.396 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M115.394 568.877 L145.07 568.877 L145.07 572.812 L115.394 572.812 L115.394 568.877 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M155.973 581.77 L163.612 581.77 L163.612 555.405 L155.302 557.071 L155.302 552.812 L163.565 551.146 L168.241 551.146 L168.241 581.77 L175.88 581.77 L175.88 585.706 L155.973 585.706 L155.973 581.77 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M199.491 567.071 Q202.848 567.789 204.723 570.057 Q206.621 572.326 206.621 575.659 Q206.621 580.775 203.102 583.576 Q199.584 586.377 193.102 586.377 Q190.926 586.377 188.612 585.937 Q186.32 585.52 183.866 584.664 L183.866 580.15 Q185.811 581.284 188.126 581.863 Q190.44 582.442 192.963 582.442 Q197.362 582.442 199.653 580.706 Q201.968 578.969 201.968 575.659 Q201.968 572.604 199.815 570.891 Q197.686 569.155 193.866 569.155 L189.838 569.155 L189.838 565.312 L194.051 565.312 Q197.5 565.312 199.329 563.946 Q201.158 562.558 201.158 559.965 Q201.158 557.303 199.26 555.891 Q197.385 554.456 193.866 554.456 Q191.945 554.456 189.746 554.872 Q187.547 555.289 184.908 556.169 L184.908 552.002 Q187.57 551.261 189.885 550.891 Q192.223 550.521 194.283 550.521 Q199.607 550.521 202.709 552.951 Q205.811 555.358 205.811 559.479 Q205.811 562.349 204.167 564.34 Q202.524 566.308 199.491 567.071 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M116.343 217.626 L146.019 217.626 L146.019 221.562 L116.343 221.562 L116.343 217.626 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M156.922 230.52 L164.561 230.52 L164.561 204.154 L156.251 205.821 L156.251 201.562 L164.515 199.895 L169.19 199.895 L169.19 230.52 L176.829 230.52 L176.829 234.455 L156.922 234.455 L156.922 230.52 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M190.301 230.52 L206.621 230.52 L206.621 234.455 L184.676 234.455 L184.676 230.52 Q187.338 227.765 191.922 223.136 Q196.528 218.483 197.709 217.14 Q199.954 214.617 200.834 212.881 Q201.737 211.122 201.737 209.432 Q201.737 206.677 199.792 204.941 Q197.871 203.205 194.769 203.205 Q192.57 203.205 190.116 203.969 Q187.686 204.733 184.908 206.284 L184.908 201.562 Q187.732 200.427 190.186 199.849 Q192.639 199.27 194.676 199.27 Q200.047 199.27 203.241 201.955 Q206.436 204.64 206.436 209.131 Q206.436 211.261 205.625 213.182 Q204.838 215.08 202.732 217.673 Q202.153 218.344 199.051 221.562 Q195.95 224.756 190.301 230.52 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M44.7161 812.969 L47.5806 812.969 L47.5806 839.896 Q53.6281 839.514 56.8109 836.268 Q59.9619 832.989 59.9619 827.165 Q59.9619 823.791 59.1344 820.64 Q58.3069 817.457 56.6518 814.338 L62.1899 814.338 Q63.5267 817.489 64.227 820.799 Q64.9272 824.109 64.9272 827.515 Q64.9272 836.045 59.9619 841.042 Q54.9967 846.007 46.5303 846.007 Q37.7774 846.007 32.6531 841.296 Q27.4968 836.554 27.4968 828.533 Q27.4968 821.34 32.1438 817.17 Q36.7589 812.969 44.7161 812.969 M42.9973 818.826 Q38.1912 818.889 35.3266 821.531 Q32.4621 824.141 32.4621 828.47 Q32.4621 833.371 35.2312 836.331 Q38.0002 839.259 43.0292 839.705 L42.9973 818.826 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M42.4881 773.724 L64.0042 773.724 L64.0042 779.581 L42.679 779.581 Q37.6183 779.581 35.1038 781.554 Q32.5894 783.528 32.5894 787.474 Q32.5894 792.217 35.6131 794.954 Q38.6368 797.691 43.8567 797.691 L64.0042 797.691 L64.0042 803.58 L28.3562 803.58 L28.3562 797.691 L33.8944 797.691 Q30.6797 795.591 29.0883 792.758 Q27.4968 789.893 27.4968 786.169 Q27.4968 780.027 31.3163 776.876 Q35.1038 773.724 42.4881 773.724 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M44.7161 731.552 L47.5806 731.552 L47.5806 758.479 Q53.6281 758.097 56.8109 754.85 Q59.9619 751.572 59.9619 745.747 Q59.9619 742.373 59.1344 739.222 Q58.3069 736.04 56.6518 732.92 L62.1899 732.92 Q63.5267 736.071 64.227 739.382 Q64.9272 742.692 64.9272 746.097 Q64.9272 754.627 59.9619 759.624 Q54.9967 764.59 46.5303 764.59 Q37.7774 764.59 32.6531 759.879 Q27.4968 755.137 27.4968 747.116 Q27.4968 739.923 32.1438 735.753 Q36.7589 731.552 44.7161 731.552 M42.9973 737.408 Q38.1912 737.472 35.3266 740.114 Q32.4621 742.724 32.4621 747.052 Q32.4621 751.954 35.2312 754.914 Q38.0002 757.842 43.0292 758.288 L42.9973 737.408 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M33.8307 701.283 Q33.2578 702.269 33.0032 703.447 Q32.7167 704.593 32.7167 705.993 Q32.7167 710.959 35.9632 713.632 Q39.1779 716.274 45.2253 716.274 L64.0042 716.274 L64.0042 722.162 L28.3562 722.162 L28.3562 716.274 L33.8944 716.274 Q30.6479 714.428 29.0883 711.468 Q27.4968 708.508 27.4968 704.275 Q27.4968 703.67 27.5923 702.938 Q27.656 702.206 27.8151 701.315 L33.8307 701.283 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M45.7664 672.828 Q39.4007 672.828 35.8996 675.47 Q32.3984 678.08 32.3984 682.822 Q32.3984 687.533 35.8996 690.175 Q39.4007 692.785 45.7664 692.785 Q52.1003 692.785 55.6014 690.175 Q59.1026 687.533 59.1026 682.822 Q59.1026 678.08 55.6014 675.47 Q52.1003 672.828 45.7664 672.828 M59.58 666.972 Q68.683 666.972 73.1071 671.014 Q77.5631 675.056 77.5631 683.395 Q77.5631 686.482 77.0857 689.22 Q76.6401 691.957 75.6852 694.535 L69.9879 694.535 Q71.3884 691.957 72.0568 689.443 Q72.7252 686.928 72.7252 684.318 Q72.7252 678.557 69.7015 675.693 Q66.7096 672.828 60.6303 672.828 L57.7339 672.828 Q60.885 674.642 62.4446 677.475 Q64.0042 680.308 64.0042 684.254 Q64.0042 690.811 59.0071 694.822 Q54.01 698.832 45.7664 698.832 Q37.491 698.832 32.4939 694.822 Q27.4968 690.811 27.4968 684.254 Q27.4968 680.308 29.0564 677.475 Q30.616 674.642 33.7671 672.828 L28.3562 672.828 L28.3562 666.972 L59.58 666.972 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip570)" d="M67.3143 640.076 Q73.68 642.559 75.6216 644.914 Q77.5631 647.27 77.5631 651.216 L77.5631 655.895 L72.6615 655.895 L72.6615 652.458 Q72.6615 650.039 71.5157 648.702 Q70.3699 647.365 66.1048 645.742 L63.4312 644.692 L28.3562 659.11 L28.3562 652.903 L56.238 641.763 L28.3562 630.623 L28.3562 624.417 L67.3143 640.076 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><circle clip-path="url(#clip572)" cx="302.342" cy="1384.24" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip572)" cx="376.071" cy="1338.19" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip572)" cx="449.8" cy="1017.46" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip572)" cx="523.53" cy="974.553" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip572)" cx="597.259" cy="974.553" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip572)" cx="670.989" cy="663.178" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip572)" cx="744.718" cy="663.178" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip572)" cx="818.447" cy="663.178" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip572)" cx="892.177" cy="663.178" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip572)" cx="965.906" cy="635.693" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip572)" cx="1039.64" cy="635.693" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip572)" cx="1113.36" cy="610.911" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip572)" cx="1187.09" cy="345.538" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip572)" cx="1260.82" cy="345.538" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip572)" cx="1334.55" cy="345.538" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip572)" cx="1408.28" cy="345.538" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip572)" cx="1482.01" cy="344.707" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip572)" cx="1555.74" cy="344.707" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip572)" cx="1629.47" cy="308.896" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip572)" cx="1703.2" cy="272.052" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip572)" cx="1776.93" cy="272.052" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip572)" cx="1850.66" cy="272.052" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip572)" cx="1924.39" cy="272.052" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip572)" cx="1998.12" cy="121.426" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip572)" cx="2071.85" cy="121.426" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip572)" cx="2145.58" cy="86.1857" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip572)" cx="2219.31" cy="86.1857" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip572)" cx="2293.04" cy="86.1857" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
</svg>

```

!!! note "Krylov dimension"
    Note that we have specified a large Krylov dimension as degenerate eigenvalues are
    notoriously difficult for iterative methods.

## Extracting momentum

Given a state, it is possible to assign a momentum label
through the use of the translation operator. This operator can be defined in MPO language
either diagramatically as

![translation operator MPO](translation_mpo.png)

or in the code as:

````julia
id = complex(isomorphism(ℂ^2, ℂ^2))
@tensor O[-1 -2; -3 -4] := id[-1, -3] * id[-2, -4]
T = periodic_boundary_conditions(DenseMPO(O), L);
````

We can then calculate the momentum of the groundstate as the expectation value of this
operator. However, there is a subtlety because of the degeneracies in the energy
eigenvalues. The eigensolver will find an orthonormal basis within each energy subspace, but
this basis is not necessarily a basis of eigenstates of the translation operator. In order
to fix this, we diagonalize the translation operator within each energy subspace.

````julia
momentum(ψᵢ, ψⱼ=ψᵢ) = angle(dot(ψᵢ, T * ψⱼ))

function fix_degeneracies(basis)
    N = zeros(ComplexF64, length(basis), length(basis))
    M = zeros(ComplexF64, length(basis), length(basis))
    for i in eachindex(basis), j in eachindex(basis)
        N[i, j] = dot(basis[i], basis[j])
        M[i, j] = momentum(basis[i], basis[j])
    end

    vals, vecs = eigen(Hermitian(N))
    M = (vecs' * M * vecs)
    M /= diagm(vals)

    vals, vecs = eigen(M)
    return angle.(vals)
end

momenta = Float64[]
append!(momenta, fix_degeneracies(states[1:1]))
append!(momenta, fix_degeneracies(states[2:2]))
append!(momenta, fix_degeneracies(states[3:3]))
append!(momenta, fix_degeneracies(states[4:5]))
append!(momenta, fix_degeneracies(states[6:9]))
append!(momenta, fix_degeneracies(states[10:11]))
append!(momenta, fix_degeneracies(states[12:12]))
append!(momenta, fix_degeneracies(states[13:16]))
append!(momenta, fix_degeneracies(states[17:18]))

plot(
    momenta,
    real.(energies[1:18]);
    seriestype=:scatter,
    xlabel="momentum",
    ylabel="energy",
    legend=false,
)
````

```@raw html
<?xml version="1.0" encoding="utf-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="600" height="400" viewBox="0 0 2400 1600">
<defs>
  <clipPath id="clip610">
    <rect x="0" y="0" width="2400" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip610)" d="M0 1600 L2400 1600 L2400 0 L0 0  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip611">
    <rect x="480" y="0" width="1681" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip610)" d="M287.366 1423.18 L2352.76 1423.18 L2352.76 47.2441 L287.366 47.2441  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip612">
    <rect x="287" y="47" width="2066" height="1377"/>
  </clipPath>
</defs>
<polyline clip-path="url(#clip612)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="389.73,1423.18 389.73,47.2441 "/>
<polyline clip-path="url(#clip612)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="699.84,1423.18 699.84,47.2441 "/>
<polyline clip-path="url(#clip612)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1009.95,1423.18 1009.95,47.2441 "/>
<polyline clip-path="url(#clip612)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1320.06,1423.18 1320.06,47.2441 "/>
<polyline clip-path="url(#clip612)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1630.17,1423.18 1630.17,47.2441 "/>
<polyline clip-path="url(#clip612)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1940.28,1423.18 1940.28,47.2441 "/>
<polyline clip-path="url(#clip612)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="2250.39,1423.18 2250.39,47.2441 "/>
<polyline clip-path="url(#clip610)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="287.366,1423.18 2352.76,1423.18 "/>
<polyline clip-path="url(#clip610)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="389.73,1423.18 389.73,1404.28 "/>
<polyline clip-path="url(#clip610)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="699.84,1423.18 699.84,1404.28 "/>
<polyline clip-path="url(#clip610)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1009.95,1423.18 1009.95,1404.28 "/>
<polyline clip-path="url(#clip610)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1320.06,1423.18 1320.06,1404.28 "/>
<polyline clip-path="url(#clip610)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1630.17,1423.18 1630.17,1404.28 "/>
<polyline clip-path="url(#clip610)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1940.28,1423.18 1940.28,1404.28 "/>
<polyline clip-path="url(#clip610)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="2250.39,1423.18 2250.39,1404.28 "/>
<path clip-path="url(#clip610)" d="M359.197 1468.75 L388.873 1468.75 L388.873 1472.69 L359.197 1472.69 L359.197 1468.75 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M413.132 1466.95 Q416.489 1467.66 418.364 1469.93 Q420.262 1472.2 420.262 1475.53 Q420.262 1480.65 416.743 1483.45 Q413.225 1486.25 406.743 1486.25 Q404.568 1486.25 402.253 1485.81 Q399.961 1485.39 397.507 1484.54 L397.507 1480.02 Q399.452 1481.16 401.767 1481.74 Q404.081 1482.32 406.605 1482.32 Q411.003 1482.32 413.294 1480.58 Q415.609 1478.84 415.609 1475.53 Q415.609 1472.48 413.456 1470.77 Q411.327 1469.03 407.507 1469.03 L403.48 1469.03 L403.48 1465.19 L407.693 1465.19 Q411.142 1465.19 412.97 1463.82 Q414.799 1462.43 414.799 1459.84 Q414.799 1457.18 412.901 1455.77 Q411.026 1454.33 407.507 1454.33 Q405.586 1454.33 403.387 1454.75 Q401.188 1455.16 398.549 1456.04 L398.549 1451.88 Q401.211 1451.14 403.526 1450.77 Q405.864 1450.39 407.924 1450.39 Q413.248 1450.39 416.35 1452.83 Q419.452 1455.23 419.452 1459.35 Q419.452 1462.22 417.808 1464.21 Q416.165 1466.18 413.132 1466.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M669.782 1468.75 L699.458 1468.75 L699.458 1472.69 L669.782 1472.69 L669.782 1468.75 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M713.578 1481.64 L729.898 1481.64 L729.898 1485.58 L707.953 1485.58 L707.953 1481.64 Q710.616 1478.89 715.199 1474.26 Q719.805 1469.61 720.986 1468.27 Q723.231 1465.74 724.111 1464.01 Q725.014 1462.25 725.014 1460.56 Q725.014 1457.8 723.069 1456.07 Q721.148 1454.33 718.046 1454.33 Q715.847 1454.33 713.393 1455.09 Q710.963 1455.86 708.185 1457.41 L708.185 1452.69 Q711.009 1451.55 713.463 1450.97 Q715.916 1450.39 717.953 1450.39 Q723.324 1450.39 726.518 1453.08 Q729.713 1455.77 729.713 1460.26 Q729.713 1462.39 728.902 1464.31 Q728.115 1466.2 726.009 1468.8 Q725.43 1469.47 722.328 1472.69 Q719.227 1475.88 713.578 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M979.708 1468.75 L1009.38 1468.75 L1009.38 1472.69 L979.708 1472.69 L979.708 1468.75 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M1020.29 1481.64 L1027.92 1481.64 L1027.92 1455.28 L1019.61 1456.95 L1019.61 1452.69 L1027.88 1451.02 L1032.55 1451.02 L1032.55 1481.64 L1040.19 1481.64 L1040.19 1485.58 L1020.29 1485.58 L1020.29 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M1320.06 1454.1 Q1316.45 1454.1 1314.62 1457.66 Q1312.82 1461.2 1312.82 1468.33 Q1312.82 1475.44 1314.62 1479.01 Q1316.45 1482.55 1320.06 1482.55 Q1323.7 1482.55 1325.5 1479.01 Q1327.33 1475.44 1327.33 1468.33 Q1327.33 1461.2 1325.5 1457.66 Q1323.7 1454.1 1320.06 1454.1 M1320.06 1450.39 Q1325.87 1450.39 1328.93 1455 Q1332.01 1459.58 1332.01 1468.33 Q1332.01 1477.06 1328.93 1481.67 Q1325.87 1486.25 1320.06 1486.25 Q1314.25 1486.25 1311.17 1481.67 Q1308.12 1477.06 1308.12 1468.33 Q1308.12 1459.58 1311.17 1455 Q1314.25 1450.39 1320.06 1450.39 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M1620.55 1481.64 L1628.19 1481.64 L1628.19 1455.28 L1619.88 1456.95 L1619.88 1452.69 L1628.15 1451.02 L1632.82 1451.02 L1632.82 1481.64 L1640.46 1481.64 L1640.46 1485.58 L1620.55 1485.58 L1620.55 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M1934.93 1481.64 L1951.25 1481.64 L1951.25 1485.58 L1929.31 1485.58 L1929.31 1481.64 Q1931.97 1478.89 1936.55 1474.26 Q1941.16 1469.61 1942.34 1468.27 Q1944.59 1465.74 1945.47 1464.01 Q1946.37 1462.25 1946.37 1460.56 Q1946.37 1457.8 1944.43 1456.07 Q1942.5 1454.33 1939.4 1454.33 Q1937.2 1454.33 1934.75 1455.09 Q1932.32 1455.86 1929.54 1457.41 L1929.54 1452.69 Q1932.37 1451.55 1934.82 1450.97 Q1937.27 1450.39 1939.31 1450.39 Q1944.68 1450.39 1947.87 1453.08 Q1951.07 1455.77 1951.07 1460.26 Q1951.07 1462.39 1950.26 1464.31 Q1949.47 1466.2 1947.37 1468.8 Q1946.79 1469.47 1943.68 1472.69 Q1940.58 1475.88 1934.93 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M2254.64 1466.95 Q2258 1467.66 2259.87 1469.93 Q2261.77 1472.2 2261.77 1475.53 Q2261.77 1480.65 2258.25 1483.45 Q2254.73 1486.25 2248.25 1486.25 Q2246.08 1486.25 2243.76 1485.81 Q2241.47 1485.39 2239.01 1484.54 L2239.01 1480.02 Q2240.96 1481.16 2243.27 1481.74 Q2245.59 1482.32 2248.11 1482.32 Q2252.51 1482.32 2254.8 1480.58 Q2257.12 1478.84 2257.12 1475.53 Q2257.12 1472.48 2254.96 1470.77 Q2252.83 1469.03 2249.01 1469.03 L2244.99 1469.03 L2244.99 1465.19 L2249.2 1465.19 Q2252.65 1465.19 2254.48 1463.82 Q2256.31 1462.43 2256.31 1459.84 Q2256.31 1457.18 2254.41 1455.77 Q2252.53 1454.33 2249.01 1454.33 Q2247.09 1454.33 2244.89 1454.75 Q2242.7 1455.16 2240.06 1456.04 L2240.06 1451.88 Q2242.72 1451.14 2245.03 1450.77 Q2247.37 1450.39 2249.43 1450.39 Q2254.76 1450.39 2257.86 1452.83 Q2260.96 1455.23 2260.96 1459.35 Q2260.96 1462.22 2259.32 1464.21 Q2257.67 1466.18 2254.64 1466.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M1164.44 1539.24 Q1166.63 1535.29 1169.69 1533.41 Q1172.74 1531.54 1176.88 1531.54 Q1182.45 1531.54 1185.47 1535.45 Q1188.5 1539.33 1188.5 1546.53 L1188.5 1568.04 L1182.61 1568.04 L1182.61 1546.72 Q1182.61 1541.59 1180.8 1539.11 Q1178.98 1536.63 1175.26 1536.63 Q1170.71 1536.63 1168.06 1539.65 Q1165.42 1542.68 1165.42 1547.9 L1165.42 1568.04 L1159.53 1568.04 L1159.53 1546.72 Q1159.53 1541.56 1157.72 1539.11 Q1155.91 1536.63 1152.12 1536.63 Q1147.63 1536.63 1144.99 1539.68 Q1142.35 1542.71 1142.35 1547.9 L1142.35 1568.04 L1136.46 1568.04 L1136.46 1532.4 L1142.35 1532.4 L1142.35 1537.93 Q1144.35 1534.66 1147.15 1533.1 Q1149.95 1531.54 1153.8 1531.54 Q1157.69 1531.54 1160.39 1533.51 Q1163.13 1535.48 1164.44 1539.24 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M1213.99 1536.5 Q1209.28 1536.5 1206.54 1540.19 Q1203.81 1543.85 1203.81 1550.25 Q1203.81 1556.65 1206.51 1560.34 Q1209.25 1564 1213.99 1564 Q1218.67 1564 1221.41 1560.31 Q1224.15 1556.62 1224.15 1550.25 Q1224.15 1543.92 1221.41 1540.23 Q1218.67 1536.5 1213.99 1536.5 M1213.99 1531.54 Q1221.63 1531.54 1225.99 1536.5 Q1230.35 1541.47 1230.35 1550.25 Q1230.35 1559 1225.99 1564 Q1221.63 1568.97 1213.99 1568.97 Q1206.32 1568.97 1201.96 1564 Q1197.63 1559 1197.63 1550.25 Q1197.63 1541.47 1201.96 1536.5 Q1206.32 1531.54 1213.99 1531.54 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M1267.81 1539.24 Q1270.01 1535.29 1273.07 1533.41 Q1276.12 1531.54 1280.26 1531.54 Q1285.83 1531.54 1288.85 1535.45 Q1291.88 1539.33 1291.88 1546.53 L1291.88 1568.04 L1285.99 1568.04 L1285.99 1546.72 Q1285.99 1541.59 1284.17 1539.11 Q1282.36 1536.63 1278.64 1536.63 Q1274.08 1536.63 1271.44 1539.65 Q1268.8 1542.68 1268.8 1547.9 L1268.8 1568.04 L1262.91 1568.04 L1262.91 1546.72 Q1262.91 1541.56 1261.1 1539.11 Q1259.28 1536.63 1255.5 1536.63 Q1251.01 1536.63 1248.37 1539.68 Q1245.73 1542.71 1245.73 1547.9 L1245.73 1568.04 L1239.84 1568.04 L1239.84 1532.4 L1245.73 1532.4 L1245.73 1537.93 Q1247.73 1534.66 1250.53 1533.1 Q1253.33 1531.54 1257.18 1531.54 Q1261.07 1531.54 1263.77 1533.51 Q1266.51 1535.48 1267.81 1539.24 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M1334.05 1548.76 L1334.05 1551.62 L1307.12 1551.62 Q1307.5 1557.67 1310.75 1560.85 Q1314.03 1564 1319.85 1564 Q1323.23 1564 1326.38 1563.17 Q1329.56 1562.35 1332.68 1560.69 L1332.68 1566.23 Q1329.53 1567.57 1326.22 1568.27 Q1322.91 1568.97 1319.5 1568.97 Q1310.97 1568.97 1305.98 1564 Q1301.01 1559.04 1301.01 1550.57 Q1301.01 1541.82 1305.72 1536.69 Q1310.46 1531.54 1318.49 1531.54 Q1325.68 1531.54 1329.85 1536.18 Q1334.05 1540.8 1334.05 1548.76 M1328.19 1547.04 Q1328.13 1542.23 1325.49 1539.37 Q1322.88 1536.5 1318.55 1536.5 Q1313.65 1536.5 1310.69 1539.27 Q1307.76 1542.04 1307.31 1547.07 L1328.19 1547.04 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M1373.29 1546.53 L1373.29 1568.04 L1367.44 1568.04 L1367.44 1546.72 Q1367.44 1541.66 1365.46 1539.14 Q1363.49 1536.63 1359.54 1536.63 Q1354.8 1536.63 1352.06 1539.65 Q1349.33 1542.68 1349.33 1547.9 L1349.33 1568.04 L1343.44 1568.04 L1343.44 1532.4 L1349.33 1532.4 L1349.33 1537.93 Q1351.43 1534.72 1354.26 1533.13 Q1357.13 1531.54 1360.85 1531.54 Q1366.99 1531.54 1370.14 1535.36 Q1373.29 1539.14 1373.29 1546.53 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M1390.77 1522.27 L1390.77 1532.4 L1402.83 1532.4 L1402.83 1536.95 L1390.77 1536.95 L1390.77 1556.3 Q1390.77 1560.66 1391.95 1561.9 Q1393.16 1563.14 1396.82 1563.14 L1402.83 1563.14 L1402.83 1568.04 L1396.82 1568.04 Q1390.04 1568.04 1387.46 1565.53 Q1384.88 1562.98 1384.88 1556.3 L1384.88 1536.95 L1380.58 1536.95 L1380.58 1532.4 L1384.88 1532.4 L1384.88 1522.27 L1390.77 1522.27 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M1409.93 1553.98 L1409.93 1532.4 L1415.79 1532.4 L1415.79 1553.75 Q1415.79 1558.81 1417.76 1561.36 Q1419.73 1563.87 1423.68 1563.87 Q1428.42 1563.87 1431.16 1560.85 Q1433.93 1557.83 1433.93 1552.61 L1433.93 1532.4 L1439.78 1532.4 L1439.78 1568.04 L1433.93 1568.04 L1433.93 1562.57 Q1431.79 1565.82 1428.96 1567.41 Q1426.16 1568.97 1422.44 1568.97 Q1416.29 1568.97 1413.11 1565.15 Q1409.93 1561.33 1409.93 1553.98 M1424.67 1531.54 L1424.67 1531.54 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M1479.6 1539.24 Q1481.8 1535.29 1484.85 1533.41 Q1487.91 1531.54 1492.05 1531.54 Q1497.62 1531.54 1500.64 1535.45 Q1503.66 1539.33 1503.66 1546.53 L1503.66 1568.04 L1497.78 1568.04 L1497.78 1546.72 Q1497.78 1541.59 1495.96 1539.11 Q1494.15 1536.63 1490.42 1536.63 Q1485.87 1536.63 1483.23 1539.65 Q1480.59 1542.68 1480.59 1547.9 L1480.59 1568.04 L1474.7 1568.04 L1474.7 1546.72 Q1474.7 1541.56 1472.89 1539.11 Q1471.07 1536.63 1467.28 1536.63 Q1462.8 1536.63 1460.15 1539.68 Q1457.51 1542.71 1457.51 1547.9 L1457.51 1568.04 L1451.62 1568.04 L1451.62 1532.4 L1457.51 1532.4 L1457.51 1537.93 Q1459.52 1534.66 1462.32 1533.1 Q1465.12 1531.54 1468.97 1531.54 Q1472.85 1531.54 1475.56 1533.51 Q1478.3 1535.48 1479.6 1539.24 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><polyline clip-path="url(#clip612)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="287.366,1242.75 2352.76,1242.75 "/>
<polyline clip-path="url(#clip612)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="287.366,1023.45 2352.76,1023.45 "/>
<polyline clip-path="url(#clip612)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="287.366,804.144 2352.76,804.144 "/>
<polyline clip-path="url(#clip612)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="287.366,584.842 2352.76,584.842 "/>
<polyline clip-path="url(#clip612)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="287.366,365.54 2352.76,365.54 "/>
<polyline clip-path="url(#clip612)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="287.366,146.239 2352.76,146.239 "/>
<polyline clip-path="url(#clip610)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="287.366,1423.18 287.366,47.2441 "/>
<polyline clip-path="url(#clip610)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="287.366,1242.75 306.264,1242.75 "/>
<polyline clip-path="url(#clip610)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="287.366,1023.45 306.264,1023.45 "/>
<polyline clip-path="url(#clip610)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="287.366,804.144 306.264,804.144 "/>
<polyline clip-path="url(#clip610)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="287.366,584.842 306.264,584.842 "/>
<polyline clip-path="url(#clip610)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="287.366,365.54 306.264,365.54 "/>
<polyline clip-path="url(#clip610)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="287.366,146.239 306.264,146.239 "/>
<path clip-path="url(#clip610)" d="M114.26 1243.2 L143.936 1243.2 L143.936 1247.13 L114.26 1247.13 L114.26 1243.2 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M154.839 1256.09 L162.477 1256.09 L162.477 1229.73 L154.167 1231.39 L154.167 1227.13 L162.431 1225.47 L167.107 1225.47 L167.107 1256.09 L174.746 1256.09 L174.746 1260.03 L154.839 1260.03 L154.839 1256.09 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M184.237 1225.47 L202.593 1225.47 L202.593 1229.4 L188.519 1229.4 L188.519 1237.87 Q189.538 1237.53 190.556 1237.37 Q191.575 1237.18 192.593 1237.18 Q198.38 1237.18 201.76 1240.35 Q205.139 1243.52 205.139 1248.94 Q205.139 1254.52 201.667 1257.62 Q198.195 1260.7 191.875 1260.7 Q189.7 1260.7 187.431 1260.33 Q185.186 1259.96 182.778 1259.22 L182.778 1254.52 Q184.862 1255.65 187.084 1256.21 Q189.306 1256.76 191.783 1256.76 Q195.787 1256.76 198.125 1254.66 Q200.463 1252.55 200.463 1248.94 Q200.463 1245.33 198.125 1243.22 Q195.787 1241.12 191.783 1241.12 Q189.908 1241.12 188.033 1241.53 Q186.181 1241.95 184.237 1242.83 L184.237 1225.47 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M214.352 1254.15 L219.236 1254.15 L219.236 1260.03 L214.352 1260.03 L214.352 1254.15 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M239.422 1228.55 Q235.81 1228.55 233.982 1232.11 Q232.176 1235.65 232.176 1242.78 Q232.176 1249.89 233.982 1253.45 Q235.81 1256.99 239.422 1256.99 Q243.056 1256.99 244.861 1253.45 Q246.69 1249.89 246.69 1242.78 Q246.69 1235.65 244.861 1232.11 Q243.056 1228.55 239.422 1228.55 M239.422 1224.84 Q245.232 1224.84 248.287 1229.45 Q251.366 1234.03 251.366 1242.78 Q251.366 1251.51 248.287 1256.11 Q245.232 1260.7 239.422 1260.7 Q233.611 1260.7 230.533 1256.11 Q227.477 1251.51 227.477 1242.78 Q227.477 1234.03 230.533 1229.45 Q233.611 1224.84 239.422 1224.84 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M115.256 1023.9 L144.931 1023.9 L144.931 1027.83 L115.256 1027.83 L115.256 1023.9 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M155.834 1036.79 L163.473 1036.79 L163.473 1010.42 L155.163 1012.09 L155.163 1007.83 L163.427 1006.17 L168.102 1006.17 L168.102 1036.79 L175.741 1036.79 L175.741 1040.73 L155.834 1040.73 L155.834 1036.79 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M198.033 1010.24 L186.227 1028.69 L198.033 1028.69 L198.033 1010.24 M196.806 1006.17 L202.686 1006.17 L202.686 1028.69 L207.616 1028.69 L207.616 1032.58 L202.686 1032.58 L202.686 1040.73 L198.033 1040.73 L198.033 1032.58 L182.431 1032.58 L182.431 1028.06 L196.806 1006.17 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M215.348 1034.85 L220.232 1034.85 L220.232 1040.73 L215.348 1040.73 L215.348 1034.85 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M230.463 1006.17 L248.82 1006.17 L248.82 1010.1 L234.746 1010.1 L234.746 1018.57 Q235.764 1018.23 236.783 1018.06 Q237.801 1017.88 238.82 1017.88 Q244.607 1017.88 247.986 1021.05 Q251.366 1024.22 251.366 1029.64 Q251.366 1035.22 247.894 1038.32 Q244.421 1041.4 238.102 1041.4 Q235.926 1041.4 233.658 1041.03 Q231.412 1040.66 229.005 1039.92 L229.005 1035.22 Q231.088 1036.35 233.31 1036.91 Q235.533 1037.46 238.009 1037.46 Q242.014 1037.46 244.352 1035.35 Q246.69 1033.25 246.69 1029.64 Q246.69 1026.03 244.352 1023.92 Q242.014 1021.81 238.009 1021.81 Q236.135 1021.81 234.26 1022.23 Q232.408 1022.65 230.463 1023.53 L230.463 1006.17 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M114.26 804.595 L143.936 804.595 L143.936 808.53 L114.26 808.53 L114.26 804.595 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M154.839 817.489 L162.477 817.489 L162.477 791.123 L154.167 792.79 L154.167 788.53 L162.431 786.864 L167.107 786.864 L167.107 817.489 L174.746 817.489 L174.746 821.424 L154.839 821.424 L154.839 817.489 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M197.037 790.938 L185.232 809.387 L197.037 809.387 L197.037 790.938 M195.811 786.864 L201.69 786.864 L201.69 809.387 L206.621 809.387 L206.621 813.276 L201.69 813.276 L201.69 821.424 L197.037 821.424 L197.037 813.276 L181.436 813.276 L181.436 808.762 L195.811 786.864 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M214.352 815.544 L219.236 815.544 L219.236 821.424 L214.352 821.424 L214.352 815.544 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M239.422 789.942 Q235.81 789.942 233.982 793.507 Q232.176 797.049 232.176 804.178 Q232.176 811.285 233.982 814.85 Q235.81 818.391 239.422 818.391 Q243.056 818.391 244.861 814.85 Q246.69 811.285 246.69 804.178 Q246.69 797.049 244.861 793.507 Q243.056 789.942 239.422 789.942 M239.422 786.239 Q245.232 786.239 248.287 790.845 Q251.366 795.428 251.366 804.178 Q251.366 812.905 248.287 817.512 Q245.232 822.095 239.422 822.095 Q233.611 822.095 230.533 817.512 Q227.477 812.905 227.477 804.178 Q227.477 795.428 230.533 790.845 Q233.611 786.239 239.422 786.239 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M115.256 585.293 L144.931 585.293 L144.931 589.229 L115.256 589.229 L115.256 585.293 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M155.834 598.187 L163.473 598.187 L163.473 571.821 L155.163 573.488 L155.163 569.229 L163.427 567.562 L168.102 567.562 L168.102 598.187 L175.741 598.187 L175.741 602.122 L155.834 602.122 L155.834 598.187 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M199.352 583.488 Q202.709 584.205 204.584 586.474 Q206.482 588.742 206.482 592.076 Q206.482 597.192 202.963 599.992 Q199.445 602.793 192.963 602.793 Q190.788 602.793 188.473 602.354 Q186.181 601.937 183.727 601.08 L183.727 596.567 Q185.672 597.701 187.987 598.279 Q190.301 598.858 192.825 598.858 Q197.223 598.858 199.514 597.122 Q201.829 595.386 201.829 592.076 Q201.829 589.02 199.676 587.307 Q197.547 585.571 193.727 585.571 L189.7 585.571 L189.7 581.729 L193.913 581.729 Q197.362 581.729 199.19 580.363 Q201.019 578.974 201.019 576.381 Q201.019 573.719 199.121 572.307 Q197.246 570.872 193.727 570.872 Q191.806 570.872 189.607 571.289 Q187.408 571.706 184.769 572.585 L184.769 568.419 Q187.431 567.678 189.746 567.307 Q192.084 566.937 194.144 566.937 Q199.468 566.937 202.57 569.368 Q205.672 571.775 205.672 575.895 Q205.672 578.766 204.028 580.756 Q202.385 582.724 199.352 583.488 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M215.348 596.242 L220.232 596.242 L220.232 602.122 L215.348 602.122 L215.348 596.242 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M230.463 567.562 L248.82 567.562 L248.82 571.497 L234.746 571.497 L234.746 579.969 Q235.764 579.622 236.783 579.46 Q237.801 579.275 238.82 579.275 Q244.607 579.275 247.986 582.446 Q251.366 585.617 251.366 591.034 Q251.366 596.613 247.894 599.715 Q244.421 602.793 238.102 602.793 Q235.926 602.793 233.658 602.423 Q231.412 602.053 229.005 601.312 L229.005 596.613 Q231.088 597.747 233.31 598.303 Q235.533 598.858 238.009 598.858 Q242.014 598.858 244.352 596.752 Q246.69 594.645 246.69 591.034 Q246.69 587.423 244.352 585.317 Q242.014 583.21 238.009 583.21 Q236.135 583.21 234.26 583.627 Q232.408 584.043 230.463 584.923 L230.463 567.562 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M114.26 365.992 L143.936 365.992 L143.936 369.927 L114.26 369.927 L114.26 365.992 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M154.839 378.885 L162.477 378.885 L162.477 352.52 L154.167 354.186 L154.167 349.927 L162.431 348.26 L167.107 348.26 L167.107 378.885 L174.746 378.885 L174.746 382.82 L154.839 382.82 L154.839 378.885 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M198.357 364.186 Q201.713 364.904 203.588 367.172 Q205.487 369.441 205.487 372.774 Q205.487 377.89 201.968 380.691 Q198.45 383.492 191.968 383.492 Q189.792 383.492 187.477 383.052 Q185.186 382.635 182.732 381.779 L182.732 377.265 Q184.676 378.399 186.991 378.978 Q189.306 379.557 191.829 379.557 Q196.227 379.557 198.519 377.82 Q200.834 376.084 200.834 372.774 Q200.834 369.719 198.681 368.006 Q196.551 366.27 192.732 366.27 L188.704 366.27 L188.704 362.427 L192.917 362.427 Q196.366 362.427 198.195 361.061 Q200.024 359.672 200.024 357.08 Q200.024 354.418 198.125 353.006 Q196.25 351.571 192.732 351.571 Q190.811 351.571 188.612 351.987 Q186.413 352.404 183.774 353.284 L183.774 349.117 Q186.436 348.376 188.75 348.006 Q191.088 347.635 193.149 347.635 Q198.473 347.635 201.575 350.066 Q204.676 352.473 204.676 356.594 Q204.676 359.464 203.033 361.455 Q201.389 363.422 198.357 364.186 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M214.352 376.941 L219.236 376.941 L219.236 382.82 L214.352 382.82 L214.352 376.941 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M239.422 351.339 Q235.81 351.339 233.982 354.904 Q232.176 358.446 232.176 365.575 Q232.176 372.682 233.982 376.246 Q235.81 379.788 239.422 379.788 Q243.056 379.788 244.861 376.246 Q246.69 372.682 246.69 365.575 Q246.69 358.446 244.861 354.904 Q243.056 351.339 239.422 351.339 M239.422 347.635 Q245.232 347.635 248.287 352.242 Q251.366 356.825 251.366 365.575 Q251.366 374.302 248.287 378.908 Q245.232 383.492 239.422 383.492 Q233.611 383.492 230.533 378.908 Q227.477 374.302 227.477 365.575 Q227.477 356.825 230.533 352.242 Q233.611 347.635 239.422 347.635 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M115.256 146.69 L144.931 146.69 L144.931 150.625 L115.256 150.625 L115.256 146.69 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M155.834 159.584 L163.473 159.584 L163.473 133.218 L155.163 134.885 L155.163 130.625 L163.427 128.959 L168.102 128.959 L168.102 159.584 L175.741 159.584 L175.741 163.519 L155.834 163.519 L155.834 159.584 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M189.213 159.584 L205.533 159.584 L205.533 163.519 L183.588 163.519 L183.588 159.584 Q186.251 156.829 190.834 152.199 Q195.44 147.547 196.621 146.204 Q198.866 143.681 199.746 141.945 Q200.649 140.186 200.649 138.496 Q200.649 135.741 198.704 134.005 Q196.783 132.269 193.681 132.269 Q191.482 132.269 189.028 133.033 Q186.598 133.797 183.82 135.348 L183.82 130.625 Q186.644 129.491 189.098 128.912 Q191.551 128.334 193.588 128.334 Q198.959 128.334 202.153 131.019 Q205.348 133.704 205.348 138.195 Q205.348 140.324 204.537 142.246 Q203.75 144.144 201.644 146.736 Q201.065 147.408 197.963 150.625 Q194.862 153.82 189.213 159.584 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M215.348 157.639 L220.232 157.639 L220.232 163.519 L215.348 163.519 L215.348 157.639 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M230.463 128.959 L248.82 128.959 L248.82 132.894 L234.746 132.894 L234.746 141.366 Q235.764 141.019 236.783 140.857 Q237.801 140.672 238.82 140.672 Q244.607 140.672 247.986 143.843 Q251.366 147.014 251.366 152.431 Q251.366 158.01 247.894 161.111 Q244.421 164.19 238.102 164.19 Q235.926 164.19 233.658 163.82 Q231.412 163.449 229.005 162.709 L229.005 158.01 Q231.088 159.144 233.31 159.699 Q235.533 160.255 238.009 160.255 Q242.014 160.255 244.352 158.148 Q246.69 156.042 246.69 152.431 Q246.69 148.82 244.352 146.713 Q242.014 144.607 238.009 144.607 Q236.135 144.607 234.26 145.023 Q232.408 145.44 230.463 146.32 L230.463 128.959 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M44.7161 812.969 L47.5806 812.969 L47.5806 839.896 Q53.6281 839.514 56.8109 836.268 Q59.9619 832.989 59.9619 827.165 Q59.9619 823.791 59.1344 820.64 Q58.3069 817.457 56.6518 814.338 L62.1899 814.338 Q63.5267 817.489 64.227 820.799 Q64.9272 824.109 64.9272 827.515 Q64.9272 836.045 59.9619 841.042 Q54.9967 846.007 46.5303 846.007 Q37.7774 846.007 32.6531 841.296 Q27.4968 836.554 27.4968 828.533 Q27.4968 821.34 32.1438 817.17 Q36.7589 812.969 44.7161 812.969 M42.9973 818.826 Q38.1912 818.889 35.3266 821.531 Q32.4621 824.141 32.4621 828.47 Q32.4621 833.371 35.2312 836.331 Q38.0002 839.259 43.0292 839.705 L42.9973 818.826 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M42.4881 773.724 L64.0042 773.724 L64.0042 779.581 L42.679 779.581 Q37.6183 779.581 35.1038 781.554 Q32.5894 783.528 32.5894 787.474 Q32.5894 792.217 35.6131 794.954 Q38.6368 797.691 43.8567 797.691 L64.0042 797.691 L64.0042 803.58 L28.3562 803.58 L28.3562 797.691 L33.8944 797.691 Q30.6797 795.591 29.0883 792.758 Q27.4968 789.893 27.4968 786.169 Q27.4968 780.027 31.3163 776.876 Q35.1038 773.724 42.4881 773.724 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M44.7161 731.552 L47.5806 731.552 L47.5806 758.479 Q53.6281 758.097 56.8109 754.85 Q59.9619 751.572 59.9619 745.747 Q59.9619 742.373 59.1344 739.222 Q58.3069 736.04 56.6518 732.92 L62.1899 732.92 Q63.5267 736.071 64.227 739.382 Q64.9272 742.692 64.9272 746.097 Q64.9272 754.627 59.9619 759.624 Q54.9967 764.59 46.5303 764.59 Q37.7774 764.59 32.6531 759.879 Q27.4968 755.137 27.4968 747.116 Q27.4968 739.923 32.1438 735.753 Q36.7589 731.552 44.7161 731.552 M42.9973 737.408 Q38.1912 737.472 35.3266 740.114 Q32.4621 742.724 32.4621 747.052 Q32.4621 751.954 35.2312 754.914 Q38.0002 757.842 43.0292 758.288 L42.9973 737.408 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M33.8307 701.283 Q33.2578 702.269 33.0032 703.447 Q32.7167 704.593 32.7167 705.993 Q32.7167 710.959 35.9632 713.632 Q39.1779 716.274 45.2253 716.274 L64.0042 716.274 L64.0042 722.162 L28.3562 722.162 L28.3562 716.274 L33.8944 716.274 Q30.6479 714.428 29.0883 711.468 Q27.4968 708.508 27.4968 704.275 Q27.4968 703.67 27.5923 702.938 Q27.656 702.206 27.8151 701.315 L33.8307 701.283 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M45.7664 672.828 Q39.4007 672.828 35.8996 675.47 Q32.3984 678.08 32.3984 682.822 Q32.3984 687.533 35.8996 690.175 Q39.4007 692.785 45.7664 692.785 Q52.1003 692.785 55.6014 690.175 Q59.1026 687.533 59.1026 682.822 Q59.1026 678.08 55.6014 675.47 Q52.1003 672.828 45.7664 672.828 M59.58 666.972 Q68.683 666.972 73.1071 671.014 Q77.5631 675.056 77.5631 683.395 Q77.5631 686.482 77.0857 689.22 Q76.6401 691.957 75.6852 694.535 L69.9879 694.535 Q71.3884 691.957 72.0568 689.443 Q72.7252 686.928 72.7252 684.318 Q72.7252 678.557 69.7015 675.693 Q66.7096 672.828 60.6303 672.828 L57.7339 672.828 Q60.885 674.642 62.4446 677.475 Q64.0042 680.308 64.0042 684.254 Q64.0042 690.811 59.0071 694.822 Q54.01 698.832 45.7664 698.832 Q37.491 698.832 32.4939 694.822 Q27.4968 690.811 27.4968 684.254 Q27.4968 680.308 29.0564 677.475 Q30.616 674.642 33.7671 672.828 L28.3562 672.828 L28.3562 666.972 L59.58 666.972 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip610)" d="M67.3143 640.076 Q73.68 642.559 75.6216 644.914 Q77.5631 647.27 77.5631 651.216 L77.5631 655.895 L72.6615 655.895 L72.6615 652.458 Q72.6615 650.039 71.5157 648.702 Q70.3699 647.365 66.1048 645.742 L63.4312 644.692 L28.3562 659.11 L28.3562 652.903 L56.238 641.763 L28.3562 630.623 L28.3562 624.417 L67.3143 640.076 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><circle clip-path="url(#clip612)" cx="2294.3" cy="1384.24" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip612)" cx="1320.06" cy="1326.74" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip612)" cx="2294.3" cy="926.245" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip612)" cx="2294.3" cy="872.668" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip612)" cx="1320.06" cy="872.668" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip612)" cx="612.483" cy="483.857" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip612)" cx="2027.64" cy="483.857" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip612)" cx="1012.25" cy="483.857" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip612)" cx="1627.87" cy="483.857" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip612)" cx="2294.3" cy="449.537" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip612)" cx="1320.06" cy="449.537" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip612)" cx="1320.06" cy="418.592" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip612)" cx="2294.3" cy="87.2223" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip612)" cx="345.82" cy="87.2223" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip612)" cx="1443.38" cy="87.2223" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip612)" cx="1196.74" cy="87.2223" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip612)" cx="2294.3" cy="86.1857" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip612)" cx="1320.06" cy="86.1857" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
</svg>

```

## Finite bond dimension

If we limit the maximum bond dimension of the MPS, we get an approximate solution, but we
can reach higher system sizes.

````julia
L_mps = 20
H_mps = periodic_boundary_conditions(transverse_field_ising(), L_mps)
D = 64
ψ, envs, δ = find_groundstate(FiniteMPS(L_mps, ℂ^2, ℂ^D), H_mps, DMRG());
````

````
┌ Info: DMRG iteration:
│   iter = 1
│   ϵ = 0.0004684927525136157
│   λ = -25.490989686356734 - 9.994538742425174e-16im
└   Δt = 2.896811139
┌ Info: DMRG iteration:
│   iter = 2
│   ϵ = 1.62661220919084e-7
│   λ = -25.49098968636408 - 9.75256839110213e-16im
└   Δt = 0.750456713
┌ Info: DMRG iteration:
│   iter = 3
│   ϵ = 2.2905332934820245e-8
│   λ = -25.490989686364177 + 1.385913948353531e-15im
└   Δt = 0.368387179
┌ Info: DMRG iteration:
│   iter = 4
│   ϵ = 1.2594537272422253e-8
│   λ = -25.49098968636423 + 4.313902137144602e-15im
└   Δt = 0.353520181
┌ Info: DMRG iteration:
│   iter = 5
│   ϵ = 7.0420380377351706e-9
│   λ = -25.4909896863642 + 4.680276128327185e-17im
└   Δt = 0.339315185
┌ Info: DMRG iteration:
│   iter = 6
│   ϵ = 3.713016295388171e-9
│   λ = -25.490989686364266 + 2.0717829184637413e-15im
└   Δt = 0.320734042
┌ Info: DMRG iteration:
│   iter = 7
│   ϵ = 2.1566598307176236e-9
│   λ = -25.490989686364184 + 4.307349314613251e-16im
└   Δt = 0.31395084
┌ Info: DMRG iteration:
│   iter = 8
│   ϵ = 1.3589879523025462e-9
│   λ = -25.49098968636425 + 2.2327421787411883e-15im
└   Δt = 0.305333971
┌ Info: DMRG iteration:
│   iter = 9
│   ϵ = 9.771827925055499e-10
│   λ = -25.490989686364195 + 3.9485185688336005e-15im
└   Δt = 0.293974996
┌ Info: DMRG iteration:
│   iter = 10
│   ϵ = 7.218449262668591e-10
│   λ = -25.490989686364237 - 2.178505313887061e-15im
└   Δt = 0.2863256
┌ Info: DMRG iteration:
│   iter = 11
│   ϵ = 5.348228539338777e-10
│   λ = -25.49098968636421 + 1.8141694866693874e-16im
└   Δt = 0.274799577
┌ Info: DMRG iteration:
│   iter = 12
│   ϵ = 4.3625977317089153e-10
│   λ = -25.490989686364227 - 1.3279871606822156e-15im
└   Δt = 0.298785714
┌ Info: DMRG iteration:
│   iter = 13
│   ϵ = 3.638244430412921e-10
│   λ = -25.49098968636422 + 7.786455693179592e-16im
└   Δt = 0.347494066
┌ Info: DMRG iteration:
│   iter = 14
│   ϵ = 3.06872357817218e-10
│   λ = -25.490989686364266 - 2.2382304032283896e-15im
└   Δt = 0.307288261
┌ Info: DMRG iteration:
│   iter = 15
│   ϵ = 2.6031328982393076e-10
│   λ = -25.490989686364244 - 1.0600748896625596e-15im
└   Δt = 0.261766264
┌ Info: DMRG iteration:
│   iter = 16
│   ϵ = 2.2147139399553878e-10
│   λ = -25.490989686364234 - 5.401928888445549e-17im
└   Δt = 0.249781343
┌ Info: DMRG iteration:
│   iter = 17
│   ϵ = 1.8873908110661737e-10
│   λ = -25.49098968636424 - 6.328190547070562e-16im
└   Δt = 0.251935567
┌ Info: DMRG iteration:
│   iter = 18
│   ϵ = 1.610138215633974e-10
│   λ = -25.490989686364223 + 1.0116733158061688e-15im
└   Δt = 0.285627341
┌ Info: DMRG iteration:
│   iter = 19
│   ϵ = 1.3746595182224267e-10
│   λ = -25.49098968636423 - 4.340158985998442e-16im
└   Δt = 0.245647619
┌ Info: DMRG iteration:
│   iter = 20
│   ϵ = 1.174343290065243e-10
│   λ = -25.490989686364244 - 2.7743490817958406e-15im
└   Δt = 0.264773235
┌ Info: DMRG iteration:
│   iter = 21
│   ϵ = 1.0037615163128376e-10
│   λ = -25.490989686364234 - 4.170937780042159e-15im
└   Δt = 0.270246298
┌ Info: DMRG iteration:
│   iter = 22
│   ϵ = 8.583882214324085e-11
│   λ = -25.490989686364223 - 9.82015917947169e-16im
└   Δt = 0.25142661
┌ Info: DMRG iteration:
│   iter = 23
│   ϵ = 7.34428591380097e-11
│   λ = -25.49098968636421 + 3.9546386884201186e-16im
└   Δt = 0.234485508
┌ Info: DMRG iteration:
│   iter = 24
│   ϵ = 6.28651615682863e-11
│   λ = -25.490989686364255 + 1.927132546695591e-15im
└   Δt = 0.234726926
┌ Info: DMRG iteration:
│   iter = 25
│   ϵ = 5.38353936770596e-11
│   λ = -25.490989686364266 - 6.281251395673773e-16im
└   Δt = 0.224845716
┌ Info: DMRG iteration:
│   iter = 26
│   ϵ = 4.612331081551156e-11
│   λ = -25.490989686364227 + 2.10311908901249e-15im
└   Δt = 0.230493082
┌ Info: DMRG iteration:
│   iter = 27
│   ϵ = 4.130436476701104e-11
│   λ = -25.490989686364188 + 1.959820096094226e-15im
└   Δt = 0.225698346
┌ Info: DMRG iteration:
│   iter = 28
│   ϵ = 3.971917330586574e-11
│   λ = -25.49098968636423 + 2.3683988274712866e-16im
└   Δt = 0.218882252
┌ Info: DMRG iteration:
│   iter = 29
│   ϵ = 3.8276806548505106e-11
│   λ = -25.49098968636424 - 3.194342589561068e-15im
└   Δt = 0.236420646
┌ Info: DMRG iteration:
│   iter = 30
│   ϵ = 3.693666933285163e-11
│   λ = -25.490989686364223 - 2.441435132078464e-15im
└   Δt = 0.232586538
┌ Info: DMRG iteration:
│   iter = 31
│   ϵ = 3.5671239607451524e-11
│   λ = -25.490989686364202 - 1.509316798970967e-15im
└   Δt = 0.221405436
┌ Info: DMRG iteration:
│   iter = 32
│   ϵ = 3.446224119118797e-11
│   λ = -25.490989686364262 + 3.6448733006657707e-16im
└   Δt = 0.207193958
┌ Info: DMRG iteration:
│   iter = 33
│   ϵ = 3.3297657422098e-11
│   λ = -25.490989686364223 - 6.970644231489031e-16im
└   Δt = 0.211644515
┌ Info: DMRG iteration:
│   iter = 34
│   ϵ = 3.216974873238228e-11
│   λ = -25.490989686364212 + 3.4114197360542393e-16im
└   Δt = 0.208773674
┌ Info: DMRG iteration:
│   iter = 35
│   ϵ = 3.10736395514446e-11
│   λ = -25.49098968636425 - 2.229120932295443e-15im
└   Δt = 0.209132677
┌ Info: DMRG iteration:
│   iter = 36
│   ϵ = 3.000635601209443e-11
│   λ = -25.490989686364205 - 4.788657145997574e-17im
└   Δt = 0.241232229
┌ Info: DMRG iteration:
│   iter = 37
│   ϵ = 2.8966169835099458e-11
│   λ = -25.490989686364266 - 1.0580405011462236e-15im
└   Δt = 0.233204083
┌ Info: DMRG iteration:
│   iter = 38
│   ϵ = 2.795213204032689e-11
│   λ = -25.490989686364202 - 3.475407340187604e-15im
└   Δt = 0.229257292
┌ Info: DMRG iteration:
│   iter = 39
│   ϵ = 2.696378968683298e-11
│   λ = -25.490989686364223 + 2.2741067048471018e-15im
└   Δt = 0.220368087
┌ Info: DMRG iteration:
│   iter = 40
│   ϵ = 2.6000972812474588e-11
│   λ = -25.490989686364262 - 9.986332216835527e-16im
└   Δt = 0.220188261
┌ Info: DMRG iteration:
│   iter = 41
│   ϵ = 2.5063658956501813e-11
│   λ = -25.490989686364188 + 1.7898557306192927e-15im
└   Δt = 0.20856605
┌ Info: DMRG iteration:
│   iter = 42
│   ϵ = 2.4151885413555805e-11
│   λ = -25.49098968636423 + 1.5712671950058498e-16im
└   Δt = 0.21597753
┌ Info: DMRG iteration:
│   iter = 43
│   ϵ = 2.3265692580080323e-11
│   λ = -25.49098968636418 - 3.740227834889579e-16im
└   Δt = 0.21750501
┌ Info: DMRG iteration:
│   iter = 44
│   ϵ = 2.2405089175550036e-11
│   λ = -25.490989686364202 - 1.6023353510429877e-15im
└   Δt = 0.214293224
┌ Info: DMRG iteration:
│   iter = 45
│   ϵ = 2.1570031203532257e-11
│   λ = -25.49098968636426 + 2.4306236189618434e-15im
└   Δt = 0.19822189
┌ Info: DMRG iteration:
│   iter = 46
│   ϵ = 2.076041752769954e-11
│   λ = -25.49098968636423 - 1.7213058322383951e-15im
└   Δt = 0.273323179
┌ Info: DMRG iteration:
│   iter = 47
│   ϵ = 1.9976078479100315e-11
│   λ = -25.490989686364244 - 2.1394278798827888e-15im
└   Δt = 0.222667717
┌ Info: DMRG iteration:
│   iter = 48
│   ϵ = 1.921678216493006e-11
│   λ = -25.49098968636423 + 1.3367789292905993e-15im
└   Δt = 0.194993825
┌ Info: DMRG iteration:
│   iter = 49
│   ϵ = 1.8482259621604202e-11
│   λ = -25.490989686364276 - 2.3778161092142274e-15im
└   Δt = 0.199849916
┌ Info: DMRG iteration:
│   iter = 50
│   ϵ = 1.7772116064282086e-11
│   λ = -25.49098968636426 - 9.21280139158587e-16im
└   Δt = 0.190303624
┌ Info: DMRG iteration:
│   iter = 51
│   ϵ = 1.708597643160538e-11
│   λ = -25.490989686364223 + 1.6319179717233914e-15im
└   Δt = 0.186984328
┌ Info: DMRG iteration:
│   iter = 52
│   ϵ = 1.6423404410474172e-11
│   λ = -25.49098968636422 - 2.277521038440418e-15im
└   Δt = 0.185500366
┌ Info: DMRG iteration:
│   iter = 53
│   ϵ = 1.578372228744138e-11
│   λ = -25.490989686364237 + 1.2594280541827693e-15im
└   Δt = 0.180444685
┌ Info: DMRG iteration:
│   iter = 54
│   ϵ = 1.5167289323075386e-11
│   λ = -25.490989686364262 - 7.885458418503361e-16im
└   Δt = 0.168960564
┌ Info: DMRG iteration:
│   iter = 55
│   ϵ = 1.457241880584868e-11
│   λ = -25.49098968636424 - 1.8822049528627506e-15im
└   Δt = 0.167813553
┌ Info: DMRG iteration:
│   iter = 56
│   ϵ = 1.3999062129957047e-11
│   λ = -25.490989686364266 + 1.3777092171757003e-15im
└   Δt = 0.172471853
┌ Info: DMRG iteration:
│   iter = 57
│   ϵ = 1.3446644150608534e-11
│   λ = -25.490989686364234 - 6.183524316385401e-15im
└   Δt = 0.173594007
┌ Info: DMRG iteration:
│   iter = 58
│   ϵ = 1.2914572553231402e-11
│   λ = -25.49098968636427 + 1.7637731123393629e-16im
└   Δt = 0.179260147
┌ Info: DMRG iteration:
│   iter = 59
│   ϵ = 1.2402294709877488e-11
│   λ = -25.490989686364237 + 1.2665432064486625e-15im
└   Δt = 0.173887639
┌ Info: DMRG iteration:
│   iter = 60
│   ϵ = 1.1909211142361715e-11
│   λ = -25.490989686364216 - 3.848706971564007e-17im
└   Δt = 0.169144204
┌ Info: DMRG iteration:
│   iter = 61
│   ϵ = 1.1434731987131487e-11
│   λ = -25.490989686364255 - 2.7516547283807285e-15im
└   Δt = 0.164035162
┌ Info: DMRG iteration:
│   iter = 62
│   ϵ = 1.0978271593156308e-11
│   λ = -25.490989686364262 + 8.669005114760174e-17im
└   Δt = 0.162004432
┌ Info: DMRG iteration:
│   iter = 63
│   ϵ = 1.0539248690496678e-11
│   λ = -25.490989686364205 - 1.3708960321900824e-15im
└   Δt = 0.174670006
┌ Info: DMRG iteration:
│   iter = 64
│   ϵ = 1.0117088717003324e-11
│   λ = -25.49098968636423 - 1.2754620377813489e-15im
└   Δt = 0.166473802
┌ Info: DMRG iteration:
│   iter = 65
│   ϵ = 9.711224941348948e-12
│   λ = -25.49098968636424 - 5.2194431425163374e-15im
└   Δt = 0.190644935
┌ Info: DMRG iteration:
│   iter = 66
│   ϵ = 9.321100363352468e-12
│   λ = -25.49098968636426 - 2.862122025356807e-15im
└   Δt = 0.19047662
┌ Info: DMRG iteration:
│   iter = 67
│   ϵ = 8.945858270360489e-12
│   λ = -25.490989686364244 + 1.0040537026031762e-15im
└   Δt = 0.156918756
┌ Info: DMRG iteration:
│   iter = 68
│   ϵ = 8.58563521821864e-12
│   λ = -25.49098968636425 - 1.6236095098149267e-16im
└   Δt = 0.158018612
┌ Info: DMRG iteration:
│   iter = 69
│   ϵ = 8.239503332811185e-12
│   λ = -25.49098968636423 - 5.614240346839924e-15im
└   Δt = 0.159341843
┌ Info: DMRG iteration:
│   iter = 70
│   ϵ = 7.906994217751753e-12
│   λ = -25.49098968636425 + 2.857362684404925e-15im
└   Δt = 0.154383938
┌ Info: DMRG iteration:
│   iter = 71
│   ϵ = 7.587611756233479e-12
│   λ = -25.490989686364273 + 1.4899904335987264e-16im
└   Δt = 0.15729133
┌ Info: DMRG iteration:
│   iter = 72
│   ϵ = 7.280871816138976e-12
│   λ = -25.490989686364262 - 3.0568421978704394e-15im
└   Δt = 0.170484321
┌ Info: DMRG iteration:
│   iter = 73
│   ϵ = 6.986304155532008e-12
│   λ = -25.49098968636423 - 1.5763351303729496e-15im
└   Δt = 0.159675084
┌ Info: DMRG iteration:
│   iter = 74
│   ϵ = 6.703437618274334e-12
│   λ = -25.49098968636426 - 1.1803375434953885e-15im
└   Δt = 0.155517564
┌ Info: DMRG iteration:
│   iter = 75
│   ϵ = 6.431865695159927e-12
│   λ = -25.49098968636425 - 3.802788899241827e-16im
└   Δt = 0.153188805
┌ Info: DMRG iteration:
│   iter = 76
│   ϵ = 6.171141057993292e-12
│   λ = -25.49098968636428 - 2.884588013734565e-15im
└   Δt = 0.154216427
┌ Info: DMRG iteration:
│   iter = 77
│   ϵ = 5.920834677755299e-12
│   λ = -25.49098968636428 - 7.847032657690099e-17im
└   Δt = 0.149601536
┌ Info: DMRG iteration:
│   iter = 78
│   ϵ = 5.680558078469131e-12
│   λ = -25.490989686364248 + 1.2677683378692893e-15im
└   Δt = 0.157926554
┌ Info: DMRG iteration:
│   iter = 79
│   ϵ = 5.449922564920425e-12
│   λ = -25.490989686364244 - 3.636316346961648e-15im
└   Δt = 0.148857622
┌ Info: DMRG iteration:
│   iter = 80
│   ϵ = 5.228481383008137e-12
│   λ = -25.49098968636423 + 4.2473680867361225e-15im
└   Δt = 0.155409062
┌ Info: DMRG iteration:
│   iter = 81
│   ϵ = 5.015872847279488e-12
│   λ = -25.49098968636424 - 1.5801653691932046e-15im
└   Δt = 0.159589083
┌ Info: DMRG iteration:
│   iter = 82
│   ϵ = 4.8121942237202594e-12
│   λ = -25.49098968636426 + 7.802856470160761e-16im
└   Δt = 0.150303988
┌ Info: DMRG iteration:
│   iter = 83
│   ϵ = 4.616477532907697e-12
│   λ = -25.490989686364273 - 2.8329039430124918e-15im
└   Δt = 0.153641273
┌ Info: DMRG iteration:
│   iter = 84
│   ϵ = 4.42855766971901e-12
│   λ = -25.490989686364262 + 2.4136420626235198e-15im
└   Δt = 0.149577836
┌ Info: DMRG iteration:
│   iter = 85
│   ϵ = 4.24848395540251e-12
│   λ = -25.490989686364248 + 4.766603056058685e-16im
└   Δt = 0.15207591
┌ Info: DMRG iteration:
│   iter = 86
│   ϵ = 4.075457996969941e-12
│   λ = -25.490989686364173 - 4.7748070406073966e-15im
└   Δt = 0.153164545
┌ Info: DMRG iteration:
│   iter = 87
│   ϵ = 3.909744315492562e-12
│   λ = -25.490989686364212 + 1.935947587185421e-16im
└   Δt = 0.15738927
┌ Info: DMRG iteration:
│   iter = 88
│   ϵ = 3.750539936521793e-12
│   λ = -25.49098968636419 + 3.712681875185555e-15im
└   Δt = 0.151357369
┌ Info: DMRG iteration:
│   iter = 89
│   ϵ = 3.5976841881367522e-12
│   λ = -25.490989686364223 + 2.0110140996047558e-15im
└   Δt = 0.151411975
┌ Info: DMRG iteration:
│   iter = 90
│   ϵ = 3.450693514303509e-12
│   λ = -25.49098968636424 - 6.098416938240174e-16im
└   Δt = 0.152905487
┌ Info: DMRG iteration:
│   iter = 91
│   ϵ = 3.3105287992297956e-12
│   λ = -25.490989686364227 + 2.973208519650964e-15im
└   Δt = 0.14798927
┌ Info: DMRG iteration:
│   iter = 92
│   ϵ = 3.175332943584959e-12
│   λ = -25.490989686364223 + 4.413133632386735e-15im
└   Δt = 0.149553964
┌ Info: DMRG iteration:
│   iter = 93
│   ϵ = 3.046054946799407e-12
│   λ = -25.49098968636422 - 4.552660704222068e-16im
└   Δt = 0.146753041
┌ Info: DMRG iteration:
│   iter = 94
│   ϵ = 2.9220449980241814e-12
│   λ = -25.49098968636425 + 3.1636522191110495e-16im
└   Δt = 0.147549502
┌ Info: DMRG iteration:
│   iter = 95
│   ϵ = 2.8029054110326236e-12
│   λ = -25.490989686364223 - 3.1085275658239436e-15im
└   Δt = 0.146347673
┌ Info: DMRG iteration:
│   iter = 96
│   ϵ = 2.6887531652267747e-12
│   λ = -25.490989686364237 - 5.4855507949630395e-15im
└   Δt = 0.147903854
┌ Info: DMRG iteration:
│   iter = 97
│   ϵ = 2.5790763533282124e-12
│   λ = -25.490989686364273 + 2.0135679569182404e-15im
└   Δt = 0.132107677
┌ Info: DMRG iteration:
│   iter = 98
│   ϵ = 2.474009606916938e-12
│   λ = -25.490989686364255 + 2.7998461012485813e-15im
└   Δt = 0.152253884
┌ Info: DMRG iteration:
│   iter = 99
│   ϵ = 2.3730191295885238e-12
│   λ = -25.49098968636423 + 1.4670863074529506e-15im
└   Δt = 0.146928539
┌ Info: DMRG iteration:
│   iter = 100
│   ϵ = 2.2762027140472275e-12
│   λ = -25.490989686364227 + 6.571904109531772e-16im
└   Δt = 0.148701543
┌ Warning: DMRG maximum iterations
│   iter = 100
│   ϵ = 2.2762027140472275e-12
│   λ = -25.490989686364227 + 6.571904109531772e-16im
└ @ MPSKit ~/Projects/Julia/SUNHeisenberg/dev/MPSKit/src/algorithms/groundstate/dmrg.jl:44
┌ Info: DMRG summary:
│   ϵ = 2.0e-12
│   λ = -25.490989686364227 + 6.571904109531772e-16im
└   Δt = 27.325116319

````

Excitations on top of the groundstate can be found through the use of the quasiparticle
ansatz. This returns quasiparticle states, which can be converted to regular `FiniteMPS`
objects.

````julia
E_ex, qps = excitations(H, QuasiparticleAnsatz(), ψ, envs; num=16)
states_mps = vcat(ψ, map(qp -> convert(FiniteMPS, qp), qps))
E_mps = map(x -> sum(expectation_value(x, H_mps)), states_mps)

T_mps = periodic_boundary_conditions(DenseMPO(O), L_mps)
momenta_mps = Float64[]
append!(momenta_mps, fix_degeneracies(states[1:1]))
append!(momenta_mps, fix_degeneracies(states[2:2]))
append!(momenta_mps, fix_degeneracies(states[3:3]))
append!(momenta_mps, fix_degeneracies(states[4:5]))
append!(momenta_mps, fix_degeneracies(states[6:9]))
append!(momenta_mps, fix_degeneracies(states[10:11]))
append!(momenta_mps, fix_degeneracies(states[12:12]))
append!(momenta_mps, fix_degeneracies(states[13:16]))

plot(
    momenta_mps,
    real.(energies[1:16]);
    seriestype=:scatter,
    xlabel="momentum",
    ylabel="energy",
    legend=false,
)
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
<path clip-path="url(#clip650)" d="M287.366 1423.18 L2352.76 1423.18 L2352.76 47.2441 L287.366 47.2441  Z" fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip652">
    <rect x="287" y="47" width="2066" height="1377"/>
  </clipPath>
</defs>
<polyline clip-path="url(#clip652)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="389.73,1423.18 389.73,47.2441 "/>
<polyline clip-path="url(#clip652)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="699.84,1423.18 699.84,47.2441 "/>
<polyline clip-path="url(#clip652)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1009.95,1423.18 1009.95,47.2441 "/>
<polyline clip-path="url(#clip652)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1320.06,1423.18 1320.06,47.2441 "/>
<polyline clip-path="url(#clip652)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1630.17,1423.18 1630.17,47.2441 "/>
<polyline clip-path="url(#clip652)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="1940.28,1423.18 1940.28,47.2441 "/>
<polyline clip-path="url(#clip652)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="2250.39,1423.18 2250.39,47.2441 "/>
<polyline clip-path="url(#clip650)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="287.366,1423.18 2352.76,1423.18 "/>
<polyline clip-path="url(#clip650)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="389.73,1423.18 389.73,1404.28 "/>
<polyline clip-path="url(#clip650)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="699.84,1423.18 699.84,1404.28 "/>
<polyline clip-path="url(#clip650)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1009.95,1423.18 1009.95,1404.28 "/>
<polyline clip-path="url(#clip650)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1320.06,1423.18 1320.06,1404.28 "/>
<polyline clip-path="url(#clip650)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1630.17,1423.18 1630.17,1404.28 "/>
<polyline clip-path="url(#clip650)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="1940.28,1423.18 1940.28,1404.28 "/>
<polyline clip-path="url(#clip650)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="2250.39,1423.18 2250.39,1404.28 "/>
<path clip-path="url(#clip650)" d="M359.197 1468.75 L388.873 1468.75 L388.873 1472.69 L359.197 1472.69 L359.197 1468.75 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M413.132 1466.95 Q416.489 1467.66 418.364 1469.93 Q420.262 1472.2 420.262 1475.53 Q420.262 1480.65 416.743 1483.45 Q413.225 1486.25 406.743 1486.25 Q404.568 1486.25 402.253 1485.81 Q399.961 1485.39 397.507 1484.54 L397.507 1480.02 Q399.452 1481.16 401.767 1481.74 Q404.081 1482.32 406.605 1482.32 Q411.003 1482.32 413.294 1480.58 Q415.609 1478.84 415.609 1475.53 Q415.609 1472.48 413.456 1470.77 Q411.327 1469.03 407.507 1469.03 L403.48 1469.03 L403.48 1465.19 L407.693 1465.19 Q411.142 1465.19 412.97 1463.82 Q414.799 1462.43 414.799 1459.84 Q414.799 1457.18 412.901 1455.77 Q411.026 1454.33 407.507 1454.33 Q405.586 1454.33 403.387 1454.75 Q401.188 1455.16 398.549 1456.04 L398.549 1451.88 Q401.211 1451.14 403.526 1450.77 Q405.864 1450.39 407.924 1450.39 Q413.248 1450.39 416.35 1452.83 Q419.452 1455.23 419.452 1459.35 Q419.452 1462.22 417.808 1464.21 Q416.165 1466.18 413.132 1466.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M669.782 1468.75 L699.458 1468.75 L699.458 1472.69 L669.782 1472.69 L669.782 1468.75 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M713.578 1481.64 L729.898 1481.64 L729.898 1485.58 L707.953 1485.58 L707.953 1481.64 Q710.616 1478.89 715.199 1474.26 Q719.805 1469.61 720.986 1468.27 Q723.231 1465.74 724.111 1464.01 Q725.014 1462.25 725.014 1460.56 Q725.014 1457.8 723.069 1456.07 Q721.148 1454.33 718.046 1454.33 Q715.847 1454.33 713.393 1455.09 Q710.963 1455.86 708.185 1457.41 L708.185 1452.69 Q711.009 1451.55 713.463 1450.97 Q715.916 1450.39 717.953 1450.39 Q723.324 1450.39 726.518 1453.08 Q729.713 1455.77 729.713 1460.26 Q729.713 1462.39 728.902 1464.31 Q728.115 1466.2 726.009 1468.8 Q725.43 1469.47 722.328 1472.69 Q719.227 1475.88 713.578 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M979.708 1468.75 L1009.38 1468.75 L1009.38 1472.69 L979.708 1472.69 L979.708 1468.75 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M1020.29 1481.64 L1027.92 1481.64 L1027.92 1455.28 L1019.61 1456.95 L1019.61 1452.69 L1027.88 1451.02 L1032.55 1451.02 L1032.55 1481.64 L1040.19 1481.64 L1040.19 1485.58 L1020.29 1485.58 L1020.29 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M1320.06 1454.1 Q1316.45 1454.1 1314.62 1457.66 Q1312.82 1461.2 1312.82 1468.33 Q1312.82 1475.44 1314.62 1479.01 Q1316.45 1482.55 1320.06 1482.55 Q1323.7 1482.55 1325.5 1479.01 Q1327.33 1475.44 1327.33 1468.33 Q1327.33 1461.2 1325.5 1457.66 Q1323.7 1454.1 1320.06 1454.1 M1320.06 1450.39 Q1325.87 1450.39 1328.93 1455 Q1332.01 1459.58 1332.01 1468.33 Q1332.01 1477.06 1328.93 1481.67 Q1325.87 1486.25 1320.06 1486.25 Q1314.25 1486.25 1311.17 1481.67 Q1308.12 1477.06 1308.12 1468.33 Q1308.12 1459.58 1311.17 1455 Q1314.25 1450.39 1320.06 1450.39 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M1620.55 1481.64 L1628.19 1481.64 L1628.19 1455.28 L1619.88 1456.95 L1619.88 1452.69 L1628.15 1451.02 L1632.82 1451.02 L1632.82 1481.64 L1640.46 1481.64 L1640.46 1485.58 L1620.55 1485.58 L1620.55 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M1934.93 1481.64 L1951.25 1481.64 L1951.25 1485.58 L1929.31 1485.58 L1929.31 1481.64 Q1931.97 1478.89 1936.55 1474.26 Q1941.16 1469.61 1942.34 1468.27 Q1944.59 1465.74 1945.47 1464.01 Q1946.37 1462.25 1946.37 1460.56 Q1946.37 1457.8 1944.43 1456.07 Q1942.5 1454.33 1939.4 1454.33 Q1937.2 1454.33 1934.75 1455.09 Q1932.32 1455.86 1929.54 1457.41 L1929.54 1452.69 Q1932.37 1451.55 1934.82 1450.97 Q1937.27 1450.39 1939.31 1450.39 Q1944.68 1450.39 1947.87 1453.08 Q1951.07 1455.77 1951.07 1460.26 Q1951.07 1462.39 1950.26 1464.31 Q1949.47 1466.2 1947.37 1468.8 Q1946.79 1469.47 1943.68 1472.69 Q1940.58 1475.88 1934.93 1481.64 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M2254.64 1466.95 Q2258 1467.66 2259.87 1469.93 Q2261.77 1472.2 2261.77 1475.53 Q2261.77 1480.65 2258.25 1483.45 Q2254.73 1486.25 2248.25 1486.25 Q2246.08 1486.25 2243.76 1485.81 Q2241.47 1485.39 2239.01 1484.54 L2239.01 1480.02 Q2240.96 1481.16 2243.27 1481.74 Q2245.59 1482.32 2248.11 1482.32 Q2252.51 1482.32 2254.8 1480.58 Q2257.12 1478.84 2257.12 1475.53 Q2257.12 1472.48 2254.96 1470.77 Q2252.83 1469.03 2249.01 1469.03 L2244.99 1469.03 L2244.99 1465.19 L2249.2 1465.19 Q2252.65 1465.19 2254.48 1463.82 Q2256.31 1462.43 2256.31 1459.84 Q2256.31 1457.18 2254.41 1455.77 Q2252.53 1454.33 2249.01 1454.33 Q2247.09 1454.33 2244.89 1454.75 Q2242.7 1455.16 2240.06 1456.04 L2240.06 1451.88 Q2242.72 1451.14 2245.03 1450.77 Q2247.37 1450.39 2249.43 1450.39 Q2254.76 1450.39 2257.86 1452.83 Q2260.96 1455.23 2260.96 1459.35 Q2260.96 1462.22 2259.32 1464.21 Q2257.67 1466.18 2254.64 1466.95 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M1164.44 1539.24 Q1166.63 1535.29 1169.69 1533.41 Q1172.74 1531.54 1176.88 1531.54 Q1182.45 1531.54 1185.47 1535.45 Q1188.5 1539.33 1188.5 1546.53 L1188.5 1568.04 L1182.61 1568.04 L1182.61 1546.72 Q1182.61 1541.59 1180.8 1539.11 Q1178.98 1536.63 1175.26 1536.63 Q1170.71 1536.63 1168.06 1539.65 Q1165.42 1542.68 1165.42 1547.9 L1165.42 1568.04 L1159.53 1568.04 L1159.53 1546.72 Q1159.53 1541.56 1157.72 1539.11 Q1155.91 1536.63 1152.12 1536.63 Q1147.63 1536.63 1144.99 1539.68 Q1142.35 1542.71 1142.35 1547.9 L1142.35 1568.04 L1136.46 1568.04 L1136.46 1532.4 L1142.35 1532.4 L1142.35 1537.93 Q1144.35 1534.66 1147.15 1533.1 Q1149.95 1531.54 1153.8 1531.54 Q1157.69 1531.54 1160.39 1533.51 Q1163.13 1535.48 1164.44 1539.24 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M1213.99 1536.5 Q1209.28 1536.5 1206.54 1540.19 Q1203.81 1543.85 1203.81 1550.25 Q1203.81 1556.65 1206.51 1560.34 Q1209.25 1564 1213.99 1564 Q1218.67 1564 1221.41 1560.31 Q1224.15 1556.62 1224.15 1550.25 Q1224.15 1543.92 1221.41 1540.23 Q1218.67 1536.5 1213.99 1536.5 M1213.99 1531.54 Q1221.63 1531.54 1225.99 1536.5 Q1230.35 1541.47 1230.35 1550.25 Q1230.35 1559 1225.99 1564 Q1221.63 1568.97 1213.99 1568.97 Q1206.32 1568.97 1201.96 1564 Q1197.63 1559 1197.63 1550.25 Q1197.63 1541.47 1201.96 1536.5 Q1206.32 1531.54 1213.99 1531.54 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M1267.81 1539.24 Q1270.01 1535.29 1273.07 1533.41 Q1276.12 1531.54 1280.26 1531.54 Q1285.83 1531.54 1288.85 1535.45 Q1291.88 1539.33 1291.88 1546.53 L1291.88 1568.04 L1285.99 1568.04 L1285.99 1546.72 Q1285.99 1541.59 1284.17 1539.11 Q1282.36 1536.63 1278.64 1536.63 Q1274.08 1536.63 1271.44 1539.65 Q1268.8 1542.68 1268.8 1547.9 L1268.8 1568.04 L1262.91 1568.04 L1262.91 1546.72 Q1262.91 1541.56 1261.1 1539.11 Q1259.28 1536.63 1255.5 1536.63 Q1251.01 1536.63 1248.37 1539.68 Q1245.73 1542.71 1245.73 1547.9 L1245.73 1568.04 L1239.84 1568.04 L1239.84 1532.4 L1245.73 1532.4 L1245.73 1537.93 Q1247.73 1534.66 1250.53 1533.1 Q1253.33 1531.54 1257.18 1531.54 Q1261.07 1531.54 1263.77 1533.51 Q1266.51 1535.48 1267.81 1539.24 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M1334.05 1548.76 L1334.05 1551.62 L1307.12 1551.62 Q1307.5 1557.67 1310.75 1560.85 Q1314.03 1564 1319.85 1564 Q1323.23 1564 1326.38 1563.17 Q1329.56 1562.35 1332.68 1560.69 L1332.68 1566.23 Q1329.53 1567.57 1326.22 1568.27 Q1322.91 1568.97 1319.5 1568.97 Q1310.97 1568.97 1305.98 1564 Q1301.01 1559.04 1301.01 1550.57 Q1301.01 1541.82 1305.72 1536.69 Q1310.46 1531.54 1318.49 1531.54 Q1325.68 1531.54 1329.85 1536.18 Q1334.05 1540.8 1334.05 1548.76 M1328.19 1547.04 Q1328.13 1542.23 1325.49 1539.37 Q1322.88 1536.5 1318.55 1536.5 Q1313.65 1536.5 1310.69 1539.27 Q1307.76 1542.04 1307.31 1547.07 L1328.19 1547.04 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M1373.29 1546.53 L1373.29 1568.04 L1367.44 1568.04 L1367.44 1546.72 Q1367.44 1541.66 1365.46 1539.14 Q1363.49 1536.63 1359.54 1536.63 Q1354.8 1536.63 1352.06 1539.65 Q1349.33 1542.68 1349.33 1547.9 L1349.33 1568.04 L1343.44 1568.04 L1343.44 1532.4 L1349.33 1532.4 L1349.33 1537.93 Q1351.43 1534.72 1354.26 1533.13 Q1357.13 1531.54 1360.85 1531.54 Q1366.99 1531.54 1370.14 1535.36 Q1373.29 1539.14 1373.29 1546.53 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M1390.77 1522.27 L1390.77 1532.4 L1402.83 1532.4 L1402.83 1536.95 L1390.77 1536.95 L1390.77 1556.3 Q1390.77 1560.66 1391.95 1561.9 Q1393.16 1563.14 1396.82 1563.14 L1402.83 1563.14 L1402.83 1568.04 L1396.82 1568.04 Q1390.04 1568.04 1387.46 1565.53 Q1384.88 1562.98 1384.88 1556.3 L1384.88 1536.95 L1380.58 1536.95 L1380.58 1532.4 L1384.88 1532.4 L1384.88 1522.27 L1390.77 1522.27 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M1409.93 1553.98 L1409.93 1532.4 L1415.79 1532.4 L1415.79 1553.75 Q1415.79 1558.81 1417.76 1561.36 Q1419.73 1563.87 1423.68 1563.87 Q1428.42 1563.87 1431.16 1560.85 Q1433.93 1557.83 1433.93 1552.61 L1433.93 1532.4 L1439.78 1532.4 L1439.78 1568.04 L1433.93 1568.04 L1433.93 1562.57 Q1431.79 1565.82 1428.96 1567.41 Q1426.16 1568.97 1422.44 1568.97 Q1416.29 1568.97 1413.11 1565.15 Q1409.93 1561.33 1409.93 1553.98 M1424.67 1531.54 L1424.67 1531.54 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M1479.6 1539.24 Q1481.8 1535.29 1484.85 1533.41 Q1487.91 1531.54 1492.05 1531.54 Q1497.62 1531.54 1500.64 1535.45 Q1503.66 1539.33 1503.66 1546.53 L1503.66 1568.04 L1497.78 1568.04 L1497.78 1546.72 Q1497.78 1541.59 1495.96 1539.11 Q1494.15 1536.63 1490.42 1536.63 Q1485.87 1536.63 1483.23 1539.65 Q1480.59 1542.68 1480.59 1547.9 L1480.59 1568.04 L1474.7 1568.04 L1474.7 1546.72 Q1474.7 1541.56 1472.89 1539.11 Q1471.07 1536.63 1467.28 1536.63 Q1462.8 1536.63 1460.15 1539.68 Q1457.51 1542.71 1457.51 1547.9 L1457.51 1568.04 L1451.62 1568.04 L1451.62 1532.4 L1457.51 1532.4 L1457.51 1537.93 Q1459.52 1534.66 1462.32 1533.1 Q1465.12 1531.54 1468.97 1531.54 Q1472.85 1531.54 1475.56 1533.51 Q1478.3 1535.48 1479.6 1539.24 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><polyline clip-path="url(#clip652)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="287.366,1242.63 2352.76,1242.63 "/>
<polyline clip-path="url(#clip652)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="287.366,1023.16 2352.76,1023.16 "/>
<polyline clip-path="url(#clip652)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="287.366,803.68 2352.76,803.68 "/>
<polyline clip-path="url(#clip652)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="287.366,584.203 2352.76,584.203 "/>
<polyline clip-path="url(#clip652)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="287.366,364.726 2352.76,364.726 "/>
<polyline clip-path="url(#clip652)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="287.366,145.249 2352.76,145.249 "/>
<polyline clip-path="url(#clip650)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="287.366,1423.18 287.366,47.2441 "/>
<polyline clip-path="url(#clip650)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="287.366,1242.63 306.264,1242.63 "/>
<polyline clip-path="url(#clip650)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="287.366,1023.16 306.264,1023.16 "/>
<polyline clip-path="url(#clip650)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="287.366,803.68 306.264,803.68 "/>
<polyline clip-path="url(#clip650)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="287.366,584.203 306.264,584.203 "/>
<polyline clip-path="url(#clip650)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="287.366,364.726 306.264,364.726 "/>
<polyline clip-path="url(#clip650)" style="stroke:#000000; stroke-linecap:round; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="287.366,145.249 306.264,145.249 "/>
<path clip-path="url(#clip650)" d="M114.26 1243.09 L143.936 1243.09 L143.936 1247.02 L114.26 1247.02 L114.26 1243.09 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M154.839 1255.98 L162.477 1255.98 L162.477 1229.61 L154.167 1231.28 L154.167 1227.02 L162.431 1225.35 L167.107 1225.35 L167.107 1255.98 L174.746 1255.98 L174.746 1259.91 L154.839 1259.91 L154.839 1255.98 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M184.237 1225.35 L202.593 1225.35 L202.593 1229.29 L188.519 1229.29 L188.519 1237.76 Q189.538 1237.41 190.556 1237.25 Q191.575 1237.07 192.593 1237.07 Q198.38 1237.07 201.76 1240.24 Q205.139 1243.41 205.139 1248.83 Q205.139 1254.4 201.667 1257.51 Q198.195 1260.59 191.875 1260.59 Q189.7 1260.59 187.431 1260.21 Q185.186 1259.84 182.778 1259.1 L182.778 1254.4 Q184.862 1255.54 187.084 1256.09 Q189.306 1256.65 191.783 1256.65 Q195.787 1256.65 198.125 1254.54 Q200.463 1252.44 200.463 1248.83 Q200.463 1245.21 198.125 1243.11 Q195.787 1241 191.783 1241 Q189.908 1241 188.033 1241.42 Q186.181 1241.84 184.237 1242.71 L184.237 1225.35 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M214.352 1254.03 L219.236 1254.03 L219.236 1259.91 L214.352 1259.91 L214.352 1254.03 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M239.422 1228.43 Q235.81 1228.43 233.982 1232 Q232.176 1235.54 232.176 1242.67 Q232.176 1249.78 233.982 1253.34 Q235.81 1256.88 239.422 1256.88 Q243.056 1256.88 244.861 1253.34 Q246.69 1249.78 246.69 1242.67 Q246.69 1235.54 244.861 1232 Q243.056 1228.43 239.422 1228.43 M239.422 1224.73 Q245.232 1224.73 248.287 1229.34 Q251.366 1233.92 251.366 1242.67 Q251.366 1251.4 248.287 1256 Q245.232 1260.59 239.422 1260.59 Q233.611 1260.59 230.533 1256 Q227.477 1251.4 227.477 1242.67 Q227.477 1233.92 230.533 1229.34 Q233.611 1224.73 239.422 1224.73 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M115.256 1023.61 L144.931 1023.61 L144.931 1027.54 L115.256 1027.54 L115.256 1023.61 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M155.834 1036.5 L163.473 1036.5 L163.473 1010.14 L155.163 1011.8 L155.163 1007.54 L163.427 1005.88 L168.102 1005.88 L168.102 1036.5 L175.741 1036.5 L175.741 1040.44 L155.834 1040.44 L155.834 1036.5 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M198.033 1009.95 L186.227 1028.4 L198.033 1028.4 L198.033 1009.95 M196.806 1005.88 L202.686 1005.88 L202.686 1028.4 L207.616 1028.4 L207.616 1032.29 L202.686 1032.29 L202.686 1040.44 L198.033 1040.44 L198.033 1032.29 L182.431 1032.29 L182.431 1027.77 L196.806 1005.88 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M215.348 1034.56 L220.232 1034.56 L220.232 1040.44 L215.348 1040.44 L215.348 1034.56 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M230.463 1005.88 L248.82 1005.88 L248.82 1009.81 L234.746 1009.81 L234.746 1018.28 Q235.764 1017.94 236.783 1017.78 Q237.801 1017.59 238.82 1017.59 Q244.607 1017.59 247.986 1020.76 Q251.366 1023.93 251.366 1029.35 Q251.366 1034.93 247.894 1038.03 Q244.421 1041.11 238.102 1041.11 Q235.926 1041.11 233.658 1040.74 Q231.412 1040.37 229.005 1039.63 L229.005 1034.93 Q231.088 1036.06 233.31 1036.62 Q235.533 1037.17 238.009 1037.17 Q242.014 1037.17 244.352 1035.07 Q246.69 1032.96 246.69 1029.35 Q246.69 1025.74 244.352 1023.63 Q242.014 1021.53 238.009 1021.53 Q236.135 1021.53 234.26 1021.94 Q232.408 1022.36 230.463 1023.24 L230.463 1005.88 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M114.26 804.131 L143.936 804.131 L143.936 808.067 L114.26 808.067 L114.26 804.131 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M154.839 817.025 L162.477 817.025 L162.477 790.659 L154.167 792.326 L154.167 788.067 L162.431 786.4 L167.107 786.4 L167.107 817.025 L174.746 817.025 L174.746 820.96 L154.839 820.96 L154.839 817.025 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M197.037 790.474 L185.232 808.923 L197.037 808.923 L197.037 790.474 M195.811 786.4 L201.69 786.4 L201.69 808.923 L206.621 808.923 L206.621 812.812 L201.69 812.812 L201.69 820.96 L197.037 820.96 L197.037 812.812 L181.436 812.812 L181.436 808.298 L195.811 786.4 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M214.352 815.08 L219.236 815.08 L219.236 820.96 L214.352 820.96 L214.352 815.08 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M239.422 789.479 Q235.81 789.479 233.982 793.044 Q232.176 796.585 232.176 803.715 Q232.176 810.821 233.982 814.386 Q235.81 817.928 239.422 817.928 Q243.056 817.928 244.861 814.386 Q246.69 810.821 246.69 803.715 Q246.69 796.585 244.861 793.044 Q243.056 789.479 239.422 789.479 M239.422 785.775 Q245.232 785.775 248.287 790.381 Q251.366 794.965 251.366 803.715 Q251.366 812.442 248.287 817.048 Q245.232 821.631 239.422 821.631 Q233.611 821.631 230.533 817.048 Q227.477 812.442 227.477 803.715 Q227.477 794.965 230.533 790.381 Q233.611 785.775 239.422 785.775 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M115.256 584.654 L144.931 584.654 L144.931 588.59 L115.256 588.59 L115.256 584.654 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M155.834 597.548 L163.473 597.548 L163.473 571.182 L155.163 572.849 L155.163 568.59 L163.427 566.923 L168.102 566.923 L168.102 597.548 L175.741 597.548 L175.741 601.483 L155.834 601.483 L155.834 597.548 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M199.352 582.849 Q202.709 583.567 204.584 585.835 Q206.482 588.104 206.482 591.437 Q206.482 596.553 202.963 599.353 Q199.445 602.154 192.963 602.154 Q190.788 602.154 188.473 601.715 Q186.181 601.298 183.727 600.441 L183.727 595.928 Q185.672 597.062 187.987 597.641 Q190.301 598.219 192.825 598.219 Q197.223 598.219 199.514 596.483 Q201.829 594.747 201.829 591.437 Q201.829 588.381 199.676 586.668 Q197.547 584.932 193.727 584.932 L189.7 584.932 L189.7 581.09 L193.913 581.09 Q197.362 581.09 199.19 579.724 Q201.019 578.335 201.019 575.743 Q201.019 573.08 199.121 571.668 Q197.246 570.233 193.727 570.233 Q191.806 570.233 189.607 570.65 Q187.408 571.067 184.769 571.946 L184.769 567.78 Q187.431 567.039 189.746 566.668 Q192.084 566.298 194.144 566.298 Q199.468 566.298 202.57 568.729 Q205.672 571.136 205.672 575.256 Q205.672 578.127 204.028 580.117 Q202.385 582.085 199.352 582.849 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M215.348 595.604 L220.232 595.604 L220.232 601.483 L215.348 601.483 L215.348 595.604 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M230.463 566.923 L248.82 566.923 L248.82 570.858 L234.746 570.858 L234.746 579.33 Q235.764 578.983 236.783 578.821 Q237.801 578.636 238.82 578.636 Q244.607 578.636 247.986 581.807 Q251.366 584.979 251.366 590.395 Q251.366 595.974 247.894 599.076 Q244.421 602.154 238.102 602.154 Q235.926 602.154 233.658 601.784 Q231.412 601.414 229.005 600.673 L229.005 595.974 Q231.088 597.108 233.31 597.664 Q235.533 598.219 238.009 598.219 Q242.014 598.219 244.352 596.113 Q246.69 594.006 246.69 590.395 Q246.69 586.784 244.352 584.678 Q242.014 582.571 238.009 582.571 Q236.135 582.571 234.26 582.988 Q232.408 583.404 230.463 584.284 L230.463 566.923 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M114.26 365.178 L143.936 365.178 L143.936 369.113 L114.26 369.113 L114.26 365.178 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M154.839 378.071 L162.477 378.071 L162.477 351.705 L154.167 353.372 L154.167 349.113 L162.431 347.446 L167.107 347.446 L167.107 378.071 L174.746 378.071 L174.746 382.006 L154.839 382.006 L154.839 378.071 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M198.357 363.372 Q201.713 364.09 203.588 366.358 Q205.487 368.627 205.487 371.96 Q205.487 377.076 201.968 379.877 Q198.45 382.677 191.968 382.677 Q189.792 382.677 187.477 382.238 Q185.186 381.821 182.732 380.965 L182.732 376.451 Q184.676 377.585 186.991 378.164 Q189.306 378.742 191.829 378.742 Q196.227 378.742 198.519 377.006 Q200.834 375.27 200.834 371.96 Q200.834 368.904 198.681 367.191 Q196.551 365.455 192.732 365.455 L188.704 365.455 L188.704 361.613 L192.917 361.613 Q196.366 361.613 198.195 360.247 Q200.024 358.858 200.024 356.266 Q200.024 353.604 198.125 352.192 Q196.25 350.756 192.732 350.756 Q190.811 350.756 188.612 351.173 Q186.413 351.59 183.774 352.469 L183.774 348.303 Q186.436 347.562 188.75 347.192 Q191.088 346.821 193.149 346.821 Q198.473 346.821 201.575 349.252 Q204.676 351.659 204.676 355.779 Q204.676 358.65 203.033 360.641 Q201.389 362.608 198.357 363.372 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M214.352 376.127 L219.236 376.127 L219.236 382.006 L214.352 382.006 L214.352 376.127 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M239.422 350.525 Q235.81 350.525 233.982 354.09 Q232.176 357.631 232.176 364.761 Q232.176 371.867 233.982 375.432 Q235.81 378.974 239.422 378.974 Q243.056 378.974 244.861 375.432 Q246.69 371.867 246.69 364.761 Q246.69 357.631 244.861 354.09 Q243.056 350.525 239.422 350.525 M239.422 346.821 Q245.232 346.821 248.287 351.428 Q251.366 356.011 251.366 364.761 Q251.366 373.488 248.287 378.094 Q245.232 382.677 239.422 382.677 Q233.611 382.677 230.533 378.094 Q227.477 373.488 227.477 364.761 Q227.477 356.011 230.533 351.428 Q233.611 346.821 239.422 346.821 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M115.256 145.701 L144.931 145.701 L144.931 149.636 L115.256 149.636 L115.256 145.701 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M155.834 158.594 L163.473 158.594 L163.473 132.228 L155.163 133.895 L155.163 129.636 L163.427 127.969 L168.102 127.969 L168.102 158.594 L175.741 158.594 L175.741 162.529 L155.834 162.529 L155.834 158.594 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M189.213 158.594 L205.533 158.594 L205.533 162.529 L183.588 162.529 L183.588 158.594 Q186.251 155.839 190.834 151.21 Q195.44 146.557 196.621 145.215 Q198.866 142.691 199.746 140.955 Q200.649 139.196 200.649 137.506 Q200.649 134.752 198.704 133.016 Q196.783 131.279 193.681 131.279 Q191.482 131.279 189.028 132.043 Q186.598 132.807 183.82 134.358 L183.82 129.636 Q186.644 128.502 189.098 127.923 Q191.551 127.344 193.588 127.344 Q198.959 127.344 202.153 130.029 Q205.348 132.715 205.348 137.205 Q205.348 139.335 204.537 141.256 Q203.75 143.154 201.644 145.747 Q201.065 146.418 197.963 149.636 Q194.862 152.83 189.213 158.594 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M215.348 156.65 L220.232 156.65 L220.232 162.529 L215.348 162.529 L215.348 156.65 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M230.463 127.969 L248.82 127.969 L248.82 131.904 L234.746 131.904 L234.746 140.377 Q235.764 140.029 236.783 139.867 Q237.801 139.682 238.82 139.682 Q244.607 139.682 247.986 142.853 Q251.366 146.025 251.366 151.441 Q251.366 157.02 247.894 160.122 Q244.421 163.201 238.102 163.201 Q235.926 163.201 233.658 162.83 Q231.412 162.46 229.005 161.719 L229.005 157.02 Q231.088 158.154 233.31 158.71 Q235.533 159.265 238.009 159.265 Q242.014 159.265 244.352 157.159 Q246.69 155.052 246.69 151.441 Q246.69 147.83 244.352 145.724 Q242.014 143.617 238.009 143.617 Q236.135 143.617 234.26 144.034 Q232.408 144.451 230.463 145.33 L230.463 127.969 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M44.7161 812.969 L47.5806 812.969 L47.5806 839.896 Q53.6281 839.514 56.8109 836.268 Q59.9619 832.989 59.9619 827.165 Q59.9619 823.791 59.1344 820.64 Q58.3069 817.457 56.6518 814.338 L62.1899 814.338 Q63.5267 817.489 64.227 820.799 Q64.9272 824.109 64.9272 827.515 Q64.9272 836.045 59.9619 841.042 Q54.9967 846.007 46.5303 846.007 Q37.7774 846.007 32.6531 841.296 Q27.4968 836.554 27.4968 828.533 Q27.4968 821.34 32.1438 817.17 Q36.7589 812.969 44.7161 812.969 M42.9973 818.826 Q38.1912 818.889 35.3266 821.531 Q32.4621 824.141 32.4621 828.47 Q32.4621 833.371 35.2312 836.331 Q38.0002 839.259 43.0292 839.705 L42.9973 818.826 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M42.4881 773.724 L64.0042 773.724 L64.0042 779.581 L42.679 779.581 Q37.6183 779.581 35.1038 781.554 Q32.5894 783.528 32.5894 787.474 Q32.5894 792.217 35.6131 794.954 Q38.6368 797.691 43.8567 797.691 L64.0042 797.691 L64.0042 803.58 L28.3562 803.58 L28.3562 797.691 L33.8944 797.691 Q30.6797 795.591 29.0883 792.758 Q27.4968 789.893 27.4968 786.169 Q27.4968 780.027 31.3163 776.876 Q35.1038 773.724 42.4881 773.724 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M44.7161 731.552 L47.5806 731.552 L47.5806 758.479 Q53.6281 758.097 56.8109 754.85 Q59.9619 751.572 59.9619 745.747 Q59.9619 742.373 59.1344 739.222 Q58.3069 736.04 56.6518 732.92 L62.1899 732.92 Q63.5267 736.071 64.227 739.382 Q64.9272 742.692 64.9272 746.097 Q64.9272 754.627 59.9619 759.624 Q54.9967 764.59 46.5303 764.59 Q37.7774 764.59 32.6531 759.879 Q27.4968 755.137 27.4968 747.116 Q27.4968 739.923 32.1438 735.753 Q36.7589 731.552 44.7161 731.552 M42.9973 737.408 Q38.1912 737.472 35.3266 740.114 Q32.4621 742.724 32.4621 747.052 Q32.4621 751.954 35.2312 754.914 Q38.0002 757.842 43.0292 758.288 L42.9973 737.408 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M33.8307 701.283 Q33.2578 702.269 33.0032 703.447 Q32.7167 704.593 32.7167 705.993 Q32.7167 710.959 35.9632 713.632 Q39.1779 716.274 45.2253 716.274 L64.0042 716.274 L64.0042 722.162 L28.3562 722.162 L28.3562 716.274 L33.8944 716.274 Q30.6479 714.428 29.0883 711.468 Q27.4968 708.508 27.4968 704.275 Q27.4968 703.67 27.5923 702.938 Q27.656 702.206 27.8151 701.315 L33.8307 701.283 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M45.7664 672.828 Q39.4007 672.828 35.8996 675.47 Q32.3984 678.08 32.3984 682.822 Q32.3984 687.533 35.8996 690.175 Q39.4007 692.785 45.7664 692.785 Q52.1003 692.785 55.6014 690.175 Q59.1026 687.533 59.1026 682.822 Q59.1026 678.08 55.6014 675.47 Q52.1003 672.828 45.7664 672.828 M59.58 666.972 Q68.683 666.972 73.1071 671.014 Q77.5631 675.056 77.5631 683.395 Q77.5631 686.482 77.0857 689.22 Q76.6401 691.957 75.6852 694.535 L69.9879 694.535 Q71.3884 691.957 72.0568 689.443 Q72.7252 686.928 72.7252 684.318 Q72.7252 678.557 69.7015 675.693 Q66.7096 672.828 60.6303 672.828 L57.7339 672.828 Q60.885 674.642 62.4446 677.475 Q64.0042 680.308 64.0042 684.254 Q64.0042 690.811 59.0071 694.822 Q54.01 698.832 45.7664 698.832 Q37.491 698.832 32.4939 694.822 Q27.4968 690.811 27.4968 684.254 Q27.4968 680.308 29.0564 677.475 Q30.616 674.642 33.7671 672.828 L28.3562 672.828 L28.3562 666.972 L59.58 666.972 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><path clip-path="url(#clip650)" d="M67.3143 640.076 Q73.68 642.559 75.6216 644.914 Q77.5631 647.27 77.5631 651.216 L77.5631 655.895 L72.6615 655.895 L72.6615 652.458 Q72.6615 650.039 71.5157 648.702 Q70.3699 647.365 66.1048 645.742 L63.4312 644.692 L28.3562 659.11 L28.3562 652.903 L56.238 641.763 L28.3562 630.623 L28.3562 624.417 L67.3143 640.076 Z" fill="#000000" fill-rule="nonzero" fill-opacity="1" /><circle clip-path="url(#clip652)" cx="2294.3" cy="1384.24" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip652)" cx="1320.06" cy="1326.7" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip652)" cx="2294.3" cy="925.878" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip652)" cx="2294.3" cy="872.259" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip652)" cx="1320.06" cy="872.259" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip652)" cx="612.483" cy="483.137" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip652)" cx="2027.64" cy="483.137" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip652)" cx="1012.25" cy="483.137" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip652)" cx="1627.87" cy="483.137" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip652)" cx="2294.3" cy="448.789" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip652)" cx="1320.06" cy="448.789" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip652)" cx="1320.06" cy="417.82" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip652)" cx="2294.3" cy="86.1857" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip652)" cx="345.82" cy="86.1857" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip652)" cx="1443.38" cy="86.1857" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<circle clip-path="url(#clip652)" cx="1196.74" cy="86.1857" r="14.4" fill="#009af9" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
</svg>

```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

