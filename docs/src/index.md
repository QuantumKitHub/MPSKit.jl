# MPSKit.jl documentation

This code track contains the numerical research and development of the Ghent Quantum Group with regard to tensor network simulation in the julia language. The purpose of this package is to facilitate efficient collaboration between different members of the group.

Topics of research on tensor networks within the realm of this track include:

- Tensor network algorithms (excitations, tdvp, vumps, ...)
- MPS routines (MPS diagonalization, Schmidt Decomposition, MPS left and right multiplication, ...)
- The study of several useful models (nearest neighbour interactions, MPO's, long range interactions, ...)

## Contents

```@contents
```
## Statement of Intent

Quantum mechanics is one of the biggest breakthroughs in the history of physics as, for instance, the successful applications of the theory to the hydrogen atom showed. Currently, almost a century later, physicists are still struggling to fully understand the implications of quantum mechanics. Especially in systems with many degrees of freedom where the complexity escalates quickly.
In recent years, it was realized that quantum entanglement can be exploited to create an efficient ansatz for the ground and  low energy states of a many-body system. These ansatz are known as ‘tensor network states’.

Tensor network states allow for a more efficient use of time and computational resources since one no longer has to look for the correct answer in the entire space of possible states, restricting to the physical subspace of tensor network states is sufficient. The local structure of these states also provide ways to get a deeper understanding of generic emergent phenomena in quantum many-body systems and their connection to entanglement.

Hence, tensor network techniques are finding their way in an increasing number of fields in physics, ranging from condensed matter theory to quantum gravity. In the [UGent quantum group](http://mathphy.ugent.be/wp/quantum/) of Prof. Verstraete and Prof. Haegeman we are at the forefront of many of these applications, with an expertise in both the theoretical and numerical aspects. With this code base we aim to create a flexible, well-optimized and inclusive tool to support and stimulate rapid expansion of this expertise.

## Getting Started

For those who want to dive in immediately, the examples folder is a good starting point. Those who want read up first can use this documentation file.

## Functions

```@autodocs
Modules = [MPSKit]
Private = false
Order   = [:module, :type, :function]
```

### Index
This is the index.

```@index
```

## Supporting and Credit

If you have any questions, bug reports, feature requests, etc., please submit an issue on github. If you would like to contribute to the moral of the coding team, please [send us food](http://www.greenway.be/). The software in this stack was developed as part of academic research of the [UGent Quantum Group](http://mathphy.ugent.be/wp/quantum/) of Prof. Verstraete and Prof. Haegeman.
