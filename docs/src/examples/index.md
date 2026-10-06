# Examples

The downloadable notebooks include the model constructors used in these examples.
Their operator definitions come from
[TensorKitTensors.jl](https://github.com/QuantumKitHub/TensorKitTensors.jl).
To run individual code snippets, first download the shared
[models.jl](https://github.com/QuantumKitHub/MPSKit.jl/blob/main/docs/src/assets/models.jl)
file and load it in Julia:

```julia
include("models.jl")
using .ExampleModels
```

## Quantum (1+1)d

```@contents
Pages = map(file -> joinpath("quantum1d", file, "index.md"), readdir("quantum1d"))
Depth = 1
```

## Classical (2+0)d

```@contents
Pages = map(file -> joinpath("classic2d", file, "index.md"), readdir("classic2d"))
Depth = 1
```
