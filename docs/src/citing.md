# Citing MPSKit

If you use MPSKit.jl in your research, please cite it.
Doing so supports the maintainers and helps others find the tools you relied on.

## How to cite the software

The canonical citation metadata lives in the
[`CITATION.cff`](https://github.com/QuantumKitHub/MPSKit.jl/blob/master/CITATION.cff)
file at the repository root, which GitHub also exposes through the "Cite this repository" button.
The software is archived on Zenodo under the concept DOI
[`10.5281/zenodo.10654900`](https://doi.org/10.5281/zenodo.10654900),
which always resolves to the latest release.

A ready-to-use BibTeX entry:

```bibtex
@software{mpskitjl,
  author  = {Devos, Lukas and Van Damme, Maarten and Haegeman, Jutho},
  title   = {{MPSKit.jl}},
  version = {v0.13.13},
  doi     = {10.5281/zenodo.10654900},
  url     = {https://github.com/QuantumKitHub/MPSKit.jl},
  year    = {2026}
}
```

## Citing the underlying methods

MPSKit implements algorithms developed in the tensor-network literature.
When a specific method is central to your results, please also consider citing the original method papers.
Some of the key references of the algorithms in this library are included in the [References](@ref) page.
