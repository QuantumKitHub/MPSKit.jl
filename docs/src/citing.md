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

<!-- REVIEW: The BibTeX `version` (v0.13.13) and `year` (2026) are copied from the
     repository README as of this writing. Please confirm they match the release you
     intend readers to cite, and update if a newer version should be referenced.
     Zenodo also mints a version-specific DOI per release; if you prefer readers cite a
     specific version rather than the concept DOI above, add that DOI here. -->

## Citing the underlying methods

MPSKit implements algorithms developed in the tensor-network literature.
When a specific method is central to your results, please also cite the original method paper.
Some of the key references, all included in the [References](@ref) bibliography, are:

- **DMRG / tangent-space methods** — [vanderstraeten2019](@cite), a review of tangent-space methods for uniform matrix product states.
- **VUMPS** — [zauner-stauber2018](@cite), variational optimization algorithms for uniform matrix product states.
- **TDVP (time evolution)** — [haegeman2011](@cite), the time-dependent variational principle for quantum lattices.
- **Quasiparticle excitations** — [haegeman2013](@cite), elementary excitations in gapped quantum spin systems.

<!-- REVIEW: The four method references above are real keys present in
     docs/src/assets/mpskit.bib, matched to their algorithms by paper title. Please
     confirm these are the citations you want highlighted for each method (e.g. whether
     DMRG should point to an original DMRG reference rather than the tangent-space
     review), and add or swap keys as appropriate. -->

The full bibliography, including works that have used MPSKit, is on the [References](@ref) page.
