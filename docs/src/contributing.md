# Contributing

Contributions to MPSKit.jl are welcome, from bug reports and questions to documentation
fixes and new algorithms.
This page summarizes the practical workflow; the authoritative version is the
[`CONTRIBUTING.md`](https://github.com/QuantumKitHub/MPSKit.jl/blob/master/CONTRIBUTING.md)
file in the repository.

## Asking questions and reporting bugs

Questions and bug reports both go through
[GitHub issues](https://github.com/QuantumKitHub/MPSKit.jl/issues), which provides two
templates: a **bug report** for things that do not work as expected, and a **question** for
anything else.
General discussion is also welcome on
[GitHub Discussions](https://github.com/QuantumKitHub/MPSKit.jl/discussions).

A good bug report includes a minimal working example (the smallest snippet that reproduces
the problem), the full error message and stacktrace, and version information for MPSKit,
TensorKit, and Julia:

```julia-repl
julia> using Pkg
julia> Pkg.status(["MPSKit", "TensorKit"])

julia> versioninfo()
```

## Development setup

Fork and clone the repository, then develop the package against your local checkout from the
Julia REPL:

```julia-repl
pkg> dev /path/to/your/clone/MPSKit.jl
```

## Running the tests

The repository root is a Julia workspace whose `Project.toml` lists `test`, `docs`, and
`examples` as workspace projects.
Run the full test suite from the repository root with:

```
julia --project=test test/runtests.jl
```

The tests are organized by topic under `test/` (`algorithms/`, `states/`, `operators/`,
`misc/`, `gpu/`).
GPU tests only run when a functional CUDA/cuTENSOR install is detected, so most contributors
exercise only the CPU tests.
A `--fast` flag is available for a quicker, reduced run while iterating.

## Code formatting

The repository is formatted with [Runic](https://github.com/fredrikekre/Runic.jl).
Formatting is checked automatically on pull requests, and a
[pre-commit](https://pre-commit.com/) hook (`.pre-commit-config.yaml`) runs Runic locally if
you use pre-commit.
Please format any Julia files you touch before opening a pull request.

## Building the documentation

The documentation uses [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl) with the
[DocumenterVitepress](https://github.com/LuxDL/DocumenterVitepress.jl) backend, so a full site
render needs Node.js (with `npm`) in addition to Julia.
For routine verification of code examples, doctests run on the Documenter side and need no
Node:

```
julia --project=docs -e 'using Documenter, MPSKit; doctest(MPSKit)'
```

A full local site build is:

```
julia --project=docs docs/make.jl
```

## Opening a pull request

Keep pull requests small and focused.
Describe what the change does and why, link any related issues, and fill in the checklist in
the PR template.
If your change is user-facing (a new feature, behavior change, bug fix, deprecation, or
removal), add an entry under the `[Unreleased]` section of the [Changelog](@ref) in the
category that matches your change.
