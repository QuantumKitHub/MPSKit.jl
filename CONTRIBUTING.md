# Contributing to MPSKit.jl

Thanks for taking the time to contribute!
This document is a short, practical guide to reporting issues, asking questions, and submitting changes.

## Asking questions and reporting bugs

Use [GitHub issues](https://github.com/QuantumKitHub/MPSKit.jl/issues) for both.
There are two templates to help you:

- **Bug report** — for something that doesn't work as expected.
- **Question** — for anything else, from "how do I model X" to "is Y possible".

A good bug report is one we can act on immediately, and it needs two things:

1. A minimal working example (MWE): the smallest snippet of code that reproduces the problem.
   Strip away everything not needed to trigger the bug — unrelated setup, unused imports, alternative approaches you also tried.
2. Version information for MPSKit, TensorKit, and Julia, obtained with:

   ```julia-repl
   julia> using Pkg
   julia> Pkg.status(["MPSKit", "TensorKit"])
   ```

   ```julia-repl
   julia> versioninfo()
   ```

Please also include the full error message and stacktrace, if there is one.

## Contributing code

### Development setup

1. Fork and clone the repository.
2. From the Julia REPL, develop the package against your local checkout:

   ```julia-repl
   pkg> dev /path/to/your/clone/MPSKit.jl
   ```

   or, from a fresh environment:

   ```julia-repl
   julia> using Pkg; Pkg.develop(path = "/path/to/your/clone/MPSKit.jl")
   ```

### Running the tests

The repository root is a Julia workspace (`Project.toml` lists `test`, `docs`, and `examples` as workspace projects), and `test/Project.toml` already points `MPSKit` back at the repo checkout.
From the repository root, run the full suite with:

```
julia --project=test test/runtests.jl
```

The test suite is organized by topic under `test/` (`algorithms/`, `states/`, `operators/`, `misc/`, `gpu/`), with shared setup code in `test/setup/`.
GPU tests only run when a functional CUDA/cuTENSOR install is detected, so most contributors will only ever exercise the CPU tests.
A `--fast` flag is available for a quicker, reduced run while iterating.

### Building the documentation

The docs use Documenter.jl with the DocumenterVitepress backend, so a full render needs Node.js in addition to Julia.

For routine verification of code examples (fast, no Node required), run the doctests directly:

```
julia --project=docs -e 'using Documenter, MPSKit; doctest(MPSKit)'
```

For a full local site build (slower, needs `npm`):

```
julia --project=docs docs/make.jl
```

### Code formatting

This repository is formatted with [Runic](https://github.com/fredrikekre/Runic.jl).
Formatting is checked automatically on pull requests, and there is a [pre-commit](https://pre-commit.com/) hook (`.pre-commit-config.yaml`) that runs Runic locally if you use pre-commit.
Please format any Julia files you touch before opening a PR.

### Changelog

If your change is user-facing (new feature, behavior change, bug fix, deprecation, or removal), add an entry under the `[Unreleased]` section of `CHANGELOG.md`, in the category that matches your change.

### Opening a pull request

Small, focused pull requests are easiest to review.
Please describe what the change does and why, link any related issues, and make sure the checklist in the PR template is filled in before requesting review.
