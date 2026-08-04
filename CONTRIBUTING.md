# Contributing to MPSKit.jl

Thanks for taking the time to contribute!
We are open to any and all suggestions and welcome new and advanced developers alike.
This document is a short, practical guide to reporting issues, asking questions, and submitting changes.

## Asking questions and reporting bugs

Use [GitHub issues](https://github.com/QuantumKitHub/MPSKit.jl/issues) for both.
There are two templates to help you:

- **Bug report** — for something that doesn't work as expected.
- **Question** — for anything else, from "how do I model X" to "is Y possible".

Open-ended discussion that isn't really a question or a bug is also welcome on [GitHub Discussions](https://github.com/QuantumKitHub/MPSKit.jl/discussions).

A good bug report is one we can act on immediately, and it typically needs two things:

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

We are very happy to take contributions, and will gladly make the time to review code.
Note that this might take a bit of time, so please be patient with us.

For small issues, fixes and typos, feel free to open a PR directly.
For larger amounts of work, it might be beneficial to open an issue first, so we can discuss the solution strategy beforehand.
Often, it might be the case that there are other ideas, or partial solutions that already were in the make, and we want to avoid duplicate or wasted work as much as possible.

There is no restriction for the use of AI-tooling to assist you, although we do ask that you take the time to review the code yourself.
Additionally, please do not just copy-paste LLM generated responses into code reviews or discussions, as we take the time and effort to review your code and expect the same amount of effort from you.
Do note however, that this means that overly large PRs might not get reviewed, since the review process for these is too resource-intensive.

### Development setup

Fork and clone the repository, then point Julia at your local checkout instead of the released version:

```julia-repl
pkg> dev /path/to/your/clone/MPSKit.jl
```

Alternatively, you can tell Julia to clone the repository automatically, which will put it in a local `dev/` folder next to your Project.toml:

```julia-repl
pkg> dev --local MPSKit
```

If you just want to try out the current `main` without a fork, `pkg> add MPSKit#main` is enough.

### Running the tests

The repository root is a Julia workspace (`Project.toml` lists `test`, `docs`, and `examples` as workspace projects), and `test/Project.toml` already points `MPSKit` back at the repo checkout.
From the repository root, you can therefore run the full suite with:

```
julia --project=test test/runtests.jl
```

The test suite is organized by topic under `test/` (`algorithms/`, `states/`, `operators/`, `misc/`, `gpu/`), with shared setup code in `test/setup/`.
GPU tests only run when a functional CUDA/cuTENSOR install is detected, so most contributors will only ever exercise the CPU tests.

Additionally, a number of CLI flags can be added to run the tests selectively, as these will be filtered by folder and filename
A `--fast` flag is available for a quicker, reduced run while iterating.

```
julia --project=test test/runtests.jl states
julia --project=test test/runtests.jl operators algorithms/groundstate
julia --project=test test/runtests.jl misc --fast
```

### Building the documentation

The docs use Documenter.jl with the DocumenterVitepress backend, so a full render needs Node.js in addition to Julia.
For routine verification of code examples (fast, no Node required), you can run the doctests directly:

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
Please format any Julia files you touch before opening a PR, and refrain from making any formatting changes to code that is not relevant to your PR.

### Changelog entries

If your change is user-facing (new feature, behavior change, bug fix, deprecation, or removal), add an entry under the `[Unreleased]` section of `docs/src/changelog.md`, in the category that matches your change.

### Opening a pull request

Small, focused pull requests are easiest to review.
Please describe what the change does and why, link any related issues, and make sure the checklist in the PR template is filled in before requesting review.
