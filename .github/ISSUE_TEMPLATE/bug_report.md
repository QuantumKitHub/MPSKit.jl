---
name: Bug report
about: Something isn't working as expected
title: ""
labels: bug
---

## Description

A clear and concise description of what went wrong, and what you expected to happen instead.

## Minimal working example (MWE)

Please include the smallest possible piece of code that reproduces the issue.
Trim away everything that is not needed to trigger the bug (unrelated setup, unused imports, etc).

```julia
using MPSKit, TensorKit

# ...
```

## Error / output

Paste the full error message and stacktrace (or the incorrect output), if any.

```
paste here
```

## Version info

Please include the versions of MPSKit and its main dependencies.
From the Julia REPL:

```julia-repl
julia> using Pkg
julia> Pkg.status(["MPSKit", "TensorKit"])
```

Also include the Julia version:

```julia-repl
julia> versioninfo()
```

## Additional context

Anything else that might help (OS, whether it also happens on the latest `main`, related issues, etc).
