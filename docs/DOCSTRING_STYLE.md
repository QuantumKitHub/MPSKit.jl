# Docstring style guide

This document defines the conventions for docstrings across MPSKit.jl.
The goal is a single, uniform presentation for all public-facing functionality, so that the API reference reads as one coherent whole.

These rules apply to **every** docstring in `src/`, exported or not.
Not every symbol needs every section — pick the template that fits (see [Templates](#templates)) — but when a docstring documents arguments, returns, fields, examples, or references, it does so in the one format described here.

## Quick rules

- **Section headers use a single hash** (`# Arguments`, `# Returns`), matching Julia Base.
- **Bullet entries** are `` - `name`: description `` — a dash, the name in backticks, a colon, one space, then the description. Add the type (`` `name::Type` ``) only when it is helpful.
- **Cross-references** use `[`name`](@ref)` for internal symbols and `[`name`](@extref Pkg.name)` for symbols in other packages.
- **Type parameter lists** put a space after each comma: `Array{T, N}`, `Union{A, B, C}` — never `Array{T,N}`.
- **Default values** put spaces around the `=`: `tol = 1e-10`, not `tol=1e-10`.
- **Literature references** use `@cite` keys, never inline DOIs or URLs.
- **Examples** that show output are runnable `jldoctest` blocks.
- **Caveats** use `!!! note`; **unstable or experimental** features use `!!! warning`.

## Section headers

Use a single hash (`#`) for all section headers inside a docstring.
The canonical section names, in the order they should appear, are:

1. `# Constructors` — constructor signatures (container types with non-trivial constructors).
2. `# Arguments` — positional arguments.
3. `# Keyword Arguments` — keyword arguments.
4. `# Returns` — the return value(s).
5. `# Fields` — the struct fields, when the raw fields are the public API (algorithm structs; rendered by `$(TYPEDFIELDS)`).
6. `# Properties` — the `getproperty` interface, when it differs from the raw storage (e.g. the gauge views of an MPS container). Use `# Fields` **or** `# Properties`, whichever describes the public surface — not both.
7. `# Notes` — conventions and caveats worth a dedicated block.
8. `# Examples` — runnable examples.
9. `# See also` — related functions; for an algorithm struct, the driver(s) that consume it.
10. `# References` — literature citations.

Omit any section that does not apply.
Do not use `## Arguments` (double hash), `# Keywords`, or other spellings.
For docstrings long enough to warrant it, split off detail into a `# Extended help` section (a Julia Base convention) so the summary line and first paragraph stay terse.

## Bullet format

Document arguments, keyword arguments, returns, and manually-listed properties as bullet lists in this form:

```
- `name`: description
- `name::Type`: description
```

A dash (not `*`), the name in backticks, a colon **with no leading space**, one trailing space, then the description.

Include the type in the backtick span **only when it earns its place** — when it constrains what the caller may pass or disambiguates an overloaded name (e.g. `` `O::Union{AbstractMPO, Pair, AbstractTensorMap}` ``).
Omit it when the type is obvious from the name, the surrounding prose, or the default value (e.g. `` `verbosity`: how much information is displayed ``).
Never repeat a type that `$(TYPEDFIELDS)` already renders from the struct definition.

Give keyword arguments their default in the backtick span when it is informative, with spaces around the `=`: `` - `tol = 1e-10`: convergence tolerance ``.
Always put spaces around `=` when writing a default value in a docstring (both in bullet entries and in signature blocks), even where the underlying code omits them.
Continuation lines of a long description are indented to align under the description text.

## Cross-references and citations

- Internal symbols: `` [`find_groundstate`](@ref) ``.
- External symbols: `` [`Householder`](@extref MatrixAlgebraKit.Householder) ``.
  Note the parentheses — `@extref` only expands the `[text](@extref target)` form, not `[text][target]`.
- Literature: `[Zauner-Stauber et al. Phys. Rev. B 97 (2018)](@cite zauner-stauber2018)`, with the key defined in the bibliography.
  Do not paste raw DOIs or arXiv links.

## Templates

Three templates cover the whole package.
Choose by what the symbol is, not by how important it is.

There are two flavours of type docstring — pick by what the type is.
Algorithm and configuration structs (`A1`) are keyword-configured bags of settings; container/data types (`A2`) hold state and are built through hand-written constructors.

### Template A1 — algorithm and configuration structs

For the keyword-configured `@kwdef` structs: every `Algorithm` subtype, and any similar options struct.
Each field carries a per-field string literal so that `$(TYPEDFIELDS)` renders the field documentation, including its type — the doc strings themselves do not repeat the type.

```julia
"""
$(TYPEDEF)

One paragraph: what the algorithm does and how.

# Fields

$(TYPEDFIELDS)

# See also

Used as the `algorithm` argument of [`find_groundstate`](@ref) and [`leading_boundary`](@ref).

# References

* [Author et al. Journal (Year)](@cite key)
"""
@kwdef struct VUMPS <: Algorithm
    "tolerance for convergence criterium"
    tol::Float64 = 1e-10
    "maximal amount of iterations"
    maxiter::Int = 200
end
```

`$(TYPEDEF)` generates the type signature — do not hand-write it.
The keyword constructor generated by `@kwdef` *is* the field list, so there is no `# Constructors` section; add one only if the type also offers a non-obvious convenience constructor.
`# See also` names the driver function(s) that accept the struct, so the algorithm is discoverable from its own page.
`# References` is optional and only appears when there is literature to cite.

### Template A2 — container and data types

For state-holding types with hand-written constructors: `FiniteMPS`, `InfiniteMPS`, `WindowMPS`, the MPO types, and similar.
Here the raw fields are internal; document the public `getproperty` interface under `# Properties`, and the constructors explicitly.

```julia
"""
$(TYPEDEF)

Type that represents a finite Matrix Product State.

# Constructors
    FiniteMPS([f, eltype], physicalspaces, maxvirtualspaces; kwargs...)
    FiniteMPS([f, eltype], N, physicalspace, maxvirtualspaces; kwargs...)
    FiniteMPS(As::Vector{<:GenericMPSTensor}; kwargs...)

Construct an MPS from physical and virtual spaces, or from a list of tensors `As`.

# Arguments
- `As`: vector of site tensors
- `f = rand`: initializer for tensor data
- `physicalspaces`: list of physical spaces

# Keyword Arguments
- `normalize = true`: normalize the constructed state
- `left`: left-most virtual space

# Properties
- `AL`: left-gauged MPS tensors
- `AR`: right-gauged MPS tensors
- `AC`: center-gauged MPS tensors
- `C`: gauge (bond) tensors

# Notes
By convention, `AL[i] * C[i] == AC[i] == C[i-1] * AR[i]`.
"""
```

Use `$(TYPEDEF)` for the top line here too, so the type signature never drifts from the definition.
The constructors are the user-facing interface and are documented separately: stack their signatures as an indented code block under `# Constructors`, then document their parameters in flat sibling `# Arguments` / `# Keyword Arguments` sections (not nested `### ` sub-headers).

### Template B — full-contract functions

Use for the user-facing verbs: `find_groundstate`, `leading_boundary`, `timestep`, `time_evolve`, `expectation_value`, `changebonds`, `approximate`, `correlator`, and the like.

```julia
"""
    funcname(ψ₀, H, [environments]; kwargs...) -> (ψ, environments, ϵ)

One paragraph describing the operation.

# Arguments
- `ψ₀::AbstractMPS`: initial guess
- `H::AbstractMPO`: the operator

# Keyword Arguments
- `tol::Float64 = 1e-10`: convergence tolerance

# Returns
- `ψ::AbstractMPS`: the converged state
- `ϵ::Float64`: final error estimate

# Examples
```jldoctest
julia> # runnable example
```

# References
* [...](@cite key)
"""
```

The top line is an indented, four-space signature; stack multiple overloads as separate signature lines.
Keep the `-> (...)` return annotation on the signature even when a `# Returns` section is present: the signature is the glanceable form, the section is the contract.
Omit any section that does not apply (a function with no keywords has no `# Keyword Arguments`).

### Template C — lightweight

Use for simple helpers and most internal functions: a signature and a one- or two-sentence description, no sections.

```julia
"""
    correlator(ψ, O1, O2, i, j)
    correlator(ψ, O12, i, j)

Compute the 2-point correlator `⟨ψ|O1[i]O2[j]|ψ⟩`.
Also accepts a range for `j`.
"""
```

## Admonitions

- `!!! note` for caveats and conventions the reader must know (e.g. gauge conventions).
- `!!! warning` for anything unstable or experimental — everything in `lib/internals`, current GPU support, and any feature that may change.

## Attachment

- Prefer a leading `"""..."""` block directly above the definition.
- Use `@doc (@doc a) b` only to alias a genuinely identical docstring onto a sibling.
- A comment between the docstring and the definition silently detaches the docstring; keep them adjacent and put any comment above the docstring.

## Physics claims

Any statement about physical behavior, convergence, or the meaning of a result that has not been verified gets a `<!-- REVIEW: ... -->` comment for the maintainer.
Correctness of physics is the maintainer's call.
