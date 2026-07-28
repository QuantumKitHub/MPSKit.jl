# Upstream notes: DocumenterVitepress

`docs/src/.vitepress/config.mts` is a vendored copy of the DocumenterVitepress (DVP)
v0.3.5 template plus two MPSKit additions.
Neither addition is MPSKit-specific: both fix bugs that every Documenter + DVP site has.
This file records what the upstream fixes should look like, so the vendored copy can be
deleted once they land.

Nothing here has been filed or submitted upstream.

## 1. Docstring section headings render as undifferentiated body text

Tracked in [MPSKit.jl#477](https://github.com/QuantumKitHub/MPSKit.jl/issues/477).

### Diagnosis

Documenter — not DVP — rewrites every markdown heading inside a docstring into a bold-only
paragraph before any writer sees it.
`recursive_heading_to_bold!`, called from `create_docsnode` in
`Documenter/src/expander_pipeline.jl`, replaces the `MarkdownAST.Heading` element with
`Paragraph > Strong`.
So `# Constructors` reaches DVP as `**Constructors**` and is rendered as
`<p><strong>Constructors</strong></p>`.
`template/src/.vitepress/theme/docstrings.css` has no rule for it, so all the structure a
docstring author wrote collapses into body text.

This is invisible on macOS and Windows but not on Linux, because of a second, independent
bug: `template/src/.vitepress/theme/style.css` sets

```css
--vp-font-family-base: "Barlow", "Inter var experimental", "Inter var",
  -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu,
  Cantarell, "Fira Sans", "Droid Sans", "Helvetica Neue", sans-serif;
```

but the only `@font-face` in that file is JuliaMono — Barlow and Inter var are never
loaded.
macOS and Windows fall through to `-apple-system` (SF Pro) and `Segoe UI`, which have real
semibold faces, so VitePress' `strong, b { font-weight: 600 }` reads as bold.
On Linux the request goes to fontconfig for a family that does not exist and the
substituted face renders 600 close to regular.
Worth reporting on its own: the declared stack should either be loaded or not declared.

### Upstream fix

Better done in DVP's writer than in the template's markdown-it config, because
`renderdoc` (`src/writer.jl`) already knows it is inside a docstring, so the pattern can be
matched exactly at the MarkdownAST level — no token walking, no scoping heuristic:

- In `renderdoc`, before rendering a docstring's `mdast`, walk its top-level children (and
  list items, mirroring `recursive_heading_to_bold!`) and detect a `Paragraph` whose single
  child is a `Strong`.
  That shape is exactly what Documenter produces for a docstring heading.
- Emit it as a tagged block rather than a plain paragraph, e.g.
  `<div class="jldocstring-heading">…</div>`.
- Add the styling to `template/src/.vitepress/theme/docstrings.css`.

Deliberately *not* a real `<h4>`: the template sets `themeConfig.outline: 'deep'`, so on an
`@autodocs` page (258 docstrings for MPSKit, ~3 sections each) the "On this page" sidebar
would be swamped.

### What we do instead

A `md.core.ruler` rule in the vendored `config.mts` that tracks whether it is inside a
`<details class='jldocstring …'>` html_block and tags any `paragraph_open` whose inline
content is exactly one `strong`, plus styling in
`docs/src/.vitepress/theme/custom.css`.
Both the docstring scoping and the exact-match requirement are needed: a bare
`p > strong:only-child` CSS selector also matches a paragraph that merely *starts* with
bold text, which occurs in real docstrings.

We also override `--vp-font-family-base` in `custom.css` with the upstream stack minus the
three families that are never loaded.

## 2. Local search cannot target docstrings

Tracked in [MPSKit.jl#478](https://github.com/QuantumKitHub/MPSKit.jl/issues/478).

### Diagnosis

VitePress' local search splits a page into indexable sections at headings only —
`splitPageIntoSections` matches
`/<h(\d*).*?>(<a.*? href="#.*?".*?>.*?<\/a>)<\/h\1>/gi`.

DVP renders each docstring as a raw `<details>` block whose anchor is an `<a id='…'>`
inside `<summary>`, which is not a heading.
Consequently an `@autodocs` page is indexed as one single document.
Measured on the MPSKit build: all 177 docstrings on `lib/lib` collapse into one document
`/lib/lib#lib_index`, titled "Library documentation".
Three symptoms follow:

1. No search entry exists for any individual binding.
2. What entry there is ranks badly — BM25 normalises by field length, so one enormous
   document loses to short manual sections that mention the name once.
3. A hit can only link to the top of the page, never to the docstring.

### Upstream fix

Add an `_render` hook to `themeConfig.search.options` in
`template/src/.vitepress/config.mts` that rewrites each docstring `<summary>` into a
heading *for indexing only* (the rendered page is untouched — `_render`'s output is only
consumed by the indexer):

```ts
const DOCSTRING_SUMMARY =
  /<summary><a id='([^']+)' href='([^']+)'><span class="jlbinding">(.*?)<\/span><\/a>.*?<\/summary>/g

search: {
  provider: 'local',
  options: {
    detailedView: true,
    _render(src, env, md) {
      const html = md.render(src, env)
      if (env.frontmatter?.search === false) return ''
      return html.replace(
        DOCSTRING_SUMMARY,
        (_match, id, href, name) =>
          `<h3 id="${id}">${name} <a class="header-anchor" href="${href}">&#8203;</a></h3>`
      )
    }
  }
}
```

The injected form has to mirror the shape of VitePress' own heading anchors, title *first*
and anchor *after* it, because the indexer reads the section title from the text preceding
the anchor:

```js
const headingRegex = /<h(\d*).*?>(.*?<a.*? href="#.*?".*?>.*?<\/a>)<\/h\1>/gi
const headingContentRegex = /(.*?)<a.*? href="#(.*?)".*?>.*?<\/a>/i
// ... and then, per section:
if (!title || !content) continue
```

The obvious-looking `<h3 id="X"><a href="#X">Name</a></h3>` matches `headingRegex` but
yields an empty title, so `splitPageIntoSections` silently drops every section and the
index is unchanged — worth calling out in any upstream patch.

If DVP would rather not depend on the exact shape of its own `<summary>` output, the
alternative is to have `writer.jl` emit a stable marker (e.g. a `data-jlbinding` attribute)
and key the regex on that.

Verified against the MPSKit build: the regex matches all 258 `<summary>` lines emitted
across `build/.documenter/**/*.md`, with no misses.

### What we do instead

Exactly the snippet above, in the vendored `config.mts`.
Once upstream ships it, delete `docs/src/.vitepress/config.mts` and the template drift
guard at the top of `docs/make.jl`.
