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

## 3. Bold text does not render bold

Follow-up to §1, reported on [MPSKit.jl#449](https://github.com/QuantumKitHub/MPSKit.jl/pull/449#issuecomment-5129465790):
after the docstring headings were tagged and styled at `font-weight: 700`, the computed
weight was correct but the glyphs still rendered at regular weight on Linux.

### Diagnosis

VitePress' reset sets `font-synthesis: style` on `<body>`.
Per the CSS spec, listing only `style` permits synthetic *oblique* and **disables synthetic
bold**.
So whenever the resolved face has no real bold, `font-weight: 700` computes correctly and
still paints at regular weight — silently, and only on machines whose font stack resolves
that way.
Combined with §1's stack of families that are never loaded, that is the whole of #477.

This affects far more than docstring headings: `.custom-block-title` (600),
`.custom-block a` (600), `.jldocstring.custom-block summary` (700) and every inline
`<strong>` are all subject to it.

### Upstream fix

Either set `font-synthesis: weight style` in
`template/src/.vitepress/theme/style.css`, or — better — stop declaring families in
`--vp-font-family-base` that the template never loads, so a face with a real bold is
selected in the first place.
Ideally both: the first is a safety net for whatever the user's system resolves.

### What we do instead

`body { font-synthesis: weight style }` in `docs/src/.vitepress/theme/custom.css`.

## 4. Admonitions lost their Documenter styling

Reported on [MPSKit.jl#449](https://github.com/QuantumKitHub/MPSKit.jl/pull/449#issuecomment-5129697729).

### Diagnosis

Three things stack up:

1. VitePress custom blocks are flatter than Documenter's admonitions: the title is not
   colour-coded and carries no icon, and `--vp-custom-block-*-border` defaults to
   `transparent`, so there is no visible border either.
   Documenter draws a bordered box with a bold, category-coloured header prefixed by
   `fa-circle-exclamation`.
2. The title's `font-weight: 600` is subject to the synthetic-bold problem in §3, so on
   affected systems it is not even bold.
3. `writer.jl` maps `note` onto the `tip` container ("Julia markdown says note, but
   Vitepress says tip") and `template/src/.vitepress/theme/style.css` then overrides, under
   `:root.dark`, `--vp-custom-block-tip-bg: var(--vp-dark-gray-mute)` and
   `--vp-custom-block-tip-text: var(--vp-dark-subtext)`.
   Every `!!! note` therefore renders as a flat grey box with dimmed text in dark mode.

### Upstream fix

Careful: VitePress 1.6.4 registers only **five** `:::` containers — `tip`, `info`,
`warning`, `danger`, `details` (`containerPlugin` in its `dist/node` bundle). The
`.custom-block.note`, `.custom-block.important` and `.custom-block.caution` classes in its
stylesheet exist for GitHub-style alerts (`> [!NOTE]`), *not* for containers. Emitting
`::: note` renders the literal text `::: note Hello` as a paragraph — verified against
`createMarkdownRenderer`. That is also why
[#289](https://github.com/LuxDL/DocumenterVitepress.jl/pull/289), which passes every
Documenter category straight through, cannot work as written: `::: todo` / `::: theorem`
are not registered and fall through as text.

So the fix cannot simply be "stop remapping". It needs to keep mapping onto a registered
container while *preserving the original Documenter category* so that CSS can reach it —
which is precisely what [#288](https://github.com/LuxDL/DocumenterVitepress.jl/issues/288)
asks for ("the produced html does not even have the information i need to apply some
css") and what [#95](https://github.com/LuxDL/DocumenterVitepress.jl/issues/95) needs for
`!!! compat`. Emitting the block as raw HTML in VitePress' own custom-block shape, with an
extra `jl-admonition-$(category)` class, achieves both and is not held back by which
container names VitePress happens to register.

Alongside that:

- Reconsider the dark-mode `tip` override, which drains the colour out of the single most
  common admonition in any Julia docs.
- Consider giving `.custom-block-title` a category colour and an icon in the template, to
  close the gap with Documenter.

### What we do instead

Styling only, in `docs/src/.vitepress/theme/custom.css`: accent-coloured bold title with an
inline-SVG `fa-circle-exclamation` mask, a 4px accent left rule, body text restored to
`--vp-c-text-1`, and `tip` backgrounds restated as `--vp-c-tip-soft` so notes keep their
colour in dark mode.
All of it guarded by `:not(.jldocstring)`, because DVP's docstring `<details>` also carries
the `custom-block` class.

We do **not** try to recover the note/tip distinction. By the time the markdown reaches
markdown-it both are `::: tip`, and the only remaining signal is the title — which
Documenter defaults to `"Note"`, but 13 of this repo's 44 notes carry a custom title and
are then indistinguishable from a real `!!! tip`. Colouring only the recognisable ones
would be visibly inconsistent, which is worse than a uniform palette. This needs the
upstream fix.

## Existing upstream work

Checked before writing any of the above; §1 and §3 appear to be unreported.

| Ours | Upstream | Relationship |
| --- | --- | --- |
| §2 search | [#190](https://github.com/LuxDL/DocumenterVitepress.jl/issues/190) "Search engine seems worst than Documenter.jl" | The issue our fix closes. |
| §2 search | [#385](https://github.com/LuxDL/DocumenterVitepress.jl/pull/385), [#386](https://github.com/LuxDL/DocumenterVitepress.jl/pull/386) | **Orthogonal.** Both tune MiniSearch (Julia-aware tokenizer, stop words, boosts) but leave the section splitting alone, so a whole `@autodocs` page stays one document with no per-docstring anchor. Better tokenisation cannot deep-link to something that was never indexed as a section. The two changes compose. |
| §2 search | [#384](https://github.com/LuxDL/DocumenterVitepress.jl/pull/384) | Overlapping but far larger: a custom Vue modal plus a writer-generated second index. Its author expects it to be opt-in rather than default, and #385's comparison notes its page/section records currently carry empty text. Our fix is ~10 lines inside the index VitePress already builds. |
| §2 search | [#119](https://github.com/LuxDL/DocumenterVitepress.jl/issues/119) "fine tune miniSearch options" | Adjacent; addressed by #385/#386, not by us. |
| §4 admonitions | [#289](https://github.com/LuxDL/DocumenterVitepress.jl/pull/289) | Same area, but drops the category fallback entirely, which emits unregistered containers — see above. Stale since March 2026 and marked unstable. |
| §4 admonitions | [#288](https://github.com/LuxDL/DocumenterVitepress.jl/issues/288), [#95](https://github.com/LuxDL/DocumenterVitepress.jl/issues/95) | The issues a category-preserving fix would close. |
