// ============================================================================
// VENDORED FILE — keep in sync with DocumenterVitepress.
//
// This is a verbatim copy of the DocumenterVitepress **v0.3.5** template at
// `template/src/.vitepress/config.mts`, plus the MPSKit additions marked with
// `MPSKit:` below. DocumenterVitepress copies this file into the build as-is
// and then performs its `REPLACE_ME_DOCUMENTER_VITEPRESS*` substitutions on it
// (see `modify_config_file` in DocumenterVitepress/src/vitepress_config.jl), so
// the nav, sidebar, base, outDir, ... all keep working.
//
// We only vendor it because `themeConfig.search.options` and `markdown.config`
// are not reachable from `docs/make.jl`. Both additions below belong upstream
// in DocumenterVitepress — see `docs/UPSTREAM_DVP_NOTES.md`. Once upstream
// ships them, DELETE this file and the drift guard in `docs/make.jl`.
//
// `docs/make.jl` hashes the installed DocumenterVitepress template and warns if
// it has changed, which means this copy needs re-syncing.
// ============================================================================

import { defineConfig } from 'vitepress'
import { tabsMarkdownPlugin } from 'vitepress-plugin-tabs'
import { mathjaxPlugin } from './mathjax-plugin'
import { juliaReplTransformer } from './julia-repl-transformer'
import footnote from "markdown-it-footnote";
import path from 'path'

const mathjax = mathjaxPlugin()

function getBaseRepository(base: string): string {
  if (!base || base === '/') return '/';
  const parts = base.split('/').filter(Boolean);
  return parts.length > 0 ? `/${parts[0]}/` : '/';
}

// MPSKit: ---------------------------------------------------------------
// DocumenterVitepress renders each docstring as a raw `<details>` block whose
// anchor lives in the `<summary>`:
//
//   <summary><a id='MPSKit.FiniteMPS-lib-states' href='#MPSKit.FiniteMPS-lib-states'>
//     <span class="jlbinding">MPSKit.FiniteMPS</span></a> <Badge ... /></summary>
//
// VitePress' local search only splits a page into indexable sections at
// headings (`<hN>...<a href="#...">...</a></hN>`), so an `@autodocs` page is
// indexed as one single huge document: docstrings rank poorly against short
// manual sections and every hit links to the top of the page. Rewriting each
// summary into a heading *for indexing only* gives one search entry per
// docstring, titled with the binding and deep-linking to its anchor.
// See QuantumKitHub/MPSKit.jl#478.
//
// The replacement has to mirror the shape VitePress' own heading anchors have,
// because the indexer reads the section title from the text *before* the anchor
// (`headingContentRegex = /(.*?)<a.*? href="#(.*?)".*?>.*?<\/a>/i`) and drops any
// section whose title comes out empty.
const DOCSTRING_SUMMARY =
  /<summary><a id='([^']+)' href='([^']+)'><span class="jlbinding">(.*?)<\/span><\/a>.*?<\/summary>/g

// MPSKit: ---------------------------------------------------------------
// Documenter rewrites every markdown heading inside a docstring into a
// bold-only paragraph before any writer sees it (`recursive_heading_to_bold!`
// in Documenter/src/expander_pipeline.jl), so `# Constructors` reaches
// VitePress as `<p><strong>Constructors</strong></p>` with nothing to style.
// Tag exactly those paragraphs so `theme/custom.css` can render them as
// section headings again. See QuantumKitHub/MPSKit.jl#477.
//
// Scoped to docstring `<details>` blocks, and requires the paragraph to consist
// of nothing but one `<strong>`, so that a paragraph merely *starting* with
// bold text (e.g. "**Not every algorithm ...** — see the table below") is left
// alone.
function docstringHeadings(md) {
  md.core.ruler.push('mpskit_docstring_headings', (state) => {
    const tokens = state.tokens
    let depth = 0
    for (let i = 0; i < tokens.length; i++) {
      const token = tokens[i]

      if (token.type === 'html_block') {
        if (/<details[^>]*\bclass=['"][^'"]*\bjldocstring\b/.test(token.content)) depth++
        else if (depth > 0 && /<\/details>/.test(token.content)) depth--
        continue
      }

      if (depth === 0 || token.type !== 'paragraph_open') continue
      const inline = tokens[i + 1]
      if (!inline || inline.type !== 'inline' || !inline.children) continue

      // markdown-it emits empty `text` tokens around inline markup.
      const children = inline.children.filter(
        (c) => !(c.type === 'text' && c.content === '')
      )
      if (children.length < 2) continue
      if (children[0].type !== 'strong_open') continue
      if (children[children.length - 1].type !== 'strong_close') continue

      token.attrJoin('class', 'jldocstring-heading')
    }
  })
}
// -----------------------------------------------------------------------

const baseTemp = {
  base: 'REPLACE_ME_DOCUMENTER_VITEPRESS',// TODO: replace this in makedocs!
}

const navTemp = {
  nav: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
}

const nav = [
  ...navTemp.nav,
  {
    component: 'VersionPicker'
  }
]

// https://vitepress.dev/reference/site-config
export default defineConfig({
  base: 'REPLACE_ME_DOCUMENTER_VITEPRESS',// TODO: replace this in makedocs!
  title: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  description: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  lastUpdated: true,
  cleanUrls: true,
  outDir: 'REPLACE_ME_DOCUMENTER_VITEPRESS', // This is required for MarkdownVitepress to work correctly...
  head: [
    ['link', { rel: 'icon', href: 'REPLACE_ME_DOCUMENTER_VITEPRESS_FAVICON' }],
    ['script', {src: `${getBaseRepository(baseTemp.base)}versions.js`}],
    // ['script', {src: '/versions.js'], for custom domains, I guess if deploy_url is available.
    ['script', {src: `${baseTemp.base}siteinfo.js`}],
    // REPLACE_ME_DOCUMENTER_VITEPRESS_NOINDEX
  ],

  markdown: {
    codeTransformers: [juliaReplTransformer()],
    config(md) {
      md.use(tabsMarkdownPlugin);
      md.use(footnote);
      mathjax.markdownConfig(md);
      md.use(docstringHeadings); // MPSKit: see above
    },
    theme: {
      light: "github-light",
      dark: "github-dark"
    },
  },
  vite: {
    plugins: [
      mathjax.vitePlugin,
    ],
    define: {
      __DEPLOY_ABSPATH__: JSON.stringify('REPLACE_ME_DOCUMENTER_VITEPRESS_DEPLOY_ABSPATH'),
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '../components')
      }
    },
    optimizeDeps: {
      exclude: [
        '@nolebase/vitepress-plugin-enhanced-readabilities/client',
        'vitepress',
        '@nolebase/ui',
      ],
    },
    ssr: {
      noExternal: [
        // If there are other packages that need to be processed by Vite, you can add them here.
        '@nolebase/vitepress-plugin-enhanced-readabilities',
        '@nolebase/ui',
      ],
    },
  },
  themeConfig: {
    outline: 'deep',
    logo: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    search: {
      provider: 'local',
      options: {
        detailedView: true,
        // MPSKit: index every docstring as its own section — see above.
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
    },
    nav,
    sidebar: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    sidebarDrawer: 'REPLACE_ME_DOCUMENTER_VITEPRESS_SIDEBAR_DRAWER',
    editLink: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    socialLinks: [
      { icon: 'github', link: 'REPLACE_ME_DOCUMENTER_VITEPRESS' }
    ],
    footer: {
      message: 'Made with <a href="https://luxdl.github.io/DocumenterVitepress.jl/dev/" target="_blank"><strong>DocumenterVitepress.jl</strong></a><br>',
      copyright: `© Copyright ${new Date().getUTCFullYear()}.`
    }
  }
})
