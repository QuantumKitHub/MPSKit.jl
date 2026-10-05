// VitePress indexes headings, so give each docstring summary a heading in the
// search render only. The title must precede the anchor for its section indexer.
const DOCSTRING_SUMMARY =
  /<summary><a id='([^']+)' href='([^']+)'><span class="jlbinding">(.*?)<\/span><\/a>.*?<\/summary>/g

// Documenter turns docstring headings into bold-only paragraphs. Mark paragraphs
// wrapped in one strong span; overrides.css scopes their styling to docstrings.
// Keeping the scope in CSS avoids tracking raw HTML (including nested details).
function docstringHeadings(md) {
  md.core.ruler.push('mpskit_docstring_headings', ({ tokens }) => {
    for (let i = 0; i < tokens.length; i++) {
      if (tokens[i].type !== 'paragraph_open') continue
      const children = tokens[i + 1]?.children?.filter(
        (c) => !(c.type === 'text' && c.content === '')
      )
      if (children?.[0]?.type !== 'strong_open') continue
      // The matching close must end the paragraph, excluding text or another
      // strong span after it while allowing emphasis and links inside the span.
      const end = children.findIndex(
        (c) => c.type === 'strong_close' && c.level === children[0].level
      )
      if (end === children.length - 1) tokens[i].attrJoin('class', 'jldocstring-heading')
    }
  })
}

// Extend the generated config while retaining all upstream plugins and options.
export function withOverrides(config) {
  const configureMarkdown = config.markdown.config
  config.markdown.config = (md) => {
    configureMarkdown(md)
    md.use(docstringHeadings)
  }
  config.themeConfig.search.options._render = (src, env, md) => {
    const html = md.render(src, env)
    if (env.frontmatter?.search === false) return ''
    return html.replace(
      DOCSTRING_SUMMARY,
      (_match, id, href, name) =>
        `<h3 id="${id}">${name} <a class="header-anchor" href="${href}">&#8203;</a></h3>`
    )
  }
  return config
}
