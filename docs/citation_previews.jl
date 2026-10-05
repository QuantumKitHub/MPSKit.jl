# Keep the default DocumenterVitepress config, but parse citations as inline Vue components.
# Otherwise citations at the start of a list item are treated as raw HTML blocks.
struct CitationPreviews <: Documenter.Plugin end

function DocumenterVitepress.vitepress_config_transform(::CitationPreviews, config::String)
    markdown = r"\bmarkdown\s*:\s*\{"
    occursin(markdown, config) || error("Cannot configure citation previews: no markdown config")
    return replace(
        config,
        markdown => "markdown: {\n    component: { inlineTags: ['CitationPreview'] },";
        count = 1,
    )
end
