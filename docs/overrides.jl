# Apply local overrides after DocumenterVitepress generates its upstream config.
struct VitepressOverrides <: Documenter.Plugin end

function DocumenterVitepress.vitepress_config_transform(::VitepressOverrides, config::String)
    marker = r"(?m)^export\s+default\s+"
    occursin(marker, config) || error("DocumenterVitepress config has no default export to extend")
    return replace(config, marker => "const upstreamConfig = "; count = 1) * """

                import { withOverrides } from './overrides.mts'
                export default withOverrides(upstreamConfig)
                """
end
