<!--
  SidebarDrawerToggle.vue
  A toggle button that collapses/expands the VitePress sidebar on desktop.
  Sits in the navbar to the left of the search bar.
  Enabled via `sidebar_drawer = true` in MarkdownVitepress config.
  Inspired by: https://www.codingnepalweb.com/sidebar-with-dark-light-themes-html-javascript/
-->
<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useData } from 'vitepress'

const { theme, frontmatter } = useData()
const enabled = computed(() => (theme.value as any).sidebarDrawer === true)
const isHomePage = computed(() => frontmatter.value.layout === 'home')
const shouldShow = computed(() => enabled.value && !isHomePage.value)
const collapsed = ref(false)
const STORAGE_KEY = 'dv-sidebar-collapsed'

function toggleSidebar() {
  collapsed.value = !collapsed.value
  updateDOM()
  if (typeof localStorage !== 'undefined') {
    localStorage.setItem(STORAGE_KEY, String(collapsed.value))
  }
}

function updateDOM() {
  if (typeof document === 'undefined') return
  document.documentElement.classList.toggle('sidebar-drawer-collapsed', collapsed.value)
}

onMounted(() => {
  if (!enabled.value) return
  const saved = localStorage.getItem(STORAGE_KEY)
  if (saved === 'true') {
    collapsed.value = true
    updateDOM()
  }
})
</script>

<template>
  <button
    v-if="shouldShow"
    class="sidebar-drawer-btn"
    :class="{ 'is-collapsed': collapsed }"
    @click="toggleSidebar"
    :title="collapsed ? 'Show sidebar' : 'Hide sidebar'"
    :aria-label="collapsed ? 'Show sidebar' : 'Hide sidebar'"
  >
    <!-- "panel left close/open" icon -->
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
      class="sidebar-drawer-icon"
    >
      <!-- Outer frame -->
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <!-- Vertical divider -->
      <line x1="9" y1="3" x2="9" y2="21" />
      <!-- Arrow (collapse/expand) -->
      <polyline :points="collapsed ? '13 9 16 12 13 15' : '16 15 13 12 16 9'" />
    </svg>
  </button>
</template>

<style scoped>
.sidebar-drawer-btn {
  display: none; /* Hidden on mobile, shown on desktop */
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  padding: 0;
  margin-right: -8px;
  border-radius: 8px;
  border: none;
  background: transparent;
  color: var(--vp-c-text-1);
  cursor: pointer;
  transition: color 0.25s, background-color 0.25s;
}

.sidebar-drawer-btn:hover {
  color: var(--vp-c-text-2);
  background: var(--vp-c-default-soft);
}

.sidebar-drawer-btn:active {
  transform: scale(0.92);
}

.sidebar-drawer-icon {
  display: block;
}

/* Only show on desktop (matches VitePress sidebar breakpoint) */
@media (min-width: 960px) {
  .sidebar-drawer-btn {
    display: flex;
  }
}
</style>
