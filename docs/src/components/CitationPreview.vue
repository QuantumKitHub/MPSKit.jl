<!--
Adapted from DocumenterVitepress.jl v0.3.7 to support cross-page previews.

MIT License

Copyright (c) 2023 LuxDL

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-->

<template>
  <span
    ref="wrapperRef"
    class="dv-citation-wrapper"
    @mouseover="handleMouseOver"
    @mouseleave="handleMouseLeave"
    @focusin="handleMouseOver"
    @focusout="handleMouseLeave"
  >
    <slot />
    <Teleport to="body">
      <Transition name="dv-citation-fade">
        <div
          v-if="visible && previewContent"
          ref="popoverRef"
          class="dv-citation-popover"
          :style="popoverStyle"
          @mouseenter="clearHideTimer"
          @mouseleave="scheduleHide"
        >
          <div class="dv-citation-popover-content" v-html="previewContent"></div>
        </div>
      </Transition>
    </Teleport>
  </span>
</template>

<script>
// Shared across component instances; failed requests are evicted so hovers can retry.
const bibliographyPages = new Map()
</script>

<script setup>
import { ref, reactive, onMounted, onUnmounted } from 'vue'

const wrapperRef = ref(null)
const popoverRef = ref(null)
const visible = ref(false)
const previewContent = ref('')
const popoverStyle = reactive({
  top: '0px',
  left: '0px',
  maxWidth: '440px',
  transform: 'none',
})

let hideTimer = null
let showTimer = null
let hoverRequest = 0

function cleanReferenceHtml(el) {
  if (!el) return ''
  const container = el.closest('li') || el.closest('dd') || el.parentElement
  if (!container) return ''
  const clone = container.cloneNode(true)
  // Keep the preview focused on the reference, without its navigation backlinks.
  const backlinks = clone.querySelectorAll('.citation-backlinks, a[href*="-cite-"]')
  backlinks.forEach((bl) => bl.remove())
  // Remove anchor tag if empty
  const anchor = clone.querySelector('.dv-bib-anchor, a[id]')
  if (anchor && anchor.textContent.trim() === '') {
    anchor.remove()
  }
  return clone.innerHTML.trim()
}

function updatePosition(targetEl) {
  if (!targetEl) return
  const rect = targetEl.getBoundingClientRect()
  const popoverWidth = Math.min(440, window.innerWidth - 24)
  
  let left = rect.left + rect.width / 2 - popoverWidth / 2
  if (left < 12) left = 12
  if (left + popoverWidth > window.innerWidth - 12) {
    left = window.innerWidth - popoverWidth - 12
  }

  const spaceAbove = rect.top
  const popoverEstimatedHeight = 120
  let top = 0

  if (spaceAbove > popoverEstimatedHeight + 10) {
    top = rect.top - 8
    popoverStyle.transform = 'translateY(-100%)'
  } else {
    top = rect.bottom + 8
    popoverStyle.transform = 'none'
  }

  popoverStyle.left = `${left}px`
  popoverStyle.top = `${top}px`
  popoverStyle.maxWidth = `${popoverWidth}px`
}

async function referenceHtml(link, targetId) {
  const url = new URL(link.href)
  if (url.origin !== window.location.origin) return ''
  const current = new URL(window.location.href)
  const pagePath = (pathname) => pathname.replace(/\.html$/, '').replace(/\/$/, '')
  if (pagePath(url.pathname) === pagePath(current.pathname) && url.search === current.search) {
    return cleanReferenceHtml(document.getElementById(targetId))
  }
  url.hash = ''
  const key = url.href
  if (!bibliographyPages.has(key)) {
    const request = fetch(key).then(async (response) => {
      if (!response.ok) throw new Error('Cannot load bibliography')
      return new DOMParser().parseFromString(await response.text(), 'text/html')
    }).catch((error) => {
      bibliographyPages.delete(key)
      throw error
    })
    bibliographyPages.set(key, request)
  }
  const page = await bibliographyPages.get(key)
  const html = cleanReferenceHtml(page.getElementById(targetId))
  // Resolve links against the bibliography's URL, rather than the citing page.
  const preview = document.createElement('div')
  preview.innerHTML = html
  for (const anchor of preview.querySelectorAll('a[href]')) {
    anchor.href = new URL(anchor.getAttribute('href'), url).href
  }
  return preview.innerHTML
}

function handleMouseOver(e) {
  const link = e.target.closest('a')
  if (!link) return
  const href = link.getAttribute('href')
  if (!href) return
  
  const hashIndex = href.indexOf('#')
  if (hashIndex === -1) return
  const targetId = decodeURIComponent(href.slice(hashIndex + 1))
  if (!targetId || /-cite-\d+$/.test(targetId)) return

  clearHideTimer()
  if (showTimer) clearTimeout(showTimer)
  const request = ++hoverRequest
  
  showTimer = setTimeout(async () => {
    try {
      const html = await referenceHtml(link, targetId)
      if (!html || request !== hoverRequest) return
      previewContent.value = html
      updatePosition(link)
      visible.value = true
    } catch {
      // The citation link remains usable if its preview cannot be loaded.
    }
  }, 100)
}

function handleMouseLeave() {
  hoverRequest++
  if (showTimer) clearTimeout(showTimer)
  scheduleHide()
}

function clearHideTimer() {
  if (hideTimer) {
    clearTimeout(hideTimer)
    hideTimer = null
  }
}

function scheduleHide() {
  clearHideTimer()
  hideTimer = setTimeout(() => {
    visible.value = false
    previewContent.value = ''
  }, 200)
}

function handleScrollOrResize() {
  hoverRequest++
  if (showTimer) clearTimeout(showTimer)
  if (visible.value) {
    visible.value = false
  }
}

onMounted(() => {
  window.addEventListener('scroll', handleScrollOrResize, { passive: true })
  window.addEventListener('resize', handleScrollOrResize, { passive: true })
})

onUnmounted(() => {
  hoverRequest++
  window.removeEventListener('scroll', handleScrollOrResize)
  window.removeEventListener('resize', handleScrollOrResize)
  clearHideTimer()
  if (showTimer) clearTimeout(showTimer)
})
</script>

<style>
.dv-citation-wrapper {
  display: inline;
}

.dv-citation-popover {
  position: fixed;
  z-index: 1000;
  padding: 10px 14px;
  font-size: 0.85rem;
  line-height: 1.5;
  color: var(--vp-c-text-1);
  background-color: var(--vp-c-bg-elv);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  box-shadow: var(--vp-shadow-3);
  pointer-events: auto;
  font-family: var(--vp-font-family-base);
  word-break: break-word;
}

.dv-citation-popover-content p {
  margin: 0;
  line-height: 1.5;
}

.dv-citation-popover-content a {
  color: var(--vp-c-brand-1);
  text-decoration: underline;
  text-underline-offset: 2px;
}

.dv-citation-fade-enter-active,
.dv-citation-fade-leave-active {
  transition: opacity 0.15s ease, transform 0.15s ease;
}

.dv-citation-fade-enter-from,
.dv-citation-fade-leave-to {
  opacity: 0;
}
</style>
