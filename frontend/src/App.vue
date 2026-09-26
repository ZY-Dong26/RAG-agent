<script setup>
// App.vue —— 单页问答界面：管理本页会话、服务状态和输入框。
// 对话只在内存中保存；向后端提交时只发送当前问题，不发送页面上的历史消息。
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import {
  ArrowUp, BookOpen, Database, FileText, Moon, PanelLeftClose,
  PanelLeftOpen, Search, Sparkles, SquarePen, Sun,
} from '@lucide/vue'
import ChatMessage from './components/ChatMessage.vue'
import { askQuestion, getStatus } from './lib/api.js'

// 会话列表是当前页面的展示状态；刷新后重新创建空会话。
const conversations = ref([])
const activeId = ref(null)
const draft = ref('')
const search = ref('')
const sending = ref(false)
// status 来自 GET /api/v1/status。ready 只表示本地模型与索引已加载。
const status = ref(null)
const statusError = ref('')
const statusLoading = ref(false)
const sidebarCollapsed = ref(false)
const mobileSidebarOpen = ref(false)
// 只有主题偏好写入 localStorage，问题与回答不会持久化。
const darkMode = ref(localStorage.getItem('rag-agent-theme') === 'dark')
const messagesViewport = ref(null)
const composerInput = ref(null)
let statusTimer

const activeConversation = computed(() => conversations.value.find(item => item.id === activeId.value))
const activeMessages = computed(() => activeConversation.value?.messages ?? [])
const isReady = computed(() => Boolean(status.value?.ready))
const filteredConversations = computed(() =>
  conversations.value.filter(item => item.title.toLowerCase().includes(search.value.trim().toLowerCase()))
)

watch(darkMode, value => {
  document.documentElement.dataset.theme = value ? 'dark' : 'light'
  localStorage.setItem('rag-agent-theme', value ? 'dark' : 'light')
}, { immediate: true })

// 按输入内容调整文本框高度，最多 180px；等待 DOM 更新后才能读到 scrollHeight。
watch(draft, async () => {
  await nextTick()
  const element = composerInput.value
  if (element) {
    element.style.height = 'auto'
    element.style.height = Math.min(element.scrollHeight, 180) + 'px'
  }
})

watch(activeId, () => scrollToBottom())

/** 创建空会话并返回它，供“新对话”和首次发送共用。 */
function makeConversation() {
  const conversation = {
    id: crypto.randomUUID(),
    title: '新对话',
    messages: [],
  }
  conversations.value.unshift(conversation)
  activeId.value = conversation.id
  return conversation
}

/** 空会话不重复创建，避免用户连续点击时产生一串空白记录。 */
function newChat() {
  if (!activeConversation.value || activeConversation.value.messages.length > 0) {
    makeConversation()
  }
  draft.value = ''
  mobileSidebarOpen.value = false
  nextTick(() => composerInput.value?.focus())
}

function selectChat(id) {
  activeId.value = id
  mobileSidebarOpen.value = false
}

function setSuggestion(text) {
  draft.value = text
  nextTick(() => composerInput.value?.focus())
}

async function scrollToBottom() {
  await nextTick()
  const element = messagesViewport.value
  if (element) element.scrollTop = element.scrollHeight
}

/** 读取后端就绪状态；失败时清空旧状态，避免继续向断开的服务发送问题。 */
async function refreshStatus() {
  statusLoading.value = true
  try {
    status.value = await getStatus()
    statusError.value = ''
  } catch (error) {
    status.value = null
    statusError.value = error.message
  } finally {
    statusLoading.value = false
  }
}

/**
 * 发送一次单轮问答。先放入用户消息和等待占位，再请求 POST /api/v1/ask。
 * 输入是 draft 中的当前问题；成功时把答案、引用和耗时写回占位，失败时显示错误。
 */
async function sendQuestion() {
  const question = draft.value.trim()
  if (!question || sending.value || !isReady.value || question.length > 2000) return

  const conversation = activeConversation.value ?? makeConversation()
  if (conversation.messages.length === 0) {
    conversation.title = question.length > 29 ? question.slice(0, 29) + '…' : question
  }

  // 后端响应可能较慢，先渲染消息与等待状态；sending 同时防止重复提交。
  conversation.messages.push({ id: crypto.randomUUID(), role: 'user', text: question })
  conversation.messages.push({ id: crypto.randomUUID(), role: 'assistant', text: '', pending: true })
  const reply = conversation.messages[conversation.messages.length - 1]
  draft.value = ''
  sending.value = true
  await scrollToBottom()

  try {
    const result = await askQuestion(question)
    reply.text = result.answer || '本轮没有返回回答。'
    reply.answerable = result.answerable
    reply.reason = result.reason
    reply.citations = result.citations || []
    reply.timing = result.timing
    reply.thresholdCalibrated = result.threshold_calibrated
  } catch (error) {
    reply.error = error.message
    if (error.message.includes('连接后端')) refreshStatus()
  } finally {
    reply.pending = false
    sending.value = false
    scrollToBottom()
    nextTick(() => composerInput.value?.focus())
  }
}

// 中文输入法正在组词时不能把 Enter 当作发送键。
function handleComposerKeydown(event) {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault()
    sendQuestion()
  }
}

// 桌面端控制侧栏宽度；窄屏端改为遮罩抽屉，两种状态独立保存。
function toggleSidebar() {
  if (window.innerWidth <= 760) mobileSidebarOpen.value = !mobileSidebarOpen.value
  else sidebarCollapsed.value = !sidebarCollapsed.value
}

onMounted(() => {
  makeConversation()
  refreshStatus()
  // 状态查询只读，不触发云端 LLM；定时检查便于后端重启后自动恢复输入。
  statusTimer = window.setInterval(refreshStatus, 30000)
})

onUnmounted(() => window.clearInterval(statusTimer))

const suggestions = [
  { icon: BookOpen, title: '梳理核心流程', prompt: 'RAG有哪些标准化流程？' },
  { icon: Search, title: '理解检索策略', prompt: '混合检索和重排分别起什么作用？' },
  { icon: FileText, title: '从资料中总结', prompt: '索引、检索和生成三个阶段如何配合？' },
  { icon: Sparkles, title: '深入一个问题', prompt: '如何评价RAG系统的检索效果？' },
]
</script>

<template>
  <div class="app-shell" :class="{ 'sidebar-collapsed': sidebarCollapsed }">
    <!-- 侧栏只保存本页的会话标题和消息；列表搜索不会请求后端。 -->
    <div v-if="mobileSidebarOpen" class="mobile-overlay" @click="mobileSidebarOpen = false"></div>

    <aside class="sidebar" :class="{ 'mobile-open': mobileSidebarOpen }">
      <div class="sidebar-top">
        <div class="brand">
          <div class="brand-mark"><BookOpen :size="19" :stroke-width="2.2" /></div>
          <span>RAG Agent</span>
        </div>
        <button class="icon-button sidebar-close" type="button" title="收起侧栏" @click="toggleSidebar">
          <PanelLeftClose :size="19" />
        </button>
      </div>

      <div class="sidebar-main">
        <button class="new-chat-button" type="button" @click="newChat">
          <SquarePen :size="18" />
          <span>新对话</span>
          <span class="new-chat-plus">＋</span>
        </button>

        <label class="sidebar-search">
          <Search :size="16" />
          <input v-model="search" type="search" placeholder="搜索本页对话" aria-label="搜索本页对话" />
        </label>

        <div class="sidebar-section-heading">
          <span>本页对话</span>
          <span>{{ conversations.length }}</span>
        </div>
        <nav class="conversation-list" aria-label="本页对话">
          <button v-for="conversation in filteredConversations" :key="conversation.id"
                  class="conversation-item" :class="{ active: conversation.id === activeId }"
                  type="button" @click="selectChat(conversation.id)">
            <BookOpen :size="16" />
            <span>{{ conversation.title }}</span>
          </button>
          <p v-if="filteredConversations.length === 0" class="no-search-results">没有匹配的对话</p>
        </nav>
      </div>

      <div class="sidebar-bottom">
        <div class="library-card">
          <div class="library-icon"><Database :size="17" /></div>
          <div class="library-copy">
            <strong>本地知识库</strong>
            <span v-if="isReady">{{ status.vector_count }} 个知识片段 · {{ status.reranker_device?.toUpperCase() || 'CPU' }} 重排</span>
            <span v-else>{{ statusLoading ? '正在检查连接…' : '等待后端连接' }}</span>
          </div>
          <span class="library-dot" :class="{ offline: !isReady }"></span>
        </div>
        <div class="sidebar-footer-row">
          <button class="sidebar-footer-button" type="button" @click="darkMode = !darkMode">
            <Sun v-if="darkMode" :size="17" />
            <Moon v-else :size="17" />
            <span>{{ darkMode ? '浅色外观' : '深色外观' }}</span>
          </button>
        </div>
        <p class="sidebar-caption">对话仅在当前页面保留，刷新后会清空。</p>
      </div>
    </aside>

    <main class="main-panel">
      <header class="topbar">
        <div class="topbar-left">
          <button class="icon-button menu-button" type="button" title="切换侧栏" @click="toggleSidebar">
            <PanelLeftOpen :size="21" />
          </button>
          <div class="topbar-title">
            <strong>RAG Agent</strong>
            <span>知识库问答</span>
          </div>
        </div>

        <div class="topbar-right">
          <span class="connection-status" :class="{ offline: !isReady }">
            <i></i>{{ isReady ? '知识库已就绪' : '未连接' }}
          </span>
        </div>
      </header>

      <div ref="messagesViewport" class="messages-viewport">
        <div v-if="activeMessages.length === 0" class="welcome">
          <div class="welcome-mark"><BookOpen :size="29" :stroke-width="1.8" /></div>
          <p class="welcome-eyebrow">YOUR KNOWLEDGE, ONE QUESTION AWAY</p>
          <h1>今天想了解什么？</h1>
          <p class="welcome-subtitle">从你的资料中检索依据，得到有出处的回答。</p>

          <div class="suggestions">
            <button v-for="item in suggestions" :key="item.title" class="suggestion-card" type="button"
                    @click="setSuggestion(item.prompt)">
              <component :is="item.icon" :size="18" />
              <span>{{ item.title }}</span>
              <small>{{ item.prompt }}</small>
            </button>
          </div>
        </div>

        <div v-else class="message-list">
          <ChatMessage v-for="message in activeMessages" :key="message.id" :message="message" />
        </div>
      </div>

      <!-- 后端未就绪或已有请求在执行时禁用发送；问题长度与 API 的 2000 字限制一致。 -->
      <div class="composer-area">
        <div v-if="statusError" class="connection-warning">
          <span>{{ statusError }}</span>
          <button type="button" @click="refreshStatus">重新连接</button>
        </div>

        <div class="composer" :class="{ disabled: !isReady }">
          <textarea ref="composerInput" v-model="draft" rows="1" maxlength="2000"
                    :placeholder="isReady ? '询问你的知识库…' : '等待知识库连接…'"
                    :disabled="!isReady || sending"
                    aria-label="输入问题"
                    @keydown="handleComposerKeydown"></textarea>
          <div class="composer-bottom">
            <div class="composer-context">
              <BookOpen :size="16" />
              <span>{{ isReady ? '基于当前知识库' : '本地知识库未连接' }}</span>
            </div>
            <div class="composer-right">
              <span v-if="draft.length > 1800" class="char-count">{{ draft.length }}/2000</span>
              <button class="send-button" type="button" title="发送问题"
                      :disabled="!draft.trim() || sending || !isReady"
                      @click="sendQuestion">
                <ArrowUp :size="18" :stroke-width="2.4" />
              </button>
            </div>
          </div>
        </div>
        <p class="composer-hint">每条问题独立检索，不会读取上方的对话历史 · Enter 发送，Shift + Enter 换行</p>
      </div>
    </main>
  </div>
</template>
