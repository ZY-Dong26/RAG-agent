<script setup>
// ChatMessage.vue —— 展示一条用户消息或助手消息。
// 助手消息的 pending/error/成功状态互斥；成功时可展开最终引用与分项耗时。
import { computed, ref } from 'vue'
import { marked } from 'marked'
import markedKatex from 'marked-katex-extension'
import DOMPurify from 'dompurify'
import {
  AlertCircle, BookOpen, Check, ChevronDown, Clock3, Copy, FileText,
} from '@lucide/vue'

// nonStandard 允许 $...$ 紧贴中文；公式先转 HTML，再与普通 Markdown 一起清理。
marked.use(markedKatex({ throwOnError: false, nonStandard: true }))

// message 由 App.vue 创建。成功响应包含 text、citations、timing 和 thresholdCalibrated。
const props = defineProps({
  message: { type: Object, required: true },
})

const sourcesOpen = ref(false)
const timingOpen = ref(false)
const copied = ref(false)

// v-html 只接收清理后的内容，不能把云端模型生成的原始 HTML 直接插入页面。
const renderedAnswer = computed(() => {
  const html = marked.parse(props.message.text || '', { breaks: true, gfm: true })
  return DOMPurify.sanitize(html)
})

function seconds(value) {
  return typeof value === 'number' ? value.toFixed(2) + ' 秒' : '—'
}

function score(value) {
  return typeof value === 'number' ? value.toFixed(3) : '—'
}

// 复制原始回答文本，保留 Markdown 和公式源码，方便粘贴到笔记或编辑器。
async function copyAnswer() {
  try {
    await navigator.clipboard.writeText(props.message.text || '')
    copied.value = true
    window.setTimeout(() => { copied.value = false }, 1800)
  } catch {
    copied.value = false
  }
}
</script>

<template>
  <div v-if="message.role === 'user'" class="message user-message">
    <div class="user-bubble">{{ message.text }}</div>
  </div>

  <article v-else class="message assistant-message">
    <div class="assistant-mark" aria-hidden="true"><BookOpen :size="17" :stroke-width="2.2" /></div>
    <div class="assistant-body">
      <div class="assistant-name">RAG Agent</div>

      <!-- 等待、错误、成功三种视图只显示一种；错误不展示空白回答。 -->
      <div v-if="message.pending" class="pending-state" role="status">
        <span class="thinking-dots"><i></i><i></i><i></i></span>
        正在检索资料并生成回答…
      </div>

      <div v-else-if="message.error" class="message-error" role="alert">
        <AlertCircle :size="17" />
        <span>{{ message.error }}</span>
      </div>

      <template v-else>
        <div class="answer-content" v-html="renderedAnswer"></div>

        <p v-if="message.citations?.length && !message.thresholdCalibrated" class="calibration-note">
          重排拒答阈值尚未校准，本轮没有执行分数硬拒答。
        </p>

        <div class="answer-actions">
          <button class="icon-text-button" type="button" @click="copyAnswer" :aria-label="copied ? '已复制' : '复制回答'">
            <Check v-if="copied" :size="15" />
            <Copy v-else :size="15" />
            <span>{{ copied ? '已复制' : '复制' }}</span>
          </button>
          <button v-if="message.citations?.length" class="icon-text-button" type="button"
                  :aria-expanded="sourcesOpen" @click="sourcesOpen = !sourcesOpen">
            <FileText :size="15" />
            <span>引用来源 {{ message.citations.length }}</span>
            <ChevronDown :size="13" :class="{ rotated: sourcesOpen }" />
          </button>
          <button v-if="message.timing" class="icon-text-button" type="button"
                  :aria-expanded="timingOpen" @click="timingOpen = !timingOpen">
            <Clock3 :size="15" />
            <span>{{ seconds(message.timing.total_seconds) }}</span>
            <ChevronDown :size="13" :class="{ rotated: timingOpen }" />
          </button>
        </div>

        <!-- 来源和耗时取自同一次 API 响应，不在前端重新检索或计算分数。 -->
        <div v-if="sourcesOpen && message.citations?.length" class="sources-panel">
          <div class="panel-label">本轮使用的资料</div>
          <details v-for="(citation, index) in message.citations" :key="index" class="source-item">
            <summary>
              <span class="source-rank">{{ citation.rank ?? index + 1 }}</span>
              <span class="source-name">{{ citation.source }}</span>
              <span class="source-page">第 {{ citation.page ?? '—' }} 页</span>
              <ChevronDown :size="14" class="source-chevron" />
            </summary>
            <div class="source-detail">
              <p>{{ citation.text }}</p>
              <span v-if="citation.rerank_score != null">重排分数 {{ score(citation.rerank_score) }}</span>
            </div>
          </details>
        </div>

        <div v-if="timingOpen && message.timing" class="timing-panel">
          <div><span>召回与融合</span><strong>{{ seconds(message.timing.recall_seconds) }}</strong></div>
          <div><span>重排</span><strong>{{ seconds(message.timing.rerank_seconds) }}</strong></div>
          <div><span>检索合计</span><strong>{{ seconds(message.timing.retrieval_seconds) }}</strong></div>
          <div><span>生成</span><strong>{{ seconds(message.timing.generation_seconds) }}</strong></div>
          <div class="timing-total"><span>本轮总计</span><strong>{{ seconds(message.timing.total_seconds) }}</strong></div>
        </div>
      </template>
    </div>
  </article>
</template>
