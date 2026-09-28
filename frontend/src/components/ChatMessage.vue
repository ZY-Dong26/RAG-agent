<script setup>
// ChatMessage.vue —— 展示一条用户消息或助手消息。
// 助手消息的等待、流式正文、错误和完成状态互斥；完成后才渲染 Markdown 与公式。
import { computed, ref } from 'vue'
import { renderCitation, renderMarkdown } from '../lib/render.js'
import {
  AlertCircle, BookOpen, Check, ChevronDown, Clock3, Copy, FileText,
} from '@lucide/vue'

// message 由 App.vue 创建。成功响应包含最终 answerable、引用、耗时和阈值校准状态。
const props = defineProps({
  message: { type: Object, required: true },
})

const sourcesOpen = ref(false)
const timingOpen = ref(false)
const copied = ref(false)

// 两处 v-html 都来自 render.js，统一完成公式渲染和 HTML 清理。
const renderedAnswer = computed(() => renderMarkdown(
  props.message.text,
  (props.message.citations || []).map((citation, index) => citation.rank ?? index + 1),
))
const renderedCitations = computed(() => (props.message.citations || []).map(item => renderCitation(item.text)))

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

      <!-- 流式增量保持纯文本，避免未闭合的 Markdown/公式反复渲染。 -->
      <div v-if="message.pending" class="pending-state" role="status">
        <span class="thinking-dots"><i></i><i></i><i></i></span>
        {{ message.stage || '正在检索资料并生成回答…' }}
      </div>

      <div v-else-if="message.error" class="message-error" role="alert">
        <AlertCircle :size="17" />
        <span>{{ message.error }}</span>
      </div>

      <div v-else-if="message.streaming" class="answer-content streaming-answer" aria-live="polite">{{ message.text }}</div>

      <template v-else>
        <div class="answer-content" v-html="renderedAnswer"></div>

        <!-- LLM 已拒答时不再显示“未执行分数硬拒答”，避免和最终拒答文案冲突。 -->
        <p v-if="message.answerable && message.citations?.length && !message.thresholdCalibrated" class="calibration-note">
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
              <div class="citation-content" v-html="renderedCitations[index]"></div>
              <span v-if="citation.rerank_score != null" class="source-score">重排分数 {{ score(citation.rerank_score) }}</span>
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
