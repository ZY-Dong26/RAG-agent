// render.js —— 回答与引用片段共用的 Markdown/KaTeX 安全渲染入口。
// 回答通常带 $...$；PDF 解析出的引用公式常独占一行，却没有数学分隔符。
import { marked } from 'marked'
import markedKatex from 'marked-katex-extension'
import DOMPurify from 'dompurify'

// nonStandard 允许行内公式紧贴中文；格式错误时保留可读文本，不让页面崩溃。
marked.use(markedKatex({ throwOnError: false, nonStandard: true }))

// 只识别典型的 LaTeX 命令。普通反斜杠路径、中文说明和已有 $ 分隔符都不自动改写。
const texCommand = /\\(?:frac|dfrac|tfrac|sqrt|sum|prod|int|lim|cdot|times|alpha|beta|gamma|theta|sigma|pi|text|mathrm|mathbb|tag)(?=\s|\{|\\|$)/

/** 把 PDF 提取出的独立公式行补成块级公式；输入和输出都是文本，不执行原文 HTML。 */
export function prepareCitationMath(text) {
  return (text || '').split(/\r?\n/).map(line => {
    const formula = line.trim()
    if (!formula || formula.startsWith('$') || formula.startsWith('\\(') || formula.startsWith('\\[')) return line
    if (/[\u3400-\u9fff]/u.test(formula) || !texCommand.test(formula)) return line
    // 只处理整行像公式的内容，避免把说明句中的单个 TeX 变量当成显示公式。
    if (!/[=<>≤≥]/u.test(formula) && !/\\tag\s*\{/.test(formula) && !formula.startsWith('\\')) return line
    return `$$\n${formula}\n$$`
  }).join('\n')
}

/** Markdown 和公式转 HTML 后统一清理，再交给组件的 v-html。 */
export function renderMarkdown(text) {
  return DOMPurify.sanitize(marked.parse(text || '', { breaks: true, gfm: true }))
}

/** 引用片段先补独立公式分隔符，再复用回答的安全渲染流程。 */
export function renderCitation(text) {
  return renderMarkdown(prepareCitationMath(text))
}
