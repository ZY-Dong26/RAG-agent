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

/** 只拆分本轮真实存在的 [编号]；原始回答文本不变，供复制时保留引用标记。 */
export function splitCitationText(text, ranks) {
  const valid = new Set(ranks.map(Number))
  const parts = []
  const pattern = /\[(\d+)\]/g
  let start = 0
  for (const match of text.matchAll(pattern)) {
    const rank = Number(match[1])
    if (!valid.has(rank)) continue
    if (match.index > start) parts.push({ text: text.slice(start, match.index) })
    parts.push({ text: match[0], rank })
    start = match.index + match[0].length
  }
  if (start < text.length) parts.push({ text: text.slice(start) })
  return parts
}

/** 已清理的 HTML 中仅替换普通文本节点，避开公式、代码和链接里的方括号。 */
function superscriptCitations(html, ranks) {
  if (!ranks.length || typeof document === 'undefined') return html
  const template = document.createElement('template')
  template.innerHTML = html
  const walker = document.createTreeWalker(template.content, NodeFilter.SHOW_TEXT)
  const nodes = []
  while (walker.nextNode()) nodes.push(walker.currentNode)
  for (const node of nodes) {
    if (node.parentElement?.closest('a, code, pre, sup, .katex')) continue
    const parts = splitCitationText(node.textContent || '', ranks)
    if (!parts.some(part => part.rank !== undefined)) continue
    const replacement = document.createDocumentFragment()
    for (const part of parts) {
      if (part.rank === undefined) {
        replacement.append(document.createTextNode(part.text))
      } else {
        const sup = document.createElement('sup')
        sup.className = 'citation-ref'
        sup.setAttribute('aria-label', `引用来源 ${part.rank}`)
        sup.textContent = part.text
        replacement.append(sup)
      }
    }
    node.replaceWith(replacement)
  }
  return template.innerHTML
}

/** Markdown 与公式转 HTML 后清理；回答可按本轮引用编号添加上标。 */
export function renderMarkdown(text, citationRanks = []) {
  const safeHtml = DOMPurify.sanitize(marked.parse(text || '', { breaks: true, gfm: true }))
  return superscriptCitations(safeHtml, citationRanks)
}

/** 引用片段先补独立公式分隔符，再复用回答的安全渲染流程。 */
export function renderCitation(text) {
  return renderMarkdown(prepareCitationMath(text))
}
