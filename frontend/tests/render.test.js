// 引用公式预处理的离线测试：使用 Node 内置测试工具，不启动服务或访问知识库。
import test from 'node:test'
import assert from 'node:assert/strict'
import { marked } from 'marked'
import { prepareCitationMath, splitCitationText } from '../src/lib/render.js'

test('PDF 引用里没有 $ 的独立 LaTeX 公式会进入 KaTeX 块级渲染', () => {
  const source = [
    '准确率用于衡量模型整体预测正确性。',
    '',
    'A c c u r a c y = \\frac {T P + T N}{T P + T N + F P + F N}\\tag{1}',
    '',
    '其中 TP 表示真正例。',
  ].join('\n')
  const html = marked.parse(prepareCitationMath(source), { breaks: true, gfm: true })
  assert.match(html, /katex-display/)
  assert.match(html, /<mfrac>/)
  assert.match(html, /其中 TP 表示真正例/)
  // KaTeX 的无障碍 annotation 会保留原始 LaTeX，页面可见层仍是排版后的分式。
  assert.doesNotMatch(html, /<p>[^<]*\\frac/)
})

test('普通说明和已带数学分隔符的文本保持原样', () => {
  const source = '其中，\\alpha 为平衡参数。\n路径 C:\\notes\\file.txt\n$F_1=2PR/(P+R)$'
  assert.equal(prepareCitationMath(source), source)
})

test('只拆分本轮存在的引用编号，连续编号仍各自保留', () => {
  const text = '新的 RAG 范式[3]。增强策略[1][3]，而[99]不是本轮来源。'
  assert.deepEqual(splitCitationText(text, [1, 2, 3]), [
    { text: '新的 RAG 范式' }, { text: '[3]', rank: 3 },
    { text: '。增强策略' }, { text: '[1]', rank: 1 },
    { text: '[3]', rank: 3 }, { text: '，而[99]不是本轮来源。' },
  ])
})
