// SSE 数据可能在任意字节处拆包；此测试只验证前端分帧，不需要后端或云端模型。
import test from 'node:test'
import assert from 'node:assert/strict'
import { createSseParser } from '../src/lib/api.js'

test('分块的状态、中文增量和完成事件按顺序解析', () => {
  const events = []
  const parser = createSseParser((name, data) => events.push([name, data]))
  parser.push('event: status\r')
  parser.push('\ndata: {"stage":"retrieving"}\r\n\r\nevent: delta\ndata: {"text":"你')
  parser.push('好\\n"}\n\nevent: done\ndata: {"conversation_id":"c1"}\n\n')
  assert.deepEqual(events, [
    ['status', { stage: 'retrieving' }],
    ['delta', { text: '你好\n' }],
    ['done', { conversation_id: 'c1' }],
  ])
})
