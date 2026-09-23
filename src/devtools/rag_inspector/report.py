"""
report.py —— 把一轮 RAG 调试轨迹渲染成可直接打开的静态 HTML 诊断报告

职责：
    1. 接收 TraceRecorder 收集的一轮问答轨迹（问题、命中块、发送的 prompt、回答、耗时）。
    2. 把命中块按"已发送/未发送"着色，直观展示字符预算截掉了哪些 chunk。
    3. 补充命中块相邻 segment、最终 prompt、模型配置，帮助定位"为什么没答好"。

设计原因：
    - 纯字符串拼接 HTML，不依赖模板引擎；所有用户文本经过 html.escape，防止问题或回答里的 HTML 标签注入。
    - 输出到 data/outputs/debug/<时间戳>_<问题摘要>_<随机>/，JSON 和 HTML 各一份，方便对比多轮调试。
"""
from __future__ import annotations

import html
import json
import re
import uuid
from datetime import datetime
from pathlib import Path

from .recorder import TraceRecorder


def _escape(value) -> str:
    """所有用户文本统一转义，防止问题或 chunk 里的 HTML 标签注入报告页面。"""
    return html.escape("" if value is None else str(value))


def _request_messages(trace: dict) -> list[dict]:
    """从轨迹中取出实际发给模型的 messages；缺失时返回空列表。"""
    messages = trace.get("request", {}).get("messages", [])
    return messages if isinstance(messages, list) else []


def _sent_text(trace: dict) -> str:
    """拼接 user 消息全文，用于判断某个 chunk 是否因字符预算被截掉。"""
    return "\n".join(
        str(message.get("content", ""))
        for message in _request_messages(trace)
        if message.get("role") == "user"
    )


def _hit_rows(trace: dict) -> str:
    sent_text = _sent_text(trace)
    rows = []
    for fallback_rank, hit in enumerate(trace.get("hits", []), 1):
        metadata = hit.get("metadata", {})
        rank = metadata.get("rank", fallback_rank)
        text = str(hit.get("text", "")).strip()
        marker = f"[{rank}] {text}"
        sent = marker in sent_text
        status = "已发送" if sent else "未发送"
        status_class = "sent" if sent else "dropped"
        score = metadata.get("score")
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "—"
        rows.append(
            f"""
            <details class="chunk {status_class}" {'open' if fallback_rank == 1 else ''}>
              <summary>
                <span class="rank">#{_escape(rank)}</span>
                <span class="status">{status}</span>
                <span>{_escape(metadata.get('source', '未知来源'))} · 第{_escape(metadata.get('page', '?'))}页</span>
                <span class="meta">相似度 {score_text} · {len(text)} 字符 · {_escape(metadata.get('block_type', '未知类型'))}</span>
              </summary>
              <pre>{_escape(text)}</pre>
              <div class="chunk-meta">chunk_id: {_escape(metadata.get('chunk_id', '—'))}　segment: {_escape(metadata.get('segment_index', '—'))}</div>
            </details>
            """
        )
    return "".join(rows) or '<p class="empty">没有召回文本块。</p>'


def _message_blocks(trace: dict) -> str:
    blocks = []
    for message in _request_messages(trace):
        role = message.get("role", "unknown")
        blocks.append(
            f"<section class=\"message\"><h3>{_escape(role)}</h3>"
            f"<pre>{_escape(message.get('content', ''))}</pre></section>"
        )
    return "".join(blocks) or '<p class="empty">尚未发出模型请求。</p>'


def _neighbor_blocks(trace: dict) -> str:
    sections = []
    for rank, neighbors in trace.get("neighbors", {}).items():
        cards = []
        for chunk in neighbors:
            metadata = chunk.get("metadata", {})
            cards.append(
                f"<details class=\"neighbor\"><summary>segment {_escape(metadata.get('segment_index', '?'))} · "
                f"第{_escape(metadata.get('page', '?'))}页 · {_escape(metadata.get('block_type', '未知类型'))}</summary>"
                f"<pre>{_escape(chunk.get('text', ''))}</pre></details>"
            )
        sections.append(f"<section><h3>命中 #{_escape(rank)} 的相邻块</h3>{''.join(cards)}</section>")
    return "".join(sections) or '<p class="empty">当前命中没有可显示的相邻块。</p>'


def _safe_slug(question: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", question).strip("-")[:24]
    return slug or "query"


def _render_html(trace: dict) -> str:
    timing = trace.get("timing", {})
    retrieval = timing.get("retrieval_seconds")
    generation = timing.get("generation_seconds")
    request = trace.get("request", {})
    messages = _request_messages(trace)
    prompt_chars = sum(len(str(message.get("content", ""))) for message in messages)
    sent_text = _sent_text(trace)
    sent_count = sum(
        f"[{hit.get('metadata', {}).get('rank', index)}] {str(hit.get('text', '')).strip()}" in sent_text
        for index, hit in enumerate(trace.get("hits", []), 1)
    )
    error = trace.get("error")
    answer = trace.get("answer") or ""
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAG 问答诊断</title>
<style>
:root{{--bg:#f5f7fb;--panel:#fff;--text:#172033;--muted:#667085;--line:#e4e7ec;--accent:#315efb;--good:#067647;--good-bg:#ecfdf3;--warn:#b54708;--warn-bg:#fffaeb;--code:#101828}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.6 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:28px 20px 60px}} h1{{font-size:24px;margin:0 0 6px}} h2{{font-size:18px;margin:0 0 14px}} h3{{font-size:14px;margin:14px 0 8px}}
.question{{color:var(--muted);margin-bottom:20px}} .stats{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:20px}}
.stat,.panel{{background:var(--panel);border:1px solid var(--line);border-radius:12px}} .stat{{padding:14px}} .stat b{{display:block;font-size:20px}} .stat span,.meta,.chunk-meta,.empty{{color:var(--muted)}}
.tabs{{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px}} .tabs button{{border:1px solid var(--line);background:var(--panel);padding:8px 13px;border-radius:8px;color:var(--text);cursor:pointer}}
.tabs button.active{{background:var(--accent);border-color:var(--accent);color:#fff}} .panel{{display:none;padding:18px}} .panel.active{{display:block}}
details{{border-top:1px solid var(--line)}} details:first-child{{border-top:0}} summary{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:12px 4px;cursor:pointer}}
.rank{{font-weight:700}} .status{{padding:2px 8px;border-radius:999px}} .sent .status{{background:var(--good-bg);color:var(--good)}} .dropped .status{{background:var(--warn-bg);color:var(--warn)}}
pre{{white-space:pre-wrap;word-break:break-word;background:#f8fafc;color:var(--code);padding:14px;border-radius:8px;overflow:auto;margin:0 0 12px;font-family:ui-monospace,Consolas,monospace}}
.chunk-meta{{padding:0 4px 12px;font-size:12px}} .answer{{font-size:15px;white-space:pre-wrap}} .error{{color:#b42318}} .request-meta{{display:flex;gap:18px;flex-wrap:wrap;margin-bottom:14px;color:var(--muted)}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0f172a;--panel:#172033;--text:#eef2f7;--muted:#a8b2c1;--line:#344054;--code:#e6edf7;--good-bg:#073b2a;--good:#75e0a7;--warn-bg:#4e2d10;--warn:#fec84b}} pre{{background:#101828}}}}
@media (max-width:720px){{.stats{{grid-template-columns:repeat(2,minmax(0,1fr))}} summary .meta{{width:100%}}}}
</style>
</head>
<body><main>
<h1>RAG 问答诊断</h1>
<div class="question">{_escape(trace.get('question'))}</div>
<div class="stats">
  <div class="stat"><span>召回候选</span><b>{len(trace.get('hits', []))}</b><span>实际发送 {sent_count} 条</span></div>
  <div class="stat"><span>检索耗时</span><b>{retrieval:.2f}s</b><span>问题向量化与 FAISS</span></div>
  <div class="stat"><span>生成耗时</span><b>{generation:.2f}s</b><span>云端模型请求</span></div>
  <div class="stat"><span>Prompt 字符</span><b>{prompt_chars}</b><span>system + user</span></div>
</div>
<nav class="tabs" aria-label="诊断视图">
  <button class="active" data-tab="hits">检索文本块</button>
  <button data-tab="prompt">最终 Prompt</button>
  <button data-tab="neighbors">相邻文本块</button>
  <button data-tab="answer">回答与配置</button>
</nav>
<section id="hits" class="panel active"><h2>检索文本块</h2>{_hit_rows(trace)}</section>
<section id="prompt" class="panel"><h2>实际发送给模型的消息</h2>{_message_blocks(trace)}</section>
<section id="neighbors" class="panel"><h2>命中块前后各一个 segment</h2>{_neighbor_blocks(trace)}</section>
<section id="answer" class="panel"><h2>模型回答</h2>
  <div class="request-meta"><span>模型：{_escape(request.get('model', '—'))}</span><span>temperature：{_escape(request.get('temperature', '—'))}</span><span>max_tokens：{_escape(request.get('max_tokens', '—'))}</span></div>
  <div class="answer">{_escape(answer)}</div>{f'<p class="error">{_escape(error)}</p>' if error else ''}
</section>
</main>
<script>
document.querySelectorAll('[data-tab]').forEach(button=>button.addEventListener('click',()=>{{
  document.querySelectorAll('[data-tab]').forEach(item=>item.classList.toggle('active',item===button));
  document.querySelectorAll('.panel').forEach(panel=>panel.classList.toggle('active',panel.id===button.dataset.tab));
}}));
</script></body></html>"""


def write_trace_report(recorder: TraceRecorder, output_root: Path) -> tuple[Path, Path]:
    """
    把一轮调试轨迹写入独立运行目录，返回 (JSON 路径, HTML 路径)。

    目录名格式：<时间戳>_<问题摘要>_<随机后缀>，避免多轮调试互相覆盖。
    JSON 保留完整原始轨迹，HTML 是给人看的可视化版本。
    """
    trace = recorder.as_dict()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = Path(output_root) / f"{stamp}_{_safe_slug(recorder.question)}_{uuid.uuid4().hex[:6]}"
    directory.mkdir(parents=True, exist_ok=False)
    json_path = directory / "trace.json"
    html_path = directory / "report.html"
    json_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(_render_html(trace), encoding="utf-8")
    return json_path, html_path
