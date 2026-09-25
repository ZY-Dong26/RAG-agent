"""
mineru_adapter.py —— 格式适配层：MinerU 结构化产物 → RAG 文档列表
职责：安全解压、检查页面覆盖、提取可入库文字，并保留来源和物理页码。
输入：云端 ZIP，以及解压后的 content_list 与 layout/middle JSON。
输出：documents（文字与元数据）和 stats（页数、空文本页、块数量）。
关键设计：
    1. 优先使用 layout 的 preproc_blocks 提取页内内容；缺少该字段时兼容 content_list。
    2. 不只读 Markdown，因为普通 Markdown 无法可靠保留原 PDF 页码。
    3. 未知格式直接报错，避免猜测页码而产生错误引用。
    4. 图片只取图注等已有文字，本模块不执行图片语义理解。
"""
import json
import re
import stat
import zipfile
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath

# 更改解析字段映射或页码规则时应更新此版本，让旧的转换结果不会被当成新格式缓存。
ADAPTER_VERSION = 'cloud-page-layout-v1.3'


class ValidationError(ValueError):
    """
    解析产物未满足安全或数据完整性要求时抛出的异常。
    """
    pass


def safe_extract(archive, destination):
    """
    检查 ZIP 的所有成员路径后再解压，防止文件被写到目标目录之外。
    :param archive: 本地原始 ZIP 路径
    :param destination: 本次解压的独立目录
    不仅检查 ../，还处理绝对路径、符号链接、Windows 保留名以及大小写冲突。
    数量和解压体积上限用于控制不正常压缩包的资源消耗。
    """
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        # 1. 先读取成员清单和声明的解压大小，检查资源上限，再考虑写文件。
        members = z.infolist()
        if len(members) > 10000 or sum(m.file_size for m in members) > 2 * 1024**3:
            raise ValidationError('解析压缩包超过文件数量或解压体积限制')
        checked, names = [], set()
        for member in members:
            name = member.filename
            # ZIP 使用正斜杠分隔路径，按 POSIX 规则拆分；反斜杠另行拒绝，避免 Windows 路径歧义。
            parts = PurePosixPath(name).parts
            # ZIP 的高位属性可携带 Unix 文件类型；识别符号链接，防止链接把写入引到目录外。
            mode = member.external_attr >> 16
            if (not parts or '\\' in name or ':' in name or name.startswith('/')
                    or '..' in parts or stat.S_ISLNK(mode)
                    or any(re.match(r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)', p, re.I)
                           or p.endswith((' ', '.')) for p in parts)):
                raise ValidationError('压缩包包含不安全路径')
            # 解析最终绝对路径，确认仍位于目标目录中；casefold 同时防止大小写文件名碰撞。
            target = (destination / name).resolve()
            if not target.is_relative_to(destination) or str(target).casefold() in names:
                raise ValidationError('压缩包包含越界或重复路径')
            names.add(str(target).casefold())
            checked.append((member, target))
        # 2. 先校验全部成员路径，再开始写入任一成员文件。
        for member, target in checked:
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as source, target.open('wb') as output:
                    # 分块复制成员内容；:= 同时取得数据并判断是否读到末尾。
                    while data := source.read(1024 * 1024):
                        output.write(data)


class TableText(HTMLParser):
    """
    把 HTML 表格转成适合文本 Embedding 的简单行文本。
    <tr> 变成换行，单元格之间用竖线分隔。它保留基本行内容，
    但不会完整还原合并单元格或复杂表头语义，原始 HTML 仍在云端产物中。
    """
    def __init__(self):
        """
        初始化 HTMLParser，并准备按解析顺序收集文本片段。
        """
        super().__init__()
        self.parts = []
    def handle_starttag(self, tag, attrs):
        """
        HTMLParser 遇到开始标签时自动调用，用标签恢复基本行和单元格边界。
        """
        if tag == 'tr': self.parts.append('\n')
        elif tag in ('td', 'th'): self.parts.append(' | ')
        elif tag == 'br': self.parts.append(' ')
    def handle_data(self, data):
        """
        遇到实际文本时追加到列表，最后通过 join 还原成一个字符串。
        """
        self.parts.append(data)


def text_value(value):
    """
    统一字符串、字符串列表和空字段；陌生嵌套结构直接报错，避免把字典转成无意义文字入库。
    """
    if isinstance(value, str): return value
    if isinstance(value, list) and all(isinstance(v, str) for v in value): return '\n'.join(value)
    if value is None: return ''
    raise ValidationError('不支持的内容字段格式，请检查云端输出版本')


def page_local_blocks(pages):
    """优先还原逐页原始块，避免云端跨页合并使段落或表格引用到前一页。

    preproc_blocks 是合并前的页内块；para_blocks/content_list 可能已合并跨页文字。
    表格只读取本页的 HTML，图片只读取已有文字说明，不凭空生成图像描述。
    """
    present = ['preproc_blocks' in page for page in pages]
    if not any(present):
        return None  # 兼容仅包含页清单的旧格式，保留 content_list 的来源说明。
    if not all(present):
        raise ValidationError('逐页内容字段不完整，拒绝混用可能跨页合并的内容')

    def spans_text(block, html=False):
        """按行拼接 span，并递归读取子块；html=True 时取表格 HTML，其他情况取文字内容。"""
        parts = []
        for line in block.get('lines', []):
            line_parts = []
            for span in line.get('spans', []):
                value = span.get('html') if html else span.get('content')
                if value:
                    line_parts.append(text_value(value))
            if line_parts:
                parts.append(''.join(line_parts))
        for child in block.get('blocks', []):
            value = spans_text(child, html=html)
            if value:
                parts.append(value)
        return '\n'.join(parts)

    def page_height(page):
        """尽量取得页面高度供后处理判断 bbox；字段不稳定时返回 None，由后处理回退到块顺序。"""
        for key in ('page_height', 'height'):
            if isinstance(page.get(key), (int, float)) and page[key] > 0:
                return page[key]
        size = page.get('page_size')
        if isinstance(size, (list, tuple)) and len(size) >= 2 and isinstance(size[1], (int, float)):
            return size[1]
        return None

    normalized = []
    for page in pages:
        # 正文以原始页内块为准；从丢弃块补回脚注以及页眉页脚候选。后者不在适配器删除，
        # 而是交给独立后处理基于全文重复证据标记，避免两页短文档被无条件误删。
        raw = page['preproc_blocks']
        if not isinstance(raw, list):
            raise ValidationError('逐页内容块必须为列表')
        raw = raw + [b for b in page.get('discarded_blocks', [])
                     if b.get('type') in ('page_footnote', 'header', 'footer', 'page_number')]
        for block in raw:
            kind = block.get('type', 'unknown')
            item = {'page_idx': page['page_idx'], 'type': kind, 'bbox': block.get('bbox'),
                    'page_height': page_height(page), 'mineru_block': block}
            if kind in ('text', 'title', 'ref_text', 'page_footnote', 'list', 'code',
                        'interline_equation', 'equation', 'header', 'footer', 'page_number'):
                item['type'] = 'equation' if kind == 'interline_equation' else kind
                item['text'] = spans_text(block)
                if kind == 'title':
                    item['text_level'] = block.get('text_level')
            elif kind == 'table':
                item['table_body'] = spans_text(block, html=True)
                item['table_caption'] = [spans_text(b) for b in block.get('blocks', []) if b.get('type') == 'table_caption']
                item['table_footnote'] = [spans_text(b) for b in block.get('blocks', []) if b.get('type') == 'table_footnote']
            elif kind in ('image', 'chart'):
                item['content'] = spans_text(block)
            else:
                # 未知结构不静默丢弃：保留原始块并提取已有文字。后处理记录类型告警；
                # 没有文字的未知结构仍留在文档产物中，但不会生成检索 chunk。
                item['text'] = spans_text(block)
            normalized.append(item)
    return normalized


def adapt_result(folder, source, expected_pages):
    """
    把云端结构化内容转换成统一文档记录，并核对全文是否处理完整。
    :param folder: ZIP 已安全解压的目录
    :param source: 原 PDF 文件名，用于后续引用
    :param expected_pages: 从原 PDF 本地读取的物理页数
    :return: (docs, stats)，docs 每项包含 text 和 metadata
    物理页码指 PDF 文件的第几页，不一定等于文档印刷的页码。
    云端 page_idx 从 0 开始，写给用户看的 metadata.page 时统一加 1。
    """
    folder = Path(folder)
    # 1. 找到唯一内容列表；拒绝多份候选，防止混合不同文档或不同格式的内容。
    candidates = list(folder.rglob('*_content_list.json')) + list(folder.rglob('content_list.json'))
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1:
        raise ValidationError('需要且只能有一份 content_list JSON，当前云端输出不匹配')
    blocks = json.loads(candidates[0].read_text(encoding='utf-8'))
    if not isinstance(blocks, list) or not all(isinstance(b, dict) for b in blocks):
        raise ValidationError('不支持的 content_list 格式')
    # 2. 另外读取页面记录。内容列表没有文字的页可能不出现，不能用文字块数量代替总页数。
    layouts = list(folder.rglob('layout.json')) + list(folder.rglob('*_middle.json')) + list(folder.rglob('middle.json'))
    layouts = list(dict.fromkeys(layouts))
    if len(layouts) != 1:
        raise ValidationError('缺少唯一的 layout/middle JSON，无法验证完整页数')
    layout = json.loads(layouts[0].read_text(encoding='utf-8'))
    pages = layout.get('pdf_info') if isinstance(layout, dict) else None
    if not isinstance(pages, list):
        raise ValidationError('layout 中没有 pdf_info 页记录，无法验证完整文档')
    # 要求恰好覆盖 0 到 expected_pages-1：既检查数量，也检查重复、缺页和越界。
    page_ids = [p.get('page_idx') for p in pages if isinstance(p, dict)]
    if (len(page_ids) != expected_pages or any(type(p) is not int for p in page_ids)
            or set(page_ids) != set(range(expected_pages))):
        raise ValidationError(f'解析页覆盖不完整：预期 {expected_pages} 页，收到 {len(page_ids)} 条页记录')
    # 3. docs 收集可入库文字，text_pages 统计实际有文字的页，types 统计云端返回的块类型。
    local_blocks = page_local_blocks(pages)
    if local_blocks is not None:
        blocks = local_blocks
    docs, text_pages, types = [], set(), {}
    for index, block in enumerate(blocks):
        page = block.get('page_idx')
        if type(page) is not int or page not in set(page_ids):
            raise ValidationError('内容块页码缺失或越界')
        kind = block.get('type', 'unknown')
        types[kind] = types.get(kind, 0) + 1
        # 正文、公式、代码和列表统一为字符串；同一页可以生成多个独立文档块。
        # 云端 v1 还会返回页脚注释和参考文献正文；它们含有效文字，不能当作未知类型丢弃。
        # page_footnote 是脚注内容，与仅用于排版的 footer（页脚）不同。
        if kind in ('text', 'title', 'equation', 'code', 'list', 'page_footnote', 'ref_text',
                    'header', 'footer', 'page_number'):
            value = text_value(block.get('text') or block.get('list_items'))
        # 表格由标题、主体和脚注组成。HTML 主体先转行文本，避免把标签本身当成知识正文。
        elif kind == 'table':
            body = text_value(block.get('table_body'))
            if '<table' in body.lower():
                parser = TableText(); parser.feed(body); body = ''.join(parser.parts)
            value = '\n'.join(filter(None, [text_value(block.get('table_caption')), body,
                                           text_value(block.get('table_footnote'))]))
        # 当前 Embedding 只接受文本，因此图片/图表只索引已有图注、脚注或文字化内容。
        elif kind in ('image', 'chart'):
            value = '\n'.join(filter(None, [text_value(block.get(kind + '_caption')),
                                           text_value(block.get(kind + '_footnote')),
                                           text_value(block.get('content'))]))
        else:
            # 透传未知类型及原始字段，不因云端新增类型丢失数据。V1 后处理会记录告警；
            # 空未知块仍保留在结构化产物中，但不会产生检索文本。
            raw_value = block.get('text', block.get('content', ''))
            value = raw_value if isinstance(raw_value, str) else ''
        if value.strip():
            text_pages.add(page)
        # 4. 统一输出协议：text 给切块器，metadata 沿切块→索引→检索传递，最终用于引用。
        docs.append({'text': value.strip(), 'metadata': {
            'source': source, 'page': page + 1, 'block_id': f'p{page + 1}_b{index}',
            'block_type': kind, 'bbox': block.get('bbox'), 'page_height': block.get('page_height'),
            'title_level': block.get('text_level'), 'mineru_block': block.get('mineru_block', block),
            'parser': 'mineru_cloud', 'adapter_version': ADAPTER_VERSION}})
    # 5. 覆盖完整不代表每页都有文字；明确记录空文本页，供人工确认是空白页还是识别遗漏。
    # body_blocks/table_blocks 按云端块类型计数，characters 按实际保留文字计数。
    stats = {'expected_pages': expected_pages, 'parsed_pages': len(page_ids),
             'text_pages': len(text_pages), 'empty_text_pages': [p + 1 for p in page_ids if p not in text_pages],
             'body_blocks': types.get('text', 0), 'table_blocks': types.get('table', 0),
             'block_types': types, 'characters': sum(len(d['text']) for d in docs),
             'page_source': 'preproc_blocks' if local_blocks is not None else 'content_list'}
    if not docs:
        raise ValidationError('所有页面均无可入库文字，拒绝发布空文档')
    return docs, stats
