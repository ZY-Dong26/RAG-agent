# 离线回归测试 —— 用可控的输入和故障检查 RAG 的可靠性
# 学习方式：先读每个 test_ 方法的中文目标，再看准备数据、执行操作、assert 断言三个部分。
# MockTransport/patch 用替身替换云端或大模型，测试不会上传真实 PDF，也不会花费 API 额度。
# TemporaryDirectory 在测试结束后清理产物，真实知识库和密钥不会被测试覆盖。

import io
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from unittest.mock import patch

import httpx
from pypdf import PdfWriter

from rag_agent.ingestion.mineru_settings import MinerUSettings, load_settings
from rag_agent.ingestion.mineru_client import MinerUCloud, MinerUError, parse_files
from rag_agent.ingestion.mineru_adapter import safe_extract, adapt_result, ValidationError
from rag_agent.common.files import read_json, atomic_json, exclusive_lock


def archive(pages=2, page_ids=None, blocks=None):
    """构造模拟云端 ZIP：包含页记录、内容块和 Markdown，用于测试适配过程，不调用真实 MinerU。"""
    output = io.BytesIO()
    page_ids = list(range(pages)) if page_ids is None else page_ids
    blocks = blocks if blocks is not None else [
        {'type': 'text', 'page_idx': 0, 'text': 'First page body'},
        {'type': 'table', 'page_idx': 1, 'table_body': '<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>'}]
    with zipfile.ZipFile(output, 'w') as z:
        z.writestr('doc/layout.json', json.dumps({'pdf_info': [{'page_idx': i} for i in page_ids]}))
        z.writestr('doc/doc_content_list.json', json.dumps(blocks))
        z.writestr('doc/full.md', '# original cloud output')
    return output.getvalue()


class CloudTests(unittest.TestCase):
    """用模拟 HTTP 服务测试任务状态、失败恢复、缓存复用和鉴权隔离。"""
    def setUp(self):
        """每个测试前建立独立临时目录与两页空白 PDF，并用 MockTransport 接管所有 HTTP 请求。"""
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.pdf = self.root / 'report.pdf'
        writer = PdfWriter()
        for _ in range(2): writer.add_blank_page(width=100, height=100)
        writer.write(self.pdf)
        self.calls = []
        self.remote = 'waiting-file'
        self.submitted_data_id = None
        self.zip = archive()
        self.fail_create = False
        self.fail_upload = False
        self.fail_download = False
        self.fail_query = False
        self.settings = MinerUSettings(token='test-secret', poll_interval=.01, max_wait=1, retries=2)
        self.client = httpx.Client(transport=httpx.MockTransport(self.handle), trust_env=False)
        self.parser = MinerUCloud(self.settings, self.root/'cache', self.client, sleep=lambda _: None)

    def tearDown(self):
        """测试后关闭客户端并清理临时目录，避免测试产物污染真实文档库。"""
        self.client.close()
        self.temp.cleanup()

    def handle(self, request):
        """模拟官方 API、上传地址和下载地址：按可控状态返回响应，同时断言 Token 只发给 API。"""
        self.calls.append((request.method, request.url.host, request.headers.get('Authorization')))
        if request.url.host == 'mineru.net':
            self.assertEqual(request.headers.get('Authorization'), 'Bearer test-secret')
            if request.method == 'POST':
                if self.fail_create: raise httpx.ReadTimeout('contains test-secret https://signed/url', request=request)
                body = json.loads(request.content)
                self.submitted_data_id = body['files'][0]['data_id']
                self.assertTrue(body['files'][0]['is_ocr'])
                self.assertNotIn('page_ranges', body['files'][0])
                return httpx.Response(200, json={'code': 0, 'data': {'batch_id': 'batch-1', 'file_urls': ['https://uploads.example/file?signature=abc']}})
            if self.fail_query: raise httpx.ReadTimeout('query timeout', request=request)
            return httpx.Response(200, json={'code': 0, 'data': {'extract_result': [
                {'state': self.remote, 'data_id': self.submitted_data_id, 'full_zip_url': 'https://results.example/result.zip',
                 'err_msg': 'parse error test-secret https://private.example/value'}]}})
        self.assertIsNone(request.headers.get('Authorization'))
        if request.method == 'PUT':
            # 上传 PDF 前，batch_id 必须已经可靠写入任务日志。
            self.assertEqual(read_json(self.journal())['batch_id'], 'batch-1')
            if self.fail_upload: raise httpx.WriteTimeout('upload timeout', request=request)
            self.remote = 'done'
            return httpx.Response(200)
        if self.fail_download: return httpx.Response(503)
        return httpx.Response(200, content=self.zip)

    def journal(self):
        """根据测试 PDF 的缓存键找到任务日志，检查断点信息是否已经保存到磁盘。"""
        return self.root/'cache'/self.parser.cache_key(self.pdf)/'manifest.json'

    def post_count(self):
        """统计创建任务次数，用于证明失败恢复没有重复提交，而不仅仅是最终结果成功。"""
        return sum(method == 'POST' for method, _, _ in self.calls)

    def test_success_cache_and_separate_auth(self):
        """验证一次完整解析后再次调用直接命中缓存，并检查上传/下载请求不携带 API 密钥。"""
        docs, stats = self.parser.parse(self.pdf)
        self.assertEqual(stats['parsed_pages'], 2)
        self.assertEqual(stats['table_blocks'], 1)
        self.assertEqual(docs[1]['metadata']['page'], 2)
        self.assertIn('1', docs[1]['text'])
        calls = len(self.calls)
        self.assertEqual(self.parser.parse(self.pdf)[0], docs)
        self.assertEqual(len(self.calls), calls)
        self.assertNotIn('test-secret', self.journal().read_text())
        self.assertEqual(self.post_count(), 1)

    def test_success_cache_without_key(self):
        """先生成有效缓存，再使用空密钥读取，证明离线缓存复用不依赖网络鉴权。"""
        self.parser.parse(self.pdf)
        offline = MinerUCloud(MinerUSettings(), self.root/'cache', self.client)
        self.assertEqual(len(offline.parse(self.pdf)[0]), 2)

    def test_create_timeout_is_not_resubmitted(self):
        """模拟创建任务超时：普通重跑不能再次 POST，只有显式 resubmit 才允许创建新任务。"""
        self.fail_create = True
        with self.assertRaises(MinerUError) as error: self.parser.parse(self.pdf)
        self.assertEqual(error.exception.stage, 'submission_unknown')
        self.assertNotIn('test-secret', str(error.exception))
        with self.assertRaises(MinerUError): self.parser.parse(self.pdf)
        self.assertEqual(self.post_count(), 1)
        self.fail_create = False
        self.parser.parse(self.pdf, resubmit=True)
        self.assertEqual(self.post_count(), 2)

    def test_upload_failure_resumes_original_batch(self):
        """模拟上传失败后恢复：batch_id 已保存，重跑使用原批次，不创建另一份任务。"""
        self.fail_upload = True
        with self.assertRaises(MinerUError) as error: self.parser.parse(self.pdf)
        self.assertEqual(error.exception.stage, 'upload_failed')
        self.fail_upload = False
        self.parser.parse(self.pdf)
        self.assertEqual(self.post_count(), 1)

    def test_download_retry_and_resume(self):
        """模拟下载服务暂时失败，检查尝试次数有上限，并能在重跑后继续原任务。"""
        self.fail_download = True
        with self.assertRaises(MinerUError) as error: self.parser.parse(self.pdf)
        self.assertEqual(error.exception.stage, 'download_failed')
        self.assertEqual(sum(host == 'results.example' for _, host, _ in self.calls), 2)
        self.fail_download = False
        self.parser.parse(self.pdf)
        self.assertEqual(self.post_count(), 1)

    def test_query_retries_without_recreate(self):
        """查询失败只重试查询，不应变成新的任务创建请求。"""
        self.fail_query = True
        with self.assertRaises(MinerUError) as error: self.parser.parse(self.pdf)
        self.assertEqual(error.exception.stage, 'query_failed')
        self.fail_query = False
        self.parser.parse(self.pdf)
        self.assertEqual(self.post_count(), 1)

    def test_restart_after_remote_done_without_download(self):
        """模拟云端已完成、本地尚未下载时退出：重启应继续下载，不能把 remote_done 当成本地缓存完成。"""
        self.remote = 'done'
        folder = self.journal().parent
        atomic_json(self.journal(), {'batch_id': 'batch-1', 'status': 'remote_done'})
        self.parser.parse(self.pdf)
        self.assertEqual(self.post_count(), 0)

    def test_queued_timeout_and_resume(self):
        """用可控时钟模拟排队超时，检查任务与远端状态被保存，后续仍可查询完成结果。"""
        self.remote = 'pending'
        now = iter([0, 0, 2])
        self.parser.clock = lambda: next(now)
        with self.assertRaises(MinerUError) as error: self.parser.parse(self.pdf)
        self.assertEqual(error.exception.stage, 'wait_timeout')
        self.assertEqual(read_json(self.journal())['remote_state'], 'pending')
        self.parser.clock = lambda: 0
        self.remote = 'done'
        self.parser.parse(self.pdf)
        self.assertEqual(self.post_count(), 1)

    def test_parse_failure_is_explicit_and_redacted(self):
        """模拟云端解析失败，并在错误中夹带测试密钥和 URL，验证报告明确失败且敏感内容被清除。"""
        self.remote = 'failed'
        docs, report = parse_files([self.pdf], self.parser)
        self.assertFalse(docs)
        self.assertEqual(report[0]['stage'], 'parse_failed')
        self.assertNotIn('test-secret', json.dumps(report))
        self.assertNotIn('https://private', json.dumps(report))

    def test_incomplete_pages_rejected_and_raw_saved(self):
        """让两页 PDF 只返回一页记录，验证不会验收成功，同时原始 ZIP 仍保留便于排查。"""
        self.zip = archive(page_ids=[0])
        with self.assertRaises(MinerUError) as error: self.parser.parse(self.pdf)
        self.assertEqual(error.exception.stage, 'validation_failed')
        self.assertTrue(list(self.journal().parent.glob('result.*.zip')))
        self.assertNotEqual(read_json(self.journal())['status'], 'done')

    def test_corrupt_zip_reported(self):
        """返回非 ZIP 数据，确认损坏产物被归类为校验失败，不进入向量化流程。"""
        self.zip = b'not a zip'
        with self.assertRaises(MinerUError) as error: self.parser.parse(self.pdf)
        self.assertEqual(error.exception.stage, 'validation_failed')

    def test_cache_corruption_detected(self):
        """修改已完成缓存中的文档 JSON，验证哈希检查能发现变化，不静默使用损坏数据。"""
        self.parser.parse(self.pdf)
        (self.journal().parent/'documents.json').write_text('[]')
        with self.assertRaises(MinerUError): self.parser.parse(self.pdf)
        self.assertEqual(self.post_count(), 1)

    def test_cache_parameters_and_file_content(self):
        """分别改变解析模式与 PDF 内容，验证二者都会改变缓存键。"""
        key = self.parser.cache_key(self.pdf)
        other = MinerUCloud(MinerUSettings(model='pipeline'), self.root/'cache', self.client)
        self.assertNotEqual(key, other.cache_key(self.pdf))
        self.pdf.write_bytes(self.pdf.read_bytes() + b'\n% changed')
        self.assertNotEqual(key, self.parser.cache_key(self.pdf))

    def test_missing_key_never_creates_unknown_submission(self):
        """没有密钥时应在本地报配置错误，不能写成一次真实但结果未知的提交。"""
        parser = MinerUCloud(MinerUSettings(), self.root/'cache', self.client)
        with self.assertRaises(MinerUError) as error: parser.parse(self.pdf)
        self.assertEqual(error.exception.stage, 'configuration')
        self.assertFalse(self.journal().exists())

    def test_downloaded_archive_resumes_without_network(self):
        """预先放入下载完成的 ZIP 和日志，模拟中途退出后直接在本地完成解压与验收。"""
        folder=self.journal().parent
        folder.mkdir(parents=True,exist_ok=True)
        raw=folder/'result.saved.zip'; raw.write_bytes(self.zip)
        from rag_agent.ingestion.mineru_client import file_hash
        atomic_json(self.journal(),{'status':'downloaded','batch_id':'batch-1',
                    'archive_name':raw.name,'archive_sha256':file_hash(raw)})
        self.parser.parse(self.pdf)
        self.assertEqual(self.calls,[])

    def test_adapter_upgrade_reuses_raw_zip_without_network(self):
        """升级本地适配器后，迁移旧 ZIP 重新验收，不重新创建任务或下载。"""
        with patch('rag_agent.ingestion.mineru_client.ADAPTER_VERSION', 'cloud-content-list-v1.1'):
            self.parser.parse(self.pdf)
        before = len(self.calls)
        docs, stats = self.parser.parse(self.pdf)
        self.assertEqual(len(docs), 2)
        self.assertEqual(len(self.calls), before)
        self.assertIn('migrated_from', read_json(self.journal()))

    def test_adapter_upgrade_resumes_original_task_data_id(self):
        """升级改变本地缓存键，但查询已提交任务必须沿用它原来的 data_id。"""
        self.fail_upload = True
        with patch('rag_agent.ingestion.mineru_client.ADAPTER_VERSION', 'cloud-content-list-v1.1'):
            with self.assertRaises(MinerUError):
                self.parser.parse(self.pdf)
        original_data_id = self.submitted_data_id
        self.fail_upload = False
        docs, stats = self.parser.parse(self.pdf)
        self.assertEqual(len(docs), 2)
        self.assertEqual(self.post_count(), 1)
        self.assertEqual(read_json(self.journal())['task_data_id'], original_data_id)
        self.assertNotEqual(self.parser.cache_key(self.pdf), original_data_id)

    def test_adapter_revision_changes_key(self):
        """适配逻辑版本改变时缓存键也改变，防止继续复用旧格式转换结果。"""
        before=self.parser.cache_key(self.pdf)
        with patch('rag_agent.ingestion.mineru_client.ADAPTER_VERSION','next'):
            self.assertNotEqual(before,self.parser.cache_key(self.pdf))

    def test_download_redirect_does_not_leak_token(self):
        """模拟下载地址跳转到 CDN，检查原下载请求和跳转后的请求都不带 Token。"""
        original=self.handle
        def redirected(request):
            """模拟结果下载跳转，检查 API Token 不会泄漏到文件服务。"""
            if request.url.host=='results.example':
                self.assertIsNone(request.headers.get('Authorization'))
                return httpx.Response(302,headers={'Location':'https://cdn.example/file.zip'})
            if request.url.host=='cdn.example':
                self.assertIsNone(request.headers.get('Authorization'))
                return httpx.Response(200,content=self.zip)
            return original(request)
        with httpx.Client(transport=httpx.MockTransport(redirected)) as client:
            self.parser.client=client
            self.parser.parse(self.pdf)



class AdapterTests(unittest.TestCase):
    """验证解压安全、页码覆盖和结构化内容转换，不调用实际云端。"""
    def test_unsafe_zip_paths_and_links(self):
        """构造越界路径、Windows 保留名和符号链接，检查安全解压在写入前拒绝它们。"""
        for name in ['../outside', '/absolute', 'C:/outside', 'x\\..\\outside', 'CON.txt', 'x./file']:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'bad.zip'
                with zipfile.ZipFile(path, 'w') as z: z.writestr(name, b'x')
                with self.assertRaises(ValidationError): safe_extract(path, Path(tmp)/'out')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bad.zip'
            member = zipfile.ZipInfo('link'); member.external_attr = 0o120777 << 16
            with zipfile.ZipFile(path, 'w') as z: z.writestr(member, '../outside')
            with self.assertRaises(ValidationError): safe_extract(path, Path(tmp)/'out')

    def test_cloud_footnotes_and_references_are_preserved(self):
        """真实云端 v1 的脚注与参考文献也是正文信息，应保留页码和类型。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'cloud.zip'
            path.write_bytes(archive(blocks=[
                {'type': 'page_footnote', 'page_idx': 0, 'text': 'Footnote evidence'},
                {'type': 'ref_text', 'page_idx': 1, 'text': 'Reference evidence'}]))
            safe_extract(path, root / 'out')
            docs, stats = adapt_result(root / 'out', 'real.pdf', 2)
            self.assertEqual([d['metadata']['block_type'] for d in docs], ['page_footnote', 'ref_text'])
            self.assertEqual([d['metadata']['page'] for d in docs], [1, 2])
            self.assertEqual(stats['empty_text_pages'], [])

    def test_premerge_layout_preserves_physical_page_citations(self):
        """内容列表把第二页合并到第一页时，必须使用 layout 中合并前的逐页内容。"""
        def block(kind, text):
            """构造包含行和 span 的页内块，复现云端跨页合并前的数据结构。"""
            if kind == 'table':
                return {'type': 'table', 'blocks': [{'type': 'table_body', 'lines': [
                    {'spans': [{'type': 'table', 'html': '<table><tr><td>' + text + '</td></tr></table>'}]}]}]}
            return {'type': 'text', 'lines': [{'spans': [{'type': 'text', 'content': text}]}]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'x_content_list.json').write_text(json.dumps([
                {'type': 'text', 'page_idx': 0, 'text': 'first second'},
                {'type': 'text', 'page_idx': 1, 'text': ''}]))
            (root/'layout.json').write_text(json.dumps({'pdf_info': [
                {'page_idx': 0, 'preproc_blocks': [block('text', 'first'), block('table', 'row one')]},
                {'page_idx': 1, 'preproc_blocks': [block('text', 'second'), block('table', 'row two')]}]}))
            docs, stats = adapt_result(root, 'source.pdf', 2)
            self.assertEqual([(d['text'], d['metadata']['page']) for d in docs],
                             [('first', 1), ('| row one', 1), ('second', 2), ('| row two', 2)])
            self.assertEqual(stats['empty_text_pages'], [])
            self.assertEqual(stats['page_source'], 'preproc_blocks')

    def test_empty_page_stats_and_unknown_schema(self):
        """完整两页中只有一页有正文时应报告空文本页；遇到未知内容类型时应明确拒绝。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); z = root/'a.zip'
            z.write_bytes(archive(blocks=[{'type':'text','page_idx':0,'text':'body'}]))
            safe_extract(z,root/'out')
            docs, stats = adapt_result(root/'out','test.pdf',2)
            self.assertEqual(stats['empty_text_pages'],[2])
            self.assertEqual(stats['text_pages'],1)
            (root/'out/doc/doc_content_list.json').write_text('[{"type":"new","page_idx":0}]')
            with self.assertRaises(ValidationError): adapt_result(root/'out','test.pdf',2)

    def test_lock_is_released_and_excludes_second_writer(self):
        """同一时间第二个写入者无法取得锁；第一个退出后可以再次取得，避免误认为永久占用。"""
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'.lock'
            with exclusive_lock(path):
                with self.assertRaises(RuntimeError):
                    with exclusive_lock(path): pass
            with exclusive_lock(path): pass

    def test_shared_env_uses_distinct_keys(self):
        """同一 .env 同时放 LLM 和 MinerU 密钥，验证 MinerU 只取自己的变量且系统环境变量优先。"""
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            root=Path(tmp)
            (root/'.env').write_text('LLM_API_KEY=llm-secret\nMINERU_API_KEY=mineru-secret\n')
            settings=load_settings(root)
            self.assertEqual(settings.token,'mineru-secret')
            self.assertNotIn('LLM_API_KEY',os.environ)
            self.assertNotIn('mineru-secret',repr(settings))
            os.environ['MINERU_API_KEY']='override'
            self.assertEqual(load_settings(root).token,'override')


if __name__ == '__main__': unittest.main()
