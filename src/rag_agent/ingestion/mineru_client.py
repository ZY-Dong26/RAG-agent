"""
mineru_client.py —— 云端解析层：PDF → MinerU 任务 → 本地解析缓存
在 RAG 中的位置：它发生在切块和 Embedding 之前，不参与每一轮聊天。
学习时建议先看 parse，再看 _parse_locked，最后看各个网络辅助方法。
关键设计：
    1. 内容哈希 + 解析参数 + 适配版本组成缓存键，避免重复上传相同文档。
    2. manifest.json 是任务日志，保存 batch_id、处理阶段和产物校验值。
    3. 创建任务只尝试一次；结果不确定时停下来，避免重复创建云端任务。
    4. 查询和下载可以有限重试；恢复时先查旧任务，而不是重新创建。
    5. API 请求携带 Token，文件上传/下载请求不携带 Token。
    6. 云端 done 只表示解析结束；本地还要下载、解压、校验页数，才能算成功。
说明：模块导入本身不会调用 API；实际网络操作从 parse 开始。
"""
import hashlib
import shutil
import json
import os
import re
import time
import uuid
import zipfile
from pathlib import Path
from urllib.parse import urlsplit, quote

from rag_agent.common.progress import report as notify, stage

import httpx
from pypdf import PdfReader

from rag_agent.common.files import atomic_json, read_json, exclusive_lock
from .mineru_adapter import ADAPTER_VERSION, adapt_result, safe_extract, ValidationError


class MinerUError(RuntimeError):
    """
    带阶段名称的异常，例如 upload_failed、query_failed，便于逐文件报告失败原因。
    """
    def __init__(self, stage, message):
        """
        保存机器可读的 stage，同时生成适合命令行展示的错误文字。
        """
        self.stage = stage
        super().__init__(f'[{stage}] {message}')


def file_hash(path):
    """
    分批读取文件，计算 SHA-256 内容指纹。
    内容相同则指纹相同，文件改名不会改变指纹；无需把整份 PDF 一次性读入内存。
    :return: 十六进制字符串，用于缓存识别或产物完整性校验
    """
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data := stream.read(1024 * 1024): digest.update(data)
    return digest.hexdigest()


def fingerprint(value):
    """
    把配置等结构化数据按稳定的键顺序序列化再哈希，避免字典键顺序影响缓存命中。
    """
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class MinerUCloud:
    """
    管理一份或多份 PDF 的云端解析生命周期，解析成功后输出本地可复用的文档记录。
    """
    def __init__(self, settings, cache_dir, client=None, sleep=time.sleep, clock=time.monotonic):
        """
        初始化客户端，但不立即联网。
        :param settings: MinerUSettings，包含密钥、解析开关和等待参数
        :param cache_dir: 各文档缓存的父目录
        :param client: 可注入 HTTP 客户端，测试时使用模拟接口
        :param sleep: 等待函数，测试时可替换为不等待的函数
        :param clock: 单调时钟，计算等待预算，避免系统时间调整影响超时判断
        """
        self.settings = settings
        self.cache_dir = Path(cache_dir)
        # 不设置全局鉴权头，API 密钥只在 _api 方法中添加。
        # 客户端本身不设默认 Authorization，避免复用连接时把密钥带给存储/CDN地址。
        # trust_env=False 表示不自动读取环境中的代理等网络配置，网络行为由此客户端显式决定。
        self.client = client or httpx.Client(timeout=settings.timeout, trust_env=False)
        self.owns_client = client is None
        self.sleep, self.clock = sleep, clock

    def close(self):
        """
        仅关闭自己创建的 HTTP 客户端；外部传入的客户端由调用方管理。
        """
        if self.owns_client: self.client.close()

    def cache_key(self, path):
        """
        计算一份文档的解析缓存键。
        不含 chunk_size：切块属于后续本地步骤，调整它不应该再次上传 PDF。
        不含 API Key：更换密钥后仍可以复用已经完成的本地解析结果。
        """
        return fingerprint({'sha256': file_hash(path), 'parameters': self.settings.parameters(),
                            'adapter': ADAPTER_VERSION, 'api': self.settings.base_url})

    def _safe_message(self, message):
        """
        日志脱敏：移除实际 Token 和 URL，避免错误消息泄漏密钥或临时签名链接。
        """
        message = str(message)
        if self.settings.token: message = message.replace(self.settings.token, '[redacted]')
        return re.sub(r'https?://\S+', '[url]', message)[:300]

    def _api(self, method, endpoint, payload=None, retry=False):
        """
        调用固定官方 API，只有这里主动添加 Bearer Token。
        :param method: POST 创建任务或 GET 查询任务
        :param endpoint: 官方 API 下的相对路径
        :param payload: 创建任务用的请求字典
        :param retry: 查询允许有限重试；创建任务保持 False
        :return: 校验成功响应后的 data 字典
        HTTP 200 不等于业务成功，还要检查响应中的 code。
        """
        if not self.settings.token:
            raise MinerUError('configuration', '请在 .env 填写 MINERU_API_KEY（与 LLM_API_KEY 是不同的配置项）')
        # retry=False 时只有一次尝试；创建任务不能像读取任务那样随意重试。
        attempts = self.settings.retries if retry else 1
        for attempt in range(attempts):
            try:
                response = self.client.request(method, self.settings.base_url + endpoint,
                    headers={'Authorization': 'Bearer ' + self.settings.token}, json=payload,
                    timeout=self.settings.timeout, follow_redirects=False)
                # 429 表示限流，5xx 表示服务端错误；允许重试时做短暂递增等待，避免立即连续请求。
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt + 1 < attempts:
                        notify(f"[重试] 网络请求未完成，将进行第 {attempt + 2} 次尝试")
                    self.sleep(min(2**attempt, 10)); continue
                if response.status_code != 200:
                    raise MinerUError('api', f'HTTP {response.status_code}')
                # 先校验 HTTP 状态，再校验业务 code，最后才提取 data，逐层保证响应可用。
                result = response.json()
                if not isinstance(result, dict) or result.get('code') != 0:
                    raise MinerUError('api', self._safe_message(result.get('msg', 'API 返回失败')
                                      if isinstance(result, dict) else 'API 返回格式错误'))
                if not isinstance(result.get('data'), dict):
                    raise MinerUError('api', 'API data 格式错误')
                return result['data']
            # 不直接输出底层异常：底层异常可能含 URL 或请求细节，统一改成可展示的错误说明。
            except (httpx.HTTPError, ValueError):
                if attempt + 1 < attempts:
                    notify(f"[重试] 网络请求未完成，将进行第 {attempt + 2} 次尝试")
                    self.sleep(min(2**attempt, 10)); continue
                raise MinerUError('api', '网络超时、连接失败或响应格式错误') from None

    @staticmethod
    def _asset_url(url):
        """
        检查上传或下载地址为带主机名的 HTTPS 地址，并拒绝 URL 内嵌用户名/密码。
        """
        parsed = urlsplit(url) if isinstance(url, str) else None
        if not parsed or parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
            raise MinerUError('asset', '服务器未返回有效 HTTPS 文件地址')
        return url

    def _upload(self, path, url):
        """
        上传原 PDF 到服务返回的签名地址，不携带 MinerU API Token。
        上传超时不代表服务器没收到文件，所以这里不盲目立即重传。
        下一次 parse 先查已保存的 batch_id；只有云端还在 waiting-file 时才上传。
        """
        url = self._asset_url(url)
        try:
            # 不盲目重试上传；下次先查询磁盘记录中的同一个任务。
            with Path(path).open('rb') as stream:
                response = self.client.put(url, content=stream, headers={},
                    timeout=self.settings.timeout, follow_redirects=False)
            if response.status_code not in (200, 201, 204):
                raise MinerUError('upload_failed', f'文件上传 HTTP {response.status_code}；下次先查询原任务')
        except httpx.HTTPError:
            raise MinerUError('upload_failed', '上传连接中断或超时；下次运行会先查询原任务') from None

    def _download(self, url, output):
        """
        流式下载结果 ZIP，先写 .part，完成后才改为正式文件名。
        :param url: 查询结果给出的下载地址；它不是带 Token 的 API 请求
        :param output: 原始 ZIP 的本地保存路径
        下载失败可以有限重试，每次从头覆盖临时文件；这不是字节级断点续传。
        """
        url = self._asset_url(url)
        # 下载途中只生成 .part 文件；断线后它不被当作成功 ZIP 使用，下次从头下载。
        partial = output.with_suffix('.part')
        for attempt in range(self.settings.retries):
            try:
                with self.client.stream('GET', url, headers={}, timeout=self.settings.timeout,
                                        follow_redirects=True) as response:
                    response.raise_for_status()
                    size = 0
                    # 流式写入避免一次性把整个压缩包放入内存，同时累计大小以限制结果体积。
                    with partial.open('wb') as stream:
                        for data in response.iter_bytes(1024 * 1024):
                            size += len(data)
                            if size > 512 * 1024**2:
                                raise MinerUError('download_failed', '结果压缩包超过 512 MiB 限制')
                            stream.write(data)
                        stream.flush(); os.fsync(stream.fileno())
                # 网络读取与本地写入全部完成后，才把临时文件改为正式产物。
                os.replace(partial, output)
                return
            except httpx.HTTPError:
                if attempt + 1 < self.settings.retries:
                    notify(f"[重试] 网络请求未完成，将进行第 {attempt + 2} 次尝试")
                    self.sleep(min(2**attempt, 10)); continue
                raise MinerUError('download_failed', '结果下载失败，保留原任务供下次重试') from None

    def _entry(self, batch_id, key):
        """
        通过 batch_id 查询本文件的解析结果，使用提交时的 data_id 匹配结果；适配版本升级后它可能不同于当前缓存键。
        """
        data = self._api('GET', '/extract-results/batch/' + quote(batch_id, safe=''), retry=True)
        results = data.get('extract_result')
        if not isinstance(results, list): raise MinerUError('query_failed', '任务结果列表缺失')
        # data_id 是提交时写入的本地缓存键，batch_id 则是云端分配的查询凭据，两者用途不同。
        matches = [r for r in results if r.get('data_id') == key]
        # 每批只有一个文件；若响应省略 data_id，仅在唯一结果时兼容匹配。
        if not matches and len(results) == 1 and not results[0].get('data_id'): matches = results
        if len(matches) != 1: raise MinerUError('query_failed', '无法匹配本文件的任务结果')
        return matches[0]

    def parse(self, path, resubmit=False):
        """
        单文件解析入口：先校验原 PDF，再进入有锁保护的恢复流程。
        :param path: 本地 PDF 路径
        :param resubmit: 人工确认后显式新建任务；默认恢复旧任务或使用缓存
        :return: (documents, statistics)，供建库流程切块和生成解析报告
        """
        # 1. 上传前在本地预检文件大小和真实页数，避免任务完成后才发现只解析了部分文档。
        path = Path(path)
        if path.stat().st_size > 200_000_000:
            raise MinerUError('validation_failed', 'PDF 超过 200 MB')
        try:
            reader = PdfReader(path)
            expected_pages = len(reader.pages)
        except Exception:
            raise MinerUError('validation_failed', '无法读取 PDF 页数，请检查文件或加密状态') from None
        if not 1 <= expected_pages <= 600:
            raise MinerUError('validation_failed', 'PDF 页数必须为 1 到 600')
        # 2. 相同内容和解析参数落入同一个目录；文件名不参与指纹，改名仍可复用结果。
        key = self.cache_key(path)
        folder = self.cache_dir / key
        folder.mkdir(parents=True, exist_ok=True)
        # 3. 同一份文档的恢复流程需要互斥，防止两个进程同时看到“未提交”并创建重复任务。
        with exclusive_lock(folder / '.lock'):
            return self._parse_locked(path, key, folder, expected_pages, resubmit)

    def _parse_locked(self, path, key, folder, expected_pages, resubmit):
        """
        核心状态机：根据磁盘日志决定“读缓存、提交、上传、等待或下载”。
        正常顺序：submitting → waiting-file → uploaded → remote_running
                  → downloading → downloaded → done。
        remote_done 是云端完成；done 是本地也已通过验收，两者不能混淆。
        异常会保留 batch_id；只有提交结果不确定时无法可靠查任务，需要人工核对。
        """
        # 任务日志是断点恢复的依据。后面的状态更新都通过内部 save 函数原子落盘。
        journal = folder / 'manifest.json'
        state = read_json(journal) if journal.exists() else {}
        def save(status, **values):
            """
            更新内存中的状态字典，再原子写入任务日志；已有 batch_id 等字段会被保留。
            """
            state.update(values, status=status, updated_at=time.time())
            atomic_json(journal, state)
        def documents():
            """
            返回已验收的缓存结果；相同文件改名后，引用来源使用本次文件名。
            """
            docs = read_json(folder / 'documents.json')
            for doc in docs: doc['metadata']['source'] = path.name
            return docs, {**state['statistics'], 'cache_key': key, 'status': 'done'}

        # ===== 分支A：已验收缓存，核对 ZIP 和文档 JSON 的哈希后直接返回 =====
        # 适配器升级只需重做本地转换。按旧版本计算同一文件的精确缓存键，
        # 迁移已经下载的原始 ZIP 或已有任务 ID，不会因此重新上传或创建任务。
        if not state and not resubmit:
            legacy_key = fingerprint({'sha256': file_hash(path), 'parameters': self.settings.parameters(),
                                      'adapter': 'cloud-content-list-v1.1', 'api': self.settings.base_url})
            legacy_folder = self.cache_dir / legacy_key
            legacy_journal = legacy_folder / 'manifest.json'
            if legacy_folder != folder and legacy_journal.exists():
                legacy = read_json(legacy_journal)
                archive_name = legacy.get('archive_name')
                state.update(legacy)
                state.pop('artifacts', None)
                state.pop('statistics', None)
                state.pop('extracted_dir', None)
                if archive_name:
                    if Path(archive_name).name != archive_name:
                        raise MinerUError('validation_failed', '旧缓存产物路径无效')
                    previous = legacy_folder / archive_name
                    if not previous.is_file() or file_hash(previous) != legacy.get('archive_sha256'):
                        raise MinerUError('validation_failed', '旧版本原始 ZIP 校验失败，拒绝自动重新上传')
                    shutil.copyfile(previous, folder / archive_name)
                save('downloaded' if archive_name else legacy.get('status', 'submission_unknown'),
                     cache_key=key, adapter_version=ADAPTER_VERSION, migrated_from=legacy_key,
                     task_data_id=legacy.get('task_data_id', legacy_key))

        if state.get('status') == 'done' and not resubmit:
            for name, digest in state.get('artifacts', {}).items():
                if not (folder / name).is_file() or file_hash(folder / name) != digest:
                    raise MinerUError('validation_failed', '本地缓存缺失或损坏；保留原任务记录，请检查缓存')
            if not state.get('artifacts'):
                raise MinerUError('validation_failed', '缓存校验信息缺失')
            notify(f"[缓存] {path.name}：已验证，复用本地解析结果")
            return documents()
        # ===== 分支B：用户显式要求新任务，先归档旧日志，旧原始 ZIP 不删除 =====
        if resubmit and state:
            # 只有显式重新提交才走这里，原任务日志与原始产物仍保留。
            atomic_json(folder / ('manifest.previous.' + uuid.uuid4().hex + '.json'), state)
            state = {}
        # ===== 分支C：没有可查询的 batch_id，才考虑创建任务 =====
        if not state.get('batch_id'):
            # submitting 可能意味着提交途中程序退出；服务端是否收到未知，不能自动当作“没提交”。
            if state.get('status') in ('submitting', 'submission_unknown'):
                raise MinerUError('submission_unknown',
                    '上次提交结果不确定，禁止自动重发。先核对云端任务；确认后使用 --resubmit --file 指定文件')
            if not self.settings.token:
                raise MinerUError('configuration', '请在 .env 填写 MINERU_API_KEY')
            # 先落盘“即将提交”，再发请求。即使请求途中退出，下次也知道这是不确定状态。
            save('submitting', cache_key=key, task_data_id=key, source=path.name, expected_pages=expected_pages,
                 parameters=self.settings.parameters(), adapter_version=ADAPTER_VERSION)
            # 每批只提交一份文件，便于单文件恢复；不传局部页范围，因此目标是处理整份 PDF。
            payload = {'files': [{'name': path.name, 'data_id': key, 'is_ocr': self.settings.ocr}],
                       'model_version': self.settings.model, 'language': self.settings.language,
                       'enable_formula': self.settings.formula, 'enable_table': self.settings.table}
            try:
                with stage(f'创建解析任务：{path.name}'):
                    data = self._api('POST', '/file-urls/batch', payload)
                batch_id = data.get('batch_id')
                if not isinstance(batch_id, str) or not batch_id:
                    raise MinerUError('submission_unknown', '提交响应中缺少 batch_id')
                # 获得 batch_id 立即保存；检查上传地址和上传文件都放在后面。
                save('waiting-file', batch_id=batch_id)
                notify('[提交] 任务已创建，恢复记录已保存')
                urls = data.get('file_urls')
                if not isinstance(urls, list) or len(urls) != 1:
                    raise MinerUError('upload_failed', '提交成功，但上传地址缺失')
                save('waiting-file', upload_url=self._asset_url(urls[0]), upload_complete=False)
            except MinerUError as error:
                save('upload_failed' if state.get('batch_id') else 'submission_unknown',
                     error=self._safe_message(error))
                raise MinerUError(state['status'], state['error']) from None

        # ===== 分支D：已有 batch_id，恢复本地收尾或继续查询旧任务 =====
        notify(f"[恢复] {path.name}：使用已保存任务，不重复创建")
        last_remote = None
        deadline = self.clock() + self.settings.max_wait
        try:
            # 下载已经完成时，可以直接从本地 ZIP 恢复验收，不必再次联网。
            archive = folder / state.get('archive_name', 'result.zip')
            if state.get('archive_sha256') and archive.exists() and file_hash(archive) == state['archive_sha256']:
                return self._finish(folder, archive, path.name, expected_pages, key, save, documents)
            # 轮询：每次查询当前状态并落盘；超出本次等待预算后退出，但不取消云端任务。
            while self.clock() < deadline:
                try:
                    # 本地适配版本变化会改变缓存键，但远端任务仍使用提交时的 data_id。
                    # 恢复旧任务必须保留原 ID，不能拿新缓存键匹配旧任务结果。
                    task_data_id = state.get('task_data_id') or state.get('migrated_from') or key
                    entry = self._entry(state['batch_id'], task_data_id)
                except MinerUError as error:
                    raise MinerUError('query_failed', self._safe_message(error)) from None
                remote = entry.get('state')
                if remote != last_remote:
                    labels = {'waiting-file': '等待上传', 'pending': '云端排队', 'running': '云端解析中',
                              'converting': '云端转换中', 'done': '云端解析完成', 'failed': '云端解析失败'}
                    notify(f'[云端] {path.name}：{labels.get(remote, "未知状态")}')
                    last_remote = remote
                # 给云端状态加 remote_ 前缀，避免远端 done 被误当作本地已验证的 done。
                save('remote_' + str(remote), remote_state=remote, progress=entry.get('extract_progress', {}))
                # 云端尚未收到文件才尝试上传；本地已经成功 PUT 后，等待云端状态刷新。
                if remote == 'waiting-file':
                    if not state.get('upload_complete'):
                        if not state.get('upload_url'):
                            raise MinerUError('upload_failed', '原任务缺少上传地址，需人工核对后显式重新提交')
                        # 上传前后复核 PDF 内容，防止用户同时编辑文件导致缓存键与实际上传内容不一致。
                        if self.cache_key(path) != key:
                            raise MinerUError('validation_failed', 'PDF 在提交后发生变化，请重新运行')
                        with stage(f'上传 {path.name}（{path.stat().st_size / 1024**2:.1f} MB）'):
                            self._upload(path, state['upload_url'])
                        if self.cache_key(path) != key:
                            raise MinerUError('validation_failed', 'PDF 在上传中发生变化，拒绝缓存该结果')
                        save('uploaded', upload_complete=True)
                # pending=排队，running=解析，converting=格式转换；这些状态继续等，不创建新任务。
                elif remote in ('pending', 'running', 'converting'):
                    pass
                elif remote == 'failed':
                    raise MinerUError('parse_failed', self._safe_message(entry.get('err_msg', '云端解析失败')))
                # 云端完成只是下载的起点；只有后面的 _finish 验收通过，整个文件才算成功。
                elif remote == 'done':
                    save('downloading')
                    # 每次原始结果使用独立文件名，保留旧结果便于复现和比较。
                    archive = folder / ('result.' + uuid.uuid4().hex + '.zip')
                    with stage(f'下载解析结果：{path.name}'):
                        self._download(entry.get('full_zip_url'), archive)
                    save('downloaded', archive_name=archive.name, archive_sha256=file_hash(archive))
                    return self._finish(folder, archive, path.name, expected_pages, key, save, documents)
                else:
                    raise MinerUError('query_failed', '云端返回未知任务状态')
                self.sleep(self.settings.poll_interval)
            raise MinerUError('wait_timeout', '等待超时，已保存排队/解析进度，下次继续查询原 batch_id')
        # 保存明确的失败阶段再向上传递，让下次恢复以及批次报告都能定位问题。
        except MinerUError as error:
            save(error.stage, error=self._safe_message(error))
            raise
        except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile) as error:
            save('validation_failed', error='本地结果读取、写入或格式校验失败')
            raise MinerUError('validation_failed', '本地结果读取、写入或格式校验失败') from None

    def _finish(self, folder, archive, source, expected_pages, key, save, documents):
        """
        下载后的本地收尾：安全解压 → 适配与页数验收 → 保存文档 → 标记 done。
        只有所有步骤成功才记录最终状态和产物哈希；失败时保留原始 ZIP 便于排查。
        这里不进行 Embedding，也不修改现有向量索引。
        """
        extracted = folder / ('extracted.' + uuid.uuid4().hex)
        try:
            with stage(f"安全解压与页数验收：{source}"):
                safe_extract(archive, extracted)
                docs, stats = adapt_result(extracted, source, expected_pages)
        except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile) as error:
            message = self._safe_message(error) if isinstance(error, ValidationError) else '压缩包或解析结果格式错误'
            raise MinerUError('validation_failed', message) from None
        # 安全和页覆盖检查全部通过后，保存给 RAG 使用的统一文档记录。
        atomic_json(folder / 'documents.json', docs)
        # 最后一步才记 done，并保存内容校验值；避免下一次命中未完成的缓存。
        save('done', statistics=stats, extracted_dir=extracted.name, error=None,
             artifacts={archive.name: file_hash(archive), 'documents.json': file_hash(folder / 'documents.json')})
        return documents()


def parse_files(paths, parser, resubmit=False):
    """
    顺序解析多份文件，保留成功结果并汇总每一份失败。
    :param paths: PDF 路径列表
    :param parser: 可复用的 MinerUCloud 客户端
    :param resubmit: 是否显式新建任务；命令行入口限制为指定单个文件
    :return: (documents, report)，report 包含每份文件的成功统计或失败阶段
    这里收集失败而不中途终止；是否允许建库，由调用方统一检查。
    """
    # documents 汇总成功内容；report 记录所有文件。上层根据 report 决定是否继续建库。
    documents, report = [], []
    paths = list(paths)
    for number, path in enumerate(paths, 1):
        try:
            with stage(f"解析 [{number}/{len(paths)}] {Path(path).name}"):
                docs, stats = parser.parse(path, resubmit=resubmit)
            # extend 把单文件的多条文档记录展开加入总列表，而不是形成嵌套列表。
            documents.extend(docs)
            report.append({'source': Path(path).name, **stats})
            print(f'{Path(path).name}: {stats["parsed_pages"]} 页，'
                  f'{stats["body_blocks"]} 正文块，{stats["table_blocks"]} 表格，'
                  f'空文本页 {stats["empty_text_pages"]}')
        except (MinerUError, RuntimeError, ValueError, OSError) as error:
            message = parser._safe_message(error)
            report.append({'source': Path(path).name, 'status': 'failed',
                           'stage': getattr(error, 'stage', 'local_failed'), 'error': message})
            print(f'{Path(path).name}: {message}')
    return documents, report
