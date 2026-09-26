"""
mineru_settings.py —— MinerU 配置层：backend/.env → 类型明确的配置对象
职责：只读取和校验配置，不上传文件，也不加载模型。
关键设计：
    1. LLM 和 MinerU 共用 .env，但分别使用 LLM_*、MINERU_* 变量。
    2. dotenv_values 返回字典，不把文件中的密钥写入进程环境。
    3. 系统环境变量优先，便于在不同机器上覆盖配置。
    4. 解析参数参与缓存标识；密钥和网络等待时间不参与。
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import dotenv_values


# dataclass 可以理解为“配置表”：下面每个字段都对应一个有类型和默认值的配置项。
@dataclass(frozen=True)
class MinerUSettings:
    """
    MinerU 运行配置。dataclass 自动生成初始化方法；frozen=True 防止运行中误改配置。
    """
    # repr=False：打印整个配置对象时不显示 token，减少调试时意外暴露密钥的机会。
    token: str = field(default='', repr=False)
    base_url: str = 'https://mineru.net/api/v4'
    model: str = 'vlm'
    language: str = 'ch'
    ocr: bool = True
    formula: bool = True
    table: bool = True
    # 网络参数与解析参数分开：timeout 管一次请求，max_wait 管轮询预算，retries 包含第一次尝试。
    timeout: float = 60
    poll_interval: float = 5
    max_wait: float = 1800
    retries: int = 3
    # 主动刷新开关：云端模型变化后可手动修改此值，让相同 PDF 使用新的缓存目录。
    cache_revision: str = '1'

    def __post_init__(self):
        """
        创建配置对象后立即校验。配置错误应在上传 PDF 之前被发现。
        """
        if self.model not in ('vlm', 'pipeline'):
            raise ValueError('MINERU_MODEL 必须为 vlm 或 pipeline')
        if min(self.timeout, self.poll_interval, self.max_wait) <= 0 or not 1 <= self.retries <= 10:
            raise ValueError('MinerU 超时/轮询间隔必须为正数，重试次数为 1 到 10')
        if self.base_url != 'https://mineru.net/api/v4':
            raise ValueError('当前客户端仅支持官方 https://mineru.net/api/v4 API')

    def parameters(self):
        """
        返回会影响解析结果的参数，供缓存键使用。
        密钥、超时和重试次数只影响访问方式，不改变文档内容，因此不放进这里。
        cache_revision 是用户主动刷新的标识，不代表云端真实模型版本。
        """
        return dict(model_version=self.model, language=self.language,
                    is_ocr=self.ocr, enable_formula=self.formula,
                    enable_table=self.table, cache_revision=self.cache_revision)


def load_settings(root):
    """
    读取 backend/ 下的统一 .env，仅选取 MINERU_* 配置。
    :param root: Python 工程根目录 backend/，不是当前命令行所在目录
    :return: MinerUSettings 对象，数字与布尔开关已完成类型转换
    """
    # 这里只把 .env 解析成字典。读取配置本身不会访问 MinerU，也不会污染 LLM 的环境变量。
    values = dotenv_values(Path(root) / '.env')
    def get(name, default):
        """
        按“系统环境变量 → .env → 默认值”的优先级取配置，并统一加 MINERU_ 前缀。
        """
        return os.environ.get('MINERU_' + name, values.get('MINERU_' + name) or default)
    def boolean(name, default='true'):
        """
        环境文件中都是字符串；只接受 true/false，避免 bool("false") 仍然为真的陷阱。
        """
        value = get(name, default).lower()
        if value not in ('true', 'false'):
            raise ValueError(f'MINERU_{name} 必须为 true 或 false')
        return value == 'true'
    # 把文件中的字符串转换成 float/int/bool，再交给配置对象做范围检查。
    return MinerUSettings(
        token=get('API_KEY', ''), base_url=get('BASE_URL', 'https://mineru.net/api/v4').rstrip('/'),
        model=get('MODEL', 'vlm'), language=get('LANGUAGE', 'ch'), ocr=boolean('OCR'),
        formula=boolean('FORMULA'), table=boolean('TABLE'), timeout=float(get('TIMEOUT', '60')),
        poll_interval=float(get('POLL_INTERVAL', '5')), max_wait=float(get('MAX_WAIT', '1800')),
        retries=int(get('RETRIES', '3')), cache_revision=get('CACHE_REVISION', '1'))
