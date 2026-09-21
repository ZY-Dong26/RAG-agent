# generator.py —— 生成层：检索到的chunk上下文 + 用户问题 → 云端LLM → 答案
# 关键设计：
#   1. LLM走OpenAI兼容协议：云端大模型（DeepSeek/通义/Kimi/OpenAI等）基本都支持，
#      切换服务时在根目录 .env 配置兼容服务的 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 三件套
#   2. client可注入（依赖注入）：测试时传假client，不真的调云端；平时自动按config创建
#   3. prompt = 系统约束(只依据资料回答) + 编号资料 + 问题，句末用[编号]标注引用
#   4. 资料拼接使用字符预算 MAX_CONTEXT_CHARS；它不是完整请求的 Token 计数

from rag_agent import config


class Generator:
    """基于云端大模型的回答生成器（OpenAI兼容接口）"""

    def __init__(self, client=None, model=None, temperature=None, max_tokens=None):
        """
        :param client: openai客户端实例；为None时按config自动创建（注入点）
        :param model: 模型名，如 deepseek-chat；为None读取config.LLM_MODEL
        :param temperature: 采样温度，None用config（0.3，事实问答偏保守）
        :param max_tokens: 最大输出token数，None用config.LLM_MAX_TOKENS
        """
        self.model = model or config.LLM_MODEL
        self.temperature = config.LLM_TEMPERATURE if temperature is None else temperature
        self.max_tokens = max_tokens or config.LLM_MAX_TOKENS

        if client is not None:
            # 外部注入的客户端（测试/特殊场景），跳过密钥检查与客户端创建；模型名等默认参数仍来自 config
            self.client = client
        else:
            self._check_config()
            # 延迟导入：只有真正要调LLM时才依赖openai库
            from openai import OpenAI
            self.client = OpenAI(
                api_key=config.LLM_API_KEY,
                base_url=config.LLM_BASE_URL,
            )

    @staticmethod
    def _check_config():
        """校验云端LLM配置是否齐全，缺key时给出明确的补齐指引"""
        if not config.LLM_API_KEY:
            raise RuntimeError(
                "未配置云端大模型API Key。请在项目根目录 .env 文件中添加：\n"
                f"  LLM_API_KEY=你的key\n"
                f"  LLM_BASE_URL={config.LLM_BASE_URL}\n"
                f"  LLM_MODEL={config.LLM_MODEL}\n"
                "（换厂商只需改这三项，见config.py注释）"
            )

    @staticmethod
    def build_prompt(query, hits, max_chars=None):
        """
        拼接发给LLM的消息列表（不联网，纯本地拼装，可单独测试）
        :param query: 用户问题
        :param hits: Retriever的返回（含text/metadata/score/rank）
        :param max_chars: 资料总字符上限，遇到第一条放不下的资料即停止，不截断该块，也不再尝试后续块
        :return: messages，[{"role":"system",...}, {"role":"user",...}]
        """
        max_chars = max_chars or config.MAX_CONTEXT_CHARS

        # 把命中的chunk按排名拼成"编号+正文"的资料块，累计编号和正文长度受 max_chars 限制；块间换行、问题和系统消息不计入此预算
        blocks = []
        total = 0
        for hit in hits:
            rank = hit["metadata"].get("rank", "?")
            block = f"[{rank}] {hit['text'].strip()}"
            if total + len(block) > max_chars:   # 当前整块放不下就停止追加
                break
            blocks.append(block)
            total += len(block)
        context = "\n\n".join(blocks)

        # 系统消息：限定只能依据资料回答——这是抑制幻觉的关键
        system = (
            "你是知识库问答助手。请只依据【资料】中的内容回答问题，"
            "不要编造资料之外的事实。回答时在句末用[编号]标注依据的资料条目。"
            "如果资料中没有相关内容，请直接回答：资料中没有找到相关内容。"
        )
        # 用户消息：资料 + 问题
        user = f"【资料】\n{context}\n\n【问题】\n{query}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def generate(self, query, hits):
        """
        生成回答：hits（检索结果） + query → 云端LLM → 答案文本
        :param query: 用户问题
        :param hits: Retriever.retrieve()的返回列表
        :return: 回答字符串（可能含[编号]引用标注）
        """
        messages = self.build_prompt(query, hits)

        # OpenAI兼容接口的标准调用：服务端需要支持该协议和传入参数
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()


if __name__ == "__main__":
    # 自测：只验证prompt拼装（不联网、不花钱），看发给LLM的消息长什么样
    fake_hits = [
        {
            "text": "检索增强生成（RAG）是一种结合检索与生成的大模型技术。",
            "metadata": {"source": "RAG综述.pdf", "page": 1, "score": 0.65, "rank": 1},
        },
        {
            "text": "RAG先检索相关资料，再让大模型基于资料生成答案。",
            "metadata": {"source": "RAG综述.pdf", "page": 3, "score": 0.64, "rank": 2},
        },
    ]
    messages = Generator.build_prompt("什么是RAG？", fake_hits)
    print("===== 发给LLM的消息（system）=====")
    print(messages[0]["content"])
    print("\n===== 发给LLM的消息（user）=====")
    print(messages[1]["content"])
    print("\n说明：真正调用时执行 Generator().generate(query, hits)，需要先配好API Key。")
