# 03_embedding.py —— 文本向量化模块
# 职责：只负责【文本 → 固定维度稠密向量】，不关心向量存储、向量检索逻辑
# 模型：Qwen3-Embedding-0.6B（本地运行，中文效果优秀，输出1024维向量）
# 加载方式：sentence-transformers（官方推荐），按模型目录流水线加载：
#          Qwen3骨干 → 1_Pooling(均值池化) → 2_Normalize(归一化)，输出float32
# 重要约定：包内模块导入统一使用 from src import xxx
# 注意：文件名带数字前缀，不能用点号直接导入（如 import src.03_embedding 会语法报错）

import os
import torch
# SentenceTransformer：官方推荐的加载方式，内部完成分词、批量、池化、归一化
from sentence_transformers import SentenceTransformer

# 从项目配置文件读取全局参数（模型路径、最大长度等）
from src import config


class Embedding:
    """
    文本向量化工具类
    功能：输入句子/段落文本，输出归一化后的1024维稠密向量
    使用场景：
        1. 建库阶段：embed_texts，批量把文档chunk转为向量存入向量库
        2. 问答检索阶段：embed_query，把用户提问转为向量，用于相似度匹配
    """

    def __init__(self, model_name=None, device=None, max_length=None):
        """
        Embedding 类初始化方法：加载本地embedding模型与分词器
        :param model_name: 本地embedding模型文件夹路径；为None时读取config.EMBEDDING_MODEL
        :param device: 模型运行设备，可选cuda / cpu；None时自动优先使用NVIDIA显卡cuda
        :param max_length: 文本最大token长度，超过会截断；None读取config.EMBEDDING_MAX_LENGTH
        """
        # 模型路径优先级：传入参数 > 全局配置文件
        self.model_name = model_name or config.EMBEDDING_MODEL
        # 单条文本最大token上限
        self.max_length = max_length or config.EMBEDDING_MAX_LENGTH

        # 校验本地模型文件夹是否存在，给出友好提示，避免加载器晦涩报错
        if not os.path.isdir(self.model_name):
            raise FileNotFoundError(
                f"找不到本地模型目录：{self.model_name}\n"
                "请确认 00_config.py 的 EMBEDDING_MODEL 填写的是模型解压后的文件夹路径。"
            )

        # 自动判断设备：有可用NVIDIA GPU就用cuda加速，否则回退CPU
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # SentenceTransformer按模型目录的官方流水线加载模型：
        # Qwen3骨干 → 1_Pooling(均值池化) → 2_Normalize(归一化)
        # 输出自动为float32，避免bfloat16在CPU上无法转numpy的报错
        self.model = SentenceTransformer(self.model_name, device=self.device)

        # 设置最大输入token数（encode时自动截断超长文本）
        # 该模型默认支持32768，这里按config收敛到8192，平衡速度与效果
        self.model.max_seq_length = self.max_length

        print(f"Embedding模型已加载: {self.model_name} (device={self.device})")

    def embed_texts(self, texts):
        """
        批量文本向量化方法
        适用场景：知识库构建，批量处理文档切片chunk
        :param texts: 字符串列表，例如 ["文档片段1", "文档片段2"]
        :return: vectors，二维list，shape=[文本数量,1024]，向量已经L2归一化
                 归一化后，点积等价余弦相似度，适配FAISS IndexFlatIP索引
        """
        # 空输入直接返回空列表，防止后续处理报错
        if not texts:
            return []

        # encode：ST内部完成分词、padding、批量计算、均值池化、L2归一化
        # normalize_embeddings=True：确保输出单位向量（流水线已有归一化层，双保险）
        # 截断上限由 self.model.max_seq_length 控制（__init__里已设置）
        vectors = self.model.encode(texts, normalize_embeddings=True)
        # numpy数组转普通list，方便FAISS/JSON使用
        return vectors.tolist()

    def embed_query(self, query):
        """
        用户查询单条向量化方法
        适用场景：问答检索阶段，把用户问题转为向量
        ✅ Qwen3-Embedding官方最佳实践：检索任务，【查询文本加指令前缀】，文档chunk不加前缀，提升检索召回效果
        :param query: 用户原始提问字符串
        :return: vector，一维list，长度1024，归一化后的查询向量
        """
        # 模型目录自带官方检索指令模板(prompts.query)：
        # "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
        # prompt_name="query"自动套用该模板，与官方最佳实践完全一致
        vector = self.model.encode([query], prompt_name="query", normalize_embeddings=True)
        # encode返回二维数组(1,1024)，取第0行转成一维list
        return vector[0].tolist()


if __name__ == "__main__":
    # 模块自测入口：运行命令 python -m src.03_embedding
    # 注意：必须使用 -m 模块方式运行，不能直接 python src/03_embedding.py，否则src包导入会出错

    # 1. 实例化Embedding对象，加载本地模型（首次加载需要耗时）
    emb = Embedding()

    # 2. 单条测试：用户提问转向量
    v = emb.embed_query("什么是检索增强生成？")
    print(f"问题向量维度: {len(v)}")
    print(f"前5个值: {v[:5]}")

    # 3. 批量测试：多条文档chunk转向量
    vs = emb.embed_texts(["RAG是检索增强生成", "向量数据库用于相似度检索"])
    print(f"批量向量数量: {len(vs)}，维度: {len(vs[0])}")
