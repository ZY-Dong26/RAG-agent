"""
chunker.py —— 文档切块层：MinerU 文档记录 → 可向量化的 chunk
职责：正文使用递归字符切分，表格优先按行切分，并完整保留来源、页码和块类型。
本模块不解析 PDF、不计算向量，也不写入 FAISS。
"""
# 标准库：json用于把切分结果保存为文件，Path用于处理输出路径
import json
from pathlib import Path

# 从langchain文本分割模块导入递归字符分割器，RAG任务用来把长文本切分成小块
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextSplitter:
    """文本切分类，将PDF读取得到的长文本切分为适合向量库存储的文本块chunk"""
    def __init__(
            self,
            chunk_size=800,
            chunk_overlap=150
    ):
        """
        初始化方法：创建文本分割器实例
        参数：
            chunk_size: 每个文本块最大字符长度
            chunk_overlap: 同一原始块内相邻片段的目标重叠字符数；实际重叠受分隔边界影响
        """
        # 创建递归字符文本分割器对象
        self.chunk_size = chunk_size
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # 切分符号优先级：段落换行 → 换行 → 中文标点 → 空格 → 单字符兜底
            # 最后的空字符串用于拆分没有任何分隔符的长文本，避免超过 chunk_size
            separators=[
                "\n\n",
                "\n",
                "。",
                "！",
                "？",
                "；",
                "：",
                "，",
                " ",
                ""
            ]
        )

    def split_documents(self, documents):
        """
        对统一文档记录逐块切分；不跨内容块合并，也不在不同原始块之间建立重叠
        参数：
            documents：列表，每一项是字典，通常是一个 MinerU 内容块及其元数据；也兼容旧的整页记录
                [{"text":"页面文本", "metadata":{"source":"xxx.pdf", "page":1}}]
        返回：
            chunks：列表，切分后的文本块字典，附带来源、页码、chunk编号
        """
        # 定义空列表，保存所有切分完成的文本块
        chunks = []
        # 本次调用内的临时编号；builder 随后会将其替换为稳定的内容哈希 ID
        chunk_id = 0

        # 循环遍历每个输入内容块
        for doc in documents:
            # 取出当前内容块的文本内容
            text = doc["text"]
            # 取出当前内容块的元信息（文件名、页码）
            metadata = doc["metadata"]

            # 后处理只标记页眉页脚等噪声，不物理删除原始块。切片阶段在这里统一跳过，
            # 因而 refined_document.json 仍可审计，而向量库不会收录这些内容。
            if metadata.get("excluded_from_retrieval"):
                continue

            # 调用分割器，把当前块长文本切分成多个小段文本
            # MinerU 已区分内容类型：表格使用按行切块策略，其他内容继续使用递归字符切块。
            if metadata.get("block_type") == "table":
                split_texts = self._split_table(text)
            else:
                split_texts = self.splitter.split_text(text)

            # 遍历当前内容块切分出来的所有文本小段
            for split_text in split_texts:
                # 组装单个chunk字典：保存文本内容与元数据
                chunk = {
                    "text": split_text,
                    "metadata": {
                        # mineru_block 只用于结构审计，可能包含完整 HTML/版面树，不复制进每个 chunk。
                        **{key: value for key, value in metadata.items() if key != "mineru_block"},
                        "chunk_id": chunk_id         # 本次切分调用内的临时编号
                    }
                }
                # 将当前chunk添加到总列表
                chunks.append(chunk)
                # 文本块编号自增
                chunk_id += 1

        # 返回全部切分后的文本块列表
        return chunks

    def _split_table(self, text):
        """
        表格切块：小表格保持完整，大表格优先按行分开。
        :param text: 适配层输出的表格文本，可能包含标题、表头和正文行
        :return: 字符串列表，每段尽量保留第一行作为阅读上下文
        第一行可能是标题，也可能是表头；本函数不会推断或重建复杂表格结构。
        单行过长时按可用字符预算分开，因此还需要用真实表格验证检索效果。
        """
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 2 or len(lines[0]) >= self.chunk_size // 2:
            return self.splitter.split_text(text)
        # 把第一行作为上下文附到后续片段；注意它可能是标题，并不一定是真正表头。
        prefix = lines[0]
        result, current = [], prefix
        for line in lines[1:]:
            if len(current) + len(line) + 1 <= self.chunk_size:
                current += "\n" + line
                continue
            if current != prefix:
                result.append(current)
                current = prefix
            if len(prefix) + len(line) + 1 > self.chunk_size:
                # 先扣除前缀与换行占用的空间，再切超长行，确保总字符数不超过 chunk_size。
                budget = self.chunk_size - len(prefix) - 1
                for start in range(0, len(line), budget):
                    result.append(prefix + "\n" + line[start:start + budget])
            else:
                current += "\n" + line
        if current != prefix:
            result.append(current)
        return result

    @staticmethod
    def save_chunks(chunks, output_path):
        """
        将切分好的chunk列表保存为JSON文件（中间产物落盘）
        参数：
            chunks：切分后的文本块列表
            output_path：输出文件路径（str或Path均可）
        返回：
            无
        """
        # 转为Path对象，方便创建父目录
        output_path = Path(output_path)
        # 父目录不存在时自动创建，避免手动建目录
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # 写入JSON：ensure_ascii=False保证中文正常显示，indent=2方便人工查看
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"已保存 {len(chunks)} 个 chunk → {output_path}")


if __name__ == "__main__":
    # 旧的本地读取教学演示：没有 OCR，会导出 chunks.json；正式 MinerU 流程请运行 scripts/build_index.py。
    # 导入备用读取工具；参数与路径统一从 config 获取
    from rag_agent import config
    from rag_agent.ingestion import pypdf_fallback

    # 1.读取文件夹中全部pdf，得到每页文档列表
    docs = pypdf_fallback.load_all_pdfs(config.RAW_PDF_DIR)

    # 2.实例化文本切分器，参数来自config
    splitter = TextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    # 3.调用切分方法，执行文本分割
    chunk_result = splitter.split_documents(docs)

    # 4.切分结果落盘为JSON，供下一步向量化入库使用
    TextSplitter.save_chunks(chunk_result, config.CHUNKS_FILE)

    # 打印总chunk数量
    print(f"总共生成chunk数量：{len(chunk_result)}")
    # 打印第一个chunk预览（防空结果保护）
    if chunk_result:
        print("\n=====第一个chunk预览=====")
        print("文本：", chunk_result[0]["text"][:500])
        print("元信息：", chunk_result[0]["metadata"])
