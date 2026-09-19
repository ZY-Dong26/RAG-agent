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
            chunk_overlap: 相邻文本块重叠字符数，保留上下文信息
        """
        # 创建递归字符文本分割器对象
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # 切分符号优先级：优先段落换行，再换行，中文标点，最后空格
            separators=[
                "\n\n",
                "\n",
                "。",
                "！",
                "？",
                "；",
                "：",
                "，",
                " "
            ]
        )

    def split_documents(self, documents):
        """
        对PDF加载器输出的文档列表执行文本切分
        参数：
            documents：列表，每一项是字典，一页PDF文本+元数据
                [{"text":"页面文本", "metadata":{"source":"xxx.pdf", "page":1}}]
        返回：
            chunks：列表，切分后的文本块字典，附带来源、页码、chunk编号
        """
        # 定义空列表，保存所有切分完成的文本块
        chunks = []
        # chunk全局编号，每生成1个文本块，编号+1
        chunk_id = 0

        # 循环遍历每一页PDF文档
        for doc in documents:
            # 取出当前页面的文本内容
            text = doc["text"]
            # 取出当前页面的元信息（文件名、页码）
            metadata = doc["metadata"]

            # 调用分割器，把当前页长文本切分成多个小段文本
            split_texts = self.splitter.split_text(text)

            # 遍历当前页面切分出来的所有文本小段
            for split_text in split_texts:
                # 组装单个chunk字典：保存文本内容与元数据
                chunk = {
                    "text": split_text,
                    "metadata": {
                        "source": metadata["source"], # PDF文件名
                        "page": metadata["page"],    # 当前页码
                        "chunk_id": chunk_id         # 文本块全局编号
                    }
                }
                # 将当前chunk添加到总列表
                chunks.append(chunk)
                # 文本块编号自增
                chunk_id += 1

        # 返回全部切分后的文本块列表
        return chunks

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
    # 导入上游函数：PDF批量读取来自 document_loader 模块；参数与路径统一从config获取
    from src import config, document_loader

    # 1.读取文件夹中全部pdf，得到每页文档列表
    docs = document_loader.load_all_pdfs(config.RAW_PDF_DIR)

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
