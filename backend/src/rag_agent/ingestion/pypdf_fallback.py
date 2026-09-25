"""pypdf_fallback.py —— 本地备用教学读取器，仅提取 PDF 已有文字，不做 OCR。
MinerU 主流程不会在云端失败时自动调用它；扫描件可能得到空文本。
输出按页组织，区别于 MinerU 的页内内容块。
"""
# pathlib是Python内置库，Path类用于安全、跨平台处理文件路径
from pathlib import Path
# pypdf第三方库，PdfReader用来读取PDF文档内容
from pypdf import PdfReader


def load_pdf(file_path):
    """
    读取单个PDF文件
     参数:
        file_path: PDF文件路径（Path对象）
     返回:
        documents: 列表，列表中每个元素是字典，保存一页的文本和元信息
    """
    # 空列表，用来存放该pdf每一页的数据字典
    documents = []
    # 创建PDF读取实例，打开传入的pdf文件
    reader = PdfReader(file_path)

    # 打印当前正在读取的pdf文件名（file_path.name拿到文件名）
    print(f"正在读取: {file_path.name}")
    # reader.pages是pdf所有页的列表，len获取总页数
    print(f"总页数: {len(reader.pages)}")

    # enumerate：同时拿到页码索引page_number 和页对象page，从0开始计数
    for page_number, page in enumerate(reader.pages):
        # extract_text()：提取当前页面中的文字
        text = page.extract_text()

        # 如果提取结果是None（空白页/扫描图片pdf），强制赋值为空字符串，避免后续报错
        if text is None:
            text = ""

        # 构造字典：保存当前页文本 + 元数据（来源文件名、页码）
        document = {
            "text": text,
            "metadata": {
                "source": file_path.name,  # 记录来自哪个pdf文件
                "page": page_number + 1    # page_number从0开始，+1转为人类可读页码
            }
        }
        # 将当前页字典追加到列表
        documents.append(document)

    # 返回这个pdf全部页面数据列表
    return documents


def load_all_pdfs(folder_path):
    """
    批量读取文件夹中的所有PDF
     参数:
        folder_path: 文件夹路径，示例为data/raw目录
     返回:
        all_documents：列表，包含文件夹下全部pdf的所有页面内容
    """
    # 将传入路径转为Path路径对象，方便后续文件查找
    folder = Path(folder_path)
    # 空列表，存放所有pdf全部页面数据
    all_documents = []

    # glob("*.pdf")：匹配文件夹内后缀为.pdf的文件；sorted排序保证顺序稳定（chunk编号可复现）
    pdf_files = sorted(folder.glob("*.pdf"))

    # 空文件夹防护：一个PDF都没有时提前返回空列表，避免下游IndexError
    if not pdf_files:
        print(f"警告：{folder} 下没有PDF文件，返回空列表")
        return []

    # 打印一共找到多少个pdf文件
    print(f"发现 {len(pdf_files)} 个PDF文件")

    # 循环遍历每一个pdf文件
    for pdf_file in pdf_files:
        # 调用load_pdf读取单个pdf，得到该pdf所有页的列表
        documents = load_pdf(pdf_file)
        # extend：把当前pdf的页面列表合并进总列表（append是加整个列表，extend是把里面元素逐个加进去）
        all_documents.extend(documents)

    print("\n=====读取完成=====")
    # 统计全部pdf的总页面数量
    print(f"总页面数量: {len(all_documents)}")

    # 返回全部页面数据
    return all_documents


if __name__ == "__main__":
    # 从config取原始PDF目录：基于代码文件定位，不依赖"在哪里运行"
    from rag_agent import config

    # 调用批量读取函数，读取文件夹所有pdf
    documents = load_all_pdfs(config.RAW_PDF_DIR)

    # 空结果防护：没有读到任何页面时跳过预览
    if not documents:
        print("没有读取到任何PDF内容，请检查 data/raw 目录")
    else:
        # 打印第一页文本，只取前1000字符预览
        print("\n=====第一页内容===== ")
        print(documents[0]["text"][:1000])

        # 打印第一页的元信息（文件名、页码）
        print("\n=====第一页信息=====")
        print(documents[0]["metadata"])
