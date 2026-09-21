# -*- coding: utf-8 -*-
"""
充电桩故障智能问答（ChatPDF + FAISS）

将《充电桩故障排查知识手册》按页码切块并写入 FAISS，现场提问后检索相关段落，
由大模型给出可执行的排查步骤，并回溯手册页码。用于缩短掉线、断电等故障的定位时间。
"""
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Tongyi
from typing import List, Tuple
import os
import pickle
import sys

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "充电桩故障排查知识手册.pdf")
SAVE_DIR = os.path.join(BASE_DIR, "vector_db")

SYSTEM_PROMPT = """你是充电桩运维现场助手。只根据【手册摘录】回答，禁止编造手册中不存在的电压、电阻、故障码。
面向新员工，按下面结构作答：
【结论】一句话判断根因类别（供电 / 通信 / 枪与车辆 / 保护）
【排查步骤】给出可执行的先后顺序
【注意】需要停用、禁止强启或升级二线的情况必须写明
【依据】点明摘录中的关键判据
资料不足时明确说“手册未覆盖，请升级二线”，不要补充猜测。"""


def extract_text_with_page_numbers(pdf) -> Tuple[str, List[int]]:
    """
    从PDF中提取文本并记录每个字符对应的页码。

    参数:
        pdf: PDF文件对象

    返回:
        text: 提取的文本内容
        char_page_mapping: 每个字符对应的页码列表
    """
    text = ""
    char_page_mapping = []

    for page_number, page in enumerate(pdf.pages, start=1):
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
            char_page_mapping.extend([page_number] * len(extracted_text))
        else:
            print(f"No text found on page {page_number}.")

    return text, char_page_mapping


def process_text_with_splitter(text: str, char_page_mapping: List[int], save_path: str = None) -> FAISS:
    """
    处理文本并创建向量存储。

    参数:
        text: 提取的文本内容
        char_page_mapping: 每个字符对应的页码列表
        save_path: 可选，保存向量数据库的路径

    返回:
        knowledgeBase: 基于FAISS的向量存储对象
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", ".", " ", ""],
        chunk_size=800,
        chunk_overlap=160,
        length_function=len,
    )

    chunks = text_splitter.split_text(text)
    print(f"文本被分割成 {len(chunks)} 个块。")

    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=DASHSCOPE_API_KEY,
    )

    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    print("已从文本块创建知识库。")

    page_info = {}
    current_pos = 0

    for chunk in chunks:
        chunk_start = current_pos
        chunk_end = current_pos + len(chunk)
        chunk_pages = char_page_mapping[chunk_start:chunk_end]

        if chunk_pages:
            page_counts = {}
            for page in chunk_pages:
                page_counts[page] = page_counts.get(page, 0) + 1
            most_common_page = max(page_counts, key=page_counts.get)
            page_info[chunk] = most_common_page
        else:
            page_info[chunk] = 1

        current_pos = chunk_end

    knowledgeBase.page_info = page_info
    print(f"页码映射完成，共 {len(page_info)} 个文本块")

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        knowledgeBase.save_local(save_path)
        print(f"向量数据库已保存到: {save_path}")
        with open(os.path.join(save_path, "page_info.pkl"), "wb") as f:
            pickle.dump(page_info, f)
        print(f"页码信息已保存到: {os.path.join(save_path, 'page_info.pkl')}")

    return knowledgeBase


def load_knowledge_base(load_path: str, embeddings=None) -> FAISS:
    """
    从磁盘加载向量数据库和页码信息。

    参数:
        load_path: 向量数据库的保存路径
        embeddings: 可选，嵌入模型。如果为None，将创建一个新的DashScopeEmbeddings实例

    返回:
        knowledgeBase: 加载的FAISS向量数据库对象
    """
    if embeddings is None:
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v1",
            dashscope_api_key=DASHSCOPE_API_KEY,
        )

    knowledgeBase = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
    print(f"向量数据库已从 {load_path} 加载。")

    page_info_path = os.path.join(load_path, "page_info.pkl")
    if os.path.exists(page_info_path):
        with open(page_info_path, "rb") as f:
            page_info = pickle.load(f)
        knowledgeBase.page_info = page_info
        print("页码信息已加载。")
    else:
        print("警告: 未找到页码信息文件。")
        knowledgeBase.page_info = {}

    return knowledgeBase


def build_or_load_knowledge_base(force: bool = False) -> FAISS:
    """手册已向量化则热加载，否则重新解析 PDF。"""
    index_file = os.path.join(SAVE_DIR, "index.faiss")
    if (not force) and os.path.exists(index_file):
        return load_knowledge_base(SAVE_DIR)

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"未找到知识手册: {PDF_PATH}")

    pdf_reader = PdfReader(PDF_PATH)
    text, char_page_mapping = extract_text_with_page_numbers(pdf_reader)
    print(f"提取的文本长度: {len(text)} 个字符。")
    return process_text_with_splitter(text, char_page_mapping, save_path=SAVE_DIR)


def answer_query(knowledgeBase: FAISS, query: str, k: int = 8) -> str:
    """检索手册片段并生成现场排查回答。"""
    docs = knowledgeBase.similarity_search(query, k=k)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"""{SYSTEM_PROMPT}

【手册摘录】
{context}

【现场问题】
{query}"""

    llm = Tongyi(model_name="deepseek-v3", dashscope_api_key=DASHSCOPE_API_KEY)
    response = llm.invoke(prompt)
    print(response)
    print("来源:")

    unique_pages = set()
    page_info = getattr(knowledgeBase, "page_info", {}) or {}
    for doc in docs:
        text_content = getattr(doc, "page_content", "")
        source_page = page_info.get(text_content, page_info.get(text_content.strip(), "未知"))
        if source_page not in unique_pages:
            unique_pages.add(source_page)
            print(f"文本块页码: {source_page}")
    return response


SAMPLE_QUERIES = [
    "充电桩掉线了怎么查？平台显示离线，屏幕还能操作。",
    "充电桩突然断电黑屏，断路器跳了，能不能直接合闸？",
    "插上枪不充电是什么原因？交流桩屏幕显示已连接。",
]


if __name__ == "__main__":
    force = "--force" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    query = " ".join(args).strip() or SAMPLE_QUERIES[0]

    knowledgeBase = build_or_load_knowledge_base(force=force)
    if query:
        print(f"\n问题: {query}\n")
        answer_query(knowledgeBase, query)
