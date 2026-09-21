# -*- coding: utf-8 -*-
"""
充电桩 RAG 助手 - 知识库构建（索引入库）

解析现场手册 DOCX（含表格）与排查示意图，用多模态 embedding 写入 FAISS。
新员工提问掉线、断电、插枪无响应时，可同时命中文字步骤和路径图。
"""
import os
import base64
import json
import numpy as np
import faiss
import dashscope
from http import HTTPStatus
from docx import Document as DocxDocument

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("错误：请设置 'DASHSCOPE_API_KEY' 环境变量。")

dashscope.api_key = DASHSCOPE_API_KEY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
DOCS_DIR = "charge_knowledge_base"
IMG_DIR = os.path.join(DOCS_DIR, "images")
MULTIMODAL_EMBEDDING_MODEL = "tongyi-embedding-vision-plus"

# FAISS on Windows cannot open non-ASCII absolute paths; keep filenames relative after chdir.
INDEX_FILE = "charge_index.faiss"
METADATA_FILE = "charge_metadata.json"

# 切分参数：滑动窗口，保护故障码表与时序步骤不被腰斩
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

VIDEO_KNOWLEDGE = []


def parse_docx(file_path):
    """解析 DOCX，按 XML 顺序提取段落与表格（表格转 Markdown，避免故障码表丢失）。"""
    doc = DocxDocument(file_path)
    all_text = []

    for element in doc.element.body:
        if element.tag.endswith("p"):
            paragraph_text = ""
            for run in element.findall(
                ".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
            ):
                paragraph_text += run.text if run.text else ""
            if paragraph_text.strip():
                all_text.append(paragraph_text.strip())

        elif element.tag.endswith("tbl"):
            table = [t for t in doc.tables if t._element is element][0]
            if table.rows:
                md_table = []
                header = [cell.text.strip() for cell in table.rows[0].cells]
                md_table.append("| " + " | ".join(header) + " |")
                md_table.append("|" + "---|" * len(header))
                for row in table.rows[1:]:
                    row_data = [cell.text.strip() for cell in row.cells]
                    md_table.append("| " + " | ".join(row_data) + " |")
                all_text.append("\n".join(md_table))

    return "\n".join(all_text)


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """滑动窗口切分，兼顾故障码表与步骤上下文。"""
    chunks = []
    start = 0
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def get_text_embedding(text):
    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL,
        input=[{"text": text}],
    )
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"文本Embedding失败: {resp.message}")
    return resp.output["embeddings"][0]["embedding"]


def get_image_embedding(image_path):
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")

    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    image_data = f"data:image/{ext};base64,{base64_image}"

    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL,
        input=[{"image": image_data}],
    )
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"图片Embedding失败: {resp.message}")
    return resp.output["embeddings"][0]["embedding"]


def get_video_embedding(video_url):
    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL,
        input=[{"video": video_url}],
    )
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"视频Embedding失败: {resp.message}")

    embeddings = resp.output["embeddings"]
    if len(embeddings) > 1:
        vectors = [np.array(e["embedding"]) for e in embeddings]
        return np.mean(vectors, axis=0).tolist()
    return embeddings[0]["embedding"]


def build_and_save():
    print("\n--- 构建充电桩多模态知识库 ---")
    print(f"切分参数: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")

    metadata_store = []
    all_vectors = []
    doc_id = 0

    for filename in sorted(os.listdir(DOCS_DIR)):
        if filename.startswith(".") or os.path.isdir(os.path.join(DOCS_DIR, filename)):
            continue

        file_path = os.path.join(DOCS_DIR, filename)
        if filename.endswith(".docx"):
            print(f"  处理文档: {filename}")
            full_text = parse_docx(file_path)
            chunks = split_text(full_text)
            print(f"    文档长度: {len(full_text)} 字符, 切分为 {len(chunks)} 个chunk")

            for chunk in chunks:
                metadata = {
                    "id": doc_id,
                    "source": filename,
                    "type": "text",
                    "content": chunk,
                }
                vector = get_text_embedding(chunk)
                all_vectors.append(vector)
                metadata_store.append(metadata)
                doc_id += 1

    print("  处理图片...")
    img_files = os.listdir(IMG_DIR) if os.path.isdir(IMG_DIR) else []
    for img_filename in sorted(img_files):
        if img_filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
            img_path = os.path.join(IMG_DIR, img_filename)
            rel_path = os.path.relpath(img_path, BASE_DIR)
            print(f"    - {img_filename}")
            metadata = {
                "id": doc_id,
                "source": f"图片: {img_filename}",
                "type": "image",
                "path": rel_path.replace("\\", "/"),
                "content": f"[图片] {img_filename} 充电桩现场排查示意图",
            }
            vector = get_image_embedding(img_path)
            all_vectors.append(vector)
            metadata_store.append(metadata)
            doc_id += 1

    print("  处理视频...")
    for video_info in VIDEO_KNOWLEDGE:
        print(f"    - {video_info['description']}")
        metadata = {
            "id": doc_id,
            "source": f"视频: {video_info['description']}",
            "type": "video",
            "url": video_info["url"],
            "description": video_info["description"],
            "content": f"[视频] {video_info['description']}",
        }
        vector = get_video_embedding(video_info["url"])
        all_vectors.append(vector)
        metadata_store.append(metadata)
        doc_id += 1

    if all_vectors:
        dim = len(all_vectors[0])
        print(f"\n向量维度: {dim}")
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(all_vectors).astype("float32"))
        faiss.write_index(index, INDEX_FILE)
        print(f"索引已保存: {INDEX_FILE}")
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata_store, f, ensure_ascii=False, indent=2)
        print(f"元数据已保存: {METADATA_FILE}")

    text_count = sum(1 for m in metadata_store if m["type"] == "text")
    image_count = sum(1 for m in metadata_store if m["type"] == "image")
    video_count = sum(1 for m in metadata_store if m["type"] == "video")
    print(f"\n完成! 文本:{text_count}, 图片:{image_count}, 视频:{video_count}")

    print("\n--- 知识库内容 ---")
    for m in metadata_store:
        print(f"\n[{m['id']:2d}] [{m['type']:5s}] {m.get('source', '')}")
        print(f"    {m['content'][:80]}")


if __name__ == "__main__":
    build_and_save()
