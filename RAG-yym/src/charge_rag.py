# -*- coding: utf-8 -*-
"""充电桩故障智能问答 RAG。

多源加载 PDF / DOCX / 排查图说明，滑动窗口分块，FAISS 向量召回 + BM25 混合检索，
DashScope 重排序，LCEL 编排现场排查问答，索引可持久化热加载。
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai import OpenAI
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "charge"
PDF_DIR = DATA_DIR / "pdfs"
DOC_DIR = DATA_DIR / "docs"
IMAGE_DIR = DATA_DIR / "images"
INDEX_DIR = DATA_DIR / "vector_store"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
FAISS_PATH = INDEX_DIR / "index.faiss"
BM25_PATH = INDEX_DIR / "bm25.pkl"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
CHAT_MODEL = "qwen-turbo"
RERANK_MODEL = "gte-rerank"
VECTOR_TOP_K = 20
BM25_TOP_K = 20
RERANK_TOP_N = 6


def get_api_key() -> str:
    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise ValueError("请设置环境变量 DASHSCOPE_API_KEY，或写入项目根目录 .env")
    return key


def dashscope_client() -> OpenAI:
    return OpenAI(
        api_key=get_api_key(),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    han = re.findall(r"[\u4e00-\u9fff]", text)
    latin = re.findall(r"[a-z0-9]+", text)
    return han + latin


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    source_type: str
    page: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "source_type": self.source_type,
            "page": self.page,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        return cls(
            id=data["id"],
            text=data["text"],
            source=data["source"],
            source_type=data.get("source_type", "unknown"),
            page=data.get("page"),
            metadata=data.get("metadata") or {},
        )


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf_documents(pdf_dir: Path = PDF_DIR) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    if not pdf_dir.exists():
        return docs
    for path in sorted(pdf_dir.glob("*.pdf")):
        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text() or "")
            if not text.strip():
                continue
            docs.append({"text": text, "source": path.name, "source_type": "pdf", "page": i})
    return docs


def load_docx_documents(doc_dir: Path = DOC_DIR) -> List[Dict[str, Any]]:
    from docx import Document as DocxDocument

    docs: List[Dict[str, Any]] = []
    if not doc_dir.exists():
        return docs
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    for path in sorted(doc_dir.glob("*.docx")):
        if path.name.startswith("~$"):
            continue
        document = DocxDocument(str(path))
        parts: List[str] = []
        for element in document.element.body:
            tag = element.tag.split("}")[-1]
            if tag == "p":
                paragraph_text = "".join(
                    (node.text or "") for node in element.findall(".//w:t", ns)
                ).strip()
                if paragraph_text:
                    parts.append(paragraph_text)
            elif tag == "tbl":
                table = next((t for t in document.tables if t._element is element), None)
                if table is None or not table.rows:
                    continue
                header = [cell.text.strip() for cell in table.rows[0].cells]
                lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                for row in table.rows[1:]:
                    lines.append("| " + " | ".join(c.text.strip() for c in row.cells) + " |")
                parts.append("\n".join(lines))
        text = clean_text("\n".join(parts))
        if text.strip():
            docs.append({"text": text, "source": path.name, "source_type": "docx", "page": None})
    return docs


def load_image_captions(image_dir: Path = IMAGE_DIR) -> List[Dict[str, Any]]:
    captions = {
        "充电桩故障分流路径图.png": (
            "充电桩故障分流路径图：接到工单先问四件事（黑屏、平台离线、仅一辆车失败、同站其他桩）。"
            "整站同时故障查配电室与交换机；单桩黑屏查供电、急停、进线电压与漏电；"
            "单桩有屏或平台离线查 4G/SIM/APN 或插枪导引。"
            "十五分钟无法定位则停用拍照升级二线，禁止连续更换两块以上板件。"
        ),
        "4G掉线排查示意图.png": (
            "4G 掉线排查示意图：看网络灯（常亮已注册、慢闪搜网、快闪 SIM 异常、熄灭未上电）→"
            "检查 SIM 松动欠费锁卡 → 天线与金属柜屏蔽 → 核对 APN、平台 URL 与密钥 →"
            "手机热点验证（能上线则原链路故障）→ 重启或更换通信板并回写桩号。"
            "整站掉线先查交换机，不要更换功率模块。掉线不等于断电。"
        ),
        "充电枪接口与铭牌示意图.png": (
            "充电枪接口与铭牌示意图：直流枪含 DC+、DC-、PE、CC1、S+/S-、A+/A-。"
            "未插枪 CP 对 PE 为 12 V，插枪后应为 9 V 或 6 V。"
            "枪头超过 90℃ 降额，超过 110℃ 停充；烧蚀黑斑必须整枪更换。"
            "E051 枪头过温换枪；E031 CP 异常查 CC/CP；E030 BMS 超时先换枪再换车。"
        ),
    }
    docs: List[Dict[str, Any]] = []
    if not image_dir.exists():
        return docs
    for path in sorted(image_dir.glob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        caption = captions.get(path.name, f"充电桩现场示意图：{path.stem}")
        docs.append(
            {
                "text": f"[图片] {path.name}\n{caption}",
                "source": path.name,
                "source_type": "image",
                "page": None,
                "metadata": {"image_path": str(path)},
            }
        )
    return docs


def load_all_documents() -> List[Dict[str, Any]]:
    docs = load_pdf_documents() + load_docx_documents() + load_image_captions()
    if not docs:
        raise FileNotFoundError(f"未在 {DATA_DIR} 找到 PDF/DOCX/图片知识源")
    return docs


def sliding_window_split(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind("。"), window.rfind("\n"), window.rfind("；"))
            if cut >= chunk_size * 0.6:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def chunk_documents(docs: Sequence[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for doc in docs:
        pieces = sliding_window_split(doc["text"])
        for i, piece in enumerate(pieces):
            cid = sha1_text(f"{doc['source']}|{doc.get('page')}|{i}|{piece[:40]}")
            chunks.append(
                Chunk(
                    id=cid,
                    text=piece,
                    source=doc["source"],
                    source_type=doc["source_type"],
                    page=doc.get("page"),
                    metadata=doc.get("metadata") or {},
                )
            )
    return chunks


def embed_texts(texts: Sequence[str], batch_size: int = 10) -> np.ndarray:
    import dashscope
    from dashscope import TextEmbedding

    dashscope.api_key = get_api_key()
    vectors: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = [t[:2048] for t in texts[i : i + batch_size]]
        resp = TextEmbedding.call(model=TextEmbedding.Models.text_embedding_v1, input=batch)
        if "output" not in resp:
            raise RuntimeError(f"Embedding 失败: {resp}")
        output = resp["output"]
        if "embeddings" in output:
            ordered = sorted(output["embeddings"], key=lambda x: x.get("text_index", 0))
            for item in ordered:
                vectors.append(item["embedding"])
        elif "embedding" in output:
            vectors.append(output["embedding"])
        else:
            raise RuntimeError(f"Embedding 返回格式异常: {resp}")
    return np.array(vectors, dtype=np.float32)


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


class ChargeIndex:
    def __init__(self, chunks: List[Chunk], embeddings: np.ndarray, bm25: BM25Okapi):
        self.chunks = chunks
        self.embeddings = l2_normalize(embeddings.astype(np.float32))
        self.bm25 = bm25
        dim = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.embeddings)

    def save(self, index_dir: Path = INDEX_DIR) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(index_dir / "index.faiss"))
        with open(index_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in self.chunks], f, ensure_ascii=False, indent=2)
        with open(index_dir / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)
        np.save(index_dir / "embeddings.npy", self.embeddings)

    @classmethod
    def load(cls, index_dir: Path = INDEX_DIR) -> "ChargeIndex":
        chunks_path = index_dir / "chunks.json"
        faiss_path = index_dir / "index.faiss"
        bm25_path = index_dir / "bm25.pkl"
        emb_path = index_dir / "embeddings.npy"
        if not chunks_path.exists() or not faiss_path.exists() or not emb_path.exists():
            raise FileNotFoundError("未找到向量索引，请先运行 python build_charge_index.py")
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = [Chunk.from_dict(x) for x in json.load(f)]
        embeddings = np.load(emb_path)
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)
        obj = cls.__new__(cls)
        obj.chunks = chunks
        obj.embeddings = embeddings.astype(np.float32)
        obj.bm25 = bm25
        obj.faiss_index = faiss.read_index(str(faiss_path))
        return obj


def build_index(force: bool = False) -> ChargeIndex:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if not force and FAISS_PATH.exists() and CHUNKS_PATH.exists() and BM25_PATH.exists():
        print(f"热加载已有索引: {INDEX_DIR}")
        return ChargeIndex.load(INDEX_DIR)

    print("加载知识源（PDF / DOCX / 图片说明）...")
    docs = load_all_documents()
    print(f"  文档页/篇: {len(docs)}")
    chunks = chunk_documents(docs)
    print(f"  滑动窗口分块: {len(chunks)} 块 (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print("生成向量嵌入...")
    embeddings = embed_texts([c.text for c in chunks])
    bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
    index = ChargeIndex(chunks, embeddings, bm25)
    index.save(INDEX_DIR)
    print(f"索引已保存: {INDEX_DIR}")
    return index


def rrf_fuse(ranked_lists: Sequence[Sequence[int]], k: int = 60) -> List[Tuple[int, float]]:
    scores: Dict[int, float] = {}
    for ranking in ranked_lists:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def dashscope_rerank(query: str, documents: List[str], top_n: int) -> List[Tuple[int, float]]:
    import dashscope

    dashscope.api_key = get_api_key()
    try:
        resp = dashscope.TextReRank.call(
            model=RERANK_MODEL,
            query=query,
            documents=documents,
            top_n=min(top_n, len(documents)),
            return_documents=False,
        )
        results = None
        if isinstance(resp, dict):
            results = (resp.get("output") or {}).get("results")
        else:
            output = getattr(resp, "output", None)
            results = getattr(output, "results", None) if output is not None else None
            if results is None and isinstance(output, dict):
                results = output.get("results")
        if results:
            ranked = []
            for item in results:
                if isinstance(item, dict):
                    ranked.append((int(item["index"]), float(item.get("relevance_score", 0.0))))
                else:
                    ranked.append((int(item.index), float(getattr(item, "relevance_score", 0.0))))
            return ranked[:top_n]
    except Exception as exc:
        print(f"gte-rerank 不可用，改用词重叠重排: {exc}")

    q_tokens = set(tokenize(query))
    scored = []
    for i, doc in enumerate(documents):
        overlap = len(q_tokens & set(tokenize(doc)))
        scored.append((i, overlap / (len(q_tokens) + 1e-6)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


def hybrid_retrieve(
    query: str,
    index: ChargeIndex,
    vector_k: int = VECTOR_TOP_K,
    bm25_k: int = BM25_TOP_K,
    top_n: int = RERANK_TOP_N,
) -> List[Dict[str, Any]]:
    q_vec = l2_normalize(embed_texts([query]))
    _scores, indices = index.faiss_index.search(q_vec, min(vector_k, len(index.chunks)))
    vector_ids = [int(i) for i in indices[0] if i >= 0]
    bm25_scores = index.bm25.get_scores(tokenize(query))
    bm25_ids = list(np.argsort(bm25_scores)[::-1][:bm25_k])
    fused = rrf_fuse([vector_ids, bm25_ids])
    candidate_ids = [idx for idx, _ in fused[: max(vector_k, bm25_k)]]
    documents = [index.chunks[i].text for i in candidate_ids]
    reranked = dashscope_rerank(query, documents, top_n=top_n)
    results: List[Dict[str, Any]] = []
    for local_i, score in reranked:
        chunk = index.chunks[candidate_ids[local_i]]
        results.append(
            {
                "text": chunk.text,
                "source": chunk.source,
                "source_type": chunk.source_type,
                "page": chunk.page,
                "score": float(score),
                "metadata": chunk.metadata,
            }
        )
    return results


def split_point_items(body: str) -> List[str]:
    """把【要点】正文拆成一条一条，兼容 - / 1. / 1、 / 同一行粘连。"""
    body = (body or "").strip()
    if not body:
        return []
    body = body.replace("•", "-").replace("●", "-").replace("▪", "-")
    body = re.sub(r"(?<!^)(?<!\n)\s*(?=\d+[\.、．]\s)", "\n", body)
    body = re.sub(r"(?<!^)(?<!\n)\s*(?=-\s)", "\n", body)
    items: List[str] = []
    for line in body.split("\n"):
        line = line.strip()
        if not line:
            continue
        dashed = [p.strip(" ；;。") for p in re.split(r"(?:^|\s+)-\s+", line) if p.strip(" ；;。")]
        if len(dashed) > 1:
            items.extend(dashed)
            continue
        numbered = [p.strip(" ；;。") for p in re.split(r"(?:^|\s+)\d+[\.、．]\s+", line) if p.strip(" ；;。")]
        if len(numbered) > 1:
            items.extend(numbered)
            continue
        items.append(re.sub(r"^(?:-\s+|\d+[\.、．]\s+)", "", line).strip(" ；;"))
    return [x for x in items if x]


def compact_answer(text: str) -> str:
    """去掉模型输出里多余空行、标题井号和首尾空白，并把要点拆成独立行。"""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n【", "\n\n【", text)
    text = re.sub(r"(?<!^)(?<![\n])\s+-\s+", "\n- ", text)
    text = re.sub(r"(【要点】)\s*-\s+", r"\1\n- ", text)

    def _rewrite_points(match: re.Match) -> str:
        items = split_point_items(match.group(1))
        if not items:
            return match.group(0).rstrip()
        return "【要点】\n" + "\n".join(f"- {item}" for item in items)

    text = re.sub(r"【要点】(.*?)(?=\n【|\Z)", _rewrite_points, text, flags=re.S)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def format_context(hits: Sequence[Dict[str, Any]]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        loc = hit["source"]
        if hit.get("page"):
            loc += f" 第{hit['page']}页"
        blocks.append(f"[资料{i} | {loc} | 相关度{hit['score']:.3f}]\n{hit['text']}")
    return "\n\n".join(blocks)


SYSTEM_PROMPT = """你是充电桩运维现场助手。只根据【检索资料】作答，禁止编造未出现的电压、电阻、故障码或处理步骤。

输出必须紧凑、信息密度高，禁止空行堆砌、禁止长篇铺垫。严格按下面四段输出，每段之间只空一行，段内用短句或分号连接，不要用 Markdown 标题（#）：

【结论】一句话判断根因类别（供电 / 通信 / 枪与车辆 / 保护）及能否现场处理。
【要点】3–6 条，每条必须单独换行并以「- 」开头，禁止把多条写在同一行；写清先后顺序、测量判据、换件条件。
【警示】禁止强启、漏电连合、短接急停、屏蔽绝缘、连续换板等风险，用分号连接成一段；没有则写「资料未见额外警示」。
【来源】写成「文档名（页码）」列表，不要写「资料1」这种编号。

资料不足时在【结论】中明确说手册未覆盖、请升级二线，不要补全幻觉。面向新员工，步骤必须可执行。"""

USER_PROMPT = """【检索资料】
{context}

【现场问题】
{question}

按【结论】【要点】【警示】【来源】四段紧凑作答，不要多余空行。"""


def build_chain(index: ChargeIndex):
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("user", USER_PROMPT)]
    )
    client = dashscope_client()

    def call_llm(payload: Dict[str, str]) -> str:
        messages = prompt.format_messages(context=payload["context"], question=payload["question"])
        openai_messages = []
        for m in messages:
            role = "user" if m.type in ("human", "user") else ("system" if m.type == "system" else m.type)
            openai_messages.append({"role": role, "content": m.content})
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.1,
            messages=openai_messages,
        )
        return compact_answer(resp.choices[0].message.content or "")

    def retrieve_step(question: str) -> Dict[str, Any]:
        hits = hybrid_retrieve(question, index)
        return {"question": question, "hits": hits, "context": format_context(hits)}

    return RunnableLambda(retrieve_step) | RunnablePassthrough.assign(answer=RunnableLambda(call_llm))


def answer_question(question: str, index: Optional[ChargeIndex] = None) -> Dict[str, Any]:
    index = index or ChargeIndex.load()
    return build_chain(index).invoke(question)


def llm_only_answer(question: str) -> str:
    client = dashscope_client()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": "你是充电桩运维助手，直接根据自身知识回答，不必引用外部资料。",
            },
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content
