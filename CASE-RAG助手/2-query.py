# -*- coding: utf-8 -*-
"""
充电桩 RAG 助手 - 查询处理

加载 FAISS 索引，打印相似度排名；当问题包含“图/示意图/铭牌”等意图时附加匹配图片，
帮助新员工按手册定位掉线、断电、插枪无响应。
"""
import os
import sys
import json
import numpy as np
import faiss
import dashscope
from http import HTTPStatus
from openai import OpenAI

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("错误：请设置 'DASHSCOPE_API_KEY' 环境变量。")

dashscope.api_key = DASHSCOPE_API_KEY

client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
MULTIMODAL_EMBEDDING_MODEL = "tongyi-embedding-vision-plus"
INDEX_FILE = "charge_index.faiss"
METADATA_FILE = "charge_metadata.json"

IMAGE_KEYWORDS = [
    "图片",
    "海报",
    "照片",
    "看看",
    "长什么样",
    "图",
    "路径图",
    "示意图",
    "铭牌",
    "分流",
    "接口",
]
VIDEO_KEYWORDS = ["视频", "录像", "影片", "看一下", "播放"]
MEDIA_DISTANCE_THRESHOLD = 3.0

SYSTEM_PROMPT = (
    "你是充电桩运维现场助手。只根据检索到的手册与示意图回答，禁止编造手册中没有的电压、电阻、故障码。"
    "面向新员工，按【结论】【排查步骤】【注意】【依据】作答。"
    "资料不足时写明手册未覆盖，请升级二线。"
)


def load_index():
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"已加载索引: {index.ntotal} 条记录")
    return index, metadata


def get_text_embedding(text):
    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL,
        input=[{"text": text}],
    )
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"Embedding失败: {resp.message}")
    return resp.output["embeddings"][0]["embedding"]


def distance_to_similarity(distance):
    return 1 / (1 + distance)


def detect_media_intent(query):
    query_lower = query.lower()
    want_image = any(kw in query_lower for kw in IMAGE_KEYWORDS)
    want_video = any(kw in query_lower for kw in VIDEO_KEYWORDS)
    return want_image, want_video


def search_with_details(query, index, metadata):
    print(f"\n{'=' * 60}")
    print(f"Query: {query}")
    print("=" * 60)

    query_vec = np.array([get_text_embedding(query)]).astype("float32")
    distances, indices = index.search(query_vec, index.ntotal)

    print("\n相似度排名 (越大越相似):")
    print("-" * 80)
    print(f"{'排名':4s} {'ID':4s} {'类型':6s} {'相似度':8s} {'距离':8s} 内容")
    print("-" * 80)

    results = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        if idx == -1:
            continue
        m = metadata[idx]
        sim = distance_to_similarity(dist)
        content_preview = m["content"][:45].replace("\n", " ")
        type_tag = m["type"]
        marker = ""
        if type_tag == "image":
            marker = " <-- 图片"
        elif type_tag == "video":
            marker = " <-- 视频"
        print(
            f"{rank + 1:4d} {idx:4d} [{type_tag:5s}] {sim:6.4f}  {dist:8.4f}  {content_preview}...{marker}"
        )
        results.append({"idx": idx, "distance": dist, "similarity": sim, "metadata": m})

    return results


def rag_ask(query, index, metadata, k=3):
    results = search_with_details(query, index, metadata)

    want_image, want_video = detect_media_intent(query)
    print(f"\n意图检测: 需要图片={want_image}, 需要视频={want_video}")

    top_results = [r for r in results if r["metadata"]["type"] == "text"][:k]

    matched_image = None
    if want_image:
        image_results = [
            r
            for r in results
            if r["metadata"]["type"] == "image" and r["distance"] < MEDIA_DISTANCE_THRESHOLD
        ]
        if image_results:
            image_results.sort(key=lambda x: x["distance"])
            matched_image = image_results[0]
            print(
                f"  -> 匹配到图片: {matched_image['metadata']['path']} "
                f"(距离: {matched_image['distance']:.4f}, 相似度: {matched_image['similarity']:.4f})"
            )

    matched_video = None
    if want_video:
        video_results = [
            r
            for r in results
            if r["metadata"]["type"] == "video" and r["distance"] < MEDIA_DISTANCE_THRESHOLD
        ]
        if video_results:
            video_results.sort(key=lambda x: x["distance"])
            matched_video = video_results[0]
            print(
                f"  -> 匹配到视频: {matched_video['metadata']['url']} "
                f"(距离: {matched_video['distance']:.4f}, 相似度: {matched_video['similarity']:.4f})"
            )

    print(f"\n选取Top-{k}文本构建Prompt:")
    for r in top_results:
        print(f"  - {r['metadata']['content'][:50]}... (相似度: {r['similarity']:.4f})")

    context_str = ""
    for i, r in enumerate(top_results):
        m = r["metadata"]
        context_str += (
            f"背景知识 {i + 1} (来源: {m['source']}, 相似度: {r['similarity']:.4f}):\n{m['content']}\n\n"
        )

    prompt = f"""你是充电桩运维现场助手。请仅根据下列检索资料回答，标注来源要点，不要编造未出现的电压、电阻、故障码。
若资料不足，明确说明并建议升级二线。输出【结论】【排查步骤】【注意】【依据】，帮助新员工缩短定位时间。

[背景知识]
{context_str}
[现场问题]
{query}
"""

    print("\n调用LLM生成答案...")
    completion = client.chat.completions.create(
        model="qwen-flash",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    answer = completion.choices[0].message.content

    if matched_image:
        answer += f"\n\n[相关图片]: {matched_image['metadata']['path']}"
    if matched_video:
        answer += f"\n\n[相关视频]: {matched_video['metadata']['url']}"

    print(f"\n最终答案:\n{answer}")
    return answer


SAMPLE_QUERIES = [
    ("测试1: 掉线 - 平台离线屏幕仍可操作", "充电桩掉线了怎么查？平台显示离线，屏幕还能操作。"),
    ("测试2: 图片 - 故障分流路径图", "请给我看看充电桩故障分流路径图"),
    ("测试3: 断电 - 黑屏跳闸", "充电桩突然断电黑屏，断路器跳了，能不能直接合闸？"),
    ("测试4: 插枪无响应 + 铭牌图", "插上枪不充电是什么原因？充电枪接口示意图在哪？"),
]


if __name__ == "__main__":
    index, metadata = load_index()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        rag_ask(" ".join(args).strip(), index, metadata, k=3)
    else:
        for title, q in SAMPLE_QUERIES:
            print("\n" + "=" * 60)
            print(title)
            rag_ask(q, index, metadata, k=3)
