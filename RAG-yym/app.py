# -*- coding: utf-8 -*-
"""充电桩故障智能问答（RAG）Web 界面。"""
import html
import re
from pathlib import Path

import streamlit as st

from src.charge_rag import ChargeIndex, answer_question, build_index, split_point_items

st.set_page_config(page_title="充电桩故障智能问答（RAG）", layout="wide", page_icon="⚡")

EXAMPLES = [
    "充电桩掉线了怎么查？平台显示离线，屏幕还能操作。",
    "充电桩突然断电黑屏，断路器跳了，能不能直接合闸？",
    "插上枪不充电是什么原因？交流桩屏幕显示已连接。",
    "故障码 E021 绝缘故障能不能复位继续充？",
    "枪头发烫发黑还能继续用吗？",
    "整站多台桩同时离线应该先查什么？",
]


def render_answer_html(text: str) -> str:
    """把【结论】【要点】等段落渲染成紧凑卡片，避免 pre-wrap 留下大块空白。"""
    text = (text or "").strip()
    parts = re.split(r"(?=【[^】]{1,8}】)", text)
    cards = []
    palette = {
        "结论": ("#1f4e79", "#eef4fb"),
        "要点": ("#2e75b6", "#f4f8fc"),
        "警示": ("#c45911", "#fff6ee"),
        "来源": ("#548235", "#f3f8ef"),
    }
    for part in parts:
        part = part.strip()
        if not part:
            continue
        match = re.match(r"【([^】]+)】\s*(.*)", part, flags=re.S)
        if match:
            title, body = match.group(1), match.group(2).strip()
        else:
            title, body = "回答", part
        color, bg = palette.get(title, ("#1f4e79", "#f6f8fa"))
        if title == "要点":
            items = split_point_items(body)
            body_html = "<br>".join(f"• {html.escape(item)}" for item in items) if items else html.escape(body)
        else:
            body_html = html.escape(body)
            body_html = re.sub(r"(?m)^-\s+", "• ", body_html)
            body_html = body_html.replace("\n", "<br>")
            body_html = re.sub(r"(<br>\s*){2,}", "<br>", body_html)
        cards.append(
            f"<div style='background:{bg};border-left:4px solid {color};padding:10px 14px;"
            f"border-radius:8px;margin:0 0 10px 0;line-height:1.55;font-size:15.5px;'>"
            f"<div style='color:{color};font-weight:700;margin:0 0 6px 0;'>{html.escape(title)}</div>"
            f"<div>{body_html}</div></div>"
        )
    return "".join(cards) or f"<div style='background:#f6f8fa;padding:12px;border-radius:8px;'>{html.escape(text)}</div>"


@st.cache_resource
def get_index():
    try:
        return ChargeIndex.load()
    except FileNotFoundError:
        return build_index(force=True)


st.markdown(
    """
<div style="background: linear-gradient(90deg, #1f4e79 0%, #2e75b6 100%); padding: 22px 24px; border-radius: 12px;">
  <h2 style="color: white; margin: 0;">充电桩故障智能问答（RAG）</h2>
  <div style="color: #e8f1fb; font-size: 15px; margin-top: 6px;">
    多源解析 · 滑动窗口分块 · FAISS + BM25 混合检索 · 重排序 · LCEL 现场排查
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.caption("面向运维新员工：掉线、断电、插枪无响应按手册定位，缩短平均故障定位时间，压缩上岗培训周期。")

with st.sidebar:
    st.header("现场设置")
    example = st.selectbox("示例问题", ["自定义"] + EXAMPLES)
    default_q = EXAMPLES[0] if example == "自定义" else example
    question = st.text_area("输入现场现象", default_q, height=140)
    submit = st.button("生成排查参考", type="primary", use_container_width=True)
    st.markdown("**知识源**")
    st.write("- PDF：充电桩故障排查知识手册")
    st.write("- DOCX：掉线通信 / 断电保护 / 枪头故障码")
    st.write("- 图片：分流路径、4G 六步、枪口铭牌")

st.markdown("### 排查回答")

if submit and question.strip():
    with st.spinner("正在混合检索并生成答案..."):
        try:
            index = get_index()
            result = answer_question(question.strip(), index=index)
            st.markdown("**助手回答**")
            st.markdown(render_answer_html(result["answer"]), unsafe_allow_html=True)
            st.markdown("**检索片段（重排序后）**")
            for i, hit in enumerate(result["hits"], start=1):
                loc = hit["source"]
                if hit.get("page"):
                    loc += f" · 第{hit['page']}页"
                with st.expander(f"{i}. {loc}  |  相关度 {hit['score']:.3f}"):
                    st.write(hit["text"])
                    img_path = (hit.get("metadata") or {}).get("image_path")
                    if img_path and Path(img_path).exists():
                        st.image(img_path, width="stretch")
        except Exception as exc:
            st.error(f"生成失败: {exc}")
else:
    st.info("请在左侧输入现场现象后点击【生成排查参考】。")
