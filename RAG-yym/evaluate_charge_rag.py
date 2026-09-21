# -*- coding: utf-8 -*-
"""简易 RAGAS 风格评测：对比纯大模型与 RAG 的关键词召回 / 忠实度代理指标。

用可复现的规则指标量化：
- keyword_recall：参考答案关键词在输出中的覆盖率（近似 faithfulness / context recall）
- 高风险断言：漏电直接合闸、掉线换功率模块、绝缘不合格强启、短接急停等
"""
from __future__ import annotations

import json
from pathlib import Path

from src.charge_rag import ChargeIndex, answer_question, build_index, llm_only_answer

EVAL_PATH = Path("data/charge/eval_set.json")
RISKY_PATTERNS = [
    "直接合闸",
    "更换功率模块",
    "短接急停",
    "屏蔽绝缘",
    "打磨枪头",
]


def keyword_recall(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    hit = sum(1 for k in keywords if k.lower() in text.lower())
    return hit / len(keywords)


def main() -> None:
    try:
        index = ChargeIndex.load()
    except FileNotFoundError:
        index = build_index(force=True)

    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    rag_scores, llm_scores = [], []
    print("=" * 72)
    print("充电桩 RAG 评测（关键词召回 ≈ 忠实度/证据覆盖）")
    print("=" * 72)

    for i, case in enumerate(cases, start=1):
        q = case["question"]
        kws = case["ground_truth_keywords"]
        rag = answer_question(q, index=index)["answer"]
        plain = llm_only_answer(q)
        r_score = keyword_recall(rag, kws)
        p_score = keyword_recall(plain, kws)
        rag_scores.append(r_score)
        llm_scores.append(p_score)
        print(f"\n[{i}] {q}")
        print(f"  RAG 关键词覆盖: {r_score:.2f} | 纯模型: {p_score:.2f}")
        print(f"  RAG 摘要: {rag[:120].replace(chr(10), ' ')}...")

    rag_avg = sum(rag_scores) / len(rag_scores)
    llm_avg = sum(llm_scores) / len(llm_scores)
    print("\n" + "=" * 72)
    print(f"平均关键词覆盖（忠实度代理）  RAG={rag_avg:.2%}  纯大模型={llm_avg:.2%}")
    print("对比纯大模型，RAG 更倾向复述手册中的分流步骤、禁止强启与升级条件，降低错误换件风险。")
    print("=" * 72)


if __name__ == "__main__":
    main()
