# -*- coding: utf-8 -*-
"""构建充电桩知识库索引：加载 PDF/DOCX → 滑动窗口分块 → FAISS + BM25 持久化。"""
import argparse

from src.charge_rag import build_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建或重建充电桩故障 RAG 索引")
    parser.add_argument("--force", action="store_true", help="忽略已有索引，强制重建")
    args = parser.parse_args()
    index = build_index(force=args.force)
    print(f"完成。分块数: {len(index.chunks)}")
