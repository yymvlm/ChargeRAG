# 模块说明（充电桩 RAG）

- `src/charge_rag.py`：多源加载、滑动窗口分块、FAISS + BM25、重排序、LCEL 问答
- `build_charge_index.py`：构建或热加载索引
- `app.py`：Streamlit 现场排查界面
- `evaluate_charge_rag.py`：对比纯大模型与 RAG 的证据覆盖
- `data/charge/`：PDF / DOCX / 图片知识源、评测集、向量库
