# 充电桩故障智能问答（RAG）

充电桩出现掉线、断电、插枪无响应后，往常依赖老师傅经验，定位耗时长、新员工上手慢。本系统对现场手册做细粒度解析与语义分块，构建 FAISS 向量索引；采用 BM25 与向量检索的混合策略，辅以重排序精炼结果，并用评测集量化证据覆盖，从而抑制幻觉，帮助新员工按步骤定位故障，缩短平均定位时间、压缩上岗培训周期。

## 知识源

- `data/charge/pdfs/` 充电桩故障排查知识手册 PDF
- `data/charge/docs/` 掉线通信、断电保护、枪头故障码 DOCX（保留表格）
- `data/charge/images/` 分流路径图 / 4G 六步图 / 枪口铭牌

## 快速开始

```bash
cd RAG-yym
python -m venv venv
venv\Scripts\activate
pip install -e . -r requirements.txt
```

将 `env` 复制为 `.env`，填入 `DASHSCOPE_API_KEY`。

首次构建索引并启动网页：

```bash
python build_charge_index.py
streamlit run app.py
```

浏览器打开 [http://localhost:8501](http://localhost:8501)。

评测（对比纯大模型与 RAG）：

```bash
python evaluate_charge_rag.py
```

强制重建索引：

```bash
python build_charge_index.py --force
```

## 实现对照

| 能力 | 位置 |
| --- | --- |
| 多源加载 PDF / DOCX / 图片说明 | `src/charge_rag.py` |
| 滑动窗口分块 | `sliding_window_split`（500 / 80） |
| FAISS 持久化热加载 | `data/charge/vector_store/` |
| 向量 + BM25 混合 + 重排序 | RRF 融合后 `gte-rerank` |
| LCEL 现场问答 | `build_chain` |
| Web 界面 | `app.py` |
| 评测集 | `data/charge/eval_set.json` |
