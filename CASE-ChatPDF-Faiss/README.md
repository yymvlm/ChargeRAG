# 充电桩故障智能问答（ChatPDF + FAISS）

充电桩出现掉线、断电、插枪无响应等问题后，往常依赖老师傅经验，定位耗时长、新员工上手慢。本案例把《充电桩故障排查知识手册》做成可检索知识库：PDF 按字符记录页码、切块写入 FAISS，现场提问后只根据手册作答，并回源页码，缩短平均定位时间、压缩上岗培训周期。

语言：**Python**

## 能解决什么

| 现场现象 | 手册路径 | 新员工得到什么 |
| --- | --- | --- |
| 平台离线、屏幕仍可操作 | 掉线：4G / SIM / APN / 交换机 | 先看同站在线率，不误换功率模块 |
| 整桩黑屏、断路器跳闸 | 断电：进线电压、漏电、SPD | 漏电禁止连合，给出停用条件 |
| 插枪无电流 | 导引 CP、电子锁、BMS 握手 | 换枪/换车交叉验证 |
| 故障码 E021 / E040 / E051 | 保护与故障码表 | 是否允许复位、是否升级二线 |

## 运行

```bash
cd CASE-ChatPDF-Faiss
pip install -r requirements.txt
set DASHSCOPE_API_KEY=your_key   # Windows
python chatpdf-faiss.py
python chatpdf-faiss.py 充电桩突然断电黑屏怎么办
python chatpdf-faiss.py --force  # 手册更新后强制重建索引
```

已有 `vector_db/index.faiss` 时会热加载，跳过重新向量化。

## 流程

```
手册 PDF
  → 逐页抽文本 + 字符级页码
  → RecursiveCharacterTextSplitter（800 / 160）
  → DashScope embedding + FAISS
  → 现场问题相似度检索 Top-8
  → Tongyi 按【结论】【排查步骤】【注意】【依据】作答
  → 打印来源页码
```

## 文件

| 文件 | 作用 |
| --- | --- |
| `充电桩故障排查知识手册.pdf` | 站端电气、通信、枪头、保护、故障码 |
| `chatpdf-faiss.py` | 解析 / 建库 / 热加载 / 问答 |
| `vector_db/` | FAISS 与页码映射持久化 |
