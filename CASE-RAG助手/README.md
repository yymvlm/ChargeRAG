# 充电桩故障 RAG 助手（多模态）

充电桩掉线、断电、插枪无响应后，一线往往靠老师傅经验，定位慢、新员工难上手。本案例把三本现场手册和三张排查图做成统一向量库：DOCX 表格不丢故障码，图片走多模态 embedding。提问时可命中文字步骤，问“路径图/示意图”时附上对应图片，缩短平均定位时间、压缩培训周期。

语言：**Python**

## 知识库

| 文件 | 覆盖场景 |
| --- | --- |
| `charge_knowledge_base/充电桩掉线与通信故障排查.docx` | 心跳离线、4G/SIM/APN、整站交换机、错单 |
| `charge_knowledge_base/充电桩断电与电气保护手册.docx` | 黑屏、进线电压、漏电、绝缘、能否复位 |
| `charge_knowledge_base/充电枪启动故障与故障码速查.docx` | CP/BMS、过温烧蚀、故障码、15 分钟闭环 |
| `images/充电桩故障分流路径图.png` | 整站 / 单桩黑屏 / 通信或插枪 |
| `images/4G掉线排查示意图.png` | 网络灯到热点验证六步 |
| `images/充电枪接口与铭牌示意图.png` | 枪口定义、过温与换枪换车 |

## 运行

```bash
cd CASE-RAG助手
pip install -r requirements.txt
set DASHSCOPE_API_KEY=your_key
python 1-build_index.py
python 2-query.py
python 2-query.py 充电桩突然断电黑屏怎么办
```

默认会跑四组题：掉线、分流图、断电合闸、插枪+铭牌图。

## 流程

```
DOCX（段落+表格 Markdown） + PNG
  → tongyi-embedding-vision-plus
  → FAISS IndexFlatL2
  → 文本 Top-3 注入提示词
  → 含“图/示意图”时附加距离 < 3 的图片
  → qwen-flash 按【结论】【排查步骤】【注意】【依据】作答
```
