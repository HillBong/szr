# 智能医疗数字人系统（本地大模型 + 多模态交互）

本项目是一个结合本地大语言模型（Qwen2.5）、RAG 问答系统、多模态情绪识别、语音合成与数字人视频播报的 AI 医疗问答系统，具备医疗问答、情绪识别、语音播报、视频交互等功能，适合部署为线下医疗服务助手或医疗客服交互系统。

------

## 📌 项目亮点

- 🧠 **本地部署大语言模型**：基于 Qwen2.5-7B-Instruct 微调模型，无需外部API，保障隐私。
- 📚 **RAG语义增强问答**：文档自动索引，结合向量检索 + BM25 + rerank，多策略融合。
- 😃 **情绪识别**：识别用户输入情绪（如沮丧、开心、愤怒），用于调节语音风格。
- 🗣️ **数字人语音合成**：接入 Azure Avatar SDK，支持粤语、普通话、台湾腔、陕西话等。
- 🎥 **虚拟人交互视频**：通过 WebRTC 实现视频数字人播报、语音与图像交互。

------

## 🧩 系统架构

```mermaid
graph TD
    A[用户网页交互] -->|文本输入| B[FastAPI 服务接口]
    B -->|调用| C[RAG 问答引擎 (LangChain + Qwen)]
    B -->|调用| D[情绪识别模块]
    B -->|返回| E[前端网页]
    E -->|播报| F[Azure Avatar SDK]
    C -->|读取| G[Chroma 知识库]
    G -->|索引管理| H[SQLRecordManager.sqlite]
```

------

## 🛠️ 使用技术

| 技术模块   | 技术栈                                  |
| ---------- | --------------------------------------- |
| 后端框架   | FastAPI                                 |
| 大语言模型 | Qwen2.5-7B-Instruct (LoRA 微调可选)     |
| 向量检索   | Chroma + HuggingFace BGE Embedding      |
| 检索融合   | BM25 + Embedding + CrossEncoderReranker |
| 文档处理   | LangChain Loader + Splitter             |
| 情绪识别   | Prompt + LLM 分类器                     |
| 前端交互   | HTML + JS + Azure Avatar SDK            |
| 语音播报   | Azure Neural TTS (多语言多语调)         |
| 视频传输   | WebRTC TURN Server                      |

------

## 🚀 快速部署说明

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 本地部署大语言模型 API

确保 Qwen2.5 模型已使用如 LLaMA-Factory 微调并运行 Web 接口服务：

```bash
# 示例 Flask/FastAPI 服务监听在 http://localhost:8001/qwen
```

设置环境变量：

```bash
export LOCAL_LLM_API=http://localhost:8001/qwen
```

### 3. 初始化知识库索引

将医疗文档放入 `./chroma/knowledge/` 目录下，首次运行自动构建 Chroma 向量索引 + SQLite 索引记录。

### 4. 启动服务

```bash
python server_qwen.py
```

------

## 🧪 接口说明

### POST /chat

- 参数：`query`（用户问题字符串）
- 返回：

```json
[
  {
    "msg": "这是模型给出的医疗回答",
    "qingxu": "friendly"
  }
]
```

------

## 🖥️ 前端使用说明

前端文件为 `index_medical.html`，直接用浏览器打开即可。

注意：

- 替换 Azure Speech Key
- 浏览器需能访问 Azure 相关 CDN 和 relay token 服务

------

## 📁 项目结构

```
.
├── my_local_qwen.py
├── RAG_answer_qwen.py
├── server_qwen.py
├── index_medical.html
├── turnserver.conf
├── redis.conf
├── chroma/knowledge/
├── record_manager_cache.sql
```

------

## 📜 License

本项目仅用于学习与研究目的，如需商业部署，请遵守相关模型与服务的开源协议。

------

## 🧠 改进与反思

尽管本系统功能完整、模块清晰，但在实际应用与扩展性方面仍有若干可优化点：

### ✅ 可优化项

1. **知识库更新机制**  
   当前系统需要重启服务才能加载新的文档，建议引入用户上传接口和动态索引更新机制，提升维护效率。
2. **检索器性能与召回率优化**  
   BM25 和向量融合效果良好，但在多文档复杂语义下仍存在召回误差，可探索 Faiss/HNSW、Hybrid Search、多轮 rerank 策略等。
3. **情绪识别模块强化**  
   当前使用 prompt + LLM 推理判断情绪，建议引入训练好的情绪分类器（如 BERT、RoBERTa）提升稳定性与实时性。
4. **前端交互友好性**  
   Web 页面尚不具备历史对话记录、多轮问答、多模态输入等增强功能，建议使用 Vue/React 重构 UI 体验。
5. **部署与性能**  
   模型调用为同步阻塞方式，建议结合队列系统（如 Celery）实现任务异步化，提高并发能力。

### 🌱 拓展建议

- 接入语音识别（如 Azure STT），实现语音问答闭环。
- 加入用户权限和对话记录系统，实现个性化推荐与历史追溯。
- 部署 Docker 容器或使用 K8s 实现多节点横向扩展。

