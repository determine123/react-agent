# 🤖 Agent Project

**ReAct Agent + RAG + Memory + 10 Tools — 生产级智能 Agent 系统**

一个基于 ReAct（Reasoning + Acting）范式的多工具智能 Agent 框架，集成了 RAG 检索增强生成、双层记忆机制、LangGraph 工作流编排，以及完整的评估、API 和 Web UI。

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-green)](https://deepseek.com)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange)](https://langchain.com/langgraph)
[![ChromaDB](https://img.shields.io/badge/Vector-ChromaDB-purple)](https://trychroma.com)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-teal)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Deploy-Docker-blue)](https://docker.com)

---

## 📋 目录

- [架构总览](#架构总览)
- [快速开始](#快速开始)
- [运行模式](#运行模式)
- [Agent 工具集](#agent-工具集)
- [RAG 检索增强生成](#rag-检索增强生成)
- [记忆系统](#记忆系统)
- [评估体系](#评估体系)
- [API 服务](#api-服务)
- [Web 界面](#web-界面)
- [Docker 部署](#docker-部署)
- [配置说明](#配置说明)
- [项目结构](#项目结构)

---

## 架构总览

```
                        ┌──────────────────────────┐
                        │       Gradio Web UI       │
                        │     (http://:7860)        │
                        └────────────┬─────────────┘
                                     │
                        ┌────────────▼─────────────┐
                        │      FastAPI Server       │
                        │   POST /chat  POST /rag   │
                        └────────────┬─────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
┌───────▼───────┐          ┌────────▼────────┐          ┌───────▼───────┐
│  ReAct Agent  │          │   RAG Pipeline  │          │   LangGraph   │
│  Loop Engine  │◄────────►│ Load→Split→Store│          │  Workflow     │
│ Reason → Act  │          │ Retrieve → Gen  │          │ Agent→Tool→.. │
└───────┬───────┘          └────────┬────────┘          └───────────────┘
        │                           │
        │              ┌────────────┼────────────┐
        │              │            │            │
┌───────▼───────┐ ┌───▼────┐ ┌────▼─────┐ ┌───▼──────┐
│  Tool Registry│ │Memory  │ │ ChromaDB │ │Evaluator │
│  10+ tools    │ │2-tier  │ │  Vector  │ │RAGAS/Base│
└───────────────┘ └────────┘ └──────────┘ └──────────┘
```

**核心设计理念**：感知 → 推理 → 行动 → 观察 → 记忆，形成完整的 Agent 闭环。

---

## 快速开始

### 前提条件

- Python 3.10+
- DeepSeek API Key（或其他 OpenAI 兼容 API）

### 安装

```bash
# 1. 克隆项目
cd D:\Projects\agent-project

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量（编辑 .env）
# LLM_API_KEY=sk-...        # DeepSeek API Key
# TAVILY_API_KEY=tvly-...   # Tavily 搜索 Key (https://tavily.com)

# 4. 启动交互式聊天
python main.py
```

---

## 运行模式

`main.py` 支持 **7 种运行模式**，通过命令行参数切换：

| 命令 | 模式 | 说明 |
|------|------|------|
| `python main.py` | 交互式聊天 | ReAct Agent + 10工具 + 记忆 |
| `python main.py --stateless` | 无状态聊天 | 每轮清空上下文 |
| `python main.py --rag` | RAG 问答 | 基于索引文档的检索增强生成 |
| `python main.py --index ./docs` | 文档索引 | 将目录中的文档向量化存储 |
| `python main.py --eval` | 快速评估 | 14 条测试用例，子串匹配 |
| `python main.py --eval --ragas` | RAGAS 评估 | 语义级评估（faithfulness/relevancy/...）|
| `python main.py --api` | API 服务 | 启动 FastAPI 服务器 (:8000) |
| `python main.py --ui` | Web 界面 | 启动 Gradio 聊天界面 (:7860) |

---

## Agent 工具集

Agent 注册了 **10 个工具**，可在对话中自动调用：

| 工具 | 函数名 | 功能描述 |
|------|--------|----------|
| 🔢 **计算器** | `calculate` | AST 安全数学计算（支持 +-*/%\*\*pi/e/tau）|
| 🌐 **网络搜索** | `web_search` | Tavily API 真实搜索，无 Key 时优雅降级 |
| 🕐 **当前时间** | `get_current_time` | 获取当前日期时间，支持时区偏移 |
| 📅 **日期间隔** | `days_between` | 计算两个日期之间的天数 |
| 🌤️ **天气查询** | `get_weather` | 通过搜索 API 获取城市天气信息 |
| 📄 **网页抓取** | `fetch_page` | 抓取网页文本内容（无需额外依赖）|
| 📝 **写笔记** | `write_note` | 保存内容到本地文件 |
| 📖 **读笔记** | `read_note` | 读取之前保存的笔记 |
| 📋 **列笔记** | `list_notes` | 列出所有保存的笔记文件 |

**扩展工具**：在 `tools/registry.py` 中调用 `self.register()` 即可添加新工具。

---

## RAG 检索增强生成

### 流程

```
文档目录 → 加载器(PDF/MD/TXT) → 分割器(递归+重叠) → 向量存储(ChromaDB) → 检索 → LLM 生成
```

### 支持的 Embedding 模型

| 模型 | 配置值 | 说明 |
|------|--------|------|
| ChromaDB 默认 ONNX | `chroma-default` | 零依赖，内置 all-MiniLM-L6-v2 |
| OpenAI Embedding | `text-embedding-3-small` | 需 API Key |
| 中文 BGE 模型 | `BAAI/bge-base-zh` | 需 `pip install sentence-transformers` |

### 使用示例

```bash
# 索引文档
python main.py --index ./docs

# RAG 问答
python main.py --rag
# Question: 什么是 RAG 技术？
# Answer: 根据文档，RAG（检索增强生成）是一种...

# 编程调用
from rag.qa import RAGQA
from config import load_config

qa = RAGQA(load_config())
qa.index_documents("./docs")
result = qa.answer("你的问题", return_sources=True)
print(result["answer"])
print(result["sources"])
```

---

## 记忆系统

双层级记忆架构，确保会话连贯性和跨会话知识保留：

| 层级 | 实现 | 用途 |
|------|------|------|
| **短期记忆** | `collections.deque` 滑动窗口（默认 20 轮）| 当前会话上下文 |
| **长期记忆** | JSON Lines 文件持久化（`logs/memory.json`）| 跨会话偏好和历史 |

### 行为

- 默认模式：每次对话保留历史，Agent 可引用之前的内容
- `--stateless` 模式：每轮清空上下文，适合单轮问答
- `/clear` 命令：手动重置对话上下文

---

## 评估体系

### 快速评估（子串匹配）

```bash
python main.py --eval
```

14 条测试用例覆盖 4 种类型：
- **math**（5 条）：数学计算验证
- **search**（3 条）：搜索能力验证
- **general**（3 条）：通用知识验证
- **rag**（2 条）：RAG 问答验证

### RAGAS 语义评估

```bash
python main.py --eval --ragas
```

引入 4 个 RAGAS 指标：
- **Faithfulness**：回答是否忠实于检索上下文
- **Answer Relevancy**：回答与问题的相关度
- **Context Precision**：检索结果的精确度
- **Context Recall**：检索结果的召回度

---

## API 服务

```bash
python main.py --api --port 8000
```

访问 `http://localhost:8000/docs` 查看 Swagger 文档。

### 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `POST` | `/chat` | Agent 聊天（多轮+记忆）|
| `POST` | `/chat/stream` | 流式聊天（SSE）|
| `POST` | `/rag` | RAG 问答 |
| `POST` | `/rag/index` | 索引文档 |
| `GET` | `/tools` | 列出可用工具 |

### 调用示例

```python
import requests

# Agent 聊天
resp = requests.post("http://localhost:8000/chat", json={
    "message": "计算 (15 + 27) * 3 并搜索今天上海的天气",
    "clear_history": True
})
print(resp.json())
# {"answer": "...", "tool_calls_made": ["calculate", "web_search"], "steps": 3, "duration_ms": 2340.5}

# RAG 问答
resp = requests.post("http://localhost:8000/rag", json={
    "question": "什么是 RAG？",
    "top_k": 5,
    "return_sources": True
})
print(resp.json())
```

---

## Web 界面

```bash
python main.py --ui
# 或创建公网链接：
python main.py --ui --share
```

访问 `http://localhost:7860`，3 个功能标签页：

| 标签 | 功能 |
|------|------|
| 💬 **Chat** | Agent 对话 + 系统提示词自定义 + 状态栏（耗时/步数/工具）|
| 📚 **RAG Q&A** | 文档问答 + 来源引用 + Top-K 调整 |
| 📥 **Index** | 一键索引文档目录 |

---

## Docker 部署

```bash
# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f api

# 停止
docker-compose down
```

镜像使用多阶段构建，最终镜像体积优化，仅包含运行时所需。

---

## 配置说明

所有配置在 `.env` 文件中管理（优先级高于 `config.py` 默认值）：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `LLM_PROVIDER` | `openai` | LLM 提供商（openai/deepseek/anthropic）|
| `LLM_MODEL` | `gpt-4o` | 模型名称 |
| `LLM_API_KEY` | — | API 密钥 |
| `LLM_BASE_URL` | `https://api.openai.com` | API 端点 |
| `LLM_TEMPERATURE` | `0.0` | 生成温度 |
| `LLM_MAX_TOKENS` | `4096` | 最大输出 Token |
| `TAVILY_API_KEY` | — | Tavily 搜索 API Key |
| `RAG_CHUNK_SIZE` | `800` | 文档分块大小 |
| `RAG_CHUNK_OVERLAP` | `120` | 分块重叠量 |
| `RAG_TOP_K` | `5` | 检索返回数 |

---

## 项目结构

```
agent-project/
├── main.py                 # 主入口（7种模式）
├── config.py               # 统一配置（LLM/RAG/Memory/Search）
├── api.py                  # FastAPI REST 服务
├── ui.py                   # Gradio Web 界面
├── Dockerfile              # 多阶段构建
├── docker-compose.yml      # 一键部署
├── requirements.txt        # 依赖清单
├── .env                    # 环境变量（API Keys）
│
├── core/                   # 核心引擎
│   ├── agent_loop.py       # ReAct 循环（推理→行动→观察）
│   └── state.py            # Message + AgentState 数据模型
│
├── llm/                    # LLM 客户端
│   └── client.py           # 统一接口（OpenAI/DeepSeek/Ollama）
│
├── tools/                  # 工具系统（10个工具）
│   ├── registry.py         # 工具注册 + dispatch
│   ├── calculator.py       # AST 安全计算器
│   ├── search.py           # Tavily 搜索 + 降级
│   ├── time_tool.py        # 时间 + 日期间隔
│   ├── weather.py          # 天气查询
│   ├── web_fetch.py        # 网页抓取
│   └── file_ops.py         # 文件笔记
│
├── rag/                    # RAG 检索增强生成
│   ├── qa.py               # 检索→构建上下文→LLM生成
│   ├── loader.py           # 文档加载（PDF/MD/TXT）
│   ├── splitter.py         # 递归文本分割
│   └── vector_store.py     # ChromaDB 封装（多embedding）
│
├── memory/                 # 记忆系统
│   └── memory.py           # deque短期 + JSON长期
│
├── graph/                  # LangGraph 工作流
│   └── workflow.py         # agent→tools→finish 状态机
│
├── eval/                   # 评估体系
│   ├── evaluator.py        # 快速评估 + RAGAS
│   └── questions.json      # 14条测试用例
│
└── logs/                   # 运行时数据（向量库/记忆/笔记/报告）
```

---

## 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| LLM | DeepSeek API (OpenAI 兼容) | 推理与生成 |
| Agent | ReAct 范式 | 推理-行动循环 |
| 工具 | 自定义 ToolRegistry | 10 个可调用工具 |
| 工作流 | LangGraph | 状态机编排 |
| 向量存储 | ChromaDB | 文档索引与语义检索 |
| 记忆 | deque + JSON | 短期 + 长期记忆 |
| API | FastAPI + Uvicorn | RESTful 服务 |
| UI | Gradio | Web 聊天界面 |
| 评估 | RAGAS | 语义评估指标 |
| 部署 | Docker + Compose | 容器化 |

---

## 贡献

欢迎提交 Issue 和 Pull Request。

## License

MIT
