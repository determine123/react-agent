# 简历编译说明

## 在线编译（推荐，无需本地安装 LaTeX）

1. 打开 [Overleaf](https://www.overleaf.com) 或 [ShareLaTeX](https://www.sharelatex.com)
2. 上传整个 `resume/` 文件夹
3. 将编译器设置为 **XeLaTeX**
4. 点击编译

## 本地编译

```bash
# 1. 克隆模板
git clone https://github.com/hijiangtao/resume.git --branch master --depth 1

# 2. 将 resume-zh_CN.tex 复制到模板目录
cp resume-zh_CN.tex resume/

# 3. 编译（需要安装 Adobe 中文字体或使用 external 包）
cd resume
xelatex resume-zh_CN.tex
```

## 字体说明

当前使用 `zh_CN-Adobefonts_external` 包，Overleaf 上可直接编译。
如果有 Adobe 字体，改为 `zh_CN-Adobefonts_internal`。

## 简历核心定位

- **目标岗位**：AI Agent / 大模型应用开发 / LLM应用工程 实习生
- **核心卖点**：完整Agent系统（10工具+RAG+多Agent）+ 强化学习科研 + EI论文
- **关键词**：ReAct、LangGraph、ChromaDB、HyDE、Self-RAG、PPO、SAC
