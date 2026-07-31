[English](README.md) | [简体中文](README.zh-CN.md)

# AI Resume Optimizer

## 项目概览

AI Resume Optimizer 是一个基于 Python 3.12 的命令行简历优化工具。它接收带有
可提取文本层的 PDF 或 DOCX 简历，以及来自 UTF-8 TXT 文件或交互式粘贴的目标
岗位描述。工具通过 DeepSeek Chat Completions JSON Output 分析输入，并生成匹配分析
报告、Markdown 优化简历和可编辑的 DOCX 优化简历。

本工具采用保守改写策略，不得为了提高匹配程度而虚构经历、技能、雇主、教育背景、
日期、量化结果或其他事实。

## 功能

- 提取文本层 PDF，以及 DOCX 标题、段落、列表和表格中的文字。
- 使用经过验证的 Pydantic 模型结构化岗位要求和简历内容。
- 仅提供 `高`、`一般`、`低` 三种定性匹配评价。
- 通过 source block ID 关联事实性简历内容、匹配结论与原始证据。
- 阻断原始证据中不存在的新数字和新日期。
- 阻断把 `unsupported` 岗位要求写成用户已具备的事实。
- 显著改写必须标记为需要人工审核。
- 生成 Markdown 分析、Markdown 简历和可编辑 DOCX 简历。
- 默认拒绝覆盖任何已存在的输出文件。
- 使用 Fake 模型客户端进行可重复的完全离线测试。

工具不输出百分比、ATS 分数、通过率或招聘平台模拟结果。

## 工作流程

一次优化按以下顺序执行：

1. 预检三个输出路径，并拒绝已存在的输出文件。
2. 规范化并校验岗位描述。
3. 将 PDF 或 DOCX 简历解析为有序 source blocks。
4. 在保留 source block 证据的前提下结构化简历。
5. 提取结构化岗位概览和岗位要求。
6. 根据简历证据逐项分析岗位要求。
7. 生成带证据引用的优化简历。
8. 执行确定性真实性检查。
9. 将分析报告和简历渲染为 Markdown 与 DOCX。
10. 批量写入三个输出文件；写入失败时清理不完整输出。

所有模型响应都必须先通过明确的 Pydantic 数据模型验证，才能进入后续流程。

## 环境要求

- Python 3.12 或更高版本。
- DeepSeek API key。
- 唯一支持的 DeepSeek 模型 `deepseek-v4-flash`。
- 带有可提取文本层的 PDF，或可读取的 DOCX 文件。

不支持没有文本层的扫描版 PDF。

## 安装

从仓库源码创建虚拟环境并安装：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
```

`dev` 可选依赖会安装 pytest 和 Ruff。仅从源码正常使用时，可以不安装开发依赖：

```sh
.venv/bin/python -m pip install -e .
```

本仓库不声称已经提供正式发布的包分发。

## 配置

通过 shell、运行环境或密钥管理工具设置以下环境变量：

```sh
export DEEPSEEK_API_KEY="replace-with-your-deepseek-api-key"
export DEEPSEEK_MODEL="deepseek-v4-flash"
export DEEPSEEK_TIMEOUT_SECONDS="60"
```

- `DEEPSEEK_API_KEY` 为必填项。
- `DEEPSEEK_MODEL` 可选，默认且仅支持 `deepseek-v4-flash`。
- `DEEPSEEK_TIMEOUT_SECONDS` 可选，默认值为 `60`，并且必须是正的有限数值。
- DeepSeek API base URL 固定为 `https://api.deepseek.com`，不可配置。

项目仅使用 OpenAI Python SDK 访问 DeepSeek 的 OpenAI 兼容接口，不支持 OpenAI
模型或 Responses API。

应用不会自动加载 `.env` 文件。[`.env.example`](.env.example) 仅作为配置参考模板。
不要将 API key 提交到 Git。

## 使用方式

使用 TXT 岗位描述优化 PDF 简历：

```sh
ai-resume-optimizer optimize \
  --resume ./resume.pdf \
  --job-description ./job_description.txt \
  --output-dir ./output
```

使用 DOCX 简历：

```sh
ai-resume-optimizer optimize \
  --resume ./resume.docx \
  --job-description ./job_description.txt
```

省略 `--job-description` 后交互式粘贴岗位描述：

```sh
ai-resume-optimizer optimize \
  --resume ./resume.docx
```

单独输入一行 `END` 或发送 EOF 即可结束交互输入。未提供岗位描述文件时，不支持通过
非交互 stdin pipe 输入。

工具没有 `--overwrite` 参数。如果任一预期输出文件已经存在，命令会拒绝执行。

## 输出文件

输出目录包含：

- `analysis_report.md`
- `optimized_resume.md`
- `optimized_resume.docx`

三个文件来自同一轮运行。两种简历格式都直接由同一个经过验证的
`OptimizedResume` 模型生成；DOCX 不是通过重新解析 Markdown 生成的。分析报告
不是 ATS 分数，也不是录用结果预测。使用前必须审核全部输出。

## 真实性与人工审核

确定性检查会阻断：

- 不存在的 source block ID。
- 不存在的 requirement ID。
- 与事实性简历内容关联的 `unsupported` 岗位要求。
- 引用证据中不存在的数字。
- 引用证据中不存在的日期。
- 事实性简历正文中的明显占位内容。
- 未标记人工审核的显著改写。

这些检查是保守的安全规则，不是完整的语义验证。它们无法可靠识别所有新增公司、
学校、技能或证书，无法识别所有语义夸大、责任升级或不等价改写。提示词约束同样
不能构成事实保证。用户必须将最终简历与原始证据逐项核对。

## 隐私

简历和岗位描述会发送给 DeepSeek API，用于四项结构化任务：简历结构化、岗位要求
提取、匹配分析和简历优化。项目不声称 DeepSeek 会存储或绝不存储请求。发送敏感
数据前，请查阅 DeepSeek 当前的数据政策。

生成文件写入用户选择的本地输出目录。工具不使用数据库、云端文件存储或优化历史
服务。不要在公共终端、日志或仓库中暴露 API key 和敏感简历内容。

## 示例

以下示例输入均为完全虚构的内容：

- [示例说明](examples/README.md)
- [DOCX 示例简历](examples/sample_resume.docx)
- [示例岗位描述](examples/sample_job_description.txt)

仓库有意不包含预生成的模型输出。

## 测试

运行离线测试和质量检查：

```sh
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
```

测试默认注入 Fake 模型客户端，不调用真实 DeepSeek API，也不需要真实 API key。

## 项目结构

```text
ai-resume-optimizer/
    src/
        ai_resume_optimizer/
            parsers/
            prompts/
            renderers/
            services/
            cli.py
            config.py
            model_client.py
            models.py
            pipeline.py
    tests/
        fakes/
        integration/
        unit/
    examples/
    .github/
        workflows/
    pyproject.toml
    README.md
    README.zh-CN.md
```

## 已知限制

- 不支持 OCR 或扫描版 PDF 识别。
- 不恢复复杂多栏 PDF 或 DOCX 排版。
- 不提供 `--overwrite`。
- 不导出 PDF 简历。
- 不抓取岗位网页，也不登录招聘平台。
- 不自动投递职位。
- 不输出 ATS 分数、通过率，也不保证面试或录用。
- 真实性检查是保守规则，不是完整语义验证。
- 生成的 DOCX 使用简单的内置文档结构，不恢复原始模板。
- 超长输入会报错，不会静默截断。

## 发布状态

当前项目版本为 `0.1.0`。本次文档与验证阶段尚未创建 Git 标签或 GitHub Release。
发布操作将在文档和发布前检查通过后单独进行。
