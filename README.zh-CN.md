[English](README.md) | [简体中文](README.zh-CN.md)

# AI Resume Optimizer

## 项目概览

AI Resume Optimizer 是一个基于 Python 3.12 的简历优化库与命令行工具。公共 Python
API 接收带有可提取文本层的 PDF 或 DOCX 路径及岗位描述文本，并返回经过验证的纯内存
结果。保持兼容的 CLI 还支持 UTF-8 TXT 岗位描述或交互式输入，并导出匹配分析报告、
Markdown 优化简历和可编辑的 DOCX 优化简历。

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
- 提供稳定的纯内存 Python Runner，便于其他应用嵌入且不写入文件。
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

公共 Runner 只执行纯内存阶段，不进行输出路径预检、渲染或文件写入。CLI 在同一套共享
优化核心外围增加文件导出职责。

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

## 公共 Python API

v0.2.1 提供稳定的公共集成边界：

```python
from pathlib import Path

from ai_resume_optimizer import (
    ResumeOptimizerConfig,
    create_resume_optimizer,
)

config = ResumeOptimizerConfig(
    deepseek_api_key="replace-with-your-deepseek-api-key",
)
runner = create_resume_optimizer(config)

try:
    result = runner.optimize(
        resume_path=Path("resume.docx"),
        job_description="A concise job description.",
    )
finally:
    runner.close()
```

公共 contract 包括：

- `ResumeOptimizerConfig`：不可变的 production 配置，其 repr 不包含
  `deepseek_api_key`。
- `create_resume_optimizer(config)`：装配受支持的 DeepSeek client 和拥有资源所有权的
  `ResumeOptimizerRunner`。
- `ResumeOptimizerRunner`：接收注入的 `ModelClient`，提供 `optimize(...)` 和幂等
  `close()`。
- `ModelClient`：provider-neutral 的结构化生成依赖注入协议。
- `OptimizationResult`：Runner 返回的经过验证的结果。
- 分析与优化简历的公共 DTO，以及以 `ResumeOptimizerError` 为根的分类公共异常。

`ResumeOptimizerRunner.optimize` 只接收：

- `resume_path: Path`：现有的文本层 PDF 或可读取 DOCX 简历；规范化简历文本上限为
  50,000 字符。
- `job_description: str`：非空纯文本；规范化岗位描述上限为 30,000 字符。

结果包含 `analysis`、`optimized_resume`、`warnings` 和 `output_paths`。`analysis`
公开 `overall_rating`、`overall_evaluation`、岗位要求 `assessments`、
`main_issues`、`section_suggestions`、`keyword_suggestions`、
`truthfulness_risks` 与 `content_not_to_add`。`optimized_resume` 包含经过验证的
sections、待用户补充内容与 warnings。API 不虚构 ATS 数值分数、confidence、token
usage 或 metrics。

### Public provenance API

Production `RequirementAssessment` 保留原有 `requirement_id`、
`source_block_ids`、状态、原因和建议操作，并新增：

- `requirement: RequirementReference`：人类可读的岗位要求描述、类别、重要性与岗位
  描述原文 excerpt。
- `evidence: list[RequirementEvidence]`：原始 source block 的 kind、location、excerpt
  及明确的语义 section references。

Requirement provenance 只按稳定 requirement ID 关联。Evidence 按
`source_block_ids` 原顺序，从原始解析得到的 `SourceBlock` 确定性复制；不会由模型重新
生成，不会从 optimized resume 反推，也不使用 fuzzy matching。未知 requirement 或
source block ID 会 fail closed 并抛出 `ModelOutputError`。

这是 v0.2.1 的 additive contract change。Runner API 与 CLI 不变，现有
`requirement_id`、`source_block_ids` 保持兼容。旧调用方手动构造 DTO 时仍可省略新增
字段；正常 production Runner result 会填充 requirement reference 与对齐的 evidence
列表。

公共 Runner 始终返回 `output_paths == {}`，不会写 Markdown、写 DOCX、创建输出目录
或向标准输出 print。文件导出仍是独立的 CLI 能力。

Runner 会抛出稳定的领域异常，例如 `InputError`、`ResumeExtractionError`、
`ModelCallError`、`ModelOutputError`、`TruthfulnessError` 和
`ResumeOptimizerClosedError`。嵌入方无需解析 CLI 退出文本、stderr 或 provider SDK
异常。

公共 Runner 不支持 TXT 简历、bytes、upload object、URL、扫描版 PDF 或 OCR。CLI
可以从 UTF-8 TXT 读取岗位描述，但公共 Runner 直接接收 `str`。

## 架构与生命周期

```text
Application / CLI
        ↓
ResumeOptimizerRunner
        ↓
in-memory optimization core
        ↓
ModelClient
        ↓
DeepSeekModelClient
```

Production 装配链为
`ResumeOptimizerConfig → DeepSeekModelClient → owned ResumeOptimizerRunner`。
Factory 创建的 Runner 拥有并关闭 provider client；直接传给
`ResumeOptimizerRunner` 的 ModelClient 默认属于外部资源，Runner 不会关闭它。
`close()` 幂等；关闭后再次 optimize 会抛出 `ResumeOptimizerClosedError`。

纯内存优化核心与 Markdown/DOCX 渲染、共享原子文件导出相互分离。

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

v0.2.1 CLI 继续保留原参数、环境变量、三个输出文件、覆盖保护和分类退出码。其内部
production 路径已经统一为：

```text
environment
→ ResumeOptimizerConfig
→ create_resume_optimizer
→ ResumeOptimizerRunner.optimize（恰好一次）
→ shared atomic export
```

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
.venv/bin/python -m pip check
```

测试默认注入 Fake 模型客户端，不调用真实 DeepSeek API，也不需要真实 API key。

v0.2.1 发布基线为：336 个离线测试通过、Ruff 通过、`ruff format --check` 通过、
`pip check` 通过。项目当前没有配置 mypy。

受控 Real Public Runner E2E 也已通过：使用完全虚构的
`examples/sample_resume.docx` 与 `deepseek-v4-flash`，production factory 装配
成功，结果通过 `OptimizationResult` 验证，`output_paths == {}`，没有生成输出文件，
Runner 关闭成功。

首次运行收到空模型响应，并被 `ModelOutputError` 正确拒绝。随后安全 metadata 诊断
得到正常 `ChatCompletion` 响应，最终受控完整 E2E 通过。这不能证明 client 或 provider
存在 bug；v0.2.1 未加入自动重试、backoff 或 fallback。

使用同一虚构 fixture 与受支持模型的 Real Public Runner Provenance E2E 也已通过，
验证了非空的人类可读 requirement reference、原始 evidence excerpt、evidence/source
ID 顺序完全一致、production provenance invariant、Pydantic round-trip、
`output_paths == {}`、无文件副作用和幂等关闭。仓库不保存简历、岗位描述、prompt、
raw model response 或 API key。

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
            factory.py
            model_client.py
            models.py
            pipeline.py
            runner.py
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
- 公共 Runner 不接收 TXT 简历、bytes、upload object 或 URL。
- 无效或空模型输出会被拒绝；未实现自动重试或 fallback model。

## 发布状态

当前项目版本为 `0.2.1`。本次文档与验证阶段尚未创建 v0.2.1 Git 标签或 GitHub Release。
发布操作将在文档和发布前检查通过后单独进行。
