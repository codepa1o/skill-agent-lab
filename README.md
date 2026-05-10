# Skill Agent Lab

**基于 Skill 的高考/职业规划 Agent 测评系统**

一个用于学习 Agent 应用开发的项目：输入 GitHub Skill URL，自动拉取 `SKILL.md`，创建独立对话窗口，让 AI 按 Skill 进行多轮回答，并保存运行日志。

## 项目截图

启动服务后访问 `http://127.0.0.1:8000`，建议截图保存到 `docs/screenshot-home.png`，用于简历或 README 展示。

![首页截图](docs/screenshot-home.png)

## 功能

- 输入 GitHub `SKILL.md` 文件链接，自动转换为 raw URL 并拉取内容。
- 创建多个相互独立的对话窗口，每个窗口都有自己的消息记忆。
- 用户继续追问时，系统会携带当前对话最近 20 条消息作为上下文。
- 支持手动设置和修改对话标题。
- 保留 `runs` 运行日志，记录模型、API 模式、Base URL、推理强度、耗时和错误信息。
- 页面展示运行状态、消息流、友好错误提示、对话列表和运行日志。
- 支持测试集和多个测试用例，用另一个 AI 作为 Judge 自动评分并生成评测报告。
- 支持对照评测：同一用例分别生成 baseline 不增强回答和 enhanced 搜索/RAG 增强回答，对比得分差异。
- 支持本地资料库 RAG：上传 Markdown / TXT / PDF，自动切分、生成向量、检索并在回答中引用来源。

## 运行流程

```text
用户输入 Skill URL、对话标题和第一条消息
  -> 后端拉取 SKILL.md
  -> 创建 conversation 和 user message
  -> 构建安全边界 + Skill 指令 + 用户问题
  -> 自动检索本地资料库 RAG
  -> 需要时触发实时搜索增强
  -> 携带当前窗口最近 20 条消息
  -> 调用 OpenAI-compatible 模型接口
  -> 保存 assistant message
  -> 保存 runs 运行日志
  -> 页面展示消息流和对话列表
```

## 快速开始

```powershell
cd E:\pythonlearning\code\agent_learning\skill-agent-lab
.\skill-lab\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

访问：

```text
http://127.0.0.1:8000
```

## freemodel.dev 配置

你的中转站配置是 `wire_api = "responses"`，所以 `.env` 应使用 Responses 模式：

```text
OPENAI_API_KEY=你的中转站 key
OPENAI_BASE_URL=https://api.freemodel.dev
OPENAI_MODEL=gpt-5.5
OPENAI_REVIEW_MODEL=gpt-5.4
OPENAI_API_MODE=responses
OPENAI_REASONING_EFFORT=medium
OPENAI_DISABLE_RESPONSE_STORAGE=true
OPENAI_TIMEOUT_SECONDS=120
OPENAI_MAX_OUTPUT_TOKENS=1800
SEARCH_ENABLED=true
TAVILY_API_KEY=你的 Tavily API Key
SEARCH_MAX_RESULTS=5
DASHSCOPE_API_KEY=你的阿里云百炼 API Key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIMENSIONS=1024
RAG_ENABLED=true
RAG_TOP_K=5
RAG_MIN_SCORE=0.2
RAG_CHUNK_SIZE=800
RAG_CHUNK_OVERLAP=120
JOB_WORKER_ENABLED=true
JOB_POLL_INTERVAL_SECONDS=2
JOB_MAX_ATTEMPTS=1
```

复杂问题更容易触发中转站超时，所以第一版建议先用 `medium`。

## 数据库

本项目使用 SQLite，本地数据库文件在：

```text
data/skill_agent_lab.db
```

用命令行查看：

```powershell
sqlite3 data\skill_agent_lab.db
.tables
.schema runs
SELECT id, status, model, api_mode, reasoning_effort, latency_ms, created_at FROM runs ORDER BY id DESC LIMIT 10;
```

如果没有安装 `sqlite3` 命令，也可以用 VS Code 的 SQLite Viewer 插件打开 `data/skill_agent_lab.db`。

主要数据表：

- `conversations`：对话窗口，保存标题、Skill URL、模型配置、创建和更新时间。
- `messages`：对话消息，保存 user/assistant 消息、耗时和错误信息。
- `runs`：运行日志，保存每一次模型调用结果，方便调试和复盘。
- `rag_documents`：本地资料文档，保存文件名、类型、路径、索引状态和切片数量。
- `rag_chunks`：本地资料切片，保存正文、页码、embedding JSON 和向量维度。
- `test_suites`：测试集，保存 Skill URL 和评测说明。
- `test_cases`：测试用例，保存问题、期望行为和评分重点。
- `eval_runs`：一次批量评测运行。
- `eval_results`：每条用例的 Agent 回答、Judge 分数和评语。
- `jobs`：后台任务队列，保存资料索引和评测运行的状态、结果和错误。

## 评测中心

访问：

```text
http://127.0.0.1:8000/evals
```

系统首次启动会自动创建“张雪峰 Skill 基础评测集”，包含：

- 简单寒暄
- 普通家庭专业选择
- 浙江计算机志愿咨询
- 高风险承诺测试

Judge 评分维度：

- `role_adherence`：角色遵守度
- `constraint_adherence`：Skill 约束遵守度
- `task_completion`：任务完成度
- `factual_safety`：事实安全和幻觉风险
- `format_quality`：输出格式质量
- `source_usage`：来源引用质量
- `overall`：总分

评测报告会对每个测试用例生成两组回答：

- `baseline`：不使用搜索、不使用本地资料库，只按 Skill 回答。
- `enhanced`：自动使用本地资料库 RAG，并在需要时使用搜索增强。

报告页会展示两组回答、两组评分、来源引用分，以及 enhanced 相比 baseline 的总分差值。评测会创建后台任务，页面会自动刷新任务状态，刷新浏览器也不会丢失已完成的结果。

## 搜索增强

第三版第一阶段支持搜索增强。系统会用关键词判断是否需要搜索，命中学校、专业、就业、分数线、录取、位次、志愿等问题时，会先调用搜索 API，再把搜索摘要注入 Agent 上下文，最后在回答末尾展示来源链接。

配置：

```text
SEARCH_ENABLED=true
TAVILY_API_KEY=你的 Tavily API Key
SEARCH_MAX_RESULTS=5
```

如果没有配置 `TAVILY_API_KEY`，系统不会中断回答，但会在回答末尾提示搜索增强未启用。搜索结果也会保存到 `runs.search_results`，方便在运行日志详情页复盘。

## 本地资料库 RAG

访问：

```text
http://127.0.0.1:8000/knowledge
```

支持上传：

- Markdown：`.md`、`.markdown`
- 文本：`.txt`
- PDF：`.pdf`

上传后系统会创建后台索引任务，自动解析文档、切分文本、调用阿里云百炼 Embedding API 生成向量，并保存到 SQLite。文档详情页会显示排队中、索引中、就绪或失败状态，并支持重新索引。后续对话、单次运行和评测都会自动检索本地资料；命中资料时，回答末尾会出现 `【本地资料来源】`。

RAG 使用阿里云百炼 OpenAI 兼容 Embedding 接口，默认配置：

```text
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIMENSIONS=1024
```

参考文档：

- OpenAI 兼容 Embedding 接口：https://help.aliyun.com/zh/model-studio/embedding-interfaces-compatible-with-openai
- 文本向量同步接口：https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api/

如果没有配置 `DASHSCOPE_API_KEY`，上传资料会保存失败状态并展示错误；普通对话仍然可以继续，只是不会命中本地资料。

## 后台任务与系统诊断

第四版引入 SQLite 后台任务队列，用于避免资料索引和批量评测长时间卡住页面。服务启动后会启动一个轻量 worker，依次执行 `index_document` 和 `run_eval_suite` 任务。

配置：

```text
JOB_WORKER_ENABLED=true
JOB_POLL_INTERVAL_SECONDS=2
JOB_MAX_ATTEMPTS=1
```

访问系统诊断页：

```text
http://127.0.0.1:8000/settings/diagnostics
```

诊断页会检查 OpenAI 兼容模型、Tavily、DashScope、RAG 参数和后台任务 worker 是否配置完整。

## 测试

```powershell
pytest
```

测试覆盖：

- GitHub blob URL 到 raw URL 的转换
- 非 GitHub URL 拒绝
- 空问题提交
- 失败记录保存
- 老 SQLite 表自动补字段迁移
- 对话窗口和消息保存
- 对话标题修改
- 默认测试集初始化
- Judge JSON 解析
- 评测运行和评测结果保存
- SQLite jobs 任务创建、claim、完成和失败记录
- 后台 worker 执行资料索引任务
- 资料上传和评测运行创建后台任务
- Markdown / TXT 文档切分
- RAG 向量检索、来源拼接和失败状态保存

## 常见错误

- `404 not found`：通常是 `OPENAI_API_MODE` 和中转站 wire API 不匹配，或 `OPENAI_BASE_URL` 路径不对。freemodel.dev 应使用 `OPENAI_API_MODE=responses` 和 `OPENAI_BASE_URL=https://api.freemodel.dev`。
- `504 Gateway Time-out`：通常是中转站或上游模型超时，复杂志愿填报问题更容易触发。项目会自动重试；仍失败时，把 `OPENAI_REASONING_EFFORT` 改成 `medium` 或 `low`。
- `模型没有返回可展示的文本结果`：通常是兼容接口返回结构和官方 SDK 不完全一致，或输出 token 太少。项目已兼容多种 Responses 返回结构；仍出现时可增大 `OPENAI_MAX_OUTPUT_TOKENS`。
- `未配置 DASHSCOPE_API_KEY`：本地资料库无法生成 embedding。请在 `.env` 填写阿里云百炼 API Key。
- `PDF 没有解析出可索引的正文内容`：常见于扫描版 PDF，需要先 OCR 成文本再上传。

## 默认测试 Skill

首页会预填这个 Skill URL：

```text
https://github.com/alchaincyf/zhangxuefeng-skill/blob/main/SKILL.md
```

## 注意

这是基于 Skill 的角色模拟与测试，不代表真实人物、机构或官方建议。
