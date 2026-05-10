# Skill Agent Lab

**基于 Skill 的高考/职业规划 Agent 测评系统**

一个用于学习 Agent 应用开发的第一版闭环项目：输入 GitHub Skill URL，自动拉取 `SKILL.md`，让 AI 按 Skill 回答测试问题，并保存每次测试记录。

## 项目截图

启动服务后访问 `http://127.0.0.1:8000`，建议截图保存到 `docs/screenshot-home.png`，用于简历或 README 展示。

![首页截图](docs/screenshot-home.png)

## 功能

- 输入 GitHub `SKILL.md` 文件链接，自动转换为 raw URL 并拉取内容。
- 根据 Skill 内容构建角色型 Agent Prompt，并调用 OpenAI-compatible API。
- 页面展示运行状态、回答结果、友好错误提示和最近测试记录。
- 保存模型、API 模式、Base URL、推理强度、耗时和错误信息。
- 支持详情页查看单次运行，并支持一键重新运行。

## 运行流程

```text
用户输入 Skill URL 和测试问题
  -> 后端拉取 SKILL.md
  -> 构建安全边界 + Skill 指令 + 用户问题
  -> 调用 OpenAI-compatible 模型接口
  -> 提取回答文本并处理常见错误
  -> 保存 SQLite 测试记录
  -> 页面展示结果和历史记录
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
OPENAI_API_MODE=responses
OPENAI_REASONING_EFFORT=medium
OPENAI_DISABLE_RESPONSE_STORAGE=true
OPENAI_TIMEOUT_SECONDS=120
OPENAI_MAX_OUTPUT_TOKENS=1800
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

## 常见错误

- `404 not found`：通常是 `OPENAI_API_MODE` 和中转站 wire API 不匹配，或 `OPENAI_BASE_URL` 路径不对。freemodel.dev 应使用 `OPENAI_API_MODE=responses` 和 `OPENAI_BASE_URL=https://api.freemodel.dev`。
- `504 Gateway Time-out`：通常是中转站或上游模型超时，复杂志愿填报问题更容易触发。项目会自动重试；仍失败时，把 `OPENAI_REASONING_EFFORT` 改成 `medium` 或 `low`。
- `模型没有返回可展示的文本结果`：通常是兼容接口返回结构和官方 SDK 不完全一致，或输出 token 太少。项目已兼容多种 Responses 返回结构；仍出现时可增大 `OPENAI_MAX_OUTPUT_TOKENS`。

## 默认测试 Skill

首页会预填这个 Skill URL：

```text
https://github.com/alchaincyf/zhangxuefeng-skill/blob/main/SKILL.md
```

## 注意

这是基于 Skill 的角色模拟与测试，不代表真实人物、机构或官方建议。第一版不包含 RAG、搜索增强、LLM Judge、排行榜或多 Skill 对比。
