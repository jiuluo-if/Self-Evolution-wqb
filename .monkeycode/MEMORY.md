# User Instruction Memory

This file records user instructions, preferences, and teachings for reference in future interactions.

## Format

### User Instruction Entry
User instruction entries should follow this format:

[User Instruction Summary]
- Date: [YYYY-MM-DD]
- Context: [Mentioned scenario or time]
- Instructions:
  - [Content of user teaching or instruction, described line by line]

### Project Knowledge Entry
Entries discovered by the Agent during task execution should follow this format:

[Project Knowledge Summary]
- Date: [YYYY-MM-DD]
- Context: Discovered by Agent while performing [specific task description]
- Category: [Operations & Deployment|Build Methods|Testing Methods|Troubleshooting & Debugging|Workflow & Collaboration|Environment Configuration]
- Instructions:
  - [Specific knowledge points, described line by line]

## Deduplication Strategy
- Before adding a new entry, check for similar or identical instructions.
- If a duplicate is found, skip the new entry or merge it with the existing one.
- When merging, update the context or date information.
- This helps avoid redundant entries and keeps the memory file tidy.

## Entries

[Project Knowledge Summary]
- Date: 2026-08-17
- Context: Discovered by Agent while refactoring the WQB Alpha research agent to the full specification (dual-pool candidates, submission pool, three-tier memory, high-signal validation)
- Category: Build Methods
- Instructions:
  - 测试命令：`python -m unittest discover -s tests`，运行后确认输出 `OK` 且无 `FAIL`/`ERROR`。
  - 语法/导入检查：`python -m py_compile wqb_agent/*.py main.py tests/test_agent.py`。
  - 运行入口：`python main.py --rounds N`，支持 `--single-round` 与 `--state-dir`。
  - 运行依赖仅 `requests>=2.28`，无其他外部库，无需安装额外框架。

[Project Knowledge Summary]
- Date: 2026-08-17
- Context: Discovered by Agent while running the agent against the WorldQuant BRAIN API
- Category: Environment Configuration
- Instructions:
  - 凭据二选一：环境变量 `WQB_USERNAME` / `WQB_PASSWORD`，或凭据文件 `~/.brain_credentials.txt`（第一行用户名，第二行密码）。
  - 凭据文件不进入仓库，`.gitignore` 已忽略 `config.json`、`.env`、`credentials.json` 等敏感文件。
  - 模拟请求支持并发与 `Retry-After` 轮询，超时自动跳过，失败 alpha 保留以供重试。

[Project Knowledge Summary]
- Date: 2026-08-17
- Context: Discovered by Agent while implementing the three-tier memory and submission pool
- Category: Troubleshooting & Debugging
- Instructions:
  - 状态文件持久化在 `.wqb_state/`（`experience.json`、`trajectory.json`、`round_N.json`），已被 `.gitignore` 忽略。
  - 验证提交池时，`VALIDATED_HIGH_SIGNAL` 表示已通过扰动验证；`SUSPICIOUS_HIGH_SIGNAL` 表示待验证。
  - 调试 agent 循环可直接实例化 `Agent` 并使用假的 `WQBClient` 子类（参考 `tests/test_agent.py` 的 `FakeClient`）。
