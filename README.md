# WQB Alpha 自进化研究 Agent

面向 **WorldQuant BRAIN** 平台的小规模 Alpha 自进化研究 Agent。

```text
Memory → Hypothesis → Field Discovery → Candidate
→ Real Simulation → Reflection → Memory → Next Experiment
```

## 原则

- **按假设找字段，而不是按字段找公式。**
- **真实 Simulation 产生证据，证据形成经验，经验改变下一轮实验。**
- **目标不是单个最高 Sharpe，而是一组质量良好、相互独立、经过验证、不过度拟合的 Alpha 候选池。**

## 每轮研究（探索池 + 深化池）

1. 读取三层记忆（short-term / long-term / garbage）与提交池。
2. 规划本轮探索/深化比例（`explore_ratio`，池空时偏向探索）。
3. 探索池：由真实研究假设驱动，Field Discovery（Cache First + 每轮 API 预算）。
4. 深化池：对已有 active lineage 做单变量、单参数局部优化（field swap / window step / smoothing / neutralize）。
5. 真实 BRAIN Simulation（并发、轮询），受 `sim_budget_per_round` 约束。
6. Good Alpha 进入 submission candidate pool（先做 Field / 结构冗余检查）。
7. 异常高信号标记为 `SUSPICIOUS_HIGH_SIGNAL`，做扰动验证；稳定才升级为 `VALIDATED_HIGH_SIGNAL`。
8. 连续多次深化无提升的 lineage 停止并归档。
9. 更新 lineage、三层记忆、去重提交池，规划下一轮。

## 安装

```bash
pip install -r requirements.txt
```

## 凭据

二选一：

1. 环境变量

```bash
export WQB_USERNAME=your-username
export WQB_PASSWORD=your-password
```

2. 凭据文件 `~/.brain_credentials.txt`（第一行用户名，第二行密码）

```text
username
password
```

## 运行

```bash
cp config.example.json config.json
python main.py --rounds 5
python main.py --single-round        # 只跑一轮
python main.py --state-dir /tmp/wqb  # 指定状态目录
```

## 结构

| 模块 | 职责 |
|------|------|
| `wqb_agent/client.py` | WQB API：统一 retry（401→重新认证、429→Retry-After、5xx/超时→指数退避+抖动、400/403/404/422→快速失败）、thread-local Session、poll 快速失败 |
| `wqb_agent/failures.py` | 失败分类：RESEARCH / SYNTAX / DATA / INFRA / AUTH / RATE_LIMIT / TIMEOUT；仅研究相关失败进入研究记忆 |
| `wqb_agent/discovery.py` | 按 Hypothesis 选 dataset，分页检索字段；运行期页缓存 + 每轮 API 预算 |
| `wqb_agent/candidate.py` | 探索池 + 深化池双池构造；探索按假设类型选 operator family（reversal/momentum/revision/cross-sectional/relationship）；深化做单变量局部优化（window step / smoothing / neutralize / 语义受限 field swap） |
| `wqb_agent/diversity.py` | 字段/表达式/假设相似度、冗余判定、提交池去重、表达式实际用到的字段提取 |
| `wqb_agent/scheduler.py` | BacktestScheduler：execution group、最大并发、simulation budget、FIRST_COMPLETED 滑动窗口补任务、pending/started/completed/failed、checkpoint + crash resume（tmp + os.replace 原子写入） |
| `wqb_agent/simulator.py` | 只负责单次 submit → poll → fetch metrics |
| `wqb_agent/validation.py` | 异常高信号扰动验证：多数扰动通过 + median fitness + 相对原始 Alpha 的 performance retention + WQB checks |
| `wqb_agent/reflection.py` | Good/PROMISING/FAIL/SUSPICIOUS 分类、failure taxonomy、lineage 更新、提交池维护、三层记忆沉淀 |
| `wqb_agent/memory.py` | 三层记忆：short-term（next/active lineages）、long-term（跨多轮独立证据）、garbage（重复失败/过时/不可复现）；evidence 记录真实实验来源 |
| `wqb_agent/agent.py` | 主循环编排（memory-driven planning、预算、验证、去重、下一轮计划） |
| `wqb_agent/state.py` | ResearchState / Experiment / Trajectory / AlphaRecord 数据模型 + 原子写 JSON |

## 状态文件（`.wqb_state/`）

- `experience.json` — 三层记忆 + current_best + submission_pool + active_lineages（含 schema_version / created_at / updated_at）
- `trajectory.json` — 实验轨迹（含 lineage）
- `round_N.json` — 每轮 ResearchState
- `round_N_jobs.json` — 每轮回测 checkpoint，崩溃后自动恢复，已完成的 job 不重复执行

## 三层记忆

| 层级 | 内容 | 维护 |
|------|------|------|
| Short-term | next 计划、active lineage、低证据 lesson | 每轮压缩，过期归档到 garbage |
| Long-term | evidence 累计 ≥ 3 且来自多个独立 round 的经验、稳定失败模式、成熟 lineage | 保留 top-N |
| Garbage | 重复失败、过时经验、不可复现高信号、冗余去重 | 容量封顶，默认不进上下文 |

## 测试

```bash
python -m unittest discover -s tests
```

## 感谢

感谢 **MonkeyCode** 提供开发与运行支持。
感谢 WorldQuant BRAIN 平台提供的模拟与回测基础设施。
