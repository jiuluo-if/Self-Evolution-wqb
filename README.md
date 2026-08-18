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
- **Memory 记录的是“哪个研究命题获得了什么证据”，而不是“哪个字段成功过”。**

## 每轮研究（探索池 + 深化池）

1. 读取三层记忆（short-term / long-term / garbage）、beliefs 与提交池。
2. 规划本轮探索/深化比例（`explore_ratio`，池空时偏向探索）。
3. 探索池：由真实研究假设驱动，Field Discovery（Cache First + 每轮 API 预算）。
4. 深化池：对已有 active lineage 做单变量、单参数局部优化（field swap / window step / smoothing / neutralize）。
5. 真实 BRAIN Simulation（并发、轮询），受 `sim_budget_per_round` 约束。
6. Good Alpha 进入 submission candidate pool（先做 Field / 结构冗余检查）。
7. 异常高信号标记为 `SUSPICIOUS_HIGH_SIGNAL`，做扰动验证；稳定才升级为 `VALIDATED_HIGH_SIGNAL`。
8. 连续多次深化无提升的 lineage 停止并归档。
9. 更新 Belief（support / contradict / pending）、lineage、三层记忆、去重提交池，规划下一轮。

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
| `wqb_agent/beliefs.py` | 研究命题（Belief）身份的唯一 canonical helper：`belief_identity` / `belief_claim`；Reflection 与 Validation 共用，避免 key 漂移 |
| `wqb_agent/discovery.py` | 按 Hypothesis 选 dataset，分页检索字段；运行期页缓存 + 每轮 API 预算 |
| `wqb_agent/candidate.py` | 探索池 + 深化池双池构造；探索按假设类型选 operator family（reversal/momentum/revision/cross-sectional/relationship）；深化做单变量局部优化（window step / smoothing / neutralize / 语义受限 field swap） |
| `wqb_agent/diversity.py` | 字段/表达式/假设相似度、冗余判定、提交池去重、表达式实际用到的字段提取 |
| `wqb_agent/scheduler.py` | BacktestScheduler：execution group、最大并发、simulation budget、FIRST_COMPLETED 滑动窗口补任务、pending/started/completed/failed、checkpoint + crash resume（tmp + os.replace 原子写入） |
| `wqb_agent/simulator.py` | 只负责单次 submit → poll → fetch metrics |
| `wqb_agent/validation.py` | 异常高信号扰动验证：多数扰动通过 + median fitness + 相对原始 Alpha 的 performance retention + WQB checks |
| `wqb_agent/reflection.py` | Good/PROMISING/FAIL/SUSPICIOUS 分类、failure taxonomy、lineage 更新、提交池维护、三层记忆沉淀、Belief 双向证据记录 |
| `wqb_agent/memory.py` | 三层记忆 + Belief accounting：support/contradict/pending 聚合、独立 lineage 计数、置信度计算、反证降级、evidence_log 历史保留 |
| `wqb_agent/agent.py` | 主循环编排（memory-driven planning、预算、验证、去重、下一轮计划）；validation 结果回写原 Belief |
| `wqb_agent/state.py` | ResearchState / Experiment / Trajectory / AlphaRecord 数据模型 + 原子写 JSON |

## 信念记忆（Belief）

Memory 按**可证伪研究命题**聚合证据，而不是按字段族。每个 Belief 由唯一 key 标识：

```text
belief_key = hypothesis_id + normalized fields + direction
```

- 同字段、不同 hypothesis 或相反方向 → 不同 Belief；同 hypothesis 同字段同方向的参数变异（如 `ts_mean(x,20)` vs `ts_mean(x,21)`）→ 同一 Belief。
- 双向证据：`support` 提升置信度，`contradict` 降低置信度；SUSPICIOUS 记 `pending`，扰动验证通过升级为 support、失败转为 contradict。
- 置信度 = `0.5 + 0.2 × (独立支持 lineage 数 − 独立反驳 lineage 数)`，截断于 `[0.05, 0.95]`。
- 同一 lineage 重复微调只算一条独立证据；反证强度达到支持强度时 long-term 自动降级回 short-term。
- 基础设施失败（timeout / auth / rate-limit / 5xx）与表达式构造失败（syntax / data）不产生任何支持或反驳证据。
- 同一实验重放（resume / reflection）幂等，不重复计数；`evidence_log` 保留反证历史。

## 状态文件（`.wqb_state/`）

- `experience.json` — 三层记忆 + beliefs + current_best + submission_pool + active_lineages（含 schema_version / created_at / updated_at）
- `trajectory.json` — 实验轨迹（含 lineage）
- `round_N.json` — 每轮 ResearchState
- `round_N_jobs.json` — 每轮回测 checkpoint，崩溃后自动恢复，已完成的 job 不重复执行

## 三层记忆

| 层级 | 内容 | 维护 |
|------|------|------|
| Short-term | next 计划、active lineage、低证据 lesson | 每轮压缩，过期归档到 garbage |
| Long-term | evidence 累计 ≥ 3 且来自多个独立 round 的经验、稳定失败模式、成熟 lineage；Belief 需独立支持与跨轮确认且反证弱于支持 | 保留 top-N |
| Garbage | 重复失败、过时经验、不可复现高信号、冗余去重 | 容量封顶，默认不进上下文 |

## 测试

```bash
python3 -m unittest discover -s tests
```

## 感谢

感谢 **MonkeyCode** 提供开发与运行支持。
感谢 WorldQuant BRAIN 平台提供的模拟与回测基础设施。
