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
| `wqb_agent/client.py` | WQB API：认证、datasets/datafields 分页、simulate、Retry-After 轮询、alpha 指标 |
| `wqb_agent/discovery.py` | 按 Hypothesis 选 dataset，分页检索字段；运行期页缓存 + 每轮 API 预算 |
| `wqb_agent/candidate.py` | 探索池 + 深化池双池构造，深化按 lineage 做单变量局部优化 |
| `wqb_agent/diversity.py` | 字段/表达式/假设相似度、冗余判定、提交池去重 |
| `wqb_agent/validation.py` | 异常高信号扰动验证（SUSPICIOUS_HIGH_SIGNAL → VALIDATED_HIGH_SIGNAL） |
| `wqb_agent/simulator.py` | 并发模拟（默认最多 3 个同时跑） |
| `wqb_agent/reflection.py` | Good/PROMISING/FAIL/SUSPICIOUS 分类、lineage 更新、提交池维护、三层记忆沉淀 |
| `wqb_agent/memory.py` | 三层记忆：short-term（next/active lineages）、long-term（多次验证的经验）、garbage（重复失败/过时/不可复现） |
| `wqb_agent/agent.py` | 主循环编排（探索/深化规划、预算、验证、去重、下一轮计划） |
| `wqb_agent/state.py` | ResearchState / Experiment / Trajectory / AlphaRecord 数据模型 |

## 状态文件（`.wqb_state/`）

- `experience.json` — 三层记忆 + current_best + submission_pool + active_lineages
- `trajectory.json` — 实验轨迹（含 lineage）
- `round_N.json` — 每轮 ResearchState

## 三层记忆

| 层级 | 内容 | 维护 |
|------|------|------|
| Short-term | next 计划、active lineage、低证据 lesson | 每轮压缩，过期归档到 garbage |
| Long-term | evidence ≥ 3 的经验、稳定失败模式、成熟 lineage | 保留 top-N |
| Garbage | 重复失败、过时经验、不可复现高信号、冗余去重 | 容量封顶，默认不进上下文 |

## 测试

```bash
python -m unittest discover -s tests
```

## 感谢

感谢 **MonkeyCode** 提供开发与运行支持。
感谢 WorldQuant BRAIN 平台提供的模拟与回测基础设施。
