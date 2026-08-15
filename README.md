# WQB Alpha 自进化研究 Agent

面向 **WorldQuant BRAIN** 平台的小规模 Alpha 自进化研究 Agent（MVP）。

```text
Memory → Hypothesis → Field Discovery → Candidate
→ Real Simulation → Reflection → Memory → Next Experiment
```

## 原则

- **按假设找字段，而不是按字段找公式。**
- **真实 Simulation 产生证据，证据形成经验，经验改变下一轮实验。**

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
| `wqb_agent/discovery.py` | 按 Hypothesis 选 dataset，分页拉取字段并检索，达标即停，运行期缓存 |
| `wqb_agent/candidate.py` | 构造 5~6 个 Candidate，有 best 时优先单变量局部修改 |
| `wqb_agent/simulator.py` | 并发模拟（默认最多 3 个同时跑） |
| `wqb_agent/reflection.py` | 结果分类、FAIL 诊断、经验提取、更新 best |
| `wqb_agent/memory.py` | Experience Memory：current_best / lessons / avoid / next，JSON 持久化 + 压缩 |
| `wqb_agent/agent.py` | 主循环编排（ResearchState / Experiment / Trajectory） |
| `wqb_agent/state.py` | ResearchState、Experiment、Trajectory 数据模型 |

## 状态文件（`.wqb_state/`）

- `experience.json` — 压缩后的经验记忆（lessons / avoid / next / current_best）
- `trajectory.json` — 实验轨迹
- `round_N.json` — 每轮 ResearchState

## 每轮研究

1. 读取 current_best、lessons、avoid、next
2. 形成明确 Hypothesis（沿用 best 深化或切换种子方向）
3. 按 Hypothesis 选 dataset，通过 API 分页检索 Fields
4. 提出 5~6 个 Candidate（优先单变量局部修改）
5. 真实 BRAIN Simulation（并发、轮询）
6. 读取 Sharpe / Fitness / Turnover / Margin / Checks
7. FAIL 先诊断再针对性修改
8. 更新 Current Best 并压缩 Memory

## 测试

```bash
python -m unittest discover -s tests
```
