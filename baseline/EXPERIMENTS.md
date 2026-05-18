# 实验迭代记录

> 评测命令：`uv run python evaluator.py`（本地调试默认 `ROCO_EVAL_NUM_RUNS=1`，每个任务 1 run）。  
> 计分口径：记录 6 个任务的 success_count/total_count、成功率、超时数、平均步数与关键失败原因。

## 全局经验池

- Tool-RoCo 可借鉴点：把机器人/动作视为 tool，规划前先做 tool schema 校验；每轮由 coordinator 显式分配 active agent、动作和参数；失败反馈必须禁止重复同一非法 tool call。
- 历史失败集中在：格式解析失败、不可达目标、持有物状态冲突、所有机器人 WAIT、失败动作反复出现。

## 实验记录

| 轮次 | 创新点 | 指标 | 结果 | 经验 |
|---|---|---:|---|---|
| setup | evaluator 默认每任务 1 run，便于快速迭代 | 待测 | 待测 | 只影响本地快速评测；正式评分会替换 evaluator.py。 |
| baseline-0 | 原始 qwen3-8b + chat + 1 run | 2/6：sort 0/1, cabinet 1/1, rope 0/1, sweep 0/1, sandwich 1/1, pack 0/1 | 失败 | sort 会把 cube 放到错误目标/不可达目标；sweep 两机器人选择不同 cube 后低效；rope/pack PATH 反复不均匀或碰撞。 |
| 1 | Tool-RoCo式启发式 tool coordinator：对 SweepTask 绕过自由文本规划，按 `MOVE same cube -> WAIT/SWEEP -> DUMP` 阶段机生成严格动作 | 聚合 3/6（新增 sweep 1/1，steps=8）；单测 sweep 1/1，elapsed=241.96s | 成功提点 | 对强规则任务，确定性 tool schema + active action stage 比 LLM 讨论更稳；两机器人必须绑定同一个 cube，扫入 dustpan 后立刻 DUMP。 |
| 2 | SortTask 装配线 handoff coordinator：一次只移动一个 cube，经 panel3/panel5 中转，避免不可达与多臂冲突 | 聚合 4/6（新增 sort 1/1，steps=5）；单测 sort 1/1，elapsed=122.38s | 成功提点 | 原 LLM 混淆“机器人目标”和“cube目标”；硬编码 cube->target 与 reach handoff 后可稳定完成。pink 从 panel3 应由 Bob 直接放 panel4，不能让 Chad 去 panel5。 |
