# 多机器人协同操作 - 代码使用指南

## 📁 文件说明

```
code/
├── config.py                    # ⭐ 配置文件
├── ollama_client.py             # ① Ollama LLM 客户端封装
├── prompt_templates.py          # ② 所有Prompt模板
├── dialog_handler_template.py   # ③ 处理器模板(基础版)
│
├── format_healer.py             # V2  格式修复器
├── action_filter.py             # V3  SayCan动作过滤器
├── task_decomposer.py           # V4  任务分解器
├── role_assigner.py             # V5  动态角色分配器
├── reflection_engine.py         # V6  反思引擎
├── experience_memory.py         # V7  经验记忆库
├── loop_detector.py             # V8  循环检测器
├── candidate_voter.py           # V9  候选投票器
├── evaluation_logger.py         # V10 评估日志系统
│
├── integrated_handler.py        # V10 ⭐ 最终集成处理器(推荐)
├── test_ollama.py               # Ollama连接测试
└── README.md                    # 本文件
```

| 文件 | 版本 | 借鉴来源 | 一句话描述 |
|------|:---:|------|------|
| `config.py` | - | - | 配置模型/地址/路径 |
| `ollama_client.py` | V1 | Ollama OpenAI Compat | LLM API调用封装 |
| `prompt_templates.py` | V1 | Prompt Engineering | 6任务完整Prompt模板 |
| `format_healer.py` | V2 | Toolformer / JSON repair | 两层格式自动修复 |
| `action_filter.py` | V3 | Google SayCan | 动作合法性过滤 |
| `task_decomposer.py` | V4 | LLM-Planner / CoT | 长程任务→子目标 |
| `role_assigner.py` | V5 | RoCo / AutoGen | 动态角色分配 |
| `reflection_engine.py` | V6 | Inner Monologue / Reflexion | 失败反思重规划 |
| `experience_memory.py` | V7 | Voyager / Reflexion | Few-Shot经验记忆 |
| `loop_detector.py` | V8 | Voyager | 三种策略循环检测 |
| `candidate_voter.py` | V9 | Self-Consistency | 多候选投票 |
| `evaluation_logger.py` | V10 | AgentBench / HELM | 全自动报告生成 |
| **`integrated_handler.py`** | **⭐V10** | **以上全部** | **统一集成入口** |

---

## 🚀 快速开始

### 第一步：确认 Ollama 已安装

```bash
# 检查 Ollama 是否在运行
ollama ps

# 如果没有，启动它
ollama serve
```

### 第二步：修改配置

编辑 `config.py`，修改模型名称：

```python
OLLAMA_CONFIG = {
    "model": "qwen2.5:7b",  # 改成你拉取的模型名
    # ...
}
```

### 第三步：运行测试

```bash
# 在 code 目录下
python test_ollama.py
```

如果看到 "✅ 所有测试通过"，说明环境没问题。

### 第四步：测试 Prompt

```bash
python prompt_templates.py
```

### 第五步：集成到仿真框架

这是核心步骤。你需要把 `dialog_handler_template.py` 中的代码集成到框架中。

**对话式规划（comm_mode=dialog）：**

在 `dialog_prompter.py` 顶部添加导入：
```python
import sys
sys.path.append('/path/to/code')  # 指向你的code目录
from dialog_handler_template import DialogHandler
```

在初始化方法中创建 handler：
```python
self.dialog_handler = DialogHandler(model="qwen2.5:7b")
```

在第 264-267 行位置填写：
```python
def get_llm_response(self, observation, robot_id):
    return self.dialog_handler.get_action(robot_id, observation)
```

**集中式规划（comm_mode=plan）：**

在 `plan_prompter.py` 顶部添加导入：
```python
import sys
sys.path.append('/path/to/code')
from dialog_handler_template import PlanHandler
```

在初始化方法中创建 handler：
```python
self.plan_handler = PlanHandler(model="qwen2.5:7b")
```

在第 245-248 行位置填写：
```python
def get_llm_plan(self, observation):
    return self.plan_handler.get_all_actions(observation)
```

---

## 🔧 进阶用法

### 使用增强版处理器（含创新点）

```python
from dialog_handler_template import EnhancedPlanHandler

handler = EnhancedPlanHandler(
    model="qwen2.5:7b",
    memory_file="experience_memory.json"  # 经验记忆库
)
```

增强版功能：
- ✅ 自动保存成功案例作为 Few-Shot 示例
- ✅ 失败后自动反思（降低 temperature）
- ✅ 动态温度调整

### 切换到 70B 模型

只需修改 config.py：
```python
OLLAMA_CONFIG = {
    "model": "llama3.3:70b",  # 或 "qwen2.5:72b"
}
```

### 自定义 Prompt

编辑 `prompt_templates.py` 中的 System Prompt 模板。
记得保存原版备份！

---

## 📊 调试技巧

1. **查看 LLM 日志**：设置 `config.py` 中 `SAVE_LLM_LOG = True`，每次调用会保存到 `llm_logs/`
2. **降低模型大小以加速迭代**：开发时用 `qwen2.5:0.5b`（超快），调试 Prompt 结构
3. **打印详细日志**：`config.py` 中 `DEBUG = True`
4. **单独测试 Prompt**：修改 `prompt_templates.py` 底部的测试代码

---

## ❓ 常见问题

**Q: 提示 ModuleNotFoundError: No module named 'openai'**
```bash
pip install openai requests
```

**Q: 提示 Connection refused**
Ollama 服务未启动，运行：
```bash
ollama serve
```

**Q: 提示模型不存在**
先下载模型：
```bash
ollama pull qwen2.5:7b
```

**Q: 如何查看已下载的模型？**
```bash
ollama list
```

**Q: LLM 返回的动作不符合预期？**
1. 检查 Prompt 中可用技能是否和任务实际技能一致
2. 降低 temperature（改为 0.0 试试）
3. 在 System Prompt 中增加更严格的输出格式约束
