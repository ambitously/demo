# 多机器人协同操作 - 代码使用指南

## 📁 文件说明

```
code/
├── config.py                    # ⭐ 配置文件 - 修改模型/地址等参数
├── ollama_client.py             # Ollama LLM 客户端封装
├── prompt_templates.py          # ⭐ 所有Prompt模板（System + User + 解析）
├── dialog_handler_template.py   # ⭐ 核心处理器模板 - 你需要填写的代码
└── test_ollama.py               # Ollama 连接测试脚本
```

| 文件 | 作用 | 你需要在云实例上运行吗？ | 需要修改吗？ |
|------|------|:---:|:---:|
| `config.py` | 配置模型名称、API地址等 | ✅ | ⚠️ 需要（改为你的模型名） |
| `ollama_client.py` | 封装 LLM API 调用 | ✅ | ❌ 不需要 |
| `prompt_templates.py` | 所有 Prompt 模板 | ✅ | ⚠️ 根据效果调整 |
| `dialog_handler_template.py` | 处理器核心代码 | ✅ | ⚠️ 需要（集成到框架） |
| `test_ollama.py` | 测试 Ollama 连接 | 本地和云端都可以 | ❌ 不需要 |

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
