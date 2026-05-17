"""
配置文件 - 多机器人协同操作项目
===================================
修改这里的配置来适配你的环境。
所有代码文件都会引用这个配置。

使用方法：
    from config import OLLAMA_CONFIG, TASK_ORDER
"""

# ==================== Ollama 配置 ====================
OLLAMA_CONFIG = {
    # Ollama 服务地址
    # 本地运行: http://localhost:11434
    # 云端实例: http://localhost:11434 (如果在同一台机器上)
    "base_url": "http://localhost:11434/v1",
    
    # API Key（Ollama 本地运行时随意填写）
    "api_key": "ollama",
    
    # 模型选择
    # 开发调试用: "qwen2.5:7b"
    # 正式评测用: "llama3.3:70b" 或 "qwen2.5:72b"
    "model": "qwen2.5:7b",
    
    # 推理参数
    "temperature": 0.1,      # 越低越确定，建议 0.0~0.3
    "max_tokens": 2048,      # 最大输出token数
    "timeout": 120,          # 请求超时时间（秒）
}

# ==================== 任务配置 ====================

# 任务练习顺序（从简单到困难）
TASK_ORDER = [
    "sweep_floor",      # 1. 清扫地板 - 最简单，动作空间小
    "sort_cubes",       # 2. 分类立方体 - 清晰的空间逻辑
    "make_sandwich",    # 3. 做三明治 - 需要顺序推理
    "pack_grocery",     # 4. 打包杂货 - 需要碰撞避免
    "move_rope",        # 5. 移动绳子 - 需要精细协调
    "arrange_cabinet",  # 6. 整理柜子 - 三个机器人，最复杂
]

# 任务中文名称映射
TASK_NAMES_CN = {
    "sweep_floor": "清扫地板",
    "sort_cubes": "分类立方体",
    "make_sandwich": "做三明治",
    "pack_grocery": "打包杂货",
    "move_rope": "移动绳子",
    "arrange_cabinet": "整理柜子",
}

# 每个任务的运行次数（调试用1次，正式评测用10次以上）
DEFAULT_NUM_RUNS = 1
EVAL_NUM_RUNS = 10

# ==================== 路径配置 ====================

# 项目根目录（如果你的路径不同，修改这里）
PROJECT_ROOT = None  # None 表示自动检测

# Prompt 模板目录
PROMPT_DIR = "prompts"

# 实验结果保存目录
RESULT_DIR = "results"

# 经验记忆库路径
MEMORY_FILE = "experience_memory.json"

# ==================== 调试配置 ====================

# 是否打印详细的调试信息
DEBUG = True

# 是否保存每次 LLM 调用的输入输出日志
SAVE_LLM_LOG = True

# LLM 日志目录
LLM_LOG_DIR = "llm_logs"
