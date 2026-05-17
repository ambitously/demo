"""
对话处理器模板
=================
这是你需要填写的核心代码。
对应题目要求的：
  - dialog_prompter.py 第 264-267 行（对话式规划）
  - plan_prompter.py 第 245-248 行（集中式规划）

这个文件展示了完整的实现模式。你需要根据实际代码框架调整接口。

核心思路：
1. 接收仿真环境传来的 observation（机器人位置、物体位置等）
2. 将 observation 格式化为 LLM 可以理解的文本
3. 调用 Ollama（通过 ollama_client.py）
4. 解析 LLM 返回的动作序列
5. 将动作序列返回给仿真环境执行

注意：
- 这个文件是模板，不能直接运行，需要根据实际代码框架调整
- 关键是你需要理解这个模式，然后在 dialog_prompter.py 的指定位置填写类似代码
"""

import json
import sys
import os

# 添加 code 目录到 path（如果从其他目录运行的话）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ollama_client import OllamaClient
from prompt_templates import (
    SYSTEM_PROMPTS,
    build_user_prompt,
    parse_llm_response,
    COT_ENHANCEMENT,
    SELF_REFLECTION_PROMPT,
)


# ============================================================
# 方式一：对话式规划 (comm_mode=dialog)
# 对应 dialog_prompter.py 第 264-267 行
# ============================================================

class DialogHandler:
    """
    对话式规划处理器
    
    每个机器人"独立思考"并"互相讨论"后执行动作。
    使用 Multi-Agent 模式：每个机器人有自己的 LLM "大脑"。
    """
    
    def __init__(self, model: str = "qwen2.5:7b"):
        """
        初始化对话处理器
        
        Args:
            model: LLM 模型名称
        """
        # 每个机器人一个独立的 LLM 客户端实例
        # 可以给不同机器人不同的温度参数，模拟"性格差异"
        self.robot_clients = {
            "rob0": OllamaClient(model=model, temperature=0.1),
            "rob1": OllamaClient(model=model, temperature=0.1),
            "rob2": OllamaClient(model=model, temperature=0.1),  # 仅某些任务有
        }
        
        # 协调者 LLM（用于解决机器人之间的冲突）
        self.coordinator = OllamaClient(model=model, temperature=0.0)
        
        self.model = model
        self.task_name = None
        self.history = []  # 动作历史
    
    def set_task(self, task_name: str):
        """设置当前任务"""
        self.task_name = task_name
        self.history = []
        print(f"📋 设置任务: {task_name}")
    
    def get_action(self, robot_id: str, observation: dict) -> str:
        """
        为指定机器人生成下一步动作
        这是你在 dialog_prompter.py 第 264-267 行需要写的函数
        
        Args:
            robot_id: 机器人 ID (如 "rob0")
            observation: 环境观察字典
        
        Returns:
            动作字符串 (如 "rob0: MOVE near_cube_1")
        """
        # 1. 获取 System Prompt
        system_prompt = SYSTEM_PROMPTS.get(self.task_name, "")
        
        # 2. 构建 User Prompt
        user_prompt = build_user_prompt(
            task_name=self.task_name,
            observation=observation,
            history=self.history[-5:],  # 只保留最近5步
        )
        
        # 3. 调用 LLM
        client = self.robot_clients.get(robot_id)
        if not client:
            print(f"⚠️  未知机器人: {robot_id}")
            return f"{robot_id}: WAIT"
        
        try:
            response = client.chat_with_system(
                user_message=user_prompt,
                system_prompt=system_prompt,
            )
        except Exception as e:
            print(f"❌ LLM 调用失败: {e}")
            return f"{robot_id}: WAIT"  # 失败时等待
        
        # 4. 解析动作
        parsed = parse_llm_response(response)
        actions = parsed["actions"]
        
        # 5. 提取本机器人的动作
        my_action = None
        for action in actions:
            if action.startswith(f"{robot_id}:"):
                my_action = action
                break
        
        # 如果 LLM 没有给出本机器人的动作，默认等待
        if my_action is None:
            my_action = f"{robot_id}: WAIT"
        
        # 6. 记录历史
        self.history.append(my_action)
        
        return my_action
    
    def coordinate(self, robot_plans: dict, observation: dict) -> dict:
        """
        协调多个机器人的计划
        当机器人之间的计划可能冲突时调用
        
        Args:
            robot_plans: {robot_id: action_string}
            observation: 当前环境观察
        
        Returns:
            协调后的动作 {robot_id: action_string}
        """
        # 检查是否有冲突（简单检查：是否有两个机器人同时去同一位置）
        targets = {}
        for rid, action in robot_plans.items():
            if "MOVE" in action:
                target = action.split("MOVE")[-1].strip()
                if target in targets:
                    # 冲突！让协调者LLM来解决
                    print(f"⚠️  冲突检测: {rid} 和 {targets[target]} 都要移动到 {target}")
                    
                    prompt = f"""两个机器人的计划冲突：
- {targets[target]}: {robot_plans[targets[target]]}
- {rid}: {action}

请决定哪个机器人应该先执行，另一个应该 WAIT。
输出格式：
<DECISION>
robX: 先执行原来的动作
robY: WAIT
</DECISION>"""
                    
                    try:
                        response = self.coordinator.chat(
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.0,
                        )
                        # 解析协调结果
                        coordinated = {}
                        for line in response.split("\n"):
                            if ":" in line and ("MOVE" in line or "WAIT" in line):
                                parts = line.split(":", 1)
                                coordinated[parts[0].strip()] = parts[1].strip()
                        
                        if coordinated:
                            return coordinated
                    except Exception as e:
                        print(f"⚠️  协调失败: {e}")
                
                targets[target] = rid
        
        return robot_plans  # 没有冲突，返回原计划


# ============================================================
# 方式二：集中式规划 (comm_mode=plan)
# 对应 plan_prompter.py 第 245-248 行
# ============================================================

class PlanHandler:
    """
    集中式规划处理器
    
    一个 LLM 同时为所有机器人规划动作。
    这是更简单、更直接的方式，适合初学者。
    """
    
    def __init__(self, model: str = "qwen2.5:7b"):
        """初始化规划器"""
        self.client = OllamaClient(model=model, temperature=0.1)
        self.model = model
        self.task_name = None
        self.history = []
        self.failure_count = 0
    
    def set_task(self, task_name: str):
        """设置当前任务"""
        self.task_name = task_name
        self.history = []
        self.failure_count = 0
        print(f"📋 设置任务: {task_name}")
    
    def get_all_actions(self, observation: dict) -> dict:
        """
        为所有机器人生成动作（集中式）
        这是你在 plan_prompter.py 第 245-248 行需要写的函数
        
        Args:
            observation: 环境观察字典
        
        Returns:
            {robot_id: action_string} 如 {"rob0": "MOVE ...", "rob1": "WAIT"}
        """
        # 1. 获取 System Prompt
        system_prompt = SYSTEM_PROMPTS.get(self.task_name, "")
        
        # 2. 添加 CoT 思维链要求（创新点！）
        system_prompt += COT_ENHANCEMENT
        
        # 3. 如果之前失败过，添加反思要求
        if self.failure_count > 0:
            system_prompt += f"\n\n⚠️ 此前已有 {self.failure_count} 次失败，请反思原因并改进策略。"
        
        # 4. 构建 User Prompt
        user_prompt = build_user_prompt(
            task_name=self.task_name,
            observation=observation,
            history=self.history[-10:],  # 集中式可以保留更多历史
        )
        
        # 5. 调用 LLM
        try:
            response = self.client.chat_with_system(
                user_message=user_prompt,
                system_prompt=system_prompt,
            )
        except Exception as e:
            print(f"❌ LLM 调用失败: {e}")
            return {f"rob{i}": "WAIT" for i in range(3)}
        
        # 6. 解析动作
        parsed = parse_llm_response(response)
        actions = parsed["actions"]
        
        # 7. 转换为字典格式
        action_dict = {}
        for action in actions:
            if ":" in action:
                parts = action.split(":", 1)
                robot_id = parts[0].strip()
                action_str = parts[1].strip()
                action_dict[robot_id] = action_str
        
        # 8. 记录历史
        self.history.append(action_dict)
        
        return action_dict
    
    def record_failure(self):
        """记录一次失败（用于触发反思机制）"""
        self.failure_count += 1
        print(f"⚠️  失败计数: {self.failure_count}")
    
    def record_success(self):
        """记录成功"""
        self.failure_count = 0
        self.history = []  # 成功完成任务，清空历史


# ============================================================
# 方式三：增强型集中式规划（推荐最终使用）
# 整合所有创新点：CoT + Few-Shot + Self-Reflection
# ============================================================

class EnhancedPlanHandler(PlanHandler):
    """
    增强型集中式规划处理器
    
    在 PlanHandler 基础上增加：
    1. 经验记忆库（成功案例 Few-Shot）
    2. 自适应温度（失败后降低温度提高稳定性）
    """
    
    def __init__(self, model: str = "qwen2.5:7b", memory_file: str = "experience_memory.json"):
        super().__init__(model)
        self.memory = self._load_memory(memory_file)
        self.memory_file = memory_file
    
    def _load_memory(self, filepath: str) -> list:
        """加载经验记忆库"""
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
    
    def _save_memory(self):
        """保存经验记忆库"""
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=2)
    
    def _get_few_shot_examples(self, k: int = 3) -> list:
        """获取最相关的 k 个成功案例"""
        task_memories = [m for m in self.memory if m.get("task") == self.task_name]
        # 简单策略：返回最近的 k 个成功案例
        return task_memories[-k:] if task_memories else []
    
    def get_all_actions(self, observation: dict) -> dict:
        """增强版生成函数"""
        
        system_prompt = SYSTEM_PROMPTS.get(self.task_name, "")
        system_prompt += COT_ENHANCEMENT
        
        # 添加 Few-Shot 示例（创新点！）
        examples = self._get_few_shot_examples(k=2)
        if examples:
            system_prompt += "\n\n## 历史成功案例\n"
            for i, ex in enumerate(examples):
                system_prompt += f"\n案例{i+1}：\n状态: {ex.get('state', '')}\n动作: {ex.get('actions', '')}\n"
            system_prompt += "\n请参考以上成功案例来规划动作。"
        
        # 失败反思
        if self.failure_count > 0:
            system_prompt += f"\n\n⚠️ 此前已有 {self.failure_count} 次连续失败。"
            system_prompt += "请分析失败原因，尝试不同的策略。"
        
        # 动态温度：失败后降低温度（更保守）
        current_temp = max(0.0, 0.1 - self.failure_count * 0.02)
        
        user_prompt = build_user_prompt(
            task_name=self.task_name,
            observation=observation,
            history=self.history[-10:],
        )
        
        try:
            response = self.client.chat_with_system(
                user_message=user_prompt,
                system_prompt=system_prompt,
                temperature=current_temp,
            )
        except Exception as e:
            print(f"❌ LLM 调用失败: {e}")
            return {f"rob{i}": "WAIT" for i in range(3)}
        
        parsed = parse_llm_response(response)
        actions = parsed["actions"]
        
        action_dict = {}
        for action in actions:
            if ":" in action:
                parts = action.split(":", 1)
                action_dict[parts[0].strip()] = parts[1].strip()
        
        self.history.append(action_dict)
        
        # 如果动作合理且之前没有失败过，记录为潜在成功案例
        if self.failure_count == 0 and len(action_dict) > 0:
            self.memory.append({
                "task": self.task_name,
                "state": str(observation)[:500],  # 截断防止太大
                "actions": str(action_dict),
                "timestamp": __import__('datetime').datetime.now().isoformat(),
            })
            # 只保留最近的 50 条记忆
            if len(self.memory) > 50:
                self.memory = self.memory[-50:]
            self._save_memory()
        
        return action_dict


# ============================================================
# 使用示例（需要在云实例上运行，因为依赖仿真环境）
# ============================================================

"""
### 在 dialog_prompter.py 第 264-267 行的标准写法 ###

# 在文件顶部导入
from dialog_handler import DialogHandler

# 在初始化部分创建 handler（只创建一次）
# 把这行放在 __init__ 或 setup 方法中
self.dialog_handler = DialogHandler(model="qwen2.5:7b")

# 在 264-267 行位置，替换原来的占位代码：
def get_llm_response(self, observation, robot_id):
    '''
    原来这里是空的或者占位代码
    你在这里填写：
    '''
    return self.dialog_handler.get_action(robot_id, observation)


### 在 plan_prompter.py 第 245-248 行的标准写法 ###

# 在文件顶部导入
from dialog_handler import PlanHandler  # 或 EnhancedPlanHandler

# 在初始化部分
self.plan_handler = PlanHandler(model="qwen2.5:7b")

# 在 245-248 行位置：
def get_llm_plan(self, observation):
    '''
    原来这里是空的或者占位代码
    你在这里填写：
    '''
    return self.plan_handler.get_all_actions(observation)
"""


# ============================================================
# 本地测试（不需要仿真环境）
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("对话处理器模板 - 本地测试")
    print("=" * 60)
    print()
    print("⚠️  注意：完整测试需要在有仿真环境的云实例上运行")
    print("   本地只能测试 LLM 调用部分（需要安装 Ollama 并下载模型）")
    print()
    
    # 测试是否能连接到 Ollama
    try:
        from ollama_client import OllamaClient
        client = OllamaClient(model="qwen2.5:7b")
        
        if client.test_connection():
            print("\n✅ Ollama 连接正常，可以测试 LLM 调用")
            
            # 测试集中式规划
            print("\n📋 测试集中式规划...")
            handler = PlanHandler(model="qwen2.5:7b")
            handler.set_task("sweep_floor")
            
            test_obs = {
                "robot_positions": "rob0: 在桌子左侧, rob1: 在桌子右侧",
                "cube_states": "cube_1: 在桌子中央",
                "task_stage": "准备开始",
            }
            
            actions = handler.get_all_actions(test_obs)
            print(f"   动作: {actions}")
            
        else:
            print("\n❌ Ollama 未连接，跳过测试")
            print("   请先安装 Ollama 并拉取模型：")
            print("   ollama pull qwen2.5:7b")
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        print("   这是正常的——完整测试需要在云实例上运行")
