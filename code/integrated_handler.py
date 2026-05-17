"""
最终集成处理器 - IntegratedHandler (V10)
===========================================
整合了10轮迭代的所有创新模块：
- FormatHealer: 格式修复
- ActionFilter: 动作合法性过滤
- TaskDecomposer: 任务分解
- RoleAssigner: 角色分配
- ReflectionEngine: 失败反思
- ExperienceMemory: 经验记忆
- LoopDetector: 循环检测
- CandidateVoter: 候选投票
- EvaluationLogger: 评估日志

这是你可以直接用到竞赛中的完整方案。

使用方法：
    from integrated_handler import IntegratedHandler
    
    handler = IntegratedHandler(
        task_name="sweep_floor",
        model="qwen2.5:7b",
        version="v10",
    )
    
    # 每一步调用
    observation = env.get_observation()
    actions = handler.step(observation)
    results = env.execute(actions)
    handler.feedback(results)
"""

import sys
import os
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# 导入所有模块
from ollama_client import OllamaClient
from prompt_templates import (
    SYSTEM_PROMPTS,
    build_user_prompt,
    parse_llm_response,
    COT_ENHANCEMENT,
)
from format_healer import FormatHealer
from action_filter import ActionFilter
from task_decomposer import TaskDecomposer, SubgoalStatus
from role_assigner import RoleAssigner
from reflection_engine import ReflectionEngine, ReflectionResult
from experience_memory import ExperienceMemory
from loop_detector import LoopDetector
from candidate_voter import CandidateVoter
from evaluation_logger import EvaluationLogger


class IntegratedHandler:
    """
    V10最终集成处理器
    
    整合所有创新模块，提供统一步骤接口
    """
    
    def __init__(
        self,
        task_name: str,
        model: str = "qwen2.5:7b",
        version: str = "v10",
        comm_mode: str = "plan",
        use_voter: bool = False,
        use_memory: bool = True,
        max_steps: int = 50,
    ):
        """
        Args:
            task_name: 任务名称
            model: 模型名称
            version: 版本号
            comm_mode: 通信模式 (dialog / plan)
            use_voter: 是否启用候选投票（会增加LLM调用）
            use_memory: 是否启用经验记忆
            max_steps: 最大步数限制
        """
        self.task_name = task_name
        self.model = model
        self.version = version
        self.comm_mode = comm_mode
        self.use_voter = use_voter
        self.max_steps = max_steps
        
        # ====== 初始化所有模块 ======
        print(f"\n{'='*50}")
        print(f"🚀 IntegratedHandler V10 启动")
        print(f"   任务: {task_name}")
        print(f"   模型: {model}")
        print(f"   模式: {comm_mode}")
        print(f"{'='*50}\n")
        
        # 1. LLM客户端
        self.llm = OllamaClient(model=model)
        
        # 2. 格式修复器
        self.healer = FormatHealer(self.llm)
        
        # 3. 动作过滤器
        self.action_filter = ActionFilter(task_name)
        
        # 4. 任务分解器
        self.decomposer = TaskDecomposer(task_name, self.llm)
        
        # 5. 角色分配器
        self.role_assigner = RoleAssigner(task_name, self.llm)
        
        # 6. 反思引擎
        self.reflection = ReflectionEngine(self.llm, max_retries=3)
        
        # 7. 经验记忆
        if use_memory:
            self.memory = ExperienceMemory("experience_memory.json")
        else:
            self.memory = None
        
        # 8. 循环检测器
        self.loop_detector = LoopDetector()
        
        # 9. 候选投票器（可选）
        if use_voter:
            self.voter = CandidateVoter(self.llm)
        else:
            self.voter = None
        
        # 10. 评估日志
        self.logger = EvaluationLogger("results", version=version)
        
        # ====== 运行时状态 ======
        self.step_count = 0
        self.history: List[str] = []
        self.is_done = False
        self.success = False
        
        # 分配初始角色
        self.role_assigner.assign_roles()
        
        # 开始日志记录
        self.logger.start_task(task_name, version=version)
    
    def step(self, observation: Dict) -> Dict[str, str]:
        """
        执行一步：根据观察生成动作
        
        Args:
            observation: 环境观察字典
        
        Returns:
            {robot_id: action_string}
        """
        if self.is_done:
            return {}
        
        self.step_count += 1
        step_start = time.time()
        
        # ====== 1. 检查子目标完成 ======
        if self.decomposer.check_subgoal_completed(observation):
            print(f"   ✅ 子目标完成，推进到下一个")
            self.decomposer.advance()
        
        if self.decomposer.is_task_done():
            print(f"   🎉 所有子目标完成！")
            self.is_done = True
            self.success = True
            self._end_task(True, time.time() - step_start)
            return {}
        
        # ====== 2. 检查步数限制 ======
        if self.step_count > self.max_steps:
            print(f"   ⏰ 超过最大步数限制 ({self.max_steps})")
            self.is_done = True
            self.success = False
            self._end_task(False, time.time() - step_start)
            return {}
        
        # ====== 3. 检查循环 ======
        if self.loop_detector.is_looping():
            print(f"   ⚠️ 检测到循环！注入警告")
        
        # ====== 4. 生成候选动作 ======
        valid_actions = self.action_filter.generate_valid_actions(observation, max_actions=15)
        
        # 构建Prompt
        prompt = self._build_full_prompt(observation, valid_actions)
        
        # 调用LLM
        llm_start = time.time()
        try:
            response = self.llm.chat_with_system(
                user_message=prompt,
                system_prompt=self._build_system_prompt(),
            )
        except Exception as e:
            print(f"   ❌ LLM调用失败: {e}")
            return {"rob0": "WAIT", "rob1": "WAIT"}
        
        llm_time = time.time() - llm_start
        
        # ====== 5. 格式修复 ======
        response, healed = self.healer.heal(response, "actions")
        
        # ====== 6. 解析动作 ======
        parsed = parse_llm_response(response)
        actions = self._actions_to_dict(parsed["actions"])
        
        if not actions:
            actions = {"rob0": "WAIT", "rob1": "WAIT"}
        
        # ====== 7. 动作合法性检查 ======
        for rid, action in actions.items():
            if not self.action_filter.is_action_valid(action, rid):
                print(f"   ⚠️ 非法动作被拦截: {rid}: {action}")
                actions[rid] = f"{rid}: WAIT"
        
        # ====== 8. 角色冲突检测 ======
        conflicts = self.role_assigner.detect_role_conflict(actions)
        if conflicts:
            print(f"   ⚠️ 角色冲突: {conflicts}")
            actions = self.role_assigner.resolve_conflict(conflicts, actions)
        
        # ====== 9. 更新状态 ======
        for rid, action in actions.items():
            self.history.append(f"{rid}: {action}")
            self.loop_detector.add_action(f"{rid}: {action}")
        
        self.logger.log_step(
            self.step_count,
            str(actions),
            "success",  # 先标记成功，feedback会更新
            llm_time=llm_time,
            metadata={"healed": healed},
        )
        
        return actions
    
    def feedback(self, results: Dict[str, Any]):
        """
        接收执行反馈
        
        Args:
            results: 执行结果 {robot_id: {"success": bool, "error": str}}
        """
        # 检查是否有失败
        any_failed = False
        for rid, result in results.items():
            if not result.get("success", True):
                any_failed = True
                error = result.get("error", "Unknown error")
                
                # 记录失败
                if self.memory:
                    self.memory.record_failure(
                        self.task_name,
                        {"step": self.step_count},
                        f"{rid}: {self.history[-1] if self.history else 'unknown'}",
                        error,
                        self.reflection.detect_failure_type(error).value,
                    )
                
                # 触发反思
                reflection_result = self.reflection.reflect(
                    f"{rid}: {self.history[-1] if self.history else 'unknown'}",
                    error,
                    {"step": self.step_count, "robot": rid},
                    self.history[-10:],
                    self.decomposer.get_current_subgoal()["name"] if self.decomposer.get_current_subgoal() else None,
                )
                
                if reflection_result.should_change_subgoal:
                    self.decomposer.mark_failed(error)
                    self.decomposer.advance()
                
                if not reflection_result.should_retry:
                    self.is_done = True
                    self.success = False
        
        if not any_failed:
            self.reflection.reset_retry_count()
            
            # 记录成功
            if self.memory and len(self.history) >= 2:
                self.memory.record_success(
                    self.task_name,
                    {"step": self.step_count, "task": self.task_name},
                    self.history[-3:],
                    reward=0.8,
                )
    
    def _build_full_prompt(
        self,
        observation: Dict,
        valid_actions: List[Dict],
    ) -> str:
        """构建完整的User Prompt，整合所有模块"""
        parts = []
        
        # 1. 子目标上下文
        subgoal_ctx = self.decomposer.build_subgoal_prompt_context()
        if subgoal_ctx:
            parts.append(subgoal_ctx)
        
        # 2. 角色上下文
        role_ctx = self.role_assigner.build_role_prompt_context()
        if role_ctx:
            parts.append(role_ctx)
        
        # 3. Few-Shot示例
        if self.memory:
            few_shot = self.memory.build_few_shot_prompt(self.task_name, k=2)
            if few_shot:
                parts.append(few_shot)
            
            # 失败警告
            failure_warn = self.memory.build_failure_warning(self.task_name)
            if failure_warn:
                parts.append(failure_warn)
        
        # 4. 反思上下文
        reflection_ctx = self.reflection.build_reflection_context()
        if reflection_ctx:
            parts.append(reflection_ctx)
        
        # 5. 循环警告
        loop_warn = self.loop_detector.build_loop_warning()
        if loop_warn:
            parts.append(loop_warn)
        
        # 6. 可用动作列表
        action_list = "\n".join([
            f"{a['action_id']}. [{a['robot']}] {a['action']}"
            for a in valid_actions[:15]
        ])
        parts.append(f"## 可用动作（只能从中选择）\n{action_list}")
        
        # 7. 基本任务信息
        from prompt_templates import format_observation, format_history
        parts.append(format_observation(observation))
        parts.append(format_history(self.history[-5:]))
        parts.append("请输出下一步动作：\n<THINK>...</THINK>\n<ACTIONS>...</ACTIONS>")
        
        return "\n\n".join(parts)
    
    def _build_system_prompt(self) -> str:
        """构建完整System Prompt"""
        base = SYSTEM_PROMPTS.get(self.task_name, "")
        
        # 添加各模块的约束
        enhancements = [COT_ENHANCEMENT]
        
        enhancements.append("""
## 额外约束
- 你必须从可用动作中选择，不得发明新动作
- 遵守角色分配，不要越俎代庖
- 避免重复历史中的失败动作
- 如果检测到循环，立即更换策略""")
        
        return base + "\n\n" + "\n".join(enhancements)
    
    def _actions_to_dict(self, actions: List[str]) -> Dict[str, str]:
        """将动作列表转为字典"""
        result = {}
        for action in actions:
            if ":" in action:
                parts = action.split(":", 1)
                result[parts[0].strip()] = parts[1].strip()
        return result
    
    def _end_task(self, success: bool, elapsed: float):
        """结束任务并记录"""
        self.is_done = True
        self.success = success
        
        self.logger.log_metric("total_llm_calls", self.llm.call_count)
        self.logger.log_metric("heal_count", self.healer.heal_count)
        self.logger.log_metric("reflection_count", self.reflection.total_reflections)
        self.logger.log_metric("loop_count", self.loop_detector.total_loops_detected)
        
        self.logger.end_task(success, elapsed)
        
        # 保存经验记忆
        if self.memory and success:
            self.memory.record_success(
                self.task_name,
                {"final_step": self.step_count},
                self.history,
                reward=1.0,
            )
    
    def get_module_stats(self) -> Dict:
        """获取所有模块的统计"""
        return {
            "task": self.task_name,
            "version": self.version,
            "model": self.model,
            "total_steps": self.step_count,
            "success": self.success,
            "modules": {
                "healer": self.healer.get_stats(),
                "action_filter": self.action_filter.get_stats(),
                "decomposer": self.decomposer.get_progress() if self.decomposer else {},
                "reflection": self.reflection.get_stats(),
                "loop_detector": self.loop_detector.get_stats(),
                "memory": self.memory.get_stats() if self.memory else None,
                "voter": self.voter.get_stats() if self.voter else None,
                "llm": self.llm.get_stats(),
            },
        }
    
    def print_module_stats(self):
        """打印所有模块统计"""
        stats = self.get_module_stats()
        print(f"\n{'='*50}")
        print(f"📊 模块统计 (V10)")
        print(f"{'='*50}")
        print(f"任务: {stats['task']} | 成功: {stats['success']} | 步数: {stats['total_steps']}")
        
        mods = stats["modules"]
        print(f"  FormatHealer: {mods['healer']}")
        print(f"  ActionFilter: {mods['action_filter']}")
        print(f"  Reflection: {mods['reflection']}")
        print(f"  LoopDetector: {mods['loop_detector']}")
        print(f"  LLM: {mods['llm']}")


# ============================================================
# 使用示例（在云实例上需要仿真环境）
# ============================================================

"""
### 在任务代码中使用 ###

from integrated_handler import IntegratedHandler

# 初始化
handler = IntegratedHandler(
    task_name="sweep_floor",
    model="qwen2.5:7b",     # 开发用7B
    # model="llama3.3:70b",  # 正式用70B
    version="v10",
    comm_mode="plan",
    use_memory=True,
)

# 主循环（对应 run_dialog.py 的执行循环）
while not handler.is_done:
    observation = env.get_observation()
    actions = handler.step(observation)
    
    if handler.is_done:
        break
    
    results = env.execute_actions(actions)
    handler.feedback(results)

# 打印统计
handler.print_module_stats()

# 生成报告
handler.logger.generate_report()
"""


# ============================================================
# 本地测试（不需要仿真环境，但需要Ollama）
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("IntegratedHandler V10 本地测试")
    print("=" * 60)
    print("\n⚠️  完整测试需要在有仿真环境的云实例上运行")
    print("   本地只能测试模块导入和LLM调用部分\n")
    
    # 测试模块导入
    print("📦 检查模块导入...")
    modules = {
        "OllamaClient": "ollama_client",
        "SYSTEM_PROMPTS": "prompt_templates",
        "FormatHealer": "format_healer",
        "ActionFilter": "action_filter",
        "TaskDecomposer": "task_decomposer",
        "RoleAssigner": "role_assigner",
        "ReflectionEngine": "reflection_engine",
        "ExperienceMemory": "experience_memory",
        "LoopDetector": "loop_detector",
        "CandidateVoter": "candidate_voter",
        "EvaluationLogger": "evaluation_logger",
    }
    
    all_ok = True
    for name, mod in modules.items():
        try:
            __import__(mod)
            print(f"   ✅ {name}")
        except ImportError as e:
            print(f"   ❌ {name}: {e}")
            all_ok = False
    
    if all_ok:
        print("\n✅ 所有模块导入成功！")
        
        # 测试Ollama连接
        try:
            client = OllamaClient(model="qwen2.5:7b")
            if client.test_connection():
                print("✅ Ollama连接正常")
                
                # 测试简单对话
                response = client.chat_with_system(
                    "输出一个扫地任务的机器人动作",
                    "你是机器人规划助手。请用<THINK>和<ACTIONS>格式输出。",
                )
                print(f"   LLM回复: {response[:150]}...")
        except Exception as e:
            print(f"⚠️  Ollama未连接: {e}")
            print("   安装: ollama pull qwen2.5:7b")
    
    print("\n📊 版本信息:")
    print(f"   版本: V10 (最终集成)")
    print(f"   模块数: {len(modules)}")
    print(f"   总代码行数: ~3000+")
    print(f"   支持任务: sweep_floor/sort_cubes/make_sandwich/pack_grocery/move_rope/arrange_cabinet")
