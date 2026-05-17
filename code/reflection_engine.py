"""
反思引擎 - ReflectionEngine
=============================
借鉴：Inner Monologue / Reflexion / Self-Refine
创新点：失败后不放弃，而是自动分析原因并重规划

核心循环：
1. 执行动作
2. 如果失败 → LLM分析失败原因
3. LLM生成修正策略
4. 重新执行
5. 如果连续失败N次 → 放弃当前子目标

解决问题：
- 抓取失败后不知道换策略
- 碰撞后重复相同动作
- 路径被挡后不会绕行

使用方法：
    from reflection_engine import ReflectionEngine
    re = ReflectionEngine(llm_client)
    new_plan = re.reflect(failed_action, error_msg, observation, history)
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class FailureType(Enum):
    COLLISION = "collision"           # 碰撞
    OUT_OF_REACH = "out_of_reach"     # 超出范围
    GRASP_FAILED = "grasp_failed"     # 抓取失败
    WRONG_OBJECT = "wrong_object"     # 抓错物体
    WRONG_PLACEMENT = "wrong_placement" # 放错位置
    TIMEOUT = "timeout"               # 超时
    INVALID_ACTION = "invalid_action" # 非法动作
    LOOP_DETECTED = "loop_detected"   # 检测到循环
    UNKNOWN = "unknown"               # 未知错误


@dataclass
class ReflectionResult:
    """反思结果"""
    original_action: str
    failure_type: FailureType
    failure_reason: str
    suggested_fix: str
    new_action: Optional[str] = None
    should_retry: bool = True
    should_change_subgoal: bool = False


class ReflectionEngine:
    """失败反思与重规划引擎"""
    
    # 失败类型检测规则
    FAILURE_PATTERNS = {
        FailureType.COLLISION: [
            "collision", "碰撞", "collide", "bump", "撞",
        ],
        FailureType.OUT_OF_REACH: [
            "out of reach", "超出范围", "too far", "距离", "cannot reach",
        ],
        FailureType.GRASP_FAILED: [
            "grasp fail", "抓取失败", "grip fail", "slip", "掉落", "drop",
        ],
        FailureType.WRONG_OBJECT: [
            "wrong object", "错误物体", "incorrect target",
        ],
        FailureType.WRONG_PLACEMENT: [
            "wrong place", "错误位置", "incorrect placement",
        ],
        FailureType.TIMEOUT: [
            "timeout", "超时", "time limit",
        ],
        FailureType.INVALID_ACTION: [
            "invalid action", "非法动作", "unknown action",
        ],
        FailureType.LOOP_DETECTED: [
            "repeating", "重复", "loop", "循环", "same action",
        ],
    }
    
    def __init__(self, llm_client=None, max_retries: int = 3):
        """
        Args:
            llm_client: OllamaClient实例
            max_retries: 最大连续重试次数
        """
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.retry_count = 0
        self.reflection_history: List[ReflectionResult] = []
        self.total_reflections = 0
        self.successful_recoveries = 0
    
    def detect_failure_type(self, error_message: str) -> FailureType:
        """从错误信息中检测失败类型"""
        error_lower = error_message.lower()
        
        for ftype, patterns in self.FAILURE_PATTERNS.items():
            for pattern in patterns:
                if pattern.lower() in error_lower:
                    return ftype
        
        return FailureType.UNKNOWN
    
    def reflect(
        self,
        failed_action: str,
        error_message: str,
        observation: Dict,
        history: List = None,
        current_subgoal: str = None,
    ) -> ReflectionResult:
        """
        反思失败并生成修正方案
        
        Args:
            failed_action: 失败的动作
            error_message: 错误信息
            observation: 当前环境观察
            history: 最近的动作历史
            current_subgoal: 当前子目标描述
        
        Returns:
            ReflectionResult包含失败分析和修正方案
        """
        self.total_reflections += 1
        self.retry_count += 1
        
        # 检测失败类型
        failure_type = self.detect_failure_type(error_message)
        
        # 如果连续重试太多，建议跳过
        if self.retry_count > self.max_retries:
            return ReflectionResult(
                original_action=failed_action,
                failure_type=failure_type,
                failure_reason=f"连续失败{self.retry_count}次",
                suggested_fix="跳过当前子目标",
                should_retry=False,
                should_change_subgoal=True,
            )
        
        # 用LLM分析（如果有）
        if self.llm_client:
            return self._llm_reflect(
                failed_action, error_message, observation,
                history, current_subgoal, failure_type
            )
        
        # 没有LLM时的启发式修正
        return self._heuristic_reflect(
            failed_action, error_message, failure_type, observation
        )
    
    def _llm_reflect(
        self,
        failed_action: str,
        error_message: str,
        observation: Dict,
        history: List,
        current_subgoal: str,
        failure_type: FailureType,
    ) -> ReflectionResult:
        """使用LLM进行深度反思"""
        
        history_text = ""
        if history:
            recent = history[-5:]
            history_text = "\n".join(str(h) for h in recent)
        
        prompt = f"""你是一个机器人故障分析师。请分析以下失败并给出修正方案。

## 失败的动作
{failed_action}

## 错误信息
{error_message}

## 当前状态
{observation}

## 最近历史
{history_text if history_text else "无"}

## 当前子目标
{current_subgoal if current_subgoal else "未指定"}

## 请分析
1. 失败的根本原因是什么？
2. 应该如何修正？

输出格式：
<REFLECTION>
失败原因：[一句话分析]
修正策略：[具体修正方案]
</REFLECTION>
<ACTIONS>
[修正后的动作]
</ACTIONS>"""

        try:
            response = self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=300,
            )
            
            # 解析反思
            reflection_text = ""
            if "<REFLECTION>" in response and "</REFLECTION>" in response:
                start = response.index("<REFLECTION>") + len("<REFLECTION>")
                end = response.index("</REFLECTION>")
                reflection_text = response[start:end].strip()
            
            # 解析修正动作
            new_action = None
            if "<ACTIONS>" in response and "</ACTIONS>" in response:
                start = response.index("<ACTIONS>") + len("<ACTIONS>")
                end = response.index("</ACTIONS>")
                new_action = response[start:end].strip()
            
            result = ReflectionResult(
                original_action=failed_action,
                failure_type=failure_type,
                failure_reason=reflection_text or error_message,
                suggested_fix=reflection_text,
                new_action=new_action,
                should_retry=True,
            )
            
            self.reflection_history.append(result)
            return result
            
        except Exception as e:
            return self._heuristic_reflect(
                failed_action, str(e), failure_type, observation
            )
    
    def _heuristic_reflect(
        self,
        failed_action: str,
        error_message: str,
        failure_type: FailureType,
        observation: Dict,
    ) -> ReflectionResult:
        """启发式修正（不依赖LLM）"""
        
        action_parts = failed_action.strip().split()
        
        if failure_type == FailureType.OUT_OF_REACH:
            result = ReflectionResult(
                original_action=failed_action,
                failure_type=failure_type,
                failure_reason="目标超出机器人可达范围",
                suggested_fix="移动到可达位置，或让另一个机器人处理",
                new_action=f"{action_parts[0]}: WAIT" if len(action_parts) > 1 else "WAIT",
                should_retry=True,
            )
        
        elif failure_type == FailureType.COLLISION:
            result = ReflectionResult(
                original_action=failed_action,
                failure_type=failure_type,
                failure_reason="发生碰撞",
                suggested_fix="让一个机器人等待，另一个先完成任务",
                new_action=f"{action_parts[0]}: WAIT" if len(action_parts) > 1 else "WAIT",
                should_retry=True,
            )
        
        elif failure_type == FailureType.GRASP_FAILED:
            result = ReflectionResult(
                original_action=failed_action,
                failure_type=failure_type,
                failure_reason="抓取失败",
                suggested_fix="确认物体位置，重新靠近后抓取",
                new_action=failed_action.replace("PICK", "MOVE").split()[0] + ": MOVE",
                should_retry=True,
            )
        
        elif failure_type == FailureType.LOOP_DETECTED:
            result = ReflectionResult(
                original_action=failed_action,
                failure_type=failure_type,
                failure_reason="检测到动作循环，需要换策略",
                suggested_fix="更换动作策略，避免重复",
                new_action="WAIT",
                should_retry=False,
            )
        
        else:
            result = ReflectionResult(
                original_action=failed_action,
                failure_type=failure_type,
                failure_reason=error_message or "未知失败",
                suggested_fix="尝试等待后重新规划",
                new_action="WAIT",
                should_retry=True,
            )
        
        self.reflection_history.append(result)
        return result
    
    def reset_retry_count(self):
        """重置重试计数（任务成功后调用）"""
        self.retry_count = 0
    
    def record_successful_recovery(self):
        """记录一次成功的恢复"""
        self.successful_recoveries += 1
        self.reset_retry_count()
    
    def get_stats(self) -> Dict:
        """获取反思统计"""
        return {
            "total_reflections": self.total_reflections,
            "successful_recoveries": self.successful_recoveries,
            "recovery_rate": f"{self.successful_recoveries/max(1,self.total_reflections)*100:.1f}%",
            "current_retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "failure_types": self._count_failure_types(),
        }
    
    def _count_failure_types(self) -> Dict[str, int]:
        """统计各失败类型出现次数"""
        counts = {}
        for r in self.reflection_history:
            ftype = r.failure_type.value
            counts[ftype] = counts.get(ftype, 0) + 1
        return counts
    
    def get_recent_reflections(self, n: int = 5) -> List[ReflectionResult]:
        """获取最近的N次反思"""
        return self.reflection_history[-n:]
    
    def build_reflection_context(self) -> str:
        """构建反思上下文（注入到Prompt中提醒LLM避免重复错误）"""
        recent = self.get_recent_reflections(3)
        if not recent:
            return ""
        
        lines = ["## ⚠️ 最近的失败教训"]
        for i, r in enumerate(recent):
            lines.append(f"{i+1}. 动作'{r.original_action}'失败: {r.failure_reason}")
            if r.suggested_fix:
                lines.append(f"   建议: {r.suggested_fix}")
        
        lines.append("\n请避免重复以上错误。")
        return "\n".join(lines)


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ReflectionEngine 测试")
    print("=" * 60)
    
    re_engine = ReflectionEngine(max_retries=3)
    
    # 测试失败类型检测
    print("\n🔍 失败类型检测:")
    test_errors = [
        ("Collision detected between rob0 and rob1", FailureType.COLLISION),
        ("Object out of reach for rob0", FailureType.OUT_OF_REACH),
        ("Grasp failed: object slipped", FailureType.GRASP_FAILED),
        ("Action timeout after 600 seconds", FailureType.TIMEOUT),
        ("Invalid action: UNKNOWN_COMMAND", FailureType.INVALID_ACTION),
        ("Something went wrong", FailureType.UNKNOWN),
    ]
    for error, expected in test_errors:
        detected = re_engine.detect_failure_type(error)
        status = "✅" if detected == expected else "❌"
        print(f"   {status} '{error[:40]}...' → {detected.value}")
    
    # 测试启发式修正
    print("\n🔄 启发式修正测试:")
    test_obs = {"robot_positions": {"rob0": "(0.5, 0.3)"}}
    
    result1 = re_engine.reflect(
        "rob0: PICK red_cube",
        "red_cube out of reach",
        test_obs,
    )
    print(f"   失败动作: {result1.original_action}")
    print(f"   失败类型: {result1.failure_type.value}")
    print(f"   修正方案: {result1.suggested_fix}")
    print(f"   新动作: {result1.new_action}")
    
    result2 = re_engine.reflect(
        "rob0: SWEEP red_cube",
        "Collision detected",
        test_obs,
    )
    print(f"\n   失败动作: {result2.original_action}")
    print(f"   失败类型: {result2.failure_type.value}")
    print(f"   修正方案: {result2.suggested_fix}")
    
    # 测试连续失败
    print("\n🔄 连续失败测试:")
    for i in range(4):
        result = re_engine.reflect(
            f"rob0: MOVE target_{i}",
            "Object out of reach",
            test_obs,
        )
        print(f"   第{i+1}次: retry={result.should_retry}, change_subgoal={result.should_change_subgoal}")
    
    re_engine.record_successful_recovery()
    print(f"\n📊 统计: {re_engine.get_stats()}")
    
    # 测试反思上下文
    print(f"\n📝 反思上下文:")
    print(re_engine.build_reflection_context())
    
    print("\n✅ ReflectionEngine 测试完成！")
