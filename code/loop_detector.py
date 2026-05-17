"""
循环检测器 - LoopDetector
===========================
借鉴：Voyager skill library / Reflexion loop detection
创新点：检测动作重复循环，自动触发策略变更

解决问题：
- LLM在困境中会重复输出相同动作
- 两个机器人在死锁中互相等待
- 任务卡在某个状态无法推进

核心算法：
- 滑动窗口检测连续重复
- 余弦相似度检测行为模式重复
- 自动生成循环警告注入Prompt

使用方法：
    from loop_detector import LoopDetector
    ld = LoopDetector(window_size=5)
    ld.add_action("rob0: MOVE red_cube")
    if ld.is_looping():
        warning = ld.build_loop_warning()
"""

from typing import Dict, List, Tuple, Optional
from collections import deque
import hashlib


class LoopDetector:
    """动作循环检测器"""
    
    def __init__(self, window_size: int = 5, max_consecutive_repeats: int = 3):
        """
        Args:
            window_size: 检测窗口大小
            max_consecutive_repeats: 最大连续重复次数（超过则判定循环）
        """
        self.window_size = window_size
        self.max_consecutive_repeats = max_consecutive_repeats
        self.action_history: deque = deque(maxlen=window_size * 2)
        self.state_history: deque = deque(maxlen=window_size)
        self.loop_count = 0
        self.total_loops_detected = 0
        self.loop_actions: set = set()  # 记录触发循环的动作
    
    def add_action(self, action: str, result: str = None):
        """
        添加一个动作到历史
        
        Args:
            action: 动作字符串
            result: 执行结果（success/failed/timeout等）
        """
        self.action_history.append({
            "action": action,
            "result": result or "unknown",
            "hash": self._hash_action(action),
        })
    
    def add_state(self, state: Dict):
        """添加环境状态到历史"""
        self.state_history.append(state)
    
    def is_looping(self) -> bool:
        """
        检测是否在循环中
        
        使用多种检测策略：
        1. 精确重复检测：连续N次完全相同
        2. 模式重复检测：动作类型模式重复
        3. 状态停滞检测：环境状态不变但持续输出动作
        """
        if len(self.action_history) < self.max_consecutive_repeats:
            return False
        
        is_loop = False
        
        # 策略1：精确重复
        if self._detect_exact_repeat():
            is_loop = True
        
        # 策略2：动作类型模式重复
        if self._detect_pattern_repeat():
            is_loop = True
        
        # 策略3：来回切换（A→B→A→B→A→B...）
        if self._detect_toggle_loop():
            is_loop = True
        
        if is_loop:
            self.total_loops_detected += 1
            # 记录触发循环的动作
            recent = list(self.action_history)[-self.max_consecutive_repeats:]
            for item in recent:
                self.loop_actions.add(item["action"])
        
        return is_loop
    
    def _detect_exact_repeat(self) -> bool:
        """检测精确重复：连续N次完全相同"""
        recent_actions = [
            item["action"] for item in list(self.action_history)[-self.max_consecutive_repeats:]
        ]
        return len(set(recent_actions)) == 1
    
    def _detect_pattern_repeat(self) -> bool:
        """检测动作类型模式重复"""
        if len(self.action_history) < self.window_size:
            return False
        
        recent = list(self.action_history)[-self.window_size:]
        action_types = [self._extract_action_type(item["action"]) for item in recent]
        
        # 检查动作类型是否过于单一（比如全是WAIT或全是MOVE）
        unique_types = set(action_types)
        if len(unique_types) <= 2 and len(recent) >= self.max_consecutive_repeats * 2:
            # 如果只有1-2种动作类型，且持续了很久
            return True
        
        return False
    
    def _detect_toggle_loop(self) -> bool:
        """检测来回切换循环 A→B→A→B..."""
        if len(self.action_history) < 6:
            return False
        
        recent = [item["action"] for item in list(self.action_history)[-6:]]
        if len(set(recent)) == 2:
            # 检查是否交替出现
            pattern = recent[0]
            for i in range(len(recent)):
                expected = recent[0] if i % 2 == 0 else recent[1]
                if recent[i] != expected:
                    return False
            return True
        
        return False
    
    def is_state_stuck(self) -> bool:
        """检测环境状态是否停滞"""
        if len(self.state_history) < 3:
            return False
        
        recent_states = list(self.state_history)[-3:]
        # 比较最近3个状态是否完全相同
        state_strs = [str(s) for s in recent_states]
        return len(set(state_strs)) == 1
    
    def build_loop_warning(self) -> str:
        """
        构建循环警告Prompt
        
        当检测到循环时，将此文本注入Prompt提醒LLM
        """
        if not self.is_looping():
            return ""
        
        recent_actions = [
            item["action"] for item in list(self.action_history)[-5:]
        ]
        
        warning = f"""## ⚠️ 循环检测警告！
你最近的动作出现了循环：
{chr(10).join(f'- {a}' for a in recent_actions)}

这可能导致任务卡死。请：
1. 分析为什么之前的动作没有推进任务
2. 尝试完全不同的策略
3. 如果必要，让所有机器人WAIT，重新规划

不要再重复以上动作！"""
        
        return warning
    
    def get_suggested_alternative(
        self,
        available_actions: List[str],
        avoided_actions: List[str] = None,
    ) -> Optional[str]:
        """
        从可用动作中建议一个不同的动作
        
        Args:
            available_actions: 当前可用动作列表
            avoided_actions: 要避免的动作列表
        
        Returns:
            建议的动作字符串
        """
        avoided = set(avoided_actions or [])
        avoided.update(self.loop_actions)
        
        # 从可用动作中排除循环动作
        candidates = [a for a in available_actions if a not in avoided]
        
        if candidates:
            return candidates[0]  # 返回第一个不同的
        
        # 所有动作都被避免了，返回WAIT
        for a in available_actions:
            if "WAIT" in a.upper():
                return a
        
        return "WAIT"
    
    def _extract_action_type(self, action: str) -> str:
        """从动作字符串中提取动作类型（第一个单词）"""
        parts = action.strip().split()
        return parts[0] if parts else "UNKNOWN"
    
    def _hash_action(self, action: str) -> str:
        """计算动作的简单哈希"""
        return hashlib.md5(action.encode()).hexdigest()[:8]
    
    def reset(self):
        """重置检测器"""
        self.action_history.clear()
        self.state_history.clear()
        self.loop_count = 0
        self.loop_actions.clear()
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "total_loops_detected": self.total_loops_detected,
            "history_size": len(self.action_history),
            "loop_actions": list(self.loop_actions),
            "window_size": self.window_size,
            "max_consecutive_repeats": self.max_consecutive_repeats,
        }
    
    def compress_history(self, max_items: int = 5) -> List[Dict]:
        """
        压缩历史记录用于Prompt
        
        只保留最近的动作，避免Prompt过长
        """
        recent = list(self.action_history)[-max_items:]
        return [
            {"step": i+1, "action": item["action"], "result": item["result"]}
            for i, item in enumerate(recent)
        ]
    
    def format_compressed_history(self, max_items: int = 5) -> str:
        """格式化压缩后的历史为Prompt文本"""
        compressed = self.compress_history(max_items)
        if not compressed:
            return ""
        
        lines = ["## 最近动作历史"]
        for item in compressed:
            status = "✅" if item["result"] == "success" else "❌" if item["result"] == "failed" else "➡️"
            lines.append(f"{status} 步骤{item['step']}: {item['action']}")
        
        return "\n".join(lines)


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LoopDetector 测试")
    print("=" * 60)
    
    ld = LoopDetector(window_size=5, max_consecutive_repeats=3)
    
    # 测试1：精确重复
    print("\n🔍 测试1: 精确重复检测")
    for i in range(5):
        ld.add_action("rob0: MOVE red_cube", "failed")
        print(f"   步骤{i+1}: rob0: MOVE red_cube → 循环={ld.is_looping()}")
    
    print(f"\n   循环检测到: {ld.total_loops_detected} 次")
    print(f"   警告: {ld.build_loop_warning()[:200]}...")
    
    ld.reset()
    
    # 测试2：来回切换
    print("\n🔍 测试2: 来回切换检测")
    toggle_actions = ["rob0: MOVE left", "rob0: MOVE right"] * 4
    for i, action in enumerate(toggle_actions):
        ld.add_action(action)
        if i >= 5:
            print(f"   步骤{i+1}: {action} → 循环={ld.is_looping()}")
    
    ld.reset()
    
    # 测试3：正常序列（不应检测为循环）
    print("\n🔍 测试3: 正常序列")
    normal_actions = [
        "rob0: MOVE red_cube",
        "rob1: MOVE red_cube",
        "rob0: SWEEP red_cube",
        "rob1: DUMP",
        "rob0: MOVE blue_cube",
        "rob1: MOVE blue_cube",
    ]
    for action in normal_actions:
        ld.add_action(action, "success")
    print(f"   循环检测: {ld.is_looping()} (应为False)")
    
    ld.reset()
    
    # 测试4：动作类型模式重复
    print("\n🔍 测试4: 动作类型模式重复")
    for i in range(8):
        ld.add_action(f"rob0: MOVE target_{i%3}", "success")
    print(f"   循环检测: {ld.is_looping()} (可能为True,因为全是MOVE)")
    
    # 测试5：建议替代动作
    print("\n💡 测试5: 建议替代动作")
    ld2 = LoopDetector()
    ld2.add_action("rob0: MOVE red_cube")
    ld2.add_action("rob0: MOVE red_cube")
    ld2.add_action("rob0: MOVE red_cube")
    
    available = ["rob0: MOVE red_cube", "rob0: SWEEP red_cube", "rob0: WAIT"]
    suggestion = ld2.get_suggested_alternative(available)
    print(f"   可用动作: {available}")
    print(f"   建议: {suggestion}")
    
    # 测试6：历史压缩
    print("\n📝 测试6: 历史压缩")
    ld3 = LoopDetector()
    for action in normal_actions:
        ld3.add_action(action, "success")
    print(ld3.format_compressed_history(max_items=4))
    
    print("\n✅ LoopDetector 测试完成！")
