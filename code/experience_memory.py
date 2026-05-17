"""
经验记忆库 - ExperienceMemory
===============================
借鉴：Voyager skill library / Reflexion episodic memory / Few-Shot prompting
创新点：保存成功案例作为Few-Shot示例，失败案例作为避免模式

解决问题：
- LLM每次都从零开始决策，没有"记忆"
- 同样的成功模式每次都要重新"发现"
- 失败模式重复出现

使用方法：
    from experience_memory import ExperienceMemory
    mem = ExperienceMemory("experience_memory.json")
    mem.record_success(task, state, actions)
    examples = mem.get_few_shot_examples(task, k=3)
    failures = mem.get_failure_patterns(task)
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple


class ExperienceMemory:
    """经验记忆库：成功/失败案例的持久化存储"""
    
    def __init__(self, filepath: str = "experience_memory.json", max_entries: int = 200):
        """
        Args:
            filepath: 记忆库文件路径
            max_entries: 最大记忆条数（防止文件过大）
        """
        self.filepath = filepath
        self.max_entries = max_entries
        self.memories: List[Dict] = []
        self._load()
    
    def _load(self):
        """从文件加载记忆"""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self.memories = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.memories = []
    
    def _save(self):
        """保存记忆到文件"""
        # 限制最大条数
        if len(self.memories) > self.max_entries:
            self.memories = self.memories[-self.max_entries:]
        
        os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.memories, f, ensure_ascii=False, indent=2)
    
    def record_success(
        self,
        task_name: str,
        state: Dict,
        actions: List[str],
        reward: float = 1.0,
        metadata: Dict = None,
    ):
        """
        记录一次成功经验
        
        Args:
            task_name: 任务名称
            state: 执行前的状态描述
            actions: 成功的动作序列
            reward: 奖励分数（用于排序）
            metadata: 额外元数据
        """
        entry = {
            "type": "success",
            "task": task_name,
            "timestamp": datetime.now().isoformat(),
            "state_summary": self._summarize_state(state),
            "actions": actions,
            "reward": reward,
            "metadata": metadata or {},
        }
        self.memories.append(entry)
        self._save()
    
    def record_failure(
        self,
        task_name: str,
        state: Dict,
        failed_action: str,
        error_message: str,
        failure_type: str,
    ):
        """
        记录一次失败经验
        
        Args:
            task_name: 任务名称
            state: 失败时的状态
            failed_action: 失败的动作
            error_message: 错误信息
            failure_type: 失败类型
        """
        entry = {
            "type": "failure",
            "task": task_name,
            "timestamp": datetime.now().isoformat(),
            "state_summary": self._summarize_state(state),
            "failed_action": failed_action,
            "error": error_message,
            "failure_type": failure_type,
        }
        self.memories.append(entry)
        self._save()
    
    def get_few_shot_examples(
        self,
        task_name: str,
        k: int = 3,
        min_reward: float = 0.5,
    ) -> List[Dict]:
        """
        获取Few-Shot成功示例
        
        Args:
            task_name: 任务名称
            k: 返回示例数量
            min_reward: 最低奖励阈值
        
        Returns:
            最近的k个成功案例
        """
        successes = [
            m for m in self.memories
            if m["type"] == "success"
            and m["task"] == task_name
            and m.get("reward", 0) >= min_reward
        ]
        
        # 按奖励排序，取最近的k个
        successes.sort(key=lambda x: x.get("reward", 0), reverse=True)
        return successes[:k]
    
    def get_failure_patterns(self, task_name: str, k: int = 5) -> List[Dict]:
        """获取最近的失败模式"""
        failures = [
            m for m in self.memories
            if m["type"] == "failure" and m["task"] == task_name
        ]
        return failures[-k:]
    
    def get_common_failures(self, task_name: str) -> Dict[str, int]:
        """统计常见失败类型"""
        failure_counts = {}
        for m in self.memories:
            if m["type"] == "failure" and m["task"] == task_name:
                ftype = m.get("failure_type", "unknown")
                failure_counts[ftype] = failure_counts.get(ftype, 0) + 1
        return failure_counts
    
    def find_similar_state(
        self,
        task_name: str,
        current_state: Dict,
        k: int = 3,
    ) -> List[Dict]:
        """
        查找与当前状态最相似的历史经验
        
        使用简单的关键词匹配做相似度计算
        """
        current_text = self._summarize_state(current_state).lower()
        scored = []
        
        for m in self.memories:
            if m["task"] != task_name:
                continue
            state_text = m.get("state_summary", "").lower()
            # 简单相似度：共同词数量
            if state_text and current_text:
                common = set(current_text.split()) & set(state_text.split())
                score = len(common) / max(1, len(set(current_text.split())))
                scored.append((score, m))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:k]]
    
    def build_few_shot_prompt(self, task_name: str, k: int = 2) -> str:
        """
        构建Few-Shot Prompt前缀
        
        Returns:
            格式化的Few-Shot示例文本
        """
        examples = self.get_few_shot_examples(task_name, k)
        if not examples:
            return ""
        
        lines = ["## 历史成功案例（请参考）"]
        for i, ex in enumerate(examples):
            lines.append(f"\n案例{i+1}:")
            lines.append(f"状态: {ex.get('state_summary', 'N/A')[:200]}")
            lines.append(f"动作序列: {', '.join(ex.get('actions', [])[:5])}")
        
        lines.append("\n请参考以上成功案例来规划动作。")
        return "\n".join(lines)
    
    def build_failure_warning(self, task_name: str) -> str:
        """
        构建失败警告Prompt
        
        把常见的失败模式注入Prompt，提醒LLM避免
        """
        failures = self.get_failure_patterns(task_name, 3)
        if not failures:
            return ""
        
        common = self.get_common_failures(task_name)
        
        lines = ["## ⚠️ 常见失败模式（请避免）"]
        for ftype, count in sorted(common.items(), key=lambda x: -x[1]):
            lines.append(f"- {ftype}: 出现{count}次")
        
        if failures:
            lines.append(f"\n最近一次失败: {failures[-1].get('failed_action', 'N/A')}")
            lines.append(f"原因: {failures[-1].get('error', 'N/A')[:100]}")
        
        return "\n".join(lines)
    
    def _summarize_state(self, state: Any) -> str:
        """将状态压缩为文本摘要"""
        if isinstance(state, dict):
            return json.dumps(state, ensure_ascii=False, default=str)[:500]
        elif isinstance(state, str):
            return state[:500]
        return str(state)[:500]
    
    def get_stats(self) -> Dict:
        """获取记忆库统计"""
        success_count = sum(1 for m in self.memories if m["type"] == "success")
        failure_count = sum(1 for m in self.memories if m["type"] == "failure")
        
        tasks = {}
        for m in self.memories:
            task = m["task"]
            if task not in tasks:
                tasks[task] = {"success": 0, "failure": 0}
            tasks[task][m["type"]] += 1
        
        return {
            "total_memories": len(self.memories),
            "successes": success_count,
            "failures": failure_count,
            "by_task": tasks,
            "filepath": self.filepath,
        }
    
    def clear_task(self, task_name: str):
        """清除某个任务的所有记忆"""
        self.memories = [
            m for m in self.memories
            if m["task"] != task_name
        ]
        self._save()
    
    def clear_all(self):
        """清除所有记忆"""
        self.memories = []
        self._save()
    
    def export_summary(self) -> str:
        """导出记忆摘要（用于报告）"""
        stats = self.get_stats()
        lines = ["# 经验记忆库摘要", ""]
        lines.append(f"总记忆数: {stats['total_memories']}")
        lines.append(f"成功: {stats['successes']} | 失败: {stats['failures']}")
        lines.append("")
        
        for task, counts in stats["by_task"].items():
            lines.append(f"## {task}")
            lines.append(f"  成功: {counts['success']}, 失败: {counts['failure']}")
            common = self.get_common_failures(task)
            if common:
                lines.append("  常见失败:")
                for ftype, count in sorted(common.items(), key=lambda x: -x[1])[:3]:
                    lines.append(f"    - {ftype}: {count}次")
        
        return "\n".join(lines)


# ============================================================
# 便捷函数
# ============================================================

def create_memory(filepath: str = "experience_memory.json") -> ExperienceMemory:
    """快速创建经验记忆库"""
    return ExperienceMemory(filepath)


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    import tempfile
    
    # 使用临时文件测试
    tmp_file = os.path.join(tempfile.gettempdir(), "test_roco_memory.json")
    
    print("=" * 60)
    print("ExperienceMemory 测试")
    print("=" * 60)
    
    mem = ExperienceMemory(tmp_file)
    
    # 记录一些成功案例
    print("\n📝 记录成功案例...")
    mem.record_success(
        "sweep_floor",
        {"cube_states": {"red": "on_table"}, "task_stage": "开始"},
        ["rob0: MOVE red_cube", "rob1: MOVE red_cube", "rob0: SWEEP red_cube", "rob1: DUMP"],
        reward=1.0,
    )
    mem.record_success(
        "sweep_floor",
        {"cube_states": {"blue": "on_table", "red": "in_dustpan"}, "task_stage": "清扫中"},
        ["rob0: MOVE blue_cube", "rob1: MOVE blue_cube", "rob0: SWEEP blue_cube"],
        reward=0.8,
    )
    
    # 记录一些失败
    print("📝 记录失败案例...")
    mem.record_failure(
        "sweep_floor",
        {"cube_states": {"red": "far_away"}, "task_stage": "开始"},
        "rob0: SWEEP red_cube",
        "Object out of reach",
        "out_of_reach",
    )
    
    # 测试Few-Shot
    print("\n📖 Few-Shot示例:")
    examples = mem.get_few_shot_examples("sweep_floor", k=2)
    for i, ex in enumerate(examples):
        print(f"   案例{i+1}: 奖励={ex['reward']}, 动作数={len(ex['actions'])}")
    
    # 测试失败模式
    print("\n⚠️ 失败模式:")
    common = mem.get_common_failures("sweep_floor")
    for ftype, count in common.items():
        print(f"   {ftype}: {count}次")
    
    # 测试Prompt生成
    print("\n📝 Few-Shot Prompt:")
    fsp = mem.build_few_shot_prompt("sweep_floor", k=1)
    print(f"   {fsp[:300]}...")
    
    print("\n⚠️ 失败警告Prompt:")
    fwp = mem.build_failure_warning("sweep_floor")
    print(f"   {fwp[:300]}...")
    
    # 统计
    print(f"\n📊 记忆库统计: {mem.get_stats()}")
    
    # 导出摘要
    print(f"\n📄 记忆摘要:")
    print(mem.export_summary())
    
    # 清理
    mem.clear_all()
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
    
    print("\n✅ ExperienceMemory 测试完成！")
